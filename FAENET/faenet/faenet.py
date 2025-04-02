"""
Simplified implementation of the Frame Averaging (Rotation Invariant) GNN
"""
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Embedding, Linear
from torch_geometric.utils import dropout_edge
from torch_geometric.nn import MessagePassing, radius_graph
from torch_geometric.nn.norm import GraphNorm
from torch_scatter import scatter

from utils import swish, GaussianSmearing, ForceDecoder, get_distances, conditional_grad


class EmbeddingBlock(nn.Module):
    """Embedding block for the GNN that initializes nodes and edges embeddings"""

    def __init__(
        self,
        num_gaussians,
        num_filters,
        hidden_channels,
        act,
    ):
        super().__init__()
        self.act = act
        
        # Main embedding
        self.emb = Embedding(100, hidden_channels)  # Support up to 100 atom types
        
        # MLP
        self.lin = Linear(hidden_channels, hidden_channels)
        self.lin_2 = Linear(hidden_channels, hidden_channels)
        
        # Edge embedding
        self.lin_e1 = Linear(3, num_filters // 2)  # r_ij
        self.lin_e2 = Linear(num_gaussians, num_filters - (num_filters // 2))  # d_ij
        self.lin_e3 = Linear(num_filters, num_filters)
        
        self.reset_parameters()

    def reset_parameters(self):
        self.emb.reset_parameters()
        nn.init.xavier_uniform_(self.lin.weight)
        self.lin.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.lin_2.weight)
        self.lin_2.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.lin_e1.weight)
        self.lin_e1.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.lin_e2.weight)
        self.lin_e2.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.lin_e3.weight)
        self.lin_e3.bias.data.fill_(0)

    def forward(self, z, rel_pos, edge_attr):
        # Edge embedding
        rel_pos = self.lin_e1(rel_pos)  # r_ij
        edge_attr = self.lin_e2(edge_attr)  # d_ij
        e = torch.cat((rel_pos, edge_attr), dim=1)
        e = self.act(e)
        e = self.act(self.lin_e3(e))
        
        # Node embedding
        h = self.emb(z)
        h = self.act(self.lin(h))
        h = self.act(self.lin_2(h))
        
        return h, e


class InteractionBlock(MessagePassing):
    """Interaction block for the GNN that updates node representations"""

    def __init__(
        self,
        hidden_channels,
        num_filters,
        act,
        graph_norm=True,
        dropout=0.0
    ):
        super(InteractionBlock, self).__init__()
        self.act = act
        self.hidden_channels = hidden_channels
        self.dropout = dropout
        
        # Graph normalization
        if graph_norm:
            self.graph_norm = GraphNorm(hidden_channels)
        else:
            self.graph_norm = None
        
        # Message passing layers
        self.lin_geom = nn.Linear(num_filters, num_filters)
        self.lin_down = nn.Linear(hidden_channels, num_filters)
        self.lin_up = nn.Linear(num_filters, hidden_channels)
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_geom.weight)
        self.lin_geom.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.lin_down.weight)
        self.lin_down.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.lin_up.weight)
        self.lin_up.bias.data.fill_(0)

    def forward(self, h, edge_index, e):
        # Apply dropout
        if self.dropout > 0 and self.training:
            h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Prepare edge features
        e = self.act(self.lin_geom(e))
        
        # Message passing
        h = self.act(self.lin_down(h))  # downscale node rep.
        h = self.propagate(edge_index, x=h, W=e)  # propagate
        
        # Graph normalization
        if self.graph_norm is not None:
            h = self.act(self.graph_norm(h))
            
        # Apply dropout and upscale
        if self.dropout > 0 and self.training:
            h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.act(self.lin_up(h))  # upscale node rep.
        
        return h

    def message(self, x_j, W):
        return x_j * W


class OutputBlock(nn.Module):
    """Output block that produces energy prediction from atom embeddings"""
    
    def __init__(self, hidden_channels, act, dropout=0.0):
        super().__init__()
        self.act = act
        self.dropout = dropout
        
        # Prediction MLP
        self.lin1 = Linear(hidden_channels, hidden_channels // 2)
        self.lin2 = Linear(hidden_channels // 2, 1)
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin1.weight)
        self.lin1.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def forward(self, h, batch):
        # MLP
        if self.dropout > 0 and self.training:
            h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.lin1(h)
        h = self.act(h)
        
        if self.dropout > 0 and self.training:
            h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.lin2(h)
        
        # Global pooling
        out = scatter(h, batch, dim=0, reduce="add")
        
        return out


class FAENet(nn.Module):
    """Frame Averaging Equivariant Network (FAENet)
    
    Args:
        cutoff (float): Cutoff distance for interatomic interactions
        num_gaussians (int): Number of gaussians used for distance expansion
        hidden_channels (int): Size of hidden representations
        num_filters (int): Size of edge features
        num_interactions (int): Number of interaction blocks
        dropout (float): Dropout rate
        output_properties (list): List of properties to predict
        regress_forces (bool): Whether to predict forces
    """

    def __init__(
        self,
        cutoff=6.0,
        num_gaussians=50,
        hidden_channels=128,
        num_filters=128,
        num_interactions=4,
        dropout=0.0,
        output_properties=["energy"],
        regress_forces=False,
    ):
        super().__init__()
        
        self.cutoff = cutoff
        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.dropout = dropout
        self.output_properties = output_properties
        self.regress_forces = regress_forces
        self.training = True
        
        # Use swish activation
        self.act = swish
        
        # Distance expansion
        self.distance_expansion = GaussianSmearing(0.0, self.cutoff, num_gaussians)
        
        # Embedding block
        self.embed_block = EmbeddingBlock(
            num_gaussians,
            num_filters,
            hidden_channels,
            self.act,
        )
        
        # Interaction blocks
        self.interaction_blocks = nn.ModuleList(
            [
                InteractionBlock(
                    hidden_channels,
                    num_filters,
                    self.act,
                    True,
                    dropout
                )
                for _ in range(num_interactions)
            ]
        )
        
        # Output blocks for each property
        self.output_blocks = nn.ModuleDict()
        for prop in output_properties:
            self.output_blocks[prop] = OutputBlock(
                hidden_channels,
                self.act,
                dropout
            )
        
        # Force decoder
        if regress_forces:
            self.force_decoder = ForceDecoder(
                hidden_channels,
                hidden_channels
            )
        
    def forward(self, data):
        """Forward pass for property prediction
        
        Args:
            data: PyTorch Geometric Data object with:
                - pos: atom positions
                - atomic_numbers: atom types
                - batch: batch assignment for atoms
                - cell: unit cell for periodic boundary conditions (optional)
                - edge_index: connectivity in COO format (optional)
                - cell_offsets: offsets for periodic boundary conditions (optional)
                - neighbors: number of neighbors for each atom (optional)
                
        Returns:
            dict: Predictions for target properties and optionally forces
        """
        z = data.atomic_numbers.long()
        pos = data.pos
        batch = data.batch
        
        # Handle periodic boundary conditions if present
        if hasattr(data, 'cell') and data.cell is not None and hasattr(data, 'cell_offsets'):
            edge_index = data.edge_index
            # Get distances with periodic boundary conditions
            out = self.get_pbc_distances(
                pos,
                edge_index,
                data.cell,
                data.cell_offsets,
                data.neighbors if hasattr(data, 'neighbors') else None
            )
            edge_weight = out["distances"]
            rel_pos = out["distance_vec"]
        else:
            # Compute graph connectivity if not provided
            if not hasattr(data, 'edge_index') or data.edge_index is None:
                edge_index = radius_graph(
                    pos, 
                    r=self.cutoff,
                    batch=batch,
                    max_num_neighbors=50
                )
            else:
                edge_index = data.edge_index
                
            # Apply dropout to edges during training
            if self.dropout > 0 and self.training:
                edge_index, edge_mask = dropout_edge(
                    edge_index,
                    p=self.dropout,
                    force_undirected=True,
                    training=self.training,
                )
                
            # Compute edge features for non-periodic case
            out = get_distances(pos, edge_index)
            edge_weight = out["distances"]
            rel_pos = out["distance_vec"]
            
        # Apply distance expansion
        edge_attr = self.distance_expansion(edge_weight)
            
        # Embedding block
        h, e = self.embed_block(z, rel_pos, edge_attr)
        
        # Store intermediate results for skip connections
        h_list = [h]
        
        # Interaction blocks
        for interaction in self.interaction_blocks:
            h = h + interaction(h, edge_index, e)
            h_list.append(h)
            
        # Output blocks for each property
        preds = {
            "hidden_state": h
        }
        
        for prop in self.output_properties:
            preds[prop] = self.output_blocks[prop](h, batch)
        
        # Add forces if requested
        if self.regress_forces:
            preds["forces"] = self.forces_forward(preds)
            
        return preds
        
    def get_pbc_distances(self, pos, edge_index, cell, cell_offsets, neighbors=None):
        """Calculate distances with periodic boundary conditions
        
        Args:
            pos: Atom positions
            edge_index: Graph connectivity
            cell: Unit cell (should be a 3x3 tensor)
            cell_offsets: Offsets for periodic boundary conditions
            neighbors: Number of neighbors for each atom
            
        Returns:
            dict: Distances, relative positions, and edge index
        """
        row, col = edge_index
        
        # Convert inputs to the right types and devices
        cell_offsets = cell_offsets.to(pos.device).float()
        
        # Ensure cell is correctly shaped for matrix multiplication
        # For frame averaging, the training loop should set the correct cell for each frame
        if not isinstance(cell, torch.Tensor):
            # Handle non-tensor cell
            cell = torch.tensor(cell, device=pos.device, dtype=torch.float)
        
        # Simple shape fix for common cases
        if cell.shape != (3, 3):
            if cell.shape == (12, 3) and cell.dim() == 2:
                # This shape occurs with 2D frame averaging (4 frames, 3 rows per frame)
                cell = cell[:3].clone()
            elif cell.numel() >= 9:
                # If we have enough elements, reshape to 3x3
                cell = cell.reshape(-1)[:9].reshape(3, 3)
            else:
                # If we can't get a proper cell, use identity
                cell = torch.eye(3, device=pos.device)
        
        # Ensure float dtype
        cell = cell.float()
        
        # Convert cell offsets to cartesian
        offsets = torch.matmul(cell_offsets, cell)
        
        # Calculate relative positions with offsets
        rel_pos = pos[row] - pos[col] + offsets
        
        # Calculate distances
        distances = torch.norm(rel_pos, dim=-1)
        
        return {
            "edge_index": edge_index,
            "distances": distances,
            "distance_vec": rel_pos
        }

    @conditional_grad(torch.enable_grad())
    def forces_forward(self, preds):
        """Predict forces from atom embeddings"""
        if hasattr(self, 'force_decoder'):
            return self.force_decoder(preds["hidden_state"])
        return None


def process_batch_with_frame_averaging(model, batch, fa_method="all"):
    """Process a batch using frame averaging
    
    Args:
        model: FAENet model
        batch: PyTorch Geometric Data batch
        fa_method: Frame averaging method
        
    Returns:
        dict: Model predictions
    """
    from frame_averaging import frame_averaging_3D
    
    # Store original positions
    original_pos = batch.pos
    
    # Apply frame averaging
    fa_pos, _, fa_rot = frame_averaging_3D(batch.pos, None, fa_method)
    
    # Store predictions for each frame
    e_all, f_all = [], []
    
    # Process each frame
    for i in range(len(fa_pos)):
        # Set positions to current frame
        batch.pos = fa_pos[i]
        
        # Forward pass
        preds = model(batch)
        e_all.append(preds["energy"])
        
        # Transform forces to maintain equivariance
        if "forces" in preds:
            fa_rot_expanded = torch.repeat_interleave(
                fa_rot[i], batch.natoms, dim=0
            )
            forces = (
                preds["forces"]
                .view(-1, 1, 3)
                .bmm(fa_rot_expanded.transpose(1, 2).to(preds["forces"].device))
                .view(-1, 3)
            )
            f_all.append(forces)
    
    # Restore original positions
    batch.pos = original_pos
    
    # Average predictions
    preds = {}
    preds["energy"] = sum(e_all) / len(e_all)
    
    if f_all:
        preds["forces"] = sum(f_all) / len(f_all)
        
    return preds