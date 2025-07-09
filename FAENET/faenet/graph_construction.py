"""
Graph construction module for crystal structures.
Integrates with frame averaging for multi-frame processing.
Enhanced with CGCNN features and improved periodic boundary condition handling.
"""
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
import torch_scatter
from pymatgen.core import Structure
import structlog

# Import JARVIS toolkit for CGCNN features
from jarvis.core.specie import get_node_attributes

# Import utility functions
from faenet.utils import SmoothCutoff

# Configure structlog
logger = structlog.get_logger()


class Envelope(nn.Module):
    """Smooth cutoff envelope function.
    
    Implements smooth cutoff as described in DimeNet paper to create
    continuous edge features that smoothly go to zero at the cutoff.
    """
    def __init__(self, exponent=5):
        super(Envelope, self).__init__()
        self.p = exponent + 1
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2
        
    def forward(self, x):
        """Apply smooth cutoff envelope to normalized distances.
        
        Args:
            x: Distances normalized by cutoff radius (distance/cutoff)
            
        Returns:
            Smoothed distance weights
        """
        p, a, b, c = self.p, self.a, self.b, self.c
        x_pow_p0 = x.pow(p - 1)
        x_pow_p1 = x_pow_p0 * x
        x_pow_p2 = x_pow_p1 * x
        
        # Apply envelope function: 1/x + ax^p-1 + bx^p + cx^p+1
        return 1. / x + a * x_pow_p0 + b * x_pow_p1 + c * x_pow_p2

def get_distances(pos, edge_index):
    """Calculate distances and displacement vectors for edges.
    
    Args:
        pos (tensor): Atom positions [num_atoms, 3]
        edge_index (tensor): Edge indices [2, num_edges]
        
    Returns:
        dict: Distances and displacement vectors
    """
    src, dst = edge_index
    rel_pos = pos[dst] - pos[src]
    distances = torch.norm(rel_pos, dim=-1)
    
    return {
        "edge_index": edge_index,
        "distances": distances,
        "distance_vec": rel_pos
    }


def get_distances_pbc(pos, edge_index, cell, cell_offsets):
    """Get distances with periodic boundary conditions.
    
    Args:
        pos (tensor): Atom positions [num_atoms, 3]
        edge_index (tensor): Edge indices [2, num_edges]
        cell (tensor): Unit cell [3, 3]
        cell_offsets (tensor): Cell offsets [num_edges, 3]
        
    Returns:
        dict: Distances and displacement vectors
    """
    src, dst = edge_index
    
    # Convert offsets to cartesian
    offsets_cart = torch.matmul(cell_offsets.float(), cell)
    
    # Calculate distances with PBC
    rel_pos = pos[dst] + offsets_cart - pos[src]
    distances = torch.norm(rel_pos, dim=-1)
    
    return {
        "edge_index": edge_index,
        "distances": distances,
        "distance_vec": rel_pos
    }


def canonize_edge(src_id, dst_id, src_image, dst_image):
    """Compute canonical edge representation.
    
    Sort vertex ids and shift periodic images so the first vertex is in (0,0,0) image.
    
    Args:
        src_id: Source atom index
        dst_id: Destination atom index
        src_image: Source periodic image
        dst_image: Destination periodic image
    
    Returns:
        tuple: Canonized (src_id, dst_id, src_image, dst_image)
    """
    # Store directed edges src_id <= dst_id
    if dst_id < src_id:
        src_id, dst_id = dst_id, src_id
        src_image, dst_image = dst_image, src_image

    # Shift periodic images so that src is in (0,0,0) image
    if not np.array_equal(src_image, (0, 0, 0)):
        shift = src_image
        src_image = tuple(np.subtract(src_image, shift))
        dst_image = tuple(np.subtract(dst_image, shift))

    return src_id, dst_id, src_image, dst_image


def radius_graph_pbc(pos, cell, cutoff, max_neighbors=32, batch=None):
    """Build a radius graph with periodic boundary conditions.
    
    Args:
        pos (tensor): Atom positions [num_atoms, 3]
        cell (tensor): Unit cell [3, 3]
        cutoff (float): Cutoff distance
        max_neighbors (int): Maximum number of neighbors per atom
        batch (tensor): Batch indices for atoms
        
    Returns:
        tuple: (edge_index, cell_offsets, neighbors)
    """
    if batch is None:
        batch = torch.zeros(pos.size(0), dtype=torch.long, device=pos.device)
    
    # Get reciprocal cell
    inv_cell = torch.inverse(cell)
    
    # Get fractional coordinates
    cart_coords = pos
    frac_coords = torch.matmul(cart_coords, inv_cell)
    
    # Compute neighboring cells to consider
    cell_norm = torch.norm(cell, dim=1)
    num_cells = torch.ceil(cutoff / cell_norm).long()
    
    # Generate neighbor cells
    neighbor_cells = []
    for i in range(-num_cells[0], num_cells[0] + 1):
        for j in range(-num_cells[1], num_cells[1] + 1):
            for k in range(-num_cells[2], num_cells[2] + 1):
                neighbor_cells.append([i, j, k])
    
    neighbor_cells = torch.tensor(neighbor_cells, device=pos.device)
    
    # Build edges
    edge_index = []
    cell_offsets = []
    
    # Iterate over each atom
    for i in range(pos.size(0)):
        # Skip atoms from different batches
        batch_mask = batch == batch[i]
        
        # Calculate distances to all other atoms in all neighboring cells
        for cell_offset in neighbor_cells:
            # Skip if this is a self-interaction in the same cell
            if torch.all(cell_offset == 0) and i == i:
                continue
            
            # Calculate position with offset
            offset_cart = torch.matmul(cell_offset.float(), cell)
            other_pos = cart_coords + offset_cart
            
            # Calculate distances
            dists = torch.norm(other_pos - cart_coords[i], dim=1)
            
            # Apply cutoff and batch mask
            mask = (dists < cutoff) & batch_mask
            
            # Add edges
            if torch.any(mask):
                j_indices = torch.where(mask)[0]
                
                # Handle max_neighbors
                if len(j_indices) > max_neighbors:
                    _, top_indices = torch.topk(-dists[mask], max_neighbors)
                    j_indices = j_indices[top_indices]
                
                # Add edges
                edge_index.extend([[i, j] for j in j_indices])
                cell_offsets.extend([cell_offset for _ in j_indices])
    
    # Convert to tensors
    edge_index = torch.tensor(edge_index, device=pos.device).t()
    cell_offsets = torch.stack(cell_offsets, dim=0)
    
    # Count neighbors per atom
    neighbors = torch.zeros(pos.size(0), dtype=torch.long, device=pos.device)
    for i in range(pos.size(0)):
        neighbors[i] = (edge_index[0] == i).sum()
    
    return edge_index, cell_offsets, neighbors


def structure_to_graph(structure, cutoff=6.0, max_neighbors=40, pbc=True, use_z_embedding=False):
    """Convert a pymatgen Structure to a PyG graph with CGCNN features.
    
    Args:
        structure: Pymatgen Structure
        cutoff: Cutoff distance for neighbors
        max_neighbors: Maximum number of neighbors per atom
        pbc: Whether to use periodic boundary conditions
        use_z_embedding: Whether to extract z-coordinates for explicit symmetry breaking
        
    Returns:
        Data: PyG Data object
    """
    # Get positions and atomic numbers
    pos = torch.tensor(structure.cart_coords, dtype=torch.float)
    atomic_numbers = torch.tensor([site.specie.Z for site in structure], dtype=torch.long)
    
    # Get CGCNN features from JARVIS
    node_features = []
    for site in structure:
        element = site.specie.symbol
        try:
            feat = get_node_attributes(element, atom_features="cgcnn")
            node_features.append(feat)
        except Exception as e:
            logger.error("cgcnn_feature_error", element=element, error=str(e))
            # Fallback to simple one-hot if JARVIS fails
            feat = np.zeros(92)  # CGCNN dimension
            feat[0] = 1.0  # Default feature
            node_features.append(feat)
    
    # Convert to tensor
    x = torch.tensor(np.array(node_features), dtype=torch.float)
    
    # Extract z-coordinates if requested (for slabs)
    z_coords = None
    if use_z_embedding:
        # Get z component from cartesian coordinates
        z_coords = torch.tensor(structure.cart_coords[:, 2], dtype=torch.float).unsqueeze(1)
    
    # Create PBC information if needed
    if pbc and structure.lattice:
        cell = torch.tensor(structure.lattice.matrix, dtype=torch.float)
        edge_index, cell_offsets, neighbors = radius_graph_pbc(
            pos, cell, cutoff, max_neighbors
        )
        
        # Calculate distances with PBC
        edge_info = get_distances_pbc(pos, edge_index, cell, cell_offsets)
    else:
        # Use non-PBC radius graph
        from torch_geometric.nn import radius_graph
        edge_index = radius_graph(pos, r=cutoff, max_num_neighbors=max_neighbors)
        
        # Calculate distances
        edge_info = get_distances(pos, edge_index)
        
        # No PBC information
        cell = None
        cell_offsets = None
        neighbors = torch.tensor([max_neighbors] * pos.size(0), dtype=torch.long)
    
    # Apply smooth cutoff to edge features
    distances = edge_info["distances"]
    distance_vec = edge_info["distance_vec"]
    
    # Create smooth cutoff weights
    cutoff_fn = SmoothCutoff()
    edge_weights = cutoff_fn(distances / cutoff)
    
    # Apply weights to distance vectors for smoother feature representation
    # This preserves direction but scales magnitude by smooth cutoff function
    weighted_distance_vec = distance_vec * edge_weights.unsqueeze(-1)
    
    # Create data dictionary
    data_dict = {
        "x": x,
        "edge_index": edge_info["edge_index"],
        "edge_attr": weighted_distance_vec,  # Use weighted vectors for better features
        "pos": pos,
        "atomic_numbers": atomic_numbers,
        "cell": cell,
        "cell_offsets": cell_offsets if cell_offsets is not None else None,
        "neighbors": neighbors,
        "natoms": torch.tensor([len(structure)]),
        "distances": distances,
        "edge_weights": edge_weights  # Store edge weights for possible later use
    }
    
    # Add z-coordinates if requested
    if z_coords is not None:
        data_dict["z_coords"] = z_coords
    
    # Create Data object
    data = Data(**data_dict)
    
    return data


def structure_dict_to_graph(structure_dict, cutoff=6.0, max_neighbors=40, pbc=True, use_z_embedding=False):
    """Convert a structure dictionary to a PyG graph.
    
    Args:
        structure_dict: Structure dictionary
        cutoff: Cutoff distance for neighbors
        max_neighbors: Maximum number of neighbors per atom
        pbc: Whether to use periodic boundary conditions
        use_z_embedding: Whether to extract z-coordinates for explicit symmetry breaking
        
    Returns:
        Data: PyG Data object
    """
    # Convert string to dict if needed
    if isinstance(structure_dict, str):
        import json
        import ast
        try:
            # First try to parse as JSON
            structure_dict = json.loads(structure_dict)
        except json.JSONDecodeError:
            # If that fails, try as Python dict literal
            try:
                structure_dict = ast.literal_eval(structure_dict)
            except (SyntaxError, ValueError) as e:
                logger.error("Failed to parse structure dict", error=str(e))
                raise ValueError(f"Could not parse structure dictionary: {e}")
    
    # Convert to Structure
    try:
        structure = Structure.from_dict(structure_dict)
    except Exception as e:
        logger.error("Failed to convert dict to Structure", error=str(e))
        raise ValueError(f"Invalid structure dictionary: {e}")
    
    # Convert to graph
    return structure_to_graph(structure, cutoff, max_neighbors, pbc, use_z_embedding)