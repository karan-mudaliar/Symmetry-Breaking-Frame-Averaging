import random
from copy import deepcopy
from itertools import product
import structlog

import torch

# Configure structlog
logger = structlog.get_logger()


def compute_frames(eigenvec, pos, cell, fa_method="random", pos_3D=None, det_index=0):
    """Compute all frames for a given graph.

    Args:
        eigenvec (tensor): eigenvectors matrix
        pos (tensor): centered position vector
        cell (tensor): cell direction (dxd)
        fa_method (str): the Frame Averaging (FA) inspired technique
            chosen to select frames: stochastic-FA (random), deterministic-FA (det),
            Full-FA (all) or SE(3)-FA (se3).
        pos_3D: for 2D FA, pass atoms' 3rd position coordinate.

    Returns:
        list: 3D position tensors of projected representation
    """
    dim = pos.shape[1]  # to differentiate between 2D or 3D case
    plus_minus_list = list(product([1, -1], repeat=dim))
    plus_minus_list = [torch.tensor(x) for x in plus_minus_list]
    all_fa_pos = []
    all_cell = []
    all_rots = []
    assert fa_method in {
        "all",
        "random",
        "det",
        "se3-all",
        "se3-random",
        "se3-det",
    }
    se3 = fa_method in {
        "se3-all",
        "se3-random",
        "se3-det",
    }
    fa_cell = deepcopy(cell)

    if fa_method == "det" or fa_method == "se3-det":
        sum_eigenvec = torch.sum(eigenvec, axis=0)
        plus_minus_list = [torch.where(sum_eigenvec >= 0, 1.0, -1.0)]

    for pm in plus_minus_list:
        # Append new graph positions to list
        new_eigenvec = pm * eigenvec

        # Consider frame if it passes above check
        fa_pos = pos @ new_eigenvec

        if pos_3D is not None:
            full_eigenvec = torch.eye(3)
            fa_pos = torch.cat((fa_pos, pos_3D.unsqueeze(1)), dim=1)
            full_eigenvec[:2, :2] = new_eigenvec
            new_eigenvec = full_eigenvec

        if cell is not None:
            fa_cell = cell @ new_eigenvec

        # Check if determinant is 1 for SE(3) case
        if se3 and not torch.allclose(
            torch.linalg.det(new_eigenvec), torch.tensor(1.0), atol=1e-03
        ):
            continue

        all_fa_pos.append(fa_pos)
        all_cell.append(fa_cell)
        all_rots.append(new_eigenvec.unsqueeze(0))

    # Handle rare case where no R is positive orthogonal
    if all_fa_pos == []:
        all_fa_pos.append(fa_pos)
        all_cell.append(fa_cell)
        all_rots.append(new_eigenvec.unsqueeze(0))

    # Return frame(s) depending on method fa_method
    if fa_method == "all" or fa_method == "se3-all":
        return all_fa_pos, all_cell, all_rots

    elif fa_method == "det" or fa_method == "se3-det":
        return [all_fa_pos[det_index]], [all_cell[det_index]], [all_rots[det_index]]

    index = random.randint(0, len(all_fa_pos) - 1)
    return [all_fa_pos[index]], [all_cell[index]], [all_rots[index]]


def check_constraints(eigenval, eigenvec, dim=3):
    """Check requirements for frame averaging are satisfied

    Args:
        eigenval (tensor): eigenvalues
        eigenvec (tensor): eigenvectors
        dim (int): 2D or 3D frame averaging
    """
    # Check eigenvalues are different
    if dim == 3:
        if (eigenval[1] / eigenval[0] > 0.90) or (eigenval[2] / eigenval[1] > 0.90):
            logger.warn("similar_eigenvalues", dimension=dim, eigenvalues=eigenval.tolist())
    else:
        if eigenval[1] / eigenval[0] > 0.90:
            logger.warn("similar_eigenvalues", dimension=dim, eigenvalues=eigenval.tolist())

    # Check eigenvectors are orthonormal
    if not torch.allclose(eigenvec @ eigenvec.T, torch.eye(dim), atol=1e-03):
        logger.warn("non_orthogonal_matrix", dimension=dim)

    # Check determinant of eigenvectors is 1
    det = torch.linalg.det(eigenvec).item()
    if not torch.allclose(torch.linalg.det(eigenvec), torch.tensor(1.0), atol=1e-03):
        logger.warn("determinant_not_one", determinant=det)


def frame_averaging_3D(pos, cell=None, fa_method="random", check=False):
    """Computes new positions for the graph atoms using PCA

    Args:
        pos (tensor): positions of atoms in the graph
        cell (tensor): unit cell of the graph
        fa_method (str): FA method used
            (random, det, all, se3-all, se3-det, se3-random)
        check (bool): check if constraints are satisfied. Default: False.

    Returns:
        tensor: updated atom positions
        tensor: updated unit cell
        tensor: the rotation matrix used (PCA)
    """

    # Compute centroid and covariance
    pos = pos - pos.mean(dim=0, keepdim=True)
    C = torch.matmul(pos.t(), pos)

    # Eigendecomposition
    eigenval, eigenvec = torch.linalg.eigh(C)

    # Sort, if necessary
    idx = eigenval.argsort(descending=True)
    eigenvec = eigenvec[:, idx]
    eigenval = eigenval[idx]

    # Check if constraints are satisfied
    if check:
        check_constraints(eigenval, eigenvec, 3)

    # Compute fa_pos
    fa_pos, fa_cell, fa_rot = compute_frames(eigenvec, pos, cell, fa_method)

    # No need to update distances, they are preserved.

    return fa_pos, fa_cell, fa_rot


def frame_averaging_2D(pos, cell=None, fa_method="random", check=False):
    """Computes new positions for the graph atoms,
    based on a frame averaging building on PCA.

    Args:
        pos (tensor): positions of atoms in the graph
        cell (tensor): unit cell of the graph
        fa_method (str): FA method used (random, det, all, se3)
        check (bool): check if constraints are satisfied. Default: False.

    Returns:
        tensor: updated atom positions
        tensor: updated unit cell
        tensor: the rotation matrix used (PCA)
    """

    # Compute centroid and covariance
    pos_2D = pos[:, :2] - pos[:, :2].mean(dim=0, keepdim=True)
    C = torch.matmul(pos_2D.t(), pos_2D)

    # Eigendecomposition
    eigenval, eigenvec = torch.linalg.eigh(C)
    # Sort eigenvalues
    idx = eigenval.argsort(descending=True)
    eigenval = eigenval[idx]
    eigenvec = eigenvec[:, idx]

    # Check if constraints are satisfied
    if check:
        check_constraints(eigenval, eigenvec, 3)

    # Compute all frames
    fa_pos, fa_cell, fa_rot = compute_frames(
        eigenvec, pos_2D, cell, fa_method, pos[:, 2]
    )
    # No need to update distances, they are preserved.

    return fa_pos, fa_cell, fa_rot


class RandomRotate:
    """Randomly rotate a graph to perform data augmentation.
    
    Args:
        angle_range (list): Range of rotation angles in degrees
        axes (list): Axes to rotate around (0=x, 1=y, 2=z)
    """
    
    def __init__(self, angle_range=[-180, 180], axes=[0, 1, 2]):
        self.angle_range = angle_range
        self.axes = axes
    
    def __call__(self, data):
        # Get random rotation angle for each axis
        angles = [random.uniform(*self.angle_range) for _ in self.axes]
        
        # Create rotation matrix
        rot = torch.eye(3)
        for axis, angle in zip(self.axes, angles):
            angle_rad = angle * (3.14159 / 180.0)
            c, s = torch.cos(torch.tensor(angle_rad)), torch.sin(torch.tensor(angle_rad))
            
            if axis == 0:  # X axis
                rot_mat = torch.tensor([[1, 0, 0], [0, c, -s], [0, s, c]])
            elif axis == 1:  # Y axis
                rot_mat = torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]])
            else:  # Z axis
                rot_mat = torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            
            rot = rot @ rot_mat
        
        # Apply rotation to positions
        if hasattr(data, 'pos'):
            pos = data.pos.view(-1, 3)
            data.pos = torch.matmul(pos, rot.t())
        
        # Apply rotation to cell if it exists
        if hasattr(data, 'cell') and data.cell is not None:
            data.cell = torch.matmul(data.cell, rot.t())
        
        return data, rot, rot.t()


class ConsistencyLoss(torch.nn.Module):
    """Computes variance across frame predictions for equivariance regularization.
    
    This loss function calculates how consistent model predictions are across different
    reference frames of the same structure. Lower variance means better equivariance.
    """
    def __init__(self, normalize=True):
        """
        Args:
            normalize: Whether to normalize the loss by the squared magnitude of predictions.
        """
        super().__init__()
        self.normalize = normalize
        
    def forward(self, frame_predictions):
        """
        Args:
            frame_predictions: List of tensors with predictions for each frame.
                              Each tensor has shape [batch_size, output_dim].
        
        Returns:
            Loss tensor (scalar) representing the mean variance across frames.
        """
        # Validate input
        num_frames = len(frame_predictions)
        if num_frames == 0:
            logger.error("empty_frame_predictions", len=0)
            return torch.tensor(0.0, requires_grad=True, device=self._get_device(frame_predictions))
        
        # Stack predictions from all frames
        preds = torch.stack(frame_predictions)  # shape: [num_frames, batch_size, output_dim]
        
        # Calculate variance across frames (dim=0) for each object independently
        # Use unbiased=False to get the mathematical variance formula sum((x-mean)^2)/n
        variance_per_object = torch.var(preds, dim=0, unbiased=False)  # shape: [batch_size, output_dim]
        
        # Average across the output dimension for each object
        variance_per_object = variance_per_object.mean(dim=-1)  # shape: [batch_size]
        
        # Normalize by squared magnitude of predictions if requested
        if self.normalize:
            # Square of predictions, averaged across frames and output dimensions, plus small epsilon
            norm_factor = (preds**2).mean(dim=[0, 2]) + 1e-6
            variance_per_object = variance_per_object / norm_factor
        
        # Average across all objects in the batch
        mean_variance = variance_per_object.mean()  # scalar
        
        return mean_variance
    
    def _get_device(self, frame_predictions):
        """Helper to get the device from frame predictions."""
        if not frame_predictions:
            return torch.device('cpu')
        return frame_predictions[0].device


# Legacy function for backward compatibility
def compute_consistency_loss(frame_predictions, normalize=True):
    """Legacy function that uses ConsistencyLoss class."""
    loss_fn = ConsistencyLoss(normalize=normalize)
    return loss_fn(frame_predictions)


def data_augmentation(g, d=3, *args):
    """Data augmentation where we randomly rotate each graph
    in the dataloader transform

    Args:
        g (data.Data): single graph
        d (int): dimension of the DA rotation (2D around z-axis or 3D)

    Returns:
        (data.Data): rotated graph
    """

    # Sampling a random rotation within [-180, 180] for all axes.
    if d == 3:
        transform = RandomRotate([-180, 180], [0, 1, 2])  # 3D
    else:
        transform = RandomRotate([-180, 180], [2])  # 2D around z-axis

    # Rotate graph
    graph_rotated, _, _ = transform(g)

    return graph_rotated