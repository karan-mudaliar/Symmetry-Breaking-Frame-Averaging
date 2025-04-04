import torch
import structlog

# Initialize a default logger
logger = structlog.get_logger()

def swish(x):
    """Swish activation function: x * sigmoid(x)"""
    return x * torch.sigmoid(x)


def generate_run_name():
    """Generate a memorable run name for experiment tracking.
    
    Returns:
        str: A unique, memorable name for the run (e.g., "atomic-elephant")
    """
    try:
        from coolname import generate_slug
        return generate_slug(2)
    except ImportError:
        import random
        import time
        # Fallback if coolname is not installed
        adjectives = ["bold", "calm", "wise", "swift", "keen"]
        nouns = ["tiger", "eagle", "wolf", "dolphin", "fox"]
        return f"{random.choice(adjectives)}-{random.choice(nouns)}-{int(time.time()) % 10000}"


def get_distances(pos, edge_index):
    """Get distances between atoms based on edge index.
    
    Args:
        pos (tensor): atom positions
        edge_index (tensor): connectivity matrix in COO format
        
    Returns:
        dict: distances, relative positions, and edge index
    """
    row, col = edge_index
    
    # Get relative positions
    rel_pos = pos[row] - pos[col]
    
    # Calculate distances
    distances = torch.norm(rel_pos, dim=-1)
    
    return {
        "edge_index": edge_index,
        "distances": distances,
        "distance_vec": rel_pos
    }


def conditional_grad(torch_enabled=True):
    """Function decorator to enable/disable gradient calculation"""
    
    def decorate(func):
        def wrapper(*args, **kwargs):
            if torch_enabled:
                return func(*args, **kwargs)
            else:
                with torch.no_grad():
                    return func(*args, **kwargs)
        return wrapper
    return decorate


class GaussianSmearing(torch.nn.Module):
    """Smears a distance distribution by a Gaussian function."""

    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


