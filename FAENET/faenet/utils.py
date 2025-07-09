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


class SmoothCutoff(torch.nn.Module):
    """Smooth cutoff envelope function.
    
    Implements smooth cutoff as described in DimeNet paper to create
    continuous edge features that go to zero at the cutoff distance.
    """
    def __init__(self, exponent=5):
        """Initialize smooth cutoff function.
        
        Args:
            exponent (int): Exponent for polynomial envelope.
        """
        super(SmoothCutoff, self).__init__()
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
        # Ensure x is within valid range to avoid singularities
        x = torch.clamp(x, min=1e-8)
        
        # Compute polynomial terms
        p, a, b, c = self.p, self.a, self.b, self.c
        x_pow_p0 = x.pow(p - 1)
        x_pow_p1 = x_pow_p0 * x
        x_pow_p2 = x_pow_p1 * x
        
        # Apply envelope function: 1/x + ax^p-1 + bx^p + cx^p+1
        return 1. / x + a * x_pow_p0 + b * x_pow_p1 + c * x_pow_p2


