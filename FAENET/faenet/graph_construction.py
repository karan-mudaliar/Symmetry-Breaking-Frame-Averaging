"""
Enhanced graph construction module for crystal structures.
Integrates Comformer-inspired representations with frame averaging capabilities.
"""
import numpy as np
import torch
from torch_geometric.data import Data
import torch_scatter
from pymatgen.core import Structure
import structlog

# Configure structlog
logger = structlog.get_logger()

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


def angle_from_array(a, b, lattice):
    """Compute angle between two vectors in the lattice space.
    
    Args:
        a (np.ndarray): First vector in fractional coordinates
        b (np.ndarray): Second vector in fractional coordinates
        lattice (np.ndarray): Lattice matrix (3x3)
        
    Returns:
        float: Angle in degrees
    """
    a_new = np.dot(a, lattice)
    b_new = np.dot(b, lattice)
    value = np.sum(a_new * b_new)
    length = (np.sum(a_new ** 2) ** 0.5) * (np.sum(b_new ** 2) ** 0.5)
    cos = value / length
    # Handle numerical instability
    cos = min(1.0, max(-1.0, cos))
    angle = np.arccos(cos)
    return angle / np.pi * 180.0


def correct_coord_sys(a, b, c, lattice):
    """Check if vectors form a right-handed coordinate system.
    
    Args:
        a (np.ndarray): First vector in fractional coordinates
        b (np.ndarray): Second vector in fractional coordinates
        c (np.ndarray): Third vector in fractional coordinates
        lattice (np.ndarray): Lattice matrix (3x3)
        
    Returns:
        bool: True if it's a right-handed coordinate system
    """
    a_new = np.dot(a, lattice)
    b_new = np.dot(b, lattice)
    c_new = np.dot(c, lattice)
    
    plane_vec = np.cross(a_new, b_new)
    value = np.sum(plane_vec * c_new)
    length = (np.sum(plane_vec ** 2) ** 0.5) * (np.sum(c_new ** 2) ** 0.5)
    cos = value / length
    angle = np.arccos(cos)
    return (angle / np.pi * 180.0 <= 90.0)


def same_line(a, b):
    """Check if two vectors are collinear.
    
    Args:
        a (np.ndarray): First vector
        b (np.ndarray): Second vector
        
    Returns:
        bool: True if vectors are collinear
    """
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    
    if a_norm < 1e-6 or b_norm < 1e-6:
        return False
        
    a_new = a / a_norm
    b_new = b / b_norm
    
    dot_product = np.sum(a_new * b_new)
    return abs(abs(dot_product) - 1.0) < 1e-5


def same_plane(a, b, c):
    """Check if three vectors are coplanar.
    
    Args:
        a (np.ndarray): First vector
        b (np.ndarray): Second vector
        c (np.ndarray): Third vector
        
    Returns:
        bool: True if vectors are coplanar
    """
    return abs(np.dot(np.cross(a, b), c)) < 1e-5


def find_lattice_vectors(atoms, cutoff=8.0):
    """Find principal lattice vectors using nearest-neighbor analysis.
    Inspired by Comformer's approach to identify invariant coordinate system.
    
    Args:
        atoms: Pymatgen Structure
        cutoff: Cutoff distance for neighbors
        
    Returns:
        tuple: Three principal lattice vectors (lat1, lat2, lat3) as numpy arrays
    """
    # Use extended cutoff to find lattice vectors
    r_cut = max(atoms.lattice.a, atoms.lattice.b, atoms.lattice.c) + 1e-2
    
    # Get neighbors of the first atom
    all_neighbors = atoms.get_all_neighbors(r=r_cut)
    if not all_neighbors or len(all_neighbors[0]) < 3:
        logger.warn("insufficient_neighbors_for_lattice_vectors", 
                   available=len(all_neighbors[0]) if all_neighbors else 0,
                   needed=3)
        # Fallback to unit vectors
        return np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])
    
    # Sort neighbors by distance
    neighborlist = sorted(all_neighbors[0], key=lambda x: x[2])
    
    # Extract images (periodic cell offsets) from neighbors with same atomic number
    atoms_with_same_element = [n for n in neighborlist if n[0].specie == atoms[0].specie]
    
    if len(atoms_with_same_element) < 3:
        atoms_with_same_element = neighborlist  # Fallback to all neighbors
    
    # Extract images (should be 3D vectors)
    images = np.array([nbr[3] for nbr in atoms_with_same_element])
    
    if len(images) < 3:
        logger.warn("insufficient_periodic_images", count=len(images))
        # Fallback to unit vectors
        return np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])
    
    # Find first lattice vector (shortest path to same element)
    lat1 = images[0]
    
    # Find second lattice vector (not collinear with first)
    lat2 = None
    for i in range(1, len(images)):
        if not same_line(lat1, images[i]):
            lat2 = images[i]
            break
    
    if lat2 is None:
        logger.warn("could_not_find_independent_second_lattice_vector")
        # Fallback to orthogonal vector
        if abs(lat1[0]) > 1e-6 or abs(lat1[1]) > 1e-6:
            lat2 = np.array([-lat1[1], lat1[0], 0])
        else:
            lat2 = np.array([1, 0, 0])
    
    # Find third lattice vector (not coplanar with first two)
    lat3 = None
    for i in range(1, len(images)):
        if not same_plane(lat1, lat2, images[i]):
            lat3 = images[i]
            break
    
    if lat3 is None:
        logger.warn("could_not_find_independent_third_lattice_vector")
        # Fallback to cross product
        lat3 = np.cross(lat1, lat2)
        if np.linalg.norm(lat3) < 1e-6:
            lat3 = np.array([0, 0, 1])
    
    # Make sure the angles are all less than 90 degrees (acute)
    lattice_matrix = atoms.lattice.matrix
    if angle_from_array(lat1, lat2, lattice_matrix) > 90.0:
        lat2 = -lat2
        
    if angle_from_array(lat1, lat3, lattice_matrix) > 90.0:
        lat3 = -lat3
        
    # Make sure it's a right-handed coordinate system
    if not correct_coord_sys(lat1, lat2, lat3, lattice_matrix):
        lat1, lat2, lat3 = -lat1, -lat2, -lat3
        
    return lat1, lat2, lat3


def canonize_edge(src_id, dst_id, src_image, dst_image):
    """Compute canonical edge representation to ensure consistency.
    
    Args:
        src_id: Source atom ID
        dst_id: Destination atom ID
        src_image: Source atom image (periodic offset)
        dst_image: Destination atom image (periodic offset)
        
    Returns:
        tuple: Canonical (src_id, dst_id, src_image, dst_image)
    """
    # Store directed edges src_id <= dst_id
    if dst_id < src_id:
        src_id, dst_id = dst_id, src_id
        src_image, dst_image = dst_image, src_image

    # Shift periodic images so that src is in (0,0,0) image
    if not np.array_equal(src_image, np.array([0, 0, 0])):
        shift = src_image
        src_image = tuple(np.subtract(src_image, shift))
        dst_image = tuple(np.subtract(dst_image, shift))

    assert src_image == (0, 0, 0)

    return src_id, dst_id, src_image, dst_image


def nearest_neighbor_edges_enhanced(atoms, cutoff=8.0, max_neighbors=12, use_canonize=True):
    """Build k-nearest neighbor edges with enhanced lattice correction.
    Implements the Comformer approach to find neighbors and correct lattice vectors.
    
    Args:
        atoms: Pymatgen Structure
        cutoff: Cutoff distance for neighbors
        max_neighbors: Maximum number of neighbors per atom
        use_canonize: Whether to canonize edges
        
    Returns:
        tuple: (edges, lat1, lat2, lat3) where edges is a dictionary of edges
    """
    # Find lattice vectors using the comformer approach
    lat1, lat2, lat3 = find_lattice_vectors(atoms, cutoff)
    
    # Get all neighbors within cutoff
    all_neighbors_now = atoms.get_all_neighbors(r=cutoff)
    
    # Check if we have enough neighbors for each atom
    min_nbrs = min(len(neighborlist) for neighborlist in all_neighbors_now)
    
    # If we don't have enough neighbors, increase cutoff and try again
    attempt = 0
    while min_nbrs < max_neighbors and attempt < 3:
        r_cut = cutoff * 1.5
        logger.info("increasing_cutoff_for_neighbor_search", 
                   attempt=attempt+1, 
                   new_cutoff=r_cut)
        all_neighbors_now = atoms.get_all_neighbors(r=r_cut)
        min_nbrs = min(len(neighborlist) for neighborlist in all_neighbors_now)
        attempt += 1
        cutoff = r_cut
    
    # Build edges dictionary
    edges = {}
    
    # Process each atom and its neighbors
    for site_idx, neighborlist in enumerate(all_neighbors_now):
        # Sort neighbors by distance
        neighborlist = sorted(neighborlist, key=lambda x: x[2])
        
        # Get distances, indices and images
        distances = np.array([nbr[2] for nbr in neighborlist])
        ids = np.array([nbr[1] for nbr in neighborlist])
        images = np.array([nbr[3] for nbr in neighborlist])
        
        # Find the distance to the k-th nearest neighbor
        if len(distances) > max_neighbors:
            max_dist = distances[max_neighbors - 1]
            mask = distances <= max_dist
            ids = ids[mask]
            images = images[mask]
            distances = distances[mask]
        
        # Add edges to dictionary
        for dst, image in zip(ids, images):
            if use_canonize:
                src_id, dst_id, src_image, dst_image = canonize_edge(
                    site_idx, dst, (0, 0, 0), tuple(image)
                )
                key = (src_id, dst_id)
                if key not in edges:
                    edges[key] = set()
                edges[key].add(dst_image)
            else:
                key = (site_idx, dst)
                if key not in edges:
                    edges[key] = set()
                edges[key].add(tuple(image))
        
        # Add lattice vectors as self-loops for each atom (Comformer approach)
        key = (site_idx, site_idx)
        if key not in edges:
            edges[key] = set()
        edges[key].add(tuple(lat1))
        edges[key].add(tuple(lat2))
        edges[key].add(tuple(lat3))
    
    return edges, lat1, lat2, lat3


def build_undirected_edgedata(atoms, edges, lat1, lat2, lat3):
    """Build undirected graph data from edge set with enhanced attributes.
    
    Args:
        atoms: Pymatgen Structure
        edges: Dictionary mapping (src_id, dst_id) to set of dst_image
        lat1, lat2, lat3: Principal lattice vectors
        
    Returns:
        tuple: (u, v, r, nei, atom_lat) edge data for graph construction
    """
    u, v, r, nei, atom_lat = [], [], [], [], []
    
    # Convert lattice vectors to cartesian
    v1 = atoms.lattice.get_cartesian_coords(lat1)
    v2 = atoms.lattice.get_cartesian_coords(lat2)
    v3 = atoms.lattice.get_cartesian_coords(lat3)
    
    # Store lattice vectors for each atom (used by the model for spatial awareness)
    atom_lat.append([v1, v2, v3])
    
    # Build undirected edges
    for (src_id, dst_id), images in edges.items():
        for dst_image in images:
            # Get fractional coordinate for periodic image of dst
            dst_coord = atoms.frac_coords[dst_id] + dst_image
            
            # Calculate cartesian displacement vector from src to dst
            d = atoms.lattice.get_cartesian_coords(
                dst_coord - atoms.frac_coords[src_id]
            )
            
            # Add edges in both directions for undirected graph
            for uu, vv, dd in [(src_id, dst_id, d), (dst_id, src_id, -d)]:
                u.append(uu)
                v.append(vv)
                r.append(dd)
                # Store lattice vectors for each edge
                nei.append([v1, v2, v3])
    
    # Convert to tensors
    u = torch.tensor(u, dtype=torch.long)
    v = torch.tensor(v, dtype=torch.long)
    r = torch.tensor(r, dtype=torch.float)
    nei = torch.tensor(nei, dtype=torch.float)
    atom_lat = torch.tensor(atom_lat, dtype=torch.float)
    
    return u, v, r, nei, atom_lat


def radius_graph_pbc(pos, cell, cutoff, max_neighbors=32, batch=None):
    """Build a radius graph with periodic boundary conditions.
    
    Args:
        pos (tensor): Atom positions [num_atoms, 3]
        cell (tensor): Unit cell [3, 3]
        cutoff (float): Cutoff distance
        max_neighbors (int): Maximum number of neighbors per atom
        batch (tensor): Batch indices for atoms
        
    Returns:
        tuple: (edge_index, cell_offsets, neighbors, nei, atom_lat)
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
    
    # Add lattice information (optional, not used in original implementation)
    # Extract principal lattice vectors from the cell
    v1 = cell[0].reshape(1, 3).repeat(pos.size(0), 1, 1)
    v2 = cell[1].reshape(1, 3).repeat(pos.size(0), 1, 1)
    v3 = cell[2].reshape(1, 3).repeat(pos.size(0), 1, 1)
    
    # Store lattice vectors for each atom
    atom_lat = torch.cat([v1, v2, v3], dim=1).reshape(1, 3, 3)
    
    # Store lattice vectors for each edge
    nei = atom_lat.repeat(edge_index.size(1), 1, 1)
    
    return edge_index, cell_offsets, neighbors, nei, atom_lat


def get_atom_features(atomic_number, feature_type="cgcnn"):
    """Get atom features based on atomic number and feature type.
    
    Args:
        atomic_number: Atomic number
        feature_type: Type of features to use (one_hot, cgcnn)
        
    Returns:
        tensor: Atom features
    """
    if feature_type == "one_hot":
        # Simple one-hot encoding (limited information)
        return torch.nn.functional.one_hot(torch.tensor(atomic_number), num_classes=100).float()
    
    elif feature_type == "cgcnn":
        # CGCNN-style features (richer representation)
        # Based on commonly used atomic properties
        
        # Define feature sets for elements (most common elements in materials)
        # Format: [period, atomic_weight, covalent_radius, electronegativity, 
        #          ionization_energy, electron_affinity, specific_heat, 
        #          valence_electrons, group]
        elem_features = {
            # H
            1: [1, 1.0079, 0.31, 2.2, 1312.0, 0.754, 14.01, 1, 1],
            # He
            2: [1, 4.0026, 0.28, 0.0, 2372.3, 0.0, 5.19, 2, 18],
            # Li
            3: [2, 6.941, 1.52, 0.98, 520.2, 0.618, 29.12, 1, 1],
            # Be
            4: [2, 9.0122, 1.12, 1.57, 899.5, 0.436, 16.43, 2, 2],
            # B
            5: [2, 10.811, 0.88, 2.04, 800.6, 1.026, 12.06, 3, 13],
            # C
            6: [2, 12.0107, 0.77, 2.55, 1086.5, 1.576, 6.8, 4, 14],
            # N 
            7: [2, 14.0067, 0.71, 3.04, 1402.3, 1.795, 5.57, 5, 15],
            # O
            8: [2, 15.9994, 0.66, 3.44, 1313.9, 2.002, 5.17, 6, 16],
            # F
            9: [2, 18.9984, 0.57, 3.98, 1681.0, 2.222, 5.32, 7, 17],
            # Ne
            10: [2, 20.1797, 0.58, 0.0, 2080.7, 0.0, 1.03, 8, 18],
            # Na
            11: [3, 22.9897, 1.86, 0.93, 495.8, 0.548, 23.68, 1, 1],
            # Mg
            12: [3, 24.305, 1.60, 1.31, 737.7, 0.737, 13.97, 2, 2],
            # Al
            13: [3, 26.9815, 1.43, 1.61, 577.5, 0.969, 10.0, 3, 13],
            # Si
            14: [3, 28.0855, 1.32, 1.9, 786.5, 1.176, 9.81, 4, 14],
            # P
            15: [3, 30.9738, 1.28, 2.19, 1011.8, 1.307, 8.73, 5, 15],
            # S
            16: [3, 32.065, 1.27, 2.58, 999.6, 1.448, 8.95, 6, 16],
            # Cl
            17: [3, 35.453, 0.97, 3.16, 1251.2, 1.594, 7.76, 7, 17],
            # Ar
            18: [3, 39.948, 0.71, 0.0, 1520.6, 0.0, 5.20, 8, 18],
            # K
            19: [4, 39.0983, 2.27, 0.82, 418.8, 0.452, 45.46, 1, 1],
            # Ca
            20: [4, 40.078, 1.97, 1.0, 589.8, 0.595, 21.1, 2, 2],
            # Sc
            21: [4, 44.9559, 1.60, 1.36, 633.1, 0.188, 10.9, 2, 3],
            # Ti
            22: [4, 47.867, 1.47, 1.54, 658.8, 0.543, 9.21, 2, 4],
            # V
            23: [4, 50.9415, 1.33, 1.63, 650.9, 0.525, 7.47, 2, 5],
            # Cr
            24: [4, 51.9961, 1.25, 1.66, 652.9, 0.498, 7.23, 1, 6],
            # Mn
            25: [4, 54.938, 1.37, 1.55, 717.3, 0.477, 7.43, 2, 7],
            # Fe
            26: [4, 55.845, 1.26, 1.83, 762.5, 0.492, 7.1, 2, 8],
            # Co
            27: [4, 58.9332, 1.25, 1.88, 760.4, 0.445, 6.7, 2, 9],
            # Ni
            28: [4, 58.6934, 1.24, 1.91, 737.1, 0.444, 6.6, 2, 10],
            # Cu
            29: [4, 63.546, 1.28, 1.9, 745.5, 0.385, 7.1, 1, 11],
            # Zn
            30: [4, 65.38, 1.37, 1.65, 906.4, 0.387, 7.1, 2, 12],
            # Ga
            31: [4, 69.723, 1.53, 1.81, 578.8, 0.444, 11.8, 3, 13],
            # Ge
            32: [4, 72.64, 1.22, 2.01, 762.0, 0.375, 13.6, 4, 14],
            # As
            33: [4, 74.9216, 1.21, 2.18, 947.0, 0.328, 13.1, 5, 15],
            # Se
            34: [4, 78.96, 1.16, 2.55, 941.0, 0.321, 16.5, 6, 16],
            # Br
            35: [4, 79.904, 1.14, 2.96, 1139.9, 0.293, 13.8, 7, 17],
            # Kr
            36: [4, 83.798, 0.89, 0.0, 1350.8, 0.0, 5.20, 8, 18],
            # Rb
            37: [5, 85.4678, 2.43, 0.82, 403.0, 0.363, 55.9, 1, 1],
            # Sr
            38: [5, 87.62, 2.15, 0.95, 549.5, 0.301, 33.7, 2, 2],
            # Y
            39: [5, 88.9059, 1.81, 1.22, 600.0, 0.307, 17.2, 2, 3],
            # Zr
            40: [5, 91.224, 1.59, 1.33, 640.1, 0.278, 17.9, 2, 4],
            # Nb
            41: [5, 92.9064, 1.43, 1.6, 652.1, 0.268, 15.7, 1, 5],
            # Mo
            42: [5, 95.96, 1.37, 2.16, 684.3, 0.251, 12.5, 1, 6],
            # Tc
            43: [5, 98.0, 1.35, 1.9, 702.0, 0.322, 8.0, 2, 7],
            # Ru
            44: [5, 101.07, 1.34, 2.2, 710.2, 0.224, 6.4, 1, 8],
            # Rh
            45: [5, 102.9055, 1.34, 2.28, 719.7, 0.222, 6.2, 1, 9],
            # Pd
            46: [5, 106.42, 1.37, 2.2, 804.4, 0.557, 6.1, 0, 10],
            # Ag
            47: [5, 107.8682, 1.45, 1.93, 731.0, 0.237, 18.3, 1, 11],
            # Cd
            48: [5, 112.411, 1.57, 1.69, 867.8, 0.232, 16.5, 2, 12],
            # In
            49: [5, 114.818, 1.66, 1.78, 558.3, 0.233, 15.7, 3, 13],
            # Sn
            50: [5, 118.71, 1.55, 1.96, 708.6, 0.228, 16.3, 4, 14],
            # Sb
            51: [5, 121.76, 1.45, 2.05, 834.0, 0.207, 18.2, 5, 15],
            # Te
            52: [5, 127.6, 1.43, 2.1, 869.3, 0.202, 20.5, 6, 16],
            # I
            53: [5, 126.9045, 1.33, 2.66, 1008.4, 0.214, 25.7, 7, 17],
            # Xe
            54: [5, 131.293, 1.08, 0.0, 1170.4, 0.0, 6.2, 8, 18],
            # Cs
            55: [6, 132.9055, 2.62, 0.79, 375.7, 0.242, 70.0, 1, 1],
            # Ba
            56: [6, 137.327, 2.17, 0.89, 502.9, 0.204, 39.0, 2, 2],
            # La
            57: [6, 138.9055, 2.08, 1.1, 538.1, 0.195, 28.2, 2, 3],
            # Ce
            58: [6, 140.116, 1.99, 1.12, 534.4, 0.192, 28.2, 2, 4],
            # Hf
            72: [6, 178.49, 1.59, 1.3, 658.5, 0.146, 16.2, 2, 4],
            # Ta
            73: [6, 180.9479, 1.43, 1.5, 761.0, 0.140, 19.2, 2, 5],
            # W
            74: [6, 183.84, 1.37, 2.36, 770.0, 0.132, 18.3, 2, 6],
            # Re
            75: [6, 186.207, 1.37, 1.9, 760.0, 0.137, 19.3, 1, 7],
            # Os
            76: [6, 190.23, 1.34, 2.2, 840.0, 0.125, 8.54, 2, 8],
            # Ir
            77: [6, 192.217, 1.35, 2.2, 880.0, 0.151, 8.2, 3, 9],
            # Pt
            78: [6, 195.084, 1.38, 2.28, 870.0, 0.205, 9.1, 1, 10],
            # Au
            79: [6, 196.9666, 1.44, 2.54, 890.1, 0.129, 12.6, 1, 11],
            # Hg
            80: [6, 200.59, 1.55, 2.0, 1007.1, 0.140, 13.6, 2, 12],
            # Tl
            81: [6, 204.3833, 1.71, 2.04, 589.4, 0.129, 17.2, 3, 13],
            # Pb
            82: [6, 207.2, 1.75, 2.33, 715.6, 0.159, 18.3, 4, 14],
            # Bi
            83: [6, 208.9804, 1.70, 2.02, 703.0, 0.122, 21.3, 5, 15],
            # Po
            84: [6, 209.0, 1.40, 2.0, 812.0, 0.183, 22.1, 6, 16],
            # At
            85: [6, 210.0, 1.50, 2.2, 930.0, 0.270, 23.5, 7, 17],
            # Rn
            86: [6, 222.0, 1.45, 0.0, 1037.0, 0.0, 4.4, 8, 18],
            # Ra
            88: [7, 226.0, 2.15, 0.9, 509.3, 0.1, 31.0, 2, 2],
            # Ac
            89: [7, 227.0, 2.08, 1.1, 499.0, 0.35, 27.2, 2, 3],
            # Th
            90: [7, 232.0381, 1.99, 1.3, 587.0, 0, 16.11, 2, 4],
            # Pa
            91: [7, 231.0359, 1.63, 1.5, 568.0, 0, 12.3, 2, 5],
            # U
            92: [7, 238.0289, 1.38, 1.38, 597.6, 0.115, 15.0, 3, 6],
        }
        
        # Calculate average features for elements not in the list
        # This way, unknown elements get average properties instead of zeros
        if atomic_number not in elem_features:
            # Log the missing element
            logger.info("using_average_features_for_element", atomic_number=atomic_number)
            
            # Calculate average features (excluding zeros)
            all_features = np.array(list(elem_features.values()))
            
            # Period - use floor(atomic_number / 18) + 1 as estimate
            period = min(7, max(1, int(atomic_number / 18) + 1))
            
            # Reasonable average values for other properties
            avg_features = [
                period,                          # Period (estimated)
                np.mean(all_features[:, 1]),     # Average atomic weight
                np.mean(all_features[:, 2]),     # Average covalent radius
                np.mean(all_features[:, 3]),     # Average electronegativity
                np.mean(all_features[:, 4]),     # Average ionization energy
                np.mean(all_features[:, 5]),     # Average electron affinity
                np.mean(all_features[:, 6]),     # Average specific heat
                atomic_number % 18,              # Estimated valence electrons 
                min(18, atomic_number % 18)      # Estimated group
            ]
            
            return torch.tensor(avg_features, dtype=torch.float)
        
        # Return features for the atomic number
        return torch.tensor(elem_features[atomic_number], dtype=torch.float)
    
    else:
        # Default to one-hot encoding
        return torch.nn.functional.one_hot(torch.tensor(atomic_number), num_classes=100).float()


def structure_to_graph(structure, cutoff=6.0, max_neighbors=40, pbc=True, atom_features="cgcnn"):
    """Convert a pymatgen Structure to a PyG graph with enhanced features.
    
    Args:
        structure: Pymatgen Structure
        cutoff: Cutoff distance for neighbors
        max_neighbors: Maximum number of neighbors per atom
        pbc: Whether to use periodic boundary conditions
        atom_features: Type of atom features ("one_hot" or "cgcnn")
        
    Returns:
        Data: PyG Data object
    """
    # Get positions and atomic numbers
    pos = torch.tensor(structure.cart_coords, dtype=torch.float)
    atomic_numbers = torch.tensor([site.specie.Z for site in structure], dtype=torch.long)
    
    # Get atom features using enhanced representation
    if atom_features == "cgcnn":
        # Use CGCNN-style features for richer representation
        x = torch.stack([get_atom_features(z.item(), "cgcnn") for z in atomic_numbers])
    else:
        # Default to one-hot encoding
        x = torch.nn.functional.one_hot(atomic_numbers, num_classes=100).float()
    
    # Create PBC information if needed
    if pbc and structure.lattice:
        try:
            # Use Comformer approach for enhanced graph construction
            np_atoms = np.array(structure.cart_coords, dtype=np.float32)
            edges, lat1, lat2, lat3 = nearest_neighbor_edges_enhanced(
                structure, cutoff, max_neighbors, use_canonize=True
            )
            u, v, rel_pos, nei, atom_lat = build_undirected_edgedata(
                structure, edges, lat1, lat2, lat3
            )
            
            # Create edge index in PyG format
            edge_index = torch.stack([u, v], dim=0)
            
            # Extract cell from structure
            cell = torch.tensor(structure.lattice.matrix, dtype=torch.float)
            
            # Create cell offsets from edges
            cell_offsets = torch.zeros((edge_index.size(1), 3), dtype=torch.float)
            
            # Calculate distances
            distances = torch.norm(rel_pos, dim=1)
            
        except Exception as e:
            # Fallback to standard radius graph if Comformer approach fails
            logger.warn("comformer_graph_construction_failed", error=str(e))
            
            cell = torch.tensor(structure.lattice.matrix, dtype=torch.float)
            edge_index, cell_offsets, neighbors, nei, atom_lat = radius_graph_pbc(
                pos, cell, cutoff, max_neighbors
            )
            
            # Calculate distances with PBC
            edge_info = get_distances_pbc(pos, edge_index, cell, cell_offsets)
            distances = edge_info["distances"]
            rel_pos = edge_info["distance_vec"]
    else:
        # Use non-PBC radius graph
        from torch_geometric.nn import radius_graph
        edge_index = radius_graph(pos, r=cutoff, max_num_neighbors=max_neighbors)
        
        # Calculate distances
        edge_info = get_distances(pos, edge_index)
        distances = edge_info["distances"]
        rel_pos = edge_info["distance_vec"]
        
        # No PBC information
        cell = None
        cell_offsets = None
        neighbors = torch.tensor([max_neighbors] * pos.size(0), dtype=torch.long)
        
        # Create dummy lattice vectors
        v1 = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float).reshape(1, 3)
        v2 = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float).reshape(1, 3)
        v3 = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float).reshape(1, 3)
        
        # Stack lattice vectors
        nei = torch.cat([v1, v2, v3], dim=0).unsqueeze(0).repeat(edge_index.size(1), 1, 1)
        atom_lat = torch.cat([v1, v2, v3], dim=0).unsqueeze(0)
    
    # Calculate edge length and unit vectors (useful for angle calculations)
    edge_length = torch.norm(rel_pos, dim=1, keepdim=True)
    edge_unit_vec = rel_pos / torch.clamp(edge_length, min=1e-8)
    
    # Create Data object with enhanced attributes
    data = Data(
        x=x,  # Enhanced atom features
        edge_index=edge_index,
        edge_attr=rel_pos,  # Displacement vectors
        pos=pos,
        atomic_numbers=atomic_numbers,
        cell=cell,
        cell_offsets=cell_offsets if cell_offsets is not None else None,
        edge_length=edge_length.squeeze(-1),  # Edge lengths
        edge_unit_vec=edge_unit_vec,  # Unit vectors (for angle calculations)
        edge_lattice=nei,  # Lattice vectors for each edge
        atom_lattice=atom_lat,  # Lattice vectors for each atom
        natoms=torch.tensor([len(structure)]),
        distances=distances  # Interatomic distances
    )
    
    return data


def structure_dict_to_graph(structure_dict, cutoff=6.0, max_neighbors=40, pbc=True, atom_features="cgcnn"):
    """Convert a structure dictionary to a PyG graph.
    
    Args:
        structure_dict: Structure dictionary
        cutoff: Cutoff distance for neighbors
        max_neighbors: Maximum number of neighbors per atom
        pbc: Whether to use periodic boundary conditions
        atom_features: Type of atom features ("one_hot" or "cgcnn")
        
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
            structure_dict = ast.literal_eval(structure_dict)
    
    # Convert to Structure
    structure = Structure.from_dict(structure_dict)
    
    # Convert to graph with enhanced features
    return structure_to_graph(structure, cutoff, max_neighbors, pbc, atom_features)