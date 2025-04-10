"""
Tests for the enhanced graph construction module.
Ensures proper translation of Comformer functionality.
"""
import pytest
import torch
import numpy as np
from pymatgen.core import Structure, Lattice

from faenet.graph_construction import (
    structure_to_graph,
    find_lattice_vectors,
    nearest_neighbor_edges_enhanced,
    get_atom_features
)

def test_lattice_vector_finding():
    """Test finding principal lattice vectors."""
    # Create a simple cubic lattice
    lattice = Lattice.cubic(5.0)
    structure = Structure(lattice, ["Si"], [[0, 0, 0]])
    
    # Add atoms at corners
    structure.append("Si", [0, 0, 5])
    structure.append("Si", [0, 5, 0])
    structure.append("Si", [5, 0, 0])
    
    # Get lattice vectors
    lat1, lat2, lat3 = find_lattice_vectors(structure, cutoff=6.0)
    
    # Check that they are orthogonal
    assert np.abs(np.dot(lat1, lat2)) < 1e-6
    assert np.abs(np.dot(lat1, lat3)) < 1e-6
    assert np.abs(np.dot(lat2, lat3)) < 1e-6

def test_enhanced_neighbor_finding():
    """Test enhanced neighbor finding with lattice correction."""
    # Create a simple cubic lattice
    lattice = Lattice.cubic(5.0)
    structure = Structure(lattice, ["Si"], [[0, 0, 0]])
    
    # Add atoms at corners and midpoints
    structure.append("Si", [0, 0, 2.5])
    structure.append("Si", [0, 2.5, 0])
    structure.append("Si", [2.5, 0, 0])
    
    # Get edges using enhanced approach
    edges, lat1, lat2, lat3 = nearest_neighbor_edges_enhanced(
        structure, cutoff=6.0, max_neighbors=12, use_canonize=True
    )
    
    # Check that we found sufficient edges
    assert len(edges) > 0
    
    # Verify self-loops with lattice vectors are added
    for i in range(len(structure)):
        assert (i, i) in edges
        assert len(edges[(i, i)]) >= 3  # At least 3 lattice vectors as self-loops

def test_structure_to_graph_with_cgcnn_features():
    """Test structure to graph conversion with CGCNN features."""
    # Create a simple structure
    lattice = Lattice.cubic(5.0)
    structure = Structure(lattice, ["Si", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    
    # Convert to graph
    graph = structure_to_graph(structure, cutoff=6.0, max_neighbors=12, pbc=True, atom_features="cgcnn")
    
    # Check that the graph has the right components
    assert hasattr(graph, 'x')  # Node features
    assert hasattr(graph, 'edge_index')  # Edge indices
    assert hasattr(graph, 'edge_attr')  # Edge attributes
    assert hasattr(graph, 'pos')  # Positions
    assert hasattr(graph, 'atomic_numbers')  # Atomic numbers
    assert hasattr(graph, 'cell')  # Unit cell
    
    # Check that CGCNN features have the right dimension
    # CGCNN features should be more than just one-hot encoding
    assert graph.x.shape[1] > 0
    
    # Silicon (Z=14) and Oxygen (Z=8) should have different features
    assert not torch.allclose(graph.x[0], graph.x[1])
    
    # Check that we have enhanced attributes
    assert hasattr(graph, 'edge_length')
    assert hasattr(graph, 'edge_unit_vec')
    assert hasattr(graph, 'edge_lattice') or hasattr(graph, 'atom_lattice')

def test_structure_to_graph_with_onehot_features():
    """Test structure to graph conversion with one-hot features."""
    # Create a simple structure
    lattice = Lattice.cubic(5.0)
    structure = Structure(lattice, ["Si", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    
    # Convert to graph
    graph = structure_to_graph(structure, cutoff=6.0, max_neighbors=12, pbc=True, atom_features="one_hot")
    
    # Check that the graph has the right components
    assert hasattr(graph, 'x')
    
    # One-hot encoding should be sparse
    assert torch.sum(graph.x[0]) == 1.0
    assert torch.sum(graph.x[1]) == 1.0
    
    # The non-zero entries should be at different positions
    assert not torch.allclose(graph.x[0], graph.x[1])

def test_atom_features_handling():
    """Test handling of atom features, particularly for unknown elements."""
    # Test a common element that is in elem_features
    si_features = get_atom_features(14, "cgcnn")  # Silicon
    
    # Check that features are non-zero and have right dimension
    assert si_features.shape[0] == 9  # 9 feature dimensions
    assert torch.sum(si_features) > 0
    
    # Test a rare element that might not be in elem_features
    # Ac - Actinium (Z=89) should still get meaningful features
    ac_features = get_atom_features(89, "cgcnn")
    
    # Features should be non-zero
    assert torch.sum(ac_features) > 0
    
    # Test a hypothetical element not in the periodic table
    # Should use average/estimated features, not zeros
    unknown_features = get_atom_features(150, "cgcnn")
    
    # Features should be non-zero
    assert torch.sum(unknown_features) > 0
    
    # Period should be a reasonable estimate (element 150 would be in period 7 or 8)
    assert 7 <= unknown_features[0] <= 8
    
    # Valence electrons and group should be based on atomic number
    assert unknown_features[7] == 150 % 18  # Valence electrons
    
    # Compare features - common elements should be different from rare ones
    assert not torch.allclose(si_features, unknown_features)

def test_structure_dict_to_graph():
    """Test conversion from structure dictionary to graph."""
    # Create a simple structure
    lattice = Lattice.cubic(5.0)
    structure = Structure(lattice, ["Si", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    
    # Convert to dictionary
    structure_dict = structure.as_dict()
    
    # Import the function
    from faenet.graph_construction import structure_dict_to_graph
    
    # Convert to graph
    graph = structure_dict_to_graph(structure_dict, cutoff=6.0, max_neighbors=12, pbc=True)
    
    # Check that the graph has the right components
    assert hasattr(graph, 'x')
    assert hasattr(graph, 'edge_index')
    assert hasattr(graph, 'edge_attr')
    assert hasattr(graph, 'pos')
    assert hasattr(graph, 'atomic_numbers')
    assert hasattr(graph, 'cell')
    
    # Check atom counts
    assert len(graph.atomic_numbers) == 2
    assert graph.atomic_numbers[0].item() == 14  # Si
    assert graph.atomic_numbers[1].item() == 8   # O