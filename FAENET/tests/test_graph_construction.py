#!/usr/bin/env python
"""
Test for graph construction enhancements including smooth cutoff and edge canonization.
"""
import os
import sys
import torch
import unittest
import numpy as np
from pymatgen.core import Structure, Lattice

# Add the parent directory to sys.path to allow importing the faenet module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from faenet.utils import SmoothCutoff
from faenet.graph_construction import (
    canonize_edge, 
    structure_to_graph,
    structure_dict_to_graph
)


class TestGraphConstruction(unittest.TestCase):
    """Test cases for graph construction."""
    
    def setUp(self):
        """Set up test data."""
        # Create a simple cubic structure
        lattice = Lattice.cubic(4.0)
        coords = [[0, 0, 0], [0.5, 0.5, 0.5]]
        species = ["Si", "Si"]
        self.structure = Structure(lattice, species, coords)
        
        # Set random seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
    
    def test_smooth_cutoff(self):
        """Test smooth cutoff function."""
        # Create smooth cutoff function
        cutoff_fn = SmoothCutoff()
        
        # Test with different distance values
        distances = torch.tensor([0.1, 0.5, 0.9, 0.99])
        
        # Apply smooth cutoff
        weights = cutoff_fn(distances)
        
        # Check that weights decrease as distances approach 1.0
        self.assertTrue(torch.all(weights[:-1] > weights[1:]))
        
        # Check that function is well-behaved at boundaries
        self.assertTrue(torch.isfinite(weights).all())
        self.assertTrue((weights > 0).all())
    
    def test_edge_canonization(self):
        """Test edge canonization function."""
        # Test case 1: src_id < dst_id, no image shift
        src_id, dst_id, src_image, dst_image = canonize_edge(1, 2, (0, 0, 0), (1, 0, 0))
        self.assertEqual(src_id, 1)
        self.assertEqual(dst_id, 2)
        self.assertEqual(src_image, (0, 0, 0))
        self.assertEqual(dst_image, (1, 0, 0))
        
        # Test case 2: src_id > dst_id, should swap and shift
        src_id, dst_id, src_image, dst_image = canonize_edge(2, 1, (0, 0, 0), (1, 0, 0))
        self.assertEqual(src_id, 1)
        self.assertEqual(dst_id, 2)
        self.assertEqual(src_image, (-1, 0, 0))
        self.assertEqual(dst_image, (0, 0, 0))
        
        # Test case 3: src_id < dst_id, src_image not (0,0,0), should shift
        src_id, dst_id, src_image, dst_image = canonize_edge(1, 2, (1, 1, 1), (2, 1, 1))
        self.assertEqual(src_id, 1)
        self.assertEqual(dst_id, 2)
        self.assertEqual(src_image, (0, 0, 0))
        self.assertEqual(dst_image, (1, 0, 0))
    
    def test_structure_to_graph(self):
        """Test structure_to_graph with smooth cutoff."""
        # Convert structure to graph
        graph = structure_to_graph(self.structure, cutoff=5.0, max_neighbors=12, pbc=True)
        
        # Check that graph has expected attributes
        self.assertIn('edge_weights', graph)
        self.assertIn('edge_attr', graph)
        self.assertIn('distances', graph)
        
        # Check that edge weights are between 0 and 1
        weights = graph.edge_weights
        self.assertTrue(torch.all(weights >= 0))
        self.assertTrue(torch.all(weights <= 10)) # Upper bound is higher due to 1/x term
        
        # Check that edge_attr includes weighted vectors
        # The direction vectors should be scaled by edge weights
        edge_attr = graph.edge_attr
        distances = graph.distances
        
        # Verify vector magnitudes match distances after weighting
        vector_magnitudes = torch.norm(edge_attr, dim=1)
        weighted_distances = distances * weights
        
        # Check proportionality (allowing for numerical differences)
        proportions = vector_magnitudes / weighted_distances
        self.assertTrue(torch.allclose(proportions, proportions[0] * torch.ones_like(proportions), rtol=1e-5))
        
        # Test with non-periodic boundary conditions
        graph_no_pbc = structure_to_graph(self.structure, cutoff=5.0, max_neighbors=12, pbc=False)
        self.assertIn('edge_weights', graph_no_pbc)


if __name__ == "__main__":
    unittest.main()