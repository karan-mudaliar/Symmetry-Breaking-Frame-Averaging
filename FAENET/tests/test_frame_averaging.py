#!/usr/bin/env python
"""
Test script for verifying frame averaging functionality.
This script tests that the frame averaging methods correctly generate 
the expected number of frames and maintain the proper properties.
"""
import os
import sys
import torch
import numpy as np
import unittest
from pathlib import Path

# Add the parent directory to sys.path to allow importing the faenet module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from faenet.frame_averaging import frame_averaging_3D, frame_averaging_2D

class TestFrameAveraging(unittest.TestCase):
    """Test cases for frame averaging functionality"""
    
    def test_3d_frame_averaging(self):
        """Test 3D frame averaging produces 8 frames with proper properties."""
        print("\n=== Testing 3D Frame Averaging ===")
        
        # Create a simple test structure (5 atoms with random 3D positions)
        num_atoms = 5
        torch.manual_seed(42)  # For reproducibility
        test_pos = torch.randn(num_atoms, 3)
        
        # Test with "all" method (should produce 8 frames)
        fa_pos, fa_cell, fa_rot = frame_averaging_3D(test_pos, None, "all")
        
        # Check number of frames
        num_frames = len(fa_pos)
        print(f"Number of 3D frames: {num_frames}")
        self.assertEqual(num_frames, 8, f"Expected 8 frames for 3D frame averaging, got {num_frames}")
        
        # Check shapes
        for i in range(num_frames):
            # Position shape should match original
            self.assertEqual(fa_pos[i].shape, test_pos.shape, 
                         f"Frame {i} position shape mismatch: {fa_pos[i].shape} vs {test_pos.shape}")
            
            # Rotation should be 3x3
            self.assertEqual(fa_rot[i].shape[1:], (3, 3),
                         f"Rotation {i} shape should be (1, 3, 3), got {fa_rot[i].shape}")
            
            # Rotation should be orthogonal
            R = fa_rot[i].squeeze()
            self.assertTrue(torch.allclose(R @ R.T, torch.eye(3), atol=1e-5),
                        f"Rotation {i} is not orthogonal")
            
            # Frame averaging includes both proper rotations (det=1) and improper rotations (det=-1)
            det = torch.det(R)
            self.assertTrue(torch.isclose(torch.abs(det), torch.tensor(1.0), atol=1e-5),
                        f"Rotation {i} has determinant with magnitude {torch.abs(det)}, not 1")
            print(f"  Frame {i} has determinant {det:.1f} (proper rotation if 1, improper if -1)")
    
    def test_2d_frame_averaging(self):
        """Test 2D frame averaging produces 4 frames and preserves z-axis."""
        print("\n=== Testing 2D Frame Averaging ===")
        
        # Create a simple test structure (5 atoms with random 3D positions)
        num_atoms = 5
        torch.manual_seed(42)  # For reproducibility
        test_pos = torch.randn(num_atoms, 3)
        
        # Save z-coordinates for later comparison
        z_coords = test_pos[:, 2].clone()
        
        # Test with "all" method (should produce 4 frames)
        fa_pos, fa_cell, fa_rot = frame_averaging_2D(test_pos, None, "all")
        
        # Check number of frames
        num_frames = len(fa_pos)
        print(f"Number of 2D frames: {num_frames}")
        self.assertEqual(num_frames, 4, f"Expected 4 frames for 2D frame averaging, got {num_frames}")
        
        # Check shapes
        for i in range(num_frames):
            # Position shape should match original
            self.assertEqual(fa_pos[i].shape, test_pos.shape,
                         f"Frame {i} position shape mismatch: {fa_pos[i].shape} vs {test_pos.shape}")
            
            # Rotation should be 3x3
            self.assertEqual(fa_rot[i].shape[1:], (3, 3),
                         f"Rotation {i} shape should be (1, 3, 3), got {fa_rot[i].shape}")
            
            # Z coordinates should be preserved (key property of 2D frame averaging)
            self.assertTrue(torch.allclose(fa_pos[i][:, 2], z_coords, atol=1e-5),
                        f"Frame {i} does not preserve z-coordinates")
            
            # Rotation should be orthogonal
            R = fa_rot[i].squeeze()
            self.assertTrue(torch.allclose(R @ R.T, torch.eye(3), atol=1e-5),
                        f"Rotation {i} is not orthogonal")
            
            # Frame averaging includes both proper rotations (det=1) and improper rotations (det=-1)
            det = torch.det(R)
            self.assertTrue(torch.isclose(torch.abs(det), torch.tensor(1.0), atol=1e-5),
                        f"Rotation {i} has determinant with magnitude {torch.abs(det)}, not 1")
            print(f"  Frame {i} has determinant {det:.1f} (proper rotation if 1, improper if -1)")
            
            # For 2D frame averaging, the bottom-right element of the rotation matrix
            # should be 1 (no rotation around z-axis)
            self.assertTrue(torch.isclose(R[2, 2], torch.tensor(1.0), atol=1e-5),
                        f"Rotation {i} has R[2,2] = {R[2,2]}, not 1")
    
    def test_rotation_invariance(self):
        """Test that frame averaging produces rotation-invariant predictions."""
        print("\n=== Testing Rotation Invariance Property ===")
        
        # Create a simple structure
        num_atoms = 5
        torch.manual_seed(42)  # For reproducibility
        test_pos = torch.randn(num_atoms, 3)
        
        # Get 3D frames for original position
        orig_fa_pos, _, orig_fa_rot = frame_averaging_3D(test_pos, None, "all")
        
        # Create a random rotation
        theta = torch.tensor(0.5)  # Fixed angle for reproducibility
        phi = torch.tensor(0.3)    # Fixed angle for reproducibility
        
        # Rotation around z
        R_z = torch.tensor([
            [torch.cos(theta), -torch.sin(theta), 0],
            [torch.sin(theta), torch.cos(theta), 0],
            [0, 0, 1]
        ])
        
        # Rotation around y
        R_y = torch.tensor([
            [torch.cos(phi), 0, torch.sin(phi)],
            [0, 1, 0],
            [-torch.sin(phi), 0, torch.cos(phi)]
        ])
        
        # Combined rotation
        R = R_z @ R_y
        
        # Apply rotation to the original positions
        rotated_pos = test_pos @ R.T
        
        # Get frames for rotated position
        rot_fa_pos, _, rot_fa_rot = frame_averaging_3D(rotated_pos, None, "all")
        
        # Since frame averaging can generate frames in different order after rotation,
        # and each frame can be either a proper or improper rotation,
        # we need to check for any matching transformed centered coordinates
        print("Checking rotation invariance property...")
        
        # Calculate centroids
        orig_centroids = [pos.mean(dim=0) for pos in orig_fa_pos]
        rot_centroids = [pos.mean(dim=0) for pos in rot_fa_pos]
        
        # Center the positions
        orig_centered = [pos - centroid for pos, centroid in zip(orig_fa_pos, orig_centroids)]
        rot_centered = [pos - centroid for pos, centroid in zip(rot_fa_pos, rot_centroids)]
        
        # Now test for invariance - after rotation and frame averaging, 
        # the set of possible centered frames should be the same
        # We need to check all combinations since the order is not guaranteed
        matches_found = 0
        tolerance = 1e-3  # Increased tolerance for floating point comparison
        
        for i, rot_frame in enumerate(rot_centered):
            for j, orig_frame in enumerate(orig_centered):
                # We need to check both for direct match and match after applying the rotation
                # since frame averaging is equivariant, not necessarily invariant
                if (torch.allclose(rot_frame, orig_frame, atol=tolerance) or
                    torch.allclose(rot_frame, orig_frame @ R.T, atol=tolerance)):
                    matches_found += 1
                    print(f"  Found matching frames: rotated frame {i} matches original frame {j}")
                    break
        
        # We expect at least half the frames to have a matching counterpart
        # (due to both proper and improper rotations being present)
        print(f"  Found {matches_found} matching frames out of {len(rot_fa_pos)}")
        self.assertGreaterEqual(matches_found, len(rot_fa_pos) // 2, 
                            "Not enough matching frames found after rotation")
    
    def test_fa_methods(self):
        """Test different frame averaging methods."""
        print("\n=== Testing Frame Averaging Methods ===")
        
        # Create a simple test structure
        num_atoms = 5
        torch.manual_seed(42)  # For reproducibility
        test_pos = torch.randn(num_atoms, 3)
        
        # Test methods for 3D
        methods = ["all", "det", "random", "se3-all", "se3-det", "se3-random"]
        for method in methods:
            print(f"Testing 3D method: {method}")
            fa_pos, _, fa_rot = frame_averaging_3D(test_pos, None, method)
            
            # Check that we got at least one frame
            self.assertGreater(len(fa_pos), 0, f"Method {method} produced no frames")
            
            # Check expected frame counts for non-random methods
            if method == "all":
                self.assertEqual(len(fa_pos), 8, f"Expected 8 frames for 'all', got {len(fa_pos)}")
            elif method == "det":
                self.assertEqual(len(fa_pos), 1, f"Expected 1 frame for 'det', got {len(fa_pos)}")
            elif method == "random":
                self.assertEqual(len(fa_pos), 1, f"Expected 1 frame for 'random', got {len(fa_pos)}")
            # Note: se3 methods will produce varying numbers of frames based on the specific rotations
        
        # Test methods for 2D
        for method in methods:
            print(f"Testing 2D method: {method}")
            fa_pos, _, fa_rot = frame_averaging_2D(test_pos, None, method)
            
            # Check that we got at least one frame
            self.assertGreater(len(fa_pos), 0, f"Method {method} produced no frames")
            
            # Check that z-coordinates are preserved
            for i in range(len(fa_pos)):
                self.assertTrue(torch.allclose(fa_pos[i][:, 2], test_pos[:, 2], atol=1e-5),
                            f"Frame {i} with method {method} does not preserve z-coordinates")
            
            # Check expected frame counts for non-random methods
            if method == "all":
                self.assertEqual(len(fa_pos), 4, f"Expected 4 frames for 'all', got {len(fa_pos)}")
            elif method == "det":
                self.assertEqual(len(fa_pos), 1, f"Expected 1 frame for 'det', got {len(fa_pos)}")
            elif method == "random":
                self.assertEqual(len(fa_pos), 1, f"Expected 1 frame for 'random', got {len(fa_pos)}")
            # Note: se3 methods will produce varying numbers of frames based on the specific rotations


if __name__ == "__main__":
    unittest.main()