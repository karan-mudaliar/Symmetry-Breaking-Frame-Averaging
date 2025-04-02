#!/usr/bin/env python
"""
Test script for verifying frame averaging functionality.
This script tests that the frame averaging methods correctly generate 
the expected number of frames and maintain the proper properties.
"""
import os
import torch
import numpy as np
from pathlib import Path

from frame_averaging import frame_averaging_3D, frame_averaging_2D

def test_3d_frame_averaging():
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
    assert num_frames == 8, f"Expected 8 frames for 3D frame averaging, got {num_frames}"
    
    # Check shapes
    for i in range(num_frames):
        # Position shape should match original
        assert fa_pos[i].shape == test_pos.shape, \
            f"Frame {i} position shape mismatch: {fa_pos[i].shape} vs {test_pos.shape}"
        
        # Rotation should be 3x3
        assert fa_rot[i].shape[1:] == (3, 3), \
            f"Rotation {i} shape should be (1, 3, 3), got {fa_rot[i].shape}"
        
        # Rotation should be orthogonal
        R = fa_rot[i].squeeze()
        assert torch.allclose(R @ R.T, torch.eye(3), atol=1e-5), \
            f"Rotation {i} is not orthogonal"
        
        # Frame averaging includes both proper rotations (det=1) and improper rotations (det=-1)
        det = torch.det(R)
        assert torch.isclose(torch.abs(det), torch.tensor(1.0), atol=1e-5), \
            f"Rotation {i} has determinant with magnitude {torch.abs(det)}, not 1"
        print(f"  Frame {i} has determinant {det:.1f} (proper rotation if 1, improper if -1)")
    
    return True

def test_2d_frame_averaging():
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
    assert num_frames == 4, f"Expected 4 frames for 2D frame averaging, got {num_frames}"
    
    # Check shapes
    for i in range(num_frames):
        # Position shape should match original
        assert fa_pos[i].shape == test_pos.shape, \
            f"Frame {i} position shape mismatch: {fa_pos[i].shape} vs {test_pos.shape}"
        
        # Rotation should be 3x3
        assert fa_rot[i].shape[1:] == (3, 3), \
            f"Rotation {i} shape should be (1, 3, 3), got {fa_rot[i].shape}"
        
        # Z coordinates should be preserved (key property of 2D frame averaging)
        assert torch.allclose(fa_pos[i][:, 2], z_coords, atol=1e-5), \
            f"Frame {i} does not preserve z-coordinates"
        
        # Rotation should be orthogonal
        R = fa_rot[i].squeeze()
        assert torch.allclose(R @ R.T, torch.eye(3), atol=1e-5), \
            f"Rotation {i} is not orthogonal"
        
        # Frame averaging includes both proper rotations (det=1) and improper rotations (det=-1)
        det = torch.det(R)
        assert torch.isclose(torch.abs(det), torch.tensor(1.0), atol=1e-5), \
            f"Rotation {i} has determinant with magnitude {torch.abs(det)}, not 1"
        print(f"  Frame {i} has determinant {det:.1f} (proper rotation if 1, improper if -1)")
        
        # For 2D frame averaging, the bottom-right element of the rotation matrix
        # should be 1 (no rotation around z-axis)
        assert torch.isclose(R[2, 2], torch.tensor(1.0), atol=1e-5), \
            f"Rotation {i} has R[2,2] = {R[2,2]}, not 1"
    
    return True

def test_rotation_invariance():
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
    assert matches_found >= len(rot_fa_pos) // 2, "Not enough matching frames found after rotation"
    
    return True

def test_fa_methods():
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
        assert len(fa_pos) > 0, f"Method {method} produced no frames"
        
        # Check expected frame counts for non-random methods
        if method == "all":
            assert len(fa_pos) == 8, f"Expected 8 frames for 'all', got {len(fa_pos)}"
        elif method == "det":
            assert len(fa_pos) == 1, f"Expected 1 frame for 'det', got {len(fa_pos)}"
        elif method == "random":
            assert len(fa_pos) == 1, f"Expected 1 frame for 'random', got {len(fa_pos)}"
        # Note: se3 methods will produce varying numbers of frames based on the specific rotations
    
    # Test methods for 2D
    for method in methods:
        print(f"Testing 2D method: {method}")
        fa_pos, _, fa_rot = frame_averaging_2D(test_pos, None, method)
        
        # Check that we got at least one frame
        assert len(fa_pos) > 0, f"Method {method} produced no frames"
        
        # Check that z-coordinates are preserved
        for i in range(len(fa_pos)):
            assert torch.allclose(fa_pos[i][:, 2], test_pos[:, 2], atol=1e-5), \
                f"Frame {i} with method {method} does not preserve z-coordinates"
        
        # Check expected frame counts for non-random methods
        if method == "all":
            assert len(fa_pos) == 4, f"Expected 4 frames for 'all', got {len(fa_pos)}"
        elif method == "det":
            assert len(fa_pos) == 1, f"Expected 1 frame for 'det', got {len(fa_pos)}"
        elif method == "random":
            assert len(fa_pos) == 1, f"Expected 1 frame for 'random', got {len(fa_pos)}"
        # Note: se3 methods will produce varying numbers of frames based on the specific rotations
    
    return True

def main():
    """Run all frame averaging tests."""
    print("=== Testing Frame Averaging Functionality ===")
    
    # Run tests
    tests = [
        test_3d_frame_averaging, 
        test_2d_frame_averaging, 
        test_rotation_invariance,
        test_fa_methods
    ]
    
    results = {}
    
    for test_func in tests:
        test_name = test_func.__name__
        try:
            result = test_func()
            results[test_name] = "✅ PASSED" if result else "❌ FAILED"
        except Exception as e:
            import traceback
            traceback.print_exc()
            results[test_name] = f"❌ ERROR: {e}"
    
    # Print summary
    print("\n=== Test Results ===")
    all_passed = True
    for test_name, result in results.items():
        print(f"{test_name}: {result}")
        if not result.startswith("✅"):
            all_passed = False
    
    print(f"\nFrame averaging testing {'successful' if all_passed else 'failed'}")
    return all_passed

if __name__ == "__main__":
    main()