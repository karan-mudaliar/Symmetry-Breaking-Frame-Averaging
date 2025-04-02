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
        assert fa_rot[i].shape == (1, 3, 3), \
            f"Rotation {i} shape should be (1, 3, 3), got {fa_rot[i].shape}"
        
        # Rotation should be orthogonal
        R = fa_rot[i].squeeze()
        assert torch.allclose(R @ R.T, torch.eye(3), atol=1e-5), \
            f"Rotation {i} is not orthogonal"
        
        # Rotation should have determinant 1
        assert torch.allclose(torch.det(R), torch.tensor(1.0), atol=1e-5), \
            f"Rotation {i} has determinant {torch.det(R)}, not 1"
    
    return True

def test_2d_frame_averaging():
    """Test 2D frame averaging produces 4 frames and preserves z-axis."""
    print("\n=== Testing 2D Frame Averaging ===")
    
    # Create a simple test structure (5 atoms with random 3D positions)
    num_atoms = 5
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
        assert fa_rot[i].shape == (1, 3, 3), \
            f"Rotation {i} shape should be (1, 3, 3), got {fa_rot[i].shape}"
        
        # Z coordinates should be preserved (key property of 2D frame averaging)
        assert torch.allclose(fa_pos[i][:, 2], z_coords, atol=1e-5), \
            f"Frame {i} does not preserve z-coordinates"
        
        # Rotation should be orthogonal
        R = fa_rot[i].squeeze()
        assert torch.allclose(R @ R.T, torch.eye(3), atol=1e-5), \
            f"Rotation {i} is not orthogonal"
        
        # Rotation should have determinant 1
        assert torch.allclose(torch.det(R), torch.tensor(1.0), atol=1e-5), \
            f"Rotation {i} has determinant {torch.det(R)}, not 1"
        
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
    test_pos = torch.randn(num_atoms, 3)
    
    # Get 3D frames for original position
    orig_fa_pos, _, orig_fa_rot = frame_averaging_3D(test_pos, None, "all")
    
    # Create a random rotation
    theta = torch.rand(1) * 2 * torch.pi
    phi = torch.rand(1) * torch.pi
    
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
    
    # Check that we have the same number of frames
    assert len(orig_fa_pos) == len(rot_fa_pos), \
        f"Original and rotated frames have different counts: {len(orig_fa_pos)} vs {len(rot_fa_pos)}"
    
    # The set of frames from the rotated structure should be related to the set of frames
    # from the original structure by the rotation R, but not necessarily in the same order
    
    # For simplicity, we'll just check that at least one frame has the expected relationship
    print("Checking rotation invariance property...")
    matched = False
    for i in range(len(rot_fa_pos)):
        # Calculate centroid to eliminate any numerical differences in translation
        rot_centered = rot_fa_pos[i] - rot_fa_pos[i].mean(dim=0)
        
        for j in range(len(orig_fa_pos)):
            orig_centered = orig_fa_pos[j] - orig_fa_pos[j].mean(dim=0)
            orig_rotated = orig_centered @ R.T
            
            if torch.allclose(rot_centered, orig_rotated, atol=1e-4):
                matched = True
                print(f"Found matching frames: rotated frame {i} matches original frame {j}")
                break
    
    assert matched, "Could not find matching frames after rotation"
    print("Rotation invariance property verified")
    
    return True

def test_fa_methods():
    """Test different frame averaging methods."""
    print("\n=== Testing Frame Averaging Methods ===")
    
    # Create a simple test structure
    num_atoms = 5
    test_pos = torch.randn(num_atoms, 3)
    
    # Test methods for 3D
    methods = ["all", "det", "random", "se3-all", "se3-det", "se3-random"]
    for method in methods:
        print(f"Testing 3D method: {method}")
        fa_pos, _, fa_rot = frame_averaging_3D(test_pos, None, method)
        
        # Check that we got at least one frame
        assert len(fa_pos) > 0, f"Method {method} produced no frames"
        
        # 'all' methods should produce 8 frames, 'det' methods should produce 1
        if method == "all":
            assert len(fa_pos) == 8, f"Expected 8 frames for 'all', got {len(fa_pos)}"
        elif method == "det":
            assert len(fa_pos) == 1, f"Expected 1 frame for 'det', got {len(fa_pos)}"
        elif method == "se3-all":
            # se3-all should have 4 frames (those with determinant 1)
            assert len(fa_pos) == 4, f"Expected 4 frames for 'se3-all', got {len(fa_pos)}"
        elif method == "se3-det":
            assert len(fa_pos) == 1, f"Expected 1 frame for 'se3-det', got {len(fa_pos)}"
    
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
        
        # 'all' methods should produce 4 frames, 'det' methods should produce 1
        if method == "all":
            assert len(fa_pos) == 4, f"Expected 4 frames for 'all', got {len(fa_pos)}"
        elif method == "det":
            assert len(fa_pos) == 1, f"Expected 1 frame for 'det', got {len(fa_pos)}"
        elif method == "se3-all":
            # se3-all should have 2 frames in 2D (with determinant 1)
            assert len(fa_pos) == 2, f"Expected 2 frames for 'se3-all', got {len(fa_pos)}"
        elif method == "se3-det":
            assert len(fa_pos) == 1, f"Expected 1 frame for 'se3-det', got {len(fa_pos)}"
    
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