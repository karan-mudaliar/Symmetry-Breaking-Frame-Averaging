#!/usr/bin/env python
"""
Diagnostic script to debug Frame Averaging cell shape issues
"""
import os
import torch
import numpy as np
from pathlib import Path
import sys

from dataset import EnhancedSlabDataset, apply_frame_averaging_to_batch
from frame_averaging import frame_averaging_3D, frame_averaging_2D
from torch_geometric.loader import DataLoader
from faenet import FAENet


def print_tensor_info(name, tensor):
    """Print detailed information about a tensor"""
    if tensor is None:
        print(f"{name}: None")
        return
        
    if isinstance(tensor, list):
        print(f"{name}: List of {len(tensor)} tensors")
        for i, t in enumerate(tensor):
            if i < 3:  # Only show first 3 to avoid output being too long
                print(f"  {name}[{i}] shape: {t.shape}, dtype: {t.dtype}")
                print(f"  {name}[{i}] first few values: {t.reshape(-1)[:5]}")
        return
        
    # Regular tensor
    print(f"{name}: shape {tensor.shape}, dtype: {tensor.dtype}")
    print(f"{name} first few values: {tensor.reshape(-1)[:5]}")


def main():
    """Run diagnostics to understand the shape issue"""
    print("=== Frame Averaging Shape Diagnostics ===")
    
    # Create a single test data point
    data_path = Path("test_data/surface_prop_data_set_top_bottom.csv")
    structure_col = "slab"
    target_props = ["WF_top", "WF_bottom"]
    
    # Create dataset with a single sample
    print("\n1. Loading a single data point...")
    dataset = EnhancedSlabDataset(
        data_source=data_path,
        structure_col=structure_col,
        target_props=target_props,
        limit=1  # Only load one sample
    )
    
    # Get the first data point
    data = dataset[0]
    print("\n2. Original Data Properties:")
    print(f"Data contains the following attributes: {data.keys}")
    print_tensor_info("pos", data.pos)
    print_tensor_info("edge_index", data.edge_index)
    print_tensor_info("cell", data.cell)
    print_tensor_info("cell_offsets", data.cell_offsets)
    
    # Test different frame averaging methods
    print("\n3. Testing 2D Frame Averaging:")
    fa_pos_2d, fa_cell_2d, fa_rot_2d = frame_averaging_2D(data.pos, data.cell, "all")
    print_tensor_info("fa_pos_2d", fa_pos_2d)
    print_tensor_info("fa_cell_2d", fa_cell_2d)
    print_tensor_info("fa_rot_2d", fa_rot_2d)
    
    print("\n4. Testing 3D Frame Averaging:")
    fa_pos_3d, fa_cell_3d, fa_rot_3d = frame_averaging_3D(data.pos, data.cell, "all")
    print_tensor_info("fa_pos_3d", fa_pos_3d)
    print_tensor_info("fa_cell_3d", fa_cell_3d)
    print_tensor_info("fa_rot_3d", fa_rot_3d)
    
    # Create a small batch and apply frame averaging
    print("\n5. Testing batch frame averaging with DataLoader:")
    # Create tiny train loader with batch_size=2
    train_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    batch = next(iter(train_loader))
    
    print("Batch before frame averaging:")
    print(f"Batch contains the following attributes: {batch.keys}")
    print_tensor_info("batch.pos", batch.pos)
    print_tensor_info("batch.cell", batch.cell)
    print_tensor_info("batch.cell_offsets", batch.cell_offsets)
    
    # Apply 2D frame averaging to batch
    print("\n6. Applying 2D frame averaging to batch:")
    batch_2d = apply_frame_averaging_to_batch(batch.clone(), "all")
    print_tensor_info("batch_2d.fa_pos", batch_2d.fa_pos)
    print_tensor_info("batch_2d.fa_cell", batch_2d.fa_cell)
    print_tensor_info("batch_2d.fa_rot", batch_2d.fa_rot)
    
    # Now try to simulate what happens in the forward pass
    print("\n7. Simulating forward pass with frame averaging:")
    model = FAENet(
        cutoff=6.0,
        output_properties=target_props,
        num_gaussians=25,
        hidden_channels=64,
        num_filters=64,
        num_interactions=2
    )
    
    # Test the matrix multiplication that's failing
    print("\n8. Testing matrix multiplication that's failing:")
    for i in range(len(batch_2d.fa_pos)):
        # Set the positions from this frame
        batch_2d.pos = batch_2d.fa_pos[i]
        
        # Set the cell properly
        if isinstance(batch_2d.fa_cell, list):
            proper_cell = batch_2d.fa_cell[i]
        else:
            proper_cell = batch_2d.cell
            
        print(f"\nFrame {i}:")
        print_tensor_info("cell_offsets", batch_2d.cell_offsets)
        print_tensor_info("proper_cell", proper_cell)
        
        try:
            # Try the matrix multiplication
            offsets = torch.matmul(batch_2d.cell_offsets, proper_cell)
            print_tensor_info("offsets result", offsets)
            print("✅ Matrix multiplication successful")
        except RuntimeError as e:
            print(f"❌ Matrix multiplication failed: {e}")
            
            # Test a possible fix
            if proper_cell.shape != (3, 3):
                print("\nAttempting to fix shape...")
                if isinstance(proper_cell, list):
                    fixed_cell = proper_cell[0]
                else:
                    fixed_cell = proper_cell
                    
                # Various fixes to try
                if fixed_cell.shape == (12, 3):
                    # Take first 3 rows
                    fixed_cell = fixed_cell[:3]
                elif fixed_cell.dim() > 2:
                    # Reshape to 3x3
                    try:
                        fixed_cell = fixed_cell.reshape(-1)[:9].reshape(3, 3)
                    except:
                        fixed_cell = torch.eye(3)
                
                print_tensor_info("fixed_cell", fixed_cell)
                
                try:
                    # Try with fixed cell
                    offsets = torch.matmul(batch_2d.cell_offsets, fixed_cell)
                    print_tensor_info("offsets with fixed cell", offsets)
                    print("✅ Matrix multiplication with fixed cell successful")
                except RuntimeError as e2:
                    print(f"❌ Matrix multiplication with fixed cell still failed: {e2}")
    
    
if __name__ == "__main__":
    main()