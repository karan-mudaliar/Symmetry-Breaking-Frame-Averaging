#!/usr/bin/env python
"""
Quick test script to debug property scaling.
"""
import os
import sys
import torch
import numpy as np

# Add the parent directory to sys.path to allow importing the faenet module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from faenet.dataset import create_dataloader

# Use an absolute path to the test data file
TEST_DATA_PATH = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 
    "..", "test_data", "surface_prop_data_set_top_bottom.csv"
))

def main():
    """Test property scaling behavior directly."""
    print("\n=== Testing property scaling behavior ===")
    
    # Create dataloader with scaling explicitly ENABLED
    print("\nCreating dataloader with use_property_scaling=True...")
    train_loader, val_loader, test_loader, dataset = create_dataloader(
        data_source=TEST_DATA_PATH,
        structure_col="slab",
        target_props=["WF_bottom", "WF_top"],
        batch_size=4,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        use_property_scaling=True  # Explicitly ENABLE
    )
    
    print(f"use_scaling attribute: {dataset.use_scaling}")
    print(f"Number of scalers: {len(dataset.scalers)}")
    
    if hasattr(dataset, 'scalers') and len(dataset.scalers) > 0:
        print("\nScaler details:")
        for prop, scaler in dataset.scalers.items():
            print(f"  {prop}: mean={scaler.mean_[0]:.4f}, std={scaler.scale_[0]:.4f}")
    
    print("\nBatch sample (first batch):")
    for batch in train_loader:
        for prop in ["WF_bottom", "WF_top"]:
            if hasattr(batch, prop) and hasattr(batch, f"{prop}_orig"):
                values = getattr(batch, prop)
                orig_values = getattr(batch, f"{prop}_orig")
                print(f"\n{prop}:")
                print(f"  Scaled: {values[:2].tolist()}")
                print(f"  Original: {orig_values[:2].tolist()}")
                
                # Check if they're different (indicating scaling was applied)
                if torch.allclose(values, orig_values):
                    print("  ❌ Values are identical (scaling NOT applied)")
                else:
                    print("  ✅ Values are different (scaling applied)")
            else:
                print(f"\n❌ Missing attributes for {prop}")
        break  # Only check first batch
    
    # Now test with default (scaling disabled)
    print("\n\nCreating dataloader with default settings (scaling disabled)...")
    train_loader2, val_loader2, test_loader2, dataset2 = create_dataloader(
        data_source=TEST_DATA_PATH,
        structure_col="slab",
        target_props=["WF_bottom", "WF_top"],
        batch_size=4,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        # use_property_scaling=False (default)
    )
    
    print(f"use_scaling attribute: {dataset2.use_scaling}")
    print(f"Number of scalers: {len(dataset2.scalers)}")
    
    print("\nBatch sample (first batch):")
    for batch in train_loader2:
        for prop in ["WF_bottom", "WF_top"]:
            if hasattr(batch, prop) and hasattr(batch, f"{prop}_orig"):
                values = getattr(batch, prop)
                orig_values = getattr(batch, f"{prop}_orig")
                print(f"\n{prop}:")
                print(f"  Values: {values[:2].tolist()}")
                print(f"  Original: {orig_values[:2].tolist()}")
                
                # Check if they're identical (indicating no scaling)
                if torch.allclose(values, orig_values):
                    print("  ✅ Values are identical (no scaling applied)")
                else:
                    print("  ❌ Values are different (scaling applied)")
            else:
                print(f"\n❌ Missing attributes for {prop}")
        break  # Only check first batch

if __name__ == "__main__":
    main()