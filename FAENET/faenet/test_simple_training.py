#!/usr/bin/env python
"""
Test script for training FAENet with SimpleConfig.
This script demonstrates training a small model for a few epochs using SimpleConfig.
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import numpy as np
from pathlib import Path

from config import SimpleConfig
from faenet import FAENet
from dataset import EnhancedSlabDataset, apply_frame_averaging_to_batch
from train import train_faenet  # Use the simplified training interface

def main():
    """Run a small training job with SimpleConfig"""
    print("=== Testing FAENet Training with SimpleConfig ===")
    
    # Set up a simple configuration
    config = SimpleConfig(
        # Model parameters
        cutoff=6.0,
        max_neighbors=40,
        num_gaussians=25,  # Reduced for faster training
        hidden_channels=64,  # Reduced for faster training
        num_filters=64,     # Reduced for faster training
        num_interactions=2,  # Reduced for faster training
        dropout=0.0,
        output_properties=["WF_top", "WF_bottom"],
        regress_forces=False,
        
        # Training parameters
        batch_size=4,
        epochs=2,  # Just a quick test
        lr=0.001,
        weight_decay=1e-5,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        frame_averaging="2D",  # Use 2D frame averaging for slabs
        fa_method="all",
        force_weight=0.0,  # No force prediction
        seed=42,
        num_workers=0,  # Use 0 for testing
        
        # Data parameters
        data_dir=Path("test_data/surface_prop_data_set_top_bottom.csv"),
        structure_col="slab",
        target_properties=["WF_top", "WF_bottom", "cleavage_energy"],
        pbc=True,
        
        # General parameters
        output_dir=Path("./test_outputs"),
        device="cpu"  # Use CPU for testing
    )
    
    # Run training with SimpleConfig
    try:
        model, test_loader = train_faenet(
            data_path=config.data_dir,
            structure_col=config.structure_col,
            target_properties=config.target_properties,
            output_dir=config.output_dir,
            frame_averaging=config.frame_averaging,
            fa_method=config.fa_method,
            cutoff=config.cutoff,
            max_neighbors=config.max_neighbors,
            batch_size=config.batch_size,
            epochs=config.epochs,
            learning_rate=config.lr,
            seed=config.seed,
            device=config.device,
            num_workers=config.num_workers,
            num_gaussians=config.num_gaussians,
            hidden_channels=config.hidden_channels,
            num_filters=config.num_filters,
            num_interactions=config.num_interactions,
            dropout=config.dropout,
            regress_forces=config.regress_forces
        )
        
        print("\n✅ Training completed successfully with SimpleConfig!")
        
        # Verify the model was saved
        best_model_path = os.path.join(config.output_dir, "best_model.pt")
        if os.path.exists(best_model_path):
            print(f"✅ Best model was saved to {best_model_path}")
        
        # Verify predictions were saved
        predictions_path = os.path.join(config.output_dir, "predictions.json")
        if os.path.exists(predictions_path):
            print(f"✅ Predictions were saved to {predictions_path}")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        raise

if __name__ == "__main__":
    main()