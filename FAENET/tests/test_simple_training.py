#!/usr/bin/env python
"""
Test script for training FAENet with SimpleConfig.
This script demonstrates training a small model for a few epochs using SimpleConfig.
"""
import os
import sys
import torch
import unittest
from pathlib import Path

# Add the parent directory to sys.path to allow importing the faenet module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from faenet.config import SimpleConfig
from faenet.faenet import FAENet
from faenet.dataset import EnhancedSlabDataset, apply_frame_averaging_to_batch
from faenet.train import train_faenet

# Use an absolute path to the test data file
TEST_DATA_PATH = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 
    "..", "test_data", "surface_prop_data_set_top_bottom.csv"
))

class TestSimpleTraining(unittest.TestCase):
    """Tests for training with SimpleConfig"""
    
    def setUp(self):
        """Set up test environment"""
        # Check if test data exists
        self.test_data_exists = os.path.exists(TEST_DATA_PATH)
        print(f"Test data path: {TEST_DATA_PATH}")
        print(f"Test data exists: {self.test_data_exists}")
        
        # Create output directory if needed
        self.output_dir = Path("./test_outputs")
        os.makedirs(self.output_dir, exist_ok=True)
    
    def test_training(self):
        """Test a small training job with SimpleConfig"""
        if not self.test_data_exists:
            self.skipTest("Test data file not found")
            
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
            data_dir=TEST_DATA_PATH,
            structure_col="slab",
            target_properties=["WF_top", "WF_bottom", "cleavage_energy"],
            pbc=True,
            
            # General parameters
            output_dir=self.output_dir,
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
            self.assertTrue(os.path.exists(best_model_path), f"Best model was not saved to {best_model_path}")
            
            # Verify predictions were saved
            predictions_path = os.path.join(config.output_dir, "predictions.json")
            self.assertTrue(os.path.exists(predictions_path), f"Predictions were not saved to {predictions_path}")
            
        except Exception as e:
            self.fail(f"Training failed with error: {e}")


# This allows running the test directly
if __name__ == "__main__":
    unittest.main()