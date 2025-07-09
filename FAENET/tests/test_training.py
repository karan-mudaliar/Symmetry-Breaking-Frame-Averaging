#!/usr/bin/env python
"""
Test script for basic training functionality.
This script runs a minimal model to ensure training works.
"""
import os
import sys
import torch
import unittest
from pathlib import Path

# Add the parent directory to sys.path to allow importing the faenet module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from faenet.config import Config
from faenet.train import train_faenet

# Use an absolute path to the test data file
TEST_DATA_PATH = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 
    "..", "test_data", "surface_prop_data_set_top_bottom.csv"
))

class TestTraining(unittest.TestCase):
    """Tests for training functionality"""
    
    def setUp(self):
        """Set up test environment"""
        # Check if test data exists
        self.test_data_exists = os.path.exists(TEST_DATA_PATH)
        print(f"Test data path: {TEST_DATA_PATH}")
        print(f"Test data exists: {self.test_data_exists}")
        
        # Create output directory if needed
        self.output_dir = Path("./test_outputs/basic_training")
        os.makedirs(self.output_dir, exist_ok=True)
    
    def test_basic_training(self):
        """Test basic training functionality with and without consistency loss."""
        if not self.test_data_exists:
            self.skipTest("Test data file not found")
            
        print("\n=== Testing Basic Training ===")
        
        # Set up configuration for minimal training
        config = Config(
            # Model parameters (minimized for fast testing)
            cutoff=6.0,
            max_neighbors=20,
            num_gaussians=10,
            hidden_channels=32,
            num_filters=32,
            num_interactions=1,
            dropout=0.0,
            
            # Training parameters
            batch_size=4,
            epochs=1,  # Just one epoch for basic testing
            learning_rate=0.001,
            frame_averaging="2D",  # Enable frame averaging
            
            # Data parameters
            data_path=TEST_DATA_PATH,
            structure_col="slab",
            target_properties=["WF_top", "WF_bottom"],
            
            # MLflow settings
            use_mlflow=False,
            
            # Other settings
            output_dir=str(self.output_dir),
            device="cpu"
        )
        
        # Run training without consistency loss
        try:
            print("\nStarting basic training without consistency loss...")
            model, _ = train_faenet(**config.model_dump())
            
            # Verify the model was saved
            best_model_path = os.path.join(self.output_dir, "best_model.pt")
            self.assertTrue(os.path.exists(best_model_path), f"Best model was not saved to {best_model_path}")
            
            print("✅ Basic training without consistency loss passed!")
            
        except Exception as e:
            self.fail(f"Basic training failed with error: {e}")
        
        # Run training with consistency loss
        try:
            print("\nStarting training with consistency loss enabled...")
            
            # Update config to enable consistency loss
            consistency_config = Config(
                **config.model_dump(),
                consistency_loss=True,
                consistency_weight=0.1,
                consistency_norm=True,
                output_dir=str(self.output_dir) + "_consistency"
            )
            
            # Create the output directory
            os.makedirs(consistency_config.output_dir, exist_ok=True)
            
            # Run training
            model, _ = train_faenet(**consistency_config.model_dump())
            
            # Verify the model was saved
            best_model_path = os.path.join(consistency_config.output_dir, "best_model.pt")
            self.assertTrue(os.path.exists(best_model_path), f"Best model was not saved to {best_model_path}")
            
            print("✅ Training with consistency loss passed!")
            
        except Exception as e:
            self.fail(f"Training with consistency loss failed with error: {e}")


if __name__ == "__main__":
    unittest.main()