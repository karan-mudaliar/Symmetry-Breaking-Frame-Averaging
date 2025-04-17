#!/usr/bin/env python
"""
Test script that replicates a real training run with consistency loss enabled.
This test runs a minimal 1-epoch training on the sample data to verify end-to-end functionality.
"""
import os
import sys
import torch
import unittest
from pathlib import Path

# Add the parent directory to sys.path to allow importing the faenet module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from faenet.config import Config
from faenet.dataset import SlabDataset
from faenet.train import train_faenet

# Use an absolute path to the test data file
TEST_DATA_PATH = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 
    "..", "test_data", "surface_prop_data_set_top_bottom.csv"
))

class TestRunWithConsistency(unittest.TestCase):
    """Real-world test for training with consistency loss enabled."""
    
    def setUp(self):
        """Set up test environment"""
        # Check if test data exists
        self.test_data_exists = os.path.exists(TEST_DATA_PATH)
        print(f"Test data path: {TEST_DATA_PATH}")
        print(f"Test data exists: {self.test_data_exists}")
        
        # Create output directory if needed
        self.output_dir = Path(os.path.dirname(__file__)) / "../test_outputs"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def test_real_world_training(self):
        """Run a real training job with consistency loss enabled on sample data."""
        if not self.test_data_exists:
            self.skipTest("Test data file not found")
            
        print("\n=== Testing Real-World Training with Consistency Loss ===\n")
        
        # First, check the test data to ensure it's usable
        try:
            test_dataset = SlabDataset(
                data_source=TEST_DATA_PATH,
                structure_col="slab",
                target_props=["WF_top", "WF_bottom"],
                cutoff=6.0,
                max_neighbors=20,
                pbc=True
            )
            print(f"Test dataset loaded successfully with {len(test_dataset)} samples")
        except Exception as e:
            self.fail(f"Failed to load test dataset: {e}")
        
        # Set up configuration exactly as would be done in a real training run
        config_params = {
            # Model parameters (minimized for fast testing)
            'cutoff': 6.0,
            'max_neighbors': 20,
            'num_gaussians': 10,
            'hidden_channels': 32,
            'num_filters': 32,
            'num_interactions': 1,
            'dropout': 0.0,
            'output_properties': ["WF_top", "WF_bottom"],
            
            # Training parameters
            'batch_size': 4,
            'epochs': 2,  # Run 2 epochs to test epoch transitions
            'learning_rate': 0.001,
            'weight_decay': 1e-5,
            'frame_averaging': "2D",  # Enable frame averaging for consistency loss
            'fa_method': "all",
            
            # Data parameters
            'data_path': TEST_DATA_PATH,
            'structure_col': "slab",
            'target_properties': ["WF_top", "WF_bottom"],
            
            # MLflow disabled to simplify test
            'use_mlflow': False,
            
            # Consistency loss parameters - explicitly enabled
            'consistency_loss': True,
            'consistency_weight': 0.1,
            'consistency_norm': True,
            
            # Other settings
            'output_dir': str(self.output_dir),
            'device': "cpu",
            'seed': 42,
            'num_workers': 0,  # Use 0 for testing
            'checkpoint_interval': 1  # Save checkpoint every epoch
        }
        
        print("Creating Config with consistency_loss=True")
        config = Config(**config_params)
        print(f"Config created with consistency_loss={config.consistency_loss}")
        
        try:
            # Run training directly from the Config object
            print("\nStarting training run with consistency loss...")
            
            model, test_loader = train_faenet(**config.model_dump())
            
            print("\n✅ Training with consistency loss completed successfully!")
            
            # Verify the model was saved
            best_model_path = os.path.join(self.output_dir, "best_model.pt")
            self.assertTrue(os.path.exists(best_model_path), f"Best model was not saved to {best_model_path}")
            
            # Verify predictions were saved
            predictions_path = os.path.join(self.output_dir, "predictions.json")
            self.assertTrue(os.path.exists(predictions_path), f"Predictions were not saved to {predictions_path}")
            
            # Check model structure
            self.assertTrue(hasattr(model, 'output_properties'), "Model should have output_properties attribute")
            self.assertEqual(len(model.output_properties), 2, "Model should have two output properties")
            
            # Check checkpoint for both epochs
            checkpoint1_path = os.path.join(self.output_dir, "checkpoint_epoch_1.pt")
            self.assertTrue(os.path.exists(checkpoint1_path), f"Checkpoint for epoch 1 was not saved")
            
            checkpoint2_path = os.path.join(self.output_dir, "checkpoint_epoch_2.pt")  
            self.assertTrue(os.path.exists(checkpoint2_path), f"Checkpoint for epoch 2 was not saved")
            
            # Verify backward pass by running model inference and checking gradients
            try:
                # Create a dummy batch with a single input
                batch = next(iter(test_loader)).to("cpu")
                
                # Set model to training mode
                model.train()
                
                # Forward pass
                outputs = model(batch)
                
                # Verify outputs are tensors with gradients
                for prop in model.output_properties:
                    self.assertTrue(outputs[prop].requires_grad, f"Output for {prop} should require gradients")
                
                # Create a dummy loss
                dummy_loss = sum(outputs[prop].mean() for prop in model.output_properties)
                
                # Backward pass
                dummy_loss.backward()
                
                # Check that gradients are computed for model parameters
                param_with_grad = False
                for param in model.parameters():
                    if param.grad is not None:
                        param_with_grad = True
                        break
                
                self.assertTrue(param_with_grad, "At least one parameter should have gradients")
                
                print("✅ Gradient flow test passed!")
            except Exception as e:
                self.fail(f"Gradient test failed: {e}")
            
        except Exception as e:
            import traceback
            self.fail(f"Training failed with error: {e}\n{traceback.format_exc()}")

if __name__ == "__main__":
    unittest.main()