#!/usr/bin/env python
"""
Test script for property scaling functionality.
"""
import os
import sys
import unittest
import numpy as np
import torch
import pandas as pd
import tempfile
import pickle

# Add the parent directory to sys.path to allow importing the faenet module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from faenet.dataset import create_dataloader
from faenet.train import train_faenet
from faenet.config import Config
from sklearn.preprocessing import StandardScaler

# Use an absolute path to the test data file
TEST_DATA_PATH = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 
    "..", "test_data", "surface_prop_data_set_top_bottom.csv"
))

class TestPropertyScaling(unittest.TestCase):
    """Test cases for property scaling functionality."""
    
    def setUp(self):
        """Set up test case."""
        if not os.path.exists(TEST_DATA_PATH):
            self.skipTest(f"Test data file not found: {TEST_DATA_PATH}")
        
        # Create temporary directory for outputs
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Print test data path for debugging
        print(f"Test data path: {TEST_DATA_PATH}")
    
    def tearDown(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()
    
    def test_property_scaling_in_dataset(self):
        """Test that properties are properly scaled in the dataset."""
        # Create dataloader with property scaling
        train_loader, val_loader, test_loader, dataset = create_dataloader(
            data_source=TEST_DATA_PATH,
            structure_col="slab",
            target_props=["WF_bottom", "WF_top"],
            batch_size=4,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15
        )
        
        # Check that scalers were created
        self.assertTrue(hasattr(dataset, 'scalers'), "Dataset should have scalers attribute")
        self.assertTrue(len(dataset.scalers) > 0, "Dataset should have at least one scaler")
        
        # Check each scaler
        for prop, scaler in dataset.scalers.items():
            self.assertIsInstance(scaler, StandardScaler, f"Scaler for {prop} should be a StandardScaler")
            self.assertTrue(hasattr(scaler, 'mean_'), f"Scaler for {prop} should be fitted")
            
            # Log scaler statistics
            print(f"Property {prop}: mean={scaler.mean_[0]:.4f}, std={scaler.scale_[0]:.4f}")
    
    def test_scaled_values_in_batch(self):
        """Test that scaled values are included in the batch."""
        # Create dataloader with property scaling
        train_loader, val_loader, test_loader, dataset = create_dataloader(
            data_source=TEST_DATA_PATH,
            structure_col="slab",
            target_props=["WF_bottom", "WF_top"],
            batch_size=4,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15
        )
        
        # Get a batch of data
        for batch in train_loader:
            # Check that each property has both scaled and original values
            for prop in ["WF_bottom", "WF_top"]:
                self.assertTrue(hasattr(batch, prop), f"Batch should have {prop} attribute")
                self.assertTrue(hasattr(batch, f"{prop}_orig"), f"Batch should have {prop}_orig attribute")
                
                # Check that scaled values are different from original values
                if batch[prop].shape[0] > 0:
                    # Convert tensors to numpy for easier comparison
                    scaled = batch[prop].cpu().numpy()
                    orig = getattr(batch, f"{prop}_orig").cpu().numpy()
                    
                    # Handle different tensor shapes
                    if len(scaled.shape) == 1:
                        # 1D array (batch_size,)
                        pass  # No reshaping needed
                    elif len(scaled.shape) == 2:
                        # 2D array (batch_size, 1)
                        scaled = scaled.flatten()
                        orig = orig.flatten()
                    
                    # Verify scaled values have approximately mean 0 and std 1
                    if scaled.size > 1:  # Only if we have enough values
                        self.assertAlmostEqual(np.mean(scaled), 0.0, delta=1.0)
                        self.assertAlmostEqual(np.std(scaled), 1.0, delta=1.0)
                    
                    # Display sample values for verification
                    for i in range(min(3, len(scaled))):
                        print(f"{prop} sample {i}: orig={orig[i]:.4f}, scaled={scaled[i]:.4f}")
            
            # Only check the first batch
            break
    
    def test_saving_and_loading_scalers(self):
        """Test saving and loading scalers works correctly."""
        # Create dataloader with property scaling
        train_loader, val_loader, test_loader, dataset = create_dataloader(
            data_source=TEST_DATA_PATH,
            structure_col="slab",
            target_props=["WF_bottom", "WF_top"],
            batch_size=4,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15
        )
        
        # Save scalers to temporary file
        scaler_path = os.path.join(self.temp_dir.name, "test_scalers.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump(dataset.scalers, f)
        
        # Load scalers back
        with open(scaler_path, 'rb') as f:
            loaded_scalers = pickle.load(f)
        
        # Compare original and loaded scalers
        self.assertEqual(len(dataset.scalers), len(loaded_scalers), 
                        "Number of scalers should match")
        
        for prop in dataset.scalers:
            self.assertIn(prop, loaded_scalers, f"Property {prop} should be in loaded scalers")
            
            # Compare mean and std values
            orig_mean = dataset.scalers[prop].mean_[0]
            loaded_mean = loaded_scalers[prop].mean_[0]
            self.assertAlmostEqual(orig_mean, loaded_mean, places=5, 
                                  msg=f"Mean for {prop} should match after loading")
            
            orig_std = dataset.scalers[prop].scale_[0]
            loaded_std = loaded_scalers[prop].scale_[0]
            self.assertAlmostEqual(orig_std, loaded_std, places=5, 
                                  msg=f"Std for {prop} should match after loading")
            
            # Test inverse transform gives same results
            test_value = 1.5  # A standardized value
            orig_inverse = dataset.scalers[prop].inverse_transform([[test_value]])[0][0]
            loaded_inverse = loaded_scalers[prop].inverse_transform([[test_value]])[0][0]
            self.assertAlmostEqual(orig_inverse, loaded_inverse, places=5,
                                  msg=f"Inverse transform for {prop} should match after loading")
    
    def test_optional_property_scaling(self):
        """Test that property scaling can be disabled."""
        # Create dataloader WITHOUT property scaling
        train_loader, val_loader, test_loader, dataset = create_dataloader(
            data_source=TEST_DATA_PATH,
            structure_col="slab",
            target_props=["WF_bottom", "WF_top"],
            batch_size=4,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            use_property_scaling=False  # Disable property scaling
        )
        
        # Check that scalers were not created or are empty
        self.assertTrue(hasattr(dataset, 'scalers'), "Dataset should have scalers attribute")
        self.assertEqual(len(dataset.scalers), 0, "Dataset should have no scalers when scaling is disabled")
        
        # Verify we're keeping track of the scaling decision
        self.assertTrue(hasattr(dataset, 'use_scaling'), "Dataset should track scaling decision")
        self.assertFalse(dataset.use_scaling, "use_scaling should be False when disabled")
        
        # Check that original values are passed directly
        for batch in train_loader:
            for prop in ["WF_bottom", "WF_top"]:
                # Should have both the main property and the original property
                self.assertTrue(hasattr(batch, prop), f"Batch should have {prop} attribute")
                self.assertTrue(hasattr(batch, f"{prop}_orig"), f"Batch should have {prop}_orig attribute")
                
                # Get values
                values = getattr(batch, prop)
                orig_values = getattr(batch, f"{prop}_orig")
                
                # They should be identical since scaling is disabled
                self.assertTrue(torch.allclose(values, orig_values), 
                               f"{prop} and {prop}_orig should be identical when scaling is disabled")
                
                print(f"Property {prop}: {values}")
                print(f"Original {prop}: {orig_values}")
                
                # With scaling disabled, mean and std should reflect original data
                if values.shape[0] > 1:
                    mean = torch.mean(values).item()
                    std = torch.std(values).item()
                    print(f"{prop} statistics - Mean: {mean:.4f}, Std: {std:.4f}")
                    
                    # These should NOT be near 0 and 1 since scaling is disabled
                    self.assertNotAlmostEqual(mean, 0.0, delta=0.5, 
                                             msg=f"Mean of unscaled {prop} should NOT be near 0")
            
            # Only check first batch
            break
    
    def test_property_scaling_in_train_function(self):
        """Test that property scaling parameter is respected in the train_faenet function."""
        # Set up minimal configuration for a quick test
        output_dir = os.path.join(self.temp_dir.name, "scaling_test_output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Run with property scaling DISABLED
        print("\nTesting train_faenet with property scaling DISABLED")
        model, test_loader = train_faenet(
            data_path=TEST_DATA_PATH,
            structure_col="slab",
            target_properties=["WF_bottom", "WF_top"],
            epochs=1,  # Minimum for test
            batch_size=4,
            num_interactions=1,  # Smaller model for speed
            hidden_channels=16,
            output_dir=output_dir,
            use_mlflow=False,
            device="cpu",
            use_property_scaling=False  # Explicitly disable
        )
        
        # Verify the dataset has scaling disabled
        dataset = test_loader.dataset.dataset
        
        self.assertTrue(hasattr(dataset, 'use_scaling'), 
                       "Dataset should have use_scaling attribute")
        self.assertFalse(dataset.use_scaling, 
                        "Dataset use_scaling should be False when disabled")
        
        # Check scaler file
        scaler_path = os.path.join(output_dir, "property_scalers.pkl")
        
        # If file exists, it should contain empty or no scalers
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scalers = pickle.load(f)
            self.assertEqual(len(scalers), 0, 
                           "Property scalers should be empty when scaling is disabled")
        
        # Run with property scaling ENABLED (default)
        print("\nTesting train_faenet with property scaling ENABLED")
        output_dir = os.path.join(self.temp_dir.name, "scaling_enabled_test_output")
        os.makedirs(output_dir, exist_ok=True)
        
        model, test_loader = train_faenet(
            data_path=TEST_DATA_PATH,
            structure_col="slab",
            target_properties=["WF_bottom", "WF_top"],
            epochs=1,  # Minimum for test
            batch_size=4,
            num_interactions=1,  # Smaller model for speed
            hidden_channels=16,
            output_dir=output_dir,
            use_mlflow=False,
            device="cpu"
            # use_property_scaling=True is the default
        )
        
        # Verify the dataset has scaling enabled
        dataset = test_loader.dataset.dataset
        
        self.assertTrue(hasattr(dataset, 'use_scaling'), 
                       "Dataset should have use_scaling attribute")
        self.assertTrue(dataset.use_scaling, 
                       "Dataset use_scaling should be True when enabled")
        
        # Check scaler file
        scaler_path = os.path.join(output_dir, "property_scalers.pkl")
        self.assertTrue(os.path.exists(scaler_path), 
                       "Property scalers file should exist when scaling is enabled")
        
        with open(scaler_path, 'rb') as f:
            scalers = pickle.load(f)
        
        self.assertGreater(len(scalers), 0, 
                          "Property scalers should not be empty when scaling is enabled")
        
        # Verify scalers are properly fitted
        for prop in ["WF_bottom", "WF_top"]:
            self.assertIn(prop, scalers, f"Scaler for {prop} should exist")
            self.assertTrue(hasattr(scalers[prop], 'mean_'), 
                          f"Scaler for {prop} should be fitted")
            print(f"Scaler for {prop}: mean={scalers[prop].mean_[0]:.4f}, std={scalers[prop].scale_[0]:.4f}")


if __name__ == "__main__":
    unittest.main()