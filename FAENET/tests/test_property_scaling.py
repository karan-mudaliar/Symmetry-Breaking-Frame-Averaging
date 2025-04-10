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
    
    def test_property_scaling_in_dataset_creation(self):
        """Test that property scaling parameter is respected when explicitly passed to create_dataloader."""
        # First, verify with default settings (scaling ENABLED)
        print("\nTesting create_dataloader with property scaling ENABLED (default)")
        train_loader_1, val_loader_1, test_loader_1, dataset_1 = create_dataloader(
            data_source=TEST_DATA_PATH,
            structure_col="slab",
            target_props=["WF_bottom", "WF_top"],
            batch_size=4,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            # use_property_scaling=True (default)
        )
        
        # Verify dataset has scaling enabled
        self.assertTrue(hasattr(dataset_1, 'use_scaling'), "Dataset should have use_scaling attribute")
        self.assertTrue(dataset_1.use_scaling, "Dataset use_scaling should be True by default")
        self.assertGreater(len(dataset_1.scalers), 0, "Dataset should have scalers when scaling is enabled")
        
        # Now, try with property scaling explicitly DISABLED 
        print("\nTesting create_dataloader with property scaling DISABLED")
        train_loader_2, val_loader_2, test_loader_2, dataset_2 = create_dataloader(
            data_source=TEST_DATA_PATH,
            structure_col="slab",
            target_props=["WF_bottom", "WF_top"],
            batch_size=4,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            use_property_scaling=False  # Explicitly disable
        )
        
        # Verify dataset has scaling disabled
        self.assertTrue(hasattr(dataset_2, 'use_scaling'), "Dataset should have use_scaling attribute")
        self.assertFalse(dataset_2.use_scaling, "Dataset use_scaling should be False when disabled")
        self.assertEqual(len(dataset_2.scalers), 0, "Dataset should have empty scalers when scaling is disabled")
        
        # Check that property values are handled properly in both cases
        # First batch from scaling enabled loader
        for batch_1 in train_loader_1:
            # First batch from scaling disabled loader
            for batch_2 in train_loader_2:
                # Check same properties in both
                for prop in ["WF_bottom", "WF_top"]:
                    # Both should have the property and original property
                    self.assertTrue(hasattr(batch_1, prop), f"Batch should have {prop} attribute")
                    self.assertTrue(hasattr(batch_1, f"{prop}_orig"), f"Batch should have {prop}_orig attribute")
                    self.assertTrue(hasattr(batch_2, prop), f"Batch should have {prop} attribute")
                    self.assertTrue(hasattr(batch_2, f"{prop}_orig"), f"Batch should have {prop}_orig attribute")
                    
                    # In the enabled case, the property should be different from original
                    values_1 = getattr(batch_1, prop)
                    orig_values_1 = getattr(batch_1, f"{prop}_orig")
                    self.assertFalse(torch.allclose(values_1, orig_values_1), 
                                     f"{prop} and {prop}_orig should be different when scaling is enabled")
                    
                    # In the disabled case, the property should be identical to original
                    values_2 = getattr(batch_2, prop)
                    orig_values_2 = getattr(batch_2, f"{prop}_orig")
                    self.assertTrue(torch.allclose(values_2, orig_values_2), 
                                     f"{prop} and {prop}_orig should be identical when scaling is disabled")
                    
                    # Print some values for visualization
                    print(f"\nScaling ENABLED: {prop} (first 2 values)")
                    print(f"  Scaled: {values_1[:2].tolist()}")
                    print(f"  Original: {orig_values_1[:2].tolist()}")
                    
                    print(f"\nScaling DISABLED: {prop} (first 2 values)")
                    print(f"  Values: {values_2[:2].tolist()}")
                    print(f"  Original: {orig_values_2[:2].tolist()}")
                    
                break  # Only need first batch from disabled loader
            break  # Only need first batch from enabled loader


if __name__ == "__main__":
    unittest.main()