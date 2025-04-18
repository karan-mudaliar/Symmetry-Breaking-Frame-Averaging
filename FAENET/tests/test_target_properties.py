#!/usr/bin/env python
"""
Test script for verifying target properties are correctly passed to the model.
This test focuses on ensuring that properties specified in the command line
flow correctly through to the model.
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
from faenet.dataset import SlabDataset
from faenet.faenet import FAENet

# Use an absolute path to the test data file
TEST_DATA_PATH = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 
    "..", "test_data", "surface_prop_data_set_top_bottom.csv"
))

class TestTargetProperties(unittest.TestCase):
    """Tests for target properties handling in the model."""
    
    def setUp(self):
        """Set up test environment"""
        # Check if test data exists
        self.test_data_exists = os.path.exists(TEST_DATA_PATH)
        print(f"Test data path: {TEST_DATA_PATH}")
        print(f"Test data exists: {self.test_data_exists}")
        
        # Create output directory if needed
        self.output_dir = Path("./test_outputs/target_properties")
        os.makedirs(self.output_dir, exist_ok=True)
    
    def test_target_properties_flow(self):
        """Test that target properties flow correctly from config to model."""
        if not self.test_data_exists:
            self.skipTest("Test data file not found")
            
        print("\n=== Testing Target Properties Flow ===")
        
        # Test with multiple target properties
        target_props = ["WF_top", "WF_bottom"]
        
        # First, verify the dataset loads these properties correctly
        dataset = SlabDataset(
            data_source=TEST_DATA_PATH,
            structure_col="slab",
            target_props=target_props,
            cutoff=6.0,
            max_neighbors=20,
            pbc=True
        )
        
        # Check that target properties were loaded
        self.assertIsNotNone(dataset.target_properties, "Dataset should have target_properties")
        self.assertIn("WF_top", dataset.target_properties, "WF_top should be in dataset.target_properties")
        self.assertIn("WF_bottom", dataset.target_properties, "WF_bottom should be in dataset.target_properties")
        
        print(f"Dataset loaded with target_properties: {list(dataset.target_properties.keys())}")
        
        # Create minimal model configuration
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
            epochs=1,  # Just one epoch for testing
            learning_rate=0.001,
            
            # Data parameters
            data_path=TEST_DATA_PATH,
            structure_col="slab",
            target_properties=target_props,
            
            # Disable MLflow for testing
            use_mlflow=False,
            
            # Other settings
            output_dir=str(self.output_dir),
            device="cpu"
        )
        
        # Extract model parameters
        kwargs = config.model_dump()
        
        try:
            print("\nStarting training with multiple target properties...")
            model, _ = train_faenet(**kwargs)
            
            # Verify model output properties
            self.assertTrue(hasattr(model, 'output_properties'), "Model should have output_properties attribute")
            self.assertEqual(len(model.output_properties), 2, "Model should have 2 output properties")
            self.assertIn("WF_top", model.output_properties, "WF_top should be in model.output_properties")
            self.assertIn("WF_bottom", model.output_properties, "WF_bottom should be in model.output_properties")
            
            print(f"Model output_properties: {model.output_properties}")
            print("Output blocks in model:")
            for block_name in model.output_blocks:
                print(f"  - {block_name}")
            
            # Check model's output blocks
            self.assertIn("WF_top", model.output_blocks, "Model should have output block for WF_top")
            self.assertIn("WF_bottom", model.output_blocks, "Model should have output block for WF_bottom")
            
            print("✅ Target properties test passed!")
            
        except Exception as e:
            import traceback
            self.fail(f"Test failed with error: {e}\n{traceback.format_exc()}")
    
    def test_output_blocks_match_properties(self):
        """Test that the model's output blocks match the specified target properties."""
        if not self.test_data_exists:
            self.skipTest("Test data file not found")
            
        print("\n=== Testing Output Blocks Match Properties ===")
        
        # Test with custom properties (reversed order to check order preservation)
        target_props = ["WF_bottom", "WF_top"]
        
        # Create a model directly (without training)
        model = FAENet(
            cutoff=6.0,
            num_gaussians=10,
            hidden_channels=32,
            num_filters=32,
            num_interactions=1,
            dropout=0.0,
            output_properties=target_props
        )
        
        # Verify model output properties
        self.assertTrue(hasattr(model, 'output_properties'), "Model should have output_properties attribute")
        self.assertEqual(len(model.output_properties), 2, "Model should have 2 output properties")
        self.assertEqual(model.output_properties, target_props, "Model output_properties should match target_props")
        
        # Check output blocks
        self.assertEqual(len(model.output_blocks), len(target_props), 
                         f"Model should have {len(target_props)} output blocks")
        
        # Verify each output block exists
        for prop in target_props:
            self.assertIn(prop, model.output_blocks, f"Model should have output block for {prop}")
        
        print("✅ Output blocks match properties test passed!")


if __name__ == "__main__":
    unittest.main()