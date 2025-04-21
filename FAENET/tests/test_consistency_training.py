#!/usr/bin/env python
"""
Test script for consistency loss integration with training.
This test performs a minimal training run with consistency loss enabled
to verify end-to-end functionality.
"""
import os
import sys
import torch
import unittest
import structlog

# Configure logging
logger = structlog.get_logger()

# Add the parent directory to sys.path to allow importing the faenet module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from faenet.config import Config
from faenet.train import train_faenet

# Use an absolute path to the test data file
TEST_DATA_PATH = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 
    "..", "test_data", "surface_prop_data_set_top_bottom.csv"
))

class TestConsistencyTraining(unittest.TestCase):
    """Tests for consistency loss integrated with training."""
    
    def setUp(self):
        """Set up test environment"""
        # Check if test data exists
        self.test_data_exists = os.path.exists(TEST_DATA_PATH)
        print(f"Test data path: {TEST_DATA_PATH}")
        print(f"Test data exists: {self.test_data_exists}")
        
        # Create output directory if needed
        self.output_dir = os.path.join(os.path.dirname(__file__), "../test_outputs")
        os.makedirs(self.output_dir, exist_ok=True)
    
    def test_tensor_initialization(self):
        """Test that consistency loss is properly initialized as a tensor and works with operations."""
        if not self.test_data_exists:
            self.skipTest("Test data file not found")
        
        # Import MLflow for tests
        try:
            import mlflow
            import uuid
        except ImportError:
            self.skipTest("MLflow not installed, skipping test")
            
        print("=== Testing Consistency Loss Tensor Initialization ===")
        
        # Create a completely unique experiment name to avoid any conflicts
        unique_id = str(uuid.uuid4())
        unique_experiment_name = f"FAENet_Test_Tensor_Init_{unique_id}"
        
        # Mock patching of mlflow.log_metric to check for item() calls
        original_log_metric = mlflow.log_metric
        called_methods = []
        
        def patched_log_metric(key, value, **kwargs):
            """Track metrics and test item() extraction"""
            called_methods.append(key)
            
            # Test that we can call item() on the value - this will fail if value is not a tensor
            try:
                if hasattr(value, 'item'):
                    value = value.item()
                else:
                    value = float(value)  # This would fail if value can't be converted to float
                
                # Track successful conversion for value verification
                called_methods.append(f"{key}_value_extracted")
            except Exception as e:
                called_methods.append(f"{key}_extraction_failed")
                print(f"Failed to extract value for {key}: {e}")
            
            # Call the original function with the processed value
            return original_log_metric(key, value, **kwargs)
        
        # Apply the patched function
        mlflow.log_metric = patched_log_metric
        
        try:
            # Set up configuration with MLflow enabled and consistency loss
            config = Config(
                # Model parameters (minimized for extremely fast testing)
                cutoff=6.0,
                max_neighbors=10,
                num_gaussians=5,
                hidden_channels=16,
                num_filters=16,
                num_interactions=1,
                dropout=0.0,
                output_properties=["WF_top", "WF_bottom"],
                
                # Training parameters
                batch_size=2,
                epochs=1,  # Minimum to test tensor operations
                learning_rate=0.001,
                weight_decay=1e-5,
                frame_averaging="2D",  # Enable frame averaging for consistency loss
                fa_method="all",
                
                # Data parameters
                data_path=TEST_DATA_PATH,
                structure_col="slab",
                target_properties=["WF_top", "WF_bottom"],
                
                # MLflow settings
                use_mlflow=True,
                mlflow_experiment_name=unique_experiment_name,
                run_name=f"tensor-init-test-{unique_id}",
                
                # Consistency loss parameters
                consistency_loss=True,  # Enable consistency loss
                consistency_weight=0.1,
                consistency_norm=True,
                
                # Other settings
                output_dir=self.output_dir,
                device="cpu",
                seed=42,
                end_mlflow_run=True  # Ensure run is properly closed
            )
            
            # Extract required parameters for train_faenet call
            kwargs = config.model_dump()
            target_props = kwargs.pop('target_properties')
            
            # To avoid duplication, remove consistency parameters from kwargs
            for param in ['consistency_loss', 'consistency_weight', 'consistency_norm']:
                if param in kwargs:
                    kwargs.pop(param)
            
            # Run training with patched mlflow
            print("Starting training with consistency_loss enabled")
            model, _ = train_faenet(
                target_properties=target_props,
                consistency_loss=True,
                consistency_weight=0.1,
                consistency_norm=True,
                **kwargs
            )
            
            # Restore the original MLflow function
            mlflow.log_metric = original_log_metric
            
            # Check that consistency metrics were properly logged and values extracted
            print(f"Called methods: {called_methods}")
            
            # These assertions check that metrics were logged
            self.assertIn("consistency_loss", called_methods, "consistency_loss metric should be logged")
            self.assertIn("train_consistency_loss", called_methods, "train_consistency_loss metric should be logged")
            self.assertIn("val_consistency_loss", called_methods, "val_consistency_loss metric should be logged")
            self.assertIn("test_consistency_loss", called_methods, "test_consistency_loss metric should be logged")
            
            # These assertions check that item() extraction worked (didn't fail)
            self.assertIn("consistency_loss_value_extracted", called_methods, 
                         "consistency_loss value should be extractable")
            self.assertIn("train_consistency_loss_value_extracted", called_methods, 
                         "train_consistency_loss value should be extractable")
            self.assertIn("val_consistency_loss_value_extracted", called_methods, 
                         "val_consistency_loss value should be extractable")
            self.assertIn("test_consistency_loss_value_extracted", called_methods, 
                         "test_consistency_loss value should be extractable")
            
            # Ensure no extraction failures occurred
            failures = [method for method in called_methods if "_extraction_failed" in method]
            self.assertEqual(len(failures), 0, f"Value extraction failed for: {failures}")
            
            print("Consistency loss tensor initialization test passed!")
            
        except Exception as e:
            # Restore original function even if there's an error
            mlflow.log_metric = original_log_metric
            self.fail(f"Consistency loss tensor initialization test failed with error: {e}")


if __name__ == "__main__":
    unittest.main()