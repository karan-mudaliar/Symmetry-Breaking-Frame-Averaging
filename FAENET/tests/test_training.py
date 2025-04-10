#!/usr/bin/env python
"""
Test script for training FAENet.
This script demonstrates training a small model for a few epochs.
"""
import os
import sys
import time
import torch
import unittest
from pathlib import Path

# Add the parent directory to sys.path to allow importing the faenet module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from faenet.config import Config
from faenet.faenet import FAENet
from faenet.dataset import SlabDataset, apply_frame_averaging_to_batch
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
        self.output_dir = Path("./test_outputs")
        os.makedirs(self.output_dir, exist_ok=True)
    
    def test_consistency_loss_and_mlflow(self):
        """Test consistency loss calculation and MLflow integration"""
        if not self.test_data_exists:
            self.skipTest("Test data file not found")
        
        # Import MLflow for tests
        try:
            import mlflow
            import uuid
        except ImportError:
            self.skipTest("MLflow not installed, skipping test")
            
        print("=== Testing Consistency Loss with MLflow Integration ===")
        
        # Create a completely unique experiment name to avoid any conflicts
        unique_id = str(uuid.uuid4())
        unique_experiment_name = f"FAENet_Test_Consistency_{unique_id}"
        
        # Set up configuration with MLflow enabled and consistency loss
        config = Config(
            # Model parameters (minimized for fast testing)
            cutoff=6.0,
            max_neighbors=20,
            num_gaussians=10,
            hidden_channels=32,
            num_filters=32,
            num_interactions=1,
            dropout=0.0,
            output_properties=["WF_top", "WF_bottom"],
            
            # Training parameters
            batch_size=4,
            epochs=1,  # Minimum to test logging
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
            run_name=f"consistency-test-{unique_id}",
            
            # Consistency loss parameters
            consistency_loss=True,  # Enable consistency loss
            consistency_weight=0.1,
            consistency_norm=True,
            
            # Other settings
            output_dir=self.output_dir,
            device="cpu",
            seed=42
        )
        
        try:
            # Save the original mlflow.log_metric function
            original_log_metric = mlflow.log_metric
            
            # Variables to track logging
            logged_metrics = []
            
            # Create a patching function to track what metrics are logged
            def patched_log_metric(key, value, **kwargs):
                logged_metrics.append(key)
                return original_log_metric(key, value, **kwargs)
            
            # Patch mlflow.log_metric for this test
            mlflow.log_metric = patched_log_metric
            
            # Run training with patched mlflow
            try:
                print("Starting training with consistency_loss enabled")
                kwargs = config.model_dump()
                target_props = kwargs.pop('target_properties')
                
                # Log key parameters for debugging
                print(f"Configuration: frame_averaging={kwargs.get('frame_averaging')}, consistency_loss={kwargs.get('consistency_loss')}")
                
                # Set end_mlflow_run=True to ensure run is properly closed
                kwargs['end_mlflow_run'] = True
                
                # IMPORTANT: We need to EITHER explicitly pass the parameters OR leave them in kwargs
                # Since we're setting them explicitly, we need to remove them from kwargs
                # to avoid "multiple values for keyword argument" error
                for param in ['consistency_loss', 'consistency_weight', 'consistency_norm']:
                    if param in kwargs:
                        kwargs.pop(param)
                
                # Add debug print statements
                print(f"Target properties: {target_props}")
                print(f"Frame averaging: {kwargs.get('frame_averaging')}")
                print(f"Consistency settings: enabled=True, weight=0.1, norm=True")
                
                model, _ = train_faenet(
                    target_properties=target_props,
                    consistency_loss=True,  # Force this to be True
                    consistency_weight=0.1,
                    consistency_norm=True,
                    **kwargs
                )
                
                # Debug model after training
                print(f"Model output properties: {model.output_properties}")
                print(f"Explicit model.output_properties check: {hasattr(model, 'output_properties')}")
                
                # Restore the original MLflow function
                mlflow.log_metric = original_log_metric
                
                # Check that expected metrics were logged
                print(f"Metrics logged during training: {logged_metrics}")
                self.assertTrue("train_loss" in logged_metrics, "train_loss should be logged")
                self.assertTrue("val_loss" in logged_metrics, "val_loss should be logged")
                self.assertTrue("test_loss" in logged_metrics, "test_loss should be logged")
                
                # Check for any consistency loss metrics
                consistency_metrics = [m for m in logged_metrics if "consistency" in m]
                self.assertTrue(len(consistency_metrics) > 0, 
                              f"No consistency metrics found in {logged_metrics}")
                
                print("✅ Consistency loss and MLflow integration test passed!")
                
            except Exception as e:
                # Restore original function even if there's an error
                mlflow.log_metric = original_log_metric
                raise e
                
        except Exception as e:
            self.fail(f"Consistency loss with MLflow integration test failed with error: {e}")
    
    def test_mlflow_integration(self):
        """Test MLflow integration during training"""
        if not self.test_data_exists:
            self.skipTest("Test data file not found")
        
        # Import MLflow for tests
        try:
            import mlflow
            import uuid
        except ImportError:
            self.skipTest("MLflow not installed, skipping test")
        
        print("=== Testing MLflow Integration ===")
        
        # Create a completely unique experiment name to avoid any conflicts
        unique_id = str(uuid.uuid4())
        unique_experiment_name = f"FAENet_Test_{unique_id}"
        
        # Set up configuration with MLflow enabled and our unique experiment name
        config = Config(
            # Model parameters (minimized for fast testing)
            cutoff=6.0,
            max_neighbors=20,
            num_gaussians=10,
            hidden_channels=32,
            num_filters=32,
            num_interactions=1,
            dropout=0.0,
            output_properties=["WF_top", "WF_bottom"],
            
            # Training parameters
            batch_size=4,
            epochs=1,  # Minimum to test logging
            learning_rate=0.001,
            weight_decay=1e-5,
            
            # Data parameters
            data_path=TEST_DATA_PATH,
            structure_col="slab",
            target_properties=["WF_top", "WF_bottom"],
            
            # MLflow settings
            use_mlflow=True,
            mlflow_experiment_name=unique_experiment_name,
            run_name=f"test-run-{unique_id}",
            
            # Other settings
            output_dir=self.output_dir,
            device="cpu",
            seed=42
        )
        
        # Create a new experiment and let the training create a new run
        run_id = None
        
        try:
            # Skip checking run counts - instead we'll patch the MLflow functions
            # to capture the correct run ID and verify metrics were logged
            
            # Save the original mlflow.log_metric function
            original_log_metric = mlflow.log_metric
            
            # Variables to track logging
            logged_metrics = []
            
            # Create a patching function to track what metrics are logged
            def patched_log_metric(key, value, **kwargs):
                logged_metrics.append(key)
                return original_log_metric(key, value, **kwargs)
            
            # Patch mlflow.log_metric for this test
            mlflow.log_metric = patched_log_metric
            
            # Run training with patched mlflow
            try:
                kwargs = config.model_dump()
                target_props = kwargs.pop('target_properties')
                
                # Set end_mlflow_run=True to ensure run is properly closed
                kwargs['end_mlflow_run'] = True
                
                model, _ = train_faenet(
                    target_properties=target_props,
                    **kwargs
                )
                
                # Restore the original MLflow function
                mlflow.log_metric = original_log_metric
                
                # Check that expected metrics were logged
                print(f"Metrics logged during training: {logged_metrics}")
                self.assertTrue("train_loss" in logged_metrics, "train_loss should be logged")
                self.assertTrue("val_loss" in logged_metrics, "val_loss should be logged")
                self.assertTrue("test_loss" in logged_metrics, "test_loss should be logged")
                
                print("✅ MLflow integration test passed!")
                
            except Exception as e:
                # Restore original function even if there's an error
                mlflow.log_metric = original_log_metric
                raise e
                
        except Exception as e:
            self.fail(f"MLflow integration test failed with error: {e}")
    
    def test_training(self):
        """Test a small training job"""
        if not self.test_data_exists:
            self.skipTest("Test data file not found")
            
        print("=== Testing FAENet Training ===")
        
        # Set up a configuration
        config = Config(
            # Model parameters
            cutoff=6.0,
            max_neighbors=40,
            num_gaussians=25,  # Reduced for faster training
            hidden_channels=64,  # Reduced for faster training
            num_filters=64,     # Reduced for faster training
            num_interactions=2,  # Reduced for faster training
            dropout=0.0,
            
            # Training parameters
            batch_size=4,
            epochs=2,  # Just a quick test
            learning_rate=0.001,
            weight_decay=1e-5,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            frame_averaging="2D",  # Use 2D frame averaging for slabs
            fa_method="all",
            
            # MLflow settings (disable for test)
            use_mlflow=False,
            seed=42,
            num_workers=0,  # Use 0 for testing
            
            # Data parameters
            data_path=TEST_DATA_PATH,
            structure_col="slab",
            target_properties=["WF_top", "WF_bottom", "cleavage_energy"],
            pbc=True,
            
            # General parameters
            output_dir=self.output_dir,
            device="cpu"  # Use CPU for testing
        )
        
        # Run training
        try:
            model, test_loader = train_faenet(
                data_path=config.data_path,
                structure_col=config.structure_col,
                target_properties=config.target_properties,
                output_dir=config.output_dir,
                frame_averaging=config.frame_averaging,
                fa_method=config.fa_method,
                cutoff=config.cutoff,
                max_neighbors=config.max_neighbors,
                batch_size=config.batch_size,
                epochs=config.epochs,
                learning_rate=config.learning_rate,
                seed=config.seed,
                device=config.device,
                num_workers=config.num_workers,
                num_gaussians=config.num_gaussians,
                hidden_channels=config.hidden_channels,
                num_filters=config.num_filters,
                num_interactions=config.num_interactions,
                dropout=config.dropout
            )
            
            print("\n✅ Training completed successfully!")
            
            # Verify the model was saved
            best_model_path = os.path.join(config.output_dir, "best_model.pt")
            self.assertTrue(os.path.exists(best_model_path), f"Best model was not saved to {best_model_path}")
            
            # Verify predictions were saved
            predictions_path = os.path.join(config.output_dir, "predictions.json")
            self.assertTrue(os.path.exists(predictions_path), f"Predictions were not saved to {predictions_path}")
            
        except Exception as e:
            self.fail(f"Training failed with error: {e}")
    
    def test_training_with_property_scaling(self):
        """Test property scaling during training process"""
        if not self.test_data_exists:
            self.skipTest("Test data file not found")
            
        print("=== Testing FAENet Training with Property Scaling ===")
        
        # Set up configuration with same parameters as basic training
        config = Config(
            # Model parameters
            cutoff=6.0,
            max_neighbors=40,
            num_gaussians=25,
            hidden_channels=64,
            num_filters=64,
            num_interactions=2,
            dropout=0.0,
            
            # Training parameters
            batch_size=4,
            epochs=1,  # Minimal for quick testing
            learning_rate=0.001,
            weight_decay=1e-5,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            frame_averaging="2D",
            fa_method="all",
            
            # MLflow settings (disable for test)
            use_mlflow=False,
            seed=42,
            num_workers=0,
            
            # Data parameters
            data_path=TEST_DATA_PATH,
            structure_col="slab",
            target_properties=["WF_top", "WF_bottom"],  # Focus on just two properties for simplicity
            pbc=True,
            
            # General parameters
            output_dir=self.output_dir / "property_scaling_test",
            device="cpu"
        )
        
        # Ensure output directory exists
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Run training
        try:
            import pickle
            import json
            import numpy as np
            
            print("Starting training with property scaling...")
            model, test_loader = train_faenet(
                data_path=config.data_path,
                structure_col=config.structure_col,
                target_properties=config.target_properties,
                output_dir=config.output_dir,
                frame_averaging=config.frame_averaging,
                fa_method=config.fa_method,
                cutoff=config.cutoff,
                max_neighbors=config.max_neighbors,
                batch_size=config.batch_size,
                epochs=config.epochs,
                learning_rate=config.learning_rate,
                seed=config.seed,
                device=config.device,
                num_workers=config.num_workers,
                num_gaussians=config.num_gaussians,
                hidden_channels=config.hidden_channels,
                num_filters=config.num_filters,
                num_interactions=config.num_interactions,
                dropout=config.dropout
            )
            
            print("✅ Training with property scaling completed!")
            
            # Verify property scalers were saved
            scaler_path = os.path.join(config.output_dir, "property_scalers.pkl")
            self.assertTrue(os.path.exists(scaler_path), 
                           f"Property scalers should be saved to {scaler_path}")
            
            # Load saved scalers and check they're properly fitted
            with open(scaler_path, 'rb') as f:
                scalers = pickle.load(f)
                
            # Verify scalers contain our target properties
            for prop in config.target_properties:
                self.assertIn(prop, scalers, f"Scaler should contain {prop}")
                self.assertTrue(hasattr(scalers[prop], 'mean_'), 
                               f"Scaler for {prop} should be fitted")
                print(f"Scaler for {prop}: mean={scalers[prop].mean_[0]:.4f}, std={scalers[prop].scale_[0]:.4f}")
            
            # Check predictions file for evidence of inverse transform
            predictions_path = os.path.join(config.output_dir, "predictions.json")
            self.assertTrue(os.path.exists(predictions_path), 
                           f"Predictions should be saved to {predictions_path}")
            
            # Load predictions and analyze
            with open(predictions_path, 'r') as f:
                predictions = json.load(f)
                
            # Verify predictions
            self.assertGreater(len(predictions), 0, "Should have at least one prediction")
            
            # Check that we have predictions from all three datasets
            datasets = [p.get("dataset", "unknown") for p in predictions]
            unique_datasets = set(datasets)
            
            print(f"Found {len(predictions)} total predictions across {len(unique_datasets)} datasets")
            for dataset in unique_datasets:
                count = datasets.count(dataset)
                print(f"  {dataset}: {count} predictions")
            
            # Verify all three datasets are represented
            self.assertIn("train", unique_datasets, "Should have predictions from training set")
            self.assertIn("val", unique_datasets, "Should have predictions from validation set")
            self.assertIn("test", unique_datasets, "Should have predictions from test set")
            
            # Get a sample prediction to check fields
            sample_pred = predictions[0]
            
            # Check prediction keys 
            prop = config.target_properties[0]  # Use the first property as an example
            expected_keys = [
                prop,               # Unscaled prediction
                f"{prop}_scaled",   # Scaled prediction
                f"{prop}_true",     # Unscaled ground truth
                f"{prop}_true_scaled"  # Scaled ground truth
            ]
            
            for key in expected_keys:
                self.assertIn(key, sample_pred, f"Prediction should contain {key}")
            
            # Print example values from one prediction for verification
            print(f"\nSample prediction values for property {prop}:")
            print(f"  Unscaled prediction: {sample_pred[prop]:.4f}")
            print(f"  Scaled prediction: {sample_pred[f'{prop}_scaled']:.4f}")
            print(f"  Unscaled ground truth: {sample_pred[f'{prop}_true']:.4f}")
            print(f"  Scaled ground truth: {sample_pred[f'{prop}_true_scaled']:.4f}")
            
            # Verify scaled and unscaled values are different
            self.assertNotEqual(sample_pred[prop], sample_pred[f"{prop}_scaled"],
                               "Scaled and unscaled predictions should be different")
            self.assertNotEqual(sample_pred[f"{prop}_true"], sample_pred[f"{prop}_true_scaled"],
                               "Scaled and unscaled ground truth should be different")
            
            # Verify scaling by checking statistics
            # Extract all values
            unscaled_preds = [p[prop] for p in predictions]
            scaled_preds = [p[f"{prop}_scaled"] for p in predictions]
            unscaled_truths = [p[f"{prop}_true"] for p in predictions]
            scaled_truths = [p[f"{prop}_true_scaled"] for p in predictions]
            
            # Print statistics
            print(f"\nStatistics for {prop}:")
            print(f"  Unscaled predictions - Mean: {np.mean(unscaled_preds):.4f}, Std: {np.std(unscaled_preds):.4f}")
            print(f"  Scaled predictions   - Mean: {np.mean(scaled_preds):.4f}, Std: {np.std(scaled_preds):.4f}")
            print(f"  Unscaled truth       - Mean: {np.mean(unscaled_truths):.4f}, Std: {np.std(unscaled_truths):.4f}")
            print(f"  Scaled truth         - Mean: {np.mean(scaled_truths):.4f}, Std: {np.std(scaled_truths):.4f}")
            
            # Verify the basic properties of standardization:
            # 1. Scaled values should have mean closer to 0 and std closer to 1 compared to unscaled
            # 2. abs(mean) of scaled values should be smaller than abs(mean) of unscaled
            # For predictions:
            self.assertLess(abs(np.mean(scaled_preds)), abs(np.mean(unscaled_preds)),
                           "Scaled predictions should have mean closer to 0")
            
            # For ground truth:
            self.assertLess(abs(np.mean(scaled_truths)), abs(np.mean(unscaled_truths)),
                           "Scaled ground truth should have mean closer to 0")
            
            # Verify predictions are stored for all datasets (train, val, test)
            for dataset_type in ["train", "val", "test"]:
                dataset_preds = [p for p in predictions if p.get("dataset") == dataset_type]
                self.assertGreater(len(dataset_preds), 0, 
                                  f"Should have predictions for {dataset_type} dataset")
                
                sample = dataset_preds[0]
                print(f"\nSample {dataset_type} prediction:")
                print(f"  ID: {sample['jid']}")
                print(f"  {prop}: {sample[prop]:.4f} (unscaled)")
                print(f"  {prop}_scaled: {sample[f'{prop}_scaled']:.4f}")
                print(f"  {prop}_true: {sample[f'{prop}_true']:.4f} (unscaled)")
                print(f"  {prop}_true_scaled: {sample[f'{prop}_true_scaled']:.4f}")
            
        except Exception as e:
            self.fail(f"Training with property scaling failed: {e}")


# This allows running the test directly
if __name__ == "__main__":
    unittest.main()