#!/usr/bin/env python
"""
Comprehensive test for frame averaging pipeline in FAENet.
This test verifies that frame averaging is correctly applied throughout the entire training process.
"""
import os
import sys
import torch
import unittest
from pathlib import Path
import numpy as np
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader
import structlog

# Configure structlog
logger = structlog.get_logger()

# Add the parent directory to sys.path to allow importing the faenet module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from faenet.config import Config
from faenet.dataset import SlabDataset, apply_frame_averaging_to_batch
from faenet.train import train_faenet, process_frames, compute_prediction_loss, compute_frame_consistency_loss
from faenet.faenet import FAENet
from faenet.frame_averaging import frame_averaging_2D, frame_averaging_3D, compute_consistency_loss, ConsistencyLoss

# Use an absolute path to the test data file
TEST_DATA_PATH = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 
    "..", "test_data", "surface_prop_data_set_top_bottom.csv"
))

class TestFrameAveragingPipeline(unittest.TestCase):
    """Comprehensive test for frame averaging pipeline."""
    
    def setUp(self):
        """Set up test environment"""
        # Check if test data exists
        self.test_data_exists = os.path.exists(TEST_DATA_PATH)
        print(f"Test data path: {TEST_DATA_PATH}")
        print(f"Test data exists: {self.test_data_exists}")
        
        # Create output directory if needed
        self.output_dir = Path("./test_outputs/frame_averaging_pipeline")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set random seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
    
    def test_dataset_frame_averaging(self):
        """Test that the dataset correctly applies frame averaging during loading."""
        if not self.test_data_exists:
            self.skipTest("Test data file not found")
            
        print("\n=== Testing Dataset Frame Averaging ===")
        
        # Create dataset with 2D frame averaging
        dataset_2d = SlabDataset(
            data_source=TEST_DATA_PATH,
            structure_col="slab",
            target_props=["WF_top", "WF_bottom"],
            cutoff=6.0,
            max_neighbors=20,
            frame_averaging="2D",
            fa_method="all"
        )
        
        # Create dataset with 3D frame averaging
        dataset_3d = SlabDataset(
            data_source=TEST_DATA_PATH,
            structure_col="slab",
            target_props=["WF_top", "WF_bottom"],
            cutoff=6.0,
            max_neighbors=20,
            frame_averaging="3D",
            fa_method="all"
        )
        
        # Create dataset without frame averaging for comparison
        dataset_no_fa = SlabDataset(
            data_source=TEST_DATA_PATH,
            structure_col="slab",
            target_props=["WF_top", "WF_bottom"],
            cutoff=6.0,
            max_neighbors=20,
            frame_averaging=None
        )
        
        # Check dataset lengths
        print(f"Dataset size: {len(dataset_2d)} structures")
        self.assertGreater(len(dataset_2d), 0, "Dataset should have at least one structure")
        
        # Verify each dataset produces the expected frame attributes
        if len(dataset_2d) > 0:
            # Get first item from each dataset
            data_2d = dataset_2d[0]
            data_3d = dataset_3d[0]
            data_no_fa = dataset_no_fa[0]
            
            # Check 2D frame averaging
            self.assertTrue(hasattr(data_2d, 'fa_pos'), "Data should have frame averaged positions")
            self.assertTrue(hasattr(data_2d, 'fa_cell'), "Data should have frame averaged cell")
            self.assertTrue(hasattr(data_2d, 'fa_rot'), "Data should have rotation matrices")
            
            # Check correct number of frames
            self.assertEqual(len(data_2d.fa_pos), 4, "2D frame averaging should produce 4 frames")
            self.assertEqual(len(data_3d.fa_pos), 8, "3D frame averaging should produce 8 frames")
            
            # Check no frame averaging dataset doesn't have these attributes
            self.assertFalse(hasattr(data_no_fa, 'fa_pos'), "Data without frame averaging should not have fa_pos")
            
            # Check that 2D frame averaging preserves z-coordinates
            z_preserved = all(torch.allclose(data_2d.fa_pos[i][:, 2], data_2d.pos[:, 2], atol=1e-5) 
                             for i in range(len(data_2d.fa_pos)))
            self.assertTrue(z_preserved, "2D frame averaging should preserve z-axis coordinates")
            
            print("✅ Dataset frame averaging checks passed!")
    
    def test_batch_frame_averaging(self):
        """Test that frame averaging is correctly applied to batches."""
        if not self.test_data_exists:
            self.skipTest("Test data file not found")
            
        print("\n=== Testing Batch Frame Averaging ===")
        
        # Create dataset without frame averaging
        dataset = SlabDataset(
            data_source=TEST_DATA_PATH,
            structure_col="slab",
            target_props=["WF_top", "WF_bottom"],
            cutoff=6.0,
            max_neighbors=20,
            frame_averaging=None
        )
        
        # Create dataloader
        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        
        # Get a batch
        for batch in loader:
            break
        
        # Apply frame averaging to batch
        batch_2d = apply_frame_averaging_to_batch(batch.clone(), "all", "2D")
        batch_3d = apply_frame_averaging_to_batch(batch.clone(), "all", "3D")
        
        # Verify frame averaging attributes
        self.assertTrue(hasattr(batch_2d, 'fa_pos'), "Batch should have frame averaged positions")
        self.assertTrue(hasattr(batch_2d, 'fa_cell'), "Batch should have frame averaged cell")
        self.assertTrue(hasattr(batch_2d, 'fa_rot'), "Batch should have rotation matrices")
        
        # Check correct number of frames
        self.assertEqual(len(batch_2d.fa_pos), 4, "2D frame averaging should produce 4 frames")
        self.assertEqual(len(batch_3d.fa_pos), 8, "3D frame averaging should produce 8 frames")
        
        # Check original batch is unchanged
        self.assertFalse(hasattr(batch, 'fa_pos'), "Original batch should not have fa_pos")
        
        print("✅ Batch frame averaging checks passed!")
    
    def test_frame_processing(self):
        """Test that frame processing works correctly in the training loop."""
        if not self.test_data_exists:
            self.skipTest("Test data file not found")
            
        print("\n=== Testing Frame Processing ===")
        
        # Create a simple model
        model = FAENet(
            cutoff=6.0,
            num_gaussians=10,
            hidden_channels=32,
            num_filters=32,
            num_interactions=1,
            target_properties=["WF_top", "WF_bottom"]
        )
        
        # Create dataset without frame averaging
        dataset = SlabDataset(
            data_source=TEST_DATA_PATH,
            structure_col="slab",
            target_props=["WF_top", "WF_bottom"],
            cutoff=6.0,
            max_neighbors=20,
            frame_averaging=None
        )
        
        # Create dataloader
        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        
        # Get a batch
        for batch in loader:
            break
        
        # Create a device
        device = torch.device("cpu")
        batch = batch.to(device)
        model = model.to(device)
        
        # Test frame processing functions
        for frame_averaging in ["2D", "3D"]:
            fa_method = "all"
            print(f"Testing process_frames with {frame_averaging} frame averaging...")
            
            # Apply frame averaging to batch
            batch_fa = apply_frame_averaging_to_batch(batch.clone(), fa_method, frame_averaging)
            
            # Process frames
            frame_outputs = process_frames(model, batch_fa, frame_averaging, fa_method)
            
            # Check outputs
            self.assertIsInstance(frame_outputs, dict, "Frame outputs should be a dictionary")
            
            # Check that we have outputs for each property
            for prop in model.target_properties:
                self.assertIn(prop, frame_outputs, f"Frame outputs should contain {prop}")
                
                # Check that we have the correct number of frames
                expected_frames = 4 if frame_averaging == "2D" else 8
                self.assertEqual(len(frame_outputs[prop]), expected_frames, 
                                f"{frame_averaging} frame averaging should produce {expected_frames} frames")
                
                # Check that each frame has the correct batch size
                for frame in frame_outputs[prop]:
                    self.assertEqual(frame.size(0), batch.num_graphs, 
                                    "Each frame should have predictions for the entire batch")
            
            print(f"✅ Process frames with {frame_averaging} frame averaging passed!")
            
    def test_consistency_loss(self):
        """Test that consistency loss is correctly calculated."""
        print("\n=== Testing Consistency Loss ===")
        
        # Create test frame predictions
        batch_size = 3
        output_dim = 1
        num_frames = 4
        
        # Case 1: All predictions are the same (perfect consistency)
        consistent_preds = [torch.ones(batch_size, output_dim) for _ in range(num_frames)]
        
        # Case 2: Predictions vary across frames (poor consistency)
        varying_preds = [torch.full((batch_size, output_dim), float(i+1)) for i in range(num_frames)]
        
        # Calculate consistency loss
        consistency_loss_fn = ConsistencyLoss(normalize=True)
        
        # Test both cases
        loss_consistent = consistency_loss_fn(consistent_preds)
        loss_varying = consistency_loss_fn(varying_preds)
        
        # Verify loss values
        self.assertAlmostEqual(loss_consistent.item(), 0.0, places=5, 
                              "Consistent predictions should have zero consistency loss")
        self.assertGreater(loss_varying.item(), 0.0, 
                          "Varying predictions should have positive consistency loss")
        
        # Test backwards compatibility function
        loss_consistent_legacy = compute_consistency_loss(consistent_preds, normalize=True)
        loss_varying_legacy = compute_consistency_loss(varying_preds, normalize=True)
        
        # Verify they give the same results
        self.assertAlmostEqual(loss_consistent.item(), loss_consistent_legacy.item(), places=5, 
                               "Legacy function should match class implementation")
        
        # Test computing frame consistency loss with multiple properties
        frame_outputs = {
            "WF_top": [torch.ones(batch_size, output_dim) for _ in range(num_frames)],
            "WF_bottom": [torch.full((batch_size, output_dim), float(i+1)) for i in range(num_frames)]
        }
        
        target_properties = ["WF_top", "WF_bottom"]
        
        # Calculate total consistency loss
        total_loss = compute_frame_consistency_loss(frame_outputs, target_properties, consistency_loss_fn)
        
        # Verify that the total loss is positive (due to varying predictions in WF_bottom)
        self.assertGreater(total_loss.item(), 0.0, 
                          "Total consistency loss should be positive when one property varies")
        
        print("✅ Consistency loss checks passed!")
    
    def test_prediction_loss(self):
        """Test that prediction loss is correctly calculated."""
        print("\n=== Testing Prediction Loss ===")
        
        # Create test frame predictions
        batch_size = 3
        output_dim = 1
        num_frames = 4
        
        # Create frame outputs dictionary
        frame_outputs = {
            "WF_top": [torch.full((batch_size, output_dim), i+1) for i in range(num_frames)],
            "WF_bottom": [torch.full((batch_size, output_dim), i+2) for i in range(num_frames)]
        }
        
        # Create batch with target properties
        batch = Data()
        batch.WF_top = torch.ones(batch_size)
        batch.WF_bottom = torch.ones(batch_size) * 2
        batch.batch = torch.zeros(batch_size, dtype=torch.long)  # Dummy batch vector
        
        # Create loss function
        criterion = torch.nn.MSELoss()
        
        # Calculate prediction loss
        total_loss = compute_prediction_loss(frame_outputs, batch, ["WF_top", "WF_bottom"], criterion)
        
        # Verify loss value (should depend on MSE between average predictions and targets)
        self.assertGreater(total_loss.item(), 0.0, "Prediction loss should be positive")
        
        # Create batch with only one property
        batch_single = Data()
        batch_single.WF_top = torch.ones(batch_size)
        batch_single.batch = torch.zeros(batch_size, dtype=torch.long)
        
        # Calculate loss for single property
        single_loss = compute_prediction_loss(frame_outputs, batch_single, ["WF_top"], criterion)
        
        # Verify loss is still calculated correctly
        self.assertGreater(single_loss.item(), 0.0, "Single property prediction loss should be positive")
        
        print("✅ Prediction loss checks passed!")
        
    def test_end_to_end_training(self):
        """Test end-to-end training with frame averaging and consistency loss."""
        if not self.test_data_exists:
            self.skipTest("Test data file not found")
            
        print("\n=== Testing End-to-End Training With Frame Averaging ===")
        
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
            batch_size=2,
            epochs=1,  # Just one epoch for basic testing
            learning_rate=0.001,
            frame_averaging="2D",  # Enable frame averaging
            fa_method="all",
            
            # Data parameters
            data_path=TEST_DATA_PATH,
            structure_col="slab",
            target_properties="WF",  # This should map to ["WF_top", "WF_bottom"]
            
            # Consistency loss parameters
            consistency_loss=True,
            consistency_weight=0.1,
            consistency_norm=True,
            
            # MLflow settings
            use_mlflow=False,
            
            # Other settings
            output_dir=str(self.output_dir),
            device="cpu"
        )
        
        # Run training
        try:
            print("Starting end-to-end training with frame averaging and consistency loss...")
            model, test_loader = train_faenet(**config.model_dump())
            
            # Verify the model was saved
            best_model_path = os.path.join(config.output_dir, "best_model.pt")
            self.assertTrue(os.path.exists(best_model_path), f"Best model was not saved to {best_model_path}")
            
            # Check that predictions file was created
            predictions_path = os.path.join(config.output_dir, "test_predictions.json")
            self.assertTrue(os.path.exists(predictions_path), f"Predictions file was not saved to {predictions_path}")
            
            # Check that model has correct target properties
            self.assertListEqual(model.target_properties, ["WF_top", "WF_bottom"], 
                               "Model should have WF_top and WF_bottom as target properties")
            
            # Check that we have predictions for train/val/test data
            train_preds_path = os.path.join(config.output_dir, "train_predictions.json")
            val_preds_path = os.path.join(config.output_dir, "val_predictions.json")
            self.assertTrue(os.path.exists(train_preds_path), f"Train predictions file not found at {train_preds_path}")
            self.assertTrue(os.path.exists(val_preds_path), f"Val predictions file not found at {val_preds_path}")
            
            # Check combined predictions file
            combined_preds_path = os.path.join(config.output_dir, "all_predictions.json")
            self.assertTrue(os.path.exists(combined_preds_path), f"Combined predictions file not found at {combined_preds_path}")
            
            # Verify model predictions
            # Load a batch from the test loader
            for batch in test_loader:
                break
            
            # Move to device and apply frame averaging
            batch = batch.to(torch.device("cpu"))
            batch_fa = apply_frame_averaging_to_batch(batch, "all", "2D")
            
            # Get model predictions
            model.eval()
            with torch.no_grad():
                # Process frames
                frame_outputs = process_frames(model, batch_fa, "2D", "all")
                
                # Check frame outputs
                for prop in model.target_properties:
                    self.assertIn(prop, frame_outputs, f"Frame outputs should contain {prop}")
                    self.assertEqual(len(frame_outputs[prop]), 4, "2D frame averaging should produce 4 frames")
                
                # Calculate consistency loss
                consistency_loss_fn = ConsistencyLoss(normalize=True)
                consistency_loss = compute_frame_consistency_loss(frame_outputs, model.target_properties, consistency_loss_fn)
                
                # Verify consistency loss
                self.assertGreater(consistency_loss.item(), 0.0, "Consistency loss should be positive")
                
                # Calculate prediction loss
                criterion = torch.nn.MSELoss()
                pred_loss = compute_prediction_loss(frame_outputs, batch, model.target_properties, criterion)
                
                # Verify prediction loss
                self.assertGreater(pred_loss.item(), 0.0, "Prediction loss should be positive")
            
            print("✅ End-to-end training with frame averaging passed!")
            
        except Exception as e:
            self.fail(f"End-to-end training failed with error: {str(e)}")

    def test_frame_averaging_count(self):
        """Test that the correct number of frames are generated."""
        print("\n=== Testing Frame Count ===")
        
        # Create test data
        num_atoms = 10
        torch.manual_seed(42)
        test_pos = torch.randn(num_atoms, 3)
        
        # Function to count frames for different methods
        def count_frames(fa_func, fa_methods, pos):
            counts = {}
            for method in fa_methods:
                fa_pos, _, _ = fa_func(pos, None, method)
                counts[method] = len(fa_pos)
            return counts
        
        # Test 2D frame averaging
        fa_methods_2d = ["all", "det", "random"]
        counts_2d = count_frames(frame_averaging_2D, fa_methods_2d, test_pos)
        
        # Expected counts for 2D
        expected_2d = {"all": 4, "det": 1, "random": 1}
        for method, expected in expected_2d.items():
            self.assertEqual(counts_2d[method], expected, 
                           f"2D frame averaging with {method} should produce {expected} frames, got {counts_2d[method]}")
        
        # Test 3D frame averaging
        fa_methods_3d = ["all", "det", "random"]
        counts_3d = count_frames(frame_averaging_3D, fa_methods_3d, test_pos)
        
        # Expected counts for 3D
        expected_3d = {"all": 8, "det": 1, "random": 1}
        for method, expected in expected_3d.items():
            self.assertEqual(counts_3d[method], expected, 
                           f"3D frame averaging with {method} should produce {expected} frames, got {counts_3d[method]}")
        
        print("✅ Frame count checks passed!")
        print(f"2D frame counts: {counts_2d}")
        print(f"3D frame counts: {counts_3d}")
        

if __name__ == "__main__":
    unittest.main()