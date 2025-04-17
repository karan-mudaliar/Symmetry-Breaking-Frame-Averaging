#!/usr/bin/env python
"""
Test script for the ConsistencyLoss implementation.
"""
import os
import sys
import torch
import unittest
import structlog
from torch_geometric.data import Batch

# Configure logging
logger = structlog.get_logger()

# Add the parent directory to sys.path to allow importing the faenet module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from faenet.frame_averaging import ConsistencyLoss
from faenet.dataset import SlabDataset, apply_frame_averaging_to_batch
from faenet.faenet import FAENet

# Use an absolute path to the test data file
TEST_DATA_PATH = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 
    "..", "test_data", "surface_prop_data_set_top_bottom.csv"
))

class ConsistencyLossTest(unittest.TestCase):
    """Tests for the ConsistencyLoss implementation."""
    
    def test_loss_calculation(self):
        """Test that the ConsistencyLoss calculates variance correctly."""
        # Create a consistency loss module
        consistency_loss = ConsistencyLoss(normalize=False)
        
        # Create test frames with varying predictions
        # 4 frames, 2 samples, 1 output dimension
        frame_preds = [
            torch.tensor([[1.0], [2.0]], requires_grad=True),
            torch.tensor([[1.0], [3.0]], requires_grad=True),
            torch.tensor([[1.0], [4.0]], requires_grad=True),
            torch.tensor([[1.0], [5.0]], requires_grad=True)
        ]
        
        # Calculate consistency loss
        loss = consistency_loss(frame_preds)
        
        # The first sample has constant predictions (variance=0)
        # The second sample has predictions [2,3,4,5] with variance=1.25
        # The mean variance should be (0 + 1.25)/2 = 0.625
        expected_value = 1.25/2
        self.assertAlmostEqual(loss.item(), expected_value, places=5, 
                             msg="Consistency loss should calculate average variance correctly")
        
    def test_normalization(self):
        """Test that normalization works correctly."""
        # Test with normalization enabled vs disabled
        loss_normalized = ConsistencyLoss(normalize=True)
        loss_no_norm = ConsistencyLoss(normalize=False)
        
        # Create frames with large values to see normalization effect
        large_frames = [
            torch.tensor([[10.0], [100.0]], requires_grad=True),
            torch.tensor([[10.0], [200.0]], requires_grad=True)
        ]
        
        # The variance is unchanged for the first sample
        # For the second sample, variance = 2500
        # Without normalization: (0 + 2500)/2 = 1250
        # With normalization: 2500 / (100^2 + 200^2)/2 = 2500 / 22500 ≈ 0.11
        
        # Calculate both losses
        normalized = loss_normalized(large_frames)
        unnormalized = loss_no_norm(large_frames)
        
        # Verify normalization reduces the value 
        self.assertLess(normalized.item(), unnormalized.item(),
                      msg="Normalized loss should be smaller than unnormalized for large values")
        
    def test_backpropagation(self):
        """Test that the loss supports backpropagation."""
        consistency_loss = ConsistencyLoss(normalize=False)
        
        # Create frames with gradients
        frames = [
            torch.tensor([[1.0], [2.0]], requires_grad=True),
            torch.tensor([[2.0], [4.0]], requires_grad=True)
        ]
        
        # Calculate loss
        loss = consistency_loss(frames)
        
        # Should be a tensor with gradient
        self.assertTrue(torch.is_tensor(loss))
        self.assertTrue(loss.requires_grad)
        
        # Backward pass
        loss.backward()
        
        # All tensors should have gradients
        for frame in frames:
            self.assertIsNotNone(frame.grad)
            
    def test_multi_property_combination(self):
        """Test combining losses from multiple properties."""
        consistency_loss = ConsistencyLoss(normalize=False)
        
        # Create frames for two properties
        prop1_frames = [
            torch.tensor([[1.0]], requires_grad=True),
            torch.tensor([[2.0]], requires_grad=True)
        ]
        
        prop2_frames = [
            torch.tensor([[3.0]], requires_grad=True),
            torch.tensor([[6.0]], requires_grad=True)
        ]
        
        # Calculate losses separately
        prop1_loss = consistency_loss(prop1_frames)  # Variance = 0.25
        prop2_loss = consistency_loss(prop2_frames)  # Variance = 2.25
        
        # Combined loss
        combined_loss = prop1_loss + prop2_loss  # Should be 0.25 + 2.25 = 2.5
        
        # Verify value
        self.assertAlmostEqual(combined_loss.item(), 2.5, places=5, 
                            msg="Combined loss should be sum of individual property losses")
        
        # Should support backpropagation
        combined_loss.backward()
        
        # All tensors should have gradients
        for frame in prop1_frames + prop2_frames:
            self.assertIsNotNone(frame.grad)


    def test_with_real_data(self):
        """Test consistency loss with real data from the test dataset."""
        # Check if test data exists
        if not os.path.exists(TEST_DATA_PATH):
            self.skipTest("Test data file not found")
        
        print("\n=== Testing Consistency Loss with Real Data ===\n")
        
        try:
            # Load the test dataset
            dataset = SlabDataset(
                data_source=TEST_DATA_PATH,
                structure_col="slab",
                target_props=["WF_top", "WF_bottom"],
                cutoff=6.0,
                max_neighbors=20,
                pbc=True
            )
            print(f"Loaded test dataset with {len(dataset)} samples")
            
            # Take first two samples to create a batch
            data_samples = [dataset[0], dataset[1]]
            batch = Batch.from_data_list(data_samples)
            device = torch.device("cpu")
            batch = batch.to(device)
            
            # Apply frame averaging (2D)
            frame_averaging = "2D"
            fa_method = "all"
            batch = apply_frame_averaging_to_batch(batch, fa_method, frame_averaging)
            
            # Create a simple model for testing
            model = FAENet(
                cutoff=6.0,
                num_gaussians=10,
                hidden_channels=32,
                num_filters=32,
                num_interactions=1,
                output_properties=["WF_top", "WF_bottom"]
            ).to(device)
            model.train()  # Set to training mode
            
            # Process each frame separately
            all_frames = []
            for i in range(len(batch.fa_pos)):
                # Create frame batch
                frame_batch = batch.clone()
                frame_batch.pos = batch.fa_pos[i]
                all_frames.append(frame_batch)
            
            # Create combined batch with all frames
            combined_batch = Batch.from_data_list(all_frames)
            
            # Forward pass
            outputs = model(combined_batch)
            
            # Organize by property and frame
            frame_outputs = {}
            batch_size = batch.num_graphs
            num_frames = len(batch.fa_pos)
            
            for prop in model.output_properties:
                frame_outputs[prop] = []
                
                # Get predictions for this property
                prop_preds = outputs[prop]
                
                # Split by frame
                for i in range(num_frames):
                    start_idx = i * batch_size
                    end_idx = (i + 1) * batch_size
                    frame_outputs[prop].append(prop_preds[start_idx:end_idx])
            
            # Create consistency loss module
            consistency_loss = ConsistencyLoss(normalize=True)
            
            # Calculate loss for each property
            for prop in model.output_properties:
                # Get frame predictions for this property
                frames = frame_outputs[prop]
                
                # Calculate consistency loss
                loss = consistency_loss(frames)
                
                # Should be a tensor
                self.assertTrue(torch.is_tensor(loss))
                self.assertTrue(loss.requires_grad)
                
                # Test backward pass
                loss.backward(retain_graph=True)
                
                print(f"Consistency loss for {prop}: {loss.item()}")
            
            # Test combined loss for all properties
            all_losses = []
            
            # Reset gradients
            model.zero_grad()
            
            for prop in model.output_properties:
                loss = consistency_loss(frame_outputs[prop])
                all_losses.append(loss)
            
            # Combined loss
            combined_loss = sum(all_losses)
            
            # Backward pass
            combined_loss.backward()
            
            # Verify parameter gradients
            has_grad = False
            for param in model.parameters():
                if param.grad is not None and torch.sum(torch.abs(param.grad)) > 0:
                    has_grad = True
                    break
            
            self.assertTrue(has_grad, "Model parameters should have gradients")
            
            print("✅ Consistency loss test with real data passed!")
            
        except Exception as e:
            import traceback
            self.fail(f"Test with real data failed: {e}\n{traceback.format_exc()}")


if __name__ == "__main__":
    unittest.main()