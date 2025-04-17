#!/usr/bin/env python
"""
Test script for the ConsistencyLoss implementation.
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

from faenet.frame_averaging import ConsistencyLoss

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


if __name__ == "__main__":
    unittest.main()