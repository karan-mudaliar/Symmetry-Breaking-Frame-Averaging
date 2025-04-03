#!/usr/bin/env python
"""
Test script to validate the integration of frame averaging with enhanced graph construction.
"""
import os
import sys
import torch
import pandas as pd
import unittest
from pathlib import Path

# Add the parent directory to sys.path to allow importing the faenet module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from faenet.config import get_config, Config
from faenet.graph_construction import structure_dict_to_graph
from faenet.frame_averaging import frame_averaging_3D, frame_averaging_2D
from faenet.dataset import EnhancedSlabDataset, create_dataloader

# Use an absolute path to the test data file
TEST_DATA_PATH = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 
    "..", "test_data", "surface_prop_data_set_top_bottom.csv"
))

class TestIntegration(unittest.TestCase):
    """Test case for FAENet integration."""
    
    def setUp(self):
        """Check if test data exists."""
        if not os.path.exists(TEST_DATA_PATH):
            self.skipTest(f"Test data file not found: {TEST_DATA_PATH}")
        
        # Print test data path for debugging
        print(f"Test data path: {TEST_DATA_PATH}")
    
    def test_csv_loading(self):
        """Test loading structures from CSV file."""
        print("\n=== Testing CSV Loading ===")
        
        # Prepare test configuration
        config = Config(
            data_dir=TEST_DATA_PATH,  # Use absolute path
            structure_col="slab",
            target_properties=["WF_top", "WF_bottom", "cleavage_energy"],
            cutoff=6.0,
            max_neighbors=40,
            frame_averaging=None,
            batch_size=2
        )
        
        # Create dataset and check first item
        dataset = EnhancedSlabDataset(
            data_source=config.data_dir,
            structure_col=config.structure_col,
            target_props=config.target_properties,
            cutoff=config.cutoff,
            max_neighbors=config.max_neighbors
        )
        
        print(f"Dataset loaded with {len(dataset)} structures")
        
        self.assertGreater(len(dataset), 0, "Dataset should have at least one structure")
        
        if len(dataset) > 0:
            data = dataset[0]
            print(f"First structure: {data.natoms} atoms")
            print(f"Properties: {[prop for prop in config.target_properties if hasattr(data, prop)]}")
            
            # Check that the graph was constructed correctly
            self.assertGreater(data.x.size(0), 0, "Data should have nodes")
            self.assertGreater(data.edge_index.size(1), 0, "Data should have edges")
            print(f"Number of nodes: {data.x.size(0)}")
            print(f"Number of edges: {data.edge_index.size(1)}")
            print(f"Edge features shape: {data.edge_attr.shape}")
    
    def test_frame_averaging(self):
        """Test frame averaging integration."""
        print("\n=== Testing Frame Averaging Integration ===")
        
        # Prepare test configuration
        config = Config(
            data_dir=TEST_DATA_PATH,  # Use absolute path
            structure_col="slab",
            target_properties=["WF_top", "WF_bottom", "cleavage_energy"],
            cutoff=6.0,
            max_neighbors=40,
            frame_averaging="3D",
            fa_method="all",
            batch_size=2
        )
        
        # Create dataset with frame averaging
        dataset = EnhancedSlabDataset(
            data_source=config.data_dir,
            structure_col=config.structure_col,
            target_props=config.target_properties,
            cutoff=config.cutoff,
            max_neighbors=config.max_neighbors,
            frame_averaging=config.frame_averaging,
            fa_method=config.fa_method
        )
        
        print(f"Dataset loaded with {len(dataset)} structures")
        
        self.assertGreater(len(dataset), 0, "Dataset should have at least one structure")
        
        if len(dataset) > 0:
            data = dataset[0]
            print(f"First structure: {data.natoms} atoms")
            
            # Check frame averaging results
            self.assertTrue(hasattr(data, 'fa_pos'), "Data should have frame averaged positions")
            
            if hasattr(data, 'fa_pos'):
                num_frames = len(data.fa_pos)
                print(f"Number of frames: {num_frames}")
                print(f"Original positions shape: {data.pos.shape}")
                print(f"Frame positions shape: {data.fa_pos[0].shape}")
                
                # Verify rotation matrices
                if hasattr(data, 'fa_rot'):
                    print(f"Rotation matrices shape: {data.fa_rot[0].shape}")
                
                self.assertGreater(num_frames, 0, "Should have at least one frame")
    
    def test_dataloader_creation(self):
        """Test creation of train, val, test dataloaders."""
        print("\n=== Testing Dataloader Creation ===")
        
        # Prepare test configuration
        config = Config(
            data_dir=TEST_DATA_PATH,  # Use absolute path
            structure_col="slab",
            target_properties=["WF_top", "WF_bottom", "cleavage_energy"],
            cutoff=6.0,
            max_neighbors=40,
            frame_averaging="3D",
            fa_method="all",
            batch_size=2,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2
        )
        
        # Create dataloaders
        train_loader, val_loader, test_loader, dataset = create_dataloader(
            data_source=config.data_dir,
            structure_col=config.structure_col,
            target_props=config.target_properties,
            cutoff=config.cutoff,
            max_neighbors=config.max_neighbors,
            frame_averaging=config.frame_averaging,
            fa_method=config.fa_method,
            batch_size=config.batch_size,
            train_ratio=config.train_ratio,
            val_ratio=config.val_ratio,
            test_ratio=config.test_ratio
        )
        
        # Check splits
        total = len(dataset)
        print(f"Dataset size: {total} structures")
        print(f"Train loader: {len(train_loader.dataset)} structures")
        print(f"Val loader: {len(val_loader.dataset)} structures")
        print(f"Test loader: {len(test_loader.dataset)} structures")
        
        self.assertGreater(len(train_loader.dataset), 0, "Train loader should have data")
        self.assertGreater(len(val_loader.dataset), 0, "Val loader should have data")
        self.assertGreater(len(test_loader.dataset), 0, "Test loader should have data")
        
        # Try loading a batch
        for batch in train_loader:
            print(f"Batch size: {batch.num_graphs}")
            print(f"Batch properties: {[prop for prop in config.target_properties if hasattr(batch, prop)]}")
            print(f"Has frame averaging: {hasattr(batch, 'fa_pos')}")
            if hasattr(batch, 'fa_pos'):
                print(f"Number of frames: {len(batch.fa_pos)}")
            break
    
    def test_frame_averaging_math(self):
        """Test frame averaging mathematical properties."""
        print("\n=== Testing Frame Averaging Correctness ===")
        
        # Create a simple test structure (5 atoms with random 3D positions)
        num_atoms = 5
        test_pos = torch.randn(num_atoms, 3)
        
        # Test 3D frame averaging (should produce 8 frames)
        print("Testing 3D frame averaging...")
        fa_pos_3d, _, fa_rot_3d = frame_averaging_3D(test_pos, None, "all")
        num_frames_3d = len(fa_pos_3d)
        print(f"Number of frames (3D): {num_frames_3d}")
        
        # Verify we get expected frames in 3D mode
        self.assertEqual(num_frames_3d, 8, "3D frame averaging should produce 8 frames")
        
        # Test 2D frame averaging (should produce 4 frames)
        print("Testing 2D frame averaging...")
        fa_pos_2d, _, fa_rot_2d = frame_averaging_2D(test_pos, None, "all")
        num_frames_2d = len(fa_pos_2d)
        print(f"Number of frames (2D): {num_frames_2d}")
        
        # Verify we get expected frames in 2D mode
        self.assertEqual(num_frames_2d, 4, "2D frame averaging should produce 4 frames")
        
        # Check that 2D frame averaging preserves z-axis
        z_preserved = True
        for i in range(num_frames_2d):
            # For 2D frame averaging, z coordinates should be preserved
            # since rotations are around z-axis only
            if not torch.allclose(fa_pos_2d[i][:, 2], test_pos[:, 2], atol=1e-5):
                z_preserved = False
                print(f"WARNING: Frame {i} does not preserve z-coordinates")
        
        self.assertTrue(z_preserved, "2D frame averaging should preserve z-axis coordinates")
    
    def test_config_instances(self):
        """Test multiple Config instances."""
        print("\n=== Testing Multiple Config Instances ===")
        
        # Create different config instances
        config1 = Config(
            cutoff=5.0,
            max_neighbors=30,
            hidden_channels=64,
            frame_averaging="2D",
            data_dir=TEST_DATA_PATH,
            structure_col="slab",
            target_properties=["WF_top", "WF_bottom"]
        )
        
        config2 = Config(
            cutoff=6.0,
            max_neighbors=40,
            hidden_channels=128,
            frame_averaging="3D",
            data_dir=TEST_DATA_PATH,
            structure_col="slab",
            target_properties=["WF_top", "WF_bottom", "cleavage_energy"]
        )
        
        # Verify each config has correct values
        self.assertEqual(config1.cutoff, 5.0)
        self.assertEqual(config1.hidden_channels, 64)
        self.assertEqual(config1.frame_averaging, "2D")
        
        self.assertEqual(config2.cutoff, 6.0)
        self.assertEqual(config2.hidden_channels, 128)
        self.assertEqual(config2.frame_averaging, "3D")
        
        print("Multiple Config instances work correctly!")


if __name__ == "__main__":
    unittest.main()