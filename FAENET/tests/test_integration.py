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

from faenet.config import get_config, Config, ModelConfig, DataConfig, TrainingConfig, SimpleConfig
from faenet.graph_construction import structure_dict_to_graph
from faenet.frame_averaging import frame_averaging_3D
from faenet.dataset import EnhancedSlabDataset, create_dataloader

# Use an absolute path to the test data file
TEST_DATA_PATH = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 
    "..", "test_data", "surface_prop_data_set_top_bottom.csv"
))

class TestIntegration(unittest.TestCase):
    """Tests for integration of different components."""
    
    def setUp(self):
        """Set up test environment."""
        print(f"Test data path: {TEST_DATA_PATH}")
        print(f"File exists: {os.path.exists(TEST_DATA_PATH)}")
    
    def test_csv_loading(self):
        """Test loading structures from CSV file."""
        print("\n=== Testing CSV Loading ===")
        
        print(f"Using test data file: {TEST_DATA_PATH}")
        
        # Skip if test data file doesn't exist
        if not os.path.exists(TEST_DATA_PATH):
            self.skipTest(f"Test data file not found: {TEST_DATA_PATH}")
        
        # Prepare test configuration
        config = Config(
            data=DataConfig(
                data_dir=TEST_DATA_PATH,  # Use absolute path
                structure_col="slab",
                target_properties=["WF_top", "WF_bottom", "cleavage_energy"]
            ),
            model=ModelConfig(
                cutoff=6.0,
                max_neighbors=40
            ),
            training=TrainingConfig(
                frame_averaging=None,
                batch_size=2
            )
        )
        
        # Create dataset and check first item
        dataset = EnhancedSlabDataset(
            data_source=config.data.data_dir,
            structure_col=config.data.structure_col,
            target_props=config.data.target_properties,
            cutoff=config.model.cutoff,
            max_neighbors=config.model.max_neighbors
        )
        
        print(f"Dataset loaded with {len(dataset)} structures")
        self.assertGreater(len(dataset), 0, "Dataset should not be empty")
        
        # Test the first structure
        data = dataset[0]
        print(f"First structure: {data.natoms} atoms")
        self.assertGreater(data.natoms.item(), 0, "Structure should have atoms")
        
        # Check that target properties are loaded
        props = [prop for prop in config.data.target_properties if hasattr(data, prop)]
        print(f"Properties: {props}")
        self.assertGreater(len(props), 0, "Structure should have target properties")
        
        # Check graph construction
        print(f"Number of nodes: {data.x.size(0)}")
        print(f"Number of edges: {data.edge_index.size(1)}")
        print(f"Edge features shape: {data.edge_attr.shape}")
        self.assertEqual(data.x.size(0), data.pos.size(0), "Number of nodes should match positions")
    
    def test_frame_averaging(self):
        """Test frame averaging integration."""
        print("\n=== Testing Frame Averaging Integration ===")
        
        # Skip if test data file doesn't exist
        if not os.path.exists(TEST_DATA_PATH):
            self.skipTest(f"Test data file not found: {TEST_DATA_PATH}")
        
        # Prepare test configuration
        config = Config(
            data=DataConfig(
                data_dir=TEST_DATA_PATH,  # Use absolute path
                structure_col="slab",
                target_properties=["WF_top", "WF_bottom", "cleavage_energy"]
            ),
            model=ModelConfig(
                cutoff=6.0,
                max_neighbors=40
            ),
            training=TrainingConfig(
                frame_averaging="3D",
                fa_method="all",
                batch_size=2
            )
        )
        
        # Create dataset with frame averaging
        dataset = EnhancedSlabDataset(
            data_source=config.data.data_dir,
            structure_col=config.data.structure_col,
            target_props=config.data.target_properties,
            cutoff=config.model.cutoff,
            max_neighbors=config.model.max_neighbors,
            frame_averaging=config.training.frame_averaging,
            fa_method=config.training.fa_method
        )
        
        print(f"Dataset loaded with {len(dataset)} structures")
        self.assertGreater(len(dataset), 0, "Dataset should not be empty")
        
        # Test the first structure
        data = dataset[0]
        print(f"First structure: {data.natoms} atoms")
        
        # Check frame averaging results
        self.assertTrue(hasattr(data, 'fa_pos'), "Structure should have fa_pos attribute")
        
        # Check frame properties
        num_frames = len(data.fa_pos)
        print(f"Number of frames: {num_frames}")
        print(f"Original positions shape: {data.pos.shape}")
        print(f"Frame positions shape: {data.fa_pos[0].shape}")
        self.assertGreater(num_frames, 0, "Should have at least one frame")
        self.assertEqual(data.fa_pos[0].shape, data.pos.shape, "Frame positions should match original shape")
        
        # Verify rotation matrices
        self.assertTrue(hasattr(data, 'fa_rot'), "Structure should have fa_rot attribute")
        print(f"Rotation matrices shape: {data.fa_rot[0].shape}")
        self.assertEqual(data.fa_rot[0].shape[1:], (3, 3), "Rotation matrices should be 3x3")
    
    def test_dataloader_creation(self):
        """Test creation of train, val, test dataloaders."""
        print("\n=== Testing Dataloader Creation ===")
        
        # Skip if test data file doesn't exist
        if not os.path.exists(TEST_DATA_PATH):
            self.skipTest(f"Test data file not found: {TEST_DATA_PATH}")
        
        # Prepare test configuration
        config = Config(
            data=DataConfig(
                data_dir=TEST_DATA_PATH,  # Use absolute path
                structure_col="slab",
                target_properties=["WF_top", "WF_bottom", "cleavage_energy"]
            ),
            model=ModelConfig(
                cutoff=6.0,
                max_neighbors=40
            ),
            training=TrainingConfig(
                frame_averaging="3D",
                fa_method="all",
                batch_size=2,
                train_ratio=0.6,
                val_ratio=0.2,
                test_ratio=0.2
            )
        )
        
        # Create dataloaders
        train_loader, val_loader, test_loader, dataset = create_dataloader(
            data_source=config.data.data_dir,
            structure_col=config.data.structure_col,
            target_props=config.data.target_properties,
            cutoff=config.model.cutoff,
            max_neighbors=config.model.max_neighbors,
            frame_averaging=config.training.frame_averaging,
            fa_method=config.training.fa_method,
            batch_size=config.training.batch_size,
            train_ratio=config.training.train_ratio,
            val_ratio=config.training.val_ratio,
            test_ratio=config.training.test_ratio
        )
        
        # Check splits
        total = len(dataset)
        print(f"Dataset size: {total} structures")
        print(f"Train loader: {len(train_loader.dataset)} structures")
        print(f"Val loader: {len(val_loader.dataset)} structures")
        print(f"Test loader: {len(test_loader.dataset)} structures")
        
        # Verify split ratios
        self.assertGreater(total, 0, "Dataset should not be empty")
        self.assertAlmostEqual(len(train_loader.dataset) / total, 0.6, delta=0.1, 
                           msg="Train split should be around 60%")
        self.assertAlmostEqual(len(val_loader.dataset) / total, 0.2, delta=0.1, 
                           msg="Validation split should be around 20%")
        self.assertAlmostEqual(len(test_loader.dataset) / total, 0.2, delta=0.1, 
                           msg="Test split should be around 20%")
        
        # Try loading a batch if dataset is not empty
        if len(train_loader.dataset) > 0:
            for batch in train_loader:
                print(f"Batch size: {batch.num_graphs}")
                props = [prop for prop in config.data.target_properties if hasattr(batch, prop)]
                print(f"Batch properties: {props}")
                print(f"Has frame averaging: {hasattr(batch, 'fa_pos')}")
                if hasattr(batch, 'fa_pos'):
                    print(f"Number of frames: {len(batch.fa_pos)}")
                self.assertGreater(batch.num_graphs, 0, "Batch should contain at least one graph")
                break
    
    def test_forward_with_frames(self):
        """Test frame averaging mathematical properties."""
        print("\n=== Testing Frame Averaging Correctness ===")
        
        try:
            # Create a simple test structure (5 atoms with random 3D positions)
            num_atoms = 5
            test_pos = torch.randn(num_atoms, 3)
            
            # Test 3D frame averaging (should produce 8 frames)
            print("Testing 3D frame averaging...")
            fa_pos_3d, _, fa_rot_3d = frame_averaging_3D(test_pos, None, "all")
            num_frames_3d = len(fa_pos_3d)
            print(f"Number of frames (3D): {num_frames_3d}")
            self.assertEqual(num_frames_3d, 8, "3D frame averaging should produce 8 frames")
            
            # Test 2D frame averaging (should produce 4 frames)
            print("Testing 2D frame averaging...")
            fa_pos_2d, _, fa_rot_2d = frame_averaging_2D(test_pos, None, "all")
            num_frames_2d = len(fa_pos_2d)
            print(f"Number of frames (2D): {num_frames_2d}")
            self.assertEqual(num_frames_2d, 4, "2D frame averaging should produce 4 frames")
            
            # Check that 2D frame averaging preserves z-axis
            for i in range(num_frames_2d):
                self.assertTrue(torch.allclose(fa_pos_2d[i][:, 2], test_pos[:, 2], atol=1e-5),
                            f"Frame {i} does not preserve z-coordinates")
            
            # Check determinants of rotation matrices
            for i in range(num_frames_2d):
                R = fa_rot_2d[i].squeeze()
                det = torch.det(R)
                print(f"  Frame {i} has determinant {det:.1f} (proper rotation if 1, improper if -1)")
                self.assertTrue(torch.isclose(torch.abs(det), torch.tensor(1.0), atol=1e-5),
                            f"Rotation {i} has incorrect determinant")
            
            print("✅ 2D frame averaging correctly preserves z-axis coordinates")
            
        except Exception as e:
            self.fail(f"Error in test_forward_with_frames: {e}")
    
    def test_simple_config(self):
        """Test SimpleConfig and conversion to nested Config."""
        print("\n=== Testing SimpleConfig ===")
        
        # Create a SimpleConfig
        simple_config = SimpleConfig(
            cutoff=5.0,
            max_neighbors=30,
            hidden_channels=64,
            frame_averaging="2D",
            data_dir=TEST_DATA_PATH,  # Use absolute path
            structure_col="slab",
            target_properties=["WF_top", "WF_bottom"],
            batch_size=4
        )
        
        print(f"Simple config created: cutoff={simple_config.cutoff}, frame_averaging={simple_config.frame_averaging}")
        
        # Convert to nested config
        nested_config = simple_config.to_nested_config()
        
        print(f"Converted to nested config: cutoff={nested_config.model.cutoff}, frame_averaging={nested_config.training.frame_averaging}")
        
        # Verify conversion was correct
        self.assertEqual(simple_config.cutoff, nested_config.model.cutoff, "cutoff conversion failed")
        self.assertEqual(simple_config.hidden_channels, nested_config.model.hidden_channels, "hidden_channels conversion failed")
        self.assertEqual(simple_config.frame_averaging, nested_config.training.frame_averaging, "frame_averaging conversion failed")
        self.assertEqual(simple_config.data_dir, nested_config.data.data_dir, "data_dir conversion failed")
        self.assertEqual(simple_config.target_properties, nested_config.data.target_properties, "target_properties conversion failed")
        
        print("SimpleConfig conversion test passed!")


if __name__ == "__main__":
    unittest.main()