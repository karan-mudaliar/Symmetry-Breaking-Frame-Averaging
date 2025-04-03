#!/usr/bin/env python
"""
Test script for SimpleConfig functionality.
"""
import os
import sys
import unittest
from pathlib import Path

# Add the parent directory to sys.path to allow importing the faenet module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from faenet.config import SimpleConfig, Config, ModelConfig, TrainingConfig, DataConfig

class TestSimpleConfig(unittest.TestCase):
    """Tests for SimpleConfig functionality"""
    
    def test_simple_config_creation(self):
        """Test creating a SimpleConfig with default values"""
        config = SimpleConfig()
        
        # Check default values
        self.assertEqual(config.cutoff, 6.0, "Default cutoff should be 6.0")
        self.assertEqual(config.batch_size, 32, "Default batch_size should be 32")
        self.assertEqual(config.epochs, 100, "Default epochs should be 100")
        
        # Create with custom values
        custom_config = SimpleConfig(
            cutoff=5.0,
            batch_size=16,
            lr=0.0001,
            frame_averaging="2D"
        )
        
        # Check custom values
        self.assertEqual(custom_config.cutoff, 5.0, "Custom cutoff should be 5.0")
        self.assertEqual(custom_config.batch_size, 16, "Custom batch_size should be 16")
        self.assertEqual(custom_config.lr, 0.0001, "Custom lr should be 0.0001")
        self.assertEqual(custom_config.frame_averaging, "2D", "Custom frame_averaging should be 2D")
    
    def test_config_conversion(self):
        """Test converting between SimpleConfig and nested Config"""
        # Create a simple config
        simple_config = SimpleConfig(
            cutoff=5.0,
            max_neighbors=30,
            hidden_channels=64,
            output_properties=["WF_top", "WF_bottom"],
            batch_size=16,
            epochs=50,
            lr=0.0005,
            frame_averaging="3D",
            data_dir=Path("./test_data"),
            structure_col="structure"
        )
        
        # Convert to nested config
        nested_config = simple_config.to_nested_config()
        
        # Check conversion
        self.assertEqual(simple_config.cutoff, nested_config.model.cutoff, "cutoff conversion failed")
        self.assertEqual(simple_config.max_neighbors, nested_config.model.max_neighbors, "max_neighbors conversion failed")
        self.assertEqual(simple_config.hidden_channels, nested_config.model.hidden_channels, "hidden_channels conversion failed")
        self.assertEqual(simple_config.output_properties, nested_config.model.output_properties, "output_properties conversion failed")
        self.assertEqual(simple_config.batch_size, nested_config.training.batch_size, "batch_size conversion failed")
        self.assertEqual(simple_config.epochs, nested_config.training.epochs, "epochs conversion failed")
        self.assertEqual(simple_config.lr, nested_config.training.lr, "lr conversion failed")
        self.assertEqual(simple_config.frame_averaging, nested_config.training.frame_averaging, "frame_averaging conversion failed")
        self.assertEqual(simple_config.data_dir, nested_config.data.data_dir, "data_dir conversion failed")
        self.assertEqual(simple_config.structure_col, nested_config.data.structure_col, "structure_col conversion failed")
    
    def test_nested_to_simple(self):
        """Test converting from nested Config to SimpleConfig"""
        # Create a nested config
        nested_config = Config(
            model=ModelConfig(
                cutoff=5.0,
                max_neighbors=30,
                hidden_channels=64,
                output_properties=["WF_top"]
            ),
            training=TrainingConfig(
                batch_size=16,
                epochs=50,
                lr=0.0005,
                frame_averaging="3D"
            ),
            data=DataConfig(
                data_dir=Path("./test_data"),
                structure_col="structure"
            )
        )
        
        # Convert to simple config
        simple_config = SimpleConfig.from_nested_config(nested_config)
        
        # Check conversion
        self.assertEqual(nested_config.model.cutoff, simple_config.cutoff, "cutoff conversion failed")
        self.assertEqual(nested_config.model.max_neighbors, simple_config.max_neighbors, "max_neighbors conversion failed")
        self.assertEqual(nested_config.model.hidden_channels, simple_config.hidden_channels, "hidden_channels conversion failed")
        self.assertEqual(nested_config.model.output_properties, simple_config.output_properties, "output_properties conversion failed")
        self.assertEqual(nested_config.training.batch_size, simple_config.batch_size, "batch_size conversion failed")
        self.assertEqual(nested_config.training.epochs, simple_config.epochs, "epochs conversion failed")
        self.assertEqual(nested_config.training.lr, simple_config.lr, "lr conversion failed")
        self.assertEqual(nested_config.training.frame_averaging, simple_config.frame_averaging, "frame_averaging conversion failed")
        self.assertEqual(nested_config.data.data_dir, simple_config.data_dir, "data_dir conversion failed")
        self.assertEqual(nested_config.data.structure_col, simple_config.structure_col, "structure_col conversion failed")


if __name__ == "__main__":
    unittest.main()