#!/usr/bin/env python
"""
Test script for comparing the nested Config and SimpleConfig functionality.
This script demonstrates the usage of both config types and ensures they work correctly.
"""
import os
import torch
import argparse
from pathlib import Path
import sys
import unittest

# Add the parent directory to sys.path to allow importing the faenet module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from faenet.config import Config, SimpleConfig, ModelConfig, TrainingConfig, DataConfig

class TestConfig(unittest.TestCase):
    """Test case for configuration classes."""
    
    def test_nested_config(self):
        """Test the original nested Config class"""
        print("\n=== Testing Nested Config ===")
        
        # Create a nested config
        config = Config(
            model=ModelConfig(
                cutoff=5.0,
                max_neighbors=30,
                hidden_channels=64,
                num_filters=64,
                num_interactions=3,
                output_properties=["WF_top", "WF_bottom"]
            ),
            training=TrainingConfig(
                batch_size=16,
                epochs=50,
                lr=0.0005,
                frame_averaging="2D",
                fa_method="all"
            ),
            data=DataConfig(
                data_dir=Path("./data"),  # Use a generic path to avoid file not found
                structure_col="slab",
                target_properties=["WF_top", "WF_bottom", "cleavage_energy"]
            ),
            output_dir=Path("./nested_output"),
            device="cpu"
        )
        
        # Verify configuration
        self.assertEqual(config.model.cutoff, 5.0)
        self.assertEqual(config.training.batch_size, 16)
        self.assertEqual(config.training.frame_averaging, "2D")
        self.assertEqual(config.data.data_dir, Path("./data"))
        self.assertEqual(config.output_dir, Path("./nested_output"))
        
        return config
    
    def test_simple_config(self):
        """Test the simplified flat SimpleConfig class"""
        print("\n=== Testing Simple Config ===")
        
        # Create a simple config
        config = SimpleConfig(
            cutoff=5.0,
            max_neighbors=30,
            hidden_channels=64,
            num_filters=64,
            num_interactions=3,
            output_properties=["WF_top", "WF_bottom"],
            batch_size=16,
            epochs=50,
            lr=0.0005,
            frame_averaging="2D",
            fa_method="all",
            data_dir=Path("./data"),  # Use a generic path to avoid file not found
            structure_col="slab",
            target_properties=["WF_top", "WF_bottom", "cleavage_energy"],
            output_dir=Path("./simple_output"),
            device="cpu"
        )
        
        # Verify configuration
        self.assertEqual(config.cutoff, 5.0)
        self.assertEqual(config.batch_size, 16)
        self.assertEqual(config.frame_averaging, "2D")
        self.assertEqual(config.data_dir, Path("./data"))
        self.assertEqual(config.output_dir, Path("./simple_output"))
        
        return config
    
    def test_conversion(self):
        """Test conversion between SimpleConfig and nested Config"""
        print("\n=== Testing Config Conversion ===")
        
        # Create a simple config
        simple_config = SimpleConfig(
            cutoff=5.5,
            max_neighbors=35,
            hidden_channels=96,
            num_filters=96,
            num_interactions=4,
            output_properties=["WF_top"],
            batch_size=8,
            epochs=100,
            lr=0.001,
            frame_averaging="3D",
            fa_method="all",
            data_dir=Path("./custom_data"),
            structure_col="structure",
            target_properties=["WF_top"],
            output_dir=Path("./converted_output"),
            device="cuda"
        )
        
        # Convert to nested config
        nested_config = simple_config.to_nested_config()
        
        # Verify conversion was correct
        self.assertEqual(simple_config.cutoff, nested_config.model.cutoff)
        self.assertEqual(simple_config.hidden_channels, nested_config.model.hidden_channels)
        self.assertEqual(simple_config.frame_averaging, nested_config.training.frame_averaging)
        self.assertEqual(simple_config.lr, nested_config.training.lr)
        self.assertEqual(simple_config.data_dir, nested_config.data.data_dir)
        self.assertEqual(simple_config.target_properties, nested_config.data.target_properties)
        self.assertEqual(simple_config.output_dir, nested_config.output_dir)
        self.assertEqual(simple_config.device, nested_config.device)
        
        return simple_config, nested_config
    
    def test_parameter_handling(self):
        """Simulate command-line parameter handling"""
        print("\n=== Testing Command-line Parameter Handling ===")
        
        # Simulate command line args
        args = argparse.Namespace(
            simple=True,
            cutoff=4.5,
            batch_size=24,
            frame_averaging="2D",
            data_dir="./simulated_data"
        )
        
        # Create appropriate config based on --simple flag
        if args.simple:
            # Use SimpleConfig for flat structure
            config = SimpleConfig(
                cutoff=args.cutoff,
                batch_size=args.batch_size,
                frame_averaging=args.frame_averaging,
                data_dir=Path(args.data_dir)
            )
            self.assertEqual(config.cutoff, 4.5)
            self.assertEqual(config.batch_size, 24)
        else:
            # Use nested Config
            config = Config(
                model=ModelConfig(cutoff=args.cutoff),
                training=TrainingConfig(
                    batch_size=args.batch_size,
                    frame_averaging=args.frame_averaging
                ),
                data=DataConfig(data_dir=Path(args.data_dir))
            )
            self.assertEqual(config.model.cutoff, 4.5)
            self.assertEqual(config.training.batch_size, 24)
        
        return config


if __name__ == "__main__":
    unittest.main()