#!/usr/bin/env python
"""
Test script for comparing the nested Config and SimpleConfig functionality.
This script demonstrates the usage of both config types and ensures they work correctly.
"""
import os
import torch
import argparse
from pathlib import Path

from config import Config, SimpleConfig, ModelConfig, TrainingConfig, DataConfig

def test_nested_config():
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
            data_dir=Path("./test_data/surface_prop_data_set_top_bottom.csv"),
            structure_col="slab",
            target_properties=["WF_top", "WF_bottom", "cleavage_energy"]
        ),
        output_dir=Path("./nested_output"),
        device="cpu"
    )
    
    # Print configuration
    print(f"Model cutoff: {config.model.cutoff}")
    print(f"Training batch size: {config.training.batch_size}")
    print(f"Frame averaging: {config.training.frame_averaging}")
    print(f"Data directory: {config.data.data_dir}")
    print(f"Target properties: {config.data.target_properties}")
    print(f"Output directory: {config.output_dir}")
    
    return config

def test_simple_config():
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
        data_dir=Path("./test_data/surface_prop_data_set_top_bottom.csv"),
        structure_col="slab",
        target_properties=["WF_top", "WF_bottom", "cleavage_energy"],
        output_dir=Path("./simple_output"),
        device="cpu"
    )
    
    # Print configuration
    print(f"Model cutoff: {config.cutoff}")
    print(f"Training batch size: {config.batch_size}")
    print(f"Frame averaging: {config.frame_averaging}")
    print(f"Data directory: {config.data_dir}")
    print(f"Target properties: {config.target_properties}")
    print(f"Output directory: {config.output_dir}")
    
    return config

def test_conversion():
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
    
    # Print both configs to compare
    print("--- Original SimpleConfig ---")
    print(f"Cutoff: {simple_config.cutoff}")
    print(f"Hidden channels: {simple_config.hidden_channels}")
    print(f"Frame averaging: {simple_config.frame_averaging}")
    print(f"Learning rate: {simple_config.lr}")
    
    print("\n--- Converted Nested Config ---")
    print(f"Cutoff: {nested_config.model.cutoff}")
    print(f"Hidden channels: {nested_config.model.hidden_channels}")
    print(f"Frame averaging: {nested_config.training.frame_averaging}")
    print(f"Learning rate: {nested_config.training.lr}")
    
    # Verify conversion was correct
    assert simple_config.cutoff == nested_config.model.cutoff
    assert simple_config.hidden_channels == nested_config.model.hidden_channels
    assert simple_config.frame_averaging == nested_config.training.frame_averaging
    assert simple_config.lr == nested_config.training.lr
    assert simple_config.data_dir == nested_config.data.data_dir
    assert simple_config.target_properties == nested_config.data.target_properties
    assert simple_config.output_dir == nested_config.output_dir
    assert simple_config.device == nested_config.device
    
    print("✅ All conversion checks passed!")
    return simple_config, nested_config

def simulate_parameters():
    """Simulate command-line parameter handling with argparse"""
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
        print(f"Created SimpleConfig from args: cutoff={config.cutoff}, batch_size={config.batch_size}")
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
        print(f"Created nested Config from args: cutoff={config.model.cutoff}, batch_size={config.training.batch_size}")
    
    return config

def main():
    """Run all config tests"""
    print("=== FAENet Configuration Tests ===")
    
    # Test both config types
    nested_config = test_nested_config()
    simple_config = test_simple_config()
    
    # Test conversion
    converted_simple, converted_nested = test_conversion()
    
    # Test parameter handling
    param_config = simulate_parameters()
    
    print("\n=== All Tests Completed Successfully ===")
    print("You can now use either the nested Config or SimpleConfig in your code!")

if __name__ == "__main__":
    main()