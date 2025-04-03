#!/usr/bin/env python
"""
Test script for Config functionality.
This tests the Config class without requiring any actual data files.
"""
import os
import torch
import argparse
from pathlib import Path

from config import Config, SimpleConfig

def test_default_values():
    """Test that default values are set correctly in Config"""
    print("\n=== Testing Default Values ===")
    
    # Create a Config with minimal parameters
    config = Config()
    
    # Check some default values
    print("Checking default values...")
    print(f"Cutoff: {config.cutoff}")
    print(f"Batch size: {config.batch_size}")
    print(f"Epochs: {config.epochs}")
    print(f"Learning rate: {config.lr}")
    print(f"Frame averaging: {config.frame_averaging}")
    print(f"FA method: {config.fa_method}")
    print(f"Regress forces: {config.regress_forces}")
    
    # Verify values
    assert config.cutoff == 6.0
    assert config.batch_size == 32
    assert config.epochs == 100
    assert config.lr == 0.001
    assert config.frame_averaging is None
    assert config.fa_method == "all"
    assert config.regress_forces is False
    
    print("✅ Default values are set correctly!")
    return config

def test_param_override():
    """Test that parameters can be overridden"""
    print("\n=== Testing Parameter Overrides ===")
    
    # Create a Config with custom parameters
    config = Config(
        cutoff=4.0,
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
        data_dir=Path("./data"),
        structure_col="slab",
        target_properties=["WF_top", "WF_bottom", "cleavage_energy"],
        output_dir=Path("./test_output"),
        device="cpu"
    )
    
    # Print configuration
    print(f"Cutoff: {config.cutoff}")
    print(f"Batch size: {config.batch_size}")
    print(f"Frame averaging: {config.frame_averaging}")
    print(f"Data directory: {config.data_dir}")
    print(f"Target properties: {config.target_properties}")
    print(f"Output directory: {config.output_dir}")
    
    # Check overridden values
    assert config.cutoff == 4.0
    assert config.batch_size == 16
    assert config.frame_averaging == "2D"
    assert config.fa_method == "all"
    assert config.data_dir == Path("./data")
    
    print("✅ Parameter overrides work correctly!")
    return config

def test_backward_compatibility():
    """Test backward compatibility with SimpleConfig"""
    print("\n=== Testing Backward Compatibility ===")
    
    # Create both Config and SimpleConfig instances
    config = Config(cutoff=5.0, batch_size=16, lr=0.01)
    simple_config = SimpleConfig(cutoff=5.0, batch_size=16, lr=0.01)
    
    # Compare the values
    print("Config values: ")
    print(f"Cutoff: {config.cutoff}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.lr}")
    
    print("\nSimpleConfig values: ")
    print(f"Cutoff: {simple_config.cutoff}")
    print(f"Batch size: {simple_config.batch_size}")
    print(f"Learning rate: {simple_config.lr}")
    
    # Verify same values
    assert config.cutoff == simple_config.cutoff
    assert config.batch_size == simple_config.batch_size
    assert config.lr == simple_config.lr
    
    # SimpleConfig should be a subclass of Config
    assert isinstance(simple_config, Config)
    
    print("✅ SimpleConfig is compatible with Config!")
    return config, simple_config

def test_command_line_args():
    """Test command-line parameter handling with argparse"""
    print("\n=== Testing Command-line Parameter Handling ===")
    
    # Simulate command line args
    args = argparse.Namespace(
        cutoff=4.5,
        batch_size=24,
        frame_averaging="2D",
        data_dir="./simulated_data"
    )
    
    # Create config from args
    config = Config(
        cutoff=args.cutoff,
        batch_size=args.batch_size,
        frame_averaging=args.frame_averaging,
        data_dir=Path(args.data_dir)
    )
    
    print(f"Created Config from args: cutoff={config.cutoff}, batch_size={config.batch_size}")
    
    # Verify values
    assert config.cutoff == 4.5
    assert config.batch_size == 24
    assert config.frame_averaging == "2D"
    assert config.data_dir == Path("./simulated_data")
    
    # Convert to command line arguments for use in scripts
    cmd_args = [
        f"--cutoff={config.cutoff}",
        f"--batch_size={config.batch_size}",
        f"--frame_averaging={config.frame_averaging}",
        f"--data_dir={config.data_dir}"
    ]
    print("Command line arguments: " + " ".join(cmd_args))
    
    print("✅ Command line argument handling works correctly!")
    return config

def main():
    """Run all config tests"""
    print("=== FAENet Configuration Tests ===")
    
    # Run all tests
    test_default_values()
    test_param_override()
    test_backward_compatibility()
    test_command_line_args()
    
    print("\n=== All Tests Completed Successfully ===")
    print("The Config class is working correctly!")

if __name__ == "__main__":
    main()