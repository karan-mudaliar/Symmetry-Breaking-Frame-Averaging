#!/usr/bin/env python
"""
Test script specifically for SimpleConfig functionality.
This script tests the SimpleConfig class and its conversion to nested Config
without requiring any actual data files.
"""
from pathlib import Path
from config import SimpleConfig, Config, ModelConfig, TrainingConfig, DataConfig

def test_conversion():
    """Test that SimpleConfig properly converts to nested Config"""
    # Create a SimpleConfig with all parameters
    simple_config = SimpleConfig(
        # Model parameters
        cutoff=5.5,
        max_neighbors=35,
        num_gaussians=50,
        hidden_channels=128,
        num_filters=128,
        num_interactions=4,
        dropout=0.1,
        output_properties=["energy", "forces"],
        regress_forces=True,
        
        # Training parameters
        batch_size=32,
        epochs=100,
        lr=0.001,
        weight_decay=1e-5,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        frame_averaging="3D",
        fa_method="all",
        force_weight=0.1,
        seed=42,
        num_workers=4,
        checkpoint_interval=10,
        eval_interval=5,
        early_stopping_patience=20,
        
        # Data parameters
        data_dir=Path("./data"),
        structure_col="structure",
        target_properties=["energy"],
        prop_files=["energy.json"],
        pbc=True,
        limit=1000,
        
        # General parameters
        output_dir=Path("./outputs"),
        inference_output="predictions.json",
        device="cuda"
    )
    
    # Convert to nested config
    nested_config = simple_config.to_nested_config()
    
    # Check that all parameters are correctly transferred
    print("Validating model parameters...")
    assert simple_config.cutoff == nested_config.model.cutoff
    assert simple_config.max_neighbors == nested_config.model.max_neighbors
    assert simple_config.num_gaussians == nested_config.model.num_gaussians
    assert simple_config.hidden_channels == nested_config.model.hidden_channels
    assert simple_config.num_filters == nested_config.model.num_filters
    assert simple_config.num_interactions == nested_config.model.num_interactions
    assert simple_config.dropout == nested_config.model.dropout
    assert simple_config.output_properties == nested_config.model.output_properties
    assert simple_config.regress_forces == nested_config.model.regress_forces
    
    print("Validating training parameters...")
    assert simple_config.batch_size == nested_config.training.batch_size
    assert simple_config.epochs == nested_config.training.epochs
    assert simple_config.lr == nested_config.training.lr
    assert simple_config.weight_decay == nested_config.training.weight_decay
    assert simple_config.train_ratio == nested_config.training.train_ratio
    assert simple_config.val_ratio == nested_config.training.val_ratio
    assert simple_config.test_ratio == nested_config.training.test_ratio
    assert simple_config.frame_averaging == nested_config.training.frame_averaging
    assert simple_config.fa_method == nested_config.training.fa_method
    assert simple_config.force_weight == nested_config.training.force_weight
    assert simple_config.seed == nested_config.training.seed
    assert simple_config.num_workers == nested_config.training.num_workers
    assert simple_config.checkpoint_interval == nested_config.training.checkpoint_interval
    assert simple_config.eval_interval == nested_config.training.eval_interval
    assert simple_config.early_stopping_patience == nested_config.training.early_stopping_patience
    
    print("Validating data parameters...")
    assert simple_config.data_dir == nested_config.data.data_dir
    assert simple_config.structure_col == nested_config.data.structure_col
    assert simple_config.target_properties == nested_config.data.target_properties
    assert simple_config.prop_files == nested_config.data.prop_files
    assert simple_config.pbc == nested_config.data.pbc
    assert simple_config.limit == nested_config.data.limit
    
    print("Validating general parameters...")
    assert simple_config.output_dir == nested_config.output_dir
    assert simple_config.inference_output == nested_config.inference_output
    assert simple_config.device == nested_config.device
    
    print("✅ All conversions are correct!")
    return True

def test_default_values():
    """Test that default values are set correctly in SimpleConfig"""
    # Create a SimpleConfig with minimal parameters
    simple_config = SimpleConfig()
    
    # Check some default values
    print("Checking default values...")
    assert simple_config.cutoff == 6.0
    assert simple_config.batch_size == 32
    assert simple_config.epochs == 100
    assert simple_config.lr == 0.001
    assert simple_config.frame_averaging is None
    assert simple_config.fa_method == "all"
    assert simple_config.regress_forces is False
    
    print("✅ Default values are set correctly!")
    return True

def test_param_override():
    """Test that parameters can be overridden"""
    # Create a SimpleConfig with some parameters
    simple_config = SimpleConfig(
        cutoff=4.0,
        batch_size=16,
        frame_averaging="2D"
    )
    
    # Check overridden values
    print("Checking parameter overrides...")
    assert simple_config.cutoff == 4.0
    assert simple_config.batch_size == 16
    assert simple_config.frame_averaging == "2D"
    assert simple_config.fa_method == "all"  # Default value
    
    print("✅ Parameter overrides work correctly!")
    return True

def main():
    """Run all SimpleConfig tests"""
    print("=== Testing SimpleConfig Functionality ===")
    
    # Run tests
    test_default_values()
    test_param_override()
    test_conversion()
    
    print("\n✅ All SimpleConfig tests passed!")

if __name__ == "__main__":
    main()