#!/usr/bin/env python
"""
Test script for Config functionality.
This tests the Config class without requiring any actual data files.
"""
from pathlib import Path
from config import Config, SimpleConfig

def test_default_values():
    """Test that default values are set correctly in Config"""
    # Create a Config with minimal parameters
    config = Config()
    
    # Check some default values
    print("Checking default values...")
    assert config.cutoff == 6.0
    assert config.batch_size == 32
    assert config.epochs == 100
    assert config.lr == 0.001
    assert config.frame_averaging is None
    assert config.fa_method == "all"
    assert config.regress_forces is False
    
    print("✅ Default values are set correctly!")
    return True

def test_param_override():
    """Test that parameters can be overridden"""
    # Create a Config with some parameters
    config = Config(
        cutoff=4.0,
        batch_size=16,
        frame_averaging="2D"
    )
    
    # Check overridden values
    print("Checking parameter overrides...")
    assert config.cutoff == 4.0
    assert config.batch_size == 16
    assert config.frame_averaging == "2D"
    assert config.fa_method == "all"  # Default value
    
    print("✅ Parameter overrides work correctly!")
    return True

def test_backward_compatibility():
    """Test backward compatibility with SimpleConfig"""
    # Create both Config and SimpleConfig instances
    config = Config(cutoff=5.0, batch_size=16, lr=0.01)
    simple_config = SimpleConfig(cutoff=5.0, batch_size=16, lr=0.01)
    
    # Verify same values
    print("Checking SimpleConfig compatibility...")
    assert config.cutoff == simple_config.cutoff
    assert config.batch_size == simple_config.batch_size
    assert config.lr == simple_config.lr
    
    # SimpleConfig should be a subclass of Config
    assert isinstance(simple_config, Config)
    
    print("✅ SimpleConfig is compatible with Config!")
    return True

def main():
    """Run all Config tests"""
    print("=== Testing Config Functionality ===")
    
    # Run tests
    test_default_values()
    test_param_override()
    test_backward_compatibility()
    
    print("\n✅ All Config tests passed!")

if __name__ == "__main__":
    main()