#!/usr/bin/env python
"""
Test script for Config functionality.
"""
import os
import sys
import unittest
from pathlib import Path

# Add the parent directory to sys.path to allow importing the faenet module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from faenet.config import Config, SimpleConfig

class TestConfig(unittest.TestCase):
    """Tests for Config functionality"""
    
    def test_config_creation(self):
        """Test creating a Config with default values"""
        config = Config()
        
        # Check default values
        self.assertEqual(config.cutoff, 6.0, "Default cutoff should be 6.0")
        self.assertEqual(config.batch_size, 32, "Default batch_size should be 32")
        self.assertEqual(config.epochs, 100, "Default epochs should be 100")
        
        # Create with custom values
        custom_config = Config(
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
    
    def test_backward_compatibility(self):
        """Test backward compatibility with SimpleConfig alias"""
        # Ensure SimpleConfig is an alias of Config
        config = Config(cutoff=5.0, batch_size=16)
        simple_config = SimpleConfig(cutoff=5.0, batch_size=16)
        
        # Both should have the same attributes and values
        self.assertEqual(config.cutoff, simple_config.cutoff)
        self.assertEqual(config.batch_size, simple_config.batch_size)
        
        # SimpleConfig should be a subclass of Config
        self.assertTrue(isinstance(simple_config, Config))


if __name__ == "__main__":
    unittest.main()