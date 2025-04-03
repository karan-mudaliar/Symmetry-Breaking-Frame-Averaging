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

from faenet.config import Config

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
    
    def test_advanced_features(self):
        """Test advanced Config features"""
        # Test using different configs with different values
        config1 = Config(cutoff=5.0, batch_size=16)
        config2 = Config(cutoff=6.0, batch_size=32)
        
        # Both should have correctly set attributes and values
        self.assertEqual(config1.cutoff, 5.0)
        self.assertEqual(config1.batch_size, 16)
        
        self.assertEqual(config2.cutoff, 6.0)
        self.assertEqual(config2.batch_size, 32)


if __name__ == "__main__":
    unittest.main()