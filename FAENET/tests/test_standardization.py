"""
Tests for standardization of regression targets.
"""
import pytest
import torch
import numpy as np
import os
import tempfile
import pickle

from faenet.dataset import Standardizer

def test_standardizer_init():
    """Test standardizer initialization."""
    # Initialize with a list of property names
    props = ["energy", "force"]
    standardizer = Standardizer(props)
    
    # Check that properties are stored
    assert standardizer.property_names == props
    
    # Check that it's not fitted yet
    assert not standardizer.is_fitted
    assert len(standardizer.mean) == 0
    assert len(standardizer.std) == 0

def test_standardizer_fit():
    """Test standardizer fitting."""
    # Sample data
    data = {
        "energy": [1.0, 2.0, 3.0, 4.0, 5.0],
        "force": [10.0, 20.0, 30.0, 40.0, 50.0]
    }
    
    # Initialize standardizer
    standardizer = Standardizer(["energy", "force"])
    
    # Fit the standardizer
    standardizer.fit(data)
    
    # Check that it's fitted
    assert standardizer.is_fitted
    
    # Check mean and std
    assert standardizer.mean["energy"] == 3.0
    assert standardizer.std["energy"] == np.std([1.0, 2.0, 3.0, 4.0, 5.0])
    assert standardizer.mean["force"] == 30.0
    assert standardizer.std["force"] == np.std([10.0, 20.0, 30.0, 40.0, 50.0])

def test_standardizer_transform():
    """Test standardizer transform."""
    # Sample data
    data = {
        "energy": [1.0, 2.0, 3.0, 4.0, 5.0],
        "force": [10.0, 20.0, 30.0, 40.0, 50.0]
    }
    
    # Initialize and fit standardizer
    standardizer = Standardizer(["energy", "force"])
    standardizer.fit(data)
    
    # Transform a value
    energy_std = standardizer.transform(4.0, "energy")
    force_std = standardizer.transform(40.0, "force")
    
    # Check standardized values
    expected_energy = (4.0 - 3.0) / standardizer.std["energy"]
    expected_force = (40.0 - 30.0) / standardizer.std["force"]
    assert energy_std == expected_energy
    assert force_std == expected_force

def test_standardizer_inverse_transform():
    """Test standardizer inverse transform."""
    # Sample data
    data = {
        "energy": [1.0, 2.0, 3.0, 4.0, 5.0],
        "force": [10.0, 20.0, 30.0, 40.0, 50.0]
    }
    
    # Initialize and fit standardizer
    standardizer = Standardizer(["energy", "force"])
    standardizer.fit(data)
    
    # Transform and inverse transform
    energy_std = standardizer.transform(4.0, "energy")
    energy_orig = standardizer.inverse_transform(energy_std, "energy")
    
    force_std = standardizer.transform(40.0, "force")
    force_orig = standardizer.inverse_transform(force_std, "force")
    
    # Check that we get back the original values
    assert np.isclose(energy_orig, 4.0)
    assert np.isclose(force_orig, 40.0)

def test_standardizer_save_load():
    """Test saving and loading standardizer."""
    # Sample data
    data = {
        "energy": [1.0, 2.0, 3.0, 4.0, 5.0],
        "force": [10.0, 20.0, 30.0, 40.0, 50.0]
    }
    
    # Initialize and fit standardizer
    standardizer1 = Standardizer(["energy", "force"])
    standardizer1.fit(data)
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        with open(tmp.name, 'wb') as f:
            pickle.dump(standardizer1.get_params(), f)
        
        # Load into new standardizer
        standardizer2 = Standardizer()
        with open(tmp.name, 'rb') as f:
            params = pickle.load(f)
        standardizer2.load_params(params)
    
    # Clean up
    os.unlink(tmp.name)
    
    # Check that parameters are the same
    assert standardizer2.is_fitted
    assert standardizer2.mean["energy"] == standardizer1.mean["energy"]
    assert standardizer2.std["energy"] == standardizer1.std["energy"]
    assert standardizer2.mean["force"] == standardizer1.mean["force"]
    assert standardizer2.std["force"] == standardizer1.std["force"]
    
    # Check that transformation works
    energy_std1 = standardizer1.transform(4.0, "energy")
    energy_std2 = standardizer2.transform(4.0, "energy")
    assert energy_std1 == energy_std2

def test_zero_std_handling():
    """Test handling of zero standard deviation."""
    # Sample data with zero variance
    data = {
        "energy": [1.0, 1.0, 1.0, 1.0, 1.0],
    }
    
    # Initialize and fit standardizer
    standardizer = Standardizer(["energy"])
    standardizer.fit(data)
    
    # Check that std is set to 1.0 to avoid division by zero
    assert standardizer.std["energy"] == 1.0
    
    # Transform a value
    energy_std = standardizer.transform(1.0, "energy")
    
    # Check standardized value (should be zero due to mean=1.0, std=1.0)
    assert energy_std == 0.0
    
    # Inverse transform
    energy_orig = standardizer.inverse_transform(energy_std, "energy")
    
    # Check that we get back the original value
    assert energy_orig == 1.0