"""
Test script to validate the integration of frame averaging with enhanced graph construction.
"""
import os
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from config import get_config, Config, ModelConfig, DataConfig, TrainingConfig, SimpleConfig

from graph_construction import structure_dict_to_graph
from frame_averaging import frame_averaging_3D
from dataset import EnhancedSlabDataset, create_dataloader

# Use an absolute path to the test data file
TEST_DATA_PATH = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 
    "..", "test_data", "surface_prop_data_set_top_bottom.csv"
))

# Print test data path for debugging
print(f"Test data path: {TEST_DATA_PATH}")
print(f"File exists: {os.path.exists(TEST_DATA_PATH)}")

def test_csv_loading():
    """Test loading structures from CSV file."""
    print("\n=== Testing CSV Loading ===")
    
    print(f"Using test data file: {TEST_DATA_PATH}")
    
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
    
    if len(dataset) > 0:
        data = dataset[0]
        print(f"First structure: {data.natoms} atoms")
        print(f"Properties: {[prop for prop in config.data.target_properties if hasattr(data, prop)]}")
        
        # Check that the graph was constructed correctly
        print(f"Number of nodes: {data.x.size(0)}")
        print(f"Number of edges: {data.edge_index.size(1)}")
        print(f"Edge features shape: {data.edge_attr.shape}")
        
        return True
    else:
        print("No structures were loaded!")
        return False

def test_frame_averaging():
    """Test frame averaging integration."""
    print("\n=== Testing Frame Averaging Integration ===")
    
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
    
    if len(dataset) > 0:
        data = dataset[0]
        print(f"First structure: {data.natoms} atoms")
        
        # Check frame averaging results
        if hasattr(data, 'fa_pos'):
            num_frames = len(data.fa_pos)
            print(f"Number of frames: {num_frames}")
            print(f"Original positions shape: {data.pos.shape}")
            print(f"Frame positions shape: {data.fa_pos[0].shape}")
            
            # Verify rotation matrices
            if hasattr(data, 'fa_rot'):
                print(f"Rotation matrices shape: {data.fa_rot[0].shape}")
            
            return num_frames > 0
        else:
            print("No frame averaging results found!")
            return False
    else:
        print("No structures were loaded!")
        return False

def test_dataloader_creation():
    """Test creation of train, val, test dataloaders."""
    print("\n=== Testing Dataloader Creation ===")
    
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
    
    # Try loading a batch
    for batch in train_loader:
        print(f"Batch size: {batch.num_graphs}")
        print(f"Batch properties: {[prop for prop in config.data.target_properties if hasattr(batch, prop)]}")
        print(f"Has frame averaging: {hasattr(batch, 'fa_pos')}")
        if hasattr(batch, 'fa_pos'):
            print(f"Number of frames: {len(batch.fa_pos)}")
        break
    
    return True

def test_forward_with_frames():
    """Test model forward pass with multiple frames."""
    print("\n=== Testing Forward Pass with Multiple Frames ===")
    
    try:
        from faenet import FAENet
        
        # Create a mini-batch with frame averaging
        dataset = EnhancedSlabDataset(
            data_source=TEST_DATA_PATH,  # Use absolute path
            structure_col="slab",
            target_props=["WF_top"],
            frame_averaging="3D",
            fa_method="all"
        )
        
        if len(dataset) == 0:
            print("No structures were loaded!")
            return False
        
        # Get first item and create a batch
        data = dataset[0]
        if not hasattr(data, 'fa_pos'):
            print("No frame averaging results found!")
            return False
        
        # Initialize model
        model = FAENet(
            cutoff=6.0,
            num_gaussians=50,
            hidden_channels=64,  # smaller for testing
            num_filters=64,      # smaller for testing
            num_interactions=2,  # fewer for testing
            output_properties=["WF_top"]
        )
        
        # Manual forward pass with frames
        print(f"Number of frames: {len(data.fa_pos)}")
        
        # Process each frame
        all_preds = []
        original_pos = data.pos.clone()
        original_cell = data.cell.clone() if hasattr(data, 'cell') and data.cell is not None else None
        
        for i in range(len(data.fa_pos)):
            # Set positions to current frame
            data.pos = data.fa_pos[i]
            if hasattr(data, 'fa_cell') and data.fa_cell is not None:
                data.cell = data.fa_cell[i]
            
            # Add batch dimension for single graph
            data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
            
            # Forward pass
            with torch.no_grad():
                pred = model(data)
                all_preds.append(pred)
            
            print(f"Frame {i} prediction shape: {pred['WF_top'].shape}")
        
        # Restore original positions
        data.pos = original_pos
        if original_cell is not None:
            data.cell = original_cell
        
        # Average predictions
        avg_pred = sum(p["WF_top"] for p in all_preds) / len(all_preds)
        print(f"Average prediction shape: {avg_pred.shape}")
        
        return True
        
    except Exception as e:
        print(f"Error in test_forward_with_frames: {e}")
        return False

def test_simple_config():
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
    assert simple_config.cutoff == nested_config.model.cutoff
    assert simple_config.hidden_channels == nested_config.model.hidden_channels
    assert simple_config.frame_averaging == nested_config.training.frame_averaging
    assert simple_config.data_dir == nested_config.data.data_dir
    assert simple_config.target_properties == nested_config.data.target_properties
    
    print("SimpleConfig conversion test passed!")
    return True


def run_all_tests():
    """Run all integration tests."""
    tests = [
        test_csv_loading,
        test_frame_averaging,
        test_dataloader_creation,
        test_forward_with_frames,
        test_simple_config
    ]
    
    results = {}
    
    for test_func in tests:
        test_name = test_func.__name__
        try:
            result = test_func()
            results[test_name] = "✅ PASSED" if result else "❌ FAILED"
        except Exception as e:
            print(f"Error in {test_name}: {e}")
            results[test_name] = f"❌ ERROR: {e}"
    
    # Print summary
    print("\n=== Test Results ===")
    for test_name, result in results.items():
        print(f"{test_name}: {result}")
    
    # Return overall success
    return all(result.startswith("✅") for result in results.values())

if __name__ == "__main__":
    success = run_all_tests()
    print(f"\nIntegration testing {'successful' if success else 'failed'}")
    
    if not success:
        print("\nSome tests failed. Please check the output above for details.")
    else:
        print("\nAll tests passed! The frame averaging and graph construction integration is working correctly.")