"""
Configuration management for the FAENet model using pydantic and tyro.
Integrates frame averaging and enhanced graph construction options.
"""
from pathlib import Path
from typing import List, Optional, Union, Literal, Dict, Any

import tyro
from pydantic import BaseModel, Field


class Config(BaseModel):
    """Configuration class for FAENet.
    
    This is a flat configuration structure that makes it easy to set and access
    parameters without nested objects.
    """
    # Model parameters
    cutoff: float = Field(6.0, description="Cutoff distance for interactions")
    max_neighbors: int = Field(40, description="Maximum number of neighbors per atom")
    num_gaussians: int = Field(50, description="Number of gaussians for distance expansion")
    hidden_channels: int = Field(128, description="Hidden channels in model")
    num_filters: int = Field(128, description="Number of filters in model")
    num_interactions: int = Field(4, description="Number of interaction blocks")
    dropout: float = Field(0.0, description="Dropout rate")
    output_properties: List[str] = Field(["energy"], description="Properties to predict")
    regress_forces: bool = Field(False, description="Whether to predict forces")
    
    # Training parameters
    batch_size: int = Field(32, description="Batch size")
    epochs: int = Field(100, description="Number of epochs")
    learning_rate: float = Field(0.001, description="Learning rate")
    weight_decay: float = Field(1e-5, description="Weight decay for optimizer")
    train_ratio: float = Field(0.8, description="Training set ratio")
    val_ratio: float = Field(0.1, description="Validation set ratio")
    test_ratio: float = Field(0.1, description="Test set ratio")
    frame_averaging: Optional[Literal["3D", "2D"]] = Field(None, 
        description="Frame averaging dimension (None, '2D', '3D')")
    fa_method: Literal["all", "det", "random", "se3-all", "se3-det", "se3-random"] = Field(
        "all", description="Frame averaging method"
    )
    force_weight: float = Field(0.1, description="Weight for force loss")
    seed: int = Field(42, description="Random seed")
    num_workers: int = Field(0, description="Number of worker processes for data loading")
    checkpoint_interval: int = Field(10, description="Save checkpoint every N epochs")
    eval_interval: int = Field(5, description="Evaluate model every N epochs")
    early_stopping_patience: int = Field(20, description="Patience for early stopping")
    
    # Data parameters
    data_path: Path = Field(Path("./data"), description="Directory with data files or CSV file")
    structure_col: Optional[str] = Field("slab", 
        description="Column name for structure data in CSV format")
    target_properties: List[str] = Field(["energy"], description="Target properties to predict")
    prop_files: Optional[List[str]] = Field(None, description="Files with property values")
    pbc: bool = Field(True, description="Use periodic boundary conditions")
    limit: Optional[int] = Field(None, description="Limit number of structures to process")
    
    # General parameters
    output_dir: Path = Field(Path("./outputs"), description="Output directory")
    inference_output: str = Field("predictions.json", description="File to save predictions")
    device: Literal["cuda", "cpu"] = Field("cuda", description="Device to run on")


# For backward compatibility with very old code
class FAENetConfig(Config):
    """Alias for Config to maintain backward compatibility with oldest code."""
    pass


def get_config() -> Config:
    """Parse command line arguments into a Config object."""
    config = tyro.cli(Config)
    return config


if __name__ == "__main__":
    # Example usage (this will capture command line arguments)
    config = get_config()
    
    # Print an example command for running training
    print("\nExample command for training FAENet:")
    print("python -m faenet.train --data_path=./test_data/surface_prop_data_set_top_bottom.csv " 
          "--structure_col=slab --target_properties=[WF_top,WF_bottom,cleavage_energy] "
          "--frame_averaging=3D --fa_method=all "
          "--regress_forces=False --output_dir=./results")
    
    # When imported in train.py, you would use:
    # from config import get_config
    # config = get_config()
    # # Access parameters directly:
    # cutoff = config.cutoff