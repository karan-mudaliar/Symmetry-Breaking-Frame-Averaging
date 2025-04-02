"""
Configuration management for the FAENet model using pydantic and tyro.
Integrates frame averaging and enhanced graph construction options.
"""
from pathlib import Path
from typing import List, Optional, Union, Literal, Dict, Any

import tyro
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Configuration for the FAENet model."""
    cutoff: float = Field(6.0, description="Cutoff distance for interactions")
    max_neighbors: int = Field(40, description="Maximum number of neighbors per atom")
    num_gaussians: int = Field(50, description="Number of gaussians for distance expansion")
    hidden_channels: int = Field(128, description="Hidden channels in model")
    num_filters: int = Field(128, description="Number of filters in model")
    num_interactions: int = Field(4, description="Number of interaction blocks")
    dropout: float = Field(0.0, description="Dropout rate")
    output_properties: List[str] = Field(["energy"], description="Properties to predict")
    regress_forces: bool = Field(False, description="Whether to predict forces")


class TrainingConfig(BaseModel):
    """Configuration for model training."""
    batch_size: int = Field(32, description="Batch size")
    epochs: int = Field(100, description="Number of epochs")
    lr: float = Field(0.001, description="Learning rate")
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


class DataConfig(BaseModel):
    """Configuration for dataset loading."""
    data_dir: Path = Field(Path("./data"), description="Directory with data files or CSV file")
    structure_col: Optional[str] = Field("slab", 
        description="Column name for structure data in CSV format")
    target_properties: List[str] = Field(["energy"], description="Target properties to predict")
    prop_files: Optional[List[str]] = Field(None, description="Files with property values")
    pbc: bool = Field(True, description="Use periodic boundary conditions")
    limit: Optional[int] = Field(None, description="Limit number of structures to process")


class Config(BaseModel):
    """Main configuration class."""
    model: ModelConfig = Field(ModelConfig(), description="Model configuration")
    training: TrainingConfig = Field(TrainingConfig(), description="Training configuration")
    data: DataConfig = Field(DataConfig(), description="Data configuration")
    output_dir: Path = Field(Path("./outputs"), description="Output directory")
    inference_output: str = Field("predictions.json", description="File to save predictions")
    device: Literal["cuda", "cpu"] = Field("cuda", description="Device to run on")


class SimpleConfig(BaseModel):
    """Simplified flat configuration class for FAENet.
    
    This class provides a flattened version of the nested Config, making it easier
    to set and access parameters with fewer levels of nesting.
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
    lr: float = Field(0.001, description="Learning rate")
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
    data_dir: Path = Field(Path("./data"), description="Directory with data files or CSV file")
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
    
    def to_nested_config(self) -> Config:
        """Convert the simplified configuration to the nested configuration format."""
        model_config = ModelConfig(
            cutoff=self.cutoff,
            max_neighbors=self.max_neighbors,
            num_gaussians=self.num_gaussians,
            hidden_channels=self.hidden_channels,
            num_filters=self.num_filters,
            num_interactions=self.num_interactions,
            dropout=self.dropout,
            output_properties=self.output_properties,
            regress_forces=self.regress_forces,
        )
        
        training_config = TrainingConfig(
            batch_size=self.batch_size,
            epochs=self.epochs,
            lr=self.lr,
            weight_decay=self.weight_decay,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
            frame_averaging=self.frame_averaging,
            fa_method=self.fa_method,
            force_weight=self.force_weight,
            seed=self.seed,
            num_workers=self.num_workers,
            checkpoint_interval=self.checkpoint_interval,
            eval_interval=self.eval_interval,
            early_stopping_patience=self.early_stopping_patience,
        )
        
        data_config = DataConfig(
            data_dir=self.data_dir,
            structure_col=self.structure_col,
            target_properties=self.target_properties,
            prop_files=self.prop_files,
            pbc=self.pbc,
            limit=self.limit,
        )
        
        return Config(
            model=model_config,
            training=training_config,
            data=data_config,
            output_dir=self.output_dir,
            inference_output=self.inference_output,
            device=self.device,
        )


# For backward compatibility with train.py
class FAENetConfig(Config):
    """Alias for Config to maintain backward compatibility."""
    pass


def get_config() -> Config:
    """Parse command line arguments into a Config object."""
    config = tyro.cli(Config)
    return config


def get_simple_config() -> SimpleConfig:
    """Parse command line arguments into a SimpleConfig object."""
    config = tyro.cli(SimpleConfig)
    return config


if __name__ == "__main__":
    # Example usage (this will capture command line arguments)
    config = get_config()
    print(f"Config: {config.model}")
    
    # Print an example command for running training
    print("\nExample command for training FAENet:")
    print("python train.py --data.data_dir=./test_data/surface_prop_data_set_top_bottom.csv " 
          "--data.structure_col=slab --data.target_properties=[WF_top,WF_bottom,cleavage_energy] "
          "--training.frame_averaging=3D --training.fa_method=all "
          "--model.regress_forces=False --output_dir=./results")
    
    print("\nExample command using SimpleConfig:")
    print("python train.py --simple --data_dir=./test_data/surface_prop_data_set_top_bottom.csv " 
          "--structure_col=slab --target_properties=[WF_top,WF_bottom,cleavage_energy] "
          "--frame_averaging=3D --fa_method=all "
          "--regress_forces=False --output_dir=./results")
    
    # When imported in train.py, you would use:
    # from config import get_config, get_simple_config
    # config = get_config()  # For nested config
    # # OR
    # config = get_simple_config()  # For simplified config
    # # For SimpleConfig, you can access parameters directly:
    # cutoff = config.cutoff
    # # For nested Config, you need to use the nested structure:
    # cutoff = config.model.cutoff