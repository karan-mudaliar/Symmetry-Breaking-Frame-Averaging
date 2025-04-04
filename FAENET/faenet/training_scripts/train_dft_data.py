#!/usr/bin/env python
"""
Training script for DFT data using the FAENet model.
Handles multiple surface property predictions with frame averaging.
"""
import os
import sys
from pathlib import Path
import structlog

# Configure structlog
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
        structlog.dev.ConsoleRenderer()
    ]
)
logger = structlog.get_logger()

# Ensure the parent package is in the path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from faenet.train import train_faenet
from faenet.config import get_config
from faenet.utils import generate_run_name

def main():
    """
    Main function to set up configuration and run FAENet training.
    """
    # Get configuration from command line arguments
    config = get_config()

    # Set default path if not specified through CLI
    if str(config.data_path) == "./data":
        config.data_path = Path("/home/mudaliar.k/github/Symmetry-Breaking-Frame-Averaging/data/DFT_data.csv")
        logger.info("using_default_data_path", path=str(config.data_path))
    
    # Set default output directory if not specified
    if str(config.output_dir) == "./outputs":
        # Generate a unique run name
        run_name = generate_run_name()
        config.output_dir = Path(f"/home/mudaliar.k/github/Symmetry-Breaking-Frame-Averaging/outputs/faenet_dft/{run_name}")
        logger.info("using_default_output_dir", path=str(config.output_dir), run_name=run_name)
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Log configuration
    logger.info(
        "training_configuration",
        data_path=str(config.data_path),
        output_dir=str(config.output_dir),
        target_properties=config.target_properties,
        frame_averaging=config.frame_averaging,
        fa_method=config.fa_method,
        model_size={
            "hidden_channels": config.hidden_channels,
            "num_interactions": config.num_interactions,
            "dropout": config.dropout
        },
        training_params={
            "batch_size": config.batch_size,
            "epochs": config.epochs,
            "learning_rate": config.learning_rate,
            "device": config.device
        }
    )
    
    # Run training directly with config 
    logger.info("starting_training")
    model, test_loader = train_faenet(**config.model_dump())
    
    logger.info("training_completed", output_dir=str(config.output_dir))
    
    print(f"\n{'='*80}")
    print(f"Training completed successfully!")
    print(f"Output directory: {config.output_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()