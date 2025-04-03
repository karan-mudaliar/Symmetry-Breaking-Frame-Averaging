# FAENet: Frame Averaging Equivariant Network

A rotation-invariant graph neural network implementation that uses frame averaging for crystal property prediction.

## Setup

### Environment Setup

```bash
# Create a new conda environment
conda create -n faenet python=3.10
conda activate faenet

# Install PyTorch (adjust CUDA version if needed)
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia

# Install PyTorch Geometric
conda install pyg -c pyg

# Install other dependencies
pip install tyro pydantic pymatgen ase tqdm matplotlib jupyter
```

### Additional Dependencies

The code relies on the following packages:
- PyTorch
- PyTorch Geometric
- PyMatGen (for crystal structure handling)
- Tyro & Pydantic (for configuration)
- NumPy, Matplotlib, ASE, tqdm

## Basic Usage

```python
from faenet.train import train_faenet

# Train a model with simplified interface
model, test_loader = train_faenet(
    data_path="path/to/data.csv",       # CSV file with structure data
    structure_col="atoms",              # Column containing structure dictionaries
    target_properties=["energy", "gap"], # Properties to predict
    output_dir="output",                # Where to save model and results
    frame_averaging="3D",               # Use 3D frame averaging (or "2D" or None)
    cutoff=6.0,                         # Cutoff radius for neighbors
    max_neighbors=40,                   # Maximum neighbors per atom
    batch_size=32,                      # Batch size for training
    epochs=100                          # Number of training epochs
)

# Alternatively, use the configuration system
from faenet.faenet import FAENet
from faenet.config import FAENetConfig

# Initialize model with custom configuration
config = FAENetConfig()
config.model.hidden_channels = 128
model = FAENet(
    cutoff=config.model.cutoff,
    hidden_channels=config.model.hidden_channels,
    output_properties=["energy"]
)
```

## Features

- Frame averaging for rotation invariance
- Proper handling of periodic boundary conditions for crystal structures
- Support for multiple property prediction
- Type-safe configuration system

## Command Line Usage

You can also run training directly from the command line:

```bash
# Train with default configuration
python -m faenet.train

# Train with custom parameters
python -m faenet.train --data.data_dir=data/my_structures.csv --data.structure_col=atoms --training.epochs=200 --model.hidden_channels=128
```

## Example Script

Create a simple script to train a model:

```python
# train_example.py
from faenet.train import train_faenet

if __name__ == "__main__":
    train_faenet(
        data_path="data/structures.csv",
        structure_col="atoms",
        target_properties=["formation_energy", "band_gap"],
        frame_averaging="3D",
        output_dir="results/my_model"
    )
```