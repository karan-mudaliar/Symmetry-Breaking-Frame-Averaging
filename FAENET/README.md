# FAENet: Frame Averaging Equivariant Network

This repository contains a simplified implementation of FAENet (Frame Averaging Equivariant Network) for predicting properties of 3D structures, particularly crystal slabs. The implementation includes frame averaging techniques to achieve rotational equivariance.

## Overview

This is a streamlined version of the original Symmetry-Breaking-Frame-Averaging implementation, focusing on the core functionality with:

- Clean, modular implementation with clear separation of components
- Type-safe configuration using pydantic and tyro
- Enhanced graph construction for crystal structures
- Support for multiple frame averaging approaches (3D, 2D)
- Focus on multi-property prediction
- Integration with PyTorch Geometric

## Features

- **Frame Averaging**: Break rotational symmetry with PCA-based frame transforms
  - Support for 3D (all directions) and 2D (preserving z-coordinate) transformations
  - Multiple frame methods: "all" (uses all 8 frames), "det" (deterministic frame), "random" (random frame)
  - SE(3) variants that preserve orientation
  
- **Graph Construction**: Enhanced handling of crystal structures
  - Proper periodic boundary condition (PBC) handling
  - Radius-based neighbor finding with cell offsets
  - Supports both file-based and CSV-based structure loading
  
- **Model Architecture**: Simplified FAENet implementation
  - Embedding block: Convert atom types to embeddings
  - Interaction blocks: Message passing between atoms
  - Output blocks: Property prediction from atom representations
  - Optional force prediction

- **Dataset Handling**: Flexible and efficient data loading
  - Support for both file-based and CSV-based loading
  - Integrated frame averaging during data loading
  - Multi-property prediction

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.10+
- PyTorch Geometric 2.0+
- Pymatgen
- Tyro
- Pydantic
- NumPy
- Tqdm

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/FAENET.git
cd FAENET
```

2. Install dependencies:
```bash
pip install torch torch_geometric pymatgen tyro pydantic numpy tqdm
```

## Usage

### Training

Train a model with CSV data and frame averaging:

```bash
python -m faenet.train \
    --data.data_dir=./test_data/surface_prop_data_set_top_bottom.csv \
    --data.structure_col=slab \
    --data.target_properties=[WF_top,WF_bottom,cleavage_energy] \
    --training.frame_averaging=3D \
    --training.fa_method=all \
    --model.regress_forces=False \
    --output_dir=./results
```

### Configuration

The implementation uses pydantic and tyro for type-safe configuration:

```python
# Example of configuration structure
config = Config(
    model=ModelConfig(
        cutoff=6.0,
        hidden_channels=256,
        num_interactions=5,
        output_properties=["energy", "forces"]
    ),
    training=TrainingConfig(
        batch_size=32,
        frame_averaging="3D",
        fa_method="all"
    ),
    data=DataConfig(
        data_dir="path/to/data.csv",
        structure_col="structure",
        target_properties=["energy"]
    )
)
```

## Frame Averaging Methods

The model supports different frame averaging methods:

- `all`: Use all possible frames (up to 8 for 3D, 4 for 2D)
- `det`: Use a deterministic frame
- `random`: Randomly select a frame for each batch
- `se3-all`, `se3-det`, `se3-random`: Same as above but with SE(3) equivariance constraints

## Tests

The repository includes integration tests to validate the functionality:

```bash
python -m faenet.test_integration
```

## Directory Structure

- `faenet/`
  - `config.py`: Type-safe configuration with pydantic and tyro
  - `dataset.py`: Enhanced dataset for crystal slabs
  - `faenet.py`: Main model implementation
  - `frame_averaging.py`: Frame averaging implementation
  - `graph_construction.py`: Graph construction for crystal structures
  - `train.py`: Training loop with frame averaging
  - `utils.py`: Utility functions
  - `test_integration.py`: Integration tests

## Citation

If you use this code in your research, please cite the original FAENet paper:

```
@article{duval2023symmetry,
  title={Symmetry-Breaking Frame Averaging: A Simple Way to Enhance the Performance of Equivariant Networks},
  author={Duval, Alexandre and Levie, Ron and Bronstein, Michael and Bruna, Joan},
  journal={arXiv preprint arXiv:2306.15145},
  year={2023}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.