# Symmetry-Breaking Frame Averaging

## Overview

This repository contains code for the paper [Symmetry-Breaking Frame Averaging: A Simple Way to Enhance the Performance of Equivariant Networks](https://arxiv.org/abs/2306.15145).

![Pipeline](FAENET/assets/pipeline.png)

The original implementation was complex and integrated with the OCP (Open Catalyst Project) codebase. A simplified and more accessible implementation has been created in the [FAENET](./FAENET) directory.

## Simplified Implementation

We've created a streamlined version of FAENet with frame averaging in the [FAENET](./FAENET) directory. This implementation:

- Provides a clean, modular codebase with clear separation of components
- Uses modern Python practices with type-safe configuration using pydantic and tyro
- Offers enhanced graph construction for crystal structures
- Supports both 3D and 2D frame averaging approaches
- Focuses on multi-property prediction
- Integrates seamlessly with PyTorch Geometric

**→ Please see the [FAENET](./FAENET) directory for the simplified implementation.**

## Key Components

The simplified implementation includes:

1. **Frame Averaging**: Break rotational symmetry with PCA-based frame transforms
   - Support for 3D (all directions) and 2D (preserving z-coordinate) transformations
   - Multiple frame methods: "all" (uses all 8 frames), "det" (deterministic frame), "random" (random frame)

2. **Graph Construction**: Enhanced handling of crystal structures
   - Proper periodic boundary condition (PBC) handling
   - Radius-based neighbor finding with cell offsets

3. **Model Architecture**: Simplified FAENet implementation
   - Embedding block: Convert atom types to embeddings
   - Interaction blocks: Message passing between atoms
   - Output blocks: Property prediction from atom representations

4. **Dataset Handling**: Flexible and efficient data loading
   - Support for both file-based and CSV-based loading
   - Integrated frame averaging during data loading

## Citation

```
@article{duval2023symmetry,
  title={Symmetry-Breaking Frame Averaging: A Simple Way to Enhance the Performance of Equivariant Networks},
  author={Duval, Alexandre and Levie, Ron and Bronstein, Michael and Bruna, Joan},
  journal={arXiv preprint arXiv:2306.15145},
  year={2023}
}
```

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.