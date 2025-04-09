# CLAUDE.md - DETAILED CONTEXT

This file provides extensive documentation about the Symmetry-Breaking-Frame-Averaging repository, ongoing issues, and debug context for Claude Code.

## QUICK REFERENCE: Adding Consistency Loss Feature

The consistency loss feature enhances FAENet's equivariance by penalizing variance in predictions across different frames of the same structure. Here's the core implementation steps:

1. **Configuration**: In `config.py`, add these parameters:
   ```python
   # Consistency loss parameters
   consistency_loss: bool = Field(False, description="Whether to use variance-based consistency loss")
   consistency_weight: float = Field(0.1, description="Weight for consistency loss")
   consistency_norm: bool = Field(True, description="Whether to normalize consistency loss")
   ```

2. **Loss Function**: In `frame_averaging.py`, implement:
   ```python
   def compute_consistency_loss(frame_predictions, normalize=True):
       """Compute consistency loss as variance across frame predictions for each object"""
       # Stack predictions from all frames
       preds = torch.stack(frame_predictions)  # shape: [num_frames, batch_size, output_dim]
       
       # Calculate variance across frames with unbiased=False
       variance_per_object = torch.var(preds, dim=0, unbiased=False)
       
       # Average across output dimensions
       variance_per_object = variance_per_object.mean(dim=-1)
       
       # Normalize if requested
       if normalize:
           norm_factor = (preds**2).mean(dim=[0, 2]) + 1e-6
           variance_per_object = variance_per_object / norm_factor
       
       # Average across batch
       mean_variance = variance_per_object.mean()
       
       return mean_variance
   ```

3. **Training Integration**: In `train.py`, add to the training loop:
   ```python
   # In the frame averaging section (where multiple frames are processed):
   consistency_loss = None
   if hasattr(config, 'consistency_loss') and config.consistency_loss:
       from faenet.frame_averaging import compute_consistency_loss
       consistency_loss = 0
       
   # After computing main loss for each property
   if consistency_loss is not None:
       prop_consistency = compute_consistency_loss(
           e_all[prop_idx], 
           normalize=config.consistency_norm
       )
       consistency_loss += prop_consistency
       
   # Add to total loss
   if consistency_loss is not None:
       loss += config.consistency_weight * consistency_loss
       # Log metrics
       mlflow.log_metric("consistency_loss", consistency_loss.item(), step=epoch)
   ```

4. **Tests**: Create `test_consistency_loss.py` to test variance calculation, normalization, etc.

5. **CRITICAL**: When implementing this in train_faenet, ensure consistency loss parameters are explicitly preserved:
   ```python
   # Explicitly include consistency loss parameters
   'consistency_loss': consistency_loss_enabled,
   'consistency_weight': consistency_weight,
   'consistency_norm': consistency_norm
   ```

Key points:
- Feature is completely optional (disabled by default)
- Only functional when frame_averaging is also enabled
- Requires no changes to model architecture
- Zero overhead when disabled

## Common Bugs & Solutions

1. **Parameter duplication**: Don't pass the same parameter both in kwargs and explicitly
   ```python
   # WRONG - causes "multiple values for keyword argument" error
   model, _ = train_faenet(
       consistency_loss=True,
       **kwargs  # If kwargs already contains consistency_loss
   )
   
   # CORRECT - remove from kwargs if passing explicitly
   if 'consistency_loss' in kwargs:
       kwargs.pop('consistency_loss')
   model, _ = train_faenet(
       consistency_loss=True,
       **kwargs
   )
   ```

2. **Parameter preservation**: Ensure parameters are preserved when creating configurations
   ```python
   # WRONG - parameter lost when creating new Config
   config_params = {
       'batch_size': batch_size,
       # Other parameters - consistency_loss is missing!
   }
   config = Config(**config_params)  # consistency_loss will be default (False)
   
   # CORRECT - explicitly include consistency loss parameters
   config_params = {
       'batch_size': batch_size,
       # Other parameters
       'consistency_loss': consistency_loss_enabled,  # Preserve parameter
   }
   config = Config(**config_params)
   ```

## Repository Architecture

### Overview

FAENet (Frame Averaging Equivariant Network) is a Graph Neural Network that achieves approximate SO(2)/SO(3) equivariance using frame averaging. Instead of building equivariance directly into the network architecture, it:

1. Generates multiple reference frames (orientations) of each input structure
2. Processes each frame independently 
3. Averages predictions across all frames
4. Optionally applies a consistency loss to encourage similar predictions across frames

This approach is particularly useful for crystal slabs where the z-axis (surface normal) has physical significance but rotation in the xy-plane should not affect predictions.

### Key Modules

1. **faenet/frame_averaging.py**:
   - `frame_averaging_2D`: Generates 4 frames by reflecting in xy-plane while preserving z-axis
   - `frame_averaging_3D`: Generates 8 frames by reflecting in all dimensions
   - `compute_consistency_loss`: Calculates variance across frame predictions

2. **faenet/train.py**:
   - Main training loop with frame averaging logic
   - MLflow integration for experiment tracking
   - Consistency loss integration

3. **faenet/config.py**:
   - Pydantic model for configuration
   - Configuration parameters for consistency loss

## Build/Test Commands
- Run all tests: `python -m pytest`
- Run specific test: `python -m pytest tests/test_frame_averaging.py -v`
- Run with specific pattern: `python -m pytest -k "test_consistency"`
- Training: `python -m faenet.train --data_dir=./test_data/surface_prop_data_set_top_bottom.csv --structure_col=slab --target_properties=[WF_top,WF_bottom] --frame_averaging=3D --consistency_loss=True`

## Code Style and Patterns
- Use structlog for logging, not print statements
- Ensure consistency loss is completely optional and non-intrusive
- Code follows PyTorch conventions with tensor operations
- Frame averaging is the core functionality
- Consistency loss enhances equivariance (optional feature)
- Configuration uses Pydantic models for type safety
- Follow snake_case for functions/variables, PascalCase for classes