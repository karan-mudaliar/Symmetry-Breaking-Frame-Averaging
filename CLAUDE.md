# CLAUDE.md - COMPREHENSIVE PROJECT REFERENCE

This document provides complete documentation of the Symmetry-Breaking-Frame-Averaging repository, serving as a reference for current implementation, recent enhancements, and ongoing development priorities.

## Project Overview

FAENet (Frame Averaging Equivariant Network) is a Graph Neural Network that predicts properties of crystal slabs with approximate SO(2)/SO(3) equivariance. The key innovations are:

1. **Comformer-inspired Graph Construction**: Creates rich graph representations from crystal structures
2. **Frame Averaging**: Generates multiple reference frames (orientations) of each input structure
3. **Consistency Loss**: Penalizes variance in predictions across frames to enhance equivariance
4. **Regression with Proper Standardization**: Predicts properties using standardized regression targets

This approach is particularly useful for crystal slabs where the z-axis (surface normal) has physical significance but rotation in the xy-plane should not affect predictions.

## Recent Enhancements

We recently implemented several major enhancements to the codebase:

### 1. Enhanced Graph Construction (Comformer-inspired)

The enhanced graph construction provides several improvements adapted from the Comformer implementation in `comformer_context.txt`:

1. **Lattice Correction**: Finds invariant coordinate system for crystals
   ```python
   # Finding orthogonal lattice vectors (from Comformer)
   lat1, lat2, lat3 = find_lattice_vectors(structure, cutoff=6.0)
   
   # Ensure proper angles and coordinate system
   if angle_from_array(lat1, lat2, lattice_matrix) > 90.0:
       lat2 = -lat2
   if angle_from_array(lat1, lat3, lattice_matrix) > 90.0:
       lat3 = -lat3
   if not correct_coord_sys(lat1, lat2, lat3, lattice_matrix):
       lat1, lat2, lat3 = -lat1, -lat2, -lat3
   ```

2. **CGCNN-style Atom Features**: Uses rich elemental representations instead of one-hot encoding
   ```python
   # CGCNN features for elements (periodic table properties)
   elem_features = {
       # H
       1: [1, 1.0079, 0.31, 2.2, 1312.0, 0.754, 14.01, 0, 1],
       # C
       6: [2, 12.0107, 0.77, 2.55, 1086.5, 1.576, 6.8, 4, 2],
       # O
       8: [2, 15.9994, 0.66, 3.44, 1313.9, 2.002, 5.17, 6, 2],
       # Si
       14: [3, 28.0855, 1.32, 1.9, 786.5, 1.176, 9.81, 4, 3],
       # ... and many more elements
   }
   ```

3. **K-Nearest Neighbors**: Better handling of periodic boundary conditions
   ```python
   # Comformer approach to nearest neighbors
   edges, lat1, lat2, lat3 = nearest_neighbor_edges_enhanced(
       structure, cutoff=6.0, max_neighbors=12, use_canonize=True
   )
   
   # Build graph with lattice information
   u, v, r, nei, atom_lat = build_undirected_edgedata(
       structure, edges, lat1, lat2, lat3
   )
   ```

4. **Enhanced Edge Attributes**: Additional edge features for improved learning
   ```python
   # Calculate edge length and unit vectors (for angle calculations)
   edge_length = torch.norm(rel_pos, dim=1, keepdim=True)
   edge_unit_vec = rel_pos / torch.clamp(edge_length, min=1e-8)
   
   # Add to graph data
   data = Data(
       # Standard attributes
       x=x, edge_index=edge_index, edge_attr=rel_pos, pos=pos,
       # Enhanced attributes
       edge_length=edge_length.squeeze(-1),
       edge_unit_vec=edge_unit_vec,
       edge_lattice=nei,
       atom_lattice=atom_lat,
   )
   ```

### 2. Proper Standardization for Regression

The enhanced standardization ensures regression targets are properly handled:

1. **Standardizer Class**: Dedicated class for managing standardization
   ```python
   class Standardizer:
       """Class to handle standardization of regression targets.
       
       Computes mean and standard deviation on training data,
       then applies the same transformation to validation and test sets.
       """
       
       def __init__(self, property_names=None):
           self.property_names = property_names or []
           self.mean = {}
           self.std = {}
           self.is_fitted = False
       
       def fit(self, data_values):
           """Compute mean and standard deviation from training data."""
           for prop in self.property_names:
               values = np.array(data_values[prop])
               self.mean[prop] = float(np.mean(values))
               self.std[prop] = float(np.std(values))
               # Handle zero std
               if self.std[prop] == 0 or np.isnan(self.std[prop]):
                   self.std[prop] = 1.0
           self.is_fitted = True
           return self
       
       def transform(self, value, property_name):
           """Apply standardization to a value."""
           return (value - self.mean[property_name]) / self.std[property_name]
       
       def inverse_transform(self, value, property_name):
           """Undo standardization for a value."""
           return value * self.std[property_name] + self.mean[property_name]
   ```

2. **Training Set Statistics**: Uses only training set statistics for standardization
   ```python
   # Extract property values from training set only
   train_property_values = {}
   for prop_name in full_dataset.property_values:
       train_property_values[prop_name] = [
           full_dataset.get_raw_value(idx, prop_name) 
           for idx in train_indices
       ]
   
   # Fit standardizer on training data
   standardizer = Standardizer(full_dataset.property_values.keys())
   standardizer.fit(train_property_values)
   ```

3. **Consistent Application**: Same statistics applied to validation and test sets
   ```python
   # Create datasets with same standardizer
   train_dataset = SlabDataset(
       # ... other parameters ...
       standardize=standardize,
       standardizer=standardizer  # Use fitted standardizer
   )
   
   # Use the same standardizer for validation and test
   val_dataset = Subset(train_dataset, val_indices)
   test_dataset = Subset(train_dataset, test_indices)
   ```

4. **Inverse Transform**: Properly transforms predictions back to original scale
   ```python
   # Apply inverse standardization during inference
   if standardizer is not None and standardizer.is_fitted and prop in standardizer.mean:
       orig_value = standardizer.inverse_transform(std_value, prop)
       # Store both standardized and original values
       result[f"{prop}_std"] = std_value
       result[prop] = float(orig_value)
   ```

### 3. Consistency Loss for Equivariance

The consistency loss feature enhances FAENet's equivariance by penalizing variance in predictions across different frames of the same structure:

1. **Loss Function**: Computes variance across frame predictions
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

2. **Training Integration**: Added to the training loop
   ```python
   # In the frame averaging section:
   consistency_loss = None
   if hasattr(config, 'consistency_loss') and config.consistency_loss:
       from faenet.frame_averaging import compute_consistency_loss
       consistency_loss = 0
       
   # After computing loss for each property
   if consistency_loss is not None:
       prop_consistency = compute_consistency_loss(
           e_all[prop_idx], 
           normalize=config.consistency_norm
       )
       consistency_loss += prop_consistency
       
   # Add to total loss
   if consistency_loss is not None:
       loss += config.consistency_weight * consistency_loss
       mlflow.log_metric("consistency_loss", consistency_loss.item(), step=epoch)
   ```

3. **Configuration**: Added parameters to config.py
   ```python
   # Consistency loss parameters
   consistency_loss: bool = Field(False, description="Whether to use variance-based consistency loss")
   consistency_weight: float = Field(0.1, description="Weight for consistency loss")
   consistency_norm: bool = Field(True, description="Whether to normalize consistency loss")
   ```

## Comformer Context

The repository contains a reference file `comformer_context.txt` with the original Comformer implementation code. Key elements from this file have been adapted into our implementation:

1. **Graph Construction**: The functions `nearest_neighbor_edges_submit`, `build_undirected_edgedata`, and `atom_dgl_multigraph` provide the foundation for our enhanced graph construction.

2. **Lattice Correction**: The lattice correction approach finds orthogonal basis vectors and ensures a correct coordinate system:
   ```python
   # Find the invariant corner
   if angle_from_array(lat1,lat2,lat.matrix) > 90.0:
       lat2 = - lat2
   if angle_from_array(lat1,lat3,lat.matrix) > 90.0:
       lat3 = - lat3
   # Find the invariant coord system
   if not correct_coord_sys(lat1, lat2, lat3, lat.matrix):
       lat1, lat2, lat3 = -lat1, -lat2, -lat3
   ```

3. **Standardization Approach**: The standardization technique in Comformer properly handles regression targets:
   ```python
   if mean_train == None:
       mean = self.labels.mean()
       std = self.labels.std()
       self.labels = (self.labels - mean) / std
   else:
       self.labels = (self.labels - mean_train) / std_train
       print("normalize using training mean %f and std %f" % (mean_train, std_train))
   ```

4. **Data Loading**: The approach to handling datasets ensures proper standardization across splits:
   ```python
   train_data, mean_train, std_train = get_pyg_dataset(
       dataset=dataset_train,
       # other parameters...
   )
   
   val_data,_,_ = get_pyg_dataset(
       dataset=dataset_val,
       # other parameters...
       mean_train=mean_train,
       std_train=std_train,
   )
   ```

## Key Modules

1. **faenet/graph_construction.py**:
   - Enhanced graph construction with Comformer-inspired techniques
   - Lattice correction for better periodic boundary conditions
   - CGCNN-style atom features for richer representation
   - K-nearest neighbor search with proper handling of periodic boundary conditions

2. **faenet/dataset.py**:
   - `SlabDataset`: Main dataset class with CSV and file loading support
   - `Standardizer`: Handles standardization of regression targets
   - `create_dataloader`: Creates train/val/test loaders with proper standardization

3. **faenet/frame_averaging.py**:
   - `frame_averaging_2D`: Generates 4 frames by reflecting in xy-plane while preserving z-axis
   - `frame_averaging_3D`: Generates 8 frames by reflecting in all dimensions
   - `compute_consistency_loss`: Calculates variance across frame predictions

4. **faenet/train.py**:
   - Main training loop with frame averaging and standardization
   - MLflow integration for experiment tracking
   - Consistency loss integration
   - Proper inverse standardization for predictions

5. **faenet/faenet.py**:
   - Graph neural network model architecture
   - Message passing layers
   - Output blocks for property prediction

6. **faenet/config.py**:
   - Pydantic model for configuration
   - Configuration parameters for all features

## Usage

### Command-Line Arguments

```bash
python -m faenet.train \
  --data_path=data.csv \
  --structure_col=slab \
  --target_properties=[WF_top,WF_bottom] \
  --frame_averaging=3D \
  --consistency_loss=True \
  --atom_features=cgcnn
```

### Code Example

```python
from faenet.train import train_faenet

# Train model with enhanced features
model, test_loader = train_faenet(
    data_path="data.csv",
    structure_col="slab",
    target_properties=["WF_top", "WF_bottom"],
    frame_averaging="3D",           # Enable frame averaging (3D mode)
    consistency_loss=True,          # Enable consistency loss
    consistency_weight=0.1,         # Weight for consistency loss
    atom_features="cgcnn",          # Use CGCNN atom features
    standardize=True,               # Enable standardization
    output_dir="./outputs",         # Save outputs and standardizer
    use_mlflow=True                 # Track experiment with MLflow
)
```

### Test Commands

```bash
# Run all tests
python -m pytest

# Test graph construction
python -m pytest tests/test_graph_construction.py -v

# Test standardization
python -m pytest tests/test_standardization.py -v

# Test consistency loss
python -m pytest -k "test_consistency"
```

## Common Issues & Solutions

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
   
   # CORRECT - explicitly include all parameters
   config_params = {
       'batch_size': batch_size,
       # Other parameters
       'consistency_loss': consistency_loss_enabled,  # Preserve parameter
   }
   config = Config(**config_params)
   ```

3. **Standardization**: Always standardize regression targets
   ```python
   # WRONG - inconsistent standardization
   train_loader = create_dataloader(..., standardize=True)
   val_loader = create_dataloader(..., standardize=False)  # Different stats!
   
   # CORRECT - use create_dataloader which handles standardization properly
   train_loader, val_loader, test_loader, _, standardizer = create_dataloader(
       data_source=data_path,
       standardize=True  # Applied correctly to all splits
   )
   ```

## Code Style and Patterns

- Use structlog for logging, not print statements
- Ensure new features are completely optional with sensible defaults
- Code follows PyTorch conventions with tensor operations
- Follow snake_case for functions/variables, PascalCase for classes
- Frame averaging is the core functionality
- Consistency loss enhances equivariance (optional feature)
- Configuration uses Pydantic models for type safety
- Tests use pytest and verify core functionality

## Crystal Slab Dataset and Graph Construction

The project works with crystal slabs from materials science datasets:

### Dataset Structure
- Example dataset: `test_data/surface_prop_data_set_top_bottom.csv`
- Contains structures (Ba, Cd, etc.) in PyMatgen Structure format
- Properties include surface properties like work functions (WF_top, WF_bottom)
- Unique identifiers formed from mpid, miller indices, and termination
- Supports both CSV with embedded structures and directory of POSCAR/VASP files

### Graph Construction with Frame Averaging
The key innovation of this project is combining CGCNN-style graph construction with frame averaging:

1. **Graph Construction**:
   ```python
   # Convert crystal structures to PyG graphs
   graph = structure_to_graph(
       structure, 
       cutoff=6.0, 
       max_neighbors=40, 
       pbc=True,
       atom_features="cgcnn"  # Uses element feature dictionary
   )
   ```

2. **CGCNN-style Atom Features**:
   ```python
   # Current implementation with element-specific features
   elem_features = {
       # H
       1: [1, 1.0079, 0.31, 2.2, 1312.0, 0.754, 14.01, 1, 1],
       # Format: [period, atomic_weight, covalent_radius, electronegativity, 
       #          ionization_energy, electron_affinity, specific_heat, 
       #          valence_electrons, group]
   }
   ```

3. **Frame Averaging**:
   ```python
   # Apply frame averaging to find equivariant frames
   fa_pos, fa_cell, fa_rot = frame_averaging_3D(
       graph.pos,
       graph.cell, 
       fa_method="all"  # Generate all possible frames
   )
   
   # For slab structures, typically use 2D averaging:
   fa_pos, fa_cell, fa_rot = frame_averaging_2D(
       graph.pos,
       graph.cell,
       fa_method="all"  # Preserves z-axis (surface normal)
   )
   ```

4. **Frame Processing in Training**:
   ```python
   # Apply model to each frame
   outputs_per_frame = []
   for frame_idx in range(len(fa_pos)):
       # Update positions using the frame
       graph.pos = fa_pos[frame_idx]
       
       # Get predictions for this frame
       pred = model(graph)
       outputs_per_frame.append(pred)
   
   # Average predictions across frames
   final_output = torch.stack(outputs_per_frame).mean(dim=0)
   
   # Optionally add consistency loss
   if use_consistency_loss:
       consistency_loss = compute_consistency_loss(outputs_per_frame)
       total_loss = main_loss + consistency_weight * consistency_loss
   ```

### Extension Options

1. **Alternative Element Features**: While the current implementation uses a custom dictionary of element features, more comprehensive options exist:

   ```python
   # Option 1: Using Matminer (requires additional dependency)
   from matminer.featurizers.composition import ElementProperty
   
   def get_atom_features_matminer(atomic_number):
       """Get atom features using matminer ElementProperty."""
       from pymatgen.core.periodic_table import Element
       element = Element.from_Z(atomic_number)
       featurizer = ElementProperty.from_preset("magpie")
       features = featurizer.featurize(element)
       return torch.tensor(features, dtype=torch.float)
   
   # Option 2: More fully utilize PyMatgen (already a dependency)
   from pymatgen.core.periodic_table import Element
   
   def get_atom_features_pymatgen(atomic_number):
       """Get atom features using pymatgen Element properties."""
       element = Element.from_Z(atomic_number)
       features = [
           element.row,
           element.atomic_mass,
           element.atomic_radius or 0.0,
           element.X or 0.0,
           element.ionization_energy or 0.0,
           # etc.
       ]
       return torch.tensor(features, dtype=torch.float)
   ```

2. **Development Constraints**:
   - Changes must be committed to test them in cloud environment
   - New dependencies must be added to requirements.txt or pyproject.toml
   - When possible, implement features using existing dependencies
   - Write comprehensive tests but defer running them until cloud execution

The combination of graph construction with frame averaging is a unique approach that preserves the benefits of GNNs while addressing their rotational variance limitations.

### Integration in Training Loop

The FAENet training approach with frame averaging is fundamentally different from most GNNs. Looking at the training function, we can see how graph construction and frame averaging work together:

```python
# In train() function:
# For each batch:
if frame_averaging:
    # Apply frame averaging transformation to get multiple reference frames
    batch = apply_frame_averaging_to_batch(batch, fa_method, frame_averaging)
    
    # Save original positions and cell
    original_pos = batch.pos.clone()
    original_cell = batch.cell.clone()
    
    # Create batches for each frame
    all_batches = []
    for i in range(len(batch.fa_pos)):
        # Create a copy of the batch for this frame
        frame_batch = batch.clone()
        # Update positions and cell for this frame
        frame_batch.pos = batch.fa_pos[i]
        if hasattr(batch, 'cell') and batch.cell is not None:
            frame_batch.cell = batch.fa_cell[i]
        all_batches.append(frame_batch)
    
    # Forward pass with all frames
    combined_batch = Batch.from_data_list(all_batches)
    all_preds = model(combined_batch)
    
    # Average predictions across frames
    avg_pred = sum(e_all[prop_idx]) / len(e_all[prop_idx])
    
    # Calculate main loss
    prop_loss = criterion(avg_pred, targets)
    
    # Optionally add consistency loss
    if consistency_loss is not None:
        prop_consistency = compute_consistency_loss(e_all[prop_idx])
        consistency_loss += prop_consistency
        
    # Add consistency loss to total loss if enabled
    if consistency_loss is not None:
        loss += config.consistency_weight * consistency_loss
```

This implementation shows that the graph construction approach is fully compatible with frame averaging, as the key workflow is:

1. Convert structure to graph (using CGCNN-style atom features) 
2. Generate multiple reference frames (via 2D or 3D frame averaging)
3. Process each frame through the same model 
4. Average predictions across frames
5. Optionally add consistency loss to enhance equivariance

This demonstrates that our enhanced graph construction (with the custom element feature dictionary) works seamlessly with frame averaging, providing a unique path to approximate SO(2)/SO(3) equivariance while preserving rich atom features for better property prediction.