import os
import torch
import pandas as pd
import numpy as np
import pickle
from typing import List, Optional, Union, Dict, Tuple
from tqdm import tqdm
import structlog
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import torch_geometric.transforms as T

from pymatgen.core import Structure
from faenet.frame_averaging import frame_averaging_3D, frame_averaging_2D
from faenet.graph_construction import structure_dict_to_graph, structure_to_graph

# Configure structlog
logger = structlog.get_logger()

class SlabDataset(Dataset):
    """
    Dataset for crystal slab structures with CSV or direct file loading.
    
    Supports both:
    1. Loading structures from POSCAR/VASP files
    2. Loading structures from CSV with structure dictionaries
    
    Key features:
    - Unique identifier handling (mpid_miller_term) for each slab
    - Handles flipped slabs with appropriate identifiers 
    - Integrates graph construction with customizable cutoffs
    - Supports frame averaging for symmetry learning
    """
    
    def __init__(
        self, 
        data_source,
        structure_col=None,
        target_props=None,
        cutoff=6.0,
        max_neighbors=40,
        pbc=True,
        frame_averaging=None,
        fa_method="all",
        transform=None,
        limit=None
    ):
        """
        Args:
            data_source: Directory with structure files or path to CSV
            structure_col: Column name in CSV containing structure data
            target_props: List of target properties or dict mapping props to files
            cutoff: Cutoff distance for neighbors
            max_neighbors: Maximum neighbors per atom
            pbc: Use periodic boundary conditions
            frame_averaging: Frame averaging method (None, "2D", "3D", "DA")
            fa_method: Frame averaging technique (all, det, random, se3-*)
            transform: Transform to apply
            limit: Limit number of structures to process
        """
        self.data_source = data_source
        self.structure_col = structure_col
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.pbc = pbc
        self.frame_averaging = frame_averaging
        self.fa_method = fa_method
        self.transform = transform
        self.limit = limit
        
        # Determine source type and load accordingly
        if os.path.isdir(data_source):
            self.source_type = "directory"
            self.file_list = [f for f in os.listdir(data_source) 
                             if f.endswith('.vasp') or f.endswith('.poscar')]
            self.file_list.sort()
            
            # Target properties from files
            self.target_properties = {}
            if target_props:
                if isinstance(target_props, dict):
                    for prop_name, prop_file in target_props.items():
                        file_path = os.path.join(data_source, prop_file)
                        if os.path.exists(file_path):
                            with open(file_path, 'r') as f:
                                values = [float(line.strip()) for line in f if line.strip()]
                            self.target_properties[prop_name] = values
                            
                            if len(values) != len(self.file_list):
                                logger.warn("property_length_mismatch", 
                                           property=prop_name, 
                                           property_count=len(values), 
                                           structure_count=len(self.file_list))
        else:
            # Assume CSV format
            self.source_type = "csv"
            if not structure_col:
                raise ValueError("structure_col must be provided when using CSV data source")
            
            # Load CSV
            self.df = pd.read_csv(data_source)
            if self.limit:
                self.df = self.df.head(self.limit)
            
            # Check if columns for generating unique identifiers are available
            id_columns = ['mpid', 'miller', 'term']
            has_id_columns = all(col in self.df.columns for col in id_columns)
            has_flipped = 'flipped' in self.df.columns
            
            if has_id_columns:
                # Create unique identifiers similar to Comformer approach
                if has_flipped:
                    logger.info("Using mpid+miller+term+flipped as unique identifiers")
                    # Create combined identifier including flipped status
                    self.df['uid'] = self.df.apply(
                        lambda row: f"{row['mpid']}_{row['miller']}_{row['term']}" + 
                                    (f"_flipped" if row['flipped'] == 'flipped' else ""), 
                        axis=1
                    )
                else:
                    logger.info("Using mpid+miller+term as unique identifiers")
                    # Create combined identifier without flipped status
                    self.df['uid'] = self.df.apply(
                        lambda row: f"{row['mpid']}_{row['miller']}_{row['term']}", 
                        axis=1
                    )
                
                # Use the unique identifiers for file_list
                self.file_list = self.df['uid'].tolist()
            else:
                # Fall back to index-based identifiers
                logger.info("Using DataFrame indices as identifiers (mpid/miller/term not found)")
                self.file_list = self.df.index.astype(str).tolist()  # Convert to strings for consistency
            
            # Target properties from columns
            self.target_properties = {}
            if target_props:
                if isinstance(target_props, list):
                    for prop in target_props:
                        if prop in self.df.columns:
                            self.target_properties[prop] = True
        
        # Process data or wait for lazy loading
        self.data_objects = None
    
    def _process_structure_file(self, file_path):
        """Process a structure file to create a graph."""
        # Read file, convert to pymatgen Structure, create graph
        try:
            structure = Structure.from_file(file_path)
            return structure_to_graph(structure, self.cutoff, self.max_neighbors, self.pbc)
        except Exception as e:
            logger.error("structure_processing_error", file_path=file_path, error=str(e))
            return None
    
    def _process_structure_dict(self, structure_dict):
        """Process a structure dictionary to create a graph."""
        try:
            return structure_dict_to_graph(structure_dict, self.cutoff, self.max_neighbors, self.pbc)
        except Exception as e:
            logger.error("structure_dict_processing_error", error=str(e))
            return None
    
    def __len__(self):
        """Return the number of structures."""
        return len(self.file_list)
    
    def __getitem__(self, idx):
        """Get a single data point."""
        data = None
        
        # Load structure based on source type
        if self.source_type == "directory":
            file_path = os.path.join(self.data_source, self.file_list[idx])
            data = self._process_structure_file(file_path)
            
            # Add target properties
            for prop_name, values in self.target_properties.items():
                if idx < len(values):
                    value = values[idx]
                    
                    # Store original value for reference
                    setattr(data, f"{prop_name}_orig", torch.tensor(value, dtype=torch.float))
                    
                    # Apply scaling if enabled and scaler exists
                    if hasattr(self, 'use_scaling') and self.use_scaling and hasattr(self, 'scalers') and prop_name in self.scalers:
                        value = float(self.scalers[prop_name].transform([[value]])[0][0])
                    
                    # Store (potentially scaled) value
                    setattr(data, prop_name, torch.tensor(value, dtype=torch.float))
        else:
            # CSV source
            row = self.df.iloc[idx]
            data = self._process_structure_dict(row[self.structure_col])
            
            # Add target properties
            for prop_name in self.target_properties:
                if prop_name in self.df.columns:
                    value = float(row[prop_name])
                    
                    # Store original value for reference
                    setattr(data, f"{prop_name}_orig", torch.tensor(value, dtype=torch.float))
                    
                    # Apply scaling if enabled and scaler exists
                    if hasattr(self, 'use_scaling') and self.use_scaling and hasattr(self, 'scalers') and prop_name in self.scalers:
                        value = float(self.scalers[prop_name].transform([[value]])[0][0])
                    
                    # Store (potentially scaled) value
                    setattr(data, prop_name, torch.tensor(value, dtype=torch.float))
        
        # Apply frame averaging if requested
        if self.frame_averaging and data is not None:
            if self.frame_averaging == "3D":
                fa_pos, fa_cell, fa_rot = frame_averaging_3D(
                    data.pos, data.cell, self.fa_method
                )
            elif self.frame_averaging == "2D":
                fa_pos, fa_cell, fa_rot = frame_averaging_2D(
                    data.pos, data.cell, self.fa_method
                )
            else:
                raise ValueError(f"Unknown frame averaging method: {self.frame_averaging}")
            
            # Store frame averaging results
            data.fa_pos = fa_pos
            data.fa_cell = fa_cell
            data.fa_rot = fa_rot
        
        # Apply transform if provided
        if self.transform:
            data = self.transform(data)
        
        return data

# For backward compatibility, EnhancedSlabDataset is now an alias for SlabDataset
EnhancedSlabDataset = SlabDataset

# Enhanced dataloader creation function
def create_dataloader(
    data_source,
    structure_col=None,
    target_props=None,
    cutoff=6.0,
    max_neighbors=40,
    pbc=True,
    frame_averaging=None,
    fa_method="all",
    batch_size=32,
    shuffle=True,
    num_workers=0,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=42,
    use_property_scaling=True,
    **kwargs
):
    """Create train, validation, and test dataloaders with optional property scaling.
    
    This function:
    1. Creates a dataset with the given parameters
    2. Fits property scalers on the training split if scaling is enabled
    3. Returns dataloaders for train/val/test splits
    
    Args:
        data_source: Path to data file or directory
        structure_col: Column name containing structure data
        target_props: List of target properties to predict
        cutoff: Cutoff distance for neighbor finding
        max_neighbors: Maximum number of neighbors per atom
        pbc: Whether to use periodic boundary conditions
        frame_averaging: Type of frame averaging to use (None, "2D", "3D")
        fa_method: Frame averaging method ("all", "det", "random")
        batch_size: Batch size for dataloaders
        shuffle: Whether to shuffle the data
        num_workers: Number of workers for dataloading
        train_ratio: Ratio of data to use for training
        val_ratio: Ratio of data to use for validation
        test_ratio: Ratio of data to use for testing
        seed: Random seed for reproducibility
        use_property_scaling: Whether to apply standardization to properties
        **kwargs: Additional arguments
    """
    from torch.utils.data import random_split, Subset
    from torch_geometric.loader import DataLoader
    
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create dataset
    dataset = SlabDataset(
        data_source=data_source,
        structure_col=structure_col,
        target_props=target_props,
        cutoff=cutoff,
        max_neighbors=max_neighbors,
        pbc=pbc,
        frame_averaging=frame_averaging,
        fa_method=fa_method
    )
    
    # Split dataset
    n_total = len(dataset)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val
    
    # Get indices for each split
    indices = list(range(n_total))
    if shuffle:
        np.random.shuffle(indices)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train+n_val]
    test_indices = indices[n_train+n_val:]
    
    # Check if we should apply property scaling
    use_scaling = kwargs.get('use_property_scaling', True)
    
    # Create property scalers using training data if scaling is enabled
    if isinstance(target_props, list) and len(target_props) > 0 and use_scaling:
        scalers = {}
        
        # Extract data values from training set for each property
        train_values = {prop: [] for prop in target_props}
        
        if dataset.source_type == "csv":
            # For CSV source, extract values directly from DataFrame
            for idx in train_indices:
                row = dataset.df.iloc[idx]
                for prop in target_props:
                    if prop in dataset.df.columns:
                        train_values[prop].append(float(row[prop]))
        else:
            # For directory source, extract from target_properties dict
            for prop in target_props:
                if prop in dataset.target_properties:
                    values = dataset.target_properties[prop]
                    for idx in train_indices:
                        if idx < len(values):
                            train_values[prop].append(values[idx])
        
        # Fit StandardScalers on training data
        for prop in target_props:
            if prop in train_values and len(train_values[prop]) > 0:
                scaler = StandardScaler()
                scaler.fit(np.array(train_values[prop]).reshape(-1, 1))
                scalers[prop] = scaler
                
                # Log mean and std for each property
                logger.info("property_scaled", 
                          property=prop, 
                          mean=float(scaler.mean_[0]), 
                          std=float(scaler.scale_[0]))
        
        # Store scalers on dataset for later use
        dataset.scalers = scalers
        dataset.use_scaling = True
        logger.info("property_scaling_enabled")
    else:
        dataset.scalers = {}
        dataset.use_scaling = False
        if not use_scaling:
            logger.info("property_scaling_disabled")
        elif not isinstance(target_props, list) or len(target_props) == 0:
            logger.info("property_scaling_disabled", reason="No target properties specified")
    
    # Create subset datasets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Always shuffle training data
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader, dataset

# Maintain backward compatibility
def apply_frame_averaging_to_batch(batch, fa_method="all", dimension="3D"):
    """Apply frame averaging to a batch of data
    
    Args:
        batch: PyTorch Geometric batch
        fa_method: Frame averaging method
        dimension: Dimension of frame averaging ("2D" or "3D")
        
    Returns:
        batch: Updated batch with fa_pos, fa_cell, fa_rot attributes
    """
    from faenet.frame_averaging import frame_averaging_3D, frame_averaging_2D
    
    # Apply frame averaging based on dimension
    if dimension == "2D":
        fa_pos, fa_cell, fa_rot = frame_averaging_2D(batch.pos, batch.cell, fa_method)
    else:
        fa_pos, fa_cell, fa_rot = frame_averaging_3D(batch.pos, batch.cell, fa_method)
    
    # Store frame averaging results
    batch.fa_pos = fa_pos
    batch.fa_cell = fa_cell
    batch.fa_rot = fa_rot
    
    return batch