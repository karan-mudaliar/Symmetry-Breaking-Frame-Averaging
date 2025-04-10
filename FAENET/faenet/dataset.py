"""
Dataset for crystal slabs with properties.
Integrates Comformer-inspired graph construction with frame averaging.
Implements proper standardization for regression tasks.
"""

import os
import torch
import pandas as pd
import numpy as np
from typing import List, Optional, Union, Dict, Tuple
from tqdm import tqdm
import structlog
import pickle

import torch
from torch.utils.data import Dataset, random_split, Subset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T

from pymatgen.core import Structure
from faenet.frame_averaging import frame_averaging_3D, frame_averaging_2D
from faenet.graph_construction import structure_dict_to_graph, structure_to_graph

# Configure structlog
logger = structlog.get_logger()

class Standardizer:
    """Class to handle standardization of regression targets.
    
    Computes mean and standard deviation on training data,
    then applies the same transformation to validation and test data.
    """
    
    def __init__(self, property_names=None):
        """Initialize the standardizer.
        
        Args:
            property_names: List of property names to standardize
        """
        self.property_names = property_names or []
        self.mean = {}
        self.std = {}
        self.is_fitted = False
        
    def fit(self, data_values):
        """Compute mean and standard deviation from training data.
        
        Args:
            data_values: Dictionary mapping property names to values
        """
        for prop in self.property_names:
            if prop in data_values and len(data_values[prop]) > 0:
                values = np.array(data_values[prop])
                self.mean[prop] = float(np.mean(values))
                self.std[prop] = float(np.std(values))
                
                # Handle zero std
                if self.std[prop] == 0 or np.isnan(self.std[prop]):
                    self.std[prop] = 1.0
                    
                logger.info("computed_standardization_stats", 
                           property=prop,
                           mean=self.mean[prop], 
                           std=self.std[prop])
            else:
                logger.warn("skipping_standardization_for_missing_property", property=prop)
                
        self.is_fitted = True
        return self
    
    def transform(self, value, property_name):
        """Apply standardization to a value.
        
        Args:
            value: Value to standardize
            property_name: Name of the property
            
        Returns:
            float: Standardized value
        """
        if not self.is_fitted:
            logger.warn("standardizer_not_fitted", property=property_name)
            return value
            
        if property_name not in self.mean:
            logger.warn("property_not_in_standardizer", property=property_name)
            return value
            
        return (value - self.mean[property_name]) / self.std[property_name]
    
    def inverse_transform(self, value, property_name):
        """Undo standardization for a value.
        
        Args:
            value: Standardized value
            property_name: Name of the property
            
        Returns:
            float: Original-scale value
        """
        if not self.is_fitted or property_name not in self.mean:
            return value
            
        return value * self.std[property_name] + self.mean[property_name]
    
    def get_params(self):
        """Get the standardization parameters.
        
        Returns:
            dict: Dictionary with mean and std for each property
        """
        return {
            'mean': self.mean,
            'std': self.std,
            'is_fitted': self.is_fitted
        }
    
    def load_params(self, params):
        """Load standardization parameters.
        
        Args:
            params: Dictionary with mean and std
        """
        self.mean = params.get('mean', {})
        self.std = params.get('std', {})
        self.is_fitted = params.get('is_fitted', False)
        return self


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
    - Properly standardizes target properties for regression
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
        limit=None,
        atom_features="cgcnn",
        standardize=True,
        standardizer=None
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
            atom_features: Type of atom features ("one_hot" or "cgcnn")
            standardize: Whether to standardize target properties
            standardizer: Pre-fitted standardizer (for val/test sets)
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
        self.atom_features = atom_features
        self.standardize = standardize
        
        # Initialize or use provided standardizer
        if standardizer is not None:
            self.standardizer = standardizer
        else:
            self.standardizer = Standardizer(target_props if isinstance(target_props, list) else [])
        
        # For preserving raw values (before standardization)
        self.raw_values = {}
        
        # Determine source type and load accordingly
        if os.path.isdir(data_source):
            self.source_type = "directory"
            self.file_list = [f for f in os.listdir(data_source) 
                             if f.endswith('.vasp') or f.endswith('.poscar')]
            self.file_list.sort()
            
            # Target properties from files
            self.target_properties = {}
            self.property_values = {}
            
            if target_props:
                if isinstance(target_props, dict):
                    for prop_name, prop_file in target_props.items():
                        file_path = os.path.join(data_source, prop_file)
                        if os.path.exists(file_path):
                            with open(file_path, 'r') as f:
                                values = [float(line.strip()) for line in f if line.strip()]
                            self.target_properties[prop_name] = values
                            self.property_values[prop_name] = values
                            
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
            self.property_values = {}
            
            if target_props:
                if isinstance(target_props, list):
                    for prop in target_props:
                        if prop in self.df.columns:
                            self.target_properties[prop] = True
                            # Store property values for standardization
                            self.property_values[prop] = self.df[prop].values.tolist()
        
        # If this is a training set and we should standardize, fit the standardizer now
        if standardize and standardizer is None:
            self.standardizer.fit(self.property_values)
            logger.info("standardizer_fitted_from_dataset")
        
        # Process data or wait for lazy loading
        self.data_objects = None
    
    def _process_structure_file(self, file_path):
        """Process a structure file to create a graph."""
        # Read file, convert to pymatgen Structure, create graph
        try:
            structure = Structure.from_file(file_path)
            return structure_to_graph(structure, self.cutoff, self.max_neighbors, self.pbc, self.atom_features)
        except Exception as e:
            logger.error("structure_processing_error", file_path=file_path, error=str(e))
            return None
    
    def _process_structure_dict(self, structure_dict):
        """Process a structure dictionary to create a graph."""
        try:
            return structure_dict_to_graph(structure_dict, self.cutoff, self.max_neighbors, self.pbc, self.atom_features)
        except Exception as e:
            logger.error("structure_dict_processing_error", error=str(e))
            return None
    
    def __len__(self):
        """Return the number of structures."""
        return len(self.file_list)
    
    def get_raw_value(self, idx, prop_name):
        """Get raw (unstandardized) value for a property.
        
        Args:
            idx: Sample index
            prop_name: Property name
            
        Returns:
            float: Raw property value
        """
        if self.source_type == "directory":
            if prop_name in self.target_properties and idx < len(self.target_properties[prop_name]):
                return self.target_properties[prop_name][idx]
        else:
            if prop_name in self.df.columns:
                return self.df.iloc[idx][prop_name]
        return None
    
    def __getitem__(self, idx):
        """Get a single data point."""
        data = None
        
        # Load structure based on source type
        if self.source_type == "directory":
            file_path = os.path.join(self.data_source, self.file_list[idx])
            data = self._process_structure_file(file_path)
            
            # Add target properties (standardized if needed)
            for prop_name, values in self.target_properties.items():
                if idx < len(values):
                    raw_value = values[idx]
                    
                    # Apply standardization if enabled
                    if self.standardize and self.standardizer.is_fitted:
                        std_value = self.standardizer.transform(raw_value, prop_name)
                        logger.debug("standardizing_property", 
                                   property=prop_name,
                                   raw=raw_value,
                                   standardized=std_value)
                        value = std_value
                    else:
                        value = raw_value
                    
                    # Store as tensor attribute
                    setattr(data, prop_name, torch.tensor([value], dtype=torch.float))
                    
                    # Store original value for reference
                    if not hasattr(data, f"{prop_name}_raw"):
                        setattr(data, f"{prop_name}_raw", torch.tensor([raw_value], dtype=torch.float))
        else:
            # CSV source
            row = self.df.iloc[idx]
            data = self._process_structure_dict(row[self.structure_col])
            
            # Add target properties (standardized if needed)
            for prop_name in self.target_properties:
                if prop_name in self.df.columns:
                    raw_value = row[prop_name]
                    
                    # Apply standardization if enabled
                    if self.standardize and self.standardizer.is_fitted:
                        std_value = self.standardizer.transform(raw_value, prop_name)
                        logger.debug("standardizing_property", 
                                   property=prop_name,
                                   raw=raw_value,
                                   standardized=std_value)
                        value = std_value
                    else:
                        value = raw_value
                    
                    # Store as tensor attribute
                    setattr(data, prop_name, torch.tensor([value], dtype=torch.float))
                    
                    # Store original value for reference
                    if not hasattr(data, f"{prop_name}_raw"):
                        setattr(data, f"{prop_name}_raw", torch.tensor([raw_value], dtype=torch.float))
            
            # Add identifier if available
            if 'uid' in self.df.columns:
                data.identifier = self.df.iloc[idx]['uid']
        
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
    
    def save_standardizer(self, filepath):
        """Save standardizer parameters to file.
        
        Args:
            filepath: Path to save the standardizer
        """
        params = self.standardizer.get_params()
        with open(filepath, 'wb') as f:
            pickle.dump(params, f)
        logger.info("standardizer_saved", filepath=filepath)
    
    @classmethod
    def load_standardizer(cls, filepath):
        """Load standardizer parameters from file.
        
        Args:
            filepath: Path to load the standardizer from
            
        Returns:
            Standardizer: Loaded standardizer
        """
        with open(filepath, 'rb') as f:
            params = pickle.load(f)
        standardizer = Standardizer()
        standardizer.load_params(params)
        logger.info("standardizer_loaded", filepath=filepath)
        return standardizer


# Enhanced dataloader creation function with proper standardization
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
    atom_features="cgcnn",
    standardize=True,
    output_dir=None
):
    """Create train, validation, and test dataloaders with proper standardization.
    
    Args:
        data_source: Path to data source (CSV or directory)
        structure_col: Column name for structures in CSV
        target_props: Target properties to predict
        cutoff: Cutoff distance for neighbors
        max_neighbors: Maximum number of neighbors per atom
        pbc: Use periodic boundary conditions
        frame_averaging: Frame averaging method
        fa_method: Frame averaging technique
        batch_size: Batch size for training
        shuffle: Whether to shuffle the data
        num_workers: Number of workers for data loading
        train_ratio: Ratio of data to use for training
        val_ratio: Ratio of data to use for validation
        test_ratio: Ratio of data to use for testing
        seed: Random seed for reproducibility
        atom_features: Type of atom features to use
        standardize: Whether to standardize target properties
        output_dir: Directory to save standardization parameters
        
    Returns:
        tuple: (train_loader, val_loader, test_loader, dataset, standardizer)
    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create full dataset without standardization first
    full_dataset = SlabDataset(
        data_source=data_source,
        structure_col=structure_col,
        target_props=target_props,
        cutoff=cutoff,
        max_neighbors=max_neighbors,
        pbc=pbc,
        frame_averaging=frame_averaging,
        fa_method=fa_method,
        atom_features=atom_features,
        standardize=False  # We'll handle standardization manually
    )
    
    # Split dataset indices
    n_total = len(full_dataset)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val
    
    indices = list(range(n_total))
    
    if shuffle:
        np.random.shuffle(indices)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train+n_val]
    test_indices = indices[n_train+n_val:]
    
    # Create a standardizer based on training data only
    standardizer = None
    
    if standardize:
        train_property_values = {}
        
        # Extract property values from training set only
        for prop_name in full_dataset.property_values:
            train_property_values[prop_name] = [
                full_dataset.get_raw_value(idx, prop_name) 
                for idx in train_indices
            ]
        
        # Fit standardizer on training data
        standardizer = Standardizer(full_dataset.property_values.keys())
        standardizer.fit(train_property_values)
        
        # Save standardizer if output directory is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            standardizer_path = os.path.join(output_dir, "standardizer.pkl")
            with open(standardizer_path, 'wb') as f:
                pickle.dump(standardizer.get_params(), f)
            logger.info("standardizer_saved", path=standardizer_path)
    
    # Create the standardized datasets
    train_dataset = SlabDataset(
        data_source=data_source,
        structure_col=structure_col,
        target_props=target_props,
        cutoff=cutoff,
        max_neighbors=max_neighbors,
        pbc=pbc,
        frame_averaging=frame_averaging,
        fa_method=fa_method,
        atom_features=atom_features,
        standardize=standardize,
        standardizer=standardizer  # Use fitted standardizer
    )
    
    # Use the same standardizer for validation and test
    val_dataset = Subset(train_dataset, val_indices)
    test_dataset = Subset(train_dataset, test_indices)
    
    # Create actual train subset at the end to avoid issues with dataset copies
    train_dataset = Subset(train_dataset, train_indices)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
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
    
    return train_loader, val_loader, test_loader, full_dataset, standardizer


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