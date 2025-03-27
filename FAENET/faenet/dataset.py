"""
Dataset for crystal slabs with properties.
Integrates Comformer-inspired graph construction with frame averaging.
"""

import os
import torch
import pandas as pd
import numpy as np
from typing import List, Optional, Union, Dict, Tuple
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import torch_geometric.transforms as T

from pymatgen.core import Structure
from frame_averaging import frame_averaging_3D, frame_averaging_2D
from graph_construction import structure_dict_to_graph, structure_to_graph

class EnhancedSlabDataset(Dataset):
    """
    Dataset for crystal slab structures with CSV or direct file loading.
    
    Supports both:
    1. Loading structures from POSCAR/VASP files
    2. Loading structures from CSV with structure dictionaries
    
    Integrates Comformer-inspired graph construction with frame averaging.
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
                                print(f"Warning: {prop_name} has {len(values)} values but there are {len(self.file_list)} structures")
        else:
            # Assume CSV format
            self.source_type = "csv"
            if not structure_col:
                raise ValueError("structure_col must be provided when using CSV data source")
            
            # Load CSV
            self.df = pd.read_csv(data_source)
            if self.limit:
                self.df = self.df.head(self.limit)
                
            self.file_list = self.df.index.tolist()
            
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
            print(f"Error processing {file_path}: {e}")
            return None
    
    def _process_structure_dict(self, structure_dict):
        """Process a structure dictionary to create a graph."""
        try:
            return structure_dict_to_graph(structure_dict, self.cutoff, self.max_neighbors, self.pbc)
        except Exception as e:
            print(f"Error processing structure: {e}")
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
                    setattr(data, prop_name, torch.tensor([values[idx]], dtype=torch.float))
        else:
            # CSV source
            row = self.df.iloc[idx]
            data = self._process_structure_dict(row[self.structure_col])
            
            # Add target properties
            for prop_name in self.target_properties:
                if prop_name in self.df.columns:
                    value = row[prop_name]
                    setattr(data, prop_name, torch.tensor([value], dtype=torch.float))
        
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

# Keep original SlabDataset for backward compatibility
class SlabDataset(EnhancedSlabDataset):
    """Legacy interface for SlabDataset."""
    
    def __init__(self, data_dir, target_properties=None, transform=None):
        """Initialize with original parameters."""
        super().__init__(
            data_source=data_dir,
            target_props=target_properties,
            transform=transform
        )

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
    seed=42
):
    """Create train, validation, and test dataloaders."""
    from torch.utils.data import random_split
    from torch_geometric.loader import DataLoader
    
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create dataset
    dataset = EnhancedSlabDataset(
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
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [n_train, n_val, n_test]
    )
    
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
    
    return train_loader, val_loader, test_loader, dataset

# Maintain backward compatibility
def apply_frame_averaging_to_batch(batch, fa_method="all"):
    """Apply frame averaging to a batch of data
    
    Args:
        batch: PyTorch Geometric batch
        fa_method: Frame averaging method
        
    Returns:
        batch: Updated batch with fa_pos, fa_cell, fa_rot attributes
    """
    from frame_averaging import frame_averaging_3D
    
    # Apply frame averaging
    fa_pos, fa_cell, fa_rot = frame_averaging_3D(batch.pos, batch.cell, fa_method)
    
    # Store frame averaging results
    batch.fa_pos = fa_pos
    batch.fa_cell = fa_cell
    batch.fa_rot = fa_rot
    
    return batch