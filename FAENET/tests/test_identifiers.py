#!/usr/bin/env python
"""
Test script to verify unique identifier handling in the dataset loader.
"""
import os
import sys
import structlog
from pathlib import Path

# Add the parent directory to sys.path to allow importing the faenet module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from faenet.dataset import SlabDataset

# Initialize structlog
logger = structlog.get_logger()

def test_identifiers():
    """Test unique identifier creation in EnhancedSlabDataset."""
    # Path to augmented data (adjust as needed)
    data_path = os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 
        "..", "processed_data", "augmented_DFT_data.csv"
    ))
    
    # Check if the test data exists
    if not os.path.exists(data_path):
        logger.warning("test_data_not_found", path=data_path)
        return
    
    logger.info("testing_identifiers", data_path=str(data_path))
    
    # Create dataset
    dataset = SlabDataset(
        data_source=data_path,
        structure_col="slab",
        target_props=["WF_top", "WF_bottom"]
    )
    
    # Log the first few identifiers
    logger.info("sample_identifiers", identifiers=dataset.file_list[:5])
    
    # Check for uniqueness
    unique_ids = set(dataset.file_list)
    logger.info("identifier_counts", 
               total=len(dataset.file_list), 
               unique=len(unique_ids))
    
    # Check for duplicates
    if len(unique_ids) != len(dataset.file_list):
        logger.warning("duplicate_identifiers_found", 
                      duplicates=len(dataset.file_list) - len(unique_ids))
        
        # Find the duplicates
        from collections import Counter
        counter = Counter(dataset.file_list)
        duplicates = [item for item, count in counter.items() if count > 1]
        logger.warning("duplicate_values", duplicates=duplicates[:5])
    else:
        logger.info("all_identifiers_unique")
    
    # Test if flipped slabs have different identifiers
    if 'flipped' in dataset.df.columns:
        flipped_count = (dataset.df['flipped'] == 'flipped').sum()
        logger.info("flipped_slabs_found", count=flipped_count)
        
        # Check a sample of flipped/non-flipped pairs
        if flipped_count > 0:
            # Get a sample original slab
            sample_row = dataset.df[dataset.df['flipped'] != 'flipped'].iloc[0]
            
            # Try to find its flipped counterpart
            mpid = sample_row.get('mpid')
            miller = sample_row.get('miller')
            term = sample_row.get('term')
            
            if all(x is not None for x in [mpid, miller, term]):
                # Find matching flipped row
                flipped_rows = dataset.df[
                    (dataset.df['mpid'] == mpid) & 
                    (dataset.df['miller'] == miller) & 
                    (dataset.df['term'] == term) & 
                    (dataset.df['flipped'] == 'flipped')
                ]
                
                if len(flipped_rows) > 0:
                    original_id = f"{mpid}_{miller}_{term}"
                    flipped_id = f"{mpid}_{miller}_{term}_flipped"
                    
                    logger.info("identifier_comparison",
                               original_id=original_id,
                               flipped_id=flipped_id,
                               in_file_list=[id in dataset.file_list for id in [original_id, flipped_id]])
    
    logger.info("identifier_test_complete")
    return dataset.file_list

if __name__ == "__main__":
    test_identifiers()