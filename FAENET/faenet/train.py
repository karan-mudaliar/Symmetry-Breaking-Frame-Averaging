import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils.data import random_split

from faenet.dataset import EnhancedSlabDataset, apply_frame_averaging_to_batch
from faenet.faenet import FAENet
from faenet.config import get_config, Config


def train(model, train_loader, val_loader, device, config):
    """Train the FAENet model
    
    Args:
        model: FAENet model
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to run on (cpu or cuda)
        config: Configuration object
    """
    # Get parameters directly from config
    lr = config.learning_rate
    epochs = config.epochs
    frame_averaging = config.frame_averaging
    fa_method = config.fa_method
    checkpoint_interval = config.checkpoint_interval
    output_dir = config.output_dir
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Track best model
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        # Training
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            batch = batch.to(device)
            
            # Apply frame averaging
            if frame_averaging:
                batch = apply_frame_averaging_to_batch(batch, fa_method, frame_averaging)
                
                # Process all frames at once for better equivariance
                # Create a batch containing all frames
                from torch_geometric.data import Batch
                
                # Save original positions and cell
                original_pos = batch.pos.clone()
                original_cell = batch.cell.clone() if hasattr(batch, 'cell') and isinstance(batch.cell, torch.Tensor) else batch.cell
                
                # Create a list of batches, one for each frame
                all_batches = []
                for i in range(len(batch.fa_pos)):
                    # Create a copy of the batch for this frame
                    frame_batch = batch.clone()
                    # Update positions and cell for this frame
                    frame_batch.pos = batch.fa_pos[i]
                    if hasattr(batch, 'cell') and batch.cell is not None and hasattr(batch, 'fa_cell'):
                        frame_batch.cell = batch.fa_cell[i]
                    all_batches.append(frame_batch)
                
                # Create a single large batch with all frames
                combined_batch = Batch.from_data_list(all_batches)
                
                # Forward pass with all frames at once
                all_preds = model(combined_batch)
                
                # Split predictions by property and average them
                e_all = [[] for _ in model.output_properties]
                
                # Split predictions by frame
                batch_size = batch.num_graphs
                num_frames = len(batch.fa_pos)
                
                for prop_idx, prop in enumerate(model.output_properties):
                    # Get predictions for this property
                    prop_preds = all_preds[prop]
                    
                    # Split by frame
                    for i in range(num_frames):
                        start_idx = i * batch_size
                        end_idx = (i + 1) * batch_size
                        e_all[prop_idx].append(prop_preds[start_idx:end_idx])
                
                # Restore original positions and cell
                batch.pos = original_pos
                if hasattr(batch, 'cell') and batch.cell is not None:
                    batch.cell = original_cell
                
                # Calculate loss by averaging predictions across frames
                loss = 0
                for prop_idx, prop in enumerate(model.output_properties):
                    if hasattr(batch, prop):
                        # Average predictions across frames 
                        avg_pred = sum(e_all[prop_idx]) / len(e_all[prop_idx])
                        targets = getattr(batch, prop)
                        
                        # Ensure tensors have compatible shapes for loss computation
                        if avg_pred.dim() != targets.dim():
                            if avg_pred.dim() > targets.dim():
                                avg_pred = avg_pred.squeeze()
                            else:
                                targets = targets.unsqueeze(-1)
                        
                        # Calculate loss for this property
                        prop_loss = criterion(avg_pred, targets)
                        loss += prop_loss
            
            else:
                # Standard forward pass without frame averaging
                preds = model(batch)
                
                # Calculate loss
                loss = 0
                for prop in model.output_properties:
                    if hasattr(batch, prop):
                        # Get predictions and targets, ensure they have the same shape
                        model_preds = preds[prop]
                        targets = getattr(batch, prop)
                        
                        # Make sure both are the same shape
                        if model_preds.dim() != targets.dim():
                            if model_preds.dim() > targets.dim():
                                model_preds = model_preds.squeeze()
                            else:
                                targets = targets.unsqueeze(-1)
                        
                        prop_loss = criterion(model_preds, targets)
                        loss += prop_loss
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                batch = batch.to(device)
                
                # Get frame averaging settings from config
                frame_averaging = config.frame_averaging
                fa_method = config.fa_method
                if frame_averaging:
                    batch = apply_frame_averaging_to_batch(batch, fa_method, frame_averaging)
                    
                    # Process all frames at once for better equivariance
                    from torch_geometric.data import Batch
                    
                    # Save original positions and cell
                    original_pos = batch.pos.clone()
                    original_cell = batch.cell.clone() if hasattr(batch, 'cell') and isinstance(batch.cell, torch.Tensor) else batch.cell
                    
                    # Create a list of batches, one for each frame
                    all_batches = []
                    for i in range(len(batch.fa_pos)):
                        # Create a copy of the batch for this frame
                        frame_batch = batch.clone()
                        # Update positions and cell for this frame
                        frame_batch.pos = batch.fa_pos[i]
                        if hasattr(batch, 'cell') and batch.cell is not None and hasattr(batch, 'fa_cell'):
                            frame_batch.cell = batch.fa_cell[i]
                        all_batches.append(frame_batch)
                    
                    # Create a single large batch with all frames
                    combined_batch = Batch.from_data_list(all_batches)
                    
                    # Forward pass with all frames at once
                    all_preds = model(combined_batch)
                    
                    # Split predictions by property and average them
                    e_all = [[] for _ in model.output_properties]
                    
                    # Split predictions by frame
                    batch_size = batch.num_graphs
                    num_frames = len(batch.fa_pos)
                    
                    for prop_idx, prop in enumerate(model.output_properties):
                        # Get predictions for this property
                        prop_preds = all_preds[prop]
                        
                        # Split by frame
                        for i in range(num_frames):
                            start_idx = i * batch_size
                            end_idx = (i + 1) * batch_size
                            e_all[prop_idx].append(prop_preds[start_idx:end_idx])
                    
                    # Restore original positions and cell
                    batch.pos = original_pos
                    if hasattr(batch, 'cell') and batch.cell is not None:
                        batch.cell = original_cell
                    
                    # Calculate validation loss
                    batch_loss = 0
                    for prop_idx, prop in enumerate(model.output_properties):
                        if hasattr(batch, prop):
                            # Average predictions across frames
                            avg_pred = sum(e_all[prop_idx]) / len(e_all[prop_idx])
                            targets = getattr(batch, prop)
                            
                            # Ensure tensors have compatible shapes for loss computation
                            if avg_pred.dim() != targets.dim():
                                if avg_pred.dim() > targets.dim():
                                    avg_pred = avg_pred.squeeze()
                                else:
                                    targets = targets.unsqueeze(-1)
                            
                            prop_val_loss = criterion(avg_pred, targets)
                            batch_loss += prop_val_loss
                else:
                    # Standard forward pass
                    preds = model(batch)
                    
                    # Calculate loss
                    batch_loss = 0
                    for prop in model.output_properties:
                        if hasattr(batch, prop):
                            # Get predictions and targets, ensure they have the same shape
                            model_preds = preds[prop]
                            targets = getattr(batch, prop)
                            
                            # Make sure both are the same shape
                            if model_preds.dim() != targets.dim():
                                if model_preds.dim() > targets.dim():
                                    model_preds = model_preds.squeeze()
                                else:
                                    targets = targets.unsqueeze(-1)
                            
                            prop_val_loss = criterion(model_preds, targets)
                            batch_loss += prop_val_loss
                
                val_loss += batch_loss.item()
            
            val_loss /= len(val_loader)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print results
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))
            print(f"Saved new best model with validation loss: {best_val_loss:.6f}")
        
        # Save checkpoint
        if (epoch + 1) % checkpoint_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}.pt"))
        
        # Always save latest checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, os.path.join(output_dir, "checkpoint.pt"))


def run_inference(model, data_loader, device, config, output_file):
    """Run inference on a dataset and save predictions
    
    Args:
        model: FAENet model
        data_loader: Data loader for inference
        device: Device to run on
        config: Configuration object
        output_file: File to save predictions to
    """
    model.eval()
    
    # Store predictions and file identifiers
    results = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Running inference")):
            batch = batch.to(device)
            
            # Get original file identifiers
            batch_size = config.batch_size
            file_indices = data_loader.dataset.indices[batch_idx * batch_size:
                                                     min((batch_idx + 1) * batch_size, 
                                                         len(data_loader.dataset))]
            
            file_names = [data_loader.dataset.dataset.file_list[idx] for idx in file_indices]
            
            # Apply frame averaging if requested
            frame_averaging = config.frame_averaging
            fa_method = config.fa_method
            if frame_averaging:
                batch = apply_frame_averaging_to_batch(batch, fa_method, frame_averaging)
                
                # Process all frames at once for better equivariance and efficiency
                from torch_geometric.data import Batch
                
                # Save original positions and cell
                original_pos = batch.pos.clone()
                original_cell = batch.cell.clone() if hasattr(batch, 'cell') and isinstance(batch.cell, torch.Tensor) else batch.cell
                
                # Create a list of batches, one for each frame
                all_batches = []
                for i in range(len(batch.fa_pos)):
                    # Create a copy of the batch for this frame
                    frame_batch = batch.clone()
                    # Update positions and cell for this frame
                    frame_batch.pos = batch.fa_pos[i]
                    if hasattr(batch, 'cell') and batch.cell is not None and hasattr(batch, 'fa_cell'):
                        frame_batch.cell = batch.fa_cell[i]
                    all_batches.append(frame_batch)
                
                # Create a single large batch with all frames
                combined_batch = Batch.from_data_list(all_batches)
                
                # Forward pass with all frames at once
                all_preds = model(combined_batch)
                
                # Split predictions by property and organize by frame
                e_all = {prop: [] for prop in model.output_properties}
                
                # Split predictions by frame
                batch_size = batch.num_graphs
                num_frames = len(batch.fa_pos)
                
                for prop in model.output_properties:
                    # Get predictions for this property
                    prop_preds = all_preds[prop]
                    
                    # Split by frame
                    for i in range(num_frames):
                        start_idx = i * batch_size
                        end_idx = (i + 1) * batch_size
                        e_all[prop].append(prop_preds[start_idx:end_idx])
                
                # Restore original positions and cell
                batch.pos = original_pos
                if hasattr(batch, 'cell') and batch.cell is not None:
                    batch.cell = original_cell
                
                # Average predictions across frames
                final_preds = {}
                for prop in model.output_properties:
                    final_preds[prop] = (sum(e_all[prop]) / len(e_all[prop])).cpu().numpy()
            else:
                # Standard forward pass
                preds = model(batch)
                
                # Convert predictions to numpy
                final_preds = {}
                for prop in model.output_properties:
                    final_preds[prop] = preds[prop].cpu().numpy()
            
            # Store predictions for each file
            for i, fname in enumerate(file_names):
                result = {"file_name": fname}
                
                # Add predictions
                for prop in final_preds:
                    # For graph-level properties (scalar)
                    result[prop] = float(final_preds[prop][i][0])
                
                # Add ground truth if available
                for prop in model.output_properties:
                    if hasattr(batch, prop):
                        # Handle different tensor shapes correctly
                        target_tensor = getattr(batch, prop)
                        if target_tensor.dim() == 0:
                            # 0-dim tensor
                            result[f"{prop}_true"] = float(target_tensor.item())
                        elif target_tensor.dim() == 1:
                            # 1-dim tensor
                            result[f"{prop}_true"] = float(target_tensor[i].item())
                        else:
                            # 2-dim or higher tensor
                            result[f"{prop}_true"] = float(target_tensor[i][0].cpu().numpy())
                
                
                results.append(result)
    
    # Save results to file
    import json
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved inference results for {len(results)} structures to {output_file}")


def train_faenet(
    data_path,
    structure_col=None,
    target_properties=None,
    output_dir="output",
    frame_averaging=None,
    fa_method="all",
    cutoff=6.0,
    max_neighbors=40,
    batch_size=32,
    epochs=100,
    learning_rate=0.001,
    seed=42,
    device=None,
    num_workers=4,
    pbc=True,
    test_ratio=0.1,
    val_ratio=0.1,
    **model_kwargs
):
    """
    Train a FAENet model with simplified interface
    
    Args:
        data_path: Path to data file or directory
        structure_col: Column name containing structure data (for CSV)
        target_properties: List or dict of target properties
        output_dir: Directory to save outputs
        frame_averaging: Frame averaging method ("2D", "3D", or None)
        fa_method: Frame averaging technique ("all", "det", "random")
        cutoff: Cutoff distance for neighbor finding
        max_neighbors: Maximum number of neighbors per atom
        batch_size: Batch size for training
        epochs: Number of training epochs
        learning_rate: Learning rate
        seed: Random seed
        device: Device to use ("cuda" or "cpu")
        num_workers: Number of workers for data loading
        pbc: Use periodic boundary conditions
        test_ratio: Ratio of data to use for testing
        val_ratio: Ratio of data to use for validation
        **model_kwargs: Additional arguments to pass to FAENet model
    
    Returns:
        model: Trained FAENet model
        test_loader: DataLoader for test set
    """
    # Set up device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create config object
    config = Config(
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        seed=seed,
        num_workers=num_workers,
        test_ratio=test_ratio,
        val_ratio=val_ratio,
        frame_averaging=frame_averaging,
        fa_method=fa_method,
        cutoff=cutoff,
        output_dir=output_dir
    )
    
    # Create dataset
    print(f"Loading data from {data_path}")
    dataset = EnhancedSlabDataset(
        data_source=data_path,
        structure_col=structure_col,
        target_props=target_properties,
        cutoff=cutoff,
        max_neighbors=max_neighbors,
        pbc=pbc,
        frame_averaging=frame_averaging,
        fa_method=fa_method
    )
    
    # Split into train, validation, and test sets
    test_size = int(len(dataset) * test_ratio)
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size - test_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
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
    
    # Determine output properties from target_properties
    if isinstance(target_properties, list):
        output_properties = target_properties
    elif isinstance(target_properties, dict):
        output_properties = list(target_properties.keys())
    else:
        output_properties = ["energy"]
    
    # Initialize model
    model = FAENet(
        cutoff=cutoff,
        output_properties=output_properties,
        **model_kwargs
    ).to(device)
    
    # Print model summary
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model initialized with {num_params:,} parameters")
    print(f"Using device: {device}")
    print(f"Training on {train_size} samples, validating on {val_size} samples, testing on {test_size} samples")
    
    # Train model
    train(model, train_loader, val_loader, device, config)
    print("Training complete!")
    
    # Load best model
    best_model_path = os.path.join(output_dir, "best_model.pt")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"Loaded best model from {best_model_path}")
    
    # Run inference on test set
    inference_output = os.path.join(output_dir, "predictions.json")
    run_inference(model, test_loader, device, config, inference_output)
    
    return model, test_loader


def main():
    # Get configuration
    config = get_config()
    
    # Set random seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Prepare target properties
    target_props = {}
    if config.prop_files:
        for prop, file in zip(config.target_properties, config.prop_files):
            target_props[prop] = file
    
    # Train model
    train_faenet(
        data_path=config.data_path,
        structure_col=config.structure_col,
        target_properties=config.target_properties,
        output_dir=config.output_dir,
        frame_averaging=config.frame_averaging,
        fa_method=config.fa_method,
        cutoff=config.cutoff,
        max_neighbors=config.max_neighbors,
        batch_size=config.batch_size,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        seed=config.seed,
        device=config.device,
        num_workers=config.num_workers,
        num_gaussians=config.num_gaussians,
        hidden_channels=config.hidden_channels,
        num_filters=config.num_filters,
        num_interactions=config.num_interactions,
        dropout=config.dropout
    )
    
    
    
if __name__ == "__main__":
    main()