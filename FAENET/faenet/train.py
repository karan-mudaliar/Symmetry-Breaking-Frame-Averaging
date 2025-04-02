import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils.data import random_split

from dataset import EnhancedSlabDataset, apply_frame_averaging_to_batch
from faenet import FAENet
from config import get_config, get_simple_config, FAENetConfig, SimpleConfig


def get_config_param(config, nested_path, default=None):
    """Get parameter from either nested or flat config
    
    Args:
        config: Config object (either SimpleConfig or Config)
        nested_path: Path in nested config (e.g., "training.lr")
        default: Default value if parameter not found
        
    Returns:
        Parameter value
    """
    if isinstance(config, SimpleConfig):
        # For flat config, use the last part of the path
        param_name = nested_path.split('.')[-1]
        return getattr(config, param_name, default)
    else:
        # For nested config, traverse the path
        parts = nested_path.split('.')
        current = config
        for part in parts:
            if not hasattr(current, part):
                return default
            current = getattr(current, part)
        return current


def train(model, train_loader, val_loader, device, config):
    """Train the FAENet model
    
    Args:
        model: FAENet model
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to run on (cpu or cuda)
        config: Configuration object (SimpleConfig or Config)
    """
    # Get parameters from config (works with both SimpleConfig and nested Config)
    lr = get_config_param(config, "training.lr", 0.001)
    epochs = get_config_param(config, "training.epochs", 100)
    frame_averaging = get_config_param(config, "training.frame_averaging", None)
    fa_method = get_config_param(config, "training.fa_method", "all")
    force_weight = get_config_param(config, "training.force_weight", 0.1)
    checkpoint_interval = get_config_param(config, "training.checkpoint_interval", 10)
    output_dir = get_config_param(config, "output_dir", "./outputs")
    
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
                batch = apply_frame_averaging_to_batch(batch, fa_method)
                
                # Process with frame averaging
                e_all, f_all = [], []
                for i in range(len(batch.fa_pos)):
                    # Set current frame
                    original_pos = batch.pos.clone()
                    batch.pos = batch.fa_pos[i]
                    
                    # Forward pass
                    preds = model(batch)
                    
                    # Collect predictions
                    for prop in model.output_properties:
                        if i == 0:
                            e_all.append([preds[prop]])
                        else:
                            e_all[-1].append(preds[prop])
                    
                    # Transform forces if needed
                    if model.regress_forces and "forces" in preds:
                        fa_rot = torch.repeat_interleave(
                            batch.fa_rot[i], batch.natoms, dim=0
                        ).to(device)
                        forces = (
                            preds["forces"]
                            .view(-1, 1, 3)
                            .bmm(fa_rot.transpose(1, 2))
                            .view(-1, 3)
                        )
                        f_all.append(forces)
                    
                    # Restore original positions
                    batch.pos = original_pos
                
                # Average predictions
                loss = 0
                for prop_idx, prop in enumerate(model.output_properties):
                    if hasattr(batch, prop):
                        prop_loss = criterion(
                            sum(e_all[prop_idx]) / len(e_all[prop_idx]), 
                            getattr(batch, prop)
                        )
                        loss += prop_loss
                
                # Add force loss if needed
                if model.regress_forces and hasattr(batch, "forces") and f_all:
                    force_loss = criterion(
                        sum(f_all) / len(f_all), 
                        batch.forces
                    )
                    force_weight = get_config_param(config, "training.force_weight", 0.1)
                    loss += force_weight * force_loss
            
            else:
                # Standard forward pass without frame averaging
                preds = model(batch)
                
                # Calculate loss
                loss = 0
                for prop in model.output_properties:
                    if hasattr(batch, prop):
                        prop_loss = criterion(preds[prop], getattr(batch, prop))
                        loss += prop_loss
                
                # Add force loss if needed
                if model.regress_forces and "forces" in preds and hasattr(batch, "forces"):
                    force_loss = criterion(preds["forces"], batch.forces)
                    force_weight = get_config_param(config, "training.force_weight", 0.1)
                    loss += force_weight * force_loss
            
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
                
                if config.training.frame_averaging:
                    batch = apply_frame_averaging_to_batch(batch, config.training.fa_method)
                    
                    # Process with frame averaging
                    e_all = [[] for _ in model.output_properties]
                    
                    for i in range(len(batch.fa_pos)):
                        # Set current frame
                        original_pos = batch.pos.clone()
                        batch.pos = batch.fa_pos[i]
                        
                        # Forward pass
                        preds = model(batch)
                        
                        # Collect predictions
                        for prop_idx, prop in enumerate(model.output_properties):
                            e_all[prop_idx].append(preds[prop])
                        
                        # Restore original positions
                        batch.pos = original_pos
                    
                    # Calculate validation loss
                    batch_loss = 0
                    for prop_idx, prop in enumerate(model.output_properties):
                        if hasattr(batch, prop):
                            prop_val_loss = criterion(
                                sum(e_all[prop_idx]) / len(e_all[prop_idx]),
                                getattr(batch, prop)
                            )
                            batch_loss += prop_val_loss
                else:
                    # Standard forward pass
                    preds = model(batch)
                    
                    # Calculate loss
                    batch_loss = 0
                    for prop in model.output_properties:
                        if hasattr(batch, prop):
                            prop_val_loss = criterion(preds[prop], getattr(batch, prop))
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
            batch_size = get_config_param(config, "training.batch_size", 32)
            file_indices = data_loader.dataset.indices[batch_idx * batch_size:
                                                     min((batch_idx + 1) * batch_size, 
                                                         len(data_loader.dataset))]
            
            file_names = [data_loader.dataset.dataset.file_list[idx] for idx in file_indices]
            
            # Apply frame averaging if requested
            frame_averaging = get_config_param(config, "training.frame_averaging", None)
            fa_method = get_config_param(config, "training.fa_method", "all")
            if frame_averaging:
                batch = apply_frame_averaging_to_batch(batch, fa_method)
                
                # Process with frame averaging
                e_all = {prop: [] for prop in model.output_properties}
                f_all = []
                
                for i in range(len(batch.fa_pos)):
                    # Set current frame
                    original_pos = batch.pos.clone()
                    batch.pos = batch.fa_pos[i]
                    
                    # Forward pass
                    preds = model(batch)
                    
                    # Collect predictions
                    for prop in model.output_properties:
                        e_all[prop].append(preds[prop])
                    
                    # Transform forces if needed
                    if model.regress_forces and "forces" in preds:
                        fa_rot = torch.repeat_interleave(
                            batch.fa_rot[i], batch.natoms, dim=0
                        ).to(device)
                        forces = (
                            preds["forces"]
                            .view(-1, 1, 3)
                            .bmm(fa_rot.transpose(1, 2))
                            .view(-1, 3)
                        )
                        f_all.append(forces)
                    
                    # Restore original positions
                    batch.pos = original_pos
                
                # Average predictions
                final_preds = {}
                for prop in model.output_properties:
                    final_preds[prop] = (sum(e_all[prop]) / len(e_all[prop])).cpu().numpy()
                
                if model.regress_forces and f_all:
                    final_preds["forces"] = (sum(f_all) / len(f_all)).cpu().numpy()
            else:
                # Standard forward pass
                preds = model(batch)
                
                # Convert predictions to numpy
                final_preds = {}
                for prop in model.output_properties:
                    final_preds[prop] = preds[prop].cpu().numpy()
                
                if model.regress_forces and "forces" in preds:
                    final_preds["forces"] = preds["forces"].cpu().numpy()
            
            # Store predictions for each file
            for i, fname in enumerate(file_names):
                result = {"file_name": fname}
                
                # Add predictions
                for prop in final_preds:
                    if prop != "forces":
                        # For graph-level properties (scalar)
                        result[prop] = float(final_preds[prop][i][0])
                    else:
                        # For force predictions (per atom)
                        atoms_per_graph = batch.natoms[i].item()
                        start_idx = sum(batch.natoms[:i].cpu().numpy())
                        end_idx = start_idx + atoms_per_graph
                        result[prop] = final_preds[prop][start_idx:end_idx].tolist()
                
                # Add ground truth if available
                for prop in model.output_properties:
                    if hasattr(batch, prop):
                        result[f"{prop}_true"] = float(getattr(batch, prop)[i][0].cpu().numpy())
                
                if hasattr(batch, "forces"):
                    atoms_per_graph = batch.natoms[i].item()
                    start_idx = sum(batch.natoms[:i].cpu().numpy())
                    end_idx = start_idx + atoms_per_graph
                    result["forces_true"] = batch.forces[start_idx:end_idx].cpu().numpy().tolist()
                
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
    
    # Create config object (using SimpleConfig for internal use)
    config = SimpleConfig(
        batch_size=batch_size,
        epochs=epochs,
        lr=learning_rate,
        seed=seed,
        num_workers=num_workers,
        test_ratio=test_ratio,
        val_ratio=val_ratio,
        frame_averaging=frame_averaging,
        fa_method=fa_method,
        cutoff=cutoff,
        output_dir=output_dir,
        regress_forces=False  # Default to no force regression
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
    # Parse command-line args first to check for --simple flag
    parser = argparse.ArgumentParser(description="FAENet training script")
    parser.add_argument("--simple", action="store_true", help="Use simplified flat config structure")
    
    # Only parse the --simple argument
    args, _ = parser.parse_known_args()
    
    # Get configuration based on flag
    if args.simple:
        config = get_simple_config()
        
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
        
        # Train model using the simplified interface
        train_faenet(
            data_path=config.data_dir,
            structure_col=config.structure_col,
            target_properties=config.target_properties,
            output_dir=config.output_dir,
            frame_averaging=config.frame_averaging,
            fa_method=config.fa_method,
            cutoff=config.cutoff,
            max_neighbors=config.max_neighbors,
            batch_size=config.batch_size,
            epochs=config.epochs,
            learning_rate=config.lr,
            seed=config.seed,
            device=config.device,
            num_workers=config.num_workers,
            num_gaussians=config.num_gaussians,
            hidden_channels=config.hidden_channels,
            num_filters=config.num_filters,
            num_interactions=config.num_interactions,
            dropout=config.dropout,
            regress_forces=config.regress_forces
        )
    else:
        # Use the original nested config
        config = get_config()
        
        # Set random seed
        torch.manual_seed(config.training.seed)
        np.random.seed(config.training.seed)
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Prepare target properties
        target_props = {}
        if config.data.prop_files:
            for prop, file in zip(config.data.target_properties, config.data.prop_files):
                target_props[prop] = file
        
        # Train model using the simplified interface with nested config
        train_faenet(
            data_path=config.data.data_dir,
            structure_col=config.data.structure_col,
            target_properties=config.data.target_properties,
            output_dir=config.output_dir,
            frame_averaging=config.training.frame_averaging,
            fa_method=config.training.fa_method,
            cutoff=config.model.cutoff,
            max_neighbors=config.model.max_neighbors,
            batch_size=config.training.batch_size,
            epochs=config.training.epochs,
            learning_rate=config.training.lr,
            seed=config.training.seed,
            device=config.device,
            num_workers=config.training.num_workers,
            num_gaussians=config.model.num_gaussians,
            hidden_channels=config.model.hidden_channels,
            num_filters=config.model.num_filters,
            num_interactions=config.model.num_interactions,
            dropout=config.model.dropout,
            regress_forces=config.model.regress_forces
        )
    
    
    
if __name__ == "__main__":
    main()