import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import argparse
import numpy as np
from tqdm import tqdm

from dataset import SlabDataset, apply_frame_averaging_to_batch
from faenet import FAENet


def train(model, train_loader, val_loader, device, config):
    """Train the FAENet model
    
    Args:
        model: FAENet model
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to run on (cpu or cuda)
        config: Training configuration
    """
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.training.lr)
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Track best model
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(config.training.epochs):
        model.train()
        train_loss = 0
        
        # Training
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.training.epochs}"):
            batch = batch.to(device)
            
            # Apply frame averaging
            if config.training.frame_averaging:
                batch = apply_frame_averaging_to_batch(batch, config.training.fa_method)
                
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
                    loss += config.training.force_weight * force_loss
            
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
                    loss += config.training.force_weight * force_loss
            
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
            torch.save(model.state_dict(), os.path.join(config.output_dir, "best_model.pt"))
            print(f"Saved new best model with validation loss: {best_val_loss:.6f}")
        
        # Save checkpoint
        if (epoch + 1) % config.training.checkpoint_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, os.path.join(config.output_dir, f"checkpoint_epoch_{epoch+1}.pt"))
        
        # Always save latest checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, os.path.join(config.output_dir, "checkpoint.pt"))


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
            file_indices = data_loader.dataset.indices[batch_idx * config.training.batch_size:
                                                     min((batch_idx + 1) * config.training.batch_size, 
                                                         len(data_loader.dataset))]
            
            file_names = [data_loader.dataset.dataset.file_list[idx] for idx in file_indices]
            
            # Apply frame averaging if requested
            if config.training.frame_averaging:
                batch = apply_frame_averaging_to_batch(batch, config.training.fa_method)
                
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


def main():
    # Get configuration from command line
    from config import get_config
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
    
    # Create dataset
    dataset = SlabDataset(config.data.data_dir, target_props)
    
    # Split into train, validation, and test sets
    test_size = int(len(dataset) * config.training.test_ratio)
    val_size = int(len(dataset) * config.training.val_ratio)
    train_size = len(dataset) - val_size - test_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers
    )
    
    # Initialize model
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    model = FAENet(
        cutoff=config.model.cutoff,
        num_gaussians=config.model.num_gaussians,
        hidden_channels=config.model.hidden_channels,
        num_filters=config.model.num_filters,
        num_interactions=config.model.num_interactions,
        dropout=config.model.dropout,
        output_properties=config.model.output_properties,
        regress_forces=config.model.regress_forces
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
    best_model_path = os.path.join(config.output_dir, "best_model.pt")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"Loaded best model from {best_model_path}")
    
    # Run inference on test set
    inference_output = os.path.join(config.output_dir, config.inference_output)
    run_inference(model, test_loader, device, config, inference_output)
    
    
if __name__ == "__main__":
    main()