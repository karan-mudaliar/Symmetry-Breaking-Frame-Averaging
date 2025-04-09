import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import numpy as np
from tqdm import tqdm
from torch.utils.data import random_split
import structlog
import mlflow
import traceback

# Configure structlog
logger = structlog.get_logger()

from faenet.dataset import SlabDataset, apply_frame_averaging_to_batch
from faenet.faenet import FAENet
from faenet.config import get_config, Config
from faenet.utils import generate_run_name


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
    
    # Optimizer (use weight_decay if specified)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=config.weight_decay if hasattr(config, 'weight_decay') else 0.0)
    
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
                
                # Initialize consistency loss if enabled - add detailed debug logging
                consistency_loss = None
                logger.warn("consistency_loss_check", 
                           has_attribute=hasattr(config, 'consistency_loss'),
                           config_value=config.consistency_loss if hasattr(config, 'consistency_loss') else None,
                           config_type=type(config).__name__,
                           config_dict=str(config.model_dump() if hasattr(config, 'model_dump') else vars(config)))
                
                if hasattr(config, 'consistency_loss') and config.consistency_loss:
                    from faenet.frame_averaging import compute_consistency_loss
                    consistency_loss = 0
                    logger.warn("consistency_loss_enabled", 
                               weight=config.consistency_weight,
                               normalize=config.consistency_norm)
                else:
                    logger.warn("consistency_loss_disabled", 
                               reason="Configuration parameter is not True" if hasattr(config, 'consistency_loss') else "Config doesn't have consistency_loss attribute")
                
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
                        
                        # Calculate consistency loss if enabled
                        if consistency_loss is not None:
                            # Debug e_all structure to see what's happening
                            num_frames = len(e_all[prop_idx])
                            first_shape = e_all[prop_idx][0].shape if num_frames > 0 else "empty"
                            logger.warn("e_all_structure", 
                                      property=prop,
                                      num_frames=num_frames, 
                                      first_frame_shape=str(first_shape),
                                      e_all_type=type(e_all[prop_idx]).__name__)
                            
                            try:
                                prop_consistency = compute_consistency_loss(
                                    e_all[prop_idx], 
                                    normalize=config.consistency_norm
                                )
                                consistency_loss += prop_consistency
                                logger.warn("property_consistency_loss_calculated", 
                                           property=prop, 
                                           value=prop_consistency.item())
                            except Exception as e:
                                logger.error("consistency_loss_error",
                                            property=prop,
                                            error=str(e),
                                            traceback=traceback.format_exc())
                
                # Add weighted consistency loss to total loss if enabled
                if consistency_loss is not None:
                    loss += config.consistency_weight * consistency_loss
                    
                    # Log metrics if MLflow is enabled
                    if hasattr(config, 'use_mlflow') and config.use_mlflow:
                        try:
                            metric_value = consistency_loss.item()
                            # Log with two different metric names to ensure compatibility
                            mlflow.log_metric("consistency_loss", metric_value, step=epoch)
                            mlflow.log_metric("train_consistency_loss", metric_value, step=epoch)
                            logger.warn("consistency_metric_logged", 
                                      value=metric_value, 
                                      mlflow_enabled=config.use_mlflow)
                        except Exception as e:
                            logger.error("mlflow_logging_error",
                                       error=str(e),
                                       traceback=traceback.format_exc())
            
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
                    
                    # Initialize consistency loss if enabled
                    val_consistency_loss = None
                    if hasattr(config, 'consistency_loss') and config.consistency_loss:
                        from faenet.frame_averaging import compute_consistency_loss
                        val_consistency_loss = 0
                    
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
                            
                            # Calculate consistency loss if enabled
                            if val_consistency_loss is not None:
                                prop_consistency = compute_consistency_loss(
                                    e_all[prop_idx], 
                                    normalize=config.consistency_norm
                                )
                                val_consistency_loss += prop_consistency
                    
                    # Add weighted consistency loss to total loss if enabled
                    if val_consistency_loss is not None:
                        batch_loss += config.consistency_weight * val_consistency_loss
                        
                        # Log metrics if MLflow is enabled
                        if hasattr(config, 'use_mlflow') and config.use_mlflow:
                            mlflow.log_metric("val_consistency_loss", val_consistency_loss.item(), step=epoch)
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
        
        # Log results
        logger.info("epoch_complete", 
                   epoch=epoch+1, 
                   train_loss=train_loss, 
                   val_loss=val_loss)
        
        # Log to MLflow if enabled
        if hasattr(config, 'use_mlflow') and config.use_mlflow:
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            
            # Log learning rate
            current_lr = optimizer.param_groups[0]['lr']
            mlflow.log_metric("learning_rate", current_lr, step=epoch)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(output_dir, "best_model.pt")
            torch.save(model.state_dict(), model_path)
            logger.info("best_model_saved", val_loss=best_val_loss)
            
            # Log best model to MLflow
            if hasattr(config, 'use_mlflow') and config.use_mlflow:
                mlflow.log_artifact(model_path)
        
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
            
            # Get unique identifiers for each structure in batch
            batch_size = config.batch_size
            file_indices = data_loader.dataset.indices[batch_idx * batch_size:
                                                     min((batch_idx + 1) * batch_size, 
                                                         len(data_loader.dataset))]
            
            # These are the unique identifiers (mpid_miller_term_flipped or index-based)
            unique_identifiers = [data_loader.dataset.dataset.file_list[idx] for idx in file_indices]
            logger.debug("batch_identifiers", batch_idx=batch_idx, identifiers=unique_identifiers)
            
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
            for i, identifier in enumerate(unique_identifiers):
                result = {"jid": identifier}
                
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
    
    # Log info about the identifiers used
    id_type = "unique" if any(['_' in str(r['jid']) for r in results[:5]]) else "index-based"
    has_flipped = any(['flipped' in str(r['jid']) for r in results])
    
    logger.info("inference_results_saved", 
               count=len(results), 
               output_file=output_file,
               identifier_type=id_type,
               includes_flipped_status=has_flipped)


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
    
    # Create a clean dictionary of ONLY model parameters
    # List of valid FAENet model parameters
    faenet_params = [
        'cutoff',
        'num_gaussians',
        'hidden_channels',
        'num_filters', 
        'num_interactions',
        'dropout',
        'output_properties'
    ]
    
    # Debug model_kwargs to see what's coming in
    logger.warn("train_faenet_model_kwargs", 
               has_consistency_loss='consistency_loss' in model_kwargs,
               consistency_loss_value=model_kwargs.get('consistency_loss', 'Not Present'),
               kwargs_keys=list(model_kwargs.keys()))
    
    # Create a dictionary with ONLY the allowed model parameters
    model_args = {k: model_kwargs[k] for k in faenet_params if k in model_kwargs}
    model_args['cutoff'] = cutoff  # Always include cutoff
    
    # Extract other parameters by category
    # MLflow parameters
    use_mlflow = model_kwargs.get('use_mlflow', True)
    mlflow_experiment_name = model_kwargs.get('mlflow_experiment_name', 'FAENet_Training')
    run_name = model_kwargs.get('run_name', None)
    
    # Optimizer parameters
    weight_decay = model_kwargs.get('weight_decay', 1e-5)
    
    # Consistency loss parameters - explicitly extract these
    consistency_loss_enabled = model_kwargs.get('consistency_loss', False)
    consistency_weight = model_kwargs.get('consistency_weight', 0.1)
    consistency_norm = model_kwargs.get('consistency_norm', True)
    
    # Log extraction
    logger.warn("consistency_params_extracted",
               enabled=consistency_loss_enabled,
               weight=consistency_weight,
               normalize=consistency_norm)
    
    # Generate a memorable run name if not provided
    if run_name is None:
        run_name = generate_run_name()
        logger.info("generated_run_name", run_name=run_name)
    
    # Create config object with all parameters
    config_params = {
        'batch_size': batch_size,
        'epochs': epochs,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'seed': seed,
        'num_workers': num_workers,
        'test_ratio': test_ratio,
        'val_ratio': val_ratio,
        'frame_averaging': frame_averaging,
        'fa_method': fa_method,
        'cutoff': cutoff,
        'output_dir': output_dir,
        'use_mlflow': use_mlflow,
        'mlflow_experiment_name': mlflow_experiment_name,
        'run_name': run_name,
        'data_path': data_path,
        'structure_col': structure_col,
        'target_properties': target_properties,  # Explicitly include target_properties
        
        # Explicitly include consistency loss parameters
        'consistency_loss': consistency_loss_enabled,
        'consistency_weight': consistency_weight,
        'consistency_norm': consistency_norm
    }
    
    # Log the config parameters we're creating
    logger.warn("creating_config", 
               frame_averaging=frame_averaging,
               consistency_loss=consistency_loss_enabled)
    
    # Create config from parameters
    config = Config(**config_params)
    
    # Create dataset
    logger.info("loading_dataset", data_path=str(data_path))
    dataset = SlabDataset(
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
    
    # Initialize model with clean model_args
    # If output_properties isn't in model_args, add it from the local variable
    if 'output_properties' not in model_args:
        model_args['output_properties'] = output_properties
    
    # Log the model parameters being used
    logger.info("initializing_model", 
               num_gaussians=model_args.get('num_gaussians', 50),
               hidden_channels=model_args.get('hidden_channels', 128),
               num_filters=model_args.get('num_filters', 128),
               num_interactions=model_args.get('num_interactions', 4),
               output_properties=model_args['output_properties'])
    
    model = FAENet(**model_args).to(device)
    
    # Log model summary
    num_params = sum(p.numel() for p in model.parameters())
    logger.info("model_initialized", 
               parameters=num_params, 
               device=str(device),
               train_size=train_size,
               val_size=val_size,
               test_size=test_size,
               output_properties=model.output_properties)
               
    # Debug log for consistency loss test
    logger.warn("model_output_properties",
               properties=model.output_properties,
               has_properties=hasattr(model, 'output_properties'))
    
    # MLflow setup
    if config.use_mlflow:
        mlflow.set_experiment(config.mlflow_experiment_name)
        
        # Run name should be already set at this point
        
        # Start MLflow run - create a non-context-manager run to avoid automatic end when the context exits
        # This is important for tests that check if the run exists immediately after training
        run = mlflow.start_run(run_name=config.run_name)
        
        try:
            # Log all configuration parameters
            for key, value in config.model_dump().items():
                if isinstance(value, (str, int, float, bool)) or value is None:
                    mlflow.log_param(key, value)
                elif isinstance(value, (list, tuple)) and all(isinstance(x, (str, int, float, bool)) for x in value):
                    mlflow.log_param(key, str(value))
            
            # Log system info
            mlflow.log_param("num_params", num_params)
            mlflow.log_param("train_size", train_size)
            mlflow.log_param("val_size", val_size)
            mlflow.log_param("test_size", test_size)
            
            # Train model
            train(model, train_loader, val_loader, device, config)
            
            # Evaluate on test set after training
            model.eval()
            test_loss = 0
            criterion = nn.MSELoss()
            
            with torch.no_grad():
                for batch in tqdm(test_loader, desc="Evaluating test set"):
                    batch = batch.to(device)
                    
                    # Apply frame averaging if enabled
                    if config.frame_averaging:
                        batch = apply_frame_averaging_to_batch(batch, config.fa_method, config.frame_averaging)
                        
                        # Process all frames at once (using the same code pattern as in train function)
                        from torch_geometric.data import Batch
                        
                        # Save original positions and cell
                        original_pos = batch.pos.clone()
                        original_cell = batch.cell.clone() if hasattr(batch, 'cell') and isinstance(batch.cell, torch.Tensor) else batch.cell
                        
                        # Create a list of batches, one for each frame
                        all_batches = []
                        for i in range(len(batch.fa_pos)):
                            frame_batch = batch.clone()
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
                            prop_preds = all_preds[prop]
                            
                            for i in range(num_frames):
                                start_idx = i * batch_size
                                end_idx = (i + 1) * batch_size
                                e_all[prop_idx].append(prop_preds[start_idx:end_idx])
                        
                        # Restore original positions and cell
                        batch.pos = original_pos
                        if hasattr(batch, 'cell') and batch.cell is not None:
                            batch.cell = original_cell
                        
                        # Calculate test loss
                        batch_loss = 0
                        
                        # Initialize consistency loss if enabled
                        test_consistency_loss = None
                        if hasattr(config, 'consistency_loss') and config.consistency_loss:
                            from faenet.frame_averaging import compute_consistency_loss
                            test_consistency_loss = 0
                        
                        for prop_idx, prop in enumerate(model.output_properties):
                            if hasattr(batch, prop):
                                # Average predictions across frames
                                avg_pred = sum(e_all[prop_idx]) / len(e_all[prop_idx])
                                targets = getattr(batch, prop)
                                
                                # Ensure tensors have compatible shapes
                                if avg_pred.dim() != targets.dim():
                                    if avg_pred.dim() > targets.dim():
                                        avg_pred = avg_pred.squeeze()
                                    else:
                                        targets = targets.unsqueeze(-1)
                                
                                prop_test_loss = criterion(avg_pred, targets)
                                batch_loss += prop_test_loss
                                
                                # Calculate consistency loss if enabled
                                if test_consistency_loss is not None:
                                    prop_consistency = compute_consistency_loss(
                                        e_all[prop_idx], 
                                        normalize=config.consistency_norm
                                    )
                                    test_consistency_loss += prop_consistency
                        
                        # Add weighted consistency loss to total loss if enabled
                        if test_consistency_loss is not None:
                            batch_loss += config.consistency_weight * test_consistency_loss
                            
                            # Log metrics if MLflow is enabled
                            if hasattr(config, 'use_mlflow') and config.use_mlflow:
                                mlflow.log_metric("test_consistency_loss", test_consistency_loss.item())
                    else:
                        # Standard forward pass
                        preds = model(batch)
                        
                        # Calculate loss
                        batch_loss = 0
                        for prop in model.output_properties:
                            if hasattr(batch, prop):
                                model_preds = preds[prop]
                                targets = getattr(batch, prop)
                                
                                # Make sure both are the same shape
                                if model_preds.dim() != targets.dim():
                                    if model_preds.dim() > targets.dim():
                                        model_preds = model_preds.squeeze()
                                    else:
                                        targets = targets.unsqueeze(-1)
                                
                                prop_test_loss = criterion(model_preds, targets)
                                batch_loss += prop_test_loss
                    
                    test_loss += batch_loss.item()
            
            test_loss /= len(test_loader)
            logger.info("test_evaluation_complete", test_loss=test_loss)
            
            # Log test metrics
            mlflow.log_metric("test_loss", test_loss)
            
            # Log the inference JSON as an artifact
            inference_output_path = os.path.join(output_dir, config.inference_output)
            if os.path.exists(inference_output_path):
                mlflow.log_artifact(inference_output_path)
                
            # Only end the run if specified (to support tests that need to check the run right after training)
            # We'll keep the run active during test execution, and the test itself can end it if needed
            if hasattr(config, 'end_mlflow_run') and config.end_mlflow_run:
                mlflow.end_run()
        except Exception as e:
            # Always end run on exception
            mlflow.end_run()
            raise e
    else:
        # Train without MLflow
        train(model, train_loader, val_loader, device, config)
    
    logger.info("training_complete")
    
    # Load best model
    best_model_path = os.path.join(output_dir, "best_model.pt")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        logger.info("best_model_loaded", path=best_model_path)
    
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