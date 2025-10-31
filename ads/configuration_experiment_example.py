from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import yaml
from onyx.server.features.ads.config_manager import (
from onyx.server.features.ads.experiment_tracker import (
from onyx.server.features.ads.mixed_precision_training import MixedPrecisionTrainer
from onyx.server.features.ads.profiling_optimizer import ProfilingOptimizer
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Configuration Management and Experiment Tracking Example

This example demonstrates how to use the configuration management and experiment tracking
system in a real-world machine learning project. It shows:

1. Creating and managing configurations
2. Setting up experiment tracking
3. Training with automatic checkpointing
4. Loading and resuming experiments
5. Comparing different experiments
"""

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

    ConfigManager, ModelConfig, TrainingConfig, DataConfig,
    ExperimentConfig, OptimizationConfig, DeploymentConfig,
    ConfigType
)

    ExperimentTracker, ExperimentMetadata, create_experiment_tracker,
    experiment_context
)


class SimpleClassifier(nn.Module):
    """Simple neural network classifier for demonstration."""
    
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate=0.1) -> Any:
        super().__init__()
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x) -> Any:
        return self.network(x)

def create_sample_data(num_samples=1000, input_size=100, num_classes=5) -> Any:
    """Create sample data for demonstration."""
    # Generate random features
    X = torch.randn(num_samples, input_size)
    
    # Generate random labels
    y = torch.randint(0, num_classes, (num_samples,))
    
    # Split into train/val/test
    train_size = int(0.7 * num_samples)
    val_size = int(0.15 * num_samples)
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]
    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def train_epoch(model, train_loader, optimizer, criterion, device) -> Any:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
    
    return total_loss / len(train_loader), correct / total

def validate(model, val_loader, criterion, device) -> bool:
    """Validate the model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    return total_loss / len(val_loader), correct / total

def run_experiment_with_configs(project_name="example_project", experiment_name="experiment_1") -> Any:
    """Run a complete experiment using the configuration and tracking system."""
    
    print(f"Starting experiment: {experiment_name}")
    print("=" * 50)
    
    # 1. Set up configuration management
    print("1. Setting up configuration management...")
    config_manager = ConfigManager("./configs")
    
    # Create default configurations
    config_files = config_manager.create_default_configs(project_name)
    print(f"   Created configuration files: {list(config_files.keys())}")
    
    # Load configurations
    configs = config_manager.load_all_configs(project_name)
    
    # Update configurations for this specific experiment
    configs['model'].name = f"{project_name}_model"
    configs['model'].input_size = 100
    configs['model'].output_size = 5
    configs['model'].hidden_sizes = [64, 32]
    configs['model'].dropout_rate = 0.2
    
    configs['training'].batch_size = 32
    configs['training'].learning_rate = 1e-3
    configs['training'].epochs = 10
    configs['training'].validation_split = 0.15
    configs['training'].test_split = 0.15
    
    configs['experiment'].experiment_name = experiment_name
    configs['experiment'].tracking_backend = "local"  # Use local backend for this example
    configs['experiment'].log_frequency = 10
    configs['experiment'].checkpoint_frequency = 2
    
    configs['optimization'].enable_mixed_precision = True
    configs['optimization'].enable_profiling = True
    
    # Save updated configurations
    for config_type, config in configs.items():
        config_manager.save_config(
            config, 
            f"{project_name}_{config_type}_config.yaml", 
            ConfigType[config_type.upper()]
        )
    
    # 2. Set up experiment tracking
    print("2. Setting up experiment tracking...")
    tracker = create_experiment_tracker(configs['experiment'])
    
    # Create experiment metadata
    metadata = ExperimentMetadata(
        experiment_id=f"{project_name}_{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        experiment_name=experiment_name,
        project_name=project_name,
        created_at=datetime.now(),
        tags=["example", "classification", "demo"],
        description="Example experiment demonstrating configuration management and experiment tracking",
        git_commit="example_commit",
        python_version="3.9.0",
        dependencies={
            "torch": "2.0.0",
            "numpy": "1.24.0"
        },
        hardware_info={
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "cuda_available": torch.cuda.is_available()
        }
    )
    
    # Start experiment
    tracker.start_experiment(metadata)
    
    # Log hyperparameters
    hyperparameters = {
        "model": {
            "architecture": configs['model'].architecture,
            "input_size": configs['model'].input_size,
            "output_size": configs['model'].output_size,
            "hidden_sizes": configs['model'].hidden_sizes,
            "dropout_rate": configs['model'].dropout_rate
        },
        "training": {
            "batch_size": configs['training'].batch_size,
            "learning_rate": configs['training'].learning_rate,
            "epochs": configs['training'].epochs,
            "optimizer": configs['training'].optimizer,
            "scheduler": configs['training'].scheduler
        },
        "optimization": {
            "mixed_precision": configs['optimization'].enable_mixed_precision,
            "gradient_accumulation": configs['optimization'].gradient_accumulation_steps
        }
    }
    tracker.log_hyperparameters(hyperparameters)
    
    # 3. Create model and data
    print("3. Creating model and data...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Using device: {device}")
    
    # Create model
    model = SimpleClassifier(
        input_size=configs['model'].input_size,
        hidden_sizes=configs['model'].hidden_sizes,
        output_size=configs['model'].output_size,
        dropout_rate=configs['model'].dropout_rate
    ).to(device)
    
    # Log model architecture
    tracker.log_model_architecture(model)
    
    # Create data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = create_sample_data(
        num_samples=1000,
        input_size=configs['model'].input_size,
        num_classes=configs['model'].output_size
    )
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=configs['training'].batch_size,
        shuffle=configs['data'].shuffle,
        num_workers=configs['data'].num_workers,
        pin_memory=configs['data'].pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=configs['training'].batch_size,
        shuffle=False,
        num_workers=configs['data'].num_workers,
        pin_memory=configs['data'].pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=configs['training'].batch_size,
        shuffle=False,
        num_workers=configs['data'].num_workers,
        pin_memory=configs['data'].pin_memory
    )
    
    # 4. Set up training components
    print("4. Setting up training components...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=configs['training'].learning_rate,
        **configs['training'].optimizer_params
    )
    
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=configs['training'].scheduler_params.get('step_size', 5),
        gamma=configs['training'].scheduler_params.get('gamma', 0.1)
    )
    
    # Set up mixed precision training if enabled
    if configs['optimization'].enable_mixed_precision:
        print("   Using mixed precision training")
        mp_trainer = MixedPrecisionTrainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=configs['optimization']
        )
    
    # Set up profiling if enabled
    if configs['optimization'].enable_profiling:
        print("   Setting up profiling")
        profiler = ProfilingOptimizer()
        profiler.start_profiling()
    
    # 5. Training loop
    print("5. Starting training loop...")
    best_val_loss = float('inf')
    global_step = 0
    
    for epoch in range(configs['training'].epochs):
        print(f"   Epoch {epoch + 1}/{configs['training'].epochs}")
        
        # Training
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validation
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        # Log metrics
        metrics = {
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "learning_rate": scheduler.get_last_lr()[0],
            "epoch": epoch + 1
        }
        
        tracker.log_metrics(metrics, step=global_step, epoch=epoch + 1)
        
        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
        
        if (epoch + 1) % configs['experiment'].checkpoint_frequency == 0 or is_best:
            checkpoint_path = tracker.save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                metrics=metrics,
                is_best=is_best
            )
            print(f"   Saved checkpoint: {Path(checkpoint_path).name}")
        
        global_step += len(train_loader)
        
        print(f"   Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"   Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    # 6. Final evaluation
    print("6. Final evaluation...")
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    
    final_metrics = {
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "best_val_loss": best_val_loss
    }
    
    tracker.log_metrics(final_metrics, step=global_step, epoch=configs['training'].epochs)
    
    # Save final checkpoint
    final_checkpoint_path = tracker.save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        metrics=final_metrics,
        is_best=False
    )
    
    print(f"   Final Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    
    # 7. End experiment
    print("7. Ending experiment...")
    tracker.end_experiment()
    
    # Stop profiling if enabled
    if configs['optimization'].enable_profiling:
        profiler.stop_profiling()
        profiler.generate_report()
    
    print("Experiment completed successfully!")
    print(f"Experiment ID: {tracker.experiment_id}")
    print(f"Checkpoint directory: {configs['experiment'].checkpoint_dir}")
    
    return tracker, model, final_metrics

def resume_experiment(experiment_id, checkpoint_path=None) -> Any:
    """Resume a previous experiment."""
    print(f"Resuming experiment: {experiment_id}")
    print("=" * 50)
    
    # Load experiment configuration
    config_manager = ConfigManager("./configs")
    
    # Find the experiment directory
    checkpoint_dir = Path("./checkpoints")
    experiment_dir = checkpoint_dir / experiment_id
    
    if not experiment_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")
    
    # Load experiment metadata
    metadata_file = experiment_dir / "metadata.yaml"
    with open(metadata_file, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        metadata_dict = yaml.safe_load(f)
    
    # Load hyperparameters
    hp_file = experiment_dir / "hyperparameters.yaml"
    with open(hp_file, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        hyperparameters = yaml.safe_load(f)
    
    # Create experiment config
    experiment_config = ExperimentConfig(
        experiment_name=metadata_dict['experiment_name'],
        project_name=metadata_dict['project_name'],
        track_experiments=True,
        tracking_backend="local",
        save_checkpoints=True,
        checkpoint_dir="./checkpoints"
    )
    
    # Create tracker
    tracker = create_experiment_tracker(experiment_config)
    
    # Recreate model
    model_config = hyperparameters['model']
    model = SimpleClassifier(
        input_size=model_config['input_size'],
        hidden_sizes=model_config['hidden_sizes'],
        output_size=model_config['output_size'],
        dropout_rate=model_config['dropout_rate']
    )
    
    # Load checkpoint
    if checkpoint_path is None:
        checkpoint_path = tracker.checkpoint_manager.get_latest_checkpoint(experiment_id)
    
    if checkpoint_path:
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint_info = tracker.load_checkpoint(
            model=model,
            checkpoint_path=checkpoint_path
        )
        print(f"Loaded checkpoint from epoch {checkpoint_info['epoch']}, step {checkpoint_info['step']}")
        print(f"Checkpoint metrics: {checkpoint_info['metrics']}")
    else:
        print("No checkpoint found")
    
    return tracker, model, hyperparameters

def compare_experiments(experiment_ids) -> Any:
    """Compare multiple experiments."""
    print("Comparing experiments...")
    print("=" * 50)
    
    results = {}
    
    for experiment_id in experiment_ids:
        experiment_dir = Path("./checkpoints") / experiment_id
        
        if not experiment_dir.exists():
            print(f"Experiment {experiment_id} not found")
            continue
        
        # Load experiment summary
        summary_file = experiment_dir / "experiment_summary.yaml"
        if summary_file.exists():
            with open(summary_file, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                summary = yaml.safe_load(f)
            
            results[experiment_id] = {
                'experiment_name': summary['experiment_name'],
                'total_steps': summary['total_steps'],
                'total_epochs': summary['total_epochs'],
                'final_metrics': summary['final_metrics']
            }
    
    # Print comparison
    print(f"{'Experiment':<20} {'Steps':<10} {'Epochs':<10} {'Test Loss':<12} {'Test Acc':<12}")
    print("-" * 70)
    
    for exp_id, result in results.items():
        print(f"{result['experiment_name']:<20} "
              f"{result['total_steps']:<10} "
              f"{result['total_epochs']:<10} "
              f"{result['final_metrics'].get('test_loss', 'N/A'):<12.4f} "
              f"{result['final_metrics'].get('test_accuracy', 'N/A'):<12.4f}")
    
    return results

def main():
    """Main function demonstrating the complete workflow."""
    print("Configuration Management and Experiment Tracking Example")
    print("=" * 60)
    
    # Create output directories
    os.makedirs("./configs", exist_ok=True)
    os.makedirs("./checkpoints", exist_ok=True)
    
    # Run first experiment
    print("\nRunning first experiment...")
    tracker1, model1, metrics1 = run_experiment_with_configs(
        project_name="example_project",
        experiment_name="experiment_1"
    )
    
    # Run second experiment with different hyperparameters
    print("\nRunning second experiment with different hyperparameters...")
    tracker2, model2, metrics2 = run_experiment_with_configs(
        project_name="example_project",
        experiment_name="experiment_2"
    )
    
    # Compare experiments
    print("\nComparing experiments...")
    compare_experiments([tracker1.experiment_id, tracker2.experiment_id])
    
    # Demonstrate resuming an experiment
    print("\nDemonstrating experiment resumption...")
    try:
        resumed_tracker, resumed_model, hyperparameters = resume_experiment(tracker1.experiment_id)
        print("Successfully resumed experiment!")
    except Exception as e:
        print(f"Failed to resume experiment: {e}")
    
    print("\nExample completed successfully!")
    print("Check the following directories for outputs:")
    print("  - ./configs/ - Configuration files")
    print("  - ./checkpoints/ - Experiment checkpoints and logs")

match __name__:
    case "__main__":
    main() 