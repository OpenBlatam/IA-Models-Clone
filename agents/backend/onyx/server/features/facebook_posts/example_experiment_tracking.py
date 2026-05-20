#!/usr/bin/env python3
"""
🚀 Example Usage of Experiment Tracking System
==============================================

Demonstrates how to use the experiment tracking system with a realistic
training scenario for numerical stability research.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from pathlib import Path

# Import our experiment tracking system
from experiment_tracking import (
    create_experiment_config, create_experiment_tracker
)

# Import our centralized logging configuration
from logging_config import get_logger

logger = get_logger(__name__)


def create_simple_model(input_size=10, hidden_size=64, output_size=1):
    """Create a simple neural network for demonstration."""
    return nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_size, hidden_size // 2),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_size // 2, output_size)
    )


def generate_synthetic_data(num_samples=1000, input_size=10, noise_level=0.1):
    """Generate synthetic training data."""
    # Generate random input features
    X = torch.randn(num_samples, input_size)
    
    # Create a simple target function with some non-linearity
    target = torch.sum(X[:, :5], dim=1, keepdim=True) + \
             0.1 * torch.sum(X[:, 5:], dim=1, keepdim=True) ** 2
    
    # Add noise
    target += noise_level * torch.randn_like(target)
    
    return X, target


def train_with_experiment_tracking():
    """Main training function with experiment tracking."""
    
    print("🚀 Starting Experiment Tracking Demo")
    print("=" * 50)
    
    # 1. Create experiment configuration
    print("\n1. Creating experiment configuration...")
    config = create_experiment_config(
        experiment_name="numerical_stability_demo",
        project_name="blatam_academy_facebook_posts",
        enable_tensorboard=True,
        enable_wandb=False,  # Set to True if you have wandb configured
        log_interval=10,
        log_gradients=True,
        log_nan_inf_counts=True,
        log_clipping_stats=True,
        tensorboard_dir="runs/demo_experiment",
        model_save_dir="models/demo_experiment"
    )
    
    print(f"✅ Configuration created: {config.experiment_name}")
    
    # 2. Create experiment tracker
    print("\n2. Initializing experiment tracker...")
    tracker = create_experiment_tracker(config)
    
    # 3. Log hyperparameters
    print("\n3. Logging hyperparameters...")
    hyperparams = {
        'learning_rate': 0.001,
        'batch_size': 32,
        'num_epochs': 50,
        'max_grad_norm': 1.0,
        'clipping_type': 'norm',
        'optimizer': 'Adam',
        'loss_function': 'MSE',
        'model_architecture': '3-layer MLP with dropout'
    }
    tracker.log_hyperparameters(hyperparams)
    
    # 4. Create model and data
    print("\n4. Setting up model and data...")
    model = create_simple_model()
    X_train, y_train = generate_synthetic_data(1000, 10, 0.1)
    X_val, y_val = generate_synthetic_data(200, 10, 0.1)
    
    # Convert to DataLoader-like batches
    batch_size = hyperparams['batch_size']
    num_batches = len(X_train) // batch_size
    
    # 5. Setup training components
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])
    
    # Log model architecture
    tracker.log_model_architecture(model)
    
    print(f"✅ Model created with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"✅ Training data: {len(X_train)} samples, {num_batches} batches")
    
    # 6. Training loop
    print("\n5. Starting training loop...")
    print("-" * 40)
    
    best_loss = float('inf')
    training_start_time = time.time()
    
    for epoch in range(hyperparams['num_epochs']):
        epoch_start_time = time.time()
        epoch_loss = 0.0
        epoch_grad_norm = 0.0
        epoch_nan_count = 0
        epoch_inf_count = 0
        epoch_clipping_count = 0
        
        model.train()
        
        for batch_idx in range(num_batches):
            # Get batch data
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            batch_X = X_train[start_idx:end_idx]
            batch_y = y_train[start_idx:end_idx]
            
            # Forward pass
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            
            # Check for numerical issues
            if torch.isnan(loss):
                epoch_nan_count += 1
                logger.warning(f"NaN loss detected at epoch {epoch}, batch {batch_idx}")
                continue
                
            if torch.isinf(loss):
                epoch_inf_count += 1
                logger.warning(f"Inf loss detected at epoch {epoch}, batch {batch_idx}")
                continue
            
            # Backward pass
            loss.backward()
            
            # Compute gradient norm before clipping
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            
            # Apply gradient clipping
            clipping_applied = False
            clipping_threshold = hyperparams['max_grad_norm']
            if total_norm > clipping_threshold:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clipping_threshold)
                clipping_applied = True
                epoch_clipping_count += 1
            
            # Update parameters
            optimizer.step()
            
            # Accumulate metrics
            epoch_loss += loss.item()
            epoch_grad_norm += total_norm
            
            # Log training step (every log_interval steps)
            if batch_idx % config.log_interval == 0:
                step_time = time.time() - epoch_start_time
                tracker.log_training_step(
                    loss=loss.item(),
                    accuracy=None,  # Not applicable for regression
                    learning_rate=hyperparams['learning_rate'],
                    gradient_norm=total_norm,
                    nan_count=1 if torch.isnan(loss) else 0,
                    inf_count=1 if torch.isinf(loss) else 0,
                    clipping_applied=clipping_applied,
                    clipping_threshold=clipping_threshold if clipping_applied else None,
                    training_time=step_time,
                    memory_usage=torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else None,
                    gpu_utilization=None  # Would need additional monitoring for this
                )
        
        # Compute epoch averages
        avg_loss = epoch_loss / num_batches
        avg_grad_norm = epoch_grad_norm / num_batches
        epoch_time = time.time() - epoch_start_time
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_output = model(X_val)
            val_loss = criterion(val_output, y_val).item()
        
        # Log epoch metrics
        epoch_metrics = {
            'train_loss': avg_loss,
            'val_loss': val_loss,
            'avg_gradient_norm': avg_grad_norm,
            'nan_count': epoch_nan_count,
            'inf_count': epoch_inf_count,
            'clipping_count': epoch_clipping_count,
            'epoch_time': epoch_time
        }
        tracker.log_epoch(epoch + 1, epoch_metrics)
        
        # Print progress
        print(f"Epoch {epoch+1:2d}/{hyperparams['num_epochs']} | "
              f"Train Loss: {avg_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Grad Norm: {avg_grad_norm:.4f} | "
              f"NaN: {epoch_nan_count} | "
              f"Inf: {epoch_inf_count} | "
              f"Clipping: {epoch_clipping_count}")
        
        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            tracker.save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                step=(epoch + 1) * num_batches,
                loss=val_loss,
                metrics=epoch_metrics
            )
    
    # 7. Training complete
    total_training_time = time.time() - training_start_time
    print("\n" + "=" * 50)
    print("🎉 Training Complete!")
    print(f"⏱️  Total training time: {total_training_time:.2f} seconds")
    print(f"📊 Best validation loss: {best_loss:.4f}")
    
    # 8. Create final visualization
    print("\n6. Creating training visualization...")
    viz_data = tracker.create_visualization("demo_training_progress.png")
    if viz_data:
        summary = viz_data['metrics_summary']
        print(f"✅ Visualization created with {summary['total_steps']} steps")
        print(f"📈 Final loss: {summary['final_loss']:.4f}")
        print(f"🔍 Total NaN count: {summary['total_nan_count']}")
        print(f"🔍 Total Inf count: {summary['total_inf_count']}")
    
    # 9. Get experiment summary
    print("\n7. Generating experiment summary...")
    summary = tracker.get_experiment_summary()
    print(f"📋 Experiment: {summary['experiment_name']}")
    print(f"📁 Project: {summary['project_name']}")
    print(f"📊 Total steps: {summary['total_steps']}")
    print(f"💾 Checkpoints saved: {summary['checkpoints']}")
    
    # 10. Cleanup
    print("\n8. Cleaning up...")
    tracker.close()
    print("✅ Experiment tracker closed")
    
    print("\n" + "=" * 50)
    print("🚀 Demo completed successfully!")
    print("\nNext steps:")
    print("1. Check TensorBoard logs: tensorboard --logdir=runs/demo_experiment")
    print("2. View saved models in: models/demo_experiment/")
    print("3. Check training visualization: demo_training_progress.png")
    print("4. Explore the Gradio interface: python gradio_experiment_tracking.py")


def quick_demo():
    """Quick demo with minimal setup."""
    print("🚀 Quick Experiment Tracking Demo")
    print("=" * 40)
    
    # Create minimal config
    config = create_experiment_config(
        experiment_name="quick_demo",
        enable_tensorboard=True,
        enable_wandb=False,
        log_interval=5
    )
    
    # Create tracker
    tracker = create_experiment_tracker(config)
    
    # Simulate some training steps
    for step in range(20):
        # Simulate metrics
        loss = 1.0 / (step + 1) + np.random.normal(0, 0.1)
        grad_norm = np.random.exponential(0.5)
        
        tracker.log_training_step(
            loss=loss,
            gradient_norm=grad_norm,
            nan_count=np.random.poisson(0.1),
            inf_count=np.random.poisson(0.05),
            clipping_applied=grad_norm > 1.0,
            clipping_threshold=1.0 if grad_norm > 1.0 else None
        )
        
        if step % 5 == 0:
            print(f"Step {step}: Loss={loss:.4f}, Grad Norm={grad_norm:.4f}")
    
    # Get summary
    summary = tracker.get_experiment_summary()
    print(f"\n✅ Demo completed! Total steps: {summary['total_steps']}")
    
    # Cleanup
    tracker.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Experiment Tracking Demo")
    parser.add_argument("--quick", action="store_true", 
                       help="Run quick demo instead of full training")
    
    args = parser.parse_args()
    
    if args.quick:
        quick_demo()
    else:
        train_with_experiment_tracking()






