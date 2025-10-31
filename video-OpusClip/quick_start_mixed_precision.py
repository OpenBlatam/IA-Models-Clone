#!/usr/bin/env python3
"""
Quick Start Mixed Precision Training for Video-OpusClip

Easy-to-use script for getting started with mixed precision training
using torch.cuda.amp with minimal configuration.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
import argparse
from pathlib import Path

# Import mixed precision modules
from mixed_precision_training import (
    MixedPrecisionConfig,
    MixedPrecisionTrainer,
    create_mixed_precision_config,
    benchmark_mixed_precision
)

def create_sample_model(input_size=784, hidden_size=512, num_classes=10):
    """Create a sample model for demonstration."""
    return nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_size, hidden_size // 2),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_size // 2, num_classes)
    )

def create_sample_dataset(num_samples=1000, input_size=784, num_classes=10):
    """Create a sample dataset for demonstration."""
    torch.manual_seed(42)
    data = torch.randn(num_samples, input_size)
    labels = torch.randint(0, num_classes, (num_samples,))
    return TensorDataset(data, labels)

def quick_start_basic():
    """Basic mixed precision training setup."""
    print("=== Quick Start: Basic Mixed Precision Training ===")
    
    # Check CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    # Create dataset
    dataset = create_sample_dataset()
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True, pin_memory=True)
    
    # Create model
    model = create_sample_model()
    
    # Create mixed precision configuration
    config = create_mixed_precision_config(
        enabled=True,
        dtype=torch.float16,
        init_scale=2**16
    )
    
    # Create trainer
    trainer = MixedPrecisionTrainer(
        model=model,
        train_loader=train_loader,
        config=config,
        optimizer=optim.Adam(model.parameters(), lr=1e-3),
        loss_fn=nn.CrossEntropyLoss(),
        device=device
    )
    
    # Training loop
    print("\nStarting training...")
    start_time = time.time()
    
    for epoch in range(3):
        metrics = trainer.train_epoch(epoch)
        print(f"Epoch {epoch}: Loss={metrics['train_loss']:.4f}, "
              f"Scaler Scale={metrics['scaler_scale']:.2e}")
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    
    return trainer

def quick_start_benchmark():
    """Benchmark mixed precision vs full precision."""
    print("\n=== Quick Start: Performance Benchmark ===")
    
    # Create dataset
    dataset = create_sample_dataset()
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True, pin_memory=True)
    
    # Create model
    model = create_sample_model()
    
    # Test configurations
    configs = [
        ("FP32", MixedPrecisionConfig(enabled=False)),
        ("FP16", MixedPrecisionConfig(enabled=True, dtype=torch.float16))
    ]
    
    print("Running benchmarks...")
    for name, config in configs:
        print(f"\n{name}:")
        try:
            results = benchmark_mixed_precision(
                model=model,
                train_loader=train_loader,
                config=config,
                num_steps=50
            )
            
            print(f"  Speedup: {results['speedup']:.2f}x")
            print(f"  Memory savings: {results['memory_savings']:.1f}%")
            print(f"  Training time: {results['mp_time']:.2f}s")
            
        except Exception as e:
            print(f"  Error: {e}")

def quick_start_advanced():
    """Advanced mixed precision configuration."""
    print("\n=== Quick Start: Advanced Configuration ===")
    
    # Create advanced configuration
    config = MixedPrecisionConfig(
        enabled=True,
        dtype=torch.float16,
        init_scale=2**16,
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=2000,
        log_scaling=True,
        log_frequency=50,
        save_scaler_state=True,
        handle_overflow=True
    )
    
    # Create dataset
    dataset = create_sample_dataset(num_samples=2000, input_size=1024, num_classes=20)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True, pin_memory=True)
    
    # Create model
    model = create_sample_model(input_size=1024, hidden_size=512, num_classes=20)
    
    # Create trainer with scheduler
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
    
    trainer = MixedPrecisionTrainer(
        model=model,
        train_loader=train_loader,
        config=config,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=nn.CrossEntropyLoss(),
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Training with monitoring
    print("Starting advanced training...")
    for epoch in range(2):
        metrics = trainer.train_epoch(epoch)
        
        # Get detailed status
        status = trainer.get_status()
        
        print(f"Epoch {epoch}:")
        print(f"  Loss: {metrics['train_loss']:.4f}")
        print(f"  Scaler Scale: {metrics['scaler_scale']:.2e}")
        print(f"  Overflow Count: {metrics['overflow_count']}")
        print(f"  Memory Usage: {metrics['memory_usage']['gpu_memory_allocated']:.2f}GB")
    
    return trainer

def quick_start_memory_efficient():
    """Memory-efficient mixed precision training."""
    print("\n=== Quick Start: Memory-Efficient Training ===")
    
    # Memory-efficient configuration
    config = MixedPrecisionConfig(
        enabled=True,
        dtype=torch.float16,
        memory_efficient=True,
        cache_enabled=False,
        pin_memory=False
    )
    
    # Create larger dataset
    dataset = create_sample_dataset(num_samples=3000, input_size=1024, num_classes=25)
    train_loader = DataLoader(
        dataset,
        batch_size=16,  # Smaller batch size
        shuffle=True,
        pin_memory=False,
        num_workers=2
    )
    
    # Create model
    model = create_sample_model(input_size=1024, hidden_size=512, num_classes=25)
    
    # Enable gradient checkpointing if available
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")
    
    # Create trainer
    trainer = MixedPrecisionTrainer(
        model=model,
        train_loader=train_loader,
        config=config,
        optimizer=optim.Adam(model.parameters(), lr=1e-3),
        loss_fn=nn.CrossEntropyLoss(),
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Training
    print("Starting memory-efficient training...")
    for epoch in range(2):
        metrics = trainer.train_epoch(epoch)
        print(f"Epoch {epoch}: Loss={metrics['train_loss']:.4f}, "
              f"Memory: {metrics['memory_usage']['gpu_memory_allocated']:.2f}GB")
    
    return trainer

def quick_start_checkpointing():
    """Checkpointing with mixed precision."""
    print("\n=== Quick Start: Checkpointing ===")
    
    # Create configuration
    config = create_mixed_precision_config(
        enabled=True,
        dtype=torch.float16,
        init_scale=2**16
    )
    
    # Create dataset
    dataset = create_sample_dataset()
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True, pin_memory=True)
    
    # Create model
    model = create_sample_model()
    
    # Create trainer
    trainer = MixedPrecisionTrainer(
        model=model,
        train_loader=train_loader,
        config=config,
        optimizer=optim.Adam(model.parameters(), lr=1e-3),
        loss_fn=nn.CrossEntropyLoss(),
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Create checkpoint directory
    checkpoint_dir = Path("quick_start_checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Training with checkpointing
    print("Starting training with checkpointing...")
    for epoch in range(3):
        metrics = trainer.train_epoch(epoch)
        print(f"Epoch {epoch}: Loss={metrics['train_loss']:.4f}")
        
        # Save checkpoint
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
        trainer.save_checkpoint(epoch, str(checkpoint_path))
        print(f"  Checkpoint saved: {checkpoint_path}")
    
    # Load and resume
    print("\nLoading checkpoint and resuming...")
    new_trainer = MixedPrecisionTrainer(
        model=model,
        train_loader=train_loader,
        config=config,
        optimizer=optim.Adam(model.parameters(), lr=1e-3),
        loss_fn=nn.CrossEntropyLoss(),
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    checkpoint_path = checkpoint_dir / "checkpoint_epoch_1.pth"
    new_trainer.load_checkpoint(str(checkpoint_path))
    print(f"Resumed from epoch {new_trainer.epoch}")
    
    # Continue training
    for epoch in range(new_trainer.epoch + 1, 3):
        metrics = new_trainer.train_epoch(epoch)
        print(f"Epoch {epoch}: Loss={metrics['train_loss']:.4f}")
    
    return new_trainer

def main():
    """Main function for quick start script."""
    parser = argparse.ArgumentParser(description="Quick Start Mixed Precision Training")
    parser.add_argument(
        "--mode",
        choices=["basic", "benchmark", "advanced", "memory", "checkpoint", "all"],
        default="basic",
        help="Training mode to run"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training"
    )
    
    args = parser.parse_args()
    
    print("Mixed Precision Training Quick Start")
    print("=" * 50)
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"CUDA capability: {torch.cuda.get_device_capability(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    else:
        print("CUDA not available, using CPU")
    
    print()
    
    try:
        if args.mode == "basic" or args.mode == "all":
            quick_start_basic()
        
        if args.mode == "benchmark" or args.mode == "all":
            quick_start_benchmark()
        
        if args.mode == "advanced" or args.mode == "all":
            quick_start_advanced()
        
        if args.mode == "memory" or args.mode == "all":
            quick_start_memory_efficient()
        
        if args.mode == "checkpoint" or args.mode == "all":
            quick_start_checkpointing()
        
        print("\n" + "=" * 50)
        print("Quick start completed successfully!")
        
        if args.mode == "all":
            print("\nAll modes completed. Check the output above for results.")
        
    except Exception as e:
        print(f"\nError during quick start: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 