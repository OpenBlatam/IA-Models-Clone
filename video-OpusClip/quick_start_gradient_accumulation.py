#!/usr/bin/env python3
"""
Quick Start Gradient Accumulation for Video-OpusClip

Easy setup and execution of gradient accumulation with automatic configuration
and best practices implementation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import time
import argparse
import sys
from pathlib import Path

# Import gradient accumulation components
try:
    from gradient_accumulation import (
        GradientAccumulationConfig, GradientAccumulationTrainer,
        create_accumulation_config, calculate_optimal_accumulation_steps
    )
    from gradient_accumulation_examples import (
        LargeVideoModel, MemoryIntensiveDataset
    )
    GRADIENT_ACCUMULATION_AVAILABLE = True
except ImportError as e:
    print(f"Error importing gradient accumulation components: {e}")
    print("Please ensure gradient_accumulation.py is in the same directory.")
    GRADIENT_ACCUMULATION_AVAILABLE = False

def check_requirements():
    """Check if all requirements are met for gradient accumulation."""
    print("Checking requirements...")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA is not available. Gradient accumulation requires CUDA.")
        return False
    
    # Check PyTorch version
    pytorch_version = torch.__version__
    print(f"✅ PyTorch version: {pytorch_version}")
    
    # Check GPU memory
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✅ GPU memory: {gpu_memory:.1f}GB")
        
        if gpu_memory < 4.0:
            print("⚠️  Low GPU memory detected. Consider using smaller batch sizes.")
    
    # Check gradient accumulation components
    if not GRADIENT_ACCUMULATION_AVAILABLE:
        print("❌ Gradient accumulation components not available.")
        return False
    
    print("✅ Gradient accumulation components available")
    
    return True

def get_optimal_configuration(target_batch_size=128):
    """Get optimal configuration based on available hardware."""
    if not torch.cuda.is_available():
        return None
    
    # Get GPU memory
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    # Estimate model memory (rough estimate)
    model_memory = 2.0  # GB
    
    # Calculate optimal batch size
    available_memory = gpu_memory * 0.8  # Use 80% of GPU memory
    sample_memory = 0.1  # GB per sample (estimate)
    max_batch_size = int((available_memory - model_memory) / sample_memory)
    max_batch_size = max(1, min(max_batch_size, 32))  # Clamp between 1 and 32
    
    # Calculate accumulation steps
    accumulation_steps = max(1, target_batch_size // max_batch_size)
    
    config = GradientAccumulationConfig(
        accumulation_steps=accumulation_steps,
        effective_batch_size=target_batch_size,
        max_batch_size=max_batch_size,
        use_amp=True,
        auto_adjust=True,
        memory_threshold=0.8
    )
    
    return config

def create_demo_model_and_dataset():
    """Create demo model and dataset for testing."""
    print("Creating demo model and dataset...")
    
    # Create model
    model = LargeVideoModel(input_channels=3, num_classes=10)
    
    # Create dataset
    dataset = MemoryIntensiveDataset(num_samples=500, image_size=224)
    
    print(f"✅ Created model with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"✅ Created dataset with {len(dataset)} samples")
    
    return model, dataset

def run_quick_training(config, epochs=3):
    """Run quick training session to verify setup."""
    print(f"\nStarting quick training session ({epochs} epochs)...")
    
    # Create model and dataset
    model, dataset = create_demo_model_and_dataset()
    
    # Create data loader
    train_loader = DataLoader(
        dataset,
        batch_size=config.max_batch_size,
        shuffle=True,
        num_workers=2
    )
    
    # Create trainer
    trainer = GradientAccumulationTrainer(
        model=model,
        train_loader=train_loader,
        config=config,
        optimizer=optim.Adam(model.parameters(), lr=1e-3),
        loss_fn=nn.CrossEntropyLoss()
    )
    
    # Training loop
    start_time = time.time()
    
    try:
        for epoch in range(epochs):
            metrics = trainer.train_epoch(epoch)
            
            print(f"Epoch {epoch}: Loss={metrics['train_loss']:.4f}, "
                  f"Effective Batch Size={metrics['effective_batch_size']}")
        
        training_time = time.time() - start_time
        print(f"✅ Quick training completed in {training_time:.2f} seconds")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False
    
    return True

def run_memory_benchmark():
    """Run memory usage benchmark."""
    print("\nRunning memory usage benchmark...")
    
    try:
        # Test different configurations
        configs = [
            ("Small Batch", create_accumulation_config(64, 4)),
            ("Medium Batch", create_accumulation_config(128, 8)),
            ("Large Batch", create_accumulation_config(256, 16))
        ]
        
        results = {}
        
        for name, config in configs:
            print(f"\nTesting {name} configuration...")
            
            # Create model and dataset
            model = LargeVideoModel(input_channels=3, num_classes=10)
            dataset = MemoryIntensiveDataset(num_samples=100, image_size=224)
            
            # Create data loader
            train_loader = DataLoader(
                dataset,
                batch_size=config.max_batch_size,
                shuffle=True,
                num_workers=2
            )
            
            # Create trainer
            trainer = GradientAccumulationTrainer(
                model=model,
                train_loader=train_loader,
                config=config,
                optimizer=optim.Adam(model.parameters(), lr=1e-3),
                loss_fn=nn.CrossEntropyLoss()
            )
            
            # Run one epoch
            start_time = time.time()
            metrics = trainer.train_epoch(0)
            training_time = time.time() - start_time
            
            # Get status
            status = trainer.get_status()
            
            results[name] = {
                'training_time': training_time,
                'samples_per_second': len(dataset) / training_time,
                'memory_usage': status['memory_usage']['gpu_utilization'],
                'memory_allocated': status['memory_usage']['memory_allocated'],
                'effective_batch_size': metrics['effective_batch_size']
            }
            
            print(f"  ✅ Completed: {training_time:.2f}s, "
                  f"{len(dataset) / training_time:.2f} samples/s, "
                  f"Memory: {status['memory_usage']['gpu_utilization']:.2f}")
        
        # Print summary
        print("\n" + "="*50)
        print("MEMORY BENCHMARK RESULTS")
        print("="*50)
        
        for name, result in results.items():
            print(f"\n{name}:")
            print(f"  Training time: {result['training_time']:.2f}s")
            print(f"  Samples per second: {result['samples_per_second']:.2f}")
            print(f"  Memory usage: {result['memory_usage']:.2f}")
            print(f"  Memory allocated: {result['memory_allocated']:.2f}GB")
            print(f"  Effective batch size: {result['effective_batch_size']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Memory benchmark failed: {e}")
        return False

def interactive_setup():
    """Interactive setup for gradient accumulation."""
    print("\n" + "="*60)
    print("Interactive Gradient Accumulation Setup")
    print("="*60)
    
    # Check requirements
    if not check_requirements():
        return False
    
    # Get user preferences
    print("\nConfiguration Options:")
    print("1. Small effective batch size (64)")
    print("2. Medium effective batch size (128)")
    print("3. Large effective batch size (256)")
    print("4. Custom effective batch size")
    
    choice = input("\nSelect option (1-4): ").strip()
    
    if choice == "1":
        target_batch_size = 64
    elif choice == "2":
        target_batch_size = 128
    elif choice == "3":
        target_batch_size = 256
    elif choice == "4":
        try:
            target_batch_size = int(input("Enter effective batch size: "))
        except ValueError:
            print("Invalid input, using default (128)")
            target_batch_size = 128
    else:
        print("Invalid choice, using default (128)")
        target_batch_size = 128
    
    # Get optimal configuration
    config = get_optimal_configuration(target_batch_size)
    
    print(f"\nOptimal Configuration:")
    print(f"  Target effective batch size: {target_batch_size}")
    print(f"  Physical batch size: {config.max_batch_size}")
    print(f"  Accumulation steps: {config.accumulation_steps}")
    print(f"  Mixed precision: {config.use_amp}")
    print(f"  Auto-adjust: {config.auto_adjust}")
    
    # Ask for confirmation
    response = input("\nProceed with this configuration? (y/n): ").lower()
    if response != 'y':
        print("Setup cancelled.")
        return False
    
    return config

def main():
    """Main function for quick start gradient accumulation."""
    parser = argparse.ArgumentParser(description="Quick Start Gradient Accumulation")
    parser.add_argument("--check", action="store_true", help="Check requirements only")
    parser.add_argument("--benchmark", action="store_true", help="Run memory benchmark")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--interactive", action="store_true", help="Interactive setup")
    parser.add_argument("--target-batch-size", type=int, default=128, help="Target effective batch size")
    parser.add_argument("--max-batch-size", type=int, help="Maximum physical batch size")
    
    args = parser.parse_args()
    
    print("🚀 Quick Start Gradient Accumulation for Video-OpusClip")
    print("="*60)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    if args.check:
        print("✅ Requirements check completed successfully!")
        return
    
    # Get configuration
    if args.interactive:
        config = interactive_setup()
        if not config:
            return
    else:
        # Use automatic configuration
        print("Using automatic configuration...")
        config = get_optimal_configuration(args.target_batch_size)
        
        # Override max batch size if specified
        if args.max_batch_size:
            config.max_batch_size = args.max_batch_size
            config.accumulation_steps = max(1, config.effective_batch_size // config.max_batch_size)
    
    print(f"\nConfiguration: {config}")
    
    # Run memory benchmark if requested
    if args.benchmark:
        if not run_memory_benchmark():
            sys.exit(1)
    
    # Run quick training
    if not run_quick_training(config, args.epochs):
        sys.exit(1)
    
    print("\n" + "="*60)
    print("✅ Gradient accumulation setup completed successfully!")
    print("="*60)
    print("\nNext steps:")
    print("1. Review the training logs above")
    print("2. Check the GRADIENT_ACCUMULATION_GUIDE.md for detailed usage")
    print("3. Run gradient_accumulation_examples.py for more examples")
    print("4. Customize the configuration for your specific use case")
    print("\nKey benefits achieved:")
    print(f"  • Effective batch size: {config.effective_batch_size}")
    print(f"  • Physical batch size: {config.max_batch_size}")
    print(f"  • Memory efficiency: {config.accumulation_steps}x accumulation")
    print(f"  • Mixed precision: {'Enabled' if config.use_amp else 'Disabled'}")

if __name__ == "__main__":
    main() 