#!/usr/bin/env python3
"""
Quick Start Multi-GPU Training for Video-OpusClip

Easy setup and execution of multi-GPU training with automatic configuration
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

# Import multi-GPU components
try:
    from multi_gpu_training import (
        MultiGPUConfig, MultiGPUTrainingManager,
        get_gpu_info, select_optimal_gpus,
        launch_multi_gpu_training
    )
    from multi_gpu_training_examples import (
        VideoClassificationModel, SyntheticVideoDataset
    )
    MULTI_GPU_AVAILABLE = True
except ImportError as e:
    print(f"Error importing multi-GPU components: {e}")
    print("Please ensure multi_gpu_training.py is in the same directory.")
    MULTI_GPU_AVAILABLE = False

def check_requirements():
    """Check if all requirements are met for multi-GPU training."""
    print("Checking requirements...")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA is not available. Multi-GPU training requires CUDA.")
        return False
    
    # Check PyTorch version
    pytorch_version = torch.__version__
    print(f"✅ PyTorch version: {pytorch_version}")
    
    # Check GPU availability
    gpu_count = torch.cuda.device_count()
    if gpu_count == 0:
        print("❌ No CUDA devices found.")
        return False
    
    print(f"✅ Found {gpu_count} CUDA device(s)")
    
    # Check multi-GPU components
    if not MULTI_GPU_AVAILABLE:
        print("❌ Multi-GPU training components not available.")
        return False
    
    print("✅ Multi-GPU training components available")
    
    return True

def get_optimal_configuration():
    """Get optimal configuration based on available hardware."""
    gpu_info = get_gpu_info()
    
    # Select optimal GPUs
    optimal_gpus = select_optimal_gpus(
        num_gpus=min(4, gpu_info['count']),
        min_memory_gb=4.0
    )
    
    # Calculate optimal batch size
    total_memory = sum(
        gpu_info['memory'][gpu_id]['free'] 
        for gpu_id in optimal_gpus
    )
    estimated_memory_per_sample = 0.1 * 1024**3  # 100MB per sample
    optimal_batch_size = int(total_memory * 0.7 / estimated_memory_per_sample / len(optimal_gpus))
    optimal_batch_size = max(8, min(optimal_batch_size, 128))
    
    # Determine strategy
    if len(optimal_gpus) <= 4:
        strategy = 'dataparallel'
    else:
        strategy = 'distributed'
    
    config = MultiGPUConfig(
        strategy=strategy,
        num_gpus=len(optimal_gpus),
        gpu_ids=optimal_gpus,
        batch_size=optimal_batch_size,
        num_workers=min(8, len(optimal_gpus) * 2),
        pin_memory=True,
        persistent_workers=True
    )
    
    return config

def create_demo_model_and_dataset():
    """Create demo model and dataset for testing."""
    print("Creating demo model and dataset...")
    
    # Create model
    model = VideoClassificationModel(num_classes=10)
    
    # Create datasets
    train_dataset = SyntheticVideoDataset(num_samples=1000, image_size=224)
    val_dataset = SyntheticVideoDataset(num_samples=200, image_size=224)
    
    print(f"✅ Created model with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"✅ Created datasets: {len(train_dataset)} train, {len(val_dataset)} val samples")
    
    return model, train_dataset, val_dataset

def run_quick_training(config, epochs=5):
    """Run quick training session to verify setup."""
    print(f"\nStarting quick training session ({epochs} epochs)...")
    
    # Create model and dataset functions
    def create_model():
        return VideoClassificationModel(num_classes=10)
    
    def create_datasets():
        train_dataset = SyntheticVideoDataset(num_samples=1000, image_size=224)
        val_dataset = SyntheticVideoDataset(num_samples=200, image_size=224)
        return train_dataset, val_dataset
    
    # Launch training
    start_time = time.time()
    
    try:
        launch_multi_gpu_training(
            model_fn=create_model,
            dataset_fn=create_datasets,
            config=config,
            epochs=epochs,
            learning_rate=1e-3,
            weight_decay=1e-5
        )
        
        training_time = time.time() - start_time
        print(f"✅ Quick training completed in {training_time:.2f} seconds")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False
    
    return True

def run_benchmark(config):
    """Run performance benchmark."""
    print("\nRunning performance benchmark...")
    
    try:
        from multi_gpu_training import benchmark_multi_gpu_performance
        
        # Create model and dataset
        model = VideoClassificationModel(num_classes=10)
        dataset = SyntheticVideoDataset(num_samples=500, image_size=224)
        
        # Run benchmark
        results = benchmark_multi_gpu_performance(model, dataset, config)
        
        print("📊 Benchmark Results:")
        print(f"  Training time: {results['training_time']:.2f}s")
        print(f"  Samples per second: {results['samples_per_second']:.2f}")
        print(f"  GPU utilization: {results['gpu_utilization']['gpu_info']['count']} GPUs")
        
        return True
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        return False

def interactive_setup():
    """Interactive setup for multi-GPU training."""
    print("\n" + "="*60)
    print("Interactive Multi-GPU Training Setup")
    print("="*60)
    
    # Check requirements
    if not check_requirements():
        return False
    
    # Get GPU information
    gpu_info = get_gpu_info()
    print(f"\nGPU Information:")
    for device in gpu_info['devices']:
        print(f"  GPU {device['id']}: {device['name']}")
        print(f"    Memory: {device['memory_total'] / 1024**3:.1f}GB total, "
              f"{device['memory_free'] / 1024**3:.1f}GB free")
    
    # Get optimal configuration
    config = get_optimal_configuration()
    print(f"\nOptimal Configuration:")
    print(f"  Strategy: {config.strategy}")
    print(f"  GPUs: {config.num_gpus} ({config.gpu_ids})")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Workers: {config.num_workers}")
    
    # Ask user for confirmation
    response = input("\nProceed with this configuration? (y/n): ").lower()
    if response != 'y':
        print("Setup cancelled.")
        return False
    
    return config

def main():
    """Main function for quick start multi-GPU training."""
    parser = argparse.ArgumentParser(description="Quick Start Multi-GPU Training")
    parser.add_argument("--check", action="store_true", help="Check requirements only")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--interactive", action="store_true", help="Interactive setup")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    
    args = parser.parse_args()
    
    print("🚀 Quick Start Multi-GPU Training for Video-OpusClip")
    print("="*60)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    if args.check:
        print("✅ Requirements check completed successfully!")
        return
    
    # Get configuration
    if args.config:
        # Load from file (implement if needed)
        print(f"Loading configuration from {args.config}")
        config = MultiGPUConfig()  # Default config
    elif args.interactive:
        config = interactive_setup()
        if not config:
            return
    else:
        # Use automatic configuration
        print("Using automatic configuration...")
        config = get_optimal_configuration()
    
    print(f"\nConfiguration: {config}")
    
    # Run benchmark if requested
    if args.benchmark:
        if not run_benchmark(config):
            sys.exit(1)
    
    # Run quick training
    if not run_quick_training(config, args.epochs):
        sys.exit(1)
    
    print("\n" + "="*60)
    print("✅ Multi-GPU training setup completed successfully!")
    print("="*60)
    print("\nNext steps:")
    print("1. Review the training logs above")
    print("2. Check the MULTI_GPU_TRAINING_GUIDE.md for detailed usage")
    print("3. Run multi_gpu_training_examples.py for more examples")
    print("4. Customize the configuration for your specific use case")

if __name__ == "__main__":
    main() 