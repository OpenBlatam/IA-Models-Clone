"""
Multi-GPU Training Examples for Video-OpusClip

Comprehensive examples demonstrating DataParallel and DistributedDataParallel
training with various configurations, optimizations, and best practices.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import numpy as np
import time
import json
import structlog
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import warnings

# Import multi-GPU training components
try:
    from multi_gpu_training import (
        MultiGPUConfig, MultiGPUTrainingManager,
        DataParallelTrainer, DistributedDataParallelTrainer,
        get_gpu_info, select_optimal_gpus,
        launch_distributed_training, launch_multi_gpu_training,
        benchmark_multi_gpu_performance
    )
    MULTI_GPU_AVAILABLE = True
except ImportError:
    MULTI_GPU_AVAILABLE = False
    warnings.warn("Multi-GPU training components not available")

logger = structlog.get_logger()

# =============================================================================
# EXAMPLE MODELS AND DATASETS
# =============================================================================

class VideoClassificationModel(nn.Module):
    """Example video classification model for multi-GPU training."""
    
    def __init__(self, num_classes: int = 10, input_channels: int = 3):
        super().__init__()
        
        # Feature extraction layers
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class VideoSegmentationModel(nn.Module):
    """Example video segmentation model for multi-GPU training."""
    
    def __init__(self, num_classes: int = 2, input_channels: int = 3):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, 1)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class SyntheticVideoDataset(Dataset):
    """Synthetic video dataset for testing multi-GPU training."""
    
    def __init__(self, num_samples: int = 1000, image_size: int = 224, 
                 num_frames: int = 16, num_classes: int = 10):
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_frames = num_frames
        self.num_classes = num_classes
        
        # Generate synthetic data
        self.data = torch.randn(num_samples, num_frames, 3, image_size, image_size)
        self.labels = torch.randint(0, num_classes, (num_samples,))
        
        logger.info(f"Created synthetic dataset: {num_samples} samples, "
                   f"{num_frames} frames, {image_size}x{image_size} images")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Return single frame for simplicity (can be extended to video)
        frame_idx = torch.randint(0, self.num_frames, (1,)).item()
        return self.data[idx, frame_idx], self.labels[idx]

# =============================================================================
# EXAMPLE 1: BASIC DATAPARALLEL TRAINING
# =============================================================================

def example_dataparallel_training():
    """Example 1: Basic DataParallel training setup."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic DataParallel Training")
    print("="*60)
    
    if not MULTI_GPU_AVAILABLE:
        print("Multi-GPU components not available")
        return
    
    # Check GPU availability
    gpu_info = get_gpu_info()
    print(f"Available GPUs: {gpu_info['count']}")
    
    if gpu_info['count'] < 2:
        print("Need at least 2 GPUs for DataParallel training")
        return
    
    # Create model and dataset
    model = VideoClassificationModel(num_classes=10)
    dataset = SyntheticVideoDataset(num_samples=1000, image_size=224)
    
    # Configure DataParallel training
    config = MultiGPUConfig(
        strategy='dataparallel',
        num_gpus=min(4, gpu_info['count']),
        batch_size=32,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Configuration: {config}")
    
    # Create data loader
    train_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    # Create trainer
    trainer = DataParallelTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        optimizer=optim.Adam(model.parameters(), lr=1e-3),
        loss_fn=nn.CrossEntropyLoss()
    )
    
    print("Starting DataParallel training...")
    
    # Training loop
    for epoch in range(5):
        start_time = time.time()
        
        # Training
        train_metrics = trainer.train_epoch(epoch)
        
        # Get memory stats
        memory_stats = trainer.get_memory_stats()
        
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch}: Loss={train_metrics['train_loss']:.4f}, "
              f"Time={epoch_time:.2f}s, Memory={memory_stats['total_allocated']:.2f}GB")
    
    print("DataParallel training completed!")

# =============================================================================
# EXAMPLE 2: DISTRIBUTEDDATAPARALLEL TRAINING
# =============================================================================

def example_distributed_training():
    """Example 2: DistributedDataParallel training setup."""
    print("\n" + "="*60)
    print("EXAMPLE 2: DistributedDataParallel Training")
    print("="*60)
    
    if not MULTI_GPU_AVAILABLE:
        print("Multi-GPU components not available")
        return
    
    # Check GPU availability
    gpu_info = get_gpu_info()
    print(f"Available GPUs: {gpu_info['count']}")
    
    if gpu_info['count'] < 2:
        print("Need at least 2 GPUs for distributed training")
        return
    
    # Define model and dataset functions
    def create_model():
        return VideoClassificationModel(num_classes=10)
    
    def create_datasets():
        train_dataset = SyntheticVideoDataset(num_samples=1000, image_size=224)
        val_dataset = SyntheticVideoDataset(num_samples=200, image_size=224)
        return train_dataset, val_dataset
    
    # Configure distributed training
    config = MultiGPUConfig(
        strategy='distributed',
        world_size=min(4, gpu_info['count']),
        backend='nccl',
        master_addr='localhost',
        master_port='12355',
        batch_size=32,
        num_workers=4
    )
    
    print(f"Configuration: {config}")
    print("Starting distributed training...")
    
    # Launch distributed training
    launch_distributed_training(
        rank=0,  # Will be set automatically
        world_size=config.world_size,
        model_fn=create_model,
        dataset_fn=create_datasets,
        config=config,
        epochs=5,
        learning_rate=1e-3,
        weight_decay=1e-5
    )
    
    print("Distributed training completed!")

# =============================================================================
# EXAMPLE 3: AUTOMATIC STRATEGY SELECTION
# =============================================================================

def example_automatic_strategy_selection():
    """Example 3: Automatic strategy selection based on GPU count."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Automatic Strategy Selection")
    print("="*60)
    
    if not MULTI_GPU_AVAILABLE:
        print("Multi-GPU components not available")
        return
    
    # Check GPU availability
    gpu_info = get_gpu_info()
    print(f"Available GPUs: {gpu_info['count']}")
    
    # Configure with automatic strategy selection
    config = MultiGPUConfig(
        strategy='auto',  # Will choose based on GPU count
        num_gpus=gpu_info['count'],
        batch_size=32,
        num_workers=4
    )
    
    print(f"Selected strategy: {config.strategy}")
    print(f"Configuration: {config}")
    
    # Define model and dataset functions
    def create_model():
        return VideoClassificationModel(num_classes=10)
    
    def create_datasets():
        train_dataset = SyntheticVideoDataset(num_samples=1000, image_size=224)
        val_dataset = SyntheticVideoDataset(num_samples=200, image_size=224)
        return train_dataset, val_dataset
    
    # Launch training with automatic strategy
    launch_multi_gpu_training(
        model_fn=create_model,
        dataset_fn=create_datasets,
        config=config,
        epochs=3,
        learning_rate=1e-3
    )
    
    print("Automatic strategy training completed!")

# =============================================================================
# EXAMPLE 4: PERFORMANCE BENCHMARKING
# =============================================================================

def example_performance_benchmarking():
    """Example 4: Benchmarking different multi-GPU configurations."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Performance Benchmarking")
    print("="*60)
    
    if not MULTI_GPU_AVAILABLE:
        print("Multi-GPU components not available")
        return
    
    # Check GPU availability
    gpu_info = get_gpu_info()
    print(f"Available GPUs: {gpu_info['count']}")
    
    if gpu_info['count'] < 2:
        print("Need at least 2 GPUs for benchmarking")
        return
    
    # Create model and dataset
    model = VideoClassificationModel(num_classes=10)
    dataset = SyntheticVideoDataset(num_samples=500, image_size=224)
    
    # Test different configurations
    configs = []
    
    # Single GPU baseline
    if gpu_info['count'] >= 1:
        configs.append(MultiGPUConfig(strategy='single', num_gpus=1))
    
    # DataParallel configurations
    for num_gpus in [2, 4]:
        if gpu_info['count'] >= num_gpus:
            configs.append(MultiGPUConfig(strategy='dataparallel', num_gpus=num_gpus))
    
    # Distributed configurations
    for world_size in [2, 4]:
        if gpu_info['count'] >= world_size:
            configs.append(MultiGPUConfig(strategy='distributed', world_size=world_size))
    
    print(f"Testing {len(configs)} configurations...")
    
    results = {}
    
    for i, config in enumerate(configs):
        print(f"\nBenchmarking configuration {i+1}/{len(configs)}: {config.strategy}")
        
        try:
            # Benchmark performance
            benchmark_results = benchmark_multi_gpu_performance(model, dataset, config)
            
            results[config.strategy] = {
                'training_time': benchmark_results['training_time'],
                'samples_per_second': benchmark_results['samples_per_second'],
                'gpu_utilization': benchmark_results['gpu_utilization']
            }
            
            print(f"  Training time: {benchmark_results['training_time']:.2f}s")
            print(f"  Samples per second: {benchmark_results['samples_per_second']:.2f}")
            
        except Exception as e:
            print(f"  Error benchmarking {config.strategy}: {e}")
            results[config.strategy] = {'error': str(e)}
    
    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*60)
    
    for strategy, result in results.items():
        if 'error' not in result:
            print(f"{strategy.upper()}:")
            print(f"  Training time: {result['training_time']:.2f}s")
            print(f"  Samples per second: {result['samples_per_second']:.2f}")
        else:
            print(f"{strategy.upper()}: Error - {result['error']}")
    
    # Save results
    with open('benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to benchmark_results.json")

# =============================================================================
# EXAMPLE 5: MEMORY OPTIMIZATION
# =============================================================================

def example_memory_optimization():
    """Example 5: Memory optimization techniques for multi-GPU training."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Memory Optimization")
    print("="*60)
    
    if not MULTI_GPU_AVAILABLE:
        print("Multi-GPU components not available")
        return
    
    # Check GPU availability
    gpu_info = get_gpu_info()
    print(f"Available GPUs: {gpu_info['count']}")
    
    if gpu_info['count'] < 2:
        print("Need at least 2 GPUs for memory optimization example")
        return
    
    # Create model with gradient checkpointing
    model = VideoClassificationModel(num_classes=10)
    
    # Enable gradient checkpointing if available
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")
    
    # Create dataset
    dataset = SyntheticVideoDataset(num_samples=1000, image_size=224)
    
    # Configure with memory optimizations
    config = MultiGPUConfig(
        strategy='dataparallel',
        num_gpus=min(4, gpu_info['count']),
        batch_size=16,  # Smaller batch size for memory efficiency
        num_workers=2,  # Fewer workers to reduce memory
        pin_memory=True
    )
    
    print(f"Configuration: {config}")
    
    # Create data loader with memory optimizations
    train_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    # Create trainer with memory monitoring
    trainer = DataParallelTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        optimizer=optim.Adam(model.parameters(), lr=1e-3),
        loss_fn=nn.CrossEntropyLoss()
    )
    
    print("Starting memory-optimized training...")
    
    # Training loop with memory monitoring
    for epoch in range(3):
        start_time = time.time()
        
        # Clear cache before training
        torch.cuda.empty_cache()
        
        # Training
        train_metrics = trainer.train_epoch(epoch)
        
        # Get detailed memory stats
        memory_stats = trainer.get_memory_stats()
        
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch}:")
        print(f"  Loss: {train_metrics['train_loss']:.4f}")
        print(f"  Time: {epoch_time:.2f}s")
        print(f"  Total allocated: {memory_stats['total_allocated']:.2f}GB")
        print(f"  Total cached: {memory_stats['total_cached']:.2f}GB")
        
        # Print per-device memory usage
        for device_id, device_memory in memory_stats['device_memory'].items():
            print(f"  GPU {device_id}: {device_memory['allocated']:.2f}GB allocated, "
                  f"{device_memory['cached']:.2f}GB cached")
    
    print("Memory-optimized training completed!")

# =============================================================================
# EXAMPLE 6: CUSTOM TRAINING LOOP
# =============================================================================

def example_custom_training_loop():
    """Example 6: Custom training loop with multi-GPU support."""
    print("\n" + "="*60)
    print("EXAMPLE 6: Custom Training Loop")
    print("="*60)
    
    if not MULTI_GPU_AVAILABLE:
        print("Multi-GPU components not available")
        return
    
    # Check GPU availability
    gpu_info = get_gpu_info()
    print(f"Available GPUs: {gpu_info['count']}")
    
    if gpu_info['count'] < 2:
        print("Need at least 2 GPUs for custom training loop example")
        return
    
    # Configure multi-GPU training
    config = MultiGPUConfig(
        strategy='auto',
        num_gpus=gpu_info['count'],
        batch_size=32,
        num_workers=4
    )
    
    print(f"Configuration: {config}")
    
    # Create manager
    manager = MultiGPUTrainingManager(config)
    
    # Create model and datasets
    model = VideoClassificationModel(num_classes=10)
    train_dataset = SyntheticVideoDataset(num_samples=1000, image_size=224)
    val_dataset = SyntheticVideoDataset(num_samples=200, image_size=224)
    
    # Create trainer
    trainer = manager.create_trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        optimizer=optim.Adam(model.parameters(), lr=1e-3),
        loss_fn=nn.CrossEntropyLoss()
    )
    
    print("Starting custom training loop...")
    
    # Custom training loop with advanced features
    best_loss = float('inf')
    patience = 3
    patience_counter = 0
    
    for epoch in range(10):
        # Training
        train_metrics = trainer.train_epoch(epoch)
        
        # Validation
        val_metrics = trainer.validate(epoch)
        
        # Logging (only on rank 0 for distributed training)
        if hasattr(trainer, 'rank') and trainer.rank == 0:
            print(f"Epoch {epoch}:")
            print(f"  Train Loss: {train_metrics['train_loss']:.4f}")
            if val_metrics:
                print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
        
        # Early stopping
        if val_metrics and val_metrics['val_loss'] < best_loss:
            best_loss = val_metrics['val_loss']
            patience_counter = 0
            
            # Save best model
            if hasattr(trainer, 'save_checkpoint'):
                trainer.save_checkpoint(epoch, "best_model.pth")
                print(f"  New best model saved (loss: {best_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping triggered after {patience} epochs without improvement")
                break
        
        # Get performance stats
        stats = manager.get_performance_stats()
        if hasattr(trainer, 'rank') and trainer.rank == 0:
            print(f"  GPU utilization: {stats['gpu_info']['count']} GPUs active")
    
    print("Custom training loop completed!")

# =============================================================================
# EXAMPLE 7: ERROR HANDLING AND RECOVERY
# =============================================================================

def example_error_handling():
    """Example 7: Error handling and recovery in multi-GPU training."""
    print("\n" + "="*60)
    print("EXAMPLE 7: Error Handling and Recovery")
    print("="*60)
    
    if not MULTI_GPU_AVAILABLE:
        print("Multi-GPU components not available")
        return
    
    # Check GPU availability
    gpu_info = get_gpu_info()
    print(f"Available GPUs: {gpu_info['count']}")
    
    if gpu_info['count'] < 2:
        print("Need at least 2 GPUs for error handling example")
        return
    
    # Configure training
    config = MultiGPUConfig(
        strategy='dataparallel',
        num_gpus=min(4, gpu_info['count']),
        batch_size=32,
        num_workers=4
    )
    
    print(f"Configuration: {config}")
    
    # Create model and dataset
    model = VideoClassificationModel(num_classes=10)
    dataset = SyntheticVideoDataset(num_samples=1000, image_size=224)
    
    # Create data loader
    train_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    # Create trainer
    trainer = DataParallelTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        optimizer=optim.Adam(model.parameters(), lr=1e-3),
        loss_fn=nn.CrossEntropyLoss()
    )
    
    print("Starting training with error handling...")
    
    # Training loop with error handling
    for epoch in range(5):
        try:
            start_time = time.time()
            
            # Training with error recovery
            train_metrics = trainer.train_epoch(epoch)
            
            epoch_time = time.time() - start_time
            
            print(f"Epoch {epoch}: Loss={train_metrics['train_loss']:.4f}, "
                  f"Time={epoch_time:.2f}s")
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"OOM error in epoch {epoch}: {e}")
                print("Attempting recovery...")
                
                # Clear cache
                torch.cuda.empty_cache()
                
                # Reduce batch size
                config.batch_size = config.batch_size // 2
                print(f"Reduced batch size to {config.batch_size}")
                
                # Recreate data loader with new batch size
                train_loader = DataLoader(
                    dataset,
                    batch_size=config.batch_size,
                    shuffle=True,
                    num_workers=config.num_workers,
                    pin_memory=config.pin_memory
                )
                
                # Update trainer
                trainer.train_loader = train_loader
                
                print("Recovery completed, continuing training...")
                
            else:
                print(f"Runtime error in epoch {epoch}: {e}")
                raise
        
        except Exception as e:
            print(f"Unexpected error in epoch {epoch}: {e}")
            print("Attempting to continue...")
            continue
    
    print("Error handling training completed!")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_all_examples():
    """Run all multi-GPU training examples."""
    print("Multi-GPU Training Examples for Video-OpusClip")
    print("="*60)
    
    if not MULTI_GPU_AVAILABLE:
        print("Multi-GPU training components not available.")
        print("Please ensure multi_gpu_training.py is properly imported.")
        return
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("CUDA is not available. Multi-GPU training requires CUDA.")
        return
    
    # Run examples
    examples = [
        example_dataparallel_training,
        example_distributed_training,
        example_automatic_strategy_selection,
        example_performance_benchmarking,
        example_memory_optimization,
        example_custom_training_loop,
        example_error_handling
    ]
    
    for i, example in enumerate(examples, 1):
        try:
            print(f"\nRunning Example {i}/{len(examples)}...")
            example()
        except Exception as e:
            print(f"Error running example {i}: {e}")
            continue
    
    print("\n" + "="*60)
    print("All examples completed!")
    print("="*60)

if __name__ == "__main__":
    run_all_examples() 