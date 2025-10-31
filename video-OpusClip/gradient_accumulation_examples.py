"""
Gradient Accumulation Examples for Video-OpusClip

Comprehensive examples demonstrating gradient accumulation implementation
with various configurations, memory management, and performance optimization.
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

# Import gradient accumulation components
try:
    from gradient_accumulation import (
        GradientAccumulationConfig, GradientAccumulationManager,
        GradientAccumulationTrainer, MemoryMonitor, PerformanceTracker,
        create_accumulation_config, calculate_optimal_accumulation_steps
    )
    GRADIENT_ACCUMULATION_AVAILABLE = True
except ImportError:
    GRADIENT_ACCUMULATION_AVAILABLE = False
    warnings.warn("Gradient accumulation components not available")

logger = structlog.get_logger()

# =============================================================================
# EXAMPLE MODELS AND DATASETS
# =============================================================================

class LargeVideoModel(nn.Module):
    """Large video model that benefits from gradient accumulation."""
    
    def __init__(self, input_channels: int = 3, num_classes: int = 10):
        super().__init__()
        
        # Large feature extraction layers
        self.features = nn.Sequential(
            # First block
            nn.Conv2d(input_channels, 128, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            
            # Second block
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Third block
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            # Fourth block
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            
            # Global pooling
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Large classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes)
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

class MemoryIntensiveDataset(Dataset):
    """Memory-intensive dataset for testing gradient accumulation."""
    
    def __init__(self, num_samples: int = 1000, image_size: int = 512, 
                 num_frames: int = 32, num_classes: int = 10):
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_frames = num_frames
        self.num_classes = num_classes
        
        # Generate large synthetic data
        self.data = torch.randn(num_samples, num_frames, 3, image_size, image_size)
        self.labels = torch.randint(0, num_classes, (num_samples,))
        
        logger.info(f"Created memory-intensive dataset: {num_samples} samples, "
                   f"{num_frames} frames, {image_size}x{image_size} images")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Return single frame for simplicity (can be extended to video)
        frame_idx = torch.randint(0, self.num_frames, (1,)).item()
        return self.data[idx, frame_idx], self.labels[idx]

# =============================================================================
# EXAMPLE 1: BASIC GRADIENT ACCUMULATION
# =============================================================================

def example_basic_gradient_accumulation():
    """Example 1: Basic gradient accumulation setup."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Gradient Accumulation")
    print("="*60)
    
    if not GRADIENT_ACCUMULATION_AVAILABLE:
        print("Gradient accumulation components not available")
        return
    
    # Create model and dataset
    model = LargeVideoModel(input_channels=3, num_classes=10)
    dataset = MemoryIntensiveDataset(num_samples=500, image_size=224)
    
    # Create data loader with small batch size
    train_loader = DataLoader(
        dataset,
        batch_size=8,  # Small batch size
        shuffle=True,
        num_workers=2
    )
    
    # Create gradient accumulation configuration
    config = create_accumulation_config(
        target_batch_size=128,  # Target effective batch size
        max_batch_size=8,       # Physical batch size
        use_amp=True            # Use mixed precision
    )
    
    print(f"Configuration: {config}")
    print(f"Effective batch size: {config.effective_batch_size}")
    print(f"Accumulation steps: {config.accumulation_steps}")
    
    # Create trainer
    trainer = GradientAccumulationTrainer(
        model=model,
        train_loader=train_loader,
        config=config,
        optimizer=optim.Adam(model.parameters(), lr=1e-3),
        loss_fn=nn.CrossEntropyLoss()
    )
    
    print("Starting basic gradient accumulation training...")
    
    # Training loop
    for epoch in range(3):
        start_time = time.time()
        
        metrics = trainer.train_epoch(epoch)
        
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch}:")
        print(f"  Loss: {metrics['train_loss']:.4f}")
        print(f"  Accuracy: {metrics['train_accuracy']:.4f}")
        print(f"  Time: {epoch_time:.2f}s")
        print(f"  Effective batch size: {metrics['effective_batch_size']}")
        print(f"  Accumulation count: {metrics['accumulation_count']}")
    
    print("Basic gradient accumulation training completed!")

# =============================================================================
# EXAMPLE 2: MEMORY-AWARE GRADIENT ACCUMULATION
# =============================================================================

def example_memory_aware_accumulation():
    """Example 2: Memory-aware gradient accumulation with automatic adjustment."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Memory-Aware Gradient Accumulation")
    print("="*60)
    
    if not GRADIENT_ACCUMULATION_AVAILABLE:
        print("Gradient accumulation components not available")
        return
    
    # Create large model and dataset
    model = LargeVideoModel(input_channels=3, num_classes=10)
    dataset = MemoryIntensiveDataset(num_samples=500, image_size=256)
    
    # Create data loader
    train_loader = DataLoader(
        dataset,
        batch_size=4,  # Very small batch size
        shuffle=True,
        num_workers=2
    )
    
    # Create memory-aware configuration
    config = GradientAccumulationConfig(
        accumulation_steps=8,
        effective_batch_size=128,
        max_batch_size=4,
        strategy='adaptive',
        memory_threshold=0.7,  # Conservative threshold
        auto_adjust=True,
        use_amp=True,
        log_accumulation=True,
        log_frequency=5
    )
    
    print(f"Configuration: {config}")
    
    # Create trainer
    trainer = GradientAccumulationTrainer(
        model=model,
        train_loader=train_loader,
        config=config,
        optimizer=optim.Adam(model.parameters(), lr=1e-3),
        loss_fn=nn.CrossEntropyLoss()
    )
    
    print("Starting memory-aware gradient accumulation training...")
    
    # Training loop with memory monitoring
    for epoch in range(3):
        print(f"\nEpoch {epoch}:")
        
        metrics = trainer.train_epoch(epoch)
        
        # Get detailed status
        status = trainer.get_status()
        
        print(f"  Loss: {metrics['train_loss']:.4f}")
        print(f"  Effective batch size: {metrics['effective_batch_size']}")
        print(f"  Memory usage: {status['memory_usage']['gpu_utilization']:.2f}")
        print(f"  Memory allocated: {status['memory_usage']['memory_allocated']:.2f}GB")
        
        # Check if accumulation steps were adjusted
        if status['accumulation_steps'] != config.accumulation_steps:
            print(f"  ⚠️  Accumulation steps adjusted to: {status['accumulation_steps']}")
    
    print("Memory-aware gradient accumulation training completed!")

# =============================================================================
# EXAMPLE 3: PERFORMANCE OPTIMIZATION
# =============================================================================

def example_performance_optimization():
    """Example 3: Performance optimization with gradient accumulation."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Performance Optimization")
    print("="*60)
    
    if not GRADIENT_ACCUMULATION_AVAILABLE:
        print("Gradient accumulation components not available")
        return
    
    # Create model and dataset
    model = LargeVideoModel(input_channels=3, num_classes=10)
    dataset = MemoryIntensiveDataset(num_samples=1000, image_size=224)
    
    # Create optimized data loader
    train_loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    # Create performance-optimized configuration
    config = GradientAccumulationConfig(
        accumulation_steps=4,
        effective_batch_size=256,
        max_batch_size=16,
        use_amp=True,
        gradient_clip_norm=1.0,
        log_accumulation=True,
        log_frequency=10
    )
    
    print(f"Configuration: {config}")
    
    # Create trainer
    trainer = GradientAccumulationTrainer(
        model=model,
        train_loader=train_loader,
        config=config,
        optimizer=optim.Adam(model.parameters(), lr=1e-3),
        loss_fn=nn.CrossEntropyLoss()
    )
    
    print("Starting performance-optimized training...")
    
    # Training loop with performance monitoring
    for epoch in range(3):
        start_time = time.time()
        
        metrics = trainer.train_epoch(epoch)
        
        epoch_time = time.time() - start_time
        
        # Get performance metrics
        status = trainer.get_status()
        perf_metrics = status['performance_metrics']
        
        print(f"Epoch {epoch}:")
        print(f"  Loss: {metrics['train_loss']:.4f}")
        print(f"  Time: {epoch_time:.2f}s")
        print(f"  Samples per second: {len(dataset) / epoch_time:.2f}")
        print(f"  Avg accumulation time: {perf_metrics['accumulation_time_mean']:.3f}s")
        print(f"  Avg update time: {perf_metrics['update_time_mean']:.3f}s")
        print(f"  Memory peak: {perf_metrics['memory_peak_max']:.2f}GB")
    
    print("Performance-optimized training completed!")

# =============================================================================
# EXAMPLE 4: CUSTOM TRAINING LOOP
# =============================================================================

def example_custom_training_loop():
    """Example 4: Custom training loop with gradient accumulation."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Custom Training Loop")
    print("="*60)
    
    if not GRADIENT_ACCUMULATION_AVAILABLE:
        print("Gradient accumulation components not available")
        return
    
    # Create model and dataset
    model = LargeVideoModel(input_channels=3, num_classes=10)
    dataset = MemoryIntensiveDataset(num_samples=500, image_size=224)
    
    # Create data loader
    train_loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=2
    )
    
    # Create configuration
    config = GradientAccumulationConfig(
        accumulation_steps=4,
        effective_batch_size=128,
        max_batch_size=8,
        use_amp=True,
        auto_adjust=True,
        memory_threshold=0.8
    )
    
    # Create manager
    manager = GradientAccumulationManager(config)
    
    # Create optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    
    # Create scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    print("Starting custom training loop...")
    
    # Custom training loop
    for epoch in range(3):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        print(f"\nEpoch {epoch}:")
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            batch_start_time = time.time()
            
            # Move to device
            inputs = inputs.cuda()
            targets = targets.cuda()
            
            # Forward pass
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            
            # Accumulate gradients
            manager.accumulate_gradients(loss, model)
            
            # Record accumulation time
            accumulation_time = time.time() - batch_start_time
            manager.performance_tracker.record_accumulation_time(accumulation_time)
            
            # Check if we should update
            if manager.should_update(batch_idx):
                update_start_time = time.time()
                
                # Check memory and adjust if needed
                adjusted = manager.check_memory_and_adjust(model)
                if adjusted:
                    print(f"  ⚠️  Accumulation steps adjusted at batch {batch_idx}")
                
                # Update optimizer
                update_info = manager.update_optimizer(optimizer, model)
                
                # Update scheduler
                scheduler.step()
                
                # Record update time
                update_time = time.time() - update_start_time
                manager.performance_tracker.record_update_time(update_time)
                
                print(f"  Batch {batch_idx}: Updated optimizer, "
                      f"accumulated={update_info['gradients_accumulated']} steps")
            
            # Update metrics
            epoch_loss += loss.item()
            num_batches += 1
            
            # Log progress
            if batch_idx % 10 == 0:
                status = manager.get_status()
                print(f"  Batch {batch_idx}: Loss={loss.item():.4f}, "
                      f"Memory={status['memory_usage']['gpu_utilization']:.2f}")
        
        # Calculate epoch metrics
        avg_loss = epoch_loss / num_batches
        print(f"  Epoch {epoch} completed: Avg Loss={avg_loss:.4f}")
        
        # Get final status
        status = manager.get_status()
        print(f"  Final effective batch size: {status['effective_batch_size']}")
        print(f"  Final accumulation steps: {status['accumulation_steps']}")
    
    print("Custom training loop completed!")

# =============================================================================
# EXAMPLE 5: MEMORY MONITORING AND OPTIMIZATION
# =============================================================================

def example_memory_monitoring():
    """Example 5: Advanced memory monitoring and optimization."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Memory Monitoring and Optimization")
    print("="*60)
    
    if not GRADIENT_ACCUMULATION_AVAILABLE:
        print("Gradient accumulation components not available")
        return
    
    # Create memory monitor
    monitor = MemoryMonitor()
    
    # Create model and dataset
    model = LargeVideoModel(input_channels=3, num_classes=10)
    dataset = MemoryIntensiveDataset(num_samples=300, image_size=256)
    
    # Create data loader
    train_loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=2
    )
    
    # Create configuration with aggressive memory management
    config = GradientAccumulationConfig(
        accumulation_steps=16,  # Large accumulation for memory efficiency
        effective_batch_size=128,
        max_batch_size=4,
        strategy='adaptive',
        memory_threshold=0.6,  # Conservative threshold
        auto_adjust=True,
        use_amp=True,
        log_accumulation=True,
        log_frequency=5
    )
    
    # Create trainer
    trainer = GradientAccumulationTrainer(
        model=model,
        train_loader=train_loader,
        config=config,
        optimizer=optim.Adam(model.parameters(), lr=1e-3),
        loss_fn=nn.CrossEntropyLoss()
    )
    
    print("Starting memory-monitored training...")
    
    # Training loop with detailed memory monitoring
    for epoch in range(2):
        print(f"\nEpoch {epoch}:")
        
        # Get initial memory state
        initial_memory = monitor.get_memory_usage()
        print(f"  Initial memory: {initial_memory['memory_allocated']:.2f}GB allocated, "
              f"{initial_memory['gpu_utilization']:.2f} utilization")
        
        # Train epoch
        metrics = trainer.train_epoch(epoch)
        
        # Get final memory state
        final_memory = monitor.get_memory_usage()
        print(f"  Final memory: {final_memory['memory_allocated']:.2f}GB allocated, "
              f"{final_memory['gpu_utilization']:.2f} utilization")
        
        # Get memory trend
        trend = monitor.get_memory_trend()
        print(f"  Memory trend: {trend['trend']}, change rate: {trend['change_rate']:.3f}")
        
        # Get recommendations
        recommendations = monitor.get_memory_recommendations()
        if recommendations:
            print("  Memory recommendations:")
            for rec in recommendations:
                print(f"    - {rec}")
        
        # Get training metrics
        status = trainer.get_status()
        print(f"  Loss: {metrics['train_loss']:.4f}")
        print(f"  Effective batch size: {metrics['effective_batch_size']}")
        print(f"  Accumulation steps: {status['accumulation_steps']}")
    
    print("Memory-monitored training completed!")

# =============================================================================
# EXAMPLE 6: BENCHMARKING DIFFERENT CONFIGURATIONS
# =============================================================================

def example_benchmarking():
    """Example 6: Benchmarking different gradient accumulation configurations."""
    print("\n" + "="*60)
    print("EXAMPLE 6: Configuration Benchmarking")
    print("="*60)
    
    if not GRADIENT_ACCUMULATION_AVAILABLE:
        print("Gradient accumulation components not available")
        return
    
    # Create model and dataset
    model = LargeVideoModel(input_channels=3, num_classes=10)
    dataset = MemoryIntensiveDataset(num_samples=200, image_size=224)
    
    # Test different configurations
    configurations = [
        {
            'name': 'Small Batch, High Accumulation',
            'config': GradientAccumulationConfig(
                accumulation_steps=16,
                effective_batch_size=128,
                max_batch_size=4,
                use_amp=True
            )
        },
        {
            'name': 'Medium Batch, Medium Accumulation',
            'config': GradientAccumulationConfig(
                accumulation_steps=8,
                effective_batch_size=128,
                max_batch_size=8,
                use_amp=True
            )
        },
        {
            'name': 'Large Batch, Low Accumulation',
            'config': GradientAccumulationConfig(
                accumulation_steps=4,
                effective_batch_size=128,
                max_batch_size=16,
                use_amp=True
            )
        }
    ]
    
    results = {}
    
    for config_info in configurations:
        name = config_info['name']
        config = config_info['config']
        
        print(f"\nBenchmarking: {name}")
        print(f"  Accumulation steps: {config.accumulation_steps}")
        print(f"  Max batch size: {config.max_batch_size}")
        print(f"  Effective batch size: {config.effective_batch_size}")
        
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
        
        # Benchmark
        start_time = time.time()
        
        try:
            metrics = trainer.train_epoch(0)  # Train one epoch
            
            training_time = time.time() - start_time
            
            # Get status
            status = trainer.get_status()
            
            results[name] = {
                'training_time': training_time,
                'samples_per_second': len(dataset) / training_time,
                'final_loss': metrics['train_loss'],
                'memory_usage': status['memory_usage']['gpu_utilization'],
                'memory_allocated': status['memory_usage']['memory_allocated'],
                'performance_metrics': status['performance_metrics']
            }
            
            print(f"  ✅ Completed: {training_time:.2f}s, "
                  f"{len(dataset) / training_time:.2f} samples/s, "
                  f"Loss: {metrics['train_loss']:.4f}")
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            results[name] = {'error': str(e)}
    
    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*60)
    
    for name, result in results.items():
        print(f"\n{name}:")
        if 'error' in result:
            print(f"  Error: {result['error']}")
        else:
            print(f"  Training time: {result['training_time']:.2f}s")
            print(f"  Samples per second: {result['samples_per_second']:.2f}")
            print(f"  Final loss: {result['final_loss']:.4f}")
            print(f"  Memory usage: {result['memory_usage']:.2f}")
            print(f"  Memory allocated: {result['memory_allocated']:.2f}GB")
    
    # Save results
    with open('gradient_accumulation_benchmark.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to gradient_accumulation_benchmark.json")

# =============================================================================
# EXAMPLE 7: INTEGRATION WITH EXISTING TRAINING
# =============================================================================

def example_integration():
    """Example 7: Integration with existing training pipeline."""
    print("\n" + "="*60)
    print("EXAMPLE 7: Integration with Existing Training")
    print("="*60)
    
    if not GRADIENT_ACCUMULATION_AVAILABLE:
        print("Gradient accumulation components not available")
        return
    
    # Simulate existing training setup
    class ExistingTrainer:
        def __init__(self, model, train_loader, optimizer, loss_fn):
            self.model = model
            self.train_loader = train_loader
            self.optimizer = optimizer
            self.loss_fn = loss_fn
            self.device = "cuda"
        
        def train_epoch(self, epoch):
            self.model.train()
            total_loss = 0.0
            num_batches = 0
            
            for batch_idx, (inputs, targets) in enumerate(self.train_loader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Standard training step
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            return total_loss / num_batches
    
    # Create model and dataset
    model = LargeVideoModel(input_channels=3, num_classes=10)
    dataset = MemoryIntensiveDataset(num_samples=500, image_size=224)
    
    # Create data loader
    train_loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=2
    )
    
    # Create optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    
    # Create existing trainer
    existing_trainer = ExistingTrainer(model, train_loader, optimizer, loss_fn)
    
    print("Original training (without gradient accumulation):")
    
    # Train with original method
    start_time = time.time()
    original_loss = existing_trainer.train_epoch(0)
    original_time = time.time() - start_time
    
    print(f"  Time: {original_time:.2f}s")
    print(f"  Loss: {original_loss:.4f}")
    
    # Now enhance with gradient accumulation
    print("\nEnhanced training (with gradient accumulation):")
    
    # Create gradient accumulation configuration
    config = create_accumulation_config(
        target_batch_size=128,
        max_batch_size=8,
        use_amp=True
    )
    
    # Create enhanced trainer
    enhanced_trainer = GradientAccumulationTrainer(
        model=model,
        train_loader=train_loader,
        config=config,
        optimizer=optimizer,
        loss_fn=loss_fn
    )
    
    # Train with gradient accumulation
    start_time = time.time()
    metrics = enhanced_trainer.train_epoch(0)
    enhanced_time = time.time() - start_time
    
    print(f"  Time: {enhanced_time:.2f}s")
    print(f"  Loss: {metrics['train_loss']:.4f}")
    print(f"  Effective batch size: {metrics['effective_batch_size']}")
    
    # Compare results
    print(f"\nComparison:")
    print(f"  Original time: {original_time:.2f}s")
    print(f"  Enhanced time: {enhanced_time:.2f}s")
    print(f"  Speedup: {original_time / enhanced_time:.2f}x")
    print(f"  Original loss: {original_loss:.4f}")
    print(f"  Enhanced loss: {metrics['train_loss']:.4f}")
    
    print("Integration example completed!")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_all_examples():
    """Run all gradient accumulation examples."""
    print("Gradient Accumulation Examples for Video-OpusClip")
    print("="*60)
    
    if not GRADIENT_ACCUMULATION_AVAILABLE:
        print("Gradient accumulation components not available.")
        print("Please ensure gradient_accumulation.py is properly imported.")
        return
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("CUDA is not available. Gradient accumulation examples require CUDA.")
        return
    
    # Run examples
    examples = [
        example_basic_gradient_accumulation,
        example_memory_aware_accumulation,
        example_performance_optimization,
        example_custom_training_loop,
        example_memory_monitoring,
        example_benchmarking,
        example_integration
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