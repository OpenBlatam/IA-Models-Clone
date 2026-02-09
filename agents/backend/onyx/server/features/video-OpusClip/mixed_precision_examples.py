"""
Mixed Precision Training Examples for Video-OpusClip

Comprehensive examples demonstrating mixed precision training implementation
using torch.cuda.amp with various configurations and use cases.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
import structlog
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional, Any

# Import our mixed precision modules
from mixed_precision_training import (
    MixedPrecisionConfig,
    MixedPrecisionManager,
    MixedPrecisionTrainer,
    MixedPrecisionMemoryMonitor,
    benchmark_mixed_precision,
    create_mixed_precision_config,
    mixed_precision_context
)

logger = structlog.get_logger()

# =============================================================================
# EXAMPLE 1: BASIC MIXED PRECISION TRAINING
# =============================================================================

def example_basic_mixed_precision():
    """Basic mixed precision training example."""
    print("=== Example 1: Basic Mixed Precision Training ===")
    
    # Create synthetic dataset
    torch.manual_seed(42)
    train_data = torch.randn(1000, 784)
    train_labels = torch.randint(0, 10, (1000,))
    train_dataset = TensorDataset(train_data, train_labels)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        pin_memory=True
    )
    
    # Create model
    model = nn.Sequential(
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 10)
    )
    
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
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Training loop
    print("Starting training...")
    for epoch in range(3):
        metrics = trainer.train_epoch(epoch)
        print(f"Epoch {epoch}: Loss={metrics['train_loss']:.4f}, "
              f"Scaler Scale={metrics['scaler_scale']:.2e}")
    
    print("Training completed!")

# =============================================================================
# EXAMPLE 2: ADVANCED CONFIGURATION
# =============================================================================

def example_advanced_configuration():
    """Advanced mixed precision configuration example."""
    print("\n=== Example 2: Advanced Configuration ===")
    
    # Create advanced configuration
    config = MixedPrecisionConfig(
        # Basic settings
        enabled=True,
        dtype=torch.float16,
        
        # Gradient scaling
        init_scale=2**16,
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=2000,
        
        # Performance optimization
        cache_enabled=True,
        autocast_enabled=True,
        
        # Memory optimization
        memory_efficient=True,
        pin_memory=True,
        
        # Monitoring
        log_scaling=True,
        log_frequency=50,
        save_scaler_state=True,
        
        # Error handling
        handle_overflow=True,
        overflow_threshold=float('inf')
    )
    
    # Create synthetic dataset
    torch.manual_seed(42)
    train_data = torch.randn(2000, 1024)
    train_labels = torch.randint(0, 20, (2000,))
    train_dataset = TensorDataset(train_data, train_labels)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        pin_memory=True,
        num_workers=2
    )
    
    # Create complex model
    model = nn.Sequential(
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(0.3),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.BatchNorm1d(128),
        nn.Dropout(0.2),
        nn.Linear(128, 20)
    )
    
    # Create trainer with advanced configuration
    trainer = MixedPrecisionTrainer(
        model=model,
        train_loader=train_loader,
        config=config,
        optimizer=optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4),
        scheduler=optim.lr_scheduler.StepLR(
            optim.AdamW(model.parameters(), lr=1e-3),
            step_size=100,
            gamma=0.9
        ),
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
        print(f"  Memory Usage: {metrics['memory_usage']['gpu_memory_allocated']:.2f}GB")
        print(f"  Overflow Count: {metrics['overflow_count']}")
        print(f"  Training Time: {metrics['time_elapsed']:.2f}s")
    
    print("Advanced training completed!")

# =============================================================================
# EXAMPLE 3: CUSTOM TRAINING LOOP
# =============================================================================

def example_custom_training_loop():
    """Custom training loop with mixed precision."""
    print("\n=== Example 3: Custom Training Loop ===")
    
    # Create configuration
    config = create_mixed_precision_config(
        enabled=True,
        dtype=torch.float16,
        init_scale=2**16
    )
    
    # Create synthetic dataset
    torch.manual_seed(42)
    train_data = torch.randn(1500, 512)
    train_labels = torch.randint(0, 15, (1500,))
    train_dataset = TensorDataset(train_data, train_labels)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=48,
        shuffle=True,
        pin_memory=True
    )
    
    # Create model
    model = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 15)
    )
    
    # Move to device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    # Create optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    
    # Create mixed precision manager
    mp_manager = MixedPrecisionManager(config)
    
    # Custom training loop
    print("Starting custom training loop...")
    model.train()
    
    for epoch in range(2):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # Move to device
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with mp_manager.autocast_context():
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
            
            # Scale loss and backward pass
            scaled_loss = mp_manager.scale_loss(loss)
            scaled_loss.backward()
            
            # Handle overflow
            if mp_manager.handle_overflow(optimizer):
                print(f"Gradient overflow at batch {batch_idx}, skipping...")
                continue
            
            # Unscale optimizer and step
            mp_manager.unscale_optimizer(optimizer)
            success = mp_manager.step_optimizer(optimizer)
            
            if success:
                mp_manager.update_scaler()
            
            # Update scheduler
            scheduler.step()
            
            # Update metrics
            epoch_loss += loss.item()
            num_batches += 1
            
            # Log progress
            if batch_idx % 20 == 0:
                scaler_state = mp_manager.get_scaler_state()
                print(f"Epoch {epoch}, Batch {batch_idx}: "
                      f"Loss={loss.item():.4f}, "
                      f"Scale={scaler_state['scale']:.2e}")
        
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch} completed: Avg Loss={avg_loss:.4f}")
    
    print("Custom training loop completed!")

# =============================================================================
# EXAMPLE 4: PERFORMANCE BENCHMARKING
# =============================================================================

def example_performance_benchmarking():
    """Benchmark mixed precision vs full precision."""
    print("\n=== Example 4: Performance Benchmarking ===")
    
    # Create synthetic dataset
    torch.manual_seed(42)
    train_data = torch.randn(1000, 768)
    train_labels = torch.randint(0, 10, (1000,))
    train_dataset = TensorDataset(train_data, train_labels)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        pin_memory=True
    )
    
    # Create model
    model = nn.Sequential(
        nn.Linear(768, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )
    
    # Test different configurations
    configs = [
        ("FP32 (No Mixed Precision)", MixedPrecisionConfig(enabled=False)),
        ("FP16 (Mixed Precision)", MixedPrecisionConfig(enabled=True, dtype=torch.float16)),
        ("BF16 (Mixed Precision)", MixedPrecisionConfig(enabled=True, dtype=torch.bfloat16))
    ]
    
    print("Benchmarking different configurations...")
    for name, config in configs:
        print(f"\nTesting {name}:")
        
        try:
            results = benchmark_mixed_precision(
                model=model,
                train_loader=train_loader,
                config=config,
                num_steps=50
            )
            
            print(f"  Speedup: {results['speedup']:.2f}x")
            print(f"  Memory savings: {results['memory_savings']:.1f}%")
            print(f"  FP32 time: {results['fp32_time']:.2f}s")
            print(f"  MP time: {results['mp_time']:.2f}s")
            print(f"  FP32 memory: {results['fp32_memory']:.2f}GB")
            print(f"  MP memory: {results['mp_memory']:.2f}GB")
            
        except Exception as e:
            print(f"  Error: {e}")
    
    print("Benchmarking completed!")

# =============================================================================
# EXAMPLE 5: MEMORY OPTIMIZATION
# =============================================================================

def example_memory_optimization():
    """Memory optimization with mixed precision."""
    print("\n=== Example 5: Memory Optimization ===")
    
    # Create memory monitor
    memory_monitor = MixedPrecisionMemoryMonitor()
    
    # Create synthetic dataset
    torch.manual_seed(42)
    train_data = torch.randn(2000, 1024)
    train_labels = torch.randint(0, 25, (2000,))
    train_dataset = TensorDataset(train_data, train_labels)
    
    # Test different batch sizes
    batch_sizes = [16, 32, 64, 128]
    
    print("Testing memory usage with different batch sizes...")
    
    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size}")
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True
        )
        
        # Create model
        model = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 25)
        )
        
        # Memory-efficient configuration
        config = MixedPrecisionConfig(
            enabled=True,
            dtype=torch.float16,
            memory_efficient=True,
            cache_enabled=False,
            pin_memory=False
        )
        
        # Enable gradient checkpointing if available
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        
        # Create trainer
        trainer = MixedPrecisionTrainer(
            model=model,
            train_loader=train_loader,
            config=config,
            optimizer=optim.Adam(model.parameters(), lr=1e-3),
            loss_fn=nn.CrossEntropyLoss(),
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Get memory usage
        memory_usage = memory_monitor.get_memory_usage()
        
        print(f"  GPU Memory: {memory_usage['gpu_memory_allocated']:.2f}GB")
        print(f"  GPU Utilization: {memory_usage['gpu_utilization']:.2f}")
        print(f"  Memory Free: {memory_usage['gpu_memory_free']:.2f}GB")
        
        # Estimate memory savings
        fp32_memory = batch_size * 1024 * 4 / 1024**3  # Rough estimate
        savings = memory_monitor.estimate_memory_savings(fp32_memory)
        print(f"  Estimated memory savings: {savings:.1f}%")
    
    print("Memory optimization testing completed!")

# =============================================================================
# EXAMPLE 6: ERROR HANDLING AND RECOVERY
# =============================================================================

def example_error_handling():
    """Error handling and recovery with mixed precision."""
    print("\n=== Example 6: Error Handling and Recovery ===")
    
    # Create configuration with error handling
    config = MixedPrecisionConfig(
        enabled=True,
        dtype=torch.float16,
        init_scale=2**16,
        handle_overflow=True,
        log_scaling=True,
        log_frequency=10
    )
    
    # Create synthetic dataset with potential issues
    torch.manual_seed(42)
    train_data = torch.randn(500, 256)
    # Add some extreme values to test overflow handling
    train_data[0, :] = 1e6  # Very large values
    train_data[1, :] = 1e-6  # Very small values
    
    train_labels = torch.randint(0, 8, (500,))
    train_dataset = TensorDataset(train_data, train_labels)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        pin_memory=True
    )
    
    # Create model
    model = nn.Sequential(
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 8)
    )
    
    # Create trainer
    trainer = MixedPrecisionTrainer(
        model=model,
        train_loader=train_loader,
        config=config,
        optimizer=optim.Adam(model.parameters(), lr=1e-3),
        loss_fn=nn.CrossEntropyLoss(),
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Training with error monitoring
    print("Starting training with error handling...")
    
    for epoch in range(2):
        metrics = trainer.train_epoch(epoch)
        
        print(f"Epoch {epoch}:")
        print(f"  Loss: {metrics['train_loss']:.4f}")
        print(f"  Overflow Count: {metrics['overflow_count']}")
        print(f"  Scaler Scale: {metrics['scaler_scale']:.2e}")
        
        # Check for numerical issues
        model_params = list(trainer.model.parameters())
        has_nan = any(torch.isnan(param).any() for param in model_params)
        has_inf = any(torch.isinf(param).any() for param in model_params)
        
        if has_nan:
            print("  WARNING: NaN detected in model parameters!")
        if has_inf:
            print("  WARNING: Inf detected in model parameters!")
        
        if not (has_nan or has_inf):
            print("  ✓ Model parameters are numerically stable")
    
    print("Error handling training completed!")

# =============================================================================
# EXAMPLE 7: CHECKPOINTING AND RESUME
# =============================================================================

def example_checkpointing():
    """Checkpointing and resuming with mixed precision."""
    print("\n=== Example 7: Checkpointing and Resume ===")
    
    # Create configuration
    config = create_mixed_precision_config(
        enabled=True,
        dtype=torch.float16,
        init_scale=2**16
    )
    
    # Create synthetic dataset
    torch.manual_seed(42)
    train_data = torch.randn(800, 384)
    train_labels = torch.randint(0, 12, (800,))
    train_dataset = TensorDataset(train_data, train_labels)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=24,
        shuffle=True,
        pin_memory=True
    )
    
    # Create model
    model = nn.Sequential(
        nn.Linear(384, 192),
        nn.ReLU(),
        nn.Linear(192, 96),
        nn.ReLU(),
        nn.Linear(96, 12)
    )
    
    # Create trainer
    trainer = MixedPrecisionTrainer(
        model=model,
        train_loader=train_loader,
        config=config,
        optimizer=optim.Adam(model.parameters(), lr=1e-3),
        loss_fn=nn.CrossEntropyLoss(),
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Training with checkpointing
    print("Starting training with checkpointing...")
    
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    
    for epoch in range(3):
        metrics = trainer.train_epoch(epoch)
        
        print(f"Epoch {epoch}: Loss={metrics['train_loss']:.4f}")
        
        # Save checkpoint every epoch
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
        trainer.save_checkpoint(epoch, str(checkpoint_path))
        print(f"  Checkpoint saved: {checkpoint_path}")
        
        # Simulate interruption after epoch 1
        if epoch == 1:
            print("  Simulating training interruption...")
            break
    
    # Resume training from checkpoint
    print("\nResuming training from checkpoint...")
    
    # Create new trainer for resume
    new_trainer = MixedPrecisionTrainer(
        model=model,
        train_loader=train_loader,
        config=config,
        optimizer=optim.Adam(model.parameters(), lr=1e-3),
        loss_fn=nn.CrossEntropyLoss(),
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Load checkpoint
    checkpoint_path = checkpoint_dir / "checkpoint_epoch_1.pth"
    new_trainer.load_checkpoint(str(checkpoint_path))
    
    print(f"Resumed from epoch {new_trainer.epoch}")
    
    # Continue training
    for epoch in range(new_trainer.epoch + 1, 3):
        metrics = new_trainer.train_epoch(epoch)
        print(f"Epoch {epoch}: Loss={metrics['train_loss']:.4f}")
    
    print("Checkpointing example completed!")

# =============================================================================
# EXAMPLE 8: INTEGRATION WITH EXISTING TRAINING
# =============================================================================

def example_integration():
    """Integration with existing training code."""
    print("\n=== Example 8: Integration with Existing Training ===")
    
    # Simulate existing training setup
    class ExistingTrainer:
        def __init__(self, model, train_loader, optimizer, loss_fn):
            self.model = model
            self.train_loader = train_loader
            self.optimizer = optimizer
            self.loss_fn = loss_fn
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
        
        def train_epoch(self, epoch):
            self.model.train()
            total_loss = 0.0
            num_batches = 0
            
            for batch_idx, (inputs, targets) in enumerate(self.train_loader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            return total_loss / num_batches
    
    # Create synthetic dataset
    torch.manual_seed(42)
    train_data = torch.randn(600, 512)
    train_labels = torch.randint(0, 16, (600,))
    train_dataset = TensorDataset(train_data, train_labels)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        pin_memory=True
    )
    
    # Create model
    model = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 16)
    )
    
    # Create existing trainer
    existing_trainer = ExistingTrainer(
        model=model,
        train_loader=train_loader,
        optimizer=optim.Adam(model.parameters(), lr=1e-3),
        loss_fn=nn.CrossEntropyLoss()
    )
    
    # Train without mixed precision
    print("Training without mixed precision...")
    start_time = time.time()
    for epoch in range(2):
        loss = existing_trainer.train_epoch(epoch)
        print(f"Epoch {epoch}: Loss={loss:.4f}")
    fp32_time = time.time() - start_time
    
    # Now enhance with mixed precision
    print("\nEnhancing with mixed precision...")
    
    # Create mixed precision configuration
    config = create_mixed_precision_config(
        enabled=True,
        dtype=torch.float16,
        init_scale=2**16
    )
    
    # Create mixed precision manager
    mp_manager = MixedPrecisionManager(config)
    
    # Enhanced training loop
    model.train()
    start_time = time.time()
    
    for epoch in range(2):
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(existing_trainer.device)
            targets = targets.to(existing_trainer.device)
            
            existing_trainer.optimizer.zero_grad()
            
            # Add mixed precision
            with mp_manager.autocast_context():
                outputs = existing_trainer.model(inputs)
                loss = existing_trainer.loss_fn(outputs, targets)
            
            # Scale loss and backward pass
            scaled_loss = mp_manager.scale_loss(loss)
            scaled_loss.backward()
            
            # Handle overflow
            if mp_manager.handle_overflow(existing_trainer.optimizer):
                continue
            
            # Unscale and step
            mp_manager.unscale_optimizer(existing_trainer.optimizer)
            success = mp_manager.step_optimizer(existing_trainer.optimizer)
            
            if success:
                mp_manager.update_scaler()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch}: Loss={avg_loss:.4f}")
    
    mp_time = time.time() - start_time
    
    # Compare performance
    speedup = fp32_time / mp_time
    print(f"\nPerformance comparison:")
    print(f"  FP32 time: {fp32_time:.2f}s")
    print(f"  MP time: {mp_time:.2f}s")
    print(f"  Speedup: {speedup:.2f}x")
    
    print("Integration example completed!")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_all_examples():
    """Run all mixed precision examples."""
    print("Mixed Precision Training Examples for Video-OpusClip")
    print("=" * 60)
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"CUDA capability: {torch.cuda.get_device_capability(0)}")
    else:
        print("CUDA not available, using CPU")
    
    print()
    
    try:
        # Run examples
        example_basic_mixed_precision()
        example_advanced_configuration()
        example_custom_training_loop()
        example_performance_benchmarking()
        example_memory_optimization()
        example_error_handling()
        example_checkpointing()
        example_integration()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_all_examples() 