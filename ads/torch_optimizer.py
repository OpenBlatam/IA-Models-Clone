from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset
import time
import os
import math
import gc
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
import warnings
from onyx.utils.logger import setup_logger
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
PyTorch Optimization Module

This module provides comprehensive PyTorch optimization utilities including:
- Memory optimization and management
- Performance optimization with CUDA
- Model optimization and compilation
- Data loading optimization
- Training optimization with mixed precision
- Profiling and benchmarking utilities
"""


logger = setup_logger()

@dataclass
class TorchOptimizationConfig:
    """Configuration for PyTorch optimization."""
    
    # Memory optimization
    enable_memory_optimization: bool = True
    enable_gradient_checkpointing: bool = True
    enable_mixed_precision: bool = True
    memory_fraction: float = 0.8
    
    # Performance optimization
    enable_cuda_optimization: bool = True
    enable_model_compilation: bool = True
    enable_cudnn_benchmark: bool = True
    enable_cudnn_deterministic: bool = False
    
    # Data loading optimization
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    
    # Training optimization
    gradient_accumulation_steps: int = 1
    batch_size: int = 32
    learning_rate: float = 1e-4
    
    # Profiling
    enable_profiling: bool = True
    profile_memory: bool = True
    profile_performance: bool = True

class TorchMemoryOptimizer:
    """PyTorch memory optimization utilities."""
    
    def __init__(self, config: TorchOptimizationConfig = None):
        
    """__init__ function."""
self.config = config or TorchOptimizationConfig()
        self.memory_stats = {}
    
    @staticmethod
    def clear_cache():
        """Clear GPU memory cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
    
    @staticmethod
    def get_memory_info() -> Dict[str, float]:
        """Get current memory usage information."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            max_allocated = torch.cuda.max_memory_allocated()
            max_reserved = torch.cuda.max_memory_reserved()
            
            return {
                'allocated_mb': allocated / 1024**2,
                'reserved_mb': reserved / 1024**2,
                'max_allocated_mb': max_allocated / 1024**2,
                'max_reserved_mb': max_reserved / 1024**2,
                'memory_fragmentation': torch.cuda.memory_fragmentation()
            }
        return {}
    
    @staticmethod
    def optimize_tensor(tensor: torch.Tensor) -> torch.Tensor:
        """Optimize tensor for memory efficiency."""
        # Use appropriate dtype
        if tensor.dtype == torch.float64:
            tensor = tensor.float()
        elif tensor.dtype == torch.int64:
            tensor = tensor.int()
        
        # Use contiguous memory layout
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        
        # Pin memory if on CPU and moving to GPU
        if tensor.device.type == 'cpu' and torch.cuda.is_available():
            tensor = tensor.pin_memory()
        
        return tensor
    
    @staticmethod
    def gradient_checkpointing(model: nn.Module, enable: bool = True):
        """Enable gradient checkpointing for memory efficiency."""
        if hasattr(model, 'gradient_checkpointing_enable'):
            if enable:
                model.gradient_checkpointing_enable()
                logger.info("Gradient checkpointing enabled")
            else:
                model.gradient_checkpointing_disable()
                logger.info("Gradient checkpointing disabled")
        else:
            logger.warning("Model does not support gradient checkpointing")
    
    @staticmethod
    def mixed_precision_setup():
        """Setup mixed precision training."""
        if torch.cuda.is_available():
            scaler = torch.cuda.amp.GradScaler()
            return scaler
        return None
    
    def optimize_model_memory(self, model: nn.Module) -> nn.Module:
        """Optimize model for memory efficiency."""
        if self.config.enable_gradient_checkpointing:
            self.gradient_checkpointing(model, enable=True)
        
        # Move model to GPU if available
        if torch.cuda.is_available():
            model = model.cuda()
            
            # Set memory fraction
            if self.config.memory_fraction < 1.0:
                torch.cuda.set_per_process_memory_fraction(self.config.memory_fraction)
        
        return model

class TorchPerformanceOptimizer:
    """PyTorch performance optimization utilities."""
    
    def __init__(self, config: TorchOptimizationConfig = None):
        
    """__init__ function."""
self.config = config or TorchOptimizationConfig()
    
    def optimize_cuda_settings(self) -> Any:
        """Optimize CUDA settings for performance."""
        if torch.cuda.is_available() and self.config.enable_cuda_optimization:
            # Enable cuDNN benchmark for optimal performance
            torch.backends.cudnn.benchmark = self.config.enable_cudnn_benchmark
            
            # Disable deterministic mode for better performance
            torch.backends.cudnn.deterministic = self.config.enable_cudnn_deterministic
            
            # Enable cuDNN auto-tuner
            torch.backends.cudnn.enabled = True
            
            logger.info("CUDA optimizations applied")
    
    def compile_model(self, model: nn.Module, mode: str = 'default') -> nn.Module:
        """Compile model for better performance (PyTorch 2.0+)."""
        if not self.config.enable_model_compilation:
            return model
        
        try:
            if hasattr(torch, 'compile'):
                compiled_model = torch.compile(model, mode=mode)
                logger.info(f"Model compiled with mode: {mode}")
                return compiled_model
            else:
                logger.warning("torch.compile not available (requires PyTorch 2.0+)")
                return model
        except Exception as e:
            logger.error(f"Model compilation failed: {e}")
            return model
    
    def optimize_data_loader(self, dataloader: DataLoader) -> DataLoader:
        """Optimize DataLoader for better performance."""
        dataloader.num_workers = self.config.num_workers
        dataloader.pin_memory = self.config.pin_memory and torch.cuda.is_available()
        dataloader.persistent_workers = self.config.persistent_workers and self.config.num_workers > 0
        dataloader.prefetch_factor = self.config.prefetch_factor
        
        logger.info(f"DataLoader optimized: {self.config.num_workers} workers, pin_memory={dataloader.pin_memory}")
        return dataloader

class TorchMixedPrecisionTrainer:
    """Mixed precision training utilities."""
    
    def __init__(self, model: nn.Module, optimizer: optim.Optimizer, config: TorchOptimizationConfig = None):
        
    """__init__ function."""
self.model = model
        self.optimizer = optimizer
        self.config = config or TorchOptimizationConfig()
        self.scaler = None
        
        if self.config.enable_mixed_precision and torch.cuda.is_available():
            self.scaler = torch.cuda.amp.GradScaler()
            logger.info("Mixed precision training enabled")
    
    def train_step(self, data: torch.Tensor, target: torch.Tensor, loss_fn: nn.Module) -> float:
        """Single training step with mixed precision."""
        self.optimizer.zero_grad()
        
        if self.scaler is not None:
            with torch.cuda.amp.autocast():
                output = self.model(data)
                loss = loss_fn(output, target)
            
            # Scale loss and backward pass
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            output = self.model(data)
            loss = loss_fn(output, target)
            loss.backward()
            self.optimizer.step()
        
        return loss.item()
    
    def update_learning_rate(self, new_lr: float):
        """Update learning rate with scaling."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
    
    def get_scale(self) -> float:
        """Get current gradient scaler scale."""
        if self.scaler:
            return self.scaler.get_scale()
        return 1.0

class TorchModelOptimizer:
    """PyTorch model optimization utilities."""
    
    @staticmethod
    def quantize_model(model: nn.Module, quantization_type: str = 'dynamic') -> nn.Module:
        """Quantize model for reduced memory usage and faster inference."""
        if quantization_type == 'dynamic':
            quantized_model = torch.quantization.quantize_dynamic(
                model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
            )
            logger.info("Model quantized with dynamic quantization")
            return quantized_model
        elif quantization_type == 'static':
            # Requires calibration data
            model.eval()
            model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            torch.quantization.prepare(model, inplace=True)
            logger.info("Model prepared for static quantization")
            return model
        else:
            return model
    
    @staticmethod
    def optimize_model_for_inference(model: nn.Module) -> nn.Module:
        """Optimize model for inference."""
        model.eval()
        
        # Use TorchScript for optimization
        try:
            scripted_model = torch.jit.script(model)
            logger.info("Model optimized with TorchScript")
            return scripted_model
        except Exception as e:
            logger.warning(f"TorchScript optimization failed: {e}")
            return model
    
    @staticmethod
    def fuse_model_layers(model: nn.Module) -> nn.Module:
        """Fuse model layers for better performance."""
        if hasattr(model, 'fuse_model'):
            model.fuse_model()
            logger.info("Model layers fused")
        return model

class OptimizedTorchDataset(Dataset):
    """Optimized PyTorch dataset with memory efficiency."""
    
    def __init__(self, data: torch.Tensor, targets: torch.Tensor = None, 
                 transform=None, target_transform=None):
        
    """__init__ function."""
self.data = data
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform
        
        # Optimize data storage
        if isinstance(self.data, torch.Tensor):
            self.data = TorchMemoryOptimizer.optimize_tensor(self.data)
        
        if self.targets is not None and isinstance(self.targets, torch.Tensor):
            self.targets = TorchMemoryOptimizer.optimize_tensor(self.targets)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.data[idx]
        target = self.targets[idx] if self.targets is not None else None
        
        if self.transform is not None:
            data = self.transform(data)
        
        if target is not None and self.target_transform is not None:
            target = self.target_transform(target)
        
        return data, target

class TorchOptimizedTrainer:
    """Optimized PyTorch training utilities."""
    
    def __init__(self, model: nn.Module, optimizer: optim.Optimizer, 
                 criterion: nn.Module, config: TorchOptimizationConfig = None):
        
    """__init__ function."""
self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.config = config or TorchOptimizationConfig()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = None
        
        # Setup mixed precision
        if self.config.enable_mixed_precision and torch.cuda.is_available():
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Move model to device
        self.model.to(self.device)
        
        # Enable gradient checkpointing for memory efficiency
        if self.config.enable_gradient_checkpointing:
            TorchMemoryOptimizer.gradient_checkpointing(self.model, enable=True)
        
        logger.info(f"Trainer initialized on device: {self.device}")
    
    def train_epoch(self, dataloader: DataLoader, epoch: int = 0) -> float:
        """Train for one epoch with optimizations."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Mixed precision training
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Log progress
            if batch_idx % 100 == 0:
                logger.info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        return total_loss / num_batches
    
    def validate(self, dataloader: DataLoader) -> float:
        """Validate model with optimizations."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                
                if self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        output = self.model(data)
                        loss = self.criterion(output, target)
                else:
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def save_checkpoint(self, path: str, epoch: int, loss: float):
        """Save optimized checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None
        }
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str) -> Tuple[int, float]:
        """Load optimized checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scaler and checkpoint['scaler_state_dict']:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        logger.info(f"Checkpoint loaded from {path}")
        return checkpoint['epoch'], checkpoint['loss']

class TorchGradientAccumulator:
    """Gradient accumulation for large effective batch sizes."""
    
    def __init__(self, model: nn.Module, optimizer: optim.Optimizer, 
                 accumulation_steps: int = 4, config: TorchOptimizationConfig = None):
        
    """__init__ function."""
self.model = model
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self.config = config or TorchOptimizationConfig()
        self.scaler = None
        
        if self.config.enable_mixed_precision and torch.cuda.is_available():
            self.scaler = torch.cuda.amp.GradScaler()
    
    def train_step(self, data: torch.Tensor, target: torch.Tensor, 
                  criterion: nn.Module, step: int) -> float:
        """Training step with gradient accumulation."""
        # Forward pass
        if self.scaler is not None:
            with torch.cuda.amp.autocast():
                output = self.model(data)
                loss = criterion(output, target) / self.accumulation_steps
            
            # Scale loss and backward pass
            self.scaler.scale(loss).backward()
        else:
            output = self.model(data)
            loss = criterion(output, target) / self.accumulation_steps
            loss.backward()
        
        # Update weights every accumulation_steps
        if (step + 1) % self.accumulation_steps == 0:
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad()
        
        return loss.item() * self.accumulation_steps

class TorchProfiler:
    """PyTorch-specific profiling utilities."""
    
    def __init__(self, config: TorchOptimizationConfig = None):
        
    """__init__ function."""
self.config = config or TorchOptimizationConfig()
        self.profiler = None
    
    def start_profiling(self) -> Any:
        """Start PyTorch profiling."""
        if self.config.enable_profiling and torch.cuda.is_available():
            self.profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
                with_stack=True,
                profile_memory=self.config.profile_memory,
                use_cuda=True
            )
            self.profiler.start()
    
    def stop_profiling(self) -> Optional[str]:
        """Stop PyTorch profiling and get results."""
        if self.profiler:
            self.profiler.stop()
            return self.profiler.key_averages().table(
                sort_by="cuda_time_total", row_limit=10
            )
        return None
    
    def profile_model(self, model: nn.Module, sample_input: torch.Tensor) -> Optional[str]:
        """Profile model performance."""
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(sample_input)
        
        # Profile
        self.start_profiling()
        with torch.no_grad():
            for _ in range(100):
                _ = model(sample_input)
        results = self.stop_profiling()
        
        return results
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get detailed memory statistics."""
        if torch.cuda.is_available():
            return {
                'allocated': torch.cuda.memory_allocated(),
                'reserved': torch.cuda.memory_reserved(),
                'max_allocated': torch.cuda.max_memory_allocated(),
                'max_reserved': torch.cuda.max_memory_reserved(),
                'memory_fragmentation': torch.cuda.memory_fragmentation()
            }
        return {}

class TorchBenchmarker:
    """PyTorch benchmarking utilities."""
    
    @staticmethod
    def benchmark_model(model: nn.Module, input_shape: Tuple[int, ...], 
                       num_runs: int = 100, warmup_runs: int = 10) -> Dict[str, float]:
        """Benchmark model performance."""
        model.eval()
        device = next(model.parameters()).device
        
        # Create sample input
        sample_input = torch.randn(input_shape).to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(sample_input)
        
        # Benchmark
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(sample_input)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs
        throughput = num_runs / (end_time - start_time)
        
        return {
            'avg_inference_time_ms': avg_time * 1000,
            'throughput_fps': throughput,
            'memory_usage_mb': torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        }
    
    @staticmethod
    def benchmark_training(model: nn.Module, dataloader: DataLoader, 
                          num_epochs: int = 1) -> Dict[str, float]:
        """Benchmark training performance."""
        model.train()
        device = next(model.parameters()).device
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters())
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.time()
        
        for epoch in range(num_epochs):
            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.time()
        
        return {
            'total_training_time': end_time - start_time,
            'avg_epoch_time': (end_time - start_time) / num_epochs,
            'memory_usage_mb': torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        }

# Utility functions
def setup_torch_optimization(config: TorchOptimizationConfig = None) -> TorchOptimizationConfig:
    """Setup PyTorch optimization with default configuration."""
    if config is None:
        config = TorchOptimizationConfig()
    
    # Apply CUDA optimizations
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = config.enable_cudnn_benchmark
        torch.backends.cudnn.deterministic = config.enable_cudnn_deterministic
        torch.backends.cudnn.enabled = True
    
    logger.info("PyTorch optimization setup completed")
    return config

def create_optimized_dataloader(dataset: Dataset, config: TorchOptimizationConfig = None) -> DataLoader:
    """Create an optimized DataLoader."""
    if config is None:
        config = TorchOptimizationConfig()
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory and torch.cuda.is_available(),
        persistent_workers=config.persistent_workers and config.num_workers > 0,
        prefetch_factor=config.prefetch_factor
    )
    
    return dataloader

def optimize_model_for_training(model: nn.Module, config: TorchOptimizationConfig = None) -> nn.Module:
    """Optimize model for training."""
    if config is None:
        config = TorchOptimizationConfig()
    
    # Apply memory optimizations
    memory_optimizer = TorchMemoryOptimizer(config)
    model = memory_optimizer.optimize_model_memory(model)
    
    # Apply performance optimizations
    performance_optimizer = TorchPerformanceOptimizer(config)
    model = performance_optimizer.compile_model(model)
    
    return model

# Example usage
if __name__ == "__main__":
    # Setup optimization
    config = setup_torch_optimization()
    
    # Create sample model
    model = nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 10)
    )
    
    # Optimize model
    model = optimize_model_for_training(model, config)
    
    # Create sample data
    data = torch.randn(1000, 100)
    targets = torch.randint(0, 10, (1000,))
    dataset = OptimizedTorchDataset(data, targets)
    
    # Create optimized dataloader
    dataloader = create_optimized_dataloader(dataset, config)
    
    # Create trainer
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    trainer = TorchOptimizedTrainer(model, optimizer, criterion, config)
    
    # Train for one epoch
    loss = trainer.train_epoch(dataloader)
    print(f"Training loss: {loss:.4f}") 