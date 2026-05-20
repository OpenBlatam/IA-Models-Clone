from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Optional, Union, Callable
import asyncio
import time
import logging
from dataclasses import dataclass, field
from contextlib import contextmanager
import math
from onyx.utils.logger import setup_logger
from onyx.server.features.ads.performance_optimizer import (
from onyx.server.features.ads.multi_gpu_training import (
from onyx.server.features.ads.mixed_precision_training import (
from typing import Any, List, Dict, Optional
"""
Gradient Accumulation System for Multi-GPU Training

This module provides comprehensive gradient accumulation capabilities including:
- Automatic gradient accumulation for large effective batch sizes
- Memory-efficient training with smaller actual batch sizes
- Support for both DataParallel and DistributedDataParallel
- Dynamic batch size adjustment based on GPU memory
- Performance monitoring and optimization
- Integration with existing multi-GPU training system
"""

    performance_monitor, 
    cache_result, 
    performance_context, 
    memory_context,
    optimizer
)
    GPUConfig,
    GPUMonitor,
    gpu_monitoring_context
)
    MixedPrecisionConfig,
    MixedPrecisionTrainer,
    AdaptiveMixedPrecisionTrainer,
    MixedPrecisionGradientAccumulator,
    create_mixed_precision_config,
    should_use_mixed_precision,
    optimize_mixed_precision_settings,
    mixed_precision_context
)

logger = setup_logger()

@dataclass
class GradientAccumulationConfig:
    """Configuration for gradient accumulation."""
    
    # Accumulation settings
    accumulation_steps: int = 4
    effective_batch_size: Optional[int] = None
    target_batch_size: Optional[int] = None
    
    # Memory settings
    max_memory_usage: float = 0.9  # Maximum GPU memory usage
    memory_safety_margin: float = 0.1  # Safety margin for memory
    auto_adjust_batch_size: bool = True
    
    # Performance settings
    sync_gradients: bool = True
    gradient_scaling: bool = True
    mixed_precision: bool = True
    
    # Mixed precision settings
    amp_enabled: bool = True
    amp_dtype: torch.dtype = torch.float16
    amp_init_scale: float = 2**16
    amp_memory_efficient: bool = True
    
    # Monitoring settings
    log_accumulation: bool = True
    log_memory_usage: bool = True
    profile_accumulation: bool = False
    
    # Advanced settings
    gradient_clipping: Optional[float] = 1.0
    warmup_steps: int = 0
    accumulation_scheduler: Optional[str] = None  # "linear", "cosine", "step"

class GradientAccumulator:
    """Gradient accumulator for large batch size training."""
    
    def __init__(self, config: GradientAccumulationConfig):
        
    """__init__ function."""
self.config = config
        self.current_step = 0
        self.accumulation_step = 0
        self.total_loss = 0.0
        self.total_samples = 0
        self.gradient_norms = []
        self.memory_usage = []
        
        # Performance monitoring
        self.accumulation_times = []
        self.memory_peaks = []
        
        # Mixed precision setup
        self.mp_trainer = None
        self.scaler = None
        if self.config.mixed_precision and self.config.amp_enabled:
            mp_config = create_mixed_precision_config(
                enabled=True,
                dtype=self.config.amp_dtype,
                init_scale=self.config.amp_init_scale,
                memory_efficient=self.config.amp_memory_efficient
            )
            self.mp_trainer = MixedPrecisionTrainer(mp_config)
            self.scaler = self.mp_trainer.scaler
        
    def reset_accumulation(self) -> Any:
        """Reset accumulation state."""
        self.accumulation_step = 0
        self.total_loss = 0.0
        self.total_samples = 0
        self.gradient_norms = []
        self.memory_usage = []
    
    def should_accumulate(self) -> bool:
        """Check if gradients should be accumulated."""
        return self.accumulation_step < self.config.accumulation_steps - 1
    
    def should_update(self) -> bool:
        """Check if optimizer should be updated."""
        return self.accumulation_step == self.config.accumulation_steps - 1
    
    def get_effective_batch_size(self, actual_batch_size: int) -> int:
        """Calculate effective batch size."""
        return actual_batch_size * self.config.accumulation_steps
    
    def get_learning_rate_scale(self) -> float:
        """Get learning rate scaling factor for gradient accumulation."""
        return 1.0 / self.config.accumulation_steps
    
    @performance_monitor("gradient_accumulation")
    def accumulate_gradients(
        self,
        loss: torch.Tensor,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: Optional[torch.cuda.amp.GradScaler] = None
    ) -> Dict[str, Any]:
        """Accumulate gradients for large batch training with mixed precision."""
        start_time = time.time()
        
        # Use provided scaler or default
        if scaler is None:
            scaler = self.scaler
        
        # Scale loss for accumulation
        scaled_loss = loss / self.config.accumulation_steps
        
        # Backward pass with mixed precision if enabled
        if self.mp_trainer and self.config.mixed_precision:
            backward_stats = self.mp_trainer.backward_pass(scaled_loss, optimizer, scaler)
        else:
            # Standard backward pass
            if scaler and self.config.mixed_precision:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            backward_stats = {"backward_time": 0.0, "scaler_scale": 1.0}
        
        # Accumulate loss
        self.total_loss += loss.item()
        self.total_samples += loss.numel()
        
        # Track gradient norms
        if self.config.log_accumulation:
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            self.gradient_norms.append(total_norm)
        
        # Track memory usage
        if self.config.log_memory_usage and torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1024**3  # GB
            self.memory_usage.append(memory_used)
        
        # Increment accumulation step
        self.accumulation_step += 1
        
        # Update optimizer if accumulation is complete
        should_update = self.should_update()
        if should_update:
            # Optimizer step with mixed precision
            if self.mp_trainer and self.config.mixed_precision:
                optimizer_stats = self.mp_trainer.optimizer_step(
                    optimizer, scaler, self.config.gradient_clipping
                )
            else:
                # Standard optimizer step
                if self.config.gradient_clipping:
                    if scaler and self.config.mixed_precision:
                        scaler.unscale_(optimizer)
                    
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 
                        self.config.gradient_clipping
                    )
                
                if scaler and self.config.mixed_precision:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                optimizer_stats = {"optimizer_time": 0.0, "scaler_scale": 1.0}
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Reset accumulation
            self.reset_accumulation()
        
        # Calculate metrics
        accumulation_time = time.time() - start_time
        self.accumulation_times.append(accumulation_time)
        
        # Log accumulation progress
        if self.config.log_accumulation:
            scaler_scale = backward_stats.get("scaler_scale", 1.0)
            logger.info(f"Accumulation step {self.accumulation_step}/{self.config.accumulation_steps}, "
                       f"Loss: {loss.item():.4f}, "
                       f"Should update: {should_update}, "
                       f"Scaler scale: {scaler_scale:.2f}")
        
        return {
            "should_update": should_update,
            "accumulation_step": self.accumulation_step,
            "total_loss": self.total_loss,
            "total_samples": self.total_samples,
            "accumulation_time": accumulation_time,
            "memory_used": self.memory_usage[-1] if self.memory_usage else 0.0,
            "backward_stats": backward_stats,
            "optimizer_stats": optimizer_stats if should_update else {"optimizer_time": 0.0, "scaler_scale": 1.0},
            "mp_stats": self.mp_trainer.get_training_stats() if self.mp_trainer else {}
        }
    
    def get_accumulation_stats(self) -> Dict[str, Any]:
        """Get accumulation statistics with mixed precision info."""
        avg_gradient_norm = sum(self.gradient_norms) / len(self.gradient_norms) if self.gradient_norms else 0.0
        avg_memory_usage = sum(self.memory_usage) / len(self.memory_usage) if self.memory_usage else 0.0
        avg_accumulation_time = sum(self.accumulation_times) / len(self.accumulation_times) if self.accumulation_times else 0.0
        
        # Get mixed precision stats
        mp_stats = {}
        if self.mp_trainer:
            mp_stats = self.mp_trainer.get_training_stats()
        
        return {
            "accumulation_steps": self.config.accumulation_steps,
            "current_step": self.current_step,
            "total_loss": self.total_loss,
            "total_samples": self.total_samples,
            "avg_gradient_norm": avg_gradient_norm,
            "avg_memory_usage_gb": avg_memory_usage,
            "avg_accumulation_time": avg_accumulation_time,
            "gradient_norms": self.gradient_norms,
            "memory_usage": self.memory_usage,
            "accumulation_times": self.accumulation_times,
            "mixed_precision_enabled": self.config.mixed_precision and self.config.amp_enabled,
            "mp_stats": mp_stats
        }

class AdaptiveGradientAccumulator(GradientAccumulator):
    """Adaptive gradient accumulator with dynamic batch size adjustment and mixed precision."""
    
    def __init__(self, config: GradientAccumulationConfig):
        
    """__init__ function."""
super().__init__(config)
        self.gpu_monitor = GPUMonitor(GPUConfig())
        self.batch_size_history = []
        self.memory_thresholds = []
        
        # Enhanced mixed precision setup
        if self.config.mixed_precision and self.config.amp_enabled:
            mp_config = create_mixed_precision_config(
                enabled=True,
                dtype=self.config.amp_dtype,
                init_scale=self.config.amp_init_scale,
                memory_efficient=self.config.amp_memory_efficient
            )
            self.mp_trainer = AdaptiveMixedPrecisionTrainer(mp_config)
            self.scaler = self.mp_trainer.scaler
        
    def calculate_optimal_batch_size(self, model: nn.Module, gpu_ids: List[int]) -> int:
        """Calculate optimal batch size based on GPU memory."""
        if not self.config.auto_adjust_batch_size:
            return self.config.target_batch_size or 8
        
        # Get GPU memory info
        gpu_info = self.gpu_monitor.get_gpu_info()
        available_memory = float('inf')
        
        for gpu_id in gpu_ids:
            if f"gpu_{gpu_id}" in gpu_info:
                gpu = gpu_info[f"gpu_{gpu_id}"]
                total_memory = gpu["memory_total"] / 1024  # GB
                used_memory = gpu["memory_used"] / 1024  # GB
                free_memory = total_memory - used_memory
                available_memory = min(available_memory, free_memory)
        
        # Estimate model memory usage
        model_params = sum(p.numel() for p in model.parameters())
        model_memory_gb = model_params * 4 / 1024**3  # Assuming float32
        
        # Calculate safe batch size
        safe_memory = available_memory * (1 - self.config.memory_safety_margin)
        remaining_memory = safe_memory - model_memory_gb
        
        # Estimate memory per sample (rough approximation)
        memory_per_sample = 0.1  # GB per sample (adjust based on model)
        max_batch_size = int(remaining_memory / memory_per_sample)
        
        # Apply constraints
        min_batch_size = 1
        max_batch_size = min(max_batch_size, 32)  # Reasonable upper limit
        
        optimal_batch_size = max(min_batch_size, max_batch_size)
        
        logger.info(f"GPU memory: {available_memory:.2f}GB, "
                   f"Model memory: {model_memory_gb:.2f}GB, "
                   f"Optimal batch size: {optimal_batch_size}")
        
        return optimal_batch_size
    
    def adjust_accumulation_steps(self, target_batch_size: int, actual_batch_size: int) -> int:
        """Adjust accumulation steps to achieve target batch size."""
        if actual_batch_size <= 0:
            return self.config.accumulation_steps
        
        required_steps = math.ceil(target_batch_size / actual_batch_size)
        return max(1, required_steps)
    
    def update_config(self, model: nn.Module, gpu_ids: List[int]):
        """Update configuration based on current GPU state and model."""
        # Update accumulation config
        optimal_batch_size = self.calculate_optimal_batch_size(model, gpu_ids)
        
        if self.config.target_batch_size:
            self.config.accumulation_steps = self.adjust_accumulation_steps(
                self.config.target_batch_size, 
                optimal_batch_size
            )
        
        self.config.effective_batch_size = optimal_batch_size * self.config.accumulation_steps
        
        # Update mixed precision settings based on model and GPU state
        if self.mp_trainer and self.config.mixed_precision:
            self.mp_trainer.update_config_adaptive(model)
        
        logger.info(f"Updated config: batch_size={optimal_batch_size}, "
                   f"accumulation_steps={self.config.accumulation_steps}, "
                   f"effective_batch_size={self.config.effective_batch_size}, "
                   f"mixed_precision={self.config.mixed_precision}")

class GradientAccumulationTrainer:
    """Trainer with gradient accumulation and mixed precision support."""
    
    def __init__(self, config: GradientAccumulationConfig):
        
    """__init__ function."""
self.config = config
        self.accumulator = AdaptiveGradientAccumulator(config)
        self.scaler = None
        self.mp_trainer = None
        
        # Initialize mixed precision if enabled
        if self.config.mixed_precision and self.config.amp_enabled:
            mp_config = create_mixed_precision_config(
                enabled=True,
                dtype=self.config.amp_dtype,
                init_scale=self.config.amp_init_scale,
                memory_efficient=self.config.amp_memory_efficient
            )
            self.mp_trainer = AdaptiveMixedPrecisionTrainer(mp_config)
            self.scaler = self.mp_trainer.scaler
    
    @performance_monitor("setup_accumulation_training")
    def setup_training(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        gpu_ids: List[int]
    ):
        """Setup training with gradient accumulation and mixed precision."""
        # Update configuration based on GPU state
        self.accumulator.update_config(model, gpu_ids)
        
        # Setup optimizer with gradient accumulation
        if self.config.gradient_scaling:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= self.accumulator.get_learning_rate_scale()
        
        # Update mixed precision settings
        if self.mp_trainer:
            self.mp_trainer.update_config_adaptive(model)
        
        logger.info(f"Setup gradient accumulation with mixed precision: "
                   f"steps={self.config.accumulation_steps}, "
                   f"effective_batch_size={self.config.effective_batch_size}, "
                   f"amp_enabled={self.config.mixed_precision and self.config.amp_enabled}")
    
    @performance_monitor("accumulation_training_epoch")
    async def train_epoch(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        epoch: int
    ) -> Dict[str, Any]:
        """Train for one epoch with gradient accumulation and mixed precision."""
        model.train()
        total_loss = 0.0
        num_batches = 0
        accumulation_stats = []
        
        # Reset accumulation state
        self.accumulator.reset_accumulation()
        
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            if isinstance(batch, (list, tuple)):
                batch = [b.to(model.device) for b in batch]
            else:
                batch = batch.to(model.device)
            
            # Forward pass with mixed precision
            if self.mp_trainer and self.config.mixed_precision:
                if isinstance(batch, (list, tuple)):
                    outputs = self.mp_trainer.forward_pass(model, *batch)
                else:
                    outputs = self.mp_trainer.forward_pass(model, batch)
            else:
                # Standard forward pass
                if isinstance(batch, (list, tuple)):
                    outputs = model(*batch)
                else:
                    outputs = model(batch)
            
            # Calculate loss
            if isinstance(outputs, (list, tuple)):
                loss = criterion(*outputs)
            else:
                loss = criterion(outputs)
            
            # Accumulate gradients with mixed precision
            acc_stats = self.accumulator.accumulate_gradients(
                loss, model, optimizer, self.scaler
            )
            accumulation_stats.append(acc_stats)
            
            # Update total loss
            total_loss += loss.item()
            num_batches += 1
            
            # Log progress
            if batch_idx % 10 == 0:
                scaler_scale = acc_stats.get("backward_stats", {}).get("scaler_scale", 1.0)
                logger.info(f"Epoch {epoch}, Batch {batch_idx}, "
                           f"Loss: {loss.item():.4f}, "
                           f"Accumulation: {self.accumulator.accumulation_step}/{self.config.accumulation_steps}, "
                           f"Scaler scale: {scaler_scale:.2f}")
        
        # Calculate final metrics
        avg_loss = total_loss / num_batches
        final_stats = self.accumulator.get_accumulation_stats()
        
        return {
            "loss": avg_loss,
            "num_batches": num_batches,
            "accumulation_stats": final_stats,
            "effective_batch_size": self.config.effective_batch_size,
            "mixed_precision_enabled": self.config.mixed_precision and self.config.amp_enabled,
            "mp_stats": self.mp_trainer.get_training_stats() if self.mp_trainer else {}
        }

class GradientAccumulationAPI:
    """API for gradient accumulation configuration and monitoring."""
    
    def __init__(self) -> Any:
        self.accumulators = {}
        self.configs = {}
    
    def create_accumulator(
        self,
        config: GradientAccumulationConfig,
        training_id: str
    ) -> str:
        """Create a new gradient accumulator."""
        accumulator = AdaptiveGradientAccumulator(config)
        self.accumulators[training_id] = accumulator
        self.configs[training_id] = config
        
        return training_id
    
    def get_accumulation_stats(self, training_id: str) -> Dict[str, Any]:
        """Get accumulation statistics for a training session."""
        if training_id not in self.accumulators:
            raise ValueError(f"Training ID {training_id} not found")
        
        accumulator = self.accumulators[training_id]
        return accumulator.get_accumulation_stats()
    
    def update_config(
        self,
        training_id: str,
        config: GradientAccumulationConfig
    ):
        """Update accumulation configuration."""
        if training_id not in self.accumulators:
            raise ValueError(f"Training ID {training_id} not found")
        
        self.configs[training_id] = config
        accumulator = self.accumulators[training_id]
        accumulator.config = config
        
        logger.info(f"Updated accumulation config for {training_id}")
    
    def cleanup(self, training_id: str):
        """Cleanup accumulator for a training session."""
        if training_id in self.accumulators:
            del self.accumulators[training_id]
        if training_id in self.configs:
            del self.configs[training_id]
        
        logger.info(f"Cleaned up accumulator for {training_id}")

# Utility functions
def calculate_effective_batch_size(actual_batch_size: int, accumulation_steps: int) -> int:
    """Calculate effective batch size."""
    return actual_batch_size * accumulation_steps

def calculate_accumulation_steps(target_batch_size: int, actual_batch_size: int) -> int:
    """Calculate required accumulation steps."""
    return math.ceil(target_batch_size / actual_batch_size)

def adjust_learning_rate(base_lr: float, accumulation_steps: int) -> float:
    """Adjust learning rate for gradient accumulation."""
    return base_lr / accumulation_steps

@contextmanager
def gradient_accumulation_context(accumulator: GradientAccumulator):
    """Context manager for gradient accumulation."""
    try:
        yield accumulator
    finally:
        # Cleanup if needed
        if accumulator.accumulation_step > 0:
            logger.warning(f"Gradient accumulation incomplete: {accumulator.accumulation_step}/{accumulator.config.accumulation_steps}")

# Integration with existing training systems
def integrate_with_dataparallel(
    dataparallel_trainer,
    config: GradientAccumulationConfig
) -> GradientAccumulationTrainer:
    """Integrate gradient accumulation with DataParallel trainer."""
    accumulation_trainer = GradientAccumulationTrainer(config)
    
    # Override train_epoch method
    original_train_epoch = dataparallel_trainer.train_epoch
    
    async def enhanced_train_epoch(dataloader, epoch) -> Any:
        # Setup accumulation training with mixed precision
        accumulation_trainer.setup_training(
            dataparallel_trainer.model,
            dataparallel_trainer.optimizer,
            dataparallel_trainer.config.device_ids
        )
        
        # Use accumulation trainer with mixed precision
        return await accumulation_trainer.train_epoch(
            dataparallel_trainer.model,
            dataloader,
            dataparallel_trainer.optimizer,
            dataparallel_trainer.criterion,
            epoch
        )
    
    dataparallel_trainer.train_epoch = enhanced_train_epoch
    return accumulation_trainer

def integrate_with_distributed(
    distributed_trainer,
    config: GradientAccumulationConfig
) -> GradientAccumulationTrainer:
    """Integrate gradient accumulation with DistributedDataParallel trainer."""
    accumulation_trainer = GradientAccumulationTrainer(config)
    
    # Override train_epoch method
    original_train_epoch = distributed_trainer.train_epoch
    
    async def enhanced_train_epoch(dataloader, epoch) -> Any:
        # Setup accumulation training with mixed precision
        accumulation_trainer.setup_training(
            distributed_trainer.model,
            distributed_trainer.optimizer,
            [distributed_trainer.device.index] if distributed_trainer.device.type == "cuda" else []
        )
        
        # Use accumulation trainer with mixed precision
        return await accumulation_trainer.train_epoch(
            distributed_trainer.model,
            dataloader,
            distributed_trainer.optimizer,
            distributed_trainer.criterion,
            epoch
        )
    
    distributed_trainer.train_epoch = enhanced_train_epoch
    return accumulation_trainer

# Example usage
async def example_gradient_accumulation_training():
    """Example of gradient accumulation training."""
    # Configuration
    config = GradientAccumulationConfig(
        accumulation_steps=4,
        target_batch_size=32,
        auto_adjust_batch_size=True,
        mixed_precision=True,
        gradient_clipping=1.0,
        log_accumulation=True
    )
    
    # Create trainer
    trainer = GradientAccumulationTrainer(config)
    
    # Setup model, optimizer, etc.
    model = nn.Linear(100, 10)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    # Setup training
    trainer.setup_training(model, optimizer, [0, 1])
    
    # Create dummy dataset
    dataset = torch.randn(100, 100)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8)
    
    # Train epoch
    result = await trainer.train_epoch(model, dataloader, optimizer, criterion, 0)
    
    print(f"Training completed: {result}")
    return result 