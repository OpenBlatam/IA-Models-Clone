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
from torch.cuda.amp import autocast, GradScaler
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
from onyx.server.features.ads.gradient_accumulation import (
from typing import Any, List, Dict, Optional
"""
Mixed Precision Training System for Onyx Ads Backend

This module provides comprehensive mixed precision training capabilities including:
- Automatic mixed precision (AMP) with torch.cuda.amp
- Integration with gradient accumulation
- Memory optimization and performance monitoring
- Automatic precision selection based on model and hardware
- Training stability with gradient scaling
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
    GradientAccumulationConfig,
    GradientAccumulator,
    AdaptiveGradientAccumulator
)

logger = setup_logger()

@dataclass
class MixedPrecisionConfig:
    """Configuration for mixed precision training."""
    
    # AMP settings
    enabled: bool = True
    dtype: torch.dtype = torch.float16
    autocast_enabled: bool = True
    scaler_enabled: bool = True
    
    # Scaler settings
    init_scale: float = 2**16
    growth_factor: float = 2.0
    backoff_factor: float = 0.5
    growth_interval: int = 2000
    enabled_after_scale: Optional[float] = None
    
    # Performance settings
    memory_efficient: bool = True
    cache_enabled: bool = True
    deterministic: bool = False
    
    # Monitoring settings
    log_precision: bool = True
    log_memory_usage: bool = True
    profile_amp: bool = False
    
    # Advanced settings
    min_loss_scale: float = 1e-4
    max_loss_scale: float = 2**16
    loss_scale_window: int = 1000
    hysteresis: int = 2
    
    # Integration settings
    gradient_accumulation_compatible: bool = True
    multi_gpu_compatible: bool = True
    distributed_compatible: bool = True

class MixedPrecisionTrainer:
    """Mixed precision trainer with automatic optimization."""
    
    def __init__(self, config: MixedPrecisionConfig):
        
    """__init__ function."""
self.config = config
        self.scaler = None
        self.autocast_context = None
        self.training_stats = {
            "amp_enabled": False,
            "scaler_scale": 1.0,
            "memory_saved": 0.0,
            "training_time": 0.0,
            "overflow_count": 0,
            "underflow_count": 0
        }
        
        # Initialize scaler if enabled
        if self.config.scaler_enabled and torch.cuda.is_available():
            self.scaler = GradScaler(
                init_scale=self.config.init_scale,
                growth_factor=self.config.growth_factor,
                backoff_factor=self.config.backoff_factor,
                growth_interval=self.config.growth_interval,
                enabled_after_scale=self.config.enabled_after_scale
            )
            self.training_stats["amp_enabled"] = True
        
        # Set autocast context
        if self.config.autocast_enabled and torch.cuda.is_available():
            self.autocast_context = autocast(
                enabled=True,
                dtype=self.config.dtype,
                cache_enabled=self.config.cache_enabled,
                memory_efficient=self.config.memory_efficient,
                deterministic=self.config.deterministic
            )
    
    def should_use_mixed_precision(self, model: nn.Module) -> bool:
        """Determine if mixed precision should be used."""
        if not self.config.enabled:
            return False
        
        if not torch.cuda.is_available():
            return False
        
        # Check if model supports mixed precision
        if hasattr(model, 'supports_mixed_precision'):
            return model.supports_mixed_precision
        
        # Default to True for most models
        return True
    
    def get_memory_savings(self, model: nn.Module) -> float:
        """Calculate memory savings from mixed precision."""
        if not self.should_use_mixed_precision(model):
            return 0.0
        
        # Estimate memory savings (rough approximation)
        total_params = sum(p.numel() for p in model.parameters())
        fp32_memory = total_params * 4  # 4 bytes per parameter
        fp16_memory = total_params * 2  # 2 bytes per parameter
        
        memory_saved = (fp32_memory - fp16_memory) / 1024**3  # GB
        return memory_saved
    
    @performance_monitor("mixed_precision_forward")
    def forward_pass(self, model: nn.Module, *args, **kwargs):
        """Perform forward pass with mixed precision."""
        if self.should_use_mixed_precision(model) and self.autocast_context:
            with self.autocast_context:
                return model(*args, **kwargs)
        else:
            return model(*args, **kwargs)
    
    @performance_monitor("mixed_precision_backward")
    def backward_pass(
        self,
        loss: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        scaler: Optional[GradScaler] = None
    ) -> Dict[str, Any]:
        """Perform backward pass with mixed precision."""
        start_time = time.time()
        
        if scaler and self.config.scaler_enabled:
            # Use gradient scaler
            scaler.scale(loss).backward()
            
            # Check for overflow
            if scaler.is_enabled():
                self.training_stats["scaler_scale"] = scaler.get_scale()
                
                # Check for overflow
                if scaler.get_scale() < self.config.min_loss_scale:
                    self.training_stats["underflow_count"] += 1
                    if self.config.log_precision:
                        logger.warning(f"Gradient underflow detected. Scale: {scaler.get_scale()}")
                
                if scaler.get_scale() > self.config.max_loss_scale:
                    self.training_stats["overflow_count"] += 1
                    if self.config.log_precision:
                        logger.warning(f"Gradient overflow detected. Scale: {scaler.get_scale()}")
        else:
            # Standard backward pass
            loss.backward()
        
        backward_time = time.time() - start_time
        
        return {
            "backward_time": backward_time,
            "scaler_scale": self.training_stats["scaler_scale"],
            "overflow_detected": self.training_stats["overflow_count"] > 0,
            "underflow_detected": self.training_stats["underflow_count"] > 0
        }
    
    @performance_monitor("mixed_precision_optimizer_step")
    def optimizer_step(
        self,
        optimizer: torch.optim.Optimizer,
        scaler: Optional[GradScaler] = None,
        gradient_clipping: Optional[float] = None
    ) -> Dict[str, Any]:
        """Perform optimizer step with mixed precision."""
        start_time = time.time()
        
        if scaler and self.config.scaler_enabled:
            # Gradient clipping with scaler
            if gradient_clipping:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    optimizer.param_groups[0]['params'],
                    gradient_clipping
                )
            
            # Optimizer step with scaler
            scaler.step(optimizer)
            scaler.update()
            
            # Update training stats
            self.training_stats["scaler_scale"] = scaler.get_scale()
        else:
            # Standard optimizer step
            if gradient_clipping:
                torch.nn.utils.clip_grad_norm_(
                    optimizer.param_groups[0]['params'],
                    gradient_clipping
                )
            
            optimizer.step()
        
        optimizer_time = time.time() - start_time
        
        return {
            "optimizer_time": optimizer_time,
            "scaler_scale": self.training_stats["scaler_scale"]
        }
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get mixed precision training statistics."""
        return {
            **self.training_stats,
            "memory_savings_gb": self.training_stats["memory_saved"],
            "amp_enabled": self.training_stats["amp_enabled"],
            "scaler_enabled": self.config.scaler_enabled,
            "autocast_enabled": self.config.autocast_enabled
        }
    
    def reset_stats(self) -> Any:
        """Reset training statistics."""
        self.training_stats.update({
            "memory_saved": 0.0,
            "training_time": 0.0,
            "overflow_count": 0,
            "underflow_count": 0
        })

class AdaptiveMixedPrecisionTrainer(MixedPrecisionTrainer):
    """Adaptive mixed precision trainer with automatic configuration."""
    
    def __init__(self, config: MixedPrecisionConfig):
        
    """__init__ function."""
super().__init__(config)
        self.gpu_monitor = GPUMonitor(GPUConfig())
        self.performance_history = []
        self.memory_thresholds = []
        
    def should_use_mixed_precision_adaptive(self, model: nn.Module) -> bool:
        """Adaptive decision for mixed precision usage."""
        if not self.should_use_mixed_precision(model):
            return False
        
        # Check GPU memory availability
        gpu_info = self.gpu_monitor.get_gpu_info()
        if not gpu_info:
            return True  # Default to True if no GPU info
        
        # Check if any GPU has low memory
        for gpu_id, stats in gpu_info.items():
            memory_utilization = stats.get("memory_utilization", 0)
            if memory_utilization > 80:  # Use mixed precision if memory > 80%
                return True
        
        return True
    
    def optimize_precision_settings(self, model: nn.Module) -> Dict[str, Any]:
        """Optimize precision settings based on model and hardware."""
        gpu_info = self.gpu_monitor.get_gpu_info()
        model_params = sum(p.numel() for p in model.parameters())
        
        # Calculate optimal settings
        if model_params > 100_000_000:  # Large model (>100M params)
            recommended_dtype = torch.float16
            recommended_scale = 2**16
        elif model_params > 10_000_000:  # Medium model (10M-100M params)
            recommended_dtype = torch.float16
            recommended_scale = 2**14
        else:  # Small model (<10M params)
            recommended_dtype = torch.float32
            recommended_scale = 2**10
        
        # Adjust based on GPU memory
        if gpu_info:
            total_memory = sum(stats.get("memory_total", 0) for stats in gpu_info.values())
            if total_memory < 8 * 1024:  # Less than 8GB total
                recommended_dtype = torch.float16
                recommended_scale = 2**12
        
        return {
            "recommended_dtype": recommended_dtype,
            "recommended_scale": recommended_scale,
            "model_params": model_params,
            "gpu_memory_gb": sum(stats.get("memory_total", 0) for stats in gpu_info.values()) / 1024 if gpu_info else 0
        }
    
    def update_config_adaptive(self, model: nn.Module):
        """Update configuration adaptively."""
        optimization = self.optimize_precision_settings(model)
        
        # Update config based on recommendations
        self.config.dtype = optimization["recommended_dtype"]
        self.config.init_scale = optimization["recommended_scale"]
        
        # Reinitialize scaler with new settings
        if self.config.scaler_enabled and torch.cuda.is_available():
            self.scaler = GradScaler(
                init_scale=self.config.init_scale,
                growth_factor=self.config.growth_factor,
                backoff_factor=self.config.backoff_factor,
                growth_interval=self.config.growth_interval
            )
        
        logger.info(f"Adaptive mixed precision settings: "
                   f"dtype={self.config.dtype}, "
                   f"scale={self.config.init_scale}, "
                   f"model_params={optimization['model_params']:,}")

class MixedPrecisionGradientAccumulator:
    """Gradient accumulator with mixed precision support."""
    
    def __init__(self, config: MixedPrecisionConfig, accumulation_config: GradientAccumulationConfig):
        
    """__init__ function."""
self.mp_config = config
        self.acc_config = accumulation_config
        self.mp_trainer = AdaptiveMixedPrecisionTrainer(config)
        self.accumulator = AdaptiveGradientAccumulator(accumulation_config)
        
    def accumulate_gradients_mp(
        self,
        loss: torch.Tensor,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: Optional[GradScaler] = None
    ) -> Dict[str, Any]:
        """Accumulate gradients with mixed precision."""
        # Scale loss for accumulation
        scaled_loss = loss / self.acc_config.accumulation_steps
        
        # Backward pass with mixed precision
        backward_stats = self.mp_trainer.backward_pass(scaled_loss, optimizer, scaler)
        
        # Accumulate loss
        self.accumulator.total_loss += loss.item()
        self.accumulator.total_samples += loss.numel()
        
        # Increment accumulation step
        self.accumulator.accumulation_step += 1
        
        # Update optimizer if accumulation is complete
        should_update = self.accumulator.should_update()
        if should_update:
            # Optimizer step with mixed precision
            optimizer_stats = self.mp_trainer.optimizer_step(
                optimizer, scaler, self.acc_config.gradient_clipping
            )
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Reset accumulation
            self.accumulator.reset_accumulation()
        else:
            optimizer_stats = {"optimizer_time": 0.0, "scaler_scale": 1.0}
        
        # Combine stats
        return {
            "should_update": should_update,
            "accumulation_step": self.accumulator.accumulation_step,
            "total_loss": self.accumulator.total_loss,
            "total_samples": self.accumulator.total_samples,
            "backward_stats": backward_stats,
            "optimizer_stats": optimizer_stats,
            "mp_stats": self.mp_trainer.get_training_stats()
        }
    
    def get_combined_stats(self) -> Dict[str, Any]:
        """Get combined accumulation and mixed precision statistics."""
        acc_stats = self.accumulator.get_accumulation_stats()
        mp_stats = self.mp_trainer.get_training_stats()
        
        return {
            **acc_stats,
            **mp_stats,
            "combined_memory_savings_gb": mp_stats["memory_savings_gb"],
            "amp_enabled": mp_stats["amp_enabled"],
            "scaler_scale": mp_stats["scaler_scale"]
        }

# Utility functions
def create_mixed_precision_config(
    enabled: bool = True,
    dtype: torch.dtype = torch.float16,
    init_scale: float = 2**16,
    memory_efficient: bool = True
) -> MixedPrecisionConfig:
    """Create mixed precision configuration."""
    return MixedPrecisionConfig(
        enabled=enabled,
        dtype=dtype,
        init_scale=init_scale,
        memory_efficient=memory_efficient
    )

def should_use_mixed_precision(model: nn.Module, gpu_memory_gb: float) -> bool:
    """Determine if mixed precision should be used based on model and GPU memory."""
    if not torch.cuda.is_available():
        return False
    
    # Check model size
    model_params = sum(p.numel() for p in model.parameters())
    
    # Use mixed precision for large models or limited GPU memory
    if model_params > 50_000_000:  # >50M parameters
        return True
    
    if gpu_memory_gb < 8:  # <8GB GPU memory
        return True
    
    return False

def optimize_mixed_precision_settings(
    model: nn.Module,
    gpu_memory_gb: float,
    batch_size: int
) -> Dict[str, Any]:
    """Optimize mixed precision settings."""
    model_params = sum(p.numel() for p in model.parameters())
    
    # Base settings
    settings = {
        "enabled": should_use_mixed_precision(model, gpu_memory_gb),
        "dtype": torch.float16,
        "init_scale": 2**16,
        "memory_efficient": True
    }
    
    # Adjust based on model size
    if model_params > 100_000_000:  # Very large model
        settings["init_scale"] = 2**16
        settings["growth_interval"] = 2000
    elif model_params > 10_000_000:  # Large model
        settings["init_scale"] = 2**14
        settings["growth_interval"] = 1000
    else:  # Small model
        settings["init_scale"] = 2**10
        settings["growth_interval"] = 500
    
    # Adjust based on GPU memory
    if gpu_memory_gb < 4:
        settings["memory_efficient"] = True
        settings["cache_enabled"] = False
    elif gpu_memory_gb < 8:
        settings["memory_efficient"] = True
        settings["cache_enabled"] = True
    else:
        settings["memory_efficient"] = False
        settings["cache_enabled"] = True
    
    # Adjust based on batch size
    if batch_size > 32:
        settings["init_scale"] = min(settings["init_scale"], 2**14)
    
    return settings

@contextmanager
def mixed_precision_context(config: MixedPrecisionConfig):
    """Context manager for mixed precision training."""
    trainer = MixedPrecisionTrainer(config)
    try:
        yield trainer
    finally:
        # Cleanup if needed
        if trainer.scaler:
            trainer.scaler = None

# Integration with existing training systems
def integrate_mixed_precision_with_dataparallel(
    dataparallel_trainer,
    config: MixedPrecisionConfig
) -> MixedPrecisionTrainer:
    """Integrate mixed precision with DataParallel trainer."""
    mp_trainer = AdaptiveMixedPrecisionTrainer(config)
    
    # Override forward pass method
    original_forward = dataparallel_trainer.model.forward
    
    def mixed_precision_forward(*args, **kwargs) -> Any:
        return mp_trainer.forward_pass(dataparallel_trainer.model, *args, **kwargs)
    
    dataparallel_trainer.model.forward = mixed_precision_forward
    
    return mp_trainer

def integrate_mixed_precision_with_distributed(
    distributed_trainer,
    config: MixedPrecisionConfig
) -> MixedPrecisionTrainer:
    """Integrate mixed precision with DistributedDataParallel trainer."""
    mp_trainer = AdaptiveMixedPrecisionTrainer(config)
    
    # Override forward pass method
    original_forward = distributed_trainer.model.forward
    
    def mixed_precision_forward(*args, **kwargs) -> Any:
        return mp_trainer.forward_pass(distributed_trainer.model, *args, **kwargs)
    
    distributed_trainer.model.forward = mixed_precision_forward
    
    return mp_trainer

# Example usage
async def example_mixed_precision_training():
    """Example of mixed precision training."""
    # Configuration
    config = MixedPrecisionConfig(
        enabled=True,
        dtype=torch.float16,
        init_scale=2**16,
        memory_efficient=True,
        log_precision=True
    )
    
    # Create trainer
    trainer = AdaptiveMixedPrecisionTrainer(config)
    
    # Setup model
    model = nn.Linear(100, 10)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    # Optimize settings
    trainer.update_config_adaptive(model)
    
    # Training loop
    for epoch in range(3):
        for batch_idx in range(10):
            # Create dummy data
            inputs = torch.randn(32, 100)
            targets = torch.randint(0, 10, (32,))
            
            # Forward pass with mixed precision
            outputs = trainer.forward_pass(model, inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass with mixed precision
            backward_stats = trainer.backward_pass(loss, optimizer, trainer.scaler)
            
            # Optimizer step
            optimizer_stats = trainer.optimizer_step(optimizer, trainer.scaler)
            
            # Log progress
            if batch_idx % 5 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}: "
                      f"Loss: {loss.item():.4f}, "
                      f"Scale: {trainer.training_stats['scaler_scale']:.2f}")
    
    # Get final stats
    stats = trainer.get_training_stats()
    print(f"Training completed: {stats}")
    return stats 