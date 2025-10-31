from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
from typing import Dict, Any, List, Optional, Union, Callable
import asyncio
import time
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
import psutil
import GPUtil
from contextlib import contextmanager
from onyx.utils.logger import setup_logger
from onyx.server.features.ads.performance_optimizer import (
from onyx.server.features.ads.optimized_config import settings
from onyx.server.features.ads.gradient_accumulation import (
from onyx.server.features.ads.mixed_precision_training import (
from typing import Any, List, Dict, Optional
"""
Multi-GPU Training System for Onyx Ads Backend

This module provides comprehensive multi-GPU training capabilities including:
- DataParallel for single-node multi-GPU training
- DistributedDataParallel for multi-node distributed training
- Automatic GPU detection and configuration
- Performance monitoring and optimization
- Memory management and load balancing
- Training synchronization and checkpointing
"""

    performance_monitor, 
    cache_result, 
    performance_context, 
    memory_context,
    optimizer
)
    GradientAccumulationConfig,
    GradientAccumulator,
    AdaptiveGradientAccumulator,
    GradientAccumulationTrainer,
    calculate_effective_batch_size,
    calculate_accumulation_steps,
    adjust_learning_rate,
    gradient_accumulation_context
)
    MixedPrecisionConfig,
    MixedPrecisionTrainer,
    AdaptiveMixedPrecisionTrainer,
    create_mixed_precision_config,
    should_use_mixed_precision,
    optimize_mixed_precision_settings,
    mixed_precision_context
)

logger = setup_logger()

@dataclass
class GPUConfig:
    """Configuration for multi-GPU training."""
    
    # GPU settings
    use_multi_gpu: bool = True
    gpu_ids: List[int] = field(default_factory=list)
    distributed_training: bool = False
    backend: str = "nccl"  # nccl for GPU, gloo for CPU
    
    # DataParallel settings
    device_ids: List[int] = field(default_factory=list)
    output_device: Optional[int] = None
    dim: int = 0
    
    # DistributedDataParallel settings
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    init_method: str = "env://"
    timeout: int = 1800  # 30 minutes
    
    # Training settings
    batch_size_per_gpu: int = 8
    gradient_accumulation_steps: int = 1
    sync_batch_norm: bool = True
    find_unused_parameters: bool = False
    broadcast_buffers: bool = True
    
    # Memory settings
    memory_fraction: float = 0.9
    pin_memory: bool = True
    num_workers: int = 4
    prefetch_factor: int = 2
    
    # Performance settings
    mixed_precision: bool = True
    gradient_as_bucket_view: bool = True
    static_graph: bool = False
    
    # Mixed precision settings
    amp_enabled: bool = True
    amp_dtype: torch.dtype = torch.float16
    amp_init_scale: float = 2**16
    amp_memory_efficient: bool = True
    amp_growth_factor: float = 2.0
    amp_backoff_factor: float = 0.5
    
    # Monitoring settings
    log_gpu_memory: bool = True
    log_gpu_utilization: bool = True
    profile_gpu: bool = False
    
    # Gradient accumulation settings
    use_gradient_accumulation: bool = False
    target_effective_batch_size: Optional[int] = None
    auto_adjust_accumulation: bool = True

class GPUMonitor:
    """Monitor GPU usage and performance."""
    
    def __init__(self, config: GPUConfig):
        
    """__init__ function."""
self.config = config
        self.gpu_stats = {}
        self._monitoring = False
        
    def get_gpu_info(self) -> Dict[str, Any]:
        """Get comprehensive GPU information."""
        try:
            gpus = GPUtil.getGPUs()
            gpu_info = {}
            
            for i, gpu in enumerate(gpus):
                gpu_info[f"gpu_{i}"] = {
                    "id": gpu.id,
                    "name": gpu.name,
                    "memory_total": gpu.memoryTotal,
                    "memory_used": gpu.memoryUsed,
                    "memory_free": gpu.memoryFree,
                    "memory_utilization": gpu.memoryUtil * 100,
                    "gpu_utilization": gpu.load * 100,
                    "temperature": gpu.temperature,
                    "power_draw": gpu.power_draw if hasattr(gpu, 'power_draw') else None
                }
            
            return gpu_info
        except Exception as e:
            logger.warning(f"Failed to get GPU info: {e}")
            return {}
    
    def get_available_gpus(self) -> List[int]:
        """Get list of available GPU IDs."""
        try:
            gpus = GPUtil.getGPUs()
            available_gpus = []
            
            for gpu in gpus:
                if gpu.memoryUtil < 0.9:  # Less than 90% memory usage
                    available_gpus.append(gpu.id)
            
            return available_gpus
        except Exception as e:
            logger.warning(f"Failed to get available GPUs: {e}")
            return list(range(torch.cuda.device_count()))
    
    def monitor_gpu_usage(self, gpu_id: int) -> Dict[str, float]:
        """Monitor specific GPU usage."""
        try:
            gpu = GPUtil.getGPUs()[gpu_id]
            return {
                "memory_used": gpu.memoryUsed,
                "memory_utilization": gpu.memoryUtil * 100,
                "gpu_utilization": gpu.load * 100,
                "temperature": gpu.temperature
            }
        except Exception as e:
            logger.warning(f"Failed to monitor GPU {gpu_id}: {e}")
            return {}
    
    def log_gpu_stats(self, prefix: str = ""):
        """Log current GPU statistics."""
        gpu_info = self.get_gpu_info()
        
        for gpu_id, stats in gpu_info.items():
            logger.info(f"{prefix} {gpu_id}: "
                       f"Memory: {stats['memory_used']}/{stats['memory_total']}MB "
                       f"({stats['memory_utilization']:.1f}%), "
                       f"GPU: {stats['gpu_utilization']:.1f}%, "
                       f"Temp: {stats['temperature']}°C")

class DataParallelTrainer:
    """DataParallel trainer for single-node multi-GPU training."""
    
    def __init__(self, config: GPUConfig):
        
    """__init__ function."""
self.config = config
        self.gpu_monitor = GPUMonitor(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        
        # Gradient accumulation
        self.gradient_accumulator = None
        self.accumulation_config = None
        
        # Mixed precision training
        self.mp_trainer = None
        self.scaler = None
        if self.config.mixed_precision and self.config.amp_enabled:
            mp_config = create_mixed_precision_config(
                enabled=True,
                dtype=self.config.amp_dtype,
                init_scale=self.config.amp_init_scale,
                memory_efficient=self.config.amp_memory_efficient
            )
            self.mp_trainer = AdaptiveMixedPrecisionTrainer(mp_config)
            self.scaler = self.mp_trainer.scaler
        
    def setup_gpus(self) -> List[int]:
        """Setup GPUs for DataParallel training."""
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            return []
        
        # Get available GPUs
        if not self.config.gpu_ids:
            self.config.gpu_ids = self.gpu_monitor.get_available_gpus()
        
        if not self.config.gpu_ids:
            logger.warning("No available GPUs found")
            return []
        
        # Set device IDs for DataParallel
        self.config.device_ids = self.config.gpu_ids
        if self.config.output_device is None:
            self.config.output_device = self.config.device_ids[0]
        
        logger.info(f"Using DataParallel with GPUs: {self.config.device_ids}")
        return self.config.device_ids
    
    def setup_gradient_accumulation(self, target_effective_batch_size: Optional[int] = None):
        """Setup gradient accumulation for large batch training."""
        if not self.config.use_gradient_accumulation:
            return
        
        # Calculate accumulation steps if target batch size is provided
        if target_effective_batch_size:
            actual_batch_size = self.config.batch_size_per_gpu * len(self.config.device_ids)
            accumulation_steps = calculate_accumulation_steps(target_effective_batch_size, actual_batch_size)
        else:
            accumulation_steps = self.config.gradient_accumulation_steps
        
        # Create accumulation config
        self.accumulation_config = GradientAccumulationConfig(
            accumulation_steps=accumulation_steps,
            target_batch_size=target_effective_batch_size,
            auto_adjust_batch_size=self.config.auto_adjust_accumulation,
            mixed_precision=self.config.mixed_precision,
            gradient_clipping=1.0,
            log_accumulation=True,
            log_memory_usage=True
        )
        
        # Create accumulator
        self.gradient_accumulator = AdaptiveGradientAccumulator(self.accumulation_config)
        
        # Update learning rate for accumulation
        if self.optimizer and self.accumulation_config.gradient_scaling:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = adjust_learning_rate(
                    param_group['lr'], 
                    accumulation_steps
                )
        
        logger.info(f"Gradient accumulation setup: "
                   f"steps={accumulation_steps}, "
                   f"effective_batch_size={calculate_effective_batch_size(self.config.batch_size_per_gpu * len(self.config.device_ids), accumulation_steps)}")
    
    @performance_monitor("setup_dataparallel")
    def setup_model(self, model: nn.Module) -> nn.Module:
        """Setup model for DataParallel training with mixed precision."""
        # Setup GPUs
        device_ids = self.setup_gpus()
        
        # Move model to GPU
        model = model.to(self.device)
        
        # Wrap with DataParallel
        if len(device_ids) > 1:
            model = nn.DataParallel(
                model,
                device_ids=device_ids,
                output_device=self.config.output_device,
                dim=self.config.dim
            )
        
        # Update mixed precision settings based on model
        if self.mp_trainer and self.config.mixed_precision:
            self.mp_trainer.update_config_adaptive(model)
        
        self.model = model
        return model
    
    def setup_dataloader(self, dataset, **kwargs) -> DataLoader:
        """Setup DataLoader for DataParallel training."""
        # Calculate batch size
        total_batch_size = self.config.batch_size_per_gpu * len(self.config.device_ids)
        
        # Create DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=total_batch_size,
            shuffle=kwargs.get('shuffle', True),
            num_workers=self.config.num_workers * len(self.config.device_ids),
            pin_memory=self.config.pin_memory,
            prefetch_factor=self.config.prefetch_factor,
            **kwargs
        )
        
        logger.info(f"DataLoader configured for {len(self.config.device_ids)} GPUs, "
                   f"batch size: {total_batch_size}")
        return dataloader
    
    @performance_monitor("dataparallel_training")
    async def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch using DataParallel with gradient accumulation and mixed precision."""
        if not self.model:
            raise ValueError("Model not set up. Call setup_model() first.")
        
        self.model.train()
        total_loss = 0
        num_batches = 0
        accumulation_stats = []
        
        # Initialize gradient accumulation if enabled
        if self.gradient_accumulator:
            self.gradient_accumulator.reset_accumulation()
        
        # Log GPU stats at start
        if self.config.log_gpu_memory:
            self.gpu_monitor.log_gpu_stats(f"Epoch {epoch} start")
        
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to GPU
            if isinstance(batch, (list, tuple)):
                batch = [b.to(self.device) for b in batch]
            else:
                batch = batch.to(self.device)
            
            # Forward pass with mixed precision
            if self.mp_trainer and self.config.mixed_precision:
                if isinstance(batch, (list, tuple)):
                    outputs = self.mp_trainer.forward_pass(self.model, *batch)
                else:
                    outputs = self.mp_trainer.forward_pass(self.model, batch)
            else:
                # Standard forward pass
                if isinstance(batch, (list, tuple)):
                    outputs = self.model(*batch)
                else:
                    outputs = self.model(batch)
            
            # Calculate loss
            if self.criterion:
                if isinstance(outputs, (list, tuple)):
                    loss = self.criterion(*outputs)
                else:
                    loss = self.criterion(outputs)
            else:
                loss = outputs if isinstance(outputs, torch.Tensor) else outputs.loss
            
            # Handle gradient accumulation with mixed precision
            if self.gradient_accumulator:
                # Use gradient accumulation with mixed precision
                acc_stats = self.gradient_accumulator.accumulate_gradients(
                    loss, self.model, self.optimizer, self.scaler
                )
                accumulation_stats.append(acc_stats)
                
                # Only update scheduler on actual optimizer steps
                if acc_stats["should_update"] and self.scheduler:
                    self.scheduler.step()
            else:
                # Standard training with mixed precision
                if self.mp_trainer and self.config.mixed_precision:
                    # Backward pass with mixed precision
                    backward_stats = self.mp_trainer.backward_pass(loss, self.optimizer, self.scaler)
                    
                    # Optimizer step with mixed precision
                    optimizer_stats = self.mp_trainer.optimizer_step(
                        self.optimizer, self.scaler, gradient_clipping=1.0
                    )
                    
                    if self.scheduler:
                        self.scheduler.step()
                else:
                    # Standard training without mixed precision
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                    if self.scheduler:
                        self.scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Log progress
            if batch_idx % 10 == 0:
                scaler_scale = 1.0
                if self.scaler and self.config.mixed_precision:
                    scaler_scale = self.scaler.get_scale()
                
                logger.info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}, "
                           f"Scaler scale: {scaler_scale:.2f}")
                
                # Log accumulation progress if enabled
                if self.gradient_accumulator:
                    acc_step = self.gradient_accumulator.accumulation_step
                    acc_steps = self.gradient_accumulator.config.accumulation_steps
                    logger.info(f"  Accumulation: {acc_step}/{acc_steps}")
                
                # Monitor GPU usage
                if self.config.log_gpu_utilization:
                    for gpu_id in self.config.device_ids:
                        gpu_stats = self.gpu_monitor.monitor_gpu_usage(gpu_id)
                        if gpu_stats:
                            logger.info(f"GPU {gpu_id}: "
                                       f"Memory: {gpu_stats['memory_utilization']:.1f}%, "
                                       f"GPU: {gpu_stats['gpu_utilization']:.1f}%")
        
        avg_loss = total_loss / num_batches
        
        # Log GPU stats at end
        if self.config.log_gpu_memory:
            self.gpu_monitor.log_gpu_stats(f"Epoch {epoch} end")
        
        # Get accumulation statistics
        accumulation_info = {}
        if self.gradient_accumulator:
            accumulation_info = self.gradient_accumulator.get_accumulation_stats()
        
        # Get mixed precision statistics
        mp_stats = {}
        if self.mp_trainer:
            mp_stats = self.mp_trainer.get_training_stats()
        
        return {
            "loss": avg_loss, 
            "num_batches": num_batches,
            "accumulation_stats": accumulation_info,
            "effective_batch_size": calculate_effective_batch_size(
                self.config.batch_size_per_gpu * len(self.config.device_ids),
                self.config.gradient_accumulation_steps
            ) if self.gradient_accumulator else self.config.batch_size_per_gpu * len(self.config.device_ids),
            "mixed_precision_enabled": self.config.mixed_precision and self.config.amp_enabled,
            "mp_stats": mp_stats
        }
    
    def cleanup(self) -> Any:
        """Cleanup DataParallel resources."""
        if self.model:
            # Remove DataParallel wrapper
            if isinstance(self.model, DataParallel):
                self.model = self.model.module
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        logger.info("DataParallel cleanup completed")

class DistributedDataParallelTrainer:
    """DistributedDataParallel trainer for multi-node distributed training."""
    
    def __init__(self, config: GPUConfig):
        
    """__init__ function."""
self.config = config
        self.gpu_monitor = GPUMonitor(config)
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.device = None
        
        # Gradient accumulation
        self.gradient_accumulator = None
        self.accumulation_config = None
        
        # Mixed precision training
        self.mp_trainer = None
        self.scaler = None
        if self.config.mixed_precision and self.config.amp_enabled:
            mp_config = create_mixed_precision_config(
                enabled=True,
                dtype=self.config.amp_dtype,
                init_scale=self.config.amp_init_scale,
                memory_efficient=self.config.amp_memory_efficient
            )
            self.mp_trainer = AdaptiveMixedPrecisionTrainer(mp_config)
            self.scaler = self.mp_trainer.scaler
        
    def setup_distributed(self, rank: int, world_size: int, backend: str = "nccl"):
        """Setup distributed training environment."""
        self.config.rank = rank
        self.config.world_size = world_size
        self.config.backend = backend
        
        # Set environment variables
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['RANK'] = str(rank)
        
        # Initialize process group
        dist.init_process_group(
            backend=backend,
            init_method=self.config.init_method,
            world_size=world_size,
            rank=rank,
            timeout=datetime.timedelta(seconds=self.config.timeout)
        )
        
        # Set device
        self.device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(self.device)
        
        logger.info(f"Distributed training initialized: rank {rank}/{world_size}, "
                   f"device: {self.device}, backend: {backend}")
    
    def setup_gradient_accumulation(self, target_effective_batch_size: Optional[int] = None):
        """Setup gradient accumulation for distributed training."""
        if not self.config.use_gradient_accumulation:
            return
        
        # Calculate accumulation steps for distributed training
        if target_effective_batch_size:
            actual_batch_size = self.config.batch_size_per_gpu * self.config.world_size
            accumulation_steps = calculate_accumulation_steps(target_effective_batch_size, actual_batch_size)
        else:
            accumulation_steps = self.config.gradient_accumulation_steps
        
        # Create accumulation config
        self.accumulation_config = GradientAccumulationConfig(
            accumulation_steps=accumulation_steps,
            target_batch_size=target_effective_batch_size,
            auto_adjust_batch_size=self.config.auto_adjust_accumulation,
            mixed_precision=self.config.mixed_precision,
            gradient_clipping=1.0,
            log_accumulation=True,
            log_memory_usage=True
        )
        
        # Create accumulator
        self.gradient_accumulator = AdaptiveGradientAccumulator(self.accumulation_config)
        
        # Update learning rate for accumulation
        if self.optimizer and self.accumulation_config.gradient_scaling:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = adjust_learning_rate(
                    param_group['lr'], 
                    accumulation_steps
                )
        
        logger.info(f"Distributed gradient accumulation setup: "
                   f"steps={accumulation_steps}, "
                   f"effective_batch_size={calculate_effective_batch_size(self.config.batch_size_per_gpu * self.config.world_size, accumulation_steps)}")
    
    @performance_monitor("setup_distributed_model")
    def setup_model(self, model: nn.Module) -> nn.Module:
        """Setup model for DistributedDataParallel training with mixed precision."""
        # Setup distributed environment
        self.setup_distributed(self.config.rank, self.config.world_size, self.config.backend)
        
        # Move model to device
        model = model.to(self.device)
        
        # Wrap with DistributedDataParallel
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[self.device.index] if self.device.type == "cuda" else None,
            output_device=self.device.index if self.device.type == "cuda" else None,
            find_unused_parameters=self.config.find_unused_parameters,
            broadcast_buffers=self.config.broadcast_buffers,
            gradient_as_bucket_view=self.config.gradient_as_bucket_view,
            static_graph=self.config.static_graph
        )
        
        # Update mixed precision settings based on model
        if self.mp_trainer and self.config.mixed_precision:
            self.mp_trainer.update_config_adaptive(model)
        
        self.model = model
        return model
    
    def setup_dataloader(self, dataset, **kwargs) -> DataLoader:
        """Setup DataLoader for distributed training."""
        # Create distributed sampler
        sampler = DistributedSampler(
            dataset,
            num_replicas=self.config.world_size,
            rank=self.config.rank,
            shuffle=kwargs.get('shuffle', True)
        )
        
        # Calculate batch size
        total_batch_size = self.config.batch_size_per_gpu * self.config.world_size
        
        # Create DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size_per_gpu,
            sampler=sampler,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            prefetch_factor=self.config.prefetch_factor,
            **kwargs
        )
        
        logger.info(f"Distributed DataLoader configured: "
                   f"rank {self.config.rank}, batch size per GPU: {self.config.batch_size_per_gpu}")
        return dataloader
    
    @performance_monitor("distributed_training")
    async def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch using DistributedDataParallel with gradient accumulation and mixed precision."""
        if not self.model:
            raise ValueError("Model not set up. Call setup_model() first.")
        
        self.model.train()
        total_loss = 0
        num_batches = 0
        accumulation_stats = []
        
        # Set epoch for sampler
        if hasattr(dataloader.sampler, 'set_epoch'):
            dataloader.sampler.set_epoch(epoch)
        
        # Initialize gradient accumulation if enabled
        if self.gradient_accumulator:
            self.gradient_accumulator.reset_accumulation()
        
        # Log GPU stats at start
        if self.config.log_gpu_memory and self.device.type == "cuda":
            gpu_stats = self.gpu_monitor.monitor_gpu_usage(self.device.index)
            if gpu_stats:
                logger.info(f"Rank {self.config.rank}, Epoch {epoch} start: "
                           f"GPU {self.device.index}: "
                           f"Memory: {gpu_stats['memory_utilization']:.1f}%, "
                           f"GPU: {gpu_stats['gpu_utilization']:.1f}%")
        
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            if isinstance(batch, (list, tuple)):
                batch = [b.to(self.device) for b in batch]
            else:
                batch = batch.to(self.device)
            
            # Forward pass with mixed precision
            if self.mp_trainer and self.config.mixed_precision:
                if isinstance(batch, (list, tuple)):
                    outputs = self.mp_trainer.forward_pass(self.model, *batch)
                else:
                    outputs = self.mp_trainer.forward_pass(self.model, batch)
            else:
                # Standard forward pass
                if isinstance(batch, (list, tuple)):
                    outputs = self.model(*batch)
                else:
                    outputs = self.model(batch)
            
            # Calculate loss
            if self.criterion:
                if isinstance(outputs, (list, tuple)):
                    loss = self.criterion(*outputs)
                else:
                    loss = self.criterion(outputs)
            else:
                loss = outputs if isinstance(outputs, torch.Tensor) else outputs.loss
            
            # Handle gradient accumulation with mixed precision
            if self.gradient_accumulator:
                # Use gradient accumulation with mixed precision
                acc_stats = self.gradient_accumulator.accumulate_gradients(
                    loss, self.model, self.optimizer, self.scaler
                )
                accumulation_stats.append(acc_stats)
                
                # Only update scheduler on actual optimizer steps
                if acc_stats["should_update"] and self.scheduler:
                    self.scheduler.step()
            else:
                # Standard training with mixed precision
                if self.mp_trainer and self.config.mixed_precision:
                    # Backward pass with mixed precision
                    backward_stats = self.mp_trainer.backward_pass(loss, self.optimizer, self.scaler)
                    
                    # Optimizer step with mixed precision
                    optimizer_stats = self.mp_trainer.optimizer_step(
                        self.optimizer, self.scaler, gradient_clipping=1.0
                    )
                    
                    if self.scheduler:
                        self.scheduler.step()
                else:
                    # Standard training without mixed precision
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                    if self.scheduler:
                        self.scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Log progress
            if batch_idx % 10 == 0:
                scaler_scale = 1.0
                if self.scaler and self.config.mixed_precision:
                    scaler_scale = self.scaler.get_scale()
                
                logger.info(f"Rank {self.config.rank}, Epoch {epoch}, "
                           f"Batch {batch_idx}, Loss: {loss.item():.4f}, "
                           f"Scaler scale: {scaler_scale:.2f}")
                
                # Log accumulation progress if enabled
                if self.gradient_accumulator:
                    acc_step = self.gradient_accumulator.accumulation_step
                    acc_steps = self.gradient_accumulator.config.accumulation_steps
                    logger.info(f"  Accumulation: {acc_step}/{acc_steps}")
                
                # Monitor GPU usage
                if self.config.log_gpu_utilization and self.device.type == "cuda":
                    gpu_stats = self.gpu_monitor.monitor_gpu_usage(self.device.index)
                    if gpu_stats:
                        logger.info(f"Rank {self.config.rank}, GPU {self.device.index}: "
                                   f"Memory: {gpu_stats['memory_utilization']:.1f}%, "
                                   f"GPU: {gpu_stats['gpu_utilization']:.1f}%")
        
        avg_loss = total_loss / num_batches
        
        # Synchronize loss across processes
        loss_tensor = torch.tensor(avg_loss, device=self.device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        synchronized_loss = loss_tensor.item() / self.config.world_size
        
        # Log GPU stats at end
        if self.config.log_gpu_memory and self.device.type == "cuda":
            gpu_stats = self.gpu_monitor.monitor_gpu_usage(self.device.index)
            if gpu_stats:
                logger.info(f"Rank {self.config.rank}, Epoch {epoch} end: "
                           f"GPU {self.device.index}: "
                           f"Memory: {gpu_stats['memory_utilization']:.1f}%, "
                           f"GPU: {gpu_stats['gpu_utilization']:.1f}%")
        
        # Get accumulation statistics
        accumulation_info = {}
        if self.gradient_accumulator:
            accumulation_info = self.gradient_accumulator.get_accumulation_stats()
        
        # Get mixed precision statistics
        mp_stats = {}
        if self.mp_trainer:
            mp_stats = self.mp_trainer.get_training_stats()
        
        return {
            "loss": synchronized_loss,
            "local_loss": avg_loss,
            "num_batches": num_batches,
            "accumulation_stats": accumulation_info,
            "effective_batch_size": calculate_effective_batch_size(
                self.config.batch_size_per_gpu * self.config.world_size,
                self.config.gradient_accumulation_steps
            ) if self.gradient_accumulator else self.config.batch_size_per_gpu * self.config.world_size,
            "mixed_precision_enabled": self.config.mixed_precision and self.config.amp_enabled,
            "mp_stats": mp_stats
        }
    
    def save_checkpoint(self, checkpoint_path: str, epoch: int, **kwargs):
        """Save distributed training checkpoint."""
        if self.config.rank == 0:  # Only save on main process
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": self.model.module.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
                "config": self.config.__dict__,
                **kwargs
            }
            
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load distributed training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.module.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if self.scheduler and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        logger.info(f"Checkpoint loaded: {checkpoint_path}")
        return checkpoint.get("epoch", 0)
    
    def cleanup(self) -> Any:
        """Cleanup distributed training resources."""
        if dist.is_initialized():
            dist.destroy_process_group()
        
        if self.model:
            # Remove DistributedDataParallel wrapper
            if isinstance(self.model, DistributedDataParallel):
                self.model = self.model.module
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info(f"Distributed training cleanup completed for rank {self.config.rank}")

class MultiGPUTrainingManager:
    """Manager for multi-GPU training with automatic configuration."""
    
    def __init__(self, config: Optional[GPUConfig] = None):
        
    """__init__ function."""
self.config = config or GPUConfig()
        self.gpu_monitor = GPUMonitor(self.config)
        self.dataparallel_trainer = None
        self.distributed_trainer = None
        self.current_trainer = None
        
        # Mixed precision manager
        self.mp_trainer = None
        if self.config.mixed_precision and self.config.amp_enabled:
            mp_config = create_mixed_precision_config(
                enabled=True,
                dtype=self.config.amp_dtype,
                init_scale=self.config.amp_init_scale,
                memory_efficient=self.config.amp_memory_efficient
            )
            self.mp_trainer = AdaptiveMixedPrecisionTrainer(mp_config)
        
    def detect_gpu_configuration(self) -> GPUConfig:
        """Automatically detect and configure GPU setup."""
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, using CPU only")
            self.config.use_multi_gpu = False
            return self.config
        
        num_gpus = torch.cuda.device_count()
        logger.info(f"Detected {num_gpus} CUDA devices")
        
        if num_gpus == 0:
            self.config.use_multi_gpu = False
            return self.config
        
        # Get available GPUs
        available_gpus = self.gpu_monitor.get_available_gpus()
        
        if len(available_gpus) >= 2:
            # Multi-GPU setup
            self.config.gpu_ids = available_gpus[:min(len(available_gpus), 4)]  # Max 4 GPUs
            self.config.device_ids = self.config.gpu_ids
            self.config.output_device = self.config.gpu_ids[0]
            
            logger.info(f"Multi-GPU configuration: {self.config.gpu_ids}")
        else:
            # Single GPU setup
            self.config.gpu_ids = available_gpus
            self.config.device_ids = self.config.gpu_ids
            self.config.output_device = self.config.gpu_ids[0] if self.config.gpu_ids else None
            
            logger.info(f"Single GPU configuration: {self.config.gpu_ids}")
        
        return self.config
    
    def setup_trainer(self, distributed: bool = False, world_size: int = 1, rank: int = 0):
        """Setup appropriate trainer based on configuration."""
        if distributed:
            # Distributed training
            self.config.distributed_training = True
            self.config.world_size = world_size
            self.config.rank = rank
            
            self.distributed_trainer = DistributedDataParallelTrainer(self.config)
            self.current_trainer = self.distributed_trainer
            
            logger.info(f"Distributed trainer setup: world_size={world_size}, rank={rank}")
        else:
            # DataParallel training
            self.config.distributed_training = False
            self.dataparallel_trainer = DataParallelTrainer(self.config)
            self.current_trainer = self.dataparallel_trainer
            
            logger.info("DataParallel trainer setup")
        
        return self.current_trainer
    
    def setup_gradient_accumulation(
        self,
        target_effective_batch_size: Optional[int] = None,
        accumulation_steps: Optional[int] = None
    ):
        """Setup gradient accumulation for the current trainer."""
        if not self.current_trainer:
            raise ValueError("No trainer set up. Call setup_trainer() first.")
        
        # Update config
        if accumulation_steps:
            self.config.gradient_accumulation_steps = accumulation_steps
        if target_effective_batch_size:
            self.config.target_effective_batch_size = target_effective_batch_size
        
        self.config.use_gradient_accumulation = True
        
        # Setup accumulation for current trainer
        if self.dataparallel_trainer:
            self.dataparallel_trainer.setup_gradient_accumulation(target_effective_batch_size)
        elif self.distributed_trainer:
            self.distributed_trainer.setup_gradient_accumulation(target_effective_batch_size)
        
        logger.info(f"Gradient accumulation setup: "
                   f"target_batch_size={target_effective_batch_size}, "
                   f"accumulation_steps={self.config.gradient_accumulation_steps}")
    
    @performance_monitor("multi_gpu_training")
    async def train_model(
        self,
        model: nn.Module,
        train_dataset,
        val_dataset=None,
        epochs: int = 10,
        target_effective_batch_size: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Train model using multi-GPU setup with gradient accumulation and mixed precision."""
        if not self.current_trainer:
            raise ValueError("Trainer not set up. Call setup_trainer() first.")
        
        # Setup gradient accumulation if requested
        if target_effective_batch_size or self.config.use_gradient_accumulation:
            self.setup_gradient_accumulation(target_effective_batch_size)
        
        # Setup model with mixed precision
        model = self.current_trainer.setup_model(model)
        
        # Update mixed precision settings
        if self.mp_trainer and self.config.mixed_precision:
            self.mp_trainer.update_config_adaptive(model)
        
        # Setup dataloaders
        train_dataloader = self.current_trainer.setup_dataloader(train_dataset, shuffle=True)
        val_dataloader = None
        if val_dataset:
            val_dataloader = self.current_trainer.setup_dataloader(val_dataset, shuffle=False)
        
        # Setup optimizer and scheduler
        self.current_trainer.optimizer = kwargs.get('optimizer')
        self.current_trainer.scheduler = kwargs.get('scheduler')
        self.current_trainer.criterion = kwargs.get('criterion')
        
        # Training loop
        training_history = []
        best_loss = float('inf')
        
        for epoch in range(epochs):
            # Train epoch
            train_metrics = await self.current_trainer.train_epoch(train_dataloader, epoch)
            
            # Validation
            val_metrics = {}
            if val_dataloader:
                val_metrics = await self.current_trainer.train_epoch(val_dataloader, epoch)
            
            # Log metrics
            epoch_metrics = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "effective_batch_size": train_metrics.get("effective_batch_size", 0),
                "mixed_precision_enabled": train_metrics.get("mixed_precision_enabled", False),
                **val_metrics
            }
            
            # Add accumulation stats if available
            if "accumulation_stats" in train_metrics:
                epoch_metrics["accumulation_stats"] = train_metrics["accumulation_stats"]
            
            # Add mixed precision stats if available
            if "mp_stats" in train_metrics:
                epoch_metrics["mp_stats"] = train_metrics["mp_stats"]
            
            training_history.append(epoch_metrics)
            
            logger.info(f"Epoch {epoch}: Train Loss: {train_metrics['loss']:.4f}, "
                       f"Val Loss: {val_metrics.get('loss', 'N/A')}, "
                       f"Effective Batch Size: {train_metrics.get('effective_batch_size', 'N/A')}, "
                       f"Mixed Precision: {train_metrics.get('mixed_precision_enabled', False)}")
            
            # Save checkpoint if best loss
            if val_metrics.get('loss', float('inf')) < best_loss:
                best_loss = val_metrics['loss']
                if hasattr(self.current_trainer, 'save_checkpoint'):
                    self.current_trainer.save_checkpoint(
                        f"checkpoints/best_model_epoch_{epoch}.pt",
                        epoch,
                        **epoch_metrics
                    )
        
        return {
            "training_history": training_history,
            "best_loss": best_loss,
            "final_model": model,
            "gradient_accumulation_used": self.config.use_gradient_accumulation,
            "mixed_precision_used": self.config.mixed_precision and self.config.amp_enabled,
            "mp_stats": self.mp_trainer.get_training_stats() if self.mp_trainer else {}
        }
    
    def get_gpu_stats(self) -> Dict[str, Any]:
        """Get comprehensive GPU statistics."""
        return {
            "gpu_info": self.gpu_monitor.get_gpu_info(),
            "available_gpus": self.gpu_monitor.get_available_gpus(),
            "config": self.config.__dict__,
            "current_trainer": type(self.current_trainer).__name__ if self.current_trainer else None
        }
    
    def cleanup(self) -> Any:
        """Cleanup all training resources."""
        if self.dataparallel_trainer:
            self.dataparallel_trainer.cleanup()
        
        if self.distributed_trainer:
            self.distributed_trainer.cleanup()
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Multi-GPU training cleanup completed")

# Utility functions for distributed training
def setup_distributed_training(rank: int, world_size: int, backend: str = "nccl"):
    """Setup distributed training environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_distributed_training():
    """Cleanup distributed training environment."""
    if dist.is_initialized():
        dist.destroy_process_group()

def run_distributed_training(rank: int, world_size: int, train_func: Callable, *args, **kwargs):
    """Run distributed training function."""
    setup_distributed_training(rank, world_size)
    
    try:
        train_func(rank, world_size, *args, **kwargs)
    finally:
        cleanup_distributed_training()

def launch_distributed_training(world_size: int, train_func: Callable, *args, **kwargs):
    """Launch distributed training with multiple processes."""
    mp.spawn(
        run_distributed_training,
        args=(world_size, train_func, *args),
        nprocs=world_size,
        join=True
    )

# Performance monitoring decorators
@contextmanager
def gpu_monitoring_context(gpu_ids: List[int]):
    """Context manager for GPU monitoring."""
    gpu_monitor = GPUMonitor(GPUConfig())
    
    # Log GPU stats at start
    gpu_monitor.log_gpu_stats("Start")
    
    try:
        yield gpu_monitor
    finally:
        # Log GPU stats at end
        gpu_monitor.log_gpu_stats("End")

@performance_monitor("gpu_memory_cleanup")
def cleanup_gpu_memory():
    """Cleanup GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.info("GPU memory cleanup completed")

# Example usage functions
async def example_dataparallel_training():
    """Example DataParallel training."""
    config = GPUConfig(
        use_multi_gpu=True,
        batch_size_per_gpu=8,
        mixed_precision=True
    )
    
    manager = MultiGPUTrainingManager(config)
    manager.detect_gpu_configuration()
    trainer = manager.setup_trainer(distributed=False)
    
    # Your model and dataset setup here
    # model = YourModel()
    # dataset = YourDataset()
    
    # await manager.train_model(model, dataset, epochs=10)

async def example_distributed_training():
    """Example DistributedDataParallel training."""
    config = GPUConfig(
        use_multi_gpu=True,
        distributed_training=True,
        batch_size_per_gpu=8,
        mixed_precision=True
    )
    
    manager = MultiGPUTrainingManager(config)
    manager.detect_gpu_configuration()
    trainer = manager.setup_trainer(distributed=True, world_size=4, rank=0)
    
    # Your model and dataset setup here
    # model = YourModel()
    # dataset = YourDataset()
    
    # await manager.train_model(model, dataset, epochs=10) 