"""
Multi-GPU trainer implementation for the ads training system.

This module consolidates all multi-GPU training functionality into a unified,
clean architecture following the base trainer interface.
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
import torch.multiprocessing as mp
import time
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
import psutil
import GPUtil
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Union, Callable

from .base_trainer import BaseTrainer, TrainingConfig, TrainingMetrics, TrainingResult

logger = logging.getLogger(__name__)

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

@dataclass
class MultiGPUTrainingConfig:
    """Configuration for multi-GPU training sessions."""
    
    # Multi-GPU specific settings
    gpu_config: GPUConfig = field(default_factory=GPUConfig)
    use_data_parallel: bool = True
    use_distributed: bool = False
    master_addr: str = "localhost"
    master_port: str = "12355"
    
    # Performance optimization
    use_gradient_checkpointing: bool = False
    use_amp: bool = True
    use_fp16: bool = False
    use_bf16: bool = False
    
    # Monitoring
    monitor_gpu_memory: bool = True
    monitor_gpu_utilization: bool = True
    log_gpu_stats: bool = True

class GPUMonitor:
    """Monitor GPU usage and memory."""
    
    def __init__(self, gpu_ids: List[int]):
        self.gpu_ids = gpu_ids
        self.gpus = GPUtil.getGPUs()
    
    def get_gpu_memory_info(self) -> Dict[str, Any]:
        """Get GPU memory information."""
        memory_info = {}
        
        for gpu in self.gpus:
            if gpu.id in self.gpu_ids:
                memory_info[f"gpu_{gpu.id}"] = {
                    "memory_total": gpu.memoryTotal,
                    "memory_used": gpu.memoryUsed,
                    "memory_free": gpu.memoryFree,
                    "memory_utilization": gpu.memoryUtil * 100,
                    "gpu_utilization": gpu.load * 100,
                    "temperature": gpu.temperature
                }
        
        return memory_info
    
    def get_gpu_utilization(self) -> Dict[str, float]:
        """Get GPU utilization percentages."""
        utilization = {}
        
        for gpu in self.gpus:
            if gpu.id in self.gpu_ids:
                utilization[f"gpu_{gpu.id}"] = gpu.load * 100
        
        return utilization
    
    def log_gpu_stats(self) -> str:
        """Log current GPU statistics."""
        memory_info = self.get_gpu_memory_info()
        utilization = self.get_gpu_utilization()
        
        stats = []
        for gpu_id in self.gpu_ids:
            gpu_key = f"gpu_{gpu_id}"
            if gpu_key in memory_info:
                info = memory_info[gpu_key]
                stats.append(
                    f"GPU {gpu_id}: "
                    f"Memory {info['memory_used']}/{info['memory_total']}MB "
                    f"({info['memory_utilization']:.1f}%), "
                    f"Utilization {utilization.get(gpu_key, 0):.1f}%, "
                    f"Temp {info['temperature']}°C"
                )
        
        return "\n".join(stats)

class MultiGPUTrainer(BaseTrainer):
    """
    Multi-GPU trainer implementation.
    
    This trainer consolidates all multi-GPU training functionality including:
    - DataParallel for single-node multi-GPU training
    - DistributedDataParallel for multi-node distributed training
    - Automatic GPU detection and configuration
    - Performance monitoring and optimization
    - Memory management and load balancing
    - Training synchronization and checkpointing
    """
    
    def __init__(self, config: TrainingConfig,
                 gpu_config: Optional[GPUConfig] = None,
                 multi_gpu_config: Optional[MultiGPUTrainingConfig] = None):
        """Initialize the multi-GPU trainer."""
        super().__init__(config)
        
        self.gpu_config = gpu_config or GPUConfig()
        self.multi_gpu_config = multi_gpu_config or MultiGPUTrainingConfig()
        
        # Multi-GPU specific components
        self.gpu_monitor: Optional[GPUMonitor] = None
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[Any] = None
        self.scheduler: Optional[Any] = None
        self.criterion: Optional[nn.Module] = None
        
        # Data components
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        
        # Multi-GPU setup
        self.device = self._setup_device()
        self.gpu_ids = self._detect_gpus()
        self._setup_multi_gpu()
        
        logger.info(f"Multi-GPU trainer initialized on device: {self.device} with GPUs: {self.gpu_ids}")
    
    def _setup_device(self) -> torch.device:
        """Set up the training device."""
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            return torch.device("cpu")
        
        # Use first available GPU as primary device
        primary_gpu = 0
        if self.gpu_config.gpu_ids:
            primary_gpu = self.gpu_config.gpu_ids[0]
        
        return torch.device(f"cuda:{primary_gpu}")
    
    def _detect_gpus(self) -> List[int]:
        """Detect available GPUs."""
        if not torch.cuda.is_available():
            return []
        
        if self.gpu_config.gpu_ids:
            # Use specified GPU IDs
            available_gpus = []
            for gpu_id in self.gpu_config.gpu_ids:
                if gpu_id < torch.cuda.device_count():
                    available_gpus.append(gpu_id)
                else:
                    logger.warning(f"GPU {gpu_id} not available")
            
            if not available_gpus:
                logger.warning("No specified GPUs available, using all available GPUs")
                available_gpus = list(range(torch.cuda.device_count()))
            
            return available_gpus
        else:
            # Use all available GPUs
            return list(range(torch.cuda.device_count()))
    
    def _setup_multi_gpu(self):
        """Set up multi-GPU configuration."""
        if not self.gpu_ids:
            logger.warning("No GPUs available for multi-GPU training")
            return
        
        # Initialize GPU monitor
        self.gpu_monitor = GPUMonitor(self.gpu_ids)
        
        # Log GPU information
        if self.multi_gpu_config.log_gpu_stats:
            logger.info("Available GPUs:")
            logger.info(self.gpu_monitor.log_gpu_stats())
        
        # Set environment variables for distributed training
        if self.multi_gpu_config.use_distributed:
            os.environ['MASTER_ADDR'] = self.multi_gpu_config.master_addr
            os.environ['MASTER_PORT'] = self.multi_gpu_config.master_port
            os.environ['WORLD_SIZE'] = str(len(self.gpu_ids))
            os.environ['RANK'] = '0'  # Single node for now
            os.environ['LOCAL_RANK'] = '0'
    
    async def setup_training(self) -> bool:
        """Set up the training environment and resources."""
        try:
            # Create a simple model for demonstration
            # In production, this would be the actual model to train
            self.model = self._create_model()
            
            # Move model to device
            self.model = self.model.to(self.device)
            
            # Setup multi-GPU
            if len(self.gpu_ids) > 1:
                if self.multi_gpu_config.use_distributed:
                    await self._setup_distributed_training()
                elif self.multi_gpu_config.use_data_parallel:
                    await self._setup_data_parallel()
            
            # Create criterion
            self.criterion = nn.MSELoss()
            
            # Create optimizer
            self.optimizer = self._create_optimizer()
            
            # Create scheduler
            self.scheduler = self._create_scheduler()
            
            # Setup data
            await self._setup_data()
            
            logger.info("Multi-GPU training setup completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup multi-GPU training: {e}")
            return False
    
    def _create_model(self) -> nn.Module:
        """Create a simple model for demonstration."""
        return nn.Sequential(
            nn.Linear(10, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 1)
        )
    
    async def _setup_distributed_training(self):
        """Set up distributed training."""
        if not dist.is_initialized():
            dist.init_process_group(
                backend=self.gpu_config.backend,
                init_method=self.gpu_config.init_method,
                world_size=self.gpu_config.world_size,
                rank=self.gpu_config.rank,
                timeout=datetime.timedelta(seconds=self.gpu_config.timeout)
            )
        
        # Wrap model with DistributedDataParallel
        self.model = DistributedDataParallel(
            self.model,
            device_ids=[self.gpu_config.local_rank],
            output_device=self.gpu_config.local_rank,
            find_unused_parameters=self.gpu_config.find_unused_parameters,
            broadcast_buffers=self.gpu_config.broadcast_buffers
        )
        
        logger.info("Distributed training setup completed")
    
    async def _setup_data_parallel(self):
        """Set up data parallel training."""
        # Wrap model with DataParallel
        self.model = DataParallel(
            self.model,
            device_ids=self.gpu_config.device_ids or self.gpu_ids,
            output_device=self.gpu_config.output_device or self.gpu_ids[0],
            dim=self.gpu_config.dim
        )
        
        logger.info("Data parallel training setup completed")
    
    def _create_optimizer(self):
        """Create the optimizer."""
        return torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
    
    def _create_scheduler(self):
        """Create the learning rate scheduler."""
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.num_epochs
        )
    
    async def _setup_data(self):
        """Set up training and validation data."""
        # Generate synthetic data for demonstration
        X = torch.randn(1000, 10)
        y = torch.randn(1000, 1)
        
        # Split data
        train_size = int(0.8 * len(X))
        val_size = len(X) - train_size
        
        X_train, X_val = torch.split(X, [train_size, val_size])
        y_train, y_val = torch.split(y, [train_size, val_size])
        
        # Create datasets
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
        
        # Create samplers for distributed training
        train_sampler = None
        val_sampler = None
        
        if self.multi_gpu_config.use_distributed:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=len(self.gpu_ids),
                rank=self.gpu_config.rank
            )
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=len(self.gpu_ids),
                rank=self.gpu_config.rank
            )
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.gpu_config.batch_size_per_gpu * len(self.gpu_ids),
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            num_workers=self.gpu_config.num_workers,
            pin_memory=self.gpu_config.pin_memory
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.gpu_config.batch_size_per_gpu * len(self.gpu_ids),
            sampler=val_sampler,
            shuffle=False,
            num_workers=self.gpu_config.num_workers,
            pin_memory=self.gpu_config.pin_memory
        )
        
        logger.info(f"Data loaders created - Train: {len(self.train_loader)}, Val: {len(self.val_loader)}")
    
    async def train_epoch(self, epoch: int) -> TrainingMetrics:
        """Train for one epoch."""
        if not self.model or not self.optimizer or not self.criterion:
            raise RuntimeError("Training not properly initialized")
        
        # Set epoch for distributed sampler
        if hasattr(self.train_loader.sampler, 'set_epoch'):
            self.train_loader.sampler.set_epoch(epoch)
        
        self.model.train()
        total_loss = 0.0
        total_steps = 0
        
        # Log GPU stats at start of epoch
        if self.multi_gpu_config.log_gpu_stats and self.gpu_monitor:
            logger.info(f"GPU stats at start of epoch {epoch + 1}:")
            logger.info(self.gpu_monitor.log_gpu_stats())
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.gradient_clipping:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clipping
                )
            
            # Optimizer step
            self.optimizer.step()
            
            total_loss += loss.item()
            total_steps += 1
            
            # Log GPU stats periodically
            if (batch_idx + 1) % 100 == 0 and self.multi_gpu_config.monitor_gpu_memory:
                if self.gpu_monitor:
                    memory_info = self.gpu_monitor.get_gpu_memory_info()
                    logger.debug(f"GPU memory at batch {batch_idx + 1}: {memory_info}")
        
        # Update learning rate
        if self.scheduler:
            self.scheduler.step()
        
        # Calculate average loss
        avg_loss = total_loss / total_steps if total_steps > 0 else 0.0
        
        # Create metrics
        metrics = TrainingMetrics(
            epoch=epoch,
            step=total_steps,
            loss=avg_loss,
            learning_rate=self.optimizer.param_groups[0]['lr']
        )
        
        # Add GPU utilization if monitoring
        if self.multi_gpu_config.monitor_gpu_utilization and self.gpu_monitor:
            gpu_util = self.gpu_monitor.get_gpu_utilization()
            avg_util = sum(gpu_util.values()) / len(gpu_util) if gpu_util else 0.0
            metrics.gpu_utilization = avg_util
        
        return metrics
    
    async def validate(self, epoch: int) -> TrainingMetrics:
        """Validate the model."""
        if not self.model or not self.criterion:
            raise RuntimeError("Training not properly initialized")
        
        # Set epoch for distributed sampler
        if hasattr(self.val_loader.sampler, 'set_epoch'):
            self.val_loader.sampler.set_epoch(epoch)
        
        self.model.eval()
        total_loss = 0.0
        total_steps = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                total_steps += 1
        
        # Calculate average validation loss
        avg_val_loss = total_loss / total_steps if total_steps > 0 else 0.0
        
        # Create validation metrics
        metrics = TrainingMetrics(
            epoch=epoch,
            validation_loss=avg_val_loss
        )
        
        return metrics
    
    async def save_checkpoint(self, epoch: int, metrics: TrainingMetrics) -> str:
        """Save a training checkpoint."""
        # For distributed training, only save on rank 0
        if self.multi_gpu_config.use_distributed and self.gpu_config.rank != 0:
            return ""
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict() if self.model else None,
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics.to_dict(),
            'config': self.config.to_dict(),
            'gpu_config': self.gpu_config.__dict__,
            'multi_gpu_config': self.multi_gpu_config.__dict__,
            'gpu_ids': self.gpu_ids
        }
        
        checkpoint_path = f"{self.config.checkpoint_path}/multi_gpu_checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        logger.info(f"Multi-GPU checkpoint saved: {checkpoint_path}")
        return checkpoint_path
    
    async def load_checkpoint(self, checkpoint_path: str) -> bool:
        """Load a training checkpoint."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Load model state
            if self.model and checkpoint['model_state_dict']:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load optimizer state
            if self.optimizer and checkpoint['optimizer_state_dict']:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load scheduler state
            if self.scheduler and checkpoint['scheduler_state_dict']:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            logger.info(f"Multi-GPU checkpoint loaded successfully: {checkpoint_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load multi-GPU checkpoint: {e}")
            return False
    
    async def _get_final_model_path(self) -> Optional[str]:
        """Get the path to the final trained model."""
        if not self.model:
            return None
        
        # For distributed training, only save on rank 0
        if self.multi_gpu_config.use_distributed and self.gpu_config.rank != 0:
            return None
        
        model_path = f"{self.config.model_save_path}/{self.config.model_name}_multi_gpu_final.pt"
        torch.save(self.model.state_dict(), model_path)
        return model_path
    
    async def _get_final_checkpoint_path(self) -> Optional[str]:
        """Get the path to the final checkpoint."""
        if not self.training_history:
            return None
        
        final_epoch = len(self.training_history) - 1
        return f"{self.config.checkpoint_path}/multi_gpu_checkpoint_epoch_{final_epoch}.pt"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        if not self.model:
            return {"error": "No model initialized"}
        
        info = {
            "model_type": self.model.__class__.__name__,
            "device": str(self.device),
            "gpu_ids": self.gpu_ids,
            "num_gpus": len(self.gpu_ids),
            "distributed": self.multi_gpu_config.use_distributed,
            "data_parallel": self.multi_gpu_config.use_data_parallel
        }
        
        # Add parameter counts
        if hasattr(self.model, 'module'):
            # Model is wrapped in DataParallel or DistributedDataParallel
            total_params = sum(p.numel() for p in self.model.module.parameters())
            trainable_params = sum(p.numel() for p in self.model.module.parameters() if p.requires_grad)
        else:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        info.update({
            "total_parameters": total_params,
            "trainable_parameters": trainable_params
        })
        
        # Add GPU info
        if self.gpu_monitor:
            info["gpu_memory_info"] = self.gpu_monitor.get_gpu_memory_info()
            info["gpu_utilization"] = self.gpu_monitor.get_gpu_utilization()
        
        return info
    
    def cleanup(self):
        """Clean up distributed training resources."""
        if self.multi_gpu_config.use_distributed and dist.is_initialized():
            dist.destroy_process_group()
            logger.info("Distributed training cleaned up")
