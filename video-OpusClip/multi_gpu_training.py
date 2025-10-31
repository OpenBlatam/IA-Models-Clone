"""
Multi-GPU Training System for Video-OpusClip

Comprehensive implementation of DataParallel and DistributedDataParallel
for efficient multi-GPU training with automatic device detection,
load balancing, and performance optimization.
"""

import torch
import torch.nn as nn
import torch.nn.parallel as parallel
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data.distributed import DistributedSampler
import os
import sys
import time
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import structlog
from dataclasses import dataclass, field
import numpy as np
import warnings
import subprocess
import socket
from contextlib import contextmanager

# Import existing training components
try:
    from optimized_training import (
        TrainingConfig, TrainingMetrics, TrainingState,
        OptimizedTrainer, OptimizedLRScheduler, EarlyStopping
    )
    from pytorch_debug_tools import PyTorchDebugManager
    TRAINING_COMPONENTS_AVAILABLE = True
except ImportError:
    TRAINING_COMPONENTS_AVAILABLE = False
    warnings.warn("Training components not available, using basic implementations")

logger = structlog.get_logger()

# =============================================================================
# GPU DETECTION AND UTILITIES
# =============================================================================

def get_gpu_info() -> Dict[str, Any]:
    """Get comprehensive GPU information."""
    gpu_info = {
        'count': torch.cuda.device_count(),
        'current': torch.cuda.current_device(),
        'devices': [],
        'memory': {},
        'capabilities': {}
    }
    
    if gpu_info['count'] > 0:
        for i in range(gpu_info['count']):
            device_props = torch.cuda.get_device_properties(i)
            gpu_info['devices'].append({
                'id': i,
                'name': device_props.name,
                'memory_total': device_props.total_memory,
                'memory_allocated': torch.cuda.memory_allocated(i),
                'memory_cached': torch.cuda.memory_reserved(i),
                'compute_capability': f"{device_props.major}.{device_props.minor}",
                'multi_processor_count': device_props.multi_processor_count
            })
            
            gpu_info['memory'][i] = {
                'total': device_props.total_memory,
                'allocated': torch.cuda.memory_allocated(i),
                'cached': torch.cuda.memory_reserved(i),
                'free': device_props.total_memory - torch.cuda.memory_reserved(i)
            }
            
            gpu_info['capabilities'][i] = {
                'compute_capability': f"{device_props.major}.{device_props.minor}",
                'multi_processor_count': device_props.multi_processor_count,
                'max_threads_per_block': device_props.max_threads_per_block,
                'max_shared_memory_per_block': device_props.max_shared_memory_per_block
            }
    
    return gpu_info

def select_optimal_gpus(num_gpus: Optional[int] = None, 
                       min_memory_gb: float = 4.0) -> List[int]:
    """Select optimal GPUs for training based on memory and capabilities."""
    gpu_info = get_gpu_info()
    
    if gpu_info['count'] == 0:
        logger.warning("No CUDA devices available")
        return []
    
    # Filter GPUs by memory requirements
    available_gpus = []
    min_memory_bytes = min_memory_gb * 1024**3
    
    for device_info in gpu_info['devices']:
        if device_info['memory_total'] >= min_memory_bytes:
            available_gpus.append(device_info['id'])
    
    if not available_gpus:
        logger.error(f"No GPUs with at least {min_memory_gb}GB memory available")
        return []
    
    # Sort by available memory (descending)
    available_gpus.sort(
        key=lambda gpu_id: gpu_info['memory'][gpu_id]['free'],
        reverse=True
    )
    
    # Limit to requested number
    if num_gpus is not None:
        available_gpus = available_gpus[:num_gpus]
    
    logger.info(f"Selected GPUs: {available_gpus}")
    return available_gpus

def setup_cuda_environment():
    """Setup optimal CUDA environment for multi-GPU training."""
    # Set CUDA device order
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    
    # Enable memory growth to prevent OOM
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    
    # Optimize for performance
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    # Set memory fraction if needed
    if 'CUDA_MEMORY_FRACTION' not in os.environ:
        os.environ['CUDA_MEMORY_FRACTION'] = '0.9'
    
    logger.info("CUDA environment configured for multi-GPU training")

# =============================================================================
# MULTI-GPU CONFIGURATION
# =============================================================================

@dataclass
class MultiGPUConfig:
    """Configuration for multi-GPU training."""
    # GPU selection
    num_gpus: Optional[int] = None  # None = use all available
    gpu_ids: Optional[List[int]] = None  # Specific GPU IDs
    min_memory_gb: float = 4.0
    
    # Training strategy
    strategy: str = 'auto'  # 'auto', 'dataparallel', 'distributed'
    backend: str = 'nccl'  # 'nccl', 'gloo', 'mpi'
    
    # Distributed training
    world_size: Optional[int] = None
    rank: Optional[int] = None
    local_rank: Optional[int] = None
    master_addr: str = 'localhost'
    master_port: str = '12355'
    
    # Data loading
    pin_memory: bool = True
    num_workers: int = 4
    persistent_workers: bool = True
    
    # Performance
    find_unused_parameters: bool = False
    broadcast_buffers: bool = True
    bucket_cap_mb: int = 25
    
    # Synchronization
    gradient_as_bucket_view: bool = False
    static_graph: bool = False
    
    # Memory optimization
    memory_efficient_find_unused_parameters: bool = False
    
    def __post_init__(self):
        """Validate and auto-configure settings."""
        gpu_info = get_gpu_info()
        
        if self.num_gpus is None:
            self.num_gpus = gpu_info['count']
        
        if self.gpu_ids is None:
            self.gpu_ids = select_optimal_gpus(self.num_gpus, self.min_memory_gb)
        
        if self.strategy == 'auto':
            if self.num_gpus == 1:
                self.strategy = 'single'
            elif self.num_gpus <= 4:
                self.strategy = 'dataparallel'
            else:
                self.strategy = 'distributed'
        
        # Adjust num_workers based on GPU count
        self.num_workers = min(self.num_workers * self.num_gpus, 16)

# =============================================================================
# DATA PARALLEL IMPLEMENTATION
# =============================================================================

class OptimizedDataParallel(nn.DataParallel):
    """Enhanced DataParallel with better error handling and monitoring."""
    
    def __init__(self, module: nn.Module, device_ids: Optional[List[int]] = None,
                 output_device: Optional[int] = None, dim: int = 0,
                 **kwargs):
        super().__init__(module, device_ids, output_device, dim)
        self.device_ids = device_ids or list(range(torch.cuda.device_count()))
        self.output_device = output_device or self.device_ids[0]
        
        logger.info(f"DataParallel initialized with devices: {self.device_ids}")
    
    def forward(self, *inputs, **kwargs):
        """Enhanced forward pass with error handling."""
        try:
            if not self.device_ids[1:]:  # Single GPU
                return self.module(*inputs, **kwargs)
            
            # Multi-GPU forward pass
            inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
            
            if len(self.device_ids) == 1:
                return self.module(*inputs[0], **kwargs[0])
            
            replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
            outputs = self.parallel_apply(replicas, inputs, kwargs)
            return self.gather(outputs, self.output_device)
            
        except Exception as e:
            logger.error(f"DataParallel forward error: {e}")
            raise
    
    def get_device_memory_usage(self) -> Dict[int, Dict[str, float]]:
        """Get memory usage for each device."""
        memory_info = {}
        for device_id in self.device_ids:
            memory_info[device_id] = {
                'allocated': torch.cuda.memory_allocated(device_id) / 1024**3,
                'cached': torch.cuda.memory_reserved(device_id) / 1024**3,
                'total': torch.cuda.get_device_properties(device_id).total_memory / 1024**3
            }
        return memory_info

class DataParallelTrainer:
    """Trainer optimized for DataParallel training."""
    
    def __init__(self, model: nn.Module, config: MultiGPUConfig,
                 train_loader: DataLoader, val_loader: Optional[DataLoader] = None,
                 **kwargs):
        self.config = config
        self.device_ids = config.gpu_ids
        
        # Move model to first GPU
        self.model = model.to(f'cuda:{self.device_ids[0]}')
        
        # Wrap with DataParallel
        self.model = OptimizedDataParallel(
            self.model,
            device_ids=self.device_ids,
            output_device=self.device_ids[0]
        )
        
        # Setup data loaders
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Setup optimizer and scheduler
        self.optimizer = kwargs.get('optimizer', torch.optim.Adam(self.model.parameters()))
        self.scheduler = kwargs.get('scheduler')
        self.loss_fn = kwargs.get('loss_fn', nn.CrossEntropyLoss())
        
        # Setup scaler for mixed precision
        self.scaler = torch.cuda.amp.GradScaler()
        
        logger.info(f"DataParallel trainer initialized with {len(self.device_ids)} GPUs")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch using DataParallel."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            # Move data to first GPU (DataParallel handles distribution)
            data = data.to(f'cuda:{self.device_ids[0]}')
            target = target.to(f'cuda:{self.device_ids[0]}')
            
            self.optimizer.zero_grad()
            
            # Mixed precision training
            with torch.cuda.amp.autocast():
                output = self.model(data)
                loss = self.loss_fn(output, target)
            
            # Backward pass
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 100 == 0:
                logger.info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches
        return {'train_loss': avg_loss}
    
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate using DataParallel."""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data = data.to(f'cuda:{self.device_ids[0]}')
                target = target.to(f'cuda:{self.device_ids[0]}')
                
                with torch.cuda.amp.autocast():
                    output = self.model(data)
                    loss = self.loss_fn(output, target)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return {'val_loss': avg_loss}
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics for all devices."""
        return {
            'device_memory': self.model.get_device_memory_usage(),
            'total_allocated': sum(
                torch.cuda.memory_allocated(device_id) 
                for device_id in self.device_ids
            ) / 1024**3,
            'total_cached': sum(
                torch.cuda.memory_reserved(device_id) 
                for device_id in self.device_ids
            ) / 1024**3
        }

# =============================================================================
# DISTRIBUTED DATA PARALLEL IMPLEMENTATION
# =============================================================================

def setup_distributed_training(rank: int, world_size: int, 
                             master_addr: str, master_port: str):
    """Setup distributed training environment."""
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    
    # Initialize process group
    dist.init_process_group(
        backend='nccl',
        init_method=f'tcp://{master_addr}:{master_port}',
        world_size=world_size,
        rank=rank
    )
    
    # Set device
    torch.cuda.set_device(rank)
    
    logger.info(f"Distributed training initialized: rank={rank}, world_size={world_size}")

def cleanup_distributed_training():
    """Cleanup distributed training environment."""
    if dist.is_initialized():
        dist.destroy_process_group()

class DistributedDataParallelTrainer:
    """Trainer optimized for DistributedDataParallel training."""
    
    def __init__(self, model: nn.Module, config: MultiGPUConfig,
                 train_dataset, val_dataset=None, **kwargs):
        self.config = config
        self.rank = config.rank
        self.world_size = config.world_size
        
        # Setup distributed environment
        setup_distributed_training(
            self.rank, self.world_size,
            self.config.master_addr, self.config.master_port
        )
        
        # Move model to device
        self.model = model.to(self.rank)
        
        # Wrap with DistributedDataParallel
        self.model = DDP(
            self.model,
            device_ids=[self.rank],
            output_device=self.rank,
            find_unused_parameters=self.config.find_unused_parameters,
            broadcast_buffers=self.config.broadcast_buffers,
            bucket_cap_mb=self.config.bucket_cap_mb,
            gradient_as_bucket_view=self.config.gradient_as_bucket_view,
            static_graph=self.config.static_graph
        )
        
        # Setup distributed samplers
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True
        )
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=kwargs.get('batch_size', 32),
            sampler=train_sampler,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers
        )
        
        if val_dataset is not None:
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False
            )
            
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=kwargs.get('batch_size', 32),
                sampler=val_sampler,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory,
                persistent_workers=self.config.persistent_workers
            )
        else:
            self.val_loader = None
        
        # Setup optimizer and scheduler
        self.optimizer = kwargs.get('optimizer', torch.optim.Adam(self.model.parameters()))
        self.scheduler = kwargs.get('scheduler')
        self.loss_fn = kwargs.get('loss_fn', nn.CrossEntropyLoss())
        
        # Setup scaler for mixed precision
        self.scaler = torch.cuda.amp.GradScaler()
        
        logger.info(f"DDP trainer initialized: rank={self.rank}, world_size={self.world_size}")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch using DistributedDataParallel."""
        self.train_loader.sampler.set_epoch(epoch)  # Ensure different shuffling
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data = data.to(self.rank)
            target = target.to(self.rank)
            
            self.optimizer.zero_grad()
            
            # Mixed precision training
            with torch.cuda.amp.autocast():
                output = self.model(data)
                loss = self.loss_fn(output, target)
            
            # Backward pass
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 100 == 0 and self.rank == 0:
                logger.info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        # Synchronize loss across processes
        avg_loss = total_loss / num_batches
        dist.all_reduce(torch.tensor(avg_loss, device=self.rank))
        avg_loss = avg_loss / self.world_size
        
        return {'train_loss': avg_loss}
    
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate using DistributedDataParallel."""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data = data.to(self.rank)
                target = target.to(self.rank)
                
                with torch.cuda.amp.autocast():
                    output = self.model(data)
                    loss = self.loss_fn(output, target)
                
                total_loss += loss.item()
                num_batches += 1
        
        # Synchronize loss across processes
        avg_loss = total_loss / num_batches
        dist.all_reduce(torch.tensor(avg_loss, device=self.rank))
        avg_loss = avg_loss / self.world_size
        
        return {'val_loss': avg_loss}
    
    def save_checkpoint(self, epoch: int, path: str):
        """Save checkpoint (only on rank 0)."""
        if self.rank == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.module.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'scaler_state_dict': self.scaler.state_dict(),
                'config': self.config
            }
            torch.save(checkpoint, path)
            logger.info(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, path: str):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=f'cuda:{self.rank}')
        self.model.module.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        logger.info(f"Checkpoint loaded: {path}")
        return checkpoint['epoch']

# =============================================================================
# MULTI-GPU TRAINING MANAGER
# =============================================================================

class MultiGPUTrainingManager:
    """High-level manager for multi-GPU training."""
    
    def __init__(self, config: MultiGPUConfig):
        self.config = config
        self.gpu_info = get_gpu_info()
        self.trainer = None
        
        # Setup CUDA environment
        setup_cuda_environment()
        
        logger.info(f"Multi-GPU training manager initialized with {self.config.num_gpus} GPUs")
    
    def create_trainer(self, model: nn.Module, train_dataset, 
                      val_dataset=None, **kwargs) -> Union[DataParallelTrainer, DistributedDataParallelTrainer]:
        """Create appropriate trainer based on configuration."""
        if self.config.strategy == 'dataparallel':
            # Create DataLoader for DataParallel
            train_loader = DataLoader(
                train_dataset,
                batch_size=kwargs.get('batch_size', 32),
                shuffle=True,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory,
                persistent_workers=self.config.persistent_workers
            )
            
            val_loader = None
            if val_dataset is not None:
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=kwargs.get('batch_size', 32),
                    shuffle=False,
                    num_workers=self.config.num_workers,
                    pin_memory=self.config.pin_memory,
                    persistent_workers=self.config.persistent_workers
                )
            
            self.trainer = DataParallelTrainer(
                model, self.config, train_loader, val_loader, **kwargs
            )
            
        elif self.config.strategy == 'distributed':
            self.trainer = DistributedDataParallelTrainer(
                model, self.config, train_dataset, val_dataset, **kwargs
            )
        
        else:
            raise ValueError(f"Unsupported training strategy: {self.config.strategy}")
        
        return self.trainer
    
    def train(self, epochs: int, save_path: str = "checkpoints"):
        """Execute training."""
        if self.trainer is None:
            raise ValueError("Trainer not initialized. Call create_trainer first.")
        
        save_path = Path(save_path)
        save_path.mkdir(exist_ok=True)
        
        best_loss = float('inf')
        
        for epoch in range(epochs):
            # Training
            train_metrics = self.trainer.train_epoch(epoch)
            
            # Validation
            val_metrics = self.trainer.validate(epoch)
            
            # Logging
            if hasattr(self.trainer, 'rank') and self.trainer.rank == 0:
                logger.info(f"Epoch {epoch}: Train Loss: {train_metrics['train_loss']:.4f}, "
                          f"Val Loss: {val_metrics.get('val_loss', 'N/A')}")
            
            # Save best model
            if val_metrics and val_metrics['val_loss'] < best_loss:
                best_loss = val_metrics['val_loss']
                if hasattr(self.trainer, 'save_checkpoint'):
                    self.trainer.save_checkpoint(epoch, save_path / "best_model.pth")
        
        # Cleanup
        if hasattr(self.trainer, 'cleanup'):
            self.trainer.cleanup()
        else:
            cleanup_distributed_training()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = {
            'gpu_info': self.gpu_info,
            'config': self.config,
            'memory_usage': {}
        }
        
        if self.trainer:
            if hasattr(self.trainer, 'get_memory_stats'):
                stats['memory_usage'] = self.trainer.get_memory_stats()
        
        return stats

# =============================================================================
# LAUNCHER FUNCTIONS
# =============================================================================

def launch_distributed_training(rank: int, world_size: int, 
                              model_fn: Callable, dataset_fn: Callable,
                              config: MultiGPUConfig, **kwargs):
    """Launch function for distributed training."""
    try:
        # Setup distributed environment
        setup_distributed_training(rank, world_size, config.master_addr, config.master_port)
        
        # Create model and datasets
        model = model_fn()
        train_dataset, val_dataset = dataset_fn()
        
        # Create trainer
        trainer = DistributedDataParallelTrainer(
            model, config, train_dataset, val_dataset, **kwargs
        )
        
        # Training loop
        for epoch in range(kwargs.get('epochs', 100)):
            train_metrics = trainer.train_epoch(epoch)
            val_metrics = trainer.validate(epoch)
            
            if rank == 0:
                logger.info(f"Epoch {epoch}: {train_metrics}, {val_metrics}")
        
        # Cleanup
        cleanup_distributed_training()
        
    except Exception as e:
        logger.error(f"Error in distributed training (rank {rank}): {e}")
        cleanup_distributed_training()
        raise

def launch_multi_gpu_training(model_fn: Callable, dataset_fn: Callable,
                            config: MultiGPUConfig, **kwargs):
    """Launch multi-GPU training with automatic strategy selection."""
    if config.strategy == 'distributed':
        # Launch distributed training
        mp.spawn(
            launch_distributed_training,
            args=(config.world_size, model_fn, dataset_fn, config, kwargs),
            nprocs=config.world_size,
            join=True
        )
    else:
        # Launch DataParallel training
        manager = MultiGPUTrainingManager(config)
        model = model_fn()
        train_dataset, val_dataset = dataset_fn()
        
        trainer = manager.create_trainer(model, train_dataset, val_dataset, **kwargs)
        manager.train(kwargs.get('epochs', 100))

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_optimal_batch_size(model: nn.Module, num_gpus: int, 
                          memory_per_sample: float = 0.1) -> int:
    """Calculate optimal batch size for multi-GPU training."""
    gpu_info = get_gpu_info()
    
    if num_gpus == 0:
        return 32
    
    # Get available memory per GPU
    min_memory = min(
        gpu_info['memory'][gpu_id]['free'] 
        for gpu_id in range(num_gpus)
    )
    
    # Reserve some memory for model and gradients
    available_memory = min_memory * 0.7  # 70% of free memory
    
    # Calculate batch size
    batch_size = int(available_memory / memory_per_sample)
    
    # Ensure batch size is reasonable
    batch_size = max(1, min(batch_size, 512))
    
    logger.info(f"Optimal batch size: {batch_size} (per GPU)")
    return batch_size

def benchmark_multi_gpu_performance(model: nn.Module, dataset, 
                                  config: MultiGPUConfig) -> Dict[str, float]:
    """Benchmark multi-GPU training performance."""
    manager = MultiGPUTrainingManager(config)
    
    # Create trainer
    trainer = manager.create_trainer(model, dataset)
    
    # Benchmark training speed
    start_time = time.time()
    metrics = trainer.train_epoch(0)
    end_time = time.time()
    
    training_time = end_time - start_time
    samples_per_second = len(dataset) / training_time
    
    return {
        'training_time': training_time,
        'samples_per_second': samples_per_second,
        'gpu_utilization': manager.get_performance_stats()
    }

# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def example_multi_gpu_training():
    """Example of multi-GPU training setup."""
    # Define model and dataset functions
    def create_model():
        return nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
    
    def create_datasets():
        # Create dummy datasets
        train_data = torch.randn(1000, 784)
        train_labels = torch.randint(0, 10, (1000,))
        val_data = torch.randn(200, 784)
        val_labels = torch.randint(0, 10, (200,))
        
        train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
        val_dataset = torch.utils.data.TensorDataset(val_data, val_labels)
        
        return train_dataset, val_dataset
    
    # Configure multi-GPU training
    config = MultiGPUConfig(
        strategy='auto',
        num_gpus=2,
        batch_size=64,
        num_workers=4
    )
    
    # Launch training
    launch_multi_gpu_training(
        create_model, create_datasets, config,
        epochs=10,
        learning_rate=1e-3
    )

if __name__ == "__main__":
    example_multi_gpu_training() 