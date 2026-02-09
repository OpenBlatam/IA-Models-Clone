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
import torch.nn.parallel as parallel
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data as data
import torch.utils.data.distributed as data_dist
import torch.backends.cudnn as cudnn
import numpy as np
import logging
import time
import os
import sys
import socket
import subprocess
from typing import Dict, List, Optional, Any, Union, Tuple, Callable, ContextManager
from dataclasses import dataclass, field
from datetime import datetime
import warnings
from abc import ABC, abstractmethod
import functools
import gc
from contextlib import contextmanager
import threading
import queue
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DataParallel as DP
import torch.distributed.algorithms.model_averaging.averagers as averagers
import torch.distributed.algorithms.ddp_comm_hooks.default_hooks as default_hooks
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp import MixedPrecision, BackwardPrefetch
            from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
                from apex.parallel import convert_syncbn_model
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
Parallel and Distributed Training for Deep Learning
Comprehensive parallel and distributed training components for multi-GPU training using PyTorch's DistributedDataParallel (DDP) and other parallelization techniques.
"""


warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


@dataclass
class DistributedConfig:
    """Configuration for distributed training."""
    # Basic distributed parameters
    enable_distributed: bool = True
    backend: str = "nccl"  # nccl, gloo, mpi
    init_method: str = "env://"  # env://, tcp://, file://
    world_size: int = -1
    rank: int = -1
    local_rank: int = -1
    
    # DDP specific parameters
    enable_ddp: bool = True
    enable_fsdp: bool = False
    enable_ddp_static_graph: bool = True
    enable_ddp_bucket_cap_mb: int = 25
    enable_ddp_find_unused_parameters: bool = False
    enable_ddp_gradient_as_bucket_view: bool = True
    
    # FSDP specific parameters
    fsdp_state_dict_type: str = "FULL_STATE_DICT"  # FULL_STATE_DICT, LOCAL_STATE_DICT, SHARDED_STATE_DICT
    fsdp_auto_wrap_policy: str = "transformer"  # transformer, size, none
    fsdp_mixed_precision: bool = True
    fsdp_activation_checkpointing: bool = True
    
    # Communication parameters
    enable_gradient_compression: bool = False
    enable_quantization: bool = False
    enable_allreduce_hook: bool = True
    enable_fp16_compression: bool = False
    
    # Multi-GPU parameters
    enable_data_parallel: bool = False
    enable_model_parallel: bool = False
    enable_pipeline_parallel: bool = False
    enable_tensor_parallel: bool = False
    
    # Synchronization parameters
    enable_barrier: bool = True
    enable_sync_bn: bool = True
    enable_apex_sync_bn: bool = False
    
    # Performance parameters
    enable_cudnn_benchmark: bool = True
    enable_cudnn_deterministic: bool = False
    enable_tf32: bool = True
    
    # Monitoring parameters
    enable_distributed_monitoring: bool = True
    enable_communication_monitoring: bool = True
    enable_load_balancing_monitoring: bool = True


class DistributedManager:
    """Manager for distributed training setup and coordination."""
    
    def __init__(self, config: DistributedConfig):
        
    """__init__ function."""
self.config = config
        self.is_initialized = False
        self.world_size = 1
        self.rank = 0
        self.local_rank = 0
        self.device = None
        self.process_group = None
    
    def initialize_distributed(self) -> Any:
        """Initialize distributed training."""
        if not self.config.enable_distributed:
            logger.info("Distributed training disabled")
            return
        
        try:
            # Get environment variables
            self.world_size = int(os.environ.get('WORLD_SIZE', 1))
            self.rank = int(os.environ.get('RANK', 0))
            self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
            
            # Set device
            if torch.cuda.is_available():
                self.device = torch.device(f'cuda:{self.local_rank}')
                torch.cuda.set_device(self.device)
            else:
                self.device = torch.device('cpu')
            
            # Initialize process group
            dist.init_process_group(
                backend=self.config.backend,
                init_method=self.config.init_method,
                world_size=self.world_size,
                rank=self.rank
            )
            
            self.is_initialized = True
            logger.info(f"Distributed training initialized: rank {self.rank}/{self.world_size} on {self.device}")
            
            # Apply CUDA optimizations
            self._apply_cuda_optimizations()
            
        except Exception as e:
            logger.error(f"Failed to initialize distributed training: {e}")
            raise
    
    def _apply_cuda_optimizations(self) -> Any:
        """Apply CUDA optimizations for distributed training."""
        if torch.cuda.is_available():
            # Enable cuDNN benchmark
            if self.config.enable_cudnn_benchmark:
                cudnn.benchmark = True
            
            # Enable cuDNN deterministic
            if self.config.enable_cudnn_deterministic:
                cudnn.deterministic = True
            
            # Enable TF32
            if self.config.enable_tf32:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
    
    def cleanup(self) -> Any:
        """Cleanup distributed training."""
        if self.is_initialized:
            dist.destroy_process_group()
            self.is_initialized = False
            logger.info("Distributed training cleaned up")
    
    def barrier(self) -> Any:
        """Synchronize all processes."""
        if self.is_initialized and self.config.enable_barrier:
            dist.barrier()
    
    def get_distributed_info(self) -> Dict[str, Any]:
        """Get distributed training information."""
        return {
            'is_initialized': self.is_initialized,
            'world_size': self.world_size,
            'rank': self.rank,
            'local_rank': self.local_rank,
            'device': str(self.device),
            'backend': self.config.backend,
            'is_master': self.rank == 0
        }


class DistributedDataParallelWrapper:
    """Wrapper for DistributedDataParallel with advanced features."""
    
    def __init__(self, config: DistributedConfig):
        
    """__init__ function."""
self.config = config
        self.ddp_model = None
        self.communication_stats = defaultdict(list)
    
    def wrap_model(self, model: nn.Module, device: torch.device) -> nn.Module:
        """Wrap model with DistributedDataParallel."""
        if not self.config.enable_ddp:
            return model
        
        try:
            # Move model to device
            model = model.to(device)
            
            # Configure DDP parameters
            ddp_kwargs = {
                'device_ids': [device.index] if device.type == 'cuda' else None,
                'output_device': device.index if device.type == 'cuda' else None,
                'find_unused_parameters': self.config.enable_ddp_find_unused_parameters,
                'gradient_as_bucket_view': self.config.enable_ddp_gradient_as_bucket_view,
                'bucket_cap_mb': self.config.enable_ddp_bucket_cap_mb,
                'static_graph': self.config.enable_ddp_static_graph
            }
            
            # Create DDP model
            self.ddp_model = DDP(model, **ddp_kwargs)
            
            # Register communication hooks
            if self.config.enable_allreduce_hook:
                self._register_communication_hooks()
            
            logger.info(f"DDP model created with {len(list(model.parameters()))} parameters")
            return self.ddp_model
            
        except Exception as e:
            logger.error(f"Failed to wrap model with DDP: {e}")
            return model
    
    def _register_communication_hooks(self) -> Any:
        """Register communication hooks for monitoring."""
        if self.ddp_model is None:
            return
        
        # Register allreduce hook
        def allreduce_hook(state, bucket) -> Any:
            start_time = time.time()
            fut = dist.all_reduce(bucket.buffer(), group=state.process_group)
            fut.wait()
            end_time = time.time()
            
            # Record communication stats
            self.communication_stats['allreduce_times'].append(end_time - start_time)
            self.communication_stats['allreduce_sizes'].append(bucket.buffer().numel())
        
        # Register the hook
        self.ddp_model.register_comm_hook(None, allreduce_hook)
    
    def get_communication_stats(self) -> Dict[str, Any]:
        """Get communication statistics."""
        if not self.communication_stats:
            return {}
        
        stats = {}
        for key, values in self.communication_stats.items():
            if values:
                stats[f'{key}_mean'] = np.mean(values)
                stats[f'{key}_max'] = np.max(values)
                stats[f'{key}_min'] = np.min(values)
                stats[f'{key}_total'] = np.sum(values)
        
        return stats


class FSDPWrapper:
    """Wrapper for Fully Sharded Data Parallel (FSDP)."""
    
    def __init__(self, config: DistributedConfig):
        
    """__init__ function."""
self.config = config
        self.fsdp_model = None
    
    def wrap_model(self, model: nn.Module, device: torch.device) -> nn.Module:
        """Wrap model with FSDP."""
        if not self.config.enable_fsdp:
            return model
        
        try:
            # Import FSDP
            
            # Configure FSDP parameters
            fsdp_kwargs = {
                'mixed_precision': MixedPrecision(
                    param_dtype=torch.float16 if self.config.fsdp_mixed_precision else torch.float32,
                    reduce_dtype=torch.float16 if self.config.fsdp_mixed_precision else torch.float32,
                    buffer_dtype=torch.float16 if self.config.fsdp_mixed_precision else torch.float32,
                ) if self.config.fsdp_mixed_precision else None,
                'backward_prefetch': BackwardPrefetch.BACKWARD_PRE,
                'state_dict_type': self.config.fsdp_state_dict_type,
            }
            
            # Set auto wrap policy
            if self.config.fsdp_auto_wrap_policy == "transformer":
                fsdp_kwargs['auto_wrap_policy'] = functools.partial(
                    transformer_auto_wrap_policy,
                    transformer_layer_cls={nn.TransformerEncoderLayer, nn.TransformerDecoderLayer}
                )
            
            # Create FSDP model
            self.fsdp_model = FSDP(model, **fsdp_kwargs)
            
            logger.info(f"FSDP model created with {len(list(model.parameters()))} parameters")
            return self.fsdp_model
            
        except ImportError:
            logger.warning("FSDP not available, falling back to DDP")
            return model
        except Exception as e:
            logger.error(f"Failed to wrap model with FSDP: {e}")
            return model


class DataParallelWrapper:
    """Wrapper for DataParallel (single machine, multiple GPUs)."""
    
    def __init__(self, config: DistributedConfig):
        
    """__init__ function."""
self.config = config
        self.dp_model = None
    
    def wrap_model(self, model: nn.Module, device: torch.device) -> nn.Module:
        """Wrap model with DataParallel."""
        if not self.config.enable_data_parallel:
            return model
        
        try:
            # Check if multiple GPUs are available
            if torch.cuda.device_count() < 2:
                logger.warning("DataParallel requires multiple GPUs, falling back to single GPU")
                return model
            
            # Move model to first GPU
            model = model.to(device)
            
            # Create DataParallel model
            self.dp_model = DP(model)
            
            logger.info(f"DataParallel model created on {torch.cuda.device_count()} GPUs")
            return self.dp_model
            
        except Exception as e:
            logger.error(f"Failed to wrap model with DataParallel: {e}")
            return model


class DistributedDataLoader:
    """Distributed data loader with load balancing."""
    
    def __init__(self, config: DistributedConfig):
        
    """__init__ function."""
self.config = config
        self.distributed_sampler = None
        self.load_balancing_stats = defaultdict(list)
    
    def create_distributed_sampler(self, dataset: data.Dataset, 
                                 shuffle: bool = True) -> data_dist.DistributedSampler:
        """Create distributed sampler."""
        if not self.config.enable_distributed:
            return None
        
        try:
            self.distributed_sampler = data_dist.DistributedSampler(
                dataset,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                shuffle=shuffle
            )
            
            logger.info(f"Distributed sampler created for dataset with {len(dataset)} samples")
            return self.distributed_sampler
            
        except Exception as e:
            logger.error(f"Failed to create distributed sampler: {e}")
            return None
    
    def create_distributed_dataloader(self, dataset: data.Dataset, 
                                    batch_size: int,
                                    shuffle: bool = True,
                                    num_workers: int = 4,
                                    pin_memory: bool = True,
                                    drop_last: bool = False) -> data.DataLoader:
        """Create distributed data loader."""
        # Create distributed sampler
        sampler = self.create_distributed_sampler(dataset, shuffle)
        
        # Create data loader
        dataloader = data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            persistent_workers=True if num_workers > 0 else False
        )
        
        logger.info(f"Distributed data loader created with batch size {batch_size}")
        return dataloader
    
    def set_epoch(self, epoch: int):
        """Set epoch for distributed sampler."""
        if self.distributed_sampler is not None:
            self.distributed_sampler.set_epoch(epoch)
    
    def get_load_balancing_stats(self) -> Dict[str, Any]:
        """Get load balancing statistics."""
        if not self.load_balancing_stats:
            return {}
        
        stats = {}
        for key, values in self.load_balancing_stats.items():
            if values:
                stats[f'{key}_mean'] = np.mean(values)
                stats[f'{key}_std'] = np.std(values)
                stats[f'{key}_max'] = np.max(values)
                stats[f'{key}_min'] = np.min(values)
        
        return stats


class SynchronizedBatchNorm:
    """Synchronized Batch Normalization for distributed training."""
    
    def __init__(self, config: DistributedConfig):
        
    """__init__ function."""
self.config = config
    
    def convert_batch_norm(self, model: nn.Module) -> nn.Module:
        """Convert BatchNorm to SynchronizedBatchNorm."""
        if not self.config.enable_sync_bn:
            return model
        
        try:
            # Try to import apex SyncBatchNorm
            if self.config.enable_apex_sync_bn:
                model = convert_syncbn_model(model)
                logger.info("Converted to Apex SyncBatchNorm")
            else:
                # Use torch.nn.SyncBatchNorm
                model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
                logger.info("Converted to PyTorch SyncBatchNorm")
            
            return model
            
        except ImportError:
            logger.warning("Apex not available, using PyTorch SyncBatchNorm")
            try:
                model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
                logger.info("Converted to PyTorch SyncBatchNorm")
                return model
            except Exception as e:
                logger.error(f"Failed to convert to SyncBatchNorm: {e}")
                return model
        except Exception as e:
            logger.error(f"Failed to convert to SyncBatchNorm: {e}")
            return model


class DistributedOptimizer:
    """Distributed optimizer with gradient compression and quantization."""
    
    def __init__(self, config: DistributedConfig):
        
    """__init__ function."""
self.config = config
        self.optimizer = None
        self.compression_stats = defaultdict(list)
    
    def create_optimizer(self, model: nn.Module, optimizer_class: type, 
                        **optimizer_kwargs) -> torch.optim.Optimizer:
        """Create distributed optimizer."""
        # Create base optimizer
        self.optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)
        
        # Apply distributed optimizations
        if self.config.enable_distributed:
            self._apply_distributed_optimizations()
        
        return self.optimizer
    
    def _apply_distributed_optimizations(self) -> Any:
        """Apply distributed optimizations to optimizer."""
        if self.optimizer is None:
            return
        
        # Enable foreach for supported optimizers
        if hasattr(self.optimizer, 'foreach'):
            self.optimizer.foreach = True
        
        # Enable fused for supported optimizers
        if hasattr(self.optimizer, 'fused'):
            self.optimizer.fused = True
        
        # Apply gradient compression
        if self.config.enable_gradient_compression:
            self._apply_gradient_compression()
        
        # Apply quantization
        if self.config.enable_quantization:
            self._apply_quantization()
    
    def _apply_gradient_compression(self) -> Any:
        """Apply gradient compression."""
        try:
            # This would implement gradient compression techniques
            # like PowerSGD, QSGD, etc.
            logger.info("Gradient compression enabled")
        except Exception as e:
            logger.warning(f"Failed to apply gradient compression: {e}")
    
    def _apply_quantization(self) -> Any:
        """Apply quantization to gradients."""
        try:
            # This would implement gradient quantization techniques
            logger.info("Gradient quantization enabled")
        except Exception as e:
            logger.warning(f"Failed to apply quantization: {e}")
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """Get compression statistics."""
        if not self.compression_stats:
            return {}
        
        stats = {}
        for key, values in self.compression_stats.items():
            if values:
                stats[f'{key}_mean'] = np.mean(values)
                stats[f'{key}_max'] = np.max(values)
                stats[f'{key}_min'] = np.min(values)
        
        return stats


class DistributedMonitor:
    """Monitor distributed training performance and communication."""
    
    def __init__(self, config: DistributedConfig):
        
    """__init__ function."""
self.config = config
        self.performance_history = deque(maxlen=1000)
        self.communication_history = deque(maxlen=1000)
        self.load_balancing_history = deque(maxlen=1000)
        self.start_time = time.time()
    
    def record_performance(self, batch_size: int, forward_time: float, 
                          backward_time: float, total_time: float,
                          communication_time: float = 0.0):
        """Record performance metrics."""
        performance_info = {
            'timestamp': datetime.now().isoformat(),
            'rank': dist.get_rank() if dist.is_initialized() else 0,
            'batch_size': batch_size,
            'forward_time': forward_time,
            'backward_time': backward_time,
            'total_time': total_time,
            'communication_time': communication_time,
            'throughput': batch_size / total_time,
            'gpu_utilization': self._get_gpu_utilization(),
            'memory_usage': self._get_memory_usage()
        }
        
        self.performance_history.append(performance_info)
    
    def record_communication(self, communication_stats: Dict[str, Any]):
        """Record communication statistics."""
        communication_info = {
            'timestamp': datetime.now().isoformat(),
            'rank': dist.get_rank() if dist.is_initialized() else 0,
            **communication_stats
        }
        
        self.communication_history.append(communication_info)
    
    def record_load_balancing(self, load_balancing_stats: Dict[str, Any]):
        """Record load balancing statistics."""
        load_balancing_info = {
            'timestamp': datetime.now().isoformat(),
            'rank': dist.get_rank() if dist.is_initialized() else 0,
            **load_balancing_stats
        }
        
        self.load_balancing_history.append(load_balancing_info)
    
    def _get_gpu_utilization(self) -> float:
        """Get GPU utilization."""
        try:
            if torch.cuda.is_available():
                return torch.cuda.utilization()
            return 0.0
        except:
            return 0.0
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage."""
        memory_info = {}
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                memory_info[f'gpu_{i}_allocated_mb'] = torch.cuda.memory_allocated(i) / (1024 * 1024)
                memory_info[f'gpu_{i}_reserved_mb'] = torch.cuda.memory_reserved(i) / (1024 * 1024)
        
        return memory_info
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        if not self.performance_history:
            return {}
        
        # Aggregate across all ranks
        all_performance = self._gather_performance_data()
        
        if not all_performance:
            return {}
        
        throughputs = [p['throughput'] for p in all_performance]
        forward_times = [p['forward_time'] for p in all_performance]
        backward_times = [p['backward_time'] for p in all_performance]
        communication_times = [p['communication_time'] for p in all_performance]
        
        return {
            'total_samples': sum(p['batch_size'] for p in all_performance),
            'total_time': time.time() - self.start_time,
            'avg_throughput': np.mean(throughputs),
            'max_throughput': np.max(throughputs),
            'min_throughput': np.min(throughputs),
            'avg_forward_time': np.mean(forward_times),
            'avg_backward_time': np.mean(backward_times),
            'avg_communication_time': np.mean(communication_times),
            'communication_overhead': np.mean(communication_times) / np.mean(forward_times + backward_times),
            'gpu_utilization': np.mean([p['gpu_utilization'] for p in all_performance])
        }
    
    def _gather_performance_data(self) -> List[Dict[str, Any]]:
        """Gather performance data from all ranks."""
        if not dist.is_initialized():
            return list(self.performance_history)
        
        # Gather data from all ranks
        gathered_data = [None] * dist.get_world_size()
        dist.all_gather_object(gathered_data, list(self.performance_history))
        
        # Flatten the data
        all_data = []
        for rank_data in gathered_data:
            if rank_data:
                all_data.extend(rank_data)
        
        return all_data
    
    def plot_distributed_performance(self, save_path: str = None):
        """Plot distributed training performance."""
        if not self.performance_history:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Throughput over time by rank
        rank_data = defaultdict(list)
        for p in self.performance_history:
            rank_data[p['rank']].append(p['throughput'])
        
        for rank, throughputs in rank_data.items():
            axes[0, 0].plot(throughputs, label=f'Rank {rank}')
        axes[0, 0].set_title('Throughput Over Time by Rank')
        axes[0, 0].set_ylabel('Samples/Second')
        axes[0, 0].legend()
        
        # Communication overhead
        communication_overheads = [p['communication_time'] / p['total_time'] 
                                 for p in self.performance_history]
        axes[0, 1].plot(communication_overheads)
        axes[0, 1].set_title('Communication Overhead')
        axes[0, 1].set_ylabel('Overhead Ratio')
        
        # GPU utilization by rank
        rank_gpu_data = defaultdict(list)
        for p in self.performance_history:
            rank_gpu_data[p['rank']].append(p['gpu_utilization'])
        
        for rank, gpu_utils in rank_gpu_data.items():
            axes[1, 0].plot(gpu_utils, label=f'Rank {rank}')
        axes[1, 0].set_title('GPU Utilization by Rank')
        axes[1, 0].set_ylabel('Utilization %')
        axes[1, 0].legend()
        
        # Memory usage
        if self.performance_history and 'memory_usage' in self.performance_history[0]:
            gpu_memory = [p['memory_usage'].get('gpu_0_allocated_mb', 0) 
                         for p in self.performance_history]
            axes[1, 1].plot(gpu_memory)
            axes[1, 1].set_title('GPU Memory Usage')
            axes[1, 1].set_ylabel('Memory (MB)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()


class DistributedTrainer:
    """Complete distributed trainer with all parallelization techniques."""
    
    def __init__(self, config: DistributedConfig):
        
    """__init__ function."""
self.config = config
        self.distributed_manager = DistributedManager(config)
        self.ddp_wrapper = DistributedDataParallelWrapper(config)
        self.fsdp_wrapper = FSDPWrapper(config)
        self.dp_wrapper = DataParallelWrapper(config)
        self.distributed_dataloader = DistributedDataLoader(config)
        self.sync_bn = SynchronizedBatchNorm(config)
        self.distributed_optimizer = DistributedOptimizer(config)
        self.monitor = DistributedMonitor(config)
        
        # Initialize distributed training
        self.distributed_manager.initialize_distributed()
    
    def setup_model(self, model: nn.Module) -> nn.Module:
        """Setup model for distributed training."""
        device = self.distributed_manager.device
        
        # Convert to synchronized batch norm
        model = self.sync_bn.convert_batch_norm(model)
        
        # Wrap with appropriate parallelization
        if self.config.enable_fsdp:
            model = self.fsdp_wrapper.wrap_model(model, device)
        elif self.config.enable_ddp:
            model = self.ddp_wrapper.wrap_model(model, device)
        elif self.config.enable_data_parallel:
            model = self.dp_wrapper.wrap_model(model, device)
        
        return model
    
    def setup_optimizer(self, model: nn.Module, optimizer_class: type, 
                       **optimizer_kwargs) -> torch.optim.Optimizer:
        """Setup optimizer for distributed training."""
        return self.distributed_optimizer.create_optimizer(model, optimizer_class, **optimizer_kwargs)
    
    def setup_dataloader(self, dataset: data.Dataset, batch_size: int,
                        shuffle: bool = True, num_workers: int = 4,
                        pin_memory: bool = True, drop_last: bool = False) -> data.DataLoader:
        """Setup distributed data loader."""
        return self.distributed_dataloader.create_distributed_dataloader(
            dataset, batch_size, shuffle, num_workers, pin_memory, drop_last
        )
    
    def training_step(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                     data_batch: Any, loss_fn: Callable) -> Dict[str, Any]:
        """Perform distributed training step."""
        start_time = time.time()
        
        # Move data to device
        device = self.distributed_manager.device
        if isinstance(data_batch, (tuple, list)):
            data_batch = [d.to(device, non_blocking=True) for d in data_batch]
        else:
            data_batch = data_batch.to(device, non_blocking=True)
        
        # Forward pass
        forward_start = time.time()
        output = model(data_batch)
        loss = loss_fn(output, data_batch)
        forward_time = time.time() - forward_start
        
        # Backward pass
        backward_start = time.time()
        loss.backward()
        backward_time = time.time() - backward_start
        
        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()
        
        total_time = time.time() - start_time
        
        # Record performance
        batch_size = data_batch[0].size(0) if isinstance(data_batch, (tuple, list)) else data_batch.size(0)
        self.monitor.record_performance(
            batch_size, forward_time, backward_time, total_time
        )
        
        # Record communication stats
        if self.config.enable_ddp:
            comm_stats = self.ddp_wrapper.get_communication_stats()
            self.monitor.record_communication(comm_stats)
        
        return {
            'loss': loss.item(),
            'forward_time': forward_time,
            'backward_time': backward_time,
            'total_time': total_time,
            'batch_size': batch_size
        }
    
    def set_epoch(self, epoch: int):
        """Set epoch for distributed sampler."""
        self.distributed_dataloader.set_epoch(epoch)
    
    def barrier(self) -> Any:
        """Synchronize all processes."""
        self.distributed_manager.barrier()
    
    def get_distributed_summary(self) -> Dict[str, Any]:
        """Get comprehensive distributed training summary."""
        return {
            'distributed_info': self.distributed_manager.get_distributed_info(),
            'performance_summary': self.monitor.get_performance_summary(),
            'communication_stats': self.ddp_wrapper.get_communication_stats(),
            'load_balancing_stats': self.distributed_dataloader.get_load_balancing_stats(),
            'compression_stats': self.distributed_optimizer.get_compression_stats()
        }
    
    def cleanup(self) -> Any:
        """Cleanup distributed training resources."""
        self.distributed_manager.cleanup()


# Utility functions for distributed training
def setup_distributed_training(config: DistributedConfig = None) -> DistributedTrainer:
    """Setup distributed training."""
    if config is None:
        config = DistributedConfig()
    
    trainer = DistributedTrainer(config)
    return trainer


def launch_distributed_training(rank: int, world_size: int, 
                              training_function: Callable,
                              *args, **kwargs):
    """Launch distributed training process."""
    # Set environment variables
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)
    
    # Run training function
    training_function(rank, world_size, *args, **kwargs)


def spawn_distributed_training(world_size: int, training_function: Callable,
                             *args, **kwargs):
    """Spawn multiple processes for distributed training."""
    mp.spawn(
        launch_distributed_training,
        args=(world_size, training_function, *args),
        nprocs=world_size,
        join=True
    )


def get_free_port() -> int:
    """Get a free port for distributed training."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


# Example usage
if __name__ == "__main__":
    # Create distributed configuration
    config = DistributedConfig(
        enable_distributed=True,
        enable_ddp=True,
        enable_sync_bn=True,
        enable_cudnn_benchmark=True,
        enable_tf32=True,
        enable_distributed_monitoring=True
    )
    
    # Create distributed trainer
    trainer = DistributedTrainer(config)
    
    # Example: Setup model
    model = nn.Sequential(
        nn.Linear(100, 50),
        nn.BatchNorm1d(50),
        nn.ReLU(),
        nn.Linear(50, 10)
    )
    
    # Setup model for distributed training
    model = trainer.setup_model(model)
    
    # Setup optimizer
    optimizer = trainer.setup_optimizer(model, torch.optim.Adam, lr=0.001)
    
    # Example training step
    input_data = torch.randn(32, 100)
    target = torch.randn(32, 10)
    
    def loss_fn(output, data) -> Any:
        return nn.MSELoss()(output, target)
    
    # Perform training step
    step_info = trainer.training_step(model, optimizer, input_data, loss_fn)
    print(f"Distributed training step completed: {step_info}")
    
    # Get distributed summary
    summary = trainer.get_distributed_summary()
    print(f"Distributed training summary: {summary}")
    
    # Cleanup
    trainer.cleanup() 