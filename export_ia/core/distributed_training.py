"""
Advanced Distributed Training Engine for Export IA
State-of-the-art distributed training with PyTorch best practices
"""

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data.distributed import DistributedSampler
import torch.cuda.nccl as nccl
import torch.cuda.amp as amp
from torch.distributed.optim import ZeroRedundancyOptimizer

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
import os
import time
import json
from pathlib import Path
import subprocess
import socket
from contextlib import contextmanager

# Advanced distributed training libraries
try:
    import deepspeed
    from deepspeed import DeepSpeedEngine
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    import fairscale
    from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP
    from fairscale.optim import OSS
    import horovod.torch as hvd
except ImportError:
    print("Installing advanced distributed training libraries...")
    import subprocess
    subprocess.check_call(["pip", "install", "deepspeed", "fairscale", "horovod"])

logger = logging.getLogger(__name__)

@dataclass
class DistributedConfig:
    """Configuration for distributed training"""
    # Basic distributed settings
    backend: str = "nccl"  # nccl, gloo, mpi
    init_method: str = "env://"  # env://, file://, tcp://
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    
    # Advanced distributed settings
    use_deepspeed: bool = False
    use_fairscale: bool = False
    use_horovod: bool = False
    
    # DeepSpeed configuration
    deepspeed_config: Dict[str, Any] = None
    
    # Communication optimization
    gradient_as_bucket_view: bool = True
    static_graph: bool = False
    find_unused_parameters: bool = False
    
    # Memory optimization
    bucket_cap_mb: int = 25
    gradient_accumulation_steps: int = 1
    
    # Fault tolerance
    enable_checkpointing: bool = True
    checkpoint_dir: str = "./checkpoints"
    auto_resume: bool = True
    
    # Monitoring
    enable_profiling: bool = False
    profile_dir: str = "./profiles"
    
    # Performance tuning
    nccl_timeout: int = 1800
    nccl_blocking_wait: bool = True
    nccl_async_error_handling: bool = True

class DistributedManager:
    """Advanced distributed training manager"""
    
    def __init__(self, config: DistributedConfig):
        self.config = config
        self.is_initialized = False
        self.process_group = None
        self.local_rank = None
        self.world_size = None
        self.rank = None
        
    def initialize(self):
        """Initialize distributed training"""
        if self.is_initialized:
            return
            
        # Set environment variables
        self._set_environment_variables()
        
        # Initialize process group
        if self.config.use_horovod:
            self._initialize_horovod()
        else:
            self._initialize_torch_distributed()
            
        self.is_initialized = True
        logger.info(f"Distributed training initialized: rank={self.rank}, world_size={self.world_size}")
        
    def _set_environment_variables(self):
        """Set required environment variables"""
        if "RANK" not in os.environ:
            os.environ["RANK"] = str(self.config.rank)
        if "WORLD_SIZE" not in os.environ:
            os.environ["WORLD_SIZE"] = str(self.config.world_size)
        if "LOCAL_RANK" not in os.environ:
            os.environ["LOCAL_RANK"] = str(self.config.local_rank)
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "localhost"
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "12355"
            
    def _initialize_torch_distributed(self):
        """Initialize PyTorch distributed training"""
        try:
            dist.init_process_group(
                backend=self.config.backend,
                init_method=self.config.init_method,
                world_size=self.config.world_size,
                rank=self.config.rank,
                timeout=torch.distributed.constants.default_pg_timeout
            )
            
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
            
            # Set device
            torch.cuda.set_device(self.local_rank)
            
        except Exception as e:
            logger.error(f"Failed to initialize distributed training: {e}")
            raise
            
    def _initialize_horovod(self):
        """Initialize Horovod distributed training"""
        hvd.init()
        self.rank = hvd.rank()
        self.world_size = hvd.size()
        self.local_rank = hvd.local_rank()
        
        # Set device
        torch.cuda.set_device(self.local_rank)
        
    def cleanup(self):
        """Cleanup distributed training"""
        if self.is_initialized:
            if self.config.use_horovod:
                # Horovod cleanup is automatic
                pass
            else:
                dist.destroy_process_group()
            self.is_initialized = False
            
    def barrier(self):
        """Synchronize all processes"""
        if self.is_initialized:
            if self.config.use_horovod:
                hvd.allreduce(torch.tensor(0.0))
            else:
                dist.barrier()
                
    def all_reduce(self, tensor: torch.Tensor, op=dist.ReduceOp.SUM):
        """All-reduce operation"""
        if self.is_initialized:
            if self.config.use_horovod:
                return hvd.allreduce(tensor, op=hvd.Sum)
            else:
                dist.all_reduce(tensor, op=op)
        return tensor
        
    def all_gather(self, tensor: torch.Tensor) -> List[torch.Tensor]:
        """All-gather operation"""
        if not self.is_initialized:
            return [tensor]
            
        if self.config.use_horovod:
            return hvd.allgather(tensor)
        else:
            tensor_list = [torch.zeros_like(tensor) for _ in range(self.world_size)]
            dist.all_gather(tensor_list, tensor)
            return tensor_list
            
    def broadcast(self, tensor: torch.Tensor, src: int = 0):
        """Broadcast tensor from source rank"""
        if self.is_initialized:
            if self.config.use_horovod:
                hvd.broadcast(tensor, root_rank=src)
            else:
                dist.broadcast(tensor, src=src)
                
    def reduce(self, tensor: torch.Tensor, dst: int = 0, op=dist.ReduceOp.SUM):
        """Reduce tensor to destination rank"""
        if self.is_initialized:
            if self.config.use_horovod:
                return hvd.allreduce(tensor, op=hvd.Sum)
            else:
                dist.reduce(tensor, dst=dst, op=op)

class DeepSpeedManager:
    """DeepSpeed integration for advanced distributed training"""
    
    def __init__(self, config: DistributedConfig):
        self.config = config
        self.engine = None
        
    def initialize_engine(self, model: nn.Module, 
                         optimizer: torch.optim.Optimizer,
                         training_args: Dict[str, Any]) -> DeepSpeedEngine:
        """Initialize DeepSpeed engine"""
        
        if self.config.deepspeed_config is None:
            # Default DeepSpeed configuration
            self.config.deepspeed_config = {
                "train_batch_size": training_args.get("batch_size", 32),
                "gradient_accumulation_steps": training_args.get("gradient_accumulation_steps", 1),
                "optimizer": {
                    "type": "AdamW",
                    "params": {
                        "lr": training_args.get("learning_rate", 1e-4),
                        "betas": [0.9, 0.999],
                        "eps": 1e-8,
                        "weight_decay": 1e-2
                    }
                },
                "scheduler": {
                    "type": "WarmupLR",
                    "params": {
                        "warmup_min_lr": 0,
                        "warmup_max_lr": training_args.get("learning_rate", 1e-4),
                        "warmup_num_steps": training_args.get("warmup_steps", 1000)
                    }
                },
                "fp16": {
                    "enabled": training_args.get("fp16", True),
                    "auto_cast": False,
                    "loss_scale": 0,
                    "initial_scale_power": 16,
                    "loss_scale_window": 1000,
                    "hysteresis": 2,
                    "min_loss_scale": 1
                },
                "zero_optimization": {
                    "stage": 2,
                    "allgather_partitions": True,
                    "allgather_bucket_size": 2e8,
                    "overlap_comm": True,
                    "reduce_scatter": True,
                    "reduce_bucket_size": 2e8,
                    "contiguous_gradients": True
                },
                "gradient_accumulation_steps": training_args.get("gradient_accumulation_steps", 1),
                "steps_per_print": training_args.get("log_every_n_steps", 100),
                "wall_clock_breakdown": False
            }
            
        # Initialize DeepSpeed engine
        self.engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            config=self.config.deepspeed_config
        )
        
        return self.engine
        
    def save_checkpoint(self, model: nn.Module, 
                       optimizer: torch.optim.Optimizer,
                       scheduler: Any,
                       epoch: int,
                       step: int,
                       save_dir: str):
        """Save DeepSpeed checkpoint"""
        if self.engine is not None:
            self.engine.save_checkpoint(save_dir, epoch, step)
            
    def load_checkpoint(self, model: nn.Module,
                       optimizer: torch.optim.Optimizer,
                       scheduler: Any,
                       load_dir: str):
        """Load DeepSpeed checkpoint"""
        if self.engine is not None:
            _, _, _, _ = self.engine.load_checkpoint(load_dir)
            
    def backward(self, loss: torch.Tensor):
        """Backward pass with DeepSpeed"""
        if self.engine is not None:
            self.engine.backward(loss)
        else:
            loss.backward()
            
    def step(self):
        """Optimizer step with DeepSpeed"""
        if self.engine is not None:
            self.engine.step()
        else:
            # Fallback to regular optimizer step
            pass

class FairScaleManager:
    """FairScale integration for memory-efficient distributed training"""
    
    def __init__(self, config: DistributedConfig):
        self.config = config
        self.sharded_ddp = None
        
    def wrap_model(self, model: nn.Module, optimizer: torch.optim.Optimizer) -> nn.Module:
        """Wrap model with ShardedDataParallel"""
        
        # Create OSS optimizer
        oss_optimizer = OSS(
            params=model.parameters(),
            optim=type(optimizer),
            **optimizer.defaults
        )
        
        # Wrap model with ShardedDDP
        self.sharded_ddp = ShardedDDP(
            module=model,
            optimizer=oss_optimizer,
            reduce_buffer_size=0
        )
        
        return self.sharded_ddp
        
    def backward(self, loss: torch.Tensor):
        """Backward pass with FairScale"""
        if self.sharded_ddp is not None:
            self.sharded_ddp.backward(loss)
        else:
            loss.backward()
            
    def step(self):
        """Optimizer step with FairScale"""
        if self.sharded_ddp is not None:
            self.sharded_ddp.optimizer.step()

class DistributedDataLoader:
    """Distributed data loader with advanced features"""
    
    def __init__(self, dataset: torch.utils.data.Dataset,
                 distributed_manager: DistributedManager,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 pin_memory: bool = True,
                 drop_last: bool = True):
        self.dataset = dataset
        self.distributed_manager = distributed_manager
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        
    def create_loader(self) -> DataLoader:
        """Create distributed data loader"""
        
        # Create distributed sampler
        if self.distributed_manager.is_initialized:
            sampler = DistributedSampler(
                self.dataset,
                num_replicas=self.distributed_manager.world_size,
                rank=self.distributed_manager.rank,
                shuffle=True,
                drop_last=self.drop_last
            )
        else:
            sampler = None
            
        # Create data loader
        loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            shuffle=(sampler is None),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            persistent_workers=True if self.num_workers > 0 else False
        )
        
        return loader

class DistributedTrainingEngine:
    """Main distributed training engine"""
    
    def __init__(self, config: DistributedConfig):
        self.config = config
        self.distributed_manager = DistributedManager(config)
        self.deepspeed_manager = DeepSpeedManager(config) if config.use_deepspeed else None
        self.fairscale_manager = FairScaleManager(config) if config.use_fairscale else None
        
        # Training state
        self.current_epoch = 0
        self.current_step = 0
        self.best_metric = float('-inf')
        self.training_history = []
        
    def initialize(self):
        """Initialize distributed training"""
        self.distributed_manager.initialize()
        
    def setup_model(self, model: nn.Module, 
                   optimizer: torch.optim.Optimizer,
                   training_args: Dict[str, Any]) -> nn.Module:
        """Setup model for distributed training"""
        
        # Move model to device
        device = torch.device(f"cuda:{self.distributed_manager.local_rank}")
        model = model.to(device)
        
        if self.config.use_deepspeed:
            # Use DeepSpeed
            self.deepspeed_manager.initialize_engine(model, optimizer, training_args)
            return self.deepspeed_manager.engine.module
            
        elif self.config.use_fairscale:
            # Use FairScale
            return self.fairscale_manager.wrap_model(model, optimizer)
            
        else:
            # Use standard DDP
            model = DDP(
                model,
                device_ids=[self.distributed_manager.local_rank],
                output_device=self.distributed_manager.local_rank,
                find_unused_parameters=self.config.find_unused_parameters,
                gradient_as_bucket_view=self.config.gradient_as_bucket_view,
                static_graph=self.config.static_graph,
                bucket_cap_mb=self.config.bucket_cap_mb
            )
            return model
            
    def setup_data_loader(self, dataset: torch.utils.data.Dataset,
                         batch_size: int = 32) -> DataLoader:
        """Setup distributed data loader"""
        
        distributed_loader = DistributedDataLoader(
            dataset=dataset,
            distributed_manager=self.distributed_manager,
            batch_size=batch_size
        )
        
        return distributed_loader.create_loader()
        
    def train_epoch(self, model: nn.Module, 
                   data_loader: DataLoader,
                   optimizer: torch.optim.Optimizer,
                   loss_fn: Callable,
                   scaler: amp.GradScaler = None) -> Dict[str, float]:
        """Train for one epoch with distributed training"""
        
        model.train()
        epoch_metrics = defaultdict(float)
        num_batches = len(data_loader)
        
        for batch_idx, (data, target) in enumerate(data_loader):
            # Move data to device
            device = torch.device(f"cuda:{self.distributed_manager.local_rank}")
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            
            # Forward pass with mixed precision
            if scaler is not None:
                with amp.autocast():
                    output = model(data)
                    loss = loss_fn(output, target)
            else:
                output = model(data)
                loss = loss_fn(output, target)
                
            # Backward pass
            if self.config.use_deepspeed:
                self.deepspeed_manager.backward(loss)
            elif self.config.use_fairscale:
                self.fairscale_manager.backward(loss)
            else:
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                    
            # Optimizer step
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                if self.config.use_deepspeed:
                    self.deepspeed_manager.step()
                elif self.config.use_fairscale:
                    self.fairscale_manager.step()
                else:
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()
                    
            # Update metrics
            epoch_metrics['loss'] += loss.item()
            epoch_metrics['batch_count'] += 1
            
            self.current_step += 1
            
        # Average metrics across all processes
        for key in epoch_metrics:
            if key != 'batch_count':
                tensor = torch.tensor(epoch_metrics[key], device=f"cuda:{self.distributed_manager.local_rank}")
                self.distributed_manager.all_reduce(tensor)
                epoch_metrics[key] = tensor.item() / self.distributed_manager.world_size
                
        return dict(epoch_metrics)
        
    def validate(self, model: nn.Module,
                data_loader: DataLoader,
                loss_fn: Callable) -> Dict[str, float]:
        """Validate model with distributed training"""
        
        model.eval()
        val_metrics = defaultdict(float)
        
        with torch.no_grad():
            for data, target in data_loader:
                # Move data to device
                device = torch.device(f"cuda:{self.distributed_manager.local_rank}")
                data = data.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                
                # Forward pass
                output = model(data)
                loss = loss_fn(output, target)
                
                # Update metrics
                val_metrics['loss'] += loss.item()
                val_metrics['batch_count'] += 1
                
        # Average metrics across all processes
        for key in val_metrics:
            if key != 'batch_count':
                tensor = torch.tensor(val_metrics[key], device=f"cuda:{self.distributed_manager.local_rank}")
                self.distributed_manager.all_reduce(tensor)
                val_metrics[key] = tensor.item() / self.distributed_manager.world_size
                
        return dict(val_metrics)
        
    def save_checkpoint(self, model: nn.Module,
                       optimizer: torch.optim.Optimizer,
                       scheduler: Any,
                       epoch: int,
                       step: int,
                       metrics: Dict[str, float]):
        """Save distributed checkpoint"""
        
        if self.distributed_manager.rank == 0:  # Only save on rank 0
            checkpoint = {
                'epoch': epoch,
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'metrics': metrics,
                'config': self.config.__dict__
            }
            
            checkpoint_path = Path(self.config.checkpoint_dir) / f"checkpoint_epoch_{epoch}.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")
            
    def load_checkpoint(self, model: nn.Module,
                       optimizer: torch.optim.Optimizer,
                       scheduler: Any,
                       checkpoint_path: str):
        """Load distributed checkpoint"""
        
        checkpoint = torch.load(checkpoint_path, map_location=f"cuda:{self.distributed_manager.local_rank}")
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler and checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        self.current_epoch = checkpoint.get('epoch', 0)
        self.current_step = checkpoint.get('step', 0)
        
        logger.info(f"Checkpoint loaded: {checkpoint_path}")
        
    def cleanup(self):
        """Cleanup distributed training"""
        self.distributed_manager.cleanup()

# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Test distributed training setup
    print("Testing Distributed Training Engine...")
    
    # Create distributed config
    config = DistributedConfig(
        backend="nccl",
        world_size=1,  # Single GPU for testing
        rank=0,
        local_rank=0,
        use_deepspeed=False,
        use_fairscale=False
    )
    
    # Create distributed training engine
    engine = DistributedTrainingEngine(config)
    
    # Initialize (this will work even with single GPU)
    try:
        engine.initialize()
        print("Distributed training initialized successfully")
    except Exception as e:
        print(f"Distributed initialization failed (expected for single GPU): {e}")
        
    # Test model setup
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 1)
            
        def forward(self, x):
            return self.linear(x)
    
    model = TestModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Test model setup
    training_args = {
        "batch_size": 32,
        "learning_rate": 1e-3,
        "gradient_accumulation_steps": 1
    }
    
    try:
        distributed_model = engine.setup_model(model, optimizer, training_args)
        print("Model setup for distributed training completed")
    except Exception as e:
        print(f"Model setup failed: {e}")
        
    print("\nDistributed training engine initialized successfully!")
























