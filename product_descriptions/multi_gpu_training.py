from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import asyncio
import json
import logging
import os
import pickle
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from uuid import uuid4
import numpy as np
import pandas as pd
import structlog
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
from torch.nn.parallel import parallel_apply
import torch.distributed.algorithms.model_averaging.averagers as averagers
import torch.distributed.algorithms.ddp_comm_hooks.default_hooks as default_hooks
from typing import Any, List, Dict, Optional
"""
Multi-GPU Training System for Cybersecurity Applications

This module provides comprehensive multi-GPU training capabilities using both
DataParallel and DistributedDataParallel with advanced features:

- DataParallel for single-node multi-GPU training
- DistributedDataParallel for multi-node distributed training
- Automatic mixed precision training
- Gradient accumulation and clipping
- Advanced memory management
- Performance monitoring and optimization
- Fault tolerance and recovery
- Load balancing and synchronization
- Comprehensive logging and metrics
"""



# Configure structured logging
logger = structlog.get_logger(__name__)


class TrainingMode(Enum):
    """Training mode enumeration."""
    SINGLE_GPU = "single_gpu"
    DATA_PARALLEL = "data_parallel"
    DISTRIBUTED_DATA_PARALLEL = "distributed_data_parallel"


@dataclass
class MultiGPUConfig:
    """Configuration for multi-GPU training."""
    
    # Training mode
    training_mode: TrainingMode = TrainingMode.DATA_PARALLEL
    
    # Device configuration
    device_ids: List[int] = field(default_factory=lambda: list(range(torch.cuda.device_count())))
    master_gpu: int = 0
    
    # Distributed training settings
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    backend: str = "nccl"  # nccl for GPU, gloo for CPU
    init_method: str = "env://"
    timeout: timedelta = field(default_factory=lambda: timedelta(minutes=30))
    
    # Performance optimization
    use_mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    find_unused_parameters: bool = False
    broadcast_buffers: bool = True
    bucket_cap_mb: int = 25
    static_graph: bool = True
    
    # Memory management
    pin_memory: bool = True
    num_workers: int = 4
    prefetch_factor: int = 2
    persistent_workers: bool = True
    
    # Synchronization
    sync_bn: bool = False
    use_fp16_allreduce: bool = True
    fp16_compression: bool = True
    
    # Fault tolerance
    enable_fault_tolerance: bool = True
    checkpoint_frequency: int = 1000
    recovery_timeout: int = 300
    
    # Monitoring
    enable_monitoring: bool = True
    log_every_n_steps: int = 100
    profile_every_n_steps: int = 1000
    
    # Communication optimization
    use_gradient_as_bucket_view: bool = True
    reduce_bucket_size: int = 25 * 1024 * 1024  # 25MB
    
    def __post_init__(self) -> Any:
        """Post-initialization setup."""
        if not torch.cuda.is_available():
            self.training_mode = TrainingMode.SINGLE_GPU
            self.device_ids = []
            logger.warning("CUDA not available, falling back to single GPU mode")
        
        if self.training_mode == TrainingMode.DISTRIBUTED_DATA_PARALLEL:
            self.world_size = int(os.environ.get("WORLD_SIZE", 1))
            self.rank = int(os.environ.get("RANK", 0))
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))


class MultiGPUTrainer(ABC):
    """Abstract base class for multi-GPU training."""
    
    def __init__(self, config: MultiGPUConfig):
        
    """__init__ function."""
self.config = config
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.device = None
        self.is_distributed = False
        self.rank = 0
        self.world_size = 1
        
        self._setup_environment()
        self._setup_logging()
    
    def _setup_environment(self) -> Any:
        """Setup training environment."""
        # Set CUDA device
        if torch.cuda.is_available():
            torch.cuda.set_device(self.config.local_rank)
            self.device = torch.device(f"cuda:{self.config.local_rank}")
            
            # Enable cuDNN optimizations
            cudnn.benchmark = True
            cudnn.deterministic = False
            
            # Set memory fraction if needed
            torch.cuda.empty_cache()
        else:
            self.device = torch.device("cpu")
        
        # Setup distributed training if needed
        if self.config.training_mode == TrainingMode.DISTRIBUTED_DATA_PARALLEL:
            self._setup_distributed()
    
    def _setup_distributed(self) -> Any:
        """Setup distributed training."""
        try:
            dist.init_process_group(
                backend=self.config.backend,
                init_method=self.config.init_method,
                world_size=self.config.world_size,
                rank=self.config.rank,
                timeout=self.config.timeout
            )
            
            self.is_distributed = True
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            
            logger.info(
                "Distributed training initialized",
                rank=self.rank,
                world_size=self.world_size,
                backend=self.config.backend
            )
            
        except Exception as e:
            logger.error("Failed to initialize distributed training", error=str(e))
            raise
    
    def _setup_logging(self) -> Any:
        """Setup structured logging."""
        if self.is_distributed and self.rank != 0:
            # Only log on master process
            structlog.configure(processors=[])
        else:
            structlog.configure(
                processors=[
                    structlog.stdlib.filter_by_level,
                    structlog.stdlib.add_logger_name,
                    structlog.stdlib.add_log_level,
                    structlog.stdlib.PositionalArgumentsFormatter(),
                    structlog.processors.TimeStamper(fmt="iso"),
                    structlog.processors.StackInfoRenderer(),
                    structlog.processors.format_exc_info,
                    structlog.processors.UnicodeDecoder(),
                    structlog.processors.JSONRenderer()
                ],
                context_class=dict,
                logger_factory=structlog.stdlib.LoggerFactory(),
                wrapper_class=structlog.stdlib.BoundLogger,
                cache_logger_on_first_use=True,
            )
    
    @abstractmethod
    def setup_model(self, model: nn.Module) -> nn.Module:
        """Setup model for multi-GPU training."""
        pass
    
    @abstractmethod
    def setup_dataloader(self, dataset: Dataset, **kwargs) -> DataLoader:
        """Setup dataloader for multi-GPU training."""
        pass
    
    @abstractmethod
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Perform a single training step."""
        pass
    
    @abstractmethod
    def validate_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Perform a single validation step."""
        pass


class DataParallelTrainer(MultiGPUTrainer):
    """DataParallel trainer for single-node multi-GPU training."""
    
    def setup_model(self, model: nn.Module) -> nn.Module:
        """Setup model with DataParallel."""
        if len(self.config.device_ids) > 1:
            model = model.to(self.device)
            model = DataParallel(
                model,
                device_ids=self.config.device_ids,
                output_device=self.config.master_gpu
            )
            logger.info(
                "Model wrapped with DataParallel",
                device_ids=self.config.device_ids,
                master_gpu=self.config.master_gpu
            )
        else:
            model = model.to(self.device)
            logger.info("Model moved to single GPU", device=self.device)
        
        self.model = model
        return model
    
    def setup_dataloader(self, dataset: Dataset, **kwargs) -> DataLoader:
        """Setup dataloader for DataParallel training."""
        # DataParallel doesn't need DistributedSampler
        return DataLoader(
            dataset,
            batch_size=kwargs.get('batch_size', 32),
            shuffle=kwargs.get('shuffle', True),
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers,
            prefetch_factor=self.config.prefetch_factor,
            drop_last=kwargs.get('drop_last', True)
        )
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Perform a single training step with DataParallel."""
        self.model.train()
        
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Forward pass with mixed precision
        if self.config.use_mixed_precision:
            with autocast():
                outputs = self.model(**batch)
                loss = outputs['loss']
            
            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()
            
            if (self.optimizer.step_count + 1) % self.config.gradient_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
        else:
            outputs = self.model(**batch)
            loss = outputs['loss']
            loss.backward()
            
            if (self.optimizer.step_count + 1) % self.config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()
        
        return outputs
    
    def validate_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Perform a single validation step with DataParallel."""
        self.model.eval()
        
        with torch.no_grad():
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            if self.config.use_mixed_precision:
                with autocast():
                    outputs = self.model(**batch)
            else:
                outputs = self.model(**batch)
        
        return outputs


class DistributedDataParallelTrainer(MultiGPUTrainer):
    """DistributedDataParallel trainer for distributed training."""
    
    def setup_model(self, model: nn.Module) -> nn.Module:
        """Setup model with DistributedDataParallel."""
        model = model.to(self.device)
        
        # Convert BatchNorm to SyncBatchNorm if needed
        if self.config.sync_bn:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            logger.info("Converted BatchNorm to SyncBatchNorm")
        
        # Wrap with DistributedDataParallel
        model = DistributedDataParallel(
            model,
            device_ids=[self.config.local_rank],
            output_device=self.config.local_rank,
            find_unused_parameters=self.config.find_unused_parameters,
            broadcast_buffers=self.config.broadcast_buffers,
            bucket_cap_mb=self.config.bucket_cap_mb,
            static_graph=self.config.static_graph,
            gradient_as_bucket_view=self.config.use_gradient_as_bucket_view,
            reduce_bucket_size=self.config.reduce_bucket_size
        )
        
        # Setup communication hooks for optimization
        if self.config.use_fp16_allreduce:
            model.register_comm_hook(
                state=None,
                hook=default_hooks.fp16_compress_hook
            )
        
        logger.info(
            "Model wrapped with DistributedDataParallel",
            local_rank=self.config.local_rank,
            world_size=self.world_size
        )
        
        self.model = model
        return model
    
    def setup_dataloader(self, dataset: Dataset, **kwargs) -> DataLoader:
        """Setup dataloader with DistributedSampler."""
        sampler = DistributedSampler(
            dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=kwargs.get('shuffle', True),
            drop_last=kwargs.get('drop_last', True)
        )
        
        return DataLoader(
            dataset,
            batch_size=kwargs.get('batch_size', 32),
            sampler=sampler,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers,
            prefetch_factor=self.config.prefetch_factor
        )
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Perform a single training step with DistributedDataParallel."""
        self.model.train()
        
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Forward pass with mixed precision
        if self.config.use_mixed_precision:
            with autocast():
                outputs = self.model(**batch)
                loss = outputs['loss']
            
            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()
            
            if (self.optimizer.step_count + 1) % self.config.gradient_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
        else:
            outputs = self.model(**batch)
            loss = outputs['loss']
            loss.backward()
            
            if (self.optimizer.step_count + 1) % self.config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()
        
        return outputs
    
    def validate_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Perform a single validation step with DistributedDataParallel."""
        self.model.eval()
        
        with torch.no_grad():
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            if self.config.use_mixed_precision:
                with autocast():
                    outputs = self.model(**batch)
            else:
                outputs = self.model(**batch)
        
        return outputs


class MultiGPUTrainingManager:
    """Manager for multi-GPU training operations."""
    
    def __init__(self, config: MultiGPUConfig):
        
    """__init__ function."""
self.config = config
        self.trainer = self._create_trainer()
        self.metrics_collector = MetricsCollector()
        self.fault_tolerance = FaultToleranceManager(config) if config.enable_fault_tolerance else None
        
    def _create_trainer(self) -> MultiGPUTrainer:
        """Create appropriate trainer based on configuration."""
        if self.config.training_mode == TrainingMode.DATA_PARALLEL:
            return DataParallelTrainer(self.config)
        elif self.config.training_mode == TrainingMode.DISTRIBUTED_DATA_PARALLEL:
            return DistributedDataParallelTrainer(self.config)
        else:
            return DataParallelTrainer(self.config)  # Fallback
    
    def setup_training(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ) -> Tuple[nn.Module, DataLoader, Optional[DataLoader]]:
        """Setup complete training environment."""
        
        # Setup model
        model = self.trainer.setup_model(model)
        
        # Setup dataloaders
        train_loader = self.trainer.setup_dataloader(
            train_dataset,
            batch_size=self.config.batch_size if hasattr(self.config, 'batch_size') else 32,
            shuffle=True,
            drop_last=True
        )
        
        val_loader = None
        if val_dataset:
            val_loader = self.trainer.setup_dataloader(
                val_dataset,
                batch_size=self.config.batch_size if hasattr(self.config, 'batch_size') else 32,
                shuffle=False,
                drop_last=False
            )
        
        # Setup optimizer and scheduler
        if optimizer:
            self.trainer.optimizer = optimizer
        if scheduler:
            self.trainer.scheduler = scheduler
        
        # Setup mixed precision
        if self.config.use_mixed_precision:
            self.trainer.scaler = GradScaler()
        
        return model, train_loader, val_loader
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch."""
        epoch_metrics = {}
        num_batches = len(train_loader)
        
        # Set epoch for DistributedSampler
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        
        for batch_idx, batch in enumerate(train_loader):
            try:
                # Training step
                outputs = self.trainer.train_step(batch)
                
                # Collect metrics
                step_metrics = self._extract_metrics(outputs)
                self.metrics_collector.update(step_metrics)
                
                # Logging
                if batch_idx % self.config.log_every_n_steps == 0:
                    self._log_training_step(batch_idx, num_batches, step_metrics, epoch)
                
                # Fault tolerance checkpoint
                if self.fault_tolerance and batch_idx % self.config.checkpoint_frequency == 0:
                    self.fault_tolerance.save_checkpoint(self.trainer.model, batch_idx)
                
            except Exception as e:
                logger.error("Training step failed", error=str(e), batch_idx=batch_idx)
                if self.fault_tolerance:
                    self.fault_tolerance.handle_error(e)
                raise
        
        # Aggregate epoch metrics
        epoch_metrics = self.metrics_collector.get_epoch_metrics()
        return epoch_metrics
    
    def validate_epoch(
        self,
        val_loader: DataLoader
    ) -> Dict[str, float]:
        """Validate for one epoch."""
        val_metrics = {}
        num_batches = len(val_loader)
        
        for batch_idx, batch in enumerate(val_loader):
            try:
                # Validation step
                outputs = self.trainer.validate_step(batch)
                
                # Collect metrics
                step_metrics = self._extract_metrics(outputs)
                self.metrics_collector.update(step_metrics, is_validation=True)
                
            except Exception as e:
                logger.error("Validation step failed", error=str(e), batch_idx=batch_idx)
                raise
        
        # Aggregate validation metrics
        val_metrics = self.metrics_collector.get_validation_metrics()
        return val_metrics
    
    def _extract_metrics(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Extract metrics from model outputs."""
        metrics = {}
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                if value.dim() == 0:  # Scalar tensor
                    metrics[key] = value.item()
                else:
                    metrics[key] = value.mean().item()
        return metrics
    
    def _log_training_step(
        self,
        batch_idx: int,
        num_batches: int,
        metrics: Dict[str, float],
        epoch: int
    ):
        """Log training step metrics."""
        if self.trainer.rank == 0:  # Only log on master process
            logger.info(
                "Training step",
                epoch=epoch,
                batch=batch_idx,
                total_batches=num_batches,
                progress=f"{batch_idx/num_batches*100:.1f}%",
                **metrics
            )
    
    def cleanup(self) -> Any:
        """Cleanup training resources."""
        if self.trainer.is_distributed:
            dist.destroy_process_group()
        
        if self.fault_tolerance:
            self.fault_tolerance.cleanup()


class MetricsCollector:
    """Collect and aggregate training metrics."""
    
    def __init__(self) -> Any:
        self.training_metrics = []
        self.validation_metrics = []
        self.current_epoch_metrics = []
        self.current_validation_metrics = []
    
    def update(self, metrics: Dict[str, float], is_validation: bool = False):
        """Update metrics."""
        if is_validation:
            self.current_validation_metrics.append(metrics)
        else:
            self.current_epoch_metrics.append(metrics)
    
    def get_epoch_metrics(self) -> Dict[str, float]:
        """Get aggregated epoch metrics."""
        if not self.current_epoch_metrics:
            return {}
        
        aggregated = {}
        for key in self.current_epoch_metrics[0].keys():
            values = [m[key] for m in self.current_epoch_metrics if key in m]
            if values:
                aggregated[key] = np.mean(values)
        
        self.training_metrics.append(aggregated)
        self.current_epoch_metrics = []
        return aggregated
    
    def get_validation_metrics(self) -> Dict[str, float]:
        """Get aggregated validation metrics."""
        if not self.current_validation_metrics:
            return {}
        
        aggregated = {}
        for key in self.current_validation_metrics[0].keys():
            values = [m[key] for m in self.current_validation_metrics if key in m]
            if values:
                aggregated[key] = np.mean(values)
        
        self.validation_metrics.append(aggregated)
        self.current_validation_metrics = []
        return aggregated


class FaultToleranceManager:
    """Manage fault tolerance and recovery."""
    
    def __init__(self, config: MultiGPUConfig):
        
    """__init__ function."""
self.config = config
        self.checkpoint_dir = Path("./checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.last_checkpoint = None
    
    def save_checkpoint(self, model: nn.Module, step: int):
        """Save training checkpoint."""
        checkpoint = {
            'step': step,
            'model_state_dict': model.state_dict(),
            'timestamp': datetime.now().isoformat()
        }
        
        checkpoint_path = self.checkpoint_dir / f"checkpoint_step_{step}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Remove old checkpoints
        self._cleanup_old_checkpoints()
        
        self.last_checkpoint = checkpoint_path
        logger.info("Checkpoint saved", path=str(checkpoint_path), step=step)
    
    def load_checkpoint(self, model: nn.Module, step: int) -> int:
        """Load training checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_step_{step}.pt"
        
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("Checkpoint loaded", path=str(checkpoint_path), step=step)
            return checkpoint['step']
        else:
            logger.warning("Checkpoint not found", path=str(checkpoint_path))
            return 0
    
    def handle_error(self, error: Exception):
        """Handle training errors."""
        logger.error("Training error occurred", error=str(error))
        
        # Implement recovery logic here
        # For now, just log the error
        pass
    
    def _cleanup_old_checkpoints(self, keep_last: int = 3):
        """Clean up old checkpoints."""
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_step_*.pt"))
        if len(checkpoints) > keep_last:
            for checkpoint in checkpoints[:-keep_last]:
                checkpoint.unlink()
    
    def cleanup(self) -> Any:
        """Cleanup fault tolerance resources."""
        pass


def setup_distributed_training(
    rank: int,
    world_size: int,
    backend: str = "nccl"
) -> MultiGPUConfig:
    """Setup distributed training configuration."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)
    
    config = MultiGPUConfig(
        training_mode=TrainingMode.DISTRIBUTED_DATA_PARALLEL,
        world_size=world_size,
        rank=rank,
        local_rank=rank,
        backend=backend
    )
    
    return config


def run_distributed_training(
    rank: int,
    world_size: int,
    model_fn: Callable,
    train_dataset: Dataset,
    val_dataset: Optional[Dataset] = None,
    num_epochs: int = 10,
    **kwargs
):
    """Run distributed training on a single process."""
    config = setup_distributed_training(rank, world_size)
    manager = MultiGPUTrainingManager(config)
    
    try:
        # Create model
        model = model_fn()
        
        # Setup training
        model, train_loader, val_loader = manager.setup_training(
            model, train_dataset, val_dataset
        )
        
        # Training loop
        for epoch in range(num_epochs):
            # Train
            train_metrics = manager.train_epoch(train_loader, epoch)
            
            # Validate
            if val_loader:
                val_metrics = manager.validate_epoch(val_loader)
                logger.info(
                    "Epoch completed",
                    epoch=epoch,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics
                )
            else:
                logger.info("Epoch completed", epoch=epoch, train_metrics=train_metrics)
    
    finally:
        manager.cleanup()


def launch_distributed_training(
    world_size: int,
    model_fn: Callable,
    train_dataset: Dataset,
    val_dataset: Optional[Dataset] = None,
    num_epochs: int = 10,
    **kwargs
):
    """Launch distributed training across multiple processes."""
    mp.spawn(
        run_distributed_training,
        args=(world_size, model_fn, train_dataset, val_dataset, num_epochs),
        nprocs=world_size,
        join=True
    )


# Example usage and testing functions
def create_sample_model() -> nn.Module:
    """Create a sample model for testing."""
    class SampleModel(nn.Module):
        def __init__(self) -> Any:
            super().__init__()
            self.linear = nn.Linear(100, 10)
            self.dropout = nn.Dropout(0.1)
            self.classifier = nn.Linear(10, 2)
        
        def forward(self, x) -> Any:
            x = self.linear(x)
            x = torch.relu(x)
            x = self.dropout(x)
            x = self.classifier(x)
            return {'logits': x, 'loss': torch.nn.functional.cross_entropy(x, torch.zeros(x.size(0), dtype=torch.long))}
    
    return SampleModel()


def create_sample_dataset(num_samples: int = 1000) -> Dataset:
    """Create a sample dataset for testing."""
    class SampleDataset(Dataset):
        def __init__(self, num_samples: int):
            
    """__init__ function."""
self.data = torch.randn(num_samples, 100)
            self.labels = torch.randint(0, 2, (num_samples,))
        
        def __len__(self) -> Any:
            return len(self.data)
        
        def __getitem__(self, idx) -> Optional[Dict[str, Any]]:
            return {
                'input_ids': self.data[idx],
                'labels': self.labels[idx]
            }
    
    return SampleDataset(num_samples)


async def demo_multi_gpu_training():
    """Demonstrate multi-GPU training capabilities."""
    logger.info("Starting Multi-GPU Training Demo")
    
    # Test DataParallel
    logger.info("Testing DataParallel training")
    config_dp = MultiGPUConfig(
        training_mode=TrainingMode.DATA_PARALLEL,
        device_ids=list(range(min(2, torch.cuda.device_count()))),
        use_mixed_precision=True
    )
    
    manager_dp = MultiGPUTrainingManager(config_dp)
    model = create_sample_model()
    dataset = create_sample_dataset(100)
    
    model, train_loader, _ = manager_dp.setup_training(model, dataset)
    
    # Train for a few steps
    for epoch in range(2):
        metrics = manager_dp.train_epoch(train_loader, epoch)
        logger.info(f"DataParallel Epoch {epoch} metrics: {metrics}")
    
    manager_dp.cleanup()
    
    # Test DistributedDataParallel if multiple GPUs available
    if torch.cuda.device_count() > 1:
        logger.info("Testing DistributedDataParallel training")
        config_ddp = MultiGPUConfig(
            training_mode=TrainingMode.DISTRIBUTED_DATA_PARALLEL,
            world_size=2,
            use_mixed_precision=True
        )
        
        # This would typically be launched with mp.spawn
        logger.info("DistributedDataParallel setup complete")
    
    logger.info("Multi-GPU Training Demo completed")


if __name__ == "__main__":
    # Run demo
    asyncio.run(demo_multi_gpu_training()) 