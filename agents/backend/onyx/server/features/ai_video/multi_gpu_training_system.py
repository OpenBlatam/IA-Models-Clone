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
import torch.nn.parallel
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
import os
import time
import logging
import json
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from dataclasses import dataclass, field
from contextlib import contextmanager
import numpy as np
import psutil
import gc
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
Multi-GPU Training System

Comprehensive multi-GPU training implementation using PyTorch's DataParallel
and DistributedDataParallel for AI video processing with advanced features.
"""


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MultiGPUConfig:
    """Configuration for multi-GPU training."""
    # GPU configuration
    num_gpus: int = torch.cuda.device_count()
    gpu_ids: List[int] = field(default_factory=list)
    master_gpu: int = 0
    
    # Training configuration
    batch_size_per_gpu: int = 32
    effective_batch_size: int = 128
    num_workers_per_gpu: int = 4
    pin_memory: bool = True
    
    # Distributed training configuration
    use_distributed: bool = False
    backend: str = "nccl"  # "nccl" for GPU, "gloo" for CPU
    init_method: str = "env://"
    world_size: int = 1
    rank: int = 0
    
    # Synchronization configuration
    sync_bn: bool = True
    find_unused_parameters: bool = False
    gradient_as_bucket_view: bool = True
    
    # Performance configuration
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    gradient_clipping: float = 1.0
    
    # Communication configuration
    broadcast_buffers: bool = True
    bucket_cap_mb: int = 25
    
    def __post_init__(self) -> Any:
        """Post-initialization setup."""
        if not self.gpu_ids:
            self.gpu_ids = list(range(self.num_gpus))
        
        if self.use_distributed:
            self.effective_batch_size = self.batch_size_per_gpu * self.num_gpus * self.gradient_accumulation_steps
        else:
            self.effective_batch_size = self.batch_size_per_gpu * self.num_gpus

class MultiGPUTrainer:
    """Multi-GPU training orchestrator with DataParallel and DistributedDataParallel support."""
    
    def __init__(self, config: MultiGPUConfig):
        
    """__init__ function."""
self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_distributed = config.use_distributed
        self.is_master = config.rank == 0 if self.is_distributed else True
        
        # Initialize distributed training if needed
        if self.is_distributed:
            self._init_distributed()
        
        # GPU setup
        self._setup_gpus()
        
        logger.info(f"Multi-GPU trainer initialized with {self.config.num_gpus} GPUs")
        if self.is_distributed:
            logger.info(f"Distributed training enabled (rank: {self.config.rank}, world_size: {self.config.world_size})")
    
    def _init_distributed(self) -> Any:
        """Initialize distributed training."""
        if not dist.is_initialized():
            dist.init_process_group(
                backend=self.config.backend,
                init_method=self.config.init_method,
                world_size=self.config.world_size,
                rank=self.config.rank
            )
            
            # Set device for current process
            torch.cuda.set_device(self.config.gpu_ids[self.config.rank])
            
            logger.info(f"Distributed training initialized (rank: {self.config.rank})")
    
    def _setup_gpus(self) -> Any:
        """Setup GPU configuration."""
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            self.config.num_gpus = 0
            self.config.gpu_ids = []
            return
        
        # Verify GPU availability
        available_gpus = torch.cuda.device_count()
        if self.config.num_gpus > available_gpus:
            logger.warning(f"Requested {self.config.num_gpus} GPUs, but only {available_gpus} available")
            self.config.num_gpus = available_gpus
            self.config.gpu_ids = list(range(available_gpus))
        
        # Set master GPU
        if self.config.master_gpu >= self.config.num_gpus:
            self.config.master_gpu = 0
        
        logger.info(f"GPU setup: {self.config.num_gpus} GPUs, master: {self.config.master_gpu}")
        for i in range(self.config.num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            logger.info(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    
    def wrap_model(self, model: nn.Module) -> nn.Module:
        """Wrap model for multi-GPU training."""
        if self.config.num_gpus == 0:
            return model.to(self.device)
        
        if self.is_distributed:
            # DistributedDataParallel
            model = model.to(self.device)
            
            # Synchronize batch normalization if requested
            if self.config.sync_bn and self.config.num_gpus > 1:
                model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
                logger.info("Synchronized batch normalization enabled")
            
            # Wrap with DistributedDataParallel
            model = DistributedDataParallel(
                model,
                device_ids=[self.config.gpu_ids[self.config.rank]],
                output_device=self.config.gpu_ids[self.config.rank],
                find_unused_parameters=self.config.find_unused_parameters,
                gradient_as_bucket_view=self.config.gradient_as_bucket_view,
                broadcast_buffers=self.config.broadcast_buffers,
                bucket_cap_mb=self.config.bucket_cap_mb
            )
            
            logger.info("Model wrapped with DistributedDataParallel")
            
        else:
            # DataParallel
            model = model.to(self.device)
            
            # Synchronize batch normalization if requested
            if self.config.sync_bn and self.config.num_gpus > 1:
                model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
                logger.info("Synchronized batch normalization enabled")
            
            # Wrap with DataParallel
            model = DataParallel(
                model,
                device_ids=self.config.gpu_ids,
                output_device=self.config.master_gpu
            )
            
            logger.info("Model wrapped with DataParallel")
        
        return model
    
    def create_dataloader(self, dataset, shuffle: bool = True) -> DataLoader:
        """Create DataLoader for multi-GPU training."""
        if self.is_distributed:
            # Use DistributedSampler for distributed training
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.config.world_size,
                rank=self.config.rank,
                shuffle=shuffle
            )
            
            dataloader = DataLoader(
                dataset,
                batch_size=self.config.batch_size_per_gpu,
                sampler=sampler,
                num_workers=self.config.num_workers_per_gpu,
                pin_memory=self.config.pin_memory,
                drop_last=True
            )
            
            logger.info(f"Created distributed DataLoader (rank: {self.config.rank})")
            
        else:
            # Standard DataLoader for DataParallel
            dataloader = DataLoader(
                dataset,
                batch_size=self.config.batch_size_per_gpu * self.config.num_gpus,
                shuffle=shuffle,
                num_workers=self.config.num_workers_per_gpu * self.config.num_gpus,
                pin_memory=self.config.pin_memory,
                drop_last=True
            )
            
            logger.info("Created DataParallel DataLoader")
        
        return dataloader
    
    def train_epoch(self, model: nn.Module, dataloader: DataLoader, 
                   optimizer: torch.optim.Optimizer, criterion: nn.Module,
                   epoch: int = 1, scaler: Optional[torch.cuda.amp.GradScaler] = None) -> Dict[str, float]:
        """Train for one epoch with multi-GPU support."""
        model.train()
        
        # Set epoch for distributed sampler
        if self.is_distributed and hasattr(dataloader.sampler, 'set_epoch'):
            dataloader.sampler.set_epoch(epoch)
        
        total_loss = 0.0
        num_batches = len(dataloader)
        batch_start_time = time.time()
        
        # Gradient accumulation setup
        accumulation_steps = self.config.gradient_accumulation_steps
        optimizer.zero_grad()
        
        for batch_idx, (data, targets) in enumerate(dataloader):
            data_start_time = time.time()
            
            try:
                # Move data to device
                data = data.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                # Forward pass with mixed precision
                if scaler and self.config.mixed_precision:
                    with torch.cuda.amp.autocast():
                        outputs = model(data)
                        loss = criterion(outputs, targets)
                    
                    # Scale loss for gradient accumulation
                    loss = loss / accumulation_steps
                    scaler.scale(loss).backward()
                    
                else:
                    outputs = model(data)
                    loss = criterion(outputs, targets)
                    
                    # Scale loss for gradient accumulation
                    loss = loss / accumulation_steps
                    loss.backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % accumulation_steps == 0:
                    # Gradient clipping
                    if self.config.gradient_clipping > 0:
                        if scaler:
                            scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clipping)
                    
                    # Optimizer step
                    if scaler:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    
                    optimizer.zero_grad()
                
                # Calculate metrics
                batch_loss = loss.item() * accumulation_steps
                total_loss += batch_loss
                
                # Calculate accuracy
                predicted_labels = torch.argmax(outputs, dim=1)
                accuracy = (predicted_labels == targets).float().mean().item()
                
                # Timing
                data_time = time.time() - data_start_time
                batch_time = time.time() - batch_start_time
                
                # Log progress (only on master process)
                if self.is_master and batch_idx % 50 == 0:
                    logger.info(f"Epoch {epoch}, Batch {batch_idx}/{num_batches}, "
                              f"Loss: {batch_loss:.4f}, Accuracy: {accuracy:.4f}, "
                              f"Data Time: {data_time:.3f}s, Batch Time: {batch_time:.3f}s")
                
                batch_start_time = time.time()
                
            except Exception as e:
                logger.error(f"Error in training batch {batch_idx}: {e}")
                if self.is_distributed:
                    # Synchronize processes on error
                    dist.barrier()
                continue
        
        # Final optimizer step if needed
        if batch_idx % accumulation_steps != 0:
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
        
        # Synchronize loss across processes
        if self.is_distributed:
            avg_loss = torch.tensor(total_loss / num_batches, device=self.device)
            dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
            avg_loss = avg_loss.item() / self.config.world_size
        else:
            avg_loss = total_loss / num_batches
        
        return {'loss': avg_loss}
    
    def validate(self, model: nn.Module, dataloader: DataLoader, 
                criterion: nn.Module, epoch: int = 1) -> Dict[str, float]:
        """Validate model with multi-GPU support."""
        model.eval()
        
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        try:
            with torch.no_grad():
                for batch_idx, (data, targets) in enumerate(dataloader):
                    # Move data to device
                    data = data.to(self.device, non_blocking=True)
                    targets = targets.to(self.device, non_blocking=True)
                    
                    # Forward pass
                    outputs = model(data)
                    loss = criterion(outputs, targets)
                    
                    total_loss += loss.item()
                    
                    # Calculate accuracy
                    predicted_labels = torch.argmax(outputs, dim=1)
                    correct_predictions += (predicted_labels == targets).sum().item()
                    total_predictions += targets.size(0)
            
            # Synchronize metrics across processes
            if self.is_distributed:
                # Gather loss
                avg_loss = torch.tensor(total_loss / len(dataloader), device=self.device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / self.config.world_size
                
                # Gather accuracy
                correct_tensor = torch.tensor(correct_predictions, device=self.device)
                total_tensor = torch.tensor(total_predictions, device=self.device)
                dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
                accuracy = correct_tensor.item() / total_tensor.item()
                
            else:
                avg_loss = total_loss / len(dataloader)
                accuracy = correct_predictions / total_predictions
            
            return {'loss': avg_loss, 'accuracy': accuracy}
            
        except Exception as e:
            logger.error(f"Error in validation: {e}")
            return {'loss': float('inf'), 'accuracy': 0.0}
    
    def get_model_state_dict(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """Get model state dict (handles DDP wrapper)."""
        if isinstance(model, DistributedDataParallel):
            return model.module.state_dict()
        elif isinstance(model, DataParallel):
            return model.module.state_dict()
        else:
            return model.state_dict()
    
    def load_model_state_dict(self, model: nn.Module, state_dict: Dict[str, torch.Tensor]):
        """Load model state dict (handles DDP wrapper)."""
        if isinstance(model, DistributedDataParallel):
            model.module.load_state_dict(state_dict)
        elif isinstance(model, DataParallel):
            model.module.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)
    
    def save_checkpoint(self, model: nn.Module, optimizer: torch.optim.Optimizer, 
                       epoch: int, loss: float, filename: str):
        """Save checkpoint (only on master process)."""
        if not self.is_master:
            return
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.get_model_state_dict(model),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'config': self.config.__dict__
        }
        
        torch.save(checkpoint, filename)
        logger.info(f"Checkpoint saved: {filename}")
    
    def load_checkpoint(self, model: nn.Module, optimizer: torch.optim.Optimizer, 
                       filename: str) -> Dict[str, Any]:
        """Load checkpoint."""
        if not os.path.exists(filename):
            logger.warning(f"Checkpoint not found: {filename}")
            return {}
        
        checkpoint = torch.load(filename, map_location=self.device)
        
        self.load_model_state_dict(model, checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        logger.info(f"Checkpoint loaded: {filename}")
        return checkpoint
    
    def cleanup(self) -> Any:
        """Cleanup distributed training."""
        if self.is_distributed and dist.is_initialized():
            dist.destroy_process_group()
            logger.info("Distributed process group destroyed")

class DistributedTrainingLauncher:
    """Launcher for distributed training across multiple processes."""
    
    def __init__(self, world_size: int = None):
        
    """__init__ function."""
self.world_size = world_size or torch.cuda.device_count()
    
    def launch_distributed_training(self, train_func: Callable, *args, **kwargs):
        """Launch distributed training."""
        if self.world_size <= 1:
            logger.warning("World size <= 1, running single process")
            train_func(0, 1, *args, **kwargs)
            return
        
        # Set environment variables for distributed training
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        
        # Launch processes
        mp.spawn(
            train_func,
            args=(self.world_size, *args),
            nprocs=self.world_size,
            join=True
        )

def setup_distributed_training(rank: int, world_size: int) -> MultiGPUConfig:
    """Setup distributed training configuration."""
    config = MultiGPUConfig(
        use_distributed=True,
        world_size=world_size,
        rank=rank,
        num_gpus=1,  # One GPU per process
        gpu_ids=[rank],
        master_gpu=rank,
        batch_size_per_gpu=32,
        num_workers_per_gpu=4
    )
    
    return config

def example_distributed_training(rank: int, world_size: int, model_class, dataset_class, 
                               num_epochs: int = 10):
    """Example distributed training function."""
    # Setup distributed training
    config = setup_distributed_training(rank, world_size)
    trainer = MultiGPUTrainer(config)
    
    try:
        # Create model and wrap for distributed training
        model = model_class()
        model = trainer.wrap_model(model)
        
        # Create dataset and dataloader
        dataset = dataset_class()
        dataloader = trainer.create_dataloader(dataset)
        
        # Training components
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())
        scaler = torch.cuda.amp.GradScaler() if config.mixed_precision else None
        
        # Training loop
        for epoch in range(num_epochs):
            # Train
            train_results = trainer.train_epoch(
                model, dataloader, optimizer, criterion, epoch + 1, scaler
            )
            
            # Validate
            val_results = trainer.validate(model, dataloader, criterion, epoch + 1)
            
            # Log results (only on master)
            if trainer.is_master:
                logger.info(f"Epoch {epoch + 1}: "
                          f"Train Loss: {train_results['loss']:.4f}, "
                          f"Val Loss: {val_results['loss']:.4f}, "
                          f"Val Accuracy: {val_results['accuracy']:.4f}")
        
        # Save final model (only on master)
        if trainer.is_master:
            trainer.save_checkpoint(
                model, optimizer, num_epochs, train_results['loss'], 
                f"distributed_model_rank_{rank}.pth"
            )
    
    finally:
        trainer.cleanup()

def example_dataparallel_training(model_class, dataset_class, num_epochs: int = 10):
    """Example DataParallel training function."""
    # Setup DataParallel training
    config = MultiGPUConfig(
        use_distributed=False,
        num_gpus=torch.cuda.device_count(),
        batch_size_per_gpu=32,
        num_workers_per_gpu=4
    )
    trainer = MultiGPUTrainer(config)
    
    # Create model and wrap for DataParallel
    model = model_class()
    model = trainer.wrap_model(model)
    
    # Create dataset and dataloader
    dataset = dataset_class()
    dataloader = trainer.create_dataloader(dataset)
    
    # Training components
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    scaler = torch.cuda.amp.GradScaler() if config.mixed_precision else None
    
    # Training loop
    for epoch in range(num_epochs):
        # Train
        train_results = trainer.train_epoch(
            model, dataloader, optimizer, criterion, epoch + 1, scaler
        )
        
        # Validate
        val_results = trainer.validate(model, dataloader, criterion, epoch + 1)
        
        logger.info(f"Epoch {epoch + 1}: "
                   f"Train Loss: {train_results['loss']:.4f}, "
                   f"Val Loss: {val_results['loss']:.4f}, "
                   f"Val Accuracy: {val_results['accuracy']:.4f}")
    
    # Save final model
    trainer.save_checkpoint(
        model, optimizer, num_epochs, train_results['loss'], 
        "dataparallel_model.pth"
    )

# Example usage
def example_usage():
    """Example of using multi-GPU training system."""
    
    # Example model and dataset classes
    class SimpleModel(nn.Module):
        def __init__(self) -> Any:
            super().__init__()
            self.fc = nn.Linear(784, 10)
        
        def forward(self, x) -> Any:
            return self.fc(x.view(x.size(0), -1))
    
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, num_samples=1000) -> Any:
            self.data = torch.randn(num_samples, 784)
            self.targets = torch.randint(0, 10, (num_samples,))
        
        def __len__(self) -> Any:
            return len(self.data)
        
        def __getitem__(self, idx) -> Optional[Dict[str, Any]]:
            return self.data[idx], self.targets[idx]
    
    # DataParallel training
    logger.info("Starting DataParallel training...")
    example_dataparallel_training(SimpleModel, SimpleDataset, num_epochs=2)
    
    # Distributed training
    if torch.cuda.device_count() > 1:
        logger.info("Starting distributed training...")
        launcher = DistributedTrainingLauncher(world_size=2)
        launcher.launch_distributed_training(
            example_distributed_training, SimpleModel, SimpleDataset, num_epochs=2
        )

match __name__:
    case "__main__":
    example_usage() 