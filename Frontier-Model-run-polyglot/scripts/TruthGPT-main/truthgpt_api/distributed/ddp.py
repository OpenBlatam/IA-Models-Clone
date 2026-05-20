"""
Distributed Data Parallel for TruthGPT API
=========================================

TensorFlow-like distributed training implementation.
"""

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Any, Optional, Dict, List, Tuple
import os


class DistributedDataParallel:
    """
    Distributed Data Parallel training.
    
    Similar to tf.distribute.MirroredStrategy, this class
    implements distributed training using PyTorch DDP.
    """
    
    def __init__(self, 
                 backend: str = 'nccl',
                 init_method: str = 'env://',
                 world_size: Optional[int] = None,
                 rank: Optional[int] = None,
                 name: Optional[str] = None):
        """
        Initialize DistributedDataParallel.
        
        Args:
            backend: Communication backend
            init_method: Initialization method
            world_size: Number of processes
            rank: Process rank
            name: Optional name for the distributed trainer
        """
        self.backend = backend
        self.init_method = init_method
        self.world_size = world_size or int(os.environ.get('WORLD_SIZE', 1))
        self.rank = rank or int(os.environ.get('RANK', 0))
        self.name = name or "ddp"
        
        self.is_initialized = False
        self.model = None
        self.optimizer = None
        self.device = None
    
    def initialize(self):
        """Initialize distributed training."""
        if self.is_initialized:
            return
        
        print(f"🚀 Initializing distributed training...")
        print(f"   Backend: {self.backend}")
        print(f"   World size: {self.world_size}")
        print(f"   Rank: {self.rank}")
        
        # Initialize process group
        dist.init_process_group(
            backend=self.backend,
            init_method=self.init_method,
            world_size=self.world_size,
            rank=self.rank
        )
        
        # Set device
        self.device = torch.device(f'cuda:{self.rank}' if torch.cuda.is_available() else 'cpu')
        torch.cuda.set_device(self.device)
        
        self.is_initialized = True
        print(f"✅ Distributed training initialized!")
    
    def wrap_model(self, model: Any, device_ids: Optional[List[int]] = None) -> Any:
        """
        Wrap model with DDP.
        
        Args:
            model: Model to wrap
            device_ids: Device IDs for DDP
            
        Returns:
            DDP-wrapped model
        """
        if not self.is_initialized:
            self.initialize()
        
        print(f"🔧 Wrapping model with DDP...")
        
        # Move model to device
        model = model.to(self.device)
        
        # Wrap with DDP
        if device_ids is None:
            device_ids = [self.rank]
        
        ddp_model = DDP(
            model,
            device_ids=device_ids,
            output_device=self.device,
            find_unused_parameters=True
        )
        
        self.model = ddp_model
        print(f"✅ Model wrapped with DDP!")
        
        return ddp_model
    
    def train_step(self, 
                   model: Any,
                   optimizer: Any,
                   loss_fn: Any,
                   batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        """
        Perform one training step.
        
        Args:
            model: DDP-wrapped model
            optimizer: Optimizer
            loss_fn: Loss function
            batch: Training batch
            
        Returns:
            Training metrics
        """
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        
        # Forward pass
        outputs = model(x)
        loss = loss_fn(outputs, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        with torch.no_grad():
            predictions = torch.argmax(outputs, dim=1)
            accuracy = (predictions == y).float().mean()
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy.item()
        }
    
    def validate(self, 
                 model: Any,
                 loss_fn: Any,
                 val_loader: Any) -> Dict[str, float]:
        """
        Validate model.
        
        Args:
            model: DDP-wrapped model
            loss_fn: Loss function
            val_loader: Validation data loader
            
        Returns:
            Validation metrics
        """
        model.eval()
        
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)
                
                outputs = model(x)
                loss = loss_fn(outputs, y)
                
                predictions = torch.argmax(outputs, dim=1)
                accuracy = (predictions == y).float().mean()
                
                total_loss += loss.item()
                total_accuracy += accuracy.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        
        return {
            'val_loss': avg_loss,
            'val_accuracy': avg_accuracy
        }
    
    def train_epoch(self, 
                   model: Any,
                   optimizer: Any,
                   loss_fn: Any,
                   train_loader: Any,
                   val_loader: Optional[Any] = None) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            model: DDP-wrapped model
            optimizer: Optimizer
            loss_fn: Loss function
            train_loader: Training data loader
            val_loader: Validation data loader
            
        Returns:
            Training metrics
        """
        model.train()
        
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        num_batches = 0
        
        for batch in train_loader:
            metrics = self.train_step(model, optimizer, loss_fn, batch)
            
            epoch_loss += metrics['loss']
            epoch_accuracy += metrics['accuracy']
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        avg_accuracy = epoch_accuracy / num_batches
        
        results = {
            'train_loss': avg_loss,
            'train_accuracy': avg_accuracy
        }
        
        # Validation
        if val_loader is not None:
            val_metrics = self.validate(model, loss_fn, val_loader)
            results.update(val_metrics)
        
        return results
    
    def cleanup(self):
        """Cleanup distributed training."""
        if self.is_initialized:
            dist.destroy_process_group()
            self.is_initialized = False
            print(f"✅ Distributed training cleaned up!")
    
    def get_world_size(self) -> int:
        """Get world size."""
        return self.world_size
    
    def get_rank(self) -> int:
        """Get rank."""
        return self.rank
    
    def is_master(self) -> bool:
        """Check if this is the master process."""
        return self.rank == 0
    
    def barrier(self):
        """Synchronize all processes."""
        if self.is_initialized:
            dist.barrier()
    
    def get_config(self) -> Dict[str, Any]:
        """Get distributed training configuration."""
        return {
            'name': self.name,
            'backend': self.backend,
            'world_size': self.world_size,
            'rank': self.rank,
            'is_initialized': self.is_initialized
        }
    
    def __repr__(self):
        return f"DistributedDataParallel(backend={self.backend}, world_size={self.world_size}, rank={self.rank})"









