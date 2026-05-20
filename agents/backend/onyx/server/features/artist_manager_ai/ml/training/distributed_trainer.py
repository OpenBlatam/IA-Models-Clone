"""
Distributed Trainer
===================

Distributed training support with DataParallel and DistributedDataParallel.
"""

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DataParallel, DistributedDataParallel
from typing import Dict, Any, Optional
import logging
import os

from .trainer import Trainer

logger = logging.getLogger(__name__)


class DistributedTrainer(Trainer):
    """
    Distributed trainer supporting:
    - DataParallel (single node, multiple GPUs)
    - DistributedDataParallel (multi-node, multi-GPU)
    - Gradient accumulation
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        config: Dict[str, Any],
        use_ddp: bool = False,
        world_size: Optional[int] = None,
        rank: Optional[int] = None
    ):
        """
        Initialize distributed trainer.
        
        Args:
            model: PyTorch model
            train_loader: Training dataloader
            val_loader: Validation dataloader
            criterion: Loss function
            optimizer: Optimizer
            device: Device
            config: Training configuration
            use_ddp: Whether to use DistributedDataParallel
            world_size: World size for DDP
            rank: Rank for DDP
        """
        # Initialize base trainer
        super().__init__(model, train_loader, val_loader, criterion, optimizer, device, config)
        
        self.use_ddp = use_ddp
        self.world_size = world_size
        self.rank = rank
        
        # Gradient accumulation
        self.gradient_accumulation_steps = config.get("gradient_accumulation_steps", 1)
        
        # Setup distributed training
        if use_ddp:
            self._setup_ddp()
        elif torch.cuda.device_count() > 1:
            self._setup_dataparallel()
    
    def _setup_ddp(self):
        """Setup DistributedDataParallel."""
        if not dist.is_available():
            logger.warning("Distributed training not available. Falling back to DataParallel.")
            self._setup_dataparallel()
            return
        
        if self.rank is None:
            self.rank = int(os.environ.get("LOCAL_RANK", 0))
        
        if self.world_size is None:
            self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        
        # Initialize process group
        dist.init_process_group(
            backend="nccl",
            rank=self.rank,
            world_size=self.world_size
        )
        
        # Wrap model
        self.model = DistributedDataParallel(
            self.model,
            device_ids=[self.rank],
            output_device=self.rank,
            find_unused_parameters=False
        )
        
        logger.info(f"DDP initialized: rank={self.rank}, world_size={self.world_size}")
    
    def _setup_dataparallel(self):
        """Setup DataParallel."""
        if torch.cuda.device_count() > 1:
            self.model = DataParallel(self.model)
            logger.info(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch with gradient accumulation."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        from tqdm import tqdm
        pbar = tqdm(self.train_loader, desc="Training")
        
        self.optimizer.zero_grad()
        
        for batch_idx, (features, targets) in enumerate(pbar):
            features = features.to(self.device)
            targets = targets.to(self.device)
            
            if targets.dim() == 1:
                targets = targets.unsqueeze(1)
            
            # Forward pass
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(features)
                    loss = self.criterion(outputs, targets)
                    loss = loss / self.gradient_accumulation_steps  # Scale loss
                
                # Backward pass
                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(features)
                loss = self.criterion(outputs, targets)
                loss = loss / self.gradient_accumulation_steps  # Scale loss
                
                # Backward pass
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.config.get("grad_clip", 0) > 0:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                        if isinstance(self.model, (DataParallel, DistributedDataParallel)):
                            torch.nn.utils.clip_grad_norm_(
                                self.model.module.parameters(),
                                self.config["grad_clip"]
                            )
                        else:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                self.config["grad_clip"]
                            )
                    else:
                        if isinstance(self.model, (DataParallel, DistributedDataParallel)):
                            torch.nn.utils.clip_grad_norm_(
                                self.model.module.parameters(),
                                self.config["grad_clip"]
                            )
                        else:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                self.config["grad_clip"]
                            )
                
                # Optimizer step
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
            
            # Update metrics
            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1
            
            pbar.set_postfix({"loss": loss.item() * self.gradient_accumulation_steps})
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return {"loss": avg_loss}
    
    def cleanup(self):
        """Cleanup distributed training."""
        if self.use_ddp and dist.is_initialized():
            dist.destroy_process_group()
            logger.info("DDP process group destroyed")


def setup_ddp(rank: int, world_size: int):
    """Setup DDP process group."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup_ddp():
    """Cleanup DDP process group."""
    if dist.is_initialized():
        dist.destroy_process_group()




