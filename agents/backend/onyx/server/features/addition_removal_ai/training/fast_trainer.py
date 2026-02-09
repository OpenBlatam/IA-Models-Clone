"""
Fast Training Pipeline with Optimizations
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from typing import Optional, Dict
import logging
from tqdm import tqdm
import os

from .trainer import ModelTrainer

logger = logging.getLogger(__name__)


class FastModelTrainer(ModelTrainer):
    """Fast trainer with additional optimizations"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: Optional[torch.device] = None,
        use_mixed_precision: bool = True,
        use_torch_compile: bool = True,
        gradient_accumulation_steps: int = 1,
        log_dir: str = "./logs",
        use_wandb: bool = False,
        wandb_project: str = "addition-removal-ai"
    ):
        """
        Initialize fast trainer
        
        Args:
            use_torch_compile: Use torch.compile for faster training (PyTorch 2.0+)
        """
        super().__init__(
            model, train_loader, val_loader, device,
            use_mixed_precision, gradient_accumulation_steps,
            log_dir, use_wandb, wandb_project
        )
        
        # Compile model for faster training (PyTorch 2.0+)
        if use_torch_compile and hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                logger.info("Model compiled with torch.compile")
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}")
        
        # Optimize DataLoader
        if train_loader.num_workers == 0:
            train_loader.num_workers = min(4, os.cpu_count() or 1)
            logger.info(f"Optimized DataLoader with {train_loader.num_workers} workers")
    
    def train_epoch(
        self,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        epoch: int,
        clip_grad_norm: Optional[float] = None
    ) -> Dict[str, float]:
        """Fast training epoch with optimizations"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Pre-allocate tensors if possible
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(pbar):
            # Fast batch transfer
            if isinstance(batch, (list, tuple)):
                inputs = batch[0].to(self.device, non_blocking=True)
                targets = batch[1].to(self.device, non_blocking=True)
            else:
                inputs = batch.to(self.device, non_blocking=True)
                targets = None
            
            # Forward pass with optimizations
            if self.use_mixed_precision:
                with autocast():
                    if targets is not None:
                        outputs = self.model(inputs)
                        loss = criterion(outputs, targets)
                    else:
                        outputs = self.model(inputs)
                        loss = criterion(outputs, inputs)
                    
                    loss = loss / self.gradient_accumulation_steps
                
                self.scaler.scale(loss).backward()
            else:
                if targets is not None:
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)
                else:
                    outputs = self.model(inputs)
                    loss = criterion(outputs, inputs)
                
                loss = loss / self.gradient_accumulation_steps
                loss.backward()
            
            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1
            
            # Update weights
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.use_mixed_precision:
                    if clip_grad_norm is not None:
                        self.scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), clip_grad_norm
                        )
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    if clip_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), clip_grad_norm
                        )
                    optimizer.step()
                
                optimizer.zero_grad(set_to_none=True)  # Faster zero_grad
            
            pbar.set_postfix({"loss": loss.item() * self.gradient_accumulation_steps})
        
        avg_loss = total_loss / num_batches
        return {"loss": avg_loss}


def create_fast_trainer(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    use_torch_compile: bool = True,
    **kwargs
) -> FastModelTrainer:
    """Factory function to create fast trainer"""
    return FastModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        use_torch_compile=use_torch_compile,
        **kwargs
    )

