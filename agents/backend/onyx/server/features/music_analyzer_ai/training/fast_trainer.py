"""
Fast Trainer with aggressive optimizations
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging

logger = logging.getLogger(__name__)

try:
    from .speed_optimizer import SpeedOptimizer, FastTraining, BatchOptimizer
except ImportError:
    # Fallback if not available
    SpeedOptimizer = None
    FastTraining = None
    BatchOptimizer = None


class FastMusicTrainer:
    """
    Fast trainer with all optimizations enabled
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        learning_rate: float = 1e-4,
        compile_model: bool = True,
        enable_tf32: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True
    ):
        self.device = device
        self.model = model.to(device)
        
        # Speed optimizations
        if compile_model and SpeedOptimizer and hasattr(torch, 'compile'):
            self.model = SpeedOptimizer.compile_model(self.model, mode="reduce-overhead")
        
        if enable_tf32 and device == "cuda":
            SpeedOptimizer.enable_tf32(self.model) if SpeedOptimizer else None
            SpeedOptimizer.enable_benchmark_mode() if SpeedOptimizer else None
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None
        
        # DataLoader settings
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        logger.info(f"Fast trainer initialized on {device} with optimizations")
    
    def train_epoch_fast(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Fast training epoch with all optimizations
        """
        self.model.train()
        total_loss = 0.0
        
        # Optimize dataloader if not already optimized
        if SpeedOptimizer and dataloader.num_workers == 0:
            dataloader = SpeedOptimizer.optimize_dataloader(
                dataloader,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory
            )
        
        for batch_idx, batch in enumerate(dataloader):
            # Fast batch preparation
            if BatchOptimizer:
                batch = BatchOptimizer.prepare_batch(batch, self.device, non_blocking=True)
            else:
                batch = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}
            
            self.optimizer.zero_grad()
            
            # Forward with mixed precision
            if self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model(batch["features"])
                    loss = self._compute_loss(outputs, batch)
                
                # Backward
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(batch["features"])
                loss = self._compute_loss(outputs, batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            total_loss += loss.item()
        
        return {"loss": total_loss / len(dataloader)}
    
    def _compute_loss(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute loss"""
        loss_fn = nn.CrossEntropyLoss()
        if "genre" in batch and "genre_logits" in outputs:
            return loss_fn(outputs["genre_logits"], batch["genre"].squeeze())
        return torch.tensor(0.0, device=self.device)













