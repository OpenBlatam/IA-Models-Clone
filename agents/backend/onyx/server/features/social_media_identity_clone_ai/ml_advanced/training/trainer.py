"""
Sistema completo de entrenamiento con PyTorch
"""

import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Optional, Callable
from tqdm import tqdm
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class Trainer:
    """Trainer completo con todas las mejores prácticas"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: str = "cuda",
        use_mixed_precision: bool = True,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.use_mixed_precision = use_mixed_precision and device == "cuda"
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        
        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler() if self.use_mixed_precision else None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
    def train_epoch(
        self,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ) -> Dict[str, float]:
        """Entrena una época"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(progress_bar):
            # Mover batch a device
            batch = self._move_to_device(batch)
            
            # Forward pass con mixed precision
            if self.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = self.model(**batch)
                    loss = loss_fn(outputs, batch)
                    loss = loss / self.gradient_accumulation_steps
            else:
                outputs = self.model(**batch)
                loss = loss_fn(outputs, batch)
                loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            if self.use_mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.use_mixed_precision:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm
                    )
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm
                    )
                    optimizer.step()
                
                optimizer.zero_grad()
                
                if scheduler:
                    scheduler.step()
                
                self.global_step += 1
            
            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item() * self.gradient_accumulation_steps:.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })
        
        avg_loss = total_loss / num_batches
        return {"train_loss": avg_loss}
    
    def validate(
        self,
        loss_fn: Callable
    ) -> Dict[str, float]:
        """Valida el modelo"""
        if not self.val_loader:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                batch = self._move_to_device(batch)
                
                if self.use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(**batch)
                        loss = loss_fn(outputs, batch)
                else:
                    outputs = self.model(**batch)
                    loss = loss_fn(outputs, batch)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return {"val_loss": avg_loss}
    
    def train(
        self,
        num_epochs: int,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        checkpoint_dir: Optional[str] = None,
        save_best: bool = True,
        early_stopping_patience: Optional[int] = None,
        experiment_tracker: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Entrena el modelo
        
        Args:
            num_epochs: Número de épocas
            optimizer: Optimizador
            loss_fn: Función de pérdida
            scheduler: Scheduler de learning rate
            checkpoint_dir: Directorio para guardar checkpoints
            save_best: Si guardar mejor modelo
            early_stopping_patience: Paciencia para early stopping
            experiment_tracker: Tracker de experimentos (wandb, tensorboard)
        """
        history = {
            "train_loss": [],
            "val_loss": [],
            "learning_rates": []
        }
        
        patience_counter = 0
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch(optimizer, loss_fn, scheduler)
            history["train_loss"].append(train_metrics["train_loss"])
            
            # Validate
            val_metrics = self.validate(loss_fn)
            if val_metrics:
                history["val_loss"].append(val_metrics["val_loss"])
                val_loss = val_metrics["val_loss"]
            else:
                val_loss = train_metrics["train_loss"]
            
            # Learning rate
            current_lr = optimizer.param_groups[0]["lr"]
            history["learning_rates"].append(current_lr)
            
            # Log metrics
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {train_metrics['train_loss']:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"LR: {current_lr:.2e}"
            )
            
            # Experiment tracking
            if experiment_tracker:
                metrics = {
                    "epoch": epoch + 1,
                    "train_loss": train_metrics["train_loss"],
                    "learning_rate": current_lr
                }
                if val_metrics:
                    metrics["val_loss"] = val_loss
                
                if hasattr(experiment_tracker, "log"):
                    experiment_tracker.log(metrics)
            
            # Save checkpoint
            if checkpoint_dir:
                self.save_checkpoint(
                    checkpoint_dir,
                    epoch,
                    train_metrics["train_loss"],
                    val_loss if val_metrics else None
                )
            
            # Save best model
            if save_best and val_metrics:
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    if checkpoint_dir:
                        self.save_checkpoint(
                            checkpoint_dir,
                            epoch,
                            train_metrics["train_loss"],
                            val_loss,
                            is_best=True
                        )
                    patience_counter = 0
                else:
                    patience_counter += 1
            
            # Early stopping
            if early_stopping_patience and patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        return {
            "history": history,
            "best_val_loss": self.best_val_loss,
            "epochs_trained": self.current_epoch + 1
        }
    
    def save_checkpoint(
        self,
        checkpoint_dir: str,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float] = None,
        is_best: bool = False
    ):
        """Guarda checkpoint"""
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val_loss": self.best_val_loss,
            "global_step": self.global_step
        }
        
        # Guardar checkpoint regular
        checkpoint_path = Path(checkpoint_dir) / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Guardar mejor modelo
        if is_best:
            best_path = Path(checkpoint_dir) / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Best model saved: {best_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Carga checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint.get("best_val_loss", float('inf'))
        self.global_step = checkpoint.get("global_step", 0)
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
    
    def _move_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Mueve batch a device"""
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

