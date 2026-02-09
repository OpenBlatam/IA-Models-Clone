"""
Model Training System - Sistema de entrenamiento de modelos
===========================================================
Entrenamiento eficiente con PyTorch, mixed precision, y optimizaciones
"""

import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from dataclasses import dataclass
from tqdm import tqdm
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuración de entrenamiento"""
    batch_size: int = 32
    num_epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    use_mixed_precision: bool = True
    use_gpu: bool = True
    num_workers: int = 4
    pin_memory: bool = True
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 1000
    save_checkpoint_every: int = 1000
    eval_every: int = 500


class ModelTrainer:
    """Sistema de entrenamiento de modelos"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() and config.use_gpu else "cpu")
        self.scaler = torch.cuda.amp.GradScaler() if config.use_mixed_precision and self.device.type == "cuda" else None
        self.training_history: List[Dict[str, Any]] = []
        
        logger.info(f"Initialized trainer on device: {self.device}")
    
    def train_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        epoch: int
    ) -> Dict[str, float]:
        """Entrena una época"""
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Mover batch a device
            if isinstance(batch, dict):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            elif isinstance(batch, (list, tuple)):
                batch = [item.to(self.device) if isinstance(item, torch.Tensor) else item for item in batch]
            else:
                batch = batch.to(self.device)
            
            # Forward pass con mixed precision
            if self.scaler:
                with torch.cuda.amp.autocast():
                    output = model(batch)
                    loss = criterion(output, batch) if not isinstance(batch, dict) else criterion(output, batch["target"])
                    
                    # Gradient accumulation
                    loss = loss / self.config.gradient_accumulation_steps
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    optimizer.zero_grad()
            else:
                output = model(batch)
                loss = criterion(output, batch) if not isinstance(batch, dict) else criterion(output, batch["target"])
                
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()
                
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip)
                    optimizer.step()
                    optimizer.zero_grad()
            
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            num_batches += 1
            
            # Actualizar progress bar
            progress_bar.set_postfix({"loss": f"{loss.item() * self.config.gradient_accumulation_steps:.4f}"})
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return {"train_loss": avg_loss}
    
    def validate(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        criterion: nn.Module
    ) -> Dict[str, float]:
        """Valida el modelo"""
        model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                # Mover batch a device
                if isinstance(batch, dict):
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                elif isinstance(batch, (list, tuple)):
                    batch = [item.to(self.device) if isinstance(item, torch.Tensor) else item for item in batch]
                else:
                    batch = batch.to(self.device)
                
                # Forward pass
                if self.scaler:
                    with torch.cuda.amp.autocast():
                        output = model(batch)
                        loss = criterion(output, batch) if not isinstance(batch, dict) else criterion(output, batch["target"])
                else:
                    output = model(batch)
                    loss = criterion(output, batch) if not isinstance(batch, dict) else criterion(output, batch["target"])
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return {"val_loss": avg_loss}
    
    def train(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        checkpoint_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """Entrena el modelo completo"""
        model = model.to(self.device)
        best_val_loss = float('inf')
        
        for epoch in range(self.config.num_epochs):
            # Entrenar
            train_metrics = self.train_epoch(model, train_loader, optimizer, criterion, epoch)
            
            # Validar
            val_metrics = {}
            if val_loader:
                val_metrics = self.validate(model, val_loader, criterion)
                
                # Guardar mejor modelo
                if val_metrics["val_loss"] < best_val_loss:
                    best_val_loss = val_metrics["val_loss"]
                    if checkpoint_dir:
                        torch.save({
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "val_loss": best_val_loss
                        }, f"{checkpoint_dir}/best_model.pt")
            
            # Actualizar scheduler
            if scheduler:
                scheduler.step()
            
            # Guardar historial
            metrics = {**train_metrics, **val_metrics, "epoch": epoch + 1}
            self.training_history.append(metrics)
            
            logger.info(f"Epoch {epoch+1}/{self.config.num_epochs} - Train Loss: {train_metrics['train_loss']:.4f} - Val Loss: {val_metrics.get('val_loss', 'N/A')}")
        
        return {
            "status": "completed",
            "total_epochs": self.config.num_epochs,
            "best_val_loss": best_val_loss,
            "history": self.training_history
        }




