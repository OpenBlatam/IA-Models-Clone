"""
Advanced PyTorch Trainer
=======================

Trainer modular con soporte para mixed precision, multi-GPU, y experiment tracking.
"""

import logging
from typing import Dict, Any, Optional, Callable, List
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DataParallel, DistributedDataParallel
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

from ..dl_data.dataset import create_dataloader
from ..dl_training.callbacks import Callback, EarlyStopping, ModelCheckpoint, LearningRateScheduler
from ..dl_training.optimizers import get_optimizer
from ..dl_training.schedulers import get_scheduler

logger = logging.getLogger(__name__)


class Trainer:
    """
    Trainer avanzado para modelos PyTorch.
    
    Características:
    - Mixed precision training
    - Multi-GPU support
    - Gradient accumulation
    - Early stopping
    - Model checkpointing
    - Experiment tracking
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        loss_fn: Optional[Callable] = None,
        device: Optional[torch.device] = None,
        use_amp: bool = True,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        callbacks: Optional[List[Callback]] = None,
        experiment_name: Optional[str] = None,
        checkpoint_dir: Optional[str] = None
    ):
        """
        Inicializar trainer.
        
        Args:
            model: Modelo PyTorch
            train_loader: DataLoader de entrenamiento
            val_loader: DataLoader de validación (opcional)
            optimizer: Optimizador (opcional, se crea si no se proporciona)
            scheduler: Scheduler de learning rate (opcional)
            loss_fn: Función de pérdida
            device: Dispositivo (CPU/GPU)
            use_amp: Usar mixed precision training
            gradient_accumulation_steps: Pasos de acumulación de gradiente
            max_grad_norm: Norma máxima de gradiente para clipping
            callbacks: Lista de callbacks
            experiment_name: Nombre del experimento
            checkpoint_dir: Directorio para checkpoints
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn or nn.MSELoss()
        self.use_amp = use_amp and torch.cuda.is_available()
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        
        # Device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # Mover modelo a device
        self.model = self.model.to(self.device)
        
        # Multi-GPU support
        if torch.cuda.device_count() > 1:
            logger.info(f"Using {torch.cuda.device_count()} GPUs")
            self.model = DataParallel(self.model)
        
        # Optimizer
        if optimizer is None:
            self.optimizer = get_optimizer(
                self.model,
                optimizer_type='adamw',
                lr=1e-4,
                weight_decay=1e-5
            )
        else:
            self.optimizer = optimizer
        
        # Scheduler
        self.scheduler = scheduler
        
        # Mixed precision
        if self.use_amp:
            self.scaler = GradScaler()
        else:
            self.scaler = None
        
        # Callbacks
        self.callbacks = callbacks or []
        
        # Experiment tracking
        self.experiment_name = experiment_name or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path('checkpoints')
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_model_state = None
        
        logger.info(f"Trainer initialized on device: {self.device}")
        logger.info(f"Mixed precision: {self.use_amp}")
        logger.info(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    
    def train_epoch(self) -> float:
        """
        Entrenar una época.
        
        Returns:
            Pérdida promedio de entrenamiento
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(progress_bar):
            # Mover batch a device
            batch = self._move_to_device(batch)
            
            # Forward pass con mixed precision
            if self.use_amp:
                with autocast():
                    loss = self._compute_loss(batch)
                    loss = loss / self.gradient_accumulation_steps
            else:
                loss = self._compute_loss(batch)
                loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.max_grad_norm > 0:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                # Optimizer step
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.global_step += 1
                
                # Callbacks
                for callback in self.callbacks:
                    callback.on_batch_end(
                        self,
                        batch_idx=batch_idx,
                        loss=loss.item() * self.gradient_accumulation_steps
                    )
            
            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item() * self.gradient_accumulation_steps})
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def validate(self) -> float:
        """
        Validar modelo.
        
        Returns:
            Pérdida promedio de validación
        """
        if self.val_loader is None:
            return 0.0
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                batch = self._move_to_device(batch)
                
                if self.use_amp:
                    with autocast():
                        loss = self._compute_loss(batch)
                else:
                    loss = self._compute_loss(batch)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def _compute_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        """
        Calcular pérdida.
        
        Args:
            batch: Batch de datos
            
        Returns:
            Tensor de pérdida
        """
        # Implementación específica depende del modelo
        if hasattr(self.model, 'compute_loss'):
            # Si el modelo tiene método compute_loss
            return self.model.compute_loss(batch)
        else:
            # Implementación genérica
            trajectory = batch.get('trajectory')
            if trajectory is None:
                raise ValueError("Batch must contain 'trajectory' key")
            
            # Forward pass
            if isinstance(self.model, (DataParallel, DistributedDataParallel)):
                output = self.model.module(trajectory)
            else:
                output = self.model(trajectory)
            
            # Calcular pérdida
            loss = self.loss_fn(output, trajectory)
            return loss
    
    def _move_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Mover batch a device."""
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            elif isinstance(value, (list, tuple)):
                device_batch[key] = [
                    v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for v in value
                ]
            else:
                device_batch[key] = value
        return device_batch
    
    def train(
        self,
        num_epochs: int,
        save_best: bool = True,
        save_last: bool = True
    ):
        """
        Entrenar modelo.
        
        Args:
            num_epochs: Número de épocas
            save_best: Guardar mejor modelo
            save_last: Guardar último modelo
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        
        # Callbacks on_train_begin
        for callback in self.callbacks:
            callback.on_train_begin(self)
        
        try:
            for epoch in range(num_epochs):
                self.current_epoch = epoch
                
                # Callbacks on_epoch_begin
                for callback in self.callbacks:
                    callback.on_epoch_begin(self, epoch)
                
                # Train
                train_loss = self.train_epoch()
                self.train_losses.append(train_loss)
                
                # Validate
                val_loss = self.validate()
                self.val_losses.append(val_loss)
                
                # Scheduler step
                if self.scheduler is not None:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()
                
                # Logging
                logger.info(
                    f"Epoch {epoch + 1}/{num_epochs} - "
                    f"Train Loss: {train_loss:.4f} - "
                    f"Val Loss: {val_loss:.4f}"
                )
                
                # Save best model
                if save_best and val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_model_state = self._get_model_state()
                    self._save_checkpoint(
                        self.checkpoint_dir / f"{self.experiment_name}_best.pt",
                        is_best=True
                    )
                
                # Callbacks on_epoch_end
                for callback in self.callbacks:
                    callback.on_epoch_end(self, epoch, train_loss, val_loss)
            
            # Save last model
            if save_last:
                self._save_checkpoint(
                    self.checkpoint_dir / f"{self.experiment_name}_last.pt",
                    is_best=False
                )
            
            # Callbacks on_train_end
            for callback in self.callbacks:
                callback.on_train_end(self)
            
            logger.info("Training completed")
            
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            if save_last:
                self._save_checkpoint(
                    self.checkpoint_dir / f"{self.experiment_name}_interrupted.pt",
                    is_best=False
                )
            raise
    
    def _get_model_state(self) -> Dict[str, Any]:
        """Obtener estado del modelo."""
        if isinstance(self.model, (DataParallel, DistributedDataParallel)):
            return self.model.module.state_dict()
        else:
            return self.model.state_dict()
    
    def _save_checkpoint(
        self,
        checkpoint_path: Path,
        is_best: bool = False
    ):
        """Guardar checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self._get_model_state(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'experiment_name': self.experiment_name
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: Path):
        """Cargar checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if isinstance(self.model, (DataParallel, DistributedDataParallel)):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'scaler_state_dict' in checkpoint and self.scaler is not None:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        logger.info(f"Checkpoint loaded: {checkpoint_path}")
