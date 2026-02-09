"""
Routing Model Trainer
=====================

Sistema de entrenamiento profesional para modelos de routing usando PyTorch.
Implementa mejores prácticas: early stopping, LR scheduling, gradient clipping, etc.
"""

import logging
import os
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from pathlib import Path
import time

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torch.cuda.amp import autocast, GradScaler
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Training features will be disabled.")

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    logger.warning("tqdm not available. Progress bars will be disabled.")


@dataclass
class TrainingMetrics:
    """Métricas de entrenamiento."""
    epoch: int
    train_loss: float
    val_loss: Optional[float] = None
    learning_rate: float = 0.0
    grad_norm: Optional[float] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class TrainingConfig:
    """Configuración de entrenamiento."""
    num_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 1.0
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-6
    validation_split: float = 0.2
    use_mixed_precision: bool = True
    save_best_model: bool = True
    checkpoint_dir: Optional[str] = None
    log_interval: int = 10


class LearningRateScheduler:
    """Scheduler de learning rate con múltiples estrategias."""
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        scheduler_type: str = "cosine",
        **kwargs
    ):
        """
        Inicializar scheduler.
        
        Args:
            optimizer: Optimizador de PyTorch
            scheduler_type: Tipo de scheduler ('cosine', 'step', 'plateau', 'warmup')
            **kwargs: Parámetros específicos del scheduler
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        
        self.optimizer = optimizer
        self.scheduler_type = scheduler_type.lower()
        
        if self.scheduler_type == "cosine":
            T_max = kwargs.get("T_max", 100)
            eta_min = kwargs.get("eta_min", 0)
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=T_max, eta_min=eta_min
            )
        elif self.scheduler_type == "step":
            step_size = kwargs.get("step_size", 30)
            gamma = kwargs.get("gamma", 0.1)
            self.scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=step_size, gamma=gamma
            )
        elif self.scheduler_type == "plateau":
            mode = kwargs.get("mode", "min")
            factor = kwargs.get("factor", 0.5)
            patience = kwargs.get("patience", 5)
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode=mode, factor=factor, patience=patience
            )
        elif self.scheduler_type == "warmup":
            warmup_steps = kwargs.get("warmup_steps", 10)
            total_steps = kwargs.get("total_steps", 100)
            self.scheduler = self._create_warmup_scheduler(warmup_steps, total_steps)
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    def _create_warmup_scheduler(self, warmup_steps: int, total_steps: int):
        """Crear scheduler con warmup."""
        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(
                0.0,
                float(total_steps - current_step) / float(max(1, total_steps - warmup_steps))
            )
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def step(self, metrics: Optional[float] = None):
        """Avanzar scheduler."""
        if self.scheduler_type == "plateau":
            if metrics is None:
                raise ValueError("Plateau scheduler requires metrics")
            self.scheduler.step(metrics)
        else:
            self.scheduler.step()
    
    def get_last_lr(self) -> List[float]:
        """Obtener último learning rate."""
        return self.scheduler.get_last_lr()


class EarlyStopping:
    """Early stopping para prevenir overfitting."""
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-6,
        mode: str = "min",
        restore_best_weights: bool = True
    ):
        """
        Inicializar early stopping.
        
        Args:
            patience: Número de epochs sin mejora antes de parar
            min_delta: Cambio mínimo para considerar mejora
            mode: 'min' o 'max' dependiendo de si queremos minimizar o maximizar
            restore_best_weights: Restaurar mejores pesos al final
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.best_score = float('inf') if mode == "min" else float('-inf')
        self.counter = 0
        self.best_weights: Optional[Dict[str, Any]] = None
        self.early_stop = False
    
    def __call__(self, score: float, model: nn.Module) -> bool:
        """
        Verificar si se debe hacer early stopping.
        
        Args:
            score: Score actual (loss o métrica)
            model: Modelo a monitorear
        
        Returns:
            True si se debe hacer early stopping
        """
        if self.mode == "min":
            is_better = score < (self.best_score - self.min_delta)
        else:
            is_better = score > (self.best_score + self.min_delta)
        
        if is_better:
            self.best_score = score
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights and self.best_weights:
                    model.load_state_dict(self.best_weights)
        
        return self.early_stop


class ModelTrainer:
    """Trainer profesional para modelos de routing."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[TrainingConfig] = None,
        device: Optional[torch.device] = None,
        experiment_tracker: Optional[Any] = None,
        use_distributed: bool = False,
        gradient_accumulation_steps: int = 1
    ):
        """
        Inicializar trainer.
        
        Args:
            model: Modelo PyTorch a entrenar
            train_loader: DataLoader de entrenamiento
            val_loader: DataLoader de validación (opcional)
            config: Configuración de entrenamiento
            device: Dispositivo (CPU/GPU)
            experiment_tracker: Tracker de experimentos (wandb/tensorboard)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for ModelTrainer")
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or TrainingConfig()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.experiment_tracker = experiment_tracker
        
        # Mover modelo a dispositivo
        self.model.to(self.device)
        
        # Distributed training
        self.use_distributed = use_distributed
        if use_distributed:
            try:
                from torch.nn.parallel import DistributedDataParallel as DDP
                self.model = DDP(self.model, device_ids=[self.device.index] if self.device.type == 'cuda' else None)
                logger.info("Model wrapped with DistributedDataParallel")
            except Exception as e:
                logger.warning(f"Failed to setup distributed training: {e}")
                self.use_distributed = False
        
        # Gradient accumulation
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Optimizador
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Learning rate scheduler
        self.lr_scheduler = LearningRateScheduler(
            self.optimizer,
            scheduler_type="cosine",
            T_max=self.config.num_epochs
        )
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=self.config.early_stopping_patience,
            min_delta=self.config.early_stopping_min_delta
        )
        
        # Mixed precision
        self.scaler = GradScaler() if self.config.use_mixed_precision else None
        
        # Historial
        self.history: List[TrainingMetrics] = []
        self.best_val_loss = float('inf')
        
        # Checkpointing
        if self.config.checkpoint_dir:
            self.checkpoint_dir = Path(self.config.checkpoint_dir)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.checkpoint_dir = None
    
    def train_epoch(self) -> float:
        """Entrenar un epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Habilitar detección de anomalías en modo debug
        use_anomaly_detection = os.getenv('PYTORCH_ANOMALY_DETECTION', 'False').lower() == 'true'
        
        progress_bar = tqdm(self.train_loader, desc="Training") if TQDM_AVAILABLE else self.train_loader
        
        # Gradient accumulation: solo hacer step cada N batches
        self.optimizer.zero_grad()
        accumulated_loss = 0.0
        
        # Context manager para anomaly detection
        anomaly_context = torch.autograd.detect_anomaly() if use_anomaly_detection else None
        if anomaly_context:
            anomaly_context.__enter__()
        
        for batch_idx, batch in enumerate(progress_bar):
            # Mover batch a dispositivo
            if isinstance(batch, (list, tuple)):
                inputs = [b.to(self.device) for b in batch[:-1]]
                targets = batch[-1].to(self.device)
            else:
                inputs = batch.get('features', batch.get('input')).to(self.device)
                targets = batch.get('target', batch.get('labels')).to(self.device)
            
            # Forward pass con mixed precision
            if self.config.use_mixed_precision and self.scaler:
                with autocast():
                    if isinstance(inputs, list):
                        outputs = self.model(*inputs)
                    else:
                        outputs = self.model(inputs)
                    loss = self._compute_loss(outputs, targets)
                
                # Normalizar pérdida para gradient accumulation
                loss = loss / self.gradient_accumulation_steps
                
                self.scaler.scale(loss).backward()
                accumulated_loss += loss.item()
                
                # Gradient clipping y step solo cada N batches
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    if self.config.gradient_clip_norm > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.gradient_clip_norm
                        )
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                if isinstance(inputs, list):
                    outputs = self.model(*inputs)
                else:
                    outputs = self.model(inputs)
                loss = self._compute_loss(outputs, targets)
                
                # Normalizar pérdida para gradient accumulation
                loss = loss / self.gradient_accumulation_steps
                loss.backward()
                accumulated_loss += loss.item()
                
                # Gradient clipping y step solo cada N batches
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    if self.config.gradient_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.gradient_clip_norm
                        )
                    
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            # Calcular grad norm y logging (solo cuando hacemos step)
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                grad_norm = self._compute_grad_norm()
                # Guardar pérdida antes de resetear
                batch_loss_value = accumulated_loss * self.gradient_accumulation_steps
                total_loss += batch_loss_value
                num_batches += 1
                
                # Logging
                if (batch_idx + 1) % (self.config.log_interval * self.gradient_accumulation_steps) == 0:
                    current_lr = self.lr_scheduler.get_last_lr()[0]
                    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
                    logger.info(
                        f"Batch {batch_idx + 1}/{len(self.train_loader)}, "
                        f"Loss: {avg_loss:.4f}, LR: {current_lr:.6f}, Grad: {grad_norm:.4f}"
                    )
                
                if self.experiment_tracker:
                    self.experiment_tracker.log({
                        'train/batch_loss': batch_loss_value,
                        'train/learning_rate': self.lr_scheduler.get_last_lr()[0],
                        'train/grad_norm': grad_norm
                    })
                
                # Resetear pérdida acumulada
                accumulated_loss = 0.0
        
        # Cerrar anomaly detection context
        if anomaly_context:
            anomaly_context.__exit__(None, None, None)
        
        # Manejar último batch incompleto si hay gradient accumulation
        if accumulated_loss > 0 and self.gradient_accumulation_steps > 1:
            # Hacer step final con los gradientes acumulados
            if self.config.gradient_clip_norm > 0:
                if self.config.use_mixed_precision and self.scaler:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_norm
                )
            
            if self.config.use_mixed_precision and self.scaler:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
            total_loss += accumulated_loss * self.gradient_accumulation_steps
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def validate(self) -> float:
        """Validar modelo."""
        if self.val_loader is None:
            return 0.0
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                if isinstance(batch, (list, tuple)):
                    inputs = [b.to(self.device) for b in batch[:-1]]
                    targets = batch[-1].to(self.device)
                else:
                    inputs = batch.get('features', batch.get('input')).to(self.device)
                    targets = batch.get('target', batch.get('labels')).to(self.device)
                
                if self.config.use_mixed_precision:
                    with autocast():
                        if isinstance(inputs, list):
                            outputs = self.model(*inputs)
                        else:
                            outputs = self.model(inputs)
                        loss = self._compute_loss(outputs, targets)
                else:
                    if isinstance(inputs, list):
                        outputs = self.model(*inputs)
                    else:
                        outputs = self.model(inputs)
                    loss = self._compute_loss(outputs, targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def train(self) -> Dict[str, Any]:
        """Entrenar modelo completo."""
        logger.info(f"Starting training for {self.config.num_epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(f"Mixed precision: {self.config.use_mixed_precision}")
        
        start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            epoch_start = time.time()
            
            # Training
            train_loss = self.train_epoch()
            
            # Validation
            val_loss = self.validate() if self.val_loader else None
            
            # Learning rate scheduling
            if val_loss is not None:
                self.lr_scheduler.step(val_loss)
            else:
                self.lr_scheduler.step()
            
            current_lr = self.lr_scheduler.get_last_lr()[0]
            
            # Métricas
            metrics = TrainingMetrics(
                epoch=epoch + 1,
                train_loss=train_loss,
                val_loss=val_loss,
                learning_rate=current_lr
            )
            self.history.append(metrics)
            
            # Logging
            epoch_time = time.time() - epoch_start
            logger.info(
                f"Epoch {epoch + 1}/{self.config.num_epochs} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f if val_loss else 'N/A'}, "
                f"LR: {current_lr:.6f}, "
                f"Time: {epoch_time:.2f}s"
            )
            
            # Experiment tracking
            if self.experiment_tracker:
                log_dict = {
                    'epoch': epoch + 1,
                    'train/loss': train_loss,
                    'train/learning_rate': current_lr,
                    'epoch_time': epoch_time
                }
                if val_loss is not None:
                    log_dict['val/loss'] = val_loss
                self.experiment_tracker.log(log_dict)
            
            # Early stopping
            if val_loss is not None:
                if self.early_stopping(val_loss, self.model):
                    logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                    break
                
                # Save best model
                if val_loss < self.best_val_loss and self.config.save_best_model:
                    self.best_val_loss = val_loss
                    self._save_checkpoint(epoch + 1, is_best=True)
            
            # Periodic checkpoint
            if self.checkpoint_dir and (epoch + 1) % 10 == 0:
                self._save_checkpoint(epoch + 1, is_best=False)
        
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f}s")
        
        # Final checkpoint
        if self.checkpoint_dir:
            self._save_checkpoint(self.config.num_epochs, is_best=False, is_final=True)
        
        return {
            'history': self.history,
            'best_val_loss': self.best_val_loss,
            'total_time': total_time,
            'num_epochs_trained': len(self.history)
        }
    
    def _compute_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calcular pérdida con validación de NaN/Inf.
        
        Args:
            outputs: Predicciones del modelo
            targets: Valores objetivo
        
        Returns:
            Tensor de pérdida
        """
        criterion = nn.MSELoss()
        loss = criterion(outputs, targets)
        
        # Validar pérdida
        if torch.isnan(loss):
            logger.error("NaN loss detected! This may indicate a problem with the model or data.")
            # Reemplazar con valor pequeño para continuar
            loss = torch.tensor(1e-6, device=loss.device, requires_grad=True)
        
        if torch.isinf(loss):
            logger.error("Inf loss detected! This may indicate a problem with the model or data.")
            # Reemplazar con valor grande pero finito
            loss = torch.tensor(1e6, device=loss.device, requires_grad=True)
        
        return loss
    
    def _compute_grad_norm(self) -> float:
        """Calcular norma de gradientes."""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** (1. / 2)
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False, is_final: bool = False):
        """Guardar checkpoint."""
        if self.checkpoint_dir is None:
            return
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history
        }
        
        if is_best:
            path = self.checkpoint_dir / 'best_model.pt'
        elif is_final:
            path = self.checkpoint_dir / f'final_model_epoch_{epoch}.pt'
        else:
            path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Cargar checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr_scheduler.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.history = checkpoint.get('history', [])
        logger.info(f"Checkpoint loaded: {checkpoint_path}")

