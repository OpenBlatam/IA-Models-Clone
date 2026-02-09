"""
Training Callbacks with Experiment Tracking
===========================================

Callbacks para entrenamiento con soporte para wandb y tensorboard.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional, Dict
from pathlib import Path
import torch

logger = logging.getLogger(__name__)

# Intentar importar wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

# Intentar importar tensorboard
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None


class Callback(ABC):
    """Clase base para callbacks."""
    
    def on_train_begin(self, trainer):
        """Llamado al inicio del entrenamiento."""
        pass
    
    def on_train_end(self, trainer):
        """Llamado al final del entrenamiento."""
        pass
    
    def on_epoch_begin(self, trainer, epoch: int):
        """Llamado al inicio de cada época."""
        pass
    
    def on_epoch_end(self, trainer, epoch: int, train_loss: float, val_loss: float):
        """Llamado al final de cada época."""
        pass
    
    def on_batch_end(self, trainer, batch_idx: int, loss: float):
        """Llamado al final de cada batch."""
        pass


class EarlyStopping(Callback):
    """Callback para early stopping."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, monitor: str = 'val_loss'):
        """
        Inicializar callback.
        
        Args:
            patience: Número de épocas sin mejora antes de parar
            min_delta: Mejora mínima requerida
            monitor: Métrica a monitorear ('val_loss' o 'train_loss')
        """
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.best_score = float('inf')
        self.counter = 0
        self.stopped = False
    
    def on_epoch_end(self, trainer, epoch: int, train_loss: float, val_loss: float):
        """Verificar si se debe parar."""
        if self.monitor == 'val_loss':
            current_score = val_loss
        else:
            current_score = train_loss
        
        if current_score < self.best_score - self.min_delta:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            logger.info(f"Early stopping triggered after {epoch + 1} epochs")
            self.stopped = True
            # Detener entrenamiento (el trainer debe verificar esto)
            trainer.should_stop = True


class ModelCheckpoint(Callback):
    """Callback para guardar checkpoints."""
    
    def __init__(
        self,
        checkpoint_dir: str,
        save_best: bool = True,
        save_last: bool = True,
        monitor: str = 'val_loss',
        mode: str = 'min'
    ):
        """
        Inicializar callback.
        
        Args:
            checkpoint_dir: Directorio para checkpoints
            save_best: Guardar mejor modelo
            save_last: Guardar último modelo
            monitor: Métrica a monitorear
            mode: 'min' o 'max'
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_best = save_best
        self.save_last = save_last
        self.monitor = monitor
        self.mode = mode
        self.best_score = float('inf') if mode == 'min' else float('-inf')
    
    def on_epoch_end(self, trainer, epoch: int, train_loss: float, val_loss: float):
        """Guardar checkpoint si es necesario."""
        if self.monitor == 'val_loss':
            current_score = val_loss
        else:
            current_score = train_loss
        
        # Guardar mejor modelo
        if self.save_best:
            is_best = False
            if self.mode == 'min':
                is_best = current_score < self.best_score
            else:
                is_best = current_score > self.best_score
            
            if is_best:
                self.best_score = current_score
                trainer._save_checkpoint(
                    self.checkpoint_dir / 'best_model.pt',
                    is_best=True
                )
                logger.info(f"Best model saved with {self.monitor}={current_score:.4f}")
        
        # Guardar último modelo
        if self.save_last:
            trainer._save_checkpoint(
                self.checkpoint_dir / 'last_model.pt',
                is_best=False
            )


class LearningRateScheduler(Callback):
    """Callback para scheduler de learning rate."""
    
    def __init__(self, scheduler):
        """
        Inicializar callback.
        
        Args:
            scheduler: Scheduler de PyTorch
        """
        self.scheduler = scheduler
    
    def on_epoch_end(self, trainer, epoch: int, train_loss: float, val_loss: float):
        """Actualizar learning rate."""
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(val_loss)
        else:
            self.scheduler.step()


class WandBCallback(Callback):
    """Callback para Weights & Biases."""
    
    def __init__(
        self,
        project_name: str,
        experiment_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Inicializar callback.
        
        Args:
            project_name: Nombre del proyecto en wandb
            experiment_name: Nombre del experimento
            config: Configuración a loggear
        """
        if not WANDB_AVAILABLE:
            raise ImportError("wandb is required for WandBCallback")
        
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.config = config or {}
        
        wandb.init(
            project=project_name,
            name=experiment_name,
            config=config
        )
    
    def on_train_begin(self, trainer):
        """Inicializar wandb."""
        wandb.watch(trainer.model)
    
    def on_epoch_end(self, trainer, epoch: int, train_loss: float, val_loss: float):
        """Loggear métricas."""
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'learning_rate': trainer.optimizer.param_groups[0]['lr']
        })
    
    def on_train_end(self, trainer):
        """Finalizar wandb."""
        wandb.finish()


class TensorBoardCallback(Callback):
    """Callback para TensorBoard."""
    
    def __init__(self, log_dir: str):
        """
        Inicializar callback.
        
        Args:
            log_dir: Directorio para logs
        """
        if not TENSORBOARD_AVAILABLE:
            raise ImportError("tensorboard is required for TensorBoardCallback")
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(self.log_dir))
    
    def on_epoch_end(self, trainer, epoch: int, train_loss: float, val_loss: float):
        """Loggear métricas."""
        self.writer.add_scalar('Loss/Train', train_loss, epoch)
        self.writer.add_scalar('Loss/Val', val_loss, epoch)
        self.writer.add_scalar('LearningRate', trainer.optimizer.param_groups[0]['lr'], epoch)
    
    def on_batch_end(self, trainer, batch_idx: int, loss: float):
        """Loggear pérdida por batch."""
        self.writer.add_scalar('Loss/Batch', loss, trainer.global_step)
    
    def on_train_end(self, trainer):
        """Cerrar writer."""
        self.writer.close()


class ProgressBarCallback(Callback):
    """Callback para mostrar barra de progreso."""
    
    def on_epoch_begin(self, trainer, epoch: int):
        """Mostrar información de época."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch + 1}/{trainer.current_epoch + 1}")
        logger.info(f"{'='*60}")
