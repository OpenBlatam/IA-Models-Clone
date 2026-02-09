"""
Training Callbacks
==================

Callbacks para el proceso de entrenamiento.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import os

logger = logging.getLogger(__name__)


class Callback(ABC):
    """Clase base para callbacks."""
    
    @abstractmethod
    def on_train_begin(self, trainer):
        """Llamado al inicio del entrenamiento."""
        pass
    
    @abstractmethod
    def on_train_end(self, trainer):
        """Llamado al final del entrenamiento."""
        pass
    
    @abstractmethod
    def on_epoch_begin(self, trainer, epoch: int):
        """Llamado al inicio de cada época."""
        pass
    
    @abstractmethod
    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float]) -> bool:
        """
        Llamado al final de cada época.
        
        Returns:
            True si se debe detener el entrenamiento
        """
        return False


class EarlyStopping(Callback):
    """Callback para early stopping."""
    
    def __init__(self, patience: int = 20, min_delta: float = 0.0, monitor: str = "val_loss"):
        """
        Inicializar early stopping.
        
        Args:
            patience: Número de épocas sin mejora antes de parar
            min_delta: Mejora mínima para considerar progreso
            monitor: Métrica a monitorear
        """
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.best_value = float('inf')
        self.patience_counter = 0
        self.stopped_epoch = 0
    
    def on_train_begin(self, trainer):
        """Resetear contadores."""
        self.best_value = float('inf')
        self.patience_counter = 0
        self.stopped_epoch = 0
    
    def on_train_end(self, trainer):
        """Log final."""
        if self.stopped_epoch > 0:
            logger.info(f"Early stopping activado en época {self.stopped_epoch}")
    
    def on_epoch_begin(self, trainer, epoch: int):
        """No hacer nada."""
        pass
    
    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float]) -> bool:
        """Verificar si se debe parar."""
        current_value = metrics.get(self.monitor, float('inf'))
        
        if current_value < self.best_value - self.min_delta:
            self.best_value = current_value
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        if self.patience_counter >= self.patience:
            self.stopped_epoch = epoch + 1
            return True
        
        return False


class ModelCheckpoint(Callback):
    """Callback para guardar checkpoints."""
    
    def __init__(self, checkpoint_dir: str = "./checkpoints", save_best: bool = True, 
                 save_frequency: int = 10):
        """
        Inicializar model checkpoint.
        
        Args:
            checkpoint_dir: Directorio para checkpoints
            save_best: Guardar mejor modelo
            save_frequency: Frecuencia de guardado (cada N épocas)
        """
        self.checkpoint_dir = checkpoint_dir
        self.save_best = save_best
        self.save_frequency = save_frequency
        self.best_val_loss = float('inf')
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def on_train_begin(self, trainer):
        """No hacer nada."""
        pass
    
    def on_train_end(self, trainer):
        """Guardar checkpoint final."""
        final_path = os.path.join(self.checkpoint_dir, "final_model.pt")
        trainer.model.save_checkpoint(
            final_path,
            optimizer=trainer.optimizer,
            scheduler=trainer.scheduler
        )
    
    def on_epoch_begin(self, trainer, epoch: int):
        """No hacer nada."""
        pass
    
    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float]) -> bool:
        """Guardar checkpoint si es necesario."""
        val_loss = metrics.get("val_loss", float('inf'))
        
        # Guardar mejor modelo
        if self.save_best and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_path = os.path.join(self.checkpoint_dir, "best_model.pt")
            trainer.model.save_checkpoint(
                best_path,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
                metadata={"epoch": epoch + 1, "val_loss": val_loss}
            )
        
        # Guardar periódicamente
        if (epoch + 1) % self.save_frequency == 0:
            periodic_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pt")
            trainer.model.save_checkpoint(
                periodic_path,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler
            )
        
        return False


class LearningRateScheduler(Callback):
    """Callback para logging de learning rate."""
    
    def on_train_begin(self, trainer):
        """No hacer nada."""
        pass
    
    def on_train_end(self, trainer):
        """No hacer nada."""
        pass
    
    def on_epoch_begin(self, trainer, epoch: int):
        """No hacer nada."""
        pass
    
    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float]) -> bool:
        """Log learning rate."""
        lr = trainer.optimizer.param_groups[0]['lr']
        logger.debug(f"Learning rate: {lr:.6f}")
        return False


