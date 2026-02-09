"""
Early Stopping System - Sistema de early stopping para entrenamiento
======================================================================
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class StoppingMode(Enum):
    """Modos de early stopping"""
    MIN = "min"  # Detener cuando métrica deja de disminuir
    MAX = "max"  # Detener cuando métrica deja de aumentar


@dataclass
class EarlyStoppingConfig:
    """Configuración de early stopping"""
    monitor: str = "val_loss"
    mode: StoppingMode = StoppingMode.MIN
    patience: int = 10
    min_delta: float = 0.0
    restore_best_weights: bool = True
    verbose: bool = True


class EarlyStopping:
    """Sistema de early stopping"""
    
    def __init__(self, config: EarlyStoppingConfig):
        self.config = config
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        self.stopped_epoch = 0
        self.wait = 0
        self.stopped = False
        
        if config.mode == StoppingMode.MIN:
            self.best_score = float('inf')
            self.is_better = lambda current, best: current < best - config.min_delta
        else:
            self.best_score = float('-inf')
            self.is_better = lambda current, best: current > best + config.min_delta
    
    def __call__(self, epoch: int, metrics: Dict[str, float], model: Any) -> bool:
        """Verifica si se debe detener el entrenamiento"""
        if self.config.monitor not in metrics:
            logger.warning(f"Métrica {self.config.monitor} no encontrada en metrics")
            return False
        
        current_score = metrics[self.config.monitor]
        
        if self.is_better(current_score, self.best_score):
            self.best_score = current_score
            self.counter = 0
            self.wait = 0
            
            # Guardar mejores pesos
            if self.config.restore_best_weights:
                try:
                    if hasattr(model, 'state_dict'):
                        self.best_weights = model.state_dict().copy()
                    elif hasattr(model, 'get_weights'):
                        self.best_weights = model.get_weights()
                except Exception as e:
                    logger.warning(f"Error guardando mejores pesos: {e}")
            
            if self.config.verbose:
                logger.info(
                    f"Mejora detectada en epoch {epoch}. "
                    f"{self.config.monitor}: {current_score:.4f}"
                )
        else:
            self.wait += 1
            self.counter += 1
            
            if self.config.verbose:
                logger.info(
                    f"No hay mejora por {self.wait} epochs. "
                    f"Mejor {self.config.monitor}: {self.best_score:.4f}"
                )
            
            if self.wait >= self.config.patience:
                self.stopped_epoch = epoch
                self.stopped = True
                
                if self.config.verbose:
                    logger.info(
                        f"Early stopping activado en epoch {epoch}. "
                        f"Mejor {self.config.monitor}: {self.best_score:.4f}"
                    )
                
                # Restaurar mejores pesos
                if self.config.restore_best_weights and self.best_weights is not None:
                    try:
                        if hasattr(model, 'load_state_dict'):
                            model.load_state_dict(self.best_weights)
                        elif hasattr(model, 'set_weights'):
                            model.set_weights(self.best_weights)
                        logger.info("Mejores pesos restaurados")
                    except Exception as e:
                        logger.warning(f"Error restaurando mejores pesos: {e}")
                
                return True
        
        return False
    
    def reset(self):
        """Resetea el early stopping"""
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        self.stopped_epoch = 0
        self.wait = 0
        self.stopped = False
        
        if self.config.mode == StoppingMode.MIN:
            self.best_score = float('inf')
        else:
            self.best_score = float('-inf')
    
    def get_best_score(self) -> Optional[float]:
        """Obtiene el mejor score"""
        return self.best_score




