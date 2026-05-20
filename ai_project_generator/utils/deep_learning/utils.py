"""
Utility Functions
=================

Funciones de utilidad para proyectos de deep learning.
"""

from typing import Dict, List, Any, Optional
import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)


def generate_utils_code() -> str:
    """Genera código de utilidades."""
    return '''"""
Utility Functions
=================

Funciones de utilidad para proyectos de deep learning.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json
import yaml
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """Establece semilla para reproducibilidad."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Seed set to {seed}")


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Cuenta parámetros del modelo.
    
    Returns:
        Diccionario con total, trainable y no trainable
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable
    
    return {
        'total': total,
        'trainable': trainable,
        'non_trainable': non_trainable
    }


def save_config(config: Dict[str, Any], path: str):
    """Guarda configuración en archivo YAML."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"Config saved to {path}")


def load_config(path: str) -> Dict[str, Any]:
    """Carga configuración desde archivo YAML."""
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Config loaded from {path}")
    return config


def get_device() -> torch.device:
    """Obtiene dispositivo disponible."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    
    return device


def format_time(seconds: float) -> str:
    """Formatea tiempo en formato legible."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


class EarlyStopping:
    """Early stopping para entrenamiento."""
    
    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = 'min'
    ):
        """
        Args:
            patience: Número de epochs sin mejora antes de parar
            min_delta: Cambio mínimo para considerar mejora
            mode: 'min' o 'max' para minimizar o maximizar métrica
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        Args:
            score: Score actual
        
        Returns:
            True si debe parar, False si continuar
        """
        if self.best_score is None:
            self.best_score = score
        elif self._is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def _is_better(self, current: float, best: float) -> bool:
        """Verifica si current es mejor que best."""
        if self.mode == 'min':
            return current < best - self.min_delta
        else:
            return current > best + self.min_delta


class CheckpointManager:
    """Gestiona checkpoints del modelo."""
    
    def __init__(self, save_dir: str, keep_last_n: int = 3):
        """
        Args:
            save_dir: Directorio para guardar checkpoints
            keep_last_n: Número de checkpoints a mantener
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
        self.checkpoints = []
    
    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        score: float,
        is_best: bool = False
    ):
        """
        Guarda checkpoint.
        
        Args:
            model: Modelo
            optimizer: Optimizador
            epoch: Época actual
            score: Score de la época
            is_best: Si es el mejor modelo
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'score': score,
            'timestamp': datetime.now().isoformat()
        }
        
        # Guardar checkpoint regular
        checkpoint_path = self.save_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        self.checkpoints.append((epoch, checkpoint_path, score))
        
        # Guardar mejor modelo
        if is_best:
            best_path = self.save_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Best model saved to {best_path}")
        
        # Limpiar checkpoints antiguos
        self._cleanup_old_checkpoints()
    
    def _cleanup_old_checkpoints(self):
        """Elimina checkpoints antiguos."""
        if len(self.checkpoints) > self.keep_last_n:
            # Ordenar por score (mejor primero)
            self.checkpoints.sort(key=lambda x: x[2], reverse=True)
            
            # Eliminar los peores
            for epoch, path, _ in self.checkpoints[self.keep_last_n:]:
                if path.exists():
                    path.unlink()
                    logger.info(f"Removed old checkpoint: {path}")
            
            # Mantener solo los últimos N
            self.checkpoints = self.checkpoints[:self.keep_last_n]
    
    def load(self, path: str) -> Dict[str, Any]:
        """Carga checkpoint."""
        checkpoint = torch.load(path, map_location='cpu')
        logger.info(f"Checkpoint loaded from {path}")
        return checkpoint


def log_model_info(model: nn.Module, input_shape: Tuple[int, ...]):
    """Registra información del modelo."""
    params = count_parameters(model)
    
    logger.info("=" * 50)
    logger.info("Model Information")
    logger.info("=" * 50)
    logger.info(f"Total parameters: {params['total']:,}")
    logger.info(f"Trainable parameters: {params['trainable']:,}")
    logger.info(f"Non-trainable parameters: {params['non_trainable']:,}")
    logger.info(f"Input shape: {input_shape}")
    logger.info("=" * 50)
'''

