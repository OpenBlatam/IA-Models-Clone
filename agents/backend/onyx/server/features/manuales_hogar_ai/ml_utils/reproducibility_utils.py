"""
Reproducibility Utils - Utilidades de Reproducibilidad
=======================================================

Utilidades para garantizar reproducibilidad en experimentos.
"""

import logging
import torch
import numpy as np
import random
import os
from typing import Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ReproducibilityConfig:
    """Configuración de reproducibilidad."""
    seed: int = 42
    deterministic: bool = True
    benchmark: bool = False
    cudnn_deterministic: bool = True
    cudnn_benchmark: bool = False


class ReproducibilityManager:
    """
    Gestor de reproducibilidad.
    """
    
    def __init__(self, config: Optional[ReproducibilityConfig] = None):
        """
        Inicializar gestor.
        
        Args:
            config: Configuración (opcional)
        """
        self.config = config or ReproducibilityConfig()
        self._set_seeds()
        self._set_deterministic()
    
    def _set_seeds(self):
        """Establecer semillas."""
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.seed)
            torch.cuda.manual_seed_all(self.config.seed)
    
    def _set_deterministic(self):
        """Establecer modo determinístico."""
        if self.config.deterministic:
            torch.backends.cudnn.deterministic = self.config.cudnn_deterministic
            torch.backends.cudnn.benchmark = self.config.cudnn_benchmark
        
        if torch.cuda.is_available():
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    def set_seed(self, seed: int):
        """
        Establecer nueva semilla.
        
        Args:
            seed: Nueva semilla
        """
        self.config.seed = seed
        self._set_seeds()
    
    def get_state(self) -> Dict[str, Any]:
        """
        Obtener estado de reproducibilidad.
        
        Returns:
            Estado actual
        """
        return {
            'seed': self.config.seed,
            'python_random_state': random.getstate(),
            'numpy_random_state': np.random.get_state(),
            'torch_random_state': torch.get_rng_state(),
            'cuda_random_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        }
    
    def set_state(self, state: Dict[str, Any]):
        """
        Restaurar estado de reproducibilidad.
        
        Args:
            state: Estado a restaurar
        """
        random.setstate(state['python_random_state'])
        np.random.set_state(state['numpy_random_state'])
        torch.set_rng_state(state['torch_random_state'])
        
        if torch.cuda.is_available() and state['cuda_random_state'] is not None:
            torch.cuda.set_rng_state_all(state['cuda_random_state'])


def set_seed(seed: int = 42):
    """
    Establecer semilla para reproducibilidad.
    
    Args:
        seed: Semilla
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_deterministic():
    """Hacer operaciones determinísticas."""
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ExperimentSnapshot:
    """
    Snapshot de experimento para reproducibilidad.
    """
    
    def __init__(self):
        """Inicializar snapshot."""
        self.config: Dict[str, Any] = {}
        self.random_states: Dict[str, Any] = {}
        self.model_state: Optional[Dict[str, Any]] = None
        self.optimizer_state: Optional[Dict[str, Any]] = None
    
    def capture(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Capturar snapshot.
        
        Args:
            model: Modelo
            optimizer: Optimizador (opcional)
            config: Configuración (opcional)
        """
        # Capturar estados aleatorios
        self.random_states = {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        }
        
        # Capturar estado del modelo
        self.model_state = model.state_dict()
        
        # Capturar estado del optimizador
        if optimizer is not None:
            self.optimizer_state = optimizer.state_dict()
        
        # Capturar configuración
        if config is not None:
            self.config = config
    
    def restore(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None
    ):
        """
        Restaurar snapshot.
        
        Args:
            model: Modelo
            optimizer: Optimizador (opcional)
        """
        # Restaurar estados aleatorios
        random.setstate(self.random_states['python'])
        np.random.set_state(self.random_states['numpy'])
        torch.set_rng_state(self.random_states['torch'])
        
        if torch.cuda.is_available() and self.random_states['cuda'] is not None:
            torch.cuda.set_rng_state_all(self.random_states['cuda'])
        
        # Restaurar modelo
        if self.model_state is not None:
            model.load_state_dict(self.model_state)
        
        # Restaurar optimizador
        if optimizer is not None and self.optimizer_state is not None:
            optimizer.load_state_dict(self.optimizer_state)


def save_experiment_state(
    filepath: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    config: Optional[Dict[str, Any]] = None,
    seed: int = 42
):
    """
    Guardar estado completo de experimento.
    
    Args:
        filepath: Ruta del archivo
        model: Modelo
        optimizer: Optimizador (opcional)
        config: Configuración (opcional)
        seed: Semilla
    """
    snapshot = ExperimentSnapshot()
    snapshot.capture(model, optimizer, config)
    
    state = {
        'seed': seed,
        'random_states': snapshot.random_states,
        'model_state': snapshot.model_state,
        'optimizer_state': snapshot.optimizer_state,
        'config': snapshot.config
    }
    
    torch.save(state, filepath)
    logger.info(f"Experiment state saved to {filepath}")


def load_experiment_state(
    filepath: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None
) -> Dict[str, Any]:
    """
    Cargar estado completo de experimento.
    
    Args:
        filepath: Ruta del archivo
        model: Modelo
        optimizer: Optimizador (opcional)
        
    Returns:
        Estado cargado
    """
    state = torch.load(filepath)
    
    # Restaurar semilla
    set_seed(state['seed'])
    
    # Restaurar estados aleatorios
    random.setstate(state['random_states']['python'])
    np.random.set_state(state['random_states']['numpy'])
    torch.set_rng_state(state['random_states']['torch'])
    
    if torch.cuda.is_available() and state['random_states']['cuda'] is not None:
        torch.cuda.set_rng_state_all(state['random_states']['cuda'])
    
    # Restaurar modelo
    model.load_state_dict(state['model_state'])
    
    # Restaurar optimizador
    if optimizer is not None and state['optimizer_state'] is not None:
        optimizer.load_state_dict(state['optimizer_state'])
    
    logger.info(f"Experiment state loaded from {filepath}")
    return state




