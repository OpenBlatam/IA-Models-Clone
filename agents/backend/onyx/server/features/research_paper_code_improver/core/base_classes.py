"""
Base Classes - Clases base para módulos de deep learning
=========================================================
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class BaseConfig:
    """Configuración base"""
    name: str = "default"
    enabled: bool = True
    verbose: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseManager(ABC):
    """Clase base para managers"""
    
    def __init__(self, config: Optional[BaseConfig] = None):
        self.config = config or BaseConfig()
        self.history: List[Dict[str, Any]] = []
        self.stats: Dict[str, Any] = {}
    
    def log_event(self, event_type: str, data: Dict[str, Any]):
        """Registra evento en historial"""
        event = {
            "type": event_type,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        self.history.append(event)
        if self.config.verbose:
            logger.info(f"{self.__class__.__name__}: {event_type}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas"""
        return {
            "total_events": len(self.history),
            "stats": self.stats.copy()
        }
    
    def clear_history(self):
        """Limpia historial"""
        self.history.clear()
        self.stats.clear()


class BaseTrainer(BaseManager):
    """Clase base para trainers"""
    
    def __init__(self, config: Optional[BaseConfig] = None):
        super().__init__(config)
        self.current_epoch = 0
        self.current_step = 0
        self.best_metrics: Dict[str, float] = {}
    
    @abstractmethod
    def train_step(self, *args, **kwargs):
        """Paso de entrenamiento (debe implementarse)"""
        pass
    
    @abstractmethod
    def validate_step(self, *args, **kwargs):
        """Paso de validación (debe implementarse)"""
        pass
    
    def update_metrics(self, metrics: Dict[str, float]):
        """Actualiza mejores métricas"""
        for key, value in metrics.items():
            if key not in self.best_metrics or value > self.best_metrics[key]:
                self.best_metrics[key] = value


class BaseEvaluator(BaseManager):
    """Clase base para evaluadores"""
    
    def __init__(self, config: Optional[BaseConfig] = None):
        super().__init__(config)
        self.evaluation_results: List[Dict[str, Any]] = []
    
    @abstractmethod
    def evaluate(self, *args, **kwargs) -> Dict[str, float]:
        """Evalúa modelo (debe implementarse)"""
        pass
    
    def get_latest_result(self) -> Optional[Dict[str, Any]]:
        """Obtiene último resultado"""
        return self.evaluation_results[-1] if self.evaluation_results else None


class BaseOptimizer(BaseManager):
    """Clase base para optimizadores"""
    
    def __init__(self, config: Optional[BaseConfig] = None):
        super().__init__(config)
        self.optimization_history: List[Dict[str, Any]] = []
    
    @abstractmethod
    def optimize(self, *args, **kwargs):
        """Optimiza modelo (debe implementarse)"""
        pass
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Obtiene resumen de optimización"""
        return {
            "total_optimizations": len(self.optimization_history),
            "latest": self.optimization_history[-1] if self.optimization_history else None
        }




