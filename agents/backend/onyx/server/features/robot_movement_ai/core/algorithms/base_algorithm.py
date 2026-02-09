"""
Base Optimization Algorithm
============================

Clase base abstracta para todos los algoritmos de optimización.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import numpy as np

try:
    from ..types.types import TrajectoryPoint
except ImportError:
    # Fallback if types not available
    from typing import NamedTuple
    class TrajectoryPoint(NamedTuple):
        x: float = 0.0
        y: float = 0.0
        z: float = 0.0
        qx: float = 0.0
        qy: float = 0.0
        qz: float = 0.0
        qw: float = 1.0


class BaseOptimizationAlgorithm(ABC):
    """
    Clase base para algoritmos de optimización de trayectorias.
    
    Todos los algoritmos deben implementar el método optimize.
    """
    
    def __init__(self, name: str):
        """
        Inicializar algoritmo.
        
        Args:
            name: Nombre del algoritmo
        """
        self.name = name
    
    @abstractmethod
    def optimize(
        self,
        start: TrajectoryPoint,
        goal: TrajectoryPoint,
        obstacles: Optional[List[np.ndarray]] = None,
        constraints: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[TrajectoryPoint]:
        """
        Optimizar trayectoria.
        
        Args:
            start: Punto inicial
            goal: Punto objetivo
            obstacles: Lista de obstáculos
            constraints: Restricciones adicionales
            **kwargs: Parámetros adicionales específicos del algoritmo
            
        Returns:
            Trayectoria optimizada
        """
        pass
    
    def validate_inputs(
        self,
        start: TrajectoryPoint,
        goal: TrajectoryPoint
    ) -> bool:
        """
        Validar entradas del algoritmo.
        
        Args:
            start: Punto inicial
            goal: Punto objetivo
            
        Returns:
            True si las entradas son válidas
        """
        if start is None or goal is None:
            return False
        
        if not isinstance(start.position, np.ndarray) or not isinstance(goal.position, np.ndarray):
            return False
        
        if len(start.position) != 3 or len(goal.position) != 3:
            return False
        
        return True






