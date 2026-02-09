"""
A* Algorithm
============

Implementación del algoritmo A* para optimización de trayectorias.
"""

import numpy as np
import logging
from typing import List, Optional, Dict, Any
from collections import deque

from .base_algorithm import BaseOptimizationAlgorithm
from ..types.types import TrajectoryPoint
from ..constants import (
    DEFAULT_GRID_RESOLUTION,
    A_STAR_GOAL_TOLERANCE
)
from ..utils.quaternion_utils import quaternion_slerp
from ..utils.math_utils import point_in_obstacle

logger = logging.getLogger(__name__)


class AStarAlgorithm(BaseOptimizationAlgorithm):
    """
    Algoritmo A* para optimización de trayectorias.
    
    A* es un algoritmo de búsqueda en grafos que encuentra el camino
    más corto entre dos puntos considerando obstáculos.
    """
    
    def __init__(self, grid_resolution: float = DEFAULT_GRID_RESOLUTION):
        """
        Inicializar algoritmo A*.
        
        Args:
            grid_resolution: Resolución del grid para discretización
        """
        super().__init__("A*")
        self.grid_resolution = grid_resolution
    
    def optimize(
        self,
        start: TrajectoryPoint,
        goal: TrajectoryPoint,
        obstacles: Optional[List[np.ndarray]] = None,
        constraints: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[TrajectoryPoint]:
        """
        Optimizar usando A*.
        
        Args:
            start: Punto inicial
            goal: Punto objetivo
            obstacles: Obstáculos
            constraints: Restricciones
            **kwargs: Parámetros adicionales (grid_resolution puede sobrescribir)
            
        Returns:
            Trayectoria optimizada
        """
        if not self.validate_inputs(start, goal):
            raise ValueError("Invalid inputs for A* algorithm")
        
        logger.debug("Using A* algorithm for trajectory optimization")
        
        # Permitir sobrescribir grid_resolution
        grid_resolution = kwargs.get("grid_resolution", self.grid_resolution)
        
        # Crear grid discreto
        min_bounds = np.minimum(start.position, goal.position) - 0.5
        max_bounds = np.maximum(start.position, goal.position) + 0.5
        
        # Discretizar espacio
        grid_size = ((max_bounds - min_bounds) / grid_resolution).astype(int)
        
        # Convertir posiciones a índices de grid
        start_idx = ((start.position - min_bounds) / grid_resolution).astype(int)
        goal_idx = ((goal.position - min_bounds) / grid_resolution).astype(int)
        
        # A* search
        path_indices = self._astar_search(
            start_idx, goal_idx, grid_size, obstacles, min_bounds, grid_resolution
        )
        
        # Convertir índices a trayectoria
        trajectory = []
        for i, idx in enumerate(path_indices):
            position = min_bounds + idx * grid_resolution
            # Interpolar orientación
            alpha = i / max(len(path_indices) - 1, 1)
            orientation = quaternion_slerp(start.orientation, goal.orientation, alpha)
            
            point = TrajectoryPoint(
                position=position,
                orientation=orientation,
                timestamp=i * 0.01
            )
            trajectory.append(point)
        
        return trajectory
    
    def _astar_search(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        grid_size: np.ndarray,
        obstacles: Optional[List[np.ndarray]],
        min_bounds: np.ndarray,
        grid_resolution: float
    ) -> List[np.ndarray]:
        """Implementación de A* search."""
        # Estructuras para A*
        open_set = [(0, tuple(start))]
        came_from = {}
        g_score = {tuple(start): 0}
        f_score = {tuple(start): np.linalg.norm(goal - start)}
        
        visited = set()
        
        while open_set:
            # Obtener nodo con menor f_score
            open_set.sort(key=lambda x: x[0])
            current_f, current = open_set.pop(0)
            current = np.array(current)
            
            if tuple(current) in visited:
                continue
            
            visited.add(tuple(current))
            
            # Verificar si llegamos al objetivo
            if np.allclose(current, goal, atol=A_STAR_GOAL_TOLERANCE):
                # Reconstruir camino
                path = []
                node = tuple(current)
                while node in came_from:
                    path.append(np.array(node))
                    node = came_from[node]
                path.append(start)
                path.reverse()
                return path
            
            # Explorar vecinos (6 direcciones)
            neighbors = [
                current + np.array([1, 0, 0]),
                current + np.array([-1, 0, 0]),
                current + np.array([0, 1, 0]),
                current + np.array([0, -1, 0]),
                current + np.array([0, 0, 1]),
                current + np.array([0, 0, -1]),
            ]
            
            for neighbor in neighbors:
                # Verificar límites
                if np.any(neighbor < 0) or np.any(neighbor >= grid_size):
                    continue
                
                # Verificar obstáculos
                if obstacles:
                    neighbor_pos = min_bounds + neighbor * grid_resolution
                    if any(point_in_obstacle(neighbor_pos, obs) for obs in obstacles):
                        continue
                
                neighbor_tuple = tuple(neighbor)
                tentative_g = g_score[tuple(current)] + 1
                
                if neighbor_tuple not in g_score or tentative_g < g_score[neighbor_tuple]:
                    came_from[neighbor_tuple] = tuple(current)
                    g_score[neighbor_tuple] = tentative_g
                    h_score = np.linalg.norm(goal - neighbor)
                    f_score[neighbor_tuple] = tentative_g + h_score
                    open_set.append((f_score[neighbor_tuple], neighbor_tuple))
        
        # No se encontró camino, retornar camino directo
        logger.warning("A* could not find path, using direct path")
        return [start, goal]

