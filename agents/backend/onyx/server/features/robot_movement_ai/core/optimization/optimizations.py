"""
Optimization Utilities
======================

Utilidades adicionales para optimización de performance.
"""

import numpy as np
from typing import List, Callable, Any, Optional
import functools
from numba import jit, njit
import logging

logger = logging.getLogger(__name__)


# Intentar usar numba para aceleración JIT
try:
    USE_NUMBA = True
except ImportError:
    USE_NUMBA = False
    logger.warning("Numba not available, using pure Python implementations")


def vectorized_distance(points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
    """
    Calcular distancias entre múltiples pares de puntos (vectorizado).
    
    Args:
        points1: Array de puntos [N, 3]
        points2: Array de puntos [N, 3]
        
    Returns:
        Array de distancias [N]
    """
    return np.linalg.norm(points2 - points1, axis=1)


def batch_normalize(vectors: np.ndarray) -> np.ndarray:
    """
    Normalizar múltiples vectores a la vez (vectorizado).
    
    Args:
        vectors: Array de vectores [N, 3]
        
    Returns:
        Array de vectores normalizados [N, 3]
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)  # Evitar división por cero
    return vectors / norms


def fast_interpolate(
    start: np.ndarray,
    end: np.ndarray,
    num_points: int
) -> np.ndarray:
    """
    Interpolación rápida entre dos puntos.
    
    Args:
        start: Punto inicial [3]
        end: Punto final [3]
        num_points: Número de puntos
        
    Returns:
        Array de puntos interpolados [num_points, 3]
    """
    alphas = np.linspace(0, 1, num_points)
    return start + alphas[:, np.newaxis] * (end - start)


def memoize(maxsize: int = 128):
    """
    Decorador de memoización optimizado.
    
    Args:
        maxsize: Tamaño máximo del caché
        
    Usage:
        @memoize(maxsize=256)
        def expensive_function(x, y):
            ...
    """
    def decorator(func: Callable) -> Callable:
        cache = {}
        cache_order = []
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Crear clave de caché
            import hashlib
            import pickle
            key = (args, tuple(sorted(kwargs.items())))
            key_hash = hashlib.md5(pickle.dumps(key)).hexdigest()
            
            if key_hash in cache:
                return cache[key_hash]
            
            result = func(*args, **kwargs)
            
            # Gestionar tamaño del caché
            if len(cache) >= maxsize:
                oldest_key = cache_order.pop(0)
                del cache[oldest_key]
            
            cache[key_hash] = result
            cache_order.append(key_hash)
            
            return result
        
        return wrapper
    return decorator


def lazy_property(func: Callable) -> property:
    """
    Decorador para propiedades lazy (calculadas una vez).
    
    Usage:
        class MyClass:
            @lazy_property
            def expensive_computation(self):
                return expensive_calculation()
    """
    attr_name = f"_{func.__name__}"
    
    @property
    @functools.wraps(func)
    def wrapper(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, func(self))
        return getattr(self, attr_name)
    
    return wrapper


class VectorizedOperations:
    """
    Clase para operaciones vectorizadas optimizadas.
    """
    
    @staticmethod
    def distances_to_obstacles(
        points: np.ndarray,
        obstacles: List[np.ndarray]
    ) -> np.ndarray:
        """
        Calcular distancias de múltiples puntos a múltiples obstáculos.
        
        Args:
            points: Array de puntos [N, 3]
            obstacles: Lista de obstáculos [M, 6]
            
        Returns:
            Array de distancias mínimas [N]
        """
        if len(obstacles) == 0:
            return np.full(len(points), np.inf)
        
        min_distances = np.full(len(points), np.inf)
        
        for obstacle in obstacles:
            min_corner = obstacle[:3]
            max_corner = obstacle[3:]
            center = (min_corner + max_corner) / 2
            
            # Distancia al centro del obstáculo
            distances = np.linalg.norm(points - center, axis=1)
            
            # Actualizar mínimos
            min_distances = np.minimum(min_distances, distances)
        
        return min_distances
    
    @staticmethod
    def batch_quaternion_slerp(
        q1: np.ndarray,
        q2: np.ndarray,
        alphas: np.ndarray
    ) -> np.ndarray:
        """
        SLERP para múltiples valores de alpha a la vez.
        
        Args:
            q1: Quaternion inicial [4]
            q2: Quaternion final [4]
            alphas: Array de factores de interpolación [N]
            
        Returns:
            Array de quaterniones interpolados [N, 4]
        """
        # Normalizar
        q1 = q1 / np.linalg.norm(q1)
        q2 = q2 / np.linalg.norm(q2)
        
        dot = np.clip(np.dot(q1, q2), -1.0, 1.0)
        
        if abs(dot) > 0.9995:
            # Quaterniones muy cercanos, usar interpolación lineal
            return q1 + alphas[:, np.newaxis] * (q2 - q1)
        
        theta = np.arccos(abs(dot))
        sin_theta = np.sin(theta)
        
        if sin_theta < 1e-6:
            return np.tile(q1, (len(alphas), 1))
        
        w1 = np.sin((1 - alphas) * theta) / sin_theta
        w2 = np.sin(alphas * theta) / sin_theta
        
        if dot < 0:
            q2 = -q2
        
        return w1[:, np.newaxis] * q1 + w2[:, np.newaxis] * q2


def optimize_trajectory_batch(
    starts: List[np.ndarray],
    goals: List[np.ndarray],
    optimizer_func: Callable
) -> List[np.ndarray]:
    """
    Optimizar múltiples trayectorias en paralelo.
    
    Args:
        starts: Lista de puntos iniciales
        goals: Lista de puntos objetivos
        optimizer_func: Función de optimización
        
    Returns:
        Lista de trayectorias optimizadas
    """
    from concurrent.futures import ThreadPoolExecutor
    
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(optimizer_func, start, goal)
            for start, goal in zip(starts, goals)
        ]
        return [future.result() for future in futures]


class PerformanceOptimizer:
    """
    Optimizador de performance con auto-tuning.
    """
    
    def __init__(self):
        """Inicializar optimizador."""
        self.performance_history = []
        self.current_batch_size = 10
    
    def optimize_batch_size(
        self,
        operation: Callable,
        data: List[Any],
        min_batch: int = 1,
        max_batch: int = 100
    ) -> int:
        """
        Encontrar tamaño de lote óptimo para operación.
        
        Args:
            operation: Operación a optimizar
            data: Datos a procesar
            min_batch: Tamaño mínimo de lote
            max_batch: Tamaño máximo de lote
            
        Returns:
            Tamaño de lote óptimo
        """
        import time
        
        best_batch_size = min_batch
        best_time = float('inf')
        
        for batch_size in range(min_batch, max_batch + 1, 10):
            start_time = time.time()
            
            # Procesar en lotes
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]
                operation(batch)
            
            elapsed = time.time() - start_time
            
            if elapsed < best_time:
                best_time = elapsed
                best_batch_size = batch_size
        
        return best_batch_size






