"""
Helper Functions
================

Funciones helper útiles para el sistema de movimiento robótico.
Optimizado con numba para máximo rendimiento.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import logging

try:
    from ..performance import (
        euclidean_distance_fast,
        normalize_vector_fast,
        quaternion_multiply_fast
    )
    USE_PERFORMANCE_UTILS = True
except ImportError:
    USE_PERFORMANCE_UTILS = False

logger = logging.getLogger(__name__)


def clamp(value: float, min_val: float, max_val: float) -> float:
    """
    Limitar valor entre mínimo y máximo.
    
    Args:
        value: Valor a limitar
        min_val: Valor mínimo
        max_val: Valor máximo
        
    Returns:
        Valor limitado
    """
    return max(min_val, min(value, max_val))


def lerp(start: float, end: float, t: float) -> float:
    """
    Interpolación lineal.
    
    Args:
        start: Valor inicial
        end: Valor final
        t: Factor de interpolación [0, 1]
        
    Returns:
        Valor interpolado
    """
    return start + t * (end - start)


def normalize_angle(angle: float) -> float:
    """
    Normalizar ángulo a rango [-π, π].
    
    Args:
        angle: Ángulo en radianes
        
    Returns:
        Ángulo normalizado
    """
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle


def angle_difference(angle1: float, angle2: float) -> float:
    """
    Calcular diferencia entre dos ángulos.
    
    Args:
        angle1: Primer ángulo
        angle2: Segundo ángulo
        
    Returns:
        Diferencia normalizada [-π, π]
    """
    diff = angle2 - angle1
    return normalize_angle(diff)


def euclidean_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """
    Calcular distancia euclidiana entre dos puntos.
    Usa versión optimizada con numba si está disponible.
    
    Args:
        p1: Primer punto
        p2: Segundo punto
        
    Returns:
        Distancia
    """
    if USE_PERFORMANCE_UTILS:
        return euclidean_distance_fast(p1, p2)
    return np.linalg.norm(p2 - p1)


def manhattan_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """
    Calcular distancia Manhattan entre dos puntos.
    
    Args:
        p1: Primer punto
        p2: Segundo punto
        
    Returns:
        Distancia Manhattan
    """
    return np.sum(np.abs(p2 - p1))


def is_close(a: float, b: float, rel_tol: float = 1e-9, abs_tol: float = 0.0) -> bool:
    """
    Verificar si dos valores están cerca (similar a math.isclose).
    
    Args:
        a: Primer valor
        b: Segundo valor
        rel_tol: Tolerancia relativa
        abs_tol: Tolerancia absoluta
        
    Returns:
        True si están cerca
    """
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


def smooth_step(t: float) -> float:
    """
    Función smooth step (S-curve).
    
    Args:
        t: Valor [0, 1]
        
    Returns:
        Valor suavizado [0, 1]
    """
    t = clamp(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def ease_in_out(t: float) -> float:
    """
    Función ease-in-out.
    
    Args:
        t: Valor [0, 1]
        
    Returns:
        Valor con easing [0, 1]
    """
    t = clamp(t, 0.0, 1.0)
    if t < 0.5:
        return 2 * t * t
    else:
        return -1 + (4 - 2 * t) * t


def create_grid(
    min_bounds: np.ndarray,
    max_bounds: np.ndarray,
    resolution: float
) -> List[np.ndarray]:
    """
    Crear grid de puntos.
    
    Args:
        min_bounds: Límites mínimos [x, y, z]
        max_bounds: Límites máximos [x, y, z]
        resolution: Resolución del grid
        
    Returns:
        Lista de puntos del grid
    """
    x_coords = np.arange(min_bounds[0], max_bounds[0] + resolution, resolution)
    y_coords = np.arange(min_bounds[1], max_bounds[1] + resolution, resolution)
    z_coords = np.arange(min_bounds[2], max_bounds[2] + resolution, resolution)
    
    points = []
    for x in x_coords:
        for y in y_coords:
            for z in z_coords:
                points.append(np.array([x, y, z]))
    
    return points


def find_nearest_point(target: np.ndarray, points: List[np.ndarray]) -> Tuple[int, float]:
    """
    Encontrar punto más cercano a target.
    
    Args:
        target: Punto objetivo
        points: Lista de puntos
        
    Returns:
        Tupla (índice, distancia)
    """
    if not points:
        return -1, float('inf')
    
    distances = [euclidean_distance(target, p) for p in points]
    min_idx = np.argmin(distances)
    return min_idx, distances[min_idx]


def resample_trajectory(
    trajectory: List[Any],
    num_points: int,
    method: str = "linear"
) -> List[Any]:
    """
    Re-muestrear trayectoria a número específico de puntos.
    
    Args:
        trajectory: Trayectoria original
        num_points: Número de puntos deseado
        method: Método de interpolación ("linear", "cubic")
        
    Returns:
        Trayectoria re-muestreada
    """
    if len(trajectory) <= 1:
        return trajectory
    
    if len(trajectory) == num_points:
        return trajectory
    
    # Interpolación lineal simple
    indices = np.linspace(0, len(trajectory) - 1, num_points)
    resampled = []
    
    for idx in indices:
        i = int(idx)
        t = idx - i
        
        if i >= len(trajectory) - 1:
            resampled.append(trajectory[-1])
        else:
            # Interpolación entre puntos
            p1 = trajectory[i]
            p2 = trajectory[i + 1]
            
            if hasattr(p1, 'position') and hasattr(p2, 'position'):
                # TrajectoryPoint
                from .trajectory_optimizer import TrajectoryPoint
                from .utils.quaternion_utils import quaternion_slerp
                
                new_pos = lerp(p1.position, p2.position, t)
                new_orient = quaternion_slerp(p1.orientation, p2.orientation, t)
                
                resampled.append(TrajectoryPoint(
                    position=new_pos,
                    orientation=new_orient,
                    velocity=p1.velocity,
                    acceleration=p1.acceleration,
                    timestamp=lerp(p1.timestamp, p2.timestamp, t)
                ))
            else:
                # Puntos simples
                resampled.append(lerp(p1, p2, t))
    
    return resampled


def format_duration(seconds: float) -> str:
    """
    Formatear duración en formato legible.
    
    Args:
        seconds: Duración en segundos
        
    Returns:
        String formateado (ej: "1h 23m 45s")
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    
    minutes = int(seconds // 60)
    secs = seconds % 60
    
    if minutes < 60:
        return f"{minutes}m {secs:.2f}s"
    
    hours = int(minutes // 60)
    mins = minutes % 60
    
    return f"{hours}h {mins}m {secs:.2f}s"


def format_distance(meters: float) -> str:
    """
    Formatear distancia en formato legible.
    
    Args:
        meters: Distancia en metros
        
    Returns:
        String formateado (ej: "1.23m" o "123cm")
    """
    if meters < 0.01:
        return f"{meters * 1000:.1f}mm"
    elif meters < 1.0:
        return f"{meters * 100:.1f}cm"
    else:
        return f"{meters:.2f}m"


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    División segura (evita división por cero).
    
    Args:
        numerator: Numerador
        denominator: Denominador
        default: Valor por defecto si denominador es cero
        
    Returns:
        Resultado de la división o default
    """
    if abs(denominator) < 1e-9:
        return default
    return numerator / denominator


def calculate_percentile(values: List[float], percentile: float) -> float:
    """
    Calcular percentil de lista de valores.
    
    Args:
        values: Lista de valores
        percentile: Percentil [0, 100]
        
    Returns:
        Valor del percentil
    """
    if not values:
        return 0.0
    
    sorted_values = sorted(values)
    index = int(len(sorted_values) * percentile / 100.0)
    index = min(index, len(sorted_values) - 1)
    return sorted_values[index]






