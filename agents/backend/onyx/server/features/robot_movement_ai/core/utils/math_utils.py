"""
Math Utilities
==============

Utilidades matemáticas generales.
"""

import numpy as np
from typing import List, Tuple, Optional


def point_in_obstacle(point: np.ndarray, obstacle: np.ndarray) -> bool:
    """
    Verificar si un punto está dentro de un obstáculo.
    
    Args:
        point: Punto 3D [x, y, z]
        obstacle: Bounding box [min_x, min_y, min_z, max_x, max_y, max_z]
        
    Returns:
        True si el punto está dentro del obstáculo
    """
    min_corner = obstacle[:3]
    max_corner = obstacle[3:]
    return np.all(point >= min_corner) and np.all(point <= max_corner)


def calculate_distance_to_obstacle(
    point: np.ndarray,
    obstacle: np.ndarray
) -> float:
    """
    Calcular distancia desde un punto a un obstáculo.
    
    Args:
        point: Punto 3D
        obstacle: Bounding box
        
    Returns:
        Distancia al obstáculo
    """
    obstacle_center = (obstacle[:3] + obstacle[3:]) / 2
    return np.linalg.norm(point - obstacle_center)


def normalize_vector(vector: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """
    Normalizar vector.
    
    Args:
        vector: Vector a normalizar
        epsilon: Valor pequeño para evitar división por cero
        
    Returns:
        Vector normalizado
    """
    norm = np.linalg.norm(vector)
    if norm < epsilon:
        return vector
    return vector / norm


def clamp_value(value: float, min_val: float, max_val: float) -> float:
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


def calculate_curvature(
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray
) -> float:
    """
    Calcular curvatura entre tres puntos.
    
    Args:
        p1: Punto anterior
        p2: Punto actual
        p3: Punto siguiente
        
    Returns:
        Curvatura
    """
    v1 = p2 - p1
    v2 = p3 - p2
    
    cross = np.cross(v1, v2)
    curvature = np.linalg.norm(cross) / (
        np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6
    )
    
    return curvature






