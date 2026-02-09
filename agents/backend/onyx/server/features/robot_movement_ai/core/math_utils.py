"""
Math Utilities
==============

Utilidades matemáticas optimizadas para operaciones comunes.
"""

import numpy as np
from typing import Tuple, Optional, List
from functools import lru_cache


@lru_cache(maxsize=128)
def normalize_quaternion(quaternion: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    """
    Normalizar quaternion (versión cacheada).
    
    Args:
        quaternion: Quaternion como tupla (qx, qy, qz, qw)
    
    Returns:
        Quaternion normalizado
    """
    q = np.array(quaternion)
    norm = np.linalg.norm(q)
    if norm < 1e-10:
        return (0.0, 0.0, 0.0, 1.0)
    normalized = q / norm
    return tuple(normalized.tolist())


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Multiplicar dos quaternions.
    
    Args:
        q1: Primer quaternion [qx, qy, qz, qw]
        q2: Segundo quaternion [qx, qy, qz, qw]
    
    Returns:
        Quaternion resultante
    """
    q1 = np.asarray(q1)
    q2 = np.asarray(q2)
    
    w1, x1, y1, z1 = q1[3], q1[0], q1[1], q1[2]
    w2, x2, y2, z2 = q2[3], q2[0], q2[1], q2[2]
    
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ])


def quaternion_to_euler(quaternion: np.ndarray) -> Tuple[float, float, float]:
    """
    Convertir quaternion a ángulos de Euler (roll, pitch, yaw).
    
    Args:
        quaternion: Quaternion [qx, qy, qz, qw]
    
    Returns:
        Tupla (roll, pitch, yaw) en radianes
    """
    q = np.asarray(quaternion)
    if len(q) == 4:
        x, y, z, w = q[0], q[1], q[2], q[3]
    else:
        raise ValueError("Quaternion must have 4 elements")
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return (float(roll), float(pitch), float(yaw))


def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Convertir ángulos de Euler a quaternion.
    
    Args:
        roll: Rotación alrededor del eje x (radianes)
        pitch: Rotación alrededor del eje y (radianes)
        yaw: Rotación alrededor del eje z (radianes)
    
    Returns:
        Quaternion [qx, qy, qz, qw]
    """
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    
    return np.array([
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy
    ])


def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Calcular ángulo entre dos vectores.
    
    Args:
        v1: Primer vector
        v2: Segundo vector
    
    Returns:
        Ángulo en radianes
    """
    v1 = np.asarray(v1)
    v2 = np.asarray(v2)
    
    dot = np.dot(v1, v2)
    norms = np.linalg.norm(v1) * np.linalg.norm(v2)
    
    if norms < 1e-10:
        return 0.0
    
    cos_angle = np.clip(dot / norms, -1.0, 1.0)
    return float(np.arccos(cos_angle))


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
    return float(np.clip(value, min_val, max_val))


def lerp(a: float, b: float, t: float) -> float:
    """
    Interpolación lineal.
    
    Args:
        a: Valor inicial
        b: Valor final
        t: Factor de interpolación [0, 1]
    
    Returns:
        Valor interpolado
    """
    t = clamp(t, 0.0, 1.0)
    return float(a + (b - a) * t)


def smooth_step(t: float) -> float:
    """
    Función smooth step (interpolación suave).
    
    Args:
        t: Factor [0, 1]
    
    Returns:
        Valor suavizado
    """
    t = clamp(t, 0.0, 1.0)
    return float(t * t * (3.0 - 2.0 * t))

