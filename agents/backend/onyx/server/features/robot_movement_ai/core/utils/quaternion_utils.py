"""
Quaternion Utilities
====================

Utilidades para trabajar con quaterniones.
"""

import numpy as np
from typing import Tuple


def quaternion_slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
    """
    Spherical linear interpolation para quaterniones.
    
    Args:
        q1: Quaternion inicial
        q2: Quaternion final
        t: Interpolación factor [0, 1]
        
    Returns:
        Quaternion interpolado
    """
    # Normalizar quaterniones
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    
    dot = np.clip(np.dot(q1, q2), -1.0, 1.0)
    
    if abs(dot) > 0.9995:
        # Quaterniones muy cercanos, usar interpolación lineal
        return q1 + t * (q2 - q1)
    
    theta = np.arccos(abs(dot))
    sin_theta = np.sin(theta)
    
    if sin_theta < 1e-6:
        return q1
    
    w1 = np.sin((1 - t) * theta) / sin_theta
    w2 = np.sin(t * theta) / sin_theta
    
    if dot < 0:
        q2 = -q2
    
    return w1 * q1 + w2 * q2


def quaternion_to_rotation_matrix(quat: np.ndarray) -> np.ndarray:
    """
    Convertir quaternion a matriz de rotación.
    
    Args:
        quat: Quaternion [qx, qy, qz, qw]
        
    Returns:
        Matriz de rotación 3x3
    """
    qx, qy, qz, qw = quat
    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
    ])
    return R


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """
    Convertir matriz de rotación a quaternion.
    
    Args:
        R: Matriz de rotación 3x3
        
    Returns:
        Quaternion [qx, qy, qz, qw]
    """
    trace = np.trace(R)
    
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s
    
    return np.array([qx, qy, qz, qw])






