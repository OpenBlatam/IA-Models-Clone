"""
Quaternion utilities
"""

import numpy as np
from typing import Tuple


def quaternion_slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two quaternions"""
    # Normalize quaternions
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    
    # Calculate dot product
    dot = np.dot(q1, q2)
    
    # If dot product is negative, negate one quaternion
    if dot < 0.0:
        q2 = -q2
        dot = -dot
    
    # If quaternions are very close, use linear interpolation
    if abs(dot) > 0.9995:
        return q1 + t * (q2 - q1)
    
    # Calculate angle
    theta = np.arccos(abs(dot))
    sin_theta = np.sin(theta)
    
    # Spherical interpolation
    w1 = np.sin((1 - t) * theta) / sin_theta
    w2 = np.sin(t * theta) / sin_theta
    
    return w1 * q1 + w2 * q2



