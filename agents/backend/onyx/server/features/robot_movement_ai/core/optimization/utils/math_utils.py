"""
Math utilities for trajectory optimization
"""

import numpy as np
from typing import List, Dict, Any


def point_in_obstacle(point: np.ndarray, obstacle: Dict[str, Any]) -> bool:
    """Check if a point is inside an obstacle"""
    x, y, z = point[0], point[1], point[2]
    
    return (
        obstacle.get('min_x', -np.inf) <= x <= obstacle.get('max_x', np.inf) and
        obstacle.get('min_y', -np.inf) <= y <= obstacle.get('max_y', np.inf) and
        obstacle.get('min_z', -np.inf) <= z <= obstacle.get('max_z', np.inf)
    )


def normalize_vector(v: np.ndarray) -> np.ndarray:
    """Normalize a vector"""
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm



