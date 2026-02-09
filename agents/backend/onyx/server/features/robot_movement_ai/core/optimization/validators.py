"""
Validators for Trajectory Optimization
======================================
Validation functions for trajectories and trajectory points.
"""

from typing import List, Dict, Any, Optional
import numpy as np


def validate_trajectory(trajectory: List[Dict[str, Any]]) -> bool:
    """Validate a trajectory"""
    if not trajectory:
        return False
    if not isinstance(trajectory, list):
        return False
    return True


def validate_trajectory_point(point: Dict[str, Any]) -> bool:
    """Validate a trajectory point"""
    if not isinstance(point, dict):
        return False
    required_keys = ['x', 'y', 'z']
    return all(key in point for key in required_keys)


def validate_obstacles(obstacles: List[Dict[str, Any]]) -> bool:
    """Validate obstacles list"""
    if obstacles is None:
        return True
    if not isinstance(obstacles, list):
        return False
    return True



