"""Utility modules for Robot Movement AI."""

from .quaternion_utils import quaternion_slerp, quaternion_to_rotation_matrix, rotation_matrix_to_quaternion
from .trajectory_utils import (
    calculate_trajectory_distance,
    calculate_trajectory_curvature,
    smooth_trajectory,
    validate_trajectory_continuity
)
from .math_utils import (
    point_in_obstacle,
    calculate_distance_to_obstacle,
    normalize_vector,
    clamp_value
)

__all__ = [
    # Quaternion
    "quaternion_slerp",
    "quaternion_to_rotation_matrix",
    "rotation_matrix_to_quaternion",
    # Trajectory
    "calculate_trajectory_distance",
    "calculate_trajectory_curvature",
    "smooth_trajectory",
    "validate_trajectory_continuity",
    # Math
    "point_in_obstacle",
    "calculate_distance_to_obstacle",
    "normalize_vector",
    "clamp_value",
]






