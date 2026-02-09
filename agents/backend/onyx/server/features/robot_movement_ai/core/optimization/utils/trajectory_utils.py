"""
Trajectory utilities
"""

import numpy as np
from typing import List, Dict, Any, Tuple


def interpolate_trajectory(trajectory: List[Dict[str, Any]], num_points: int) -> List[Dict[str, Any]]:
    """Interpolate trajectory to have specified number of points"""
    if not trajectory or num_points <= 0:
        return trajectory
    
    if len(trajectory) == num_points:
        return trajectory
    
    # Linear interpolation
    result = []
    for i in range(num_points):
        t = i / (num_points - 1) if num_points > 1 else 0
        idx = int(t * (len(trajectory) - 1))
        if idx >= len(trajectory) - 1:
            result.append(trajectory[-1])
        else:
            p1 = trajectory[idx]
            p2 = trajectory[idx + 1]
            alpha = t * (len(trajectory) - 1) - idx
            interpolated = {
                'x': p1.get('x', 0) * (1 - alpha) + p2.get('x', 0) * alpha,
                'y': p1.get('y', 0) * (1 - alpha) + p2.get('y', 0) * alpha,
                'z': p1.get('z', 0) * (1 - alpha) + p2.get('z', 0) * alpha,
            }
            result.append(interpolated)
    
    return result


def smooth_trajectory(trajectory: List[Dict[str, Any]], window_size: int = 3) -> List[Dict[str, Any]]:
    """Smooth trajectory using moving average"""
    if len(trajectory) <= window_size:
        return trajectory
    
    smoothed = []
    for i in range(len(trajectory)):
        start = max(0, i - window_size // 2)
        end = min(len(trajectory), i + window_size // 2 + 1)
        window = trajectory[start:end]
        
        avg_x = sum(p.get('x', 0) for p in window) / len(window)
        avg_y = sum(p.get('y', 0) for p in window) / len(window)
        avg_z = sum(p.get('z', 0) for p in window) / len(window)
        
        smoothed.append({
            'x': avg_x,
            'y': avg_y,
            'z': avg_z,
        })
    
    return smoothed


def calculate_trajectory_length(trajectory: List[Dict[str, Any]]) -> float:
    """Calculate total length of trajectory"""
    if len(trajectory) < 2:
        return 0.0
    
    total_length = 0.0
    for i in range(len(trajectory) - 1):
        p1 = trajectory[i]
        p2 = trajectory[i + 1]
        
        dx = p2.get('x', 0) - p1.get('x', 0)
        dy = p2.get('y', 0) - p1.get('y', 0)
        dz = p2.get('z', 0) - p1.get('z', 0)
        
        total_length += np.sqrt(dx*dx + dy*dy + dz*dz)
    
    return total_length


def calculate_trajectory_distance(trajectory: List[Dict[str, Any]]) -> float:
    """Calculate total distance of trajectory (alias for length)"""
    return calculate_trajectory_length(trajectory)


def calculate_trajectory_curvature(trajectory: List[Dict[str, Any]]) -> float:
    """Calculate average curvature of trajectory"""
    if len(trajectory) < 3:
        return 0.0
    
    curvatures = []
    for i in range(1, len(trajectory) - 1):
        p0 = np.array([trajectory[i-1].get('x', 0), trajectory[i-1].get('y', 0), trajectory[i-1].get('z', 0)])
        p1 = np.array([trajectory[i].get('x', 0), trajectory[i].get('y', 0), trajectory[i].get('z', 0)])
        p2 = np.array([trajectory[i+1].get('x', 0), trajectory[i+1].get('y', 0), trajectory[i+1].get('z', 0)])
        
        v1 = p1 - p0
        v2 = p2 - p1
        
        # Calculate curvature using cross product
        cross = np.cross(v1, v2)
        curvature = np.linalg.norm(cross) / (np.linalg.norm(v1) ** 3 + 1e-10)
        curvatures.append(curvature)
    
    return sum(curvatures) / len(curvatures) if curvatures else 0.0


def validate_trajectory_continuity(trajectory: List[Dict[str, Any]], max_jump: float = 0.5) -> bool:
    """Validate that trajectory is continuous (no large jumps)"""
    if len(trajectory) < 2:
        return True
    
    for i in range(len(trajectory) - 1):
        p1 = trajectory[i]
        p2 = trajectory[i + 1]
        
        dx = p2.get('x', 0) - p1.get('x', 0)
        dy = p2.get('y', 0) - p1.get('y', 0)
        dz = p2.get('z', 0) - p1.get('z', 0)
        
        distance = np.sqrt(dx*dx + dy*dy + dz*dz)
        if distance > max_jump:
            return False
    
    return True

