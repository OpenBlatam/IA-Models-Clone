"""
Constants for Trajectory Optimization
======================================
Default values and constants used in trajectory optimization algorithms.
"""

from enum import Enum

# Algorithm Configuration
DEFAULT_MAX_ITERATIONS = 1000
DEFAULT_CONVERGENCE_THRESHOLD = 1e-6
DEFAULT_CACHE_SIZE = 1000
CACHE_CLEANUP_THRESHOLD = 0.8
CACHE_CLEANUP_PERCENTAGE = 0.3

# Reinforcement Learning Parameters
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_DISCOUNT_FACTOR = 0.99
DEFAULT_EPSILON = 1.0
DEFAULT_EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01

# Trajectory Parameters
DEFAULT_TRAJECTORY_POINTS = 100
DEFAULT_TIMESTEP = 0.01
MAX_TRAJECTORY_HISTORY = 1000

# Experience Replay
EXPERIENCE_BUFFER_SIZE = 10000
PERFORMANCE_WINDOW_SIZE = 100

# RRT Parameters
DEFAULT_RRT_MAX_ITERATIONS = 5000
DEFAULT_RRT_STEP_SIZE = 0.1

# Algorithm Types
class OptimizationAlgorithm(str, Enum):
    """Available optimization algorithms"""
    ASTAR = "astar"
    RRT = "rrt"
    RRT_STAR = "rrt_star"
    PRM = "prm"
    RL = "rl"
    HYBRID = "hybrid"

# Error Messages
class ErrorMessages:
    """Error message constants"""
    INVALID_START = "Invalid start position"
    INVALID_GOAL = "Invalid goal position"
    NO_PATH_FOUND = "No path found"
    COLLISION_DETECTED = "Collision detected"
    TIMEOUT = "Optimization timeout"
    INVALID_ALGORITHM = "Invalid algorithm specified"



