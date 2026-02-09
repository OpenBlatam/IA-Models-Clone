"""
Constants - Constantes del sistema
===================================

Constantes compartidas utilizadas en todo el sistema.
"""

from enum import Enum
from typing import Final

# ============================================================================
# Algoritmos de Optimización
# ============================================================================

class OptimizationAlgorithm(Enum):
    """Algoritmos de optimización disponibles."""
    PPO = "ppo"
    DQN = "dqn"
    SAC = "sac"
    TD3 = "td3"
    A_STAR = "astar"
    RRT = "rrt"
    MULTI_OBJECTIVE = "multi_objective"
    HEURISTIC = "heuristic"


# ============================================================================
# Constantes de Performance
# ============================================================================

# Feedback System
DEFAULT_FEEDBACK_FREQUENCY: Final[int] = 1000  # Hz
MAX_FEEDBACK_FREQUENCY: Final[int] = 2000  # Hz
FEEDBACK_BUFFER_SIZE: Final[int] = 10000

# Trajectory Optimization
DEFAULT_MAX_ITERATIONS: Final[int] = 100
DEFAULT_CONVERGENCE_THRESHOLD: Final[float] = 1e-6
DEFAULT_CACHE_SIZE: Final[int] = 1000
CACHE_CLEANUP_THRESHOLD: Final[int] = 1200
CACHE_CLEANUP_PERCENTAGE: Final[float] = 0.2  # 20%

# Learning Parameters
DEFAULT_LEARNING_RATE: Final[float] = 0.001
DEFAULT_DISCOUNT_FACTOR: Final[float] = 0.95
DEFAULT_EPSILON: Final[float] = 0.1
DEFAULT_EPSILON_DECAY: Final[float] = 0.995
MIN_EPSILON: Final[float] = 0.01

# ============================================================================
# Constantes de Seguridad
# ============================================================================

# Velocidades y Aceleraciones
DEFAULT_MAX_VELOCITY: Final[float] = 1.0  # m/s
DEFAULT_MAX_ACCELERATION: Final[float] = 2.0  # m/s²
DEFAULT_MAX_JOINT_VELOCITY: Final[float] = 1.57  # rad/s

# Distancias de Seguridad
MIN_OBSTACLE_DISTANCE: Final[float] = 0.1  # 10cm
SAFE_OBSTACLE_DISTANCE: Final[float] = 0.2  # 20cm
REPLANIFICATION_THRESHOLD: Final[float] = 0.1  # 10cm

# Límites de Articulaciones
MAX_JOINT_ANGLE: Final[float] = 3.14  # rad (180°)
MAX_JOINT_TORQUE: Final[float] = 100.0  # Nm
MAX_TEMPERATURE: Final[float] = 60.0  # °C

# ============================================================================
# Constantes de Trayectoria
# ============================================================================

# Generación de Trayectoria
DEFAULT_TRAJECTORY_POINTS: Final[int] = 50
DEFAULT_TIMESTEP: Final[float] = 0.01  # 100 Hz
TRAJECTORY_UPDATE_RATE: Final[int] = 100  # Hz

# Validación
MAX_JUMP_DISTANCE: Final[float] = 0.5  # 50cm
MAX_ACCELERATION_MAGNITUDE: Final[float] = 2.0  # m/s²

# ============================================================================
# Constantes de Algoritmos
# ============================================================================

# A* Algorithm
DEFAULT_GRID_RESOLUTION: Final[float] = 0.05  # 5cm
A_STAR_GOAL_TOLERANCE: Final[int] = 1  # grid cells

# RRT Algorithm
DEFAULT_RRT_MAX_ITERATIONS: Final[int] = 1000
DEFAULT_RRT_STEP_SIZE: Final[float] = 0.1  # 10cm
RRT_GOAL_BIAS: Final[float] = 0.1  # 10% hacia el objetivo

# Multi-Objective
DEFAULT_OBJECTIVE_WEIGHTS: Final[dict] = {
    "time": 0.25,
    "energy": 0.25,
    "smoothness": 0.25,
    "safety": 0.25
}
MULTI_OBJECTIVE_CANDIDATES: Final[int] = 10

# PPO Algorithm
PPO_CLIP_RATIO: Final[float] = 0.2

# ============================================================================
# Constantes de Visual Processing
# ============================================================================

DEFAULT_CAMERA_RESOLUTION: Final[tuple] = (1920, 1080)
DEFAULT_CONFIDENCE_THRESHOLD: Final[float] = 0.5
VISUAL_PROCESSING_RATE: Final[int] = 30  # FPS

# ============================================================================
# Constantes de Historial
# ============================================================================

MAX_TRAJECTORY_HISTORY: Final[int] = 1000
MAX_MOVEMENT_HISTORY: Final[int] = 100
MAX_CONVERSATION_HISTORY: Final[int] = 10
EXPERIENCE_BUFFER_SIZE: Final[int] = 10000
PERFORMANCE_WINDOW_SIZE: Final[int] = 100

# ============================================================================
# Constantes de API
# ============================================================================

DEFAULT_API_PORT: Final[int] = 8010
DEFAULT_API_HOST: Final[str] = "0.0.0.0"
DEFAULT_LOG_LEVEL: Final[str] = "INFO"

# ============================================================================
# Constantes de Exportación
# ============================================================================

SUPPORTED_EXPORT_FORMATS: Final[list] = ["json", "csv"]
DEFAULT_EXPORT_FORMAT: Final[str] = "json"

# ============================================================================
# Mensajes de Error
# ============================================================================

class ErrorMessages:
    """Mensajes de error estándar."""
    NO_IK_SOLUTION = "No valid IK solution found"
    NO_INITIAL_POSE = "No initial pose available"
    TRAJECTORY_EMPTY = "Trajectory is empty"
    TRAJECTORY_COLLISION = "Trajectory point is inside obstacle"
    INVALID_CACHE_KEY = "Invalid cache key"
    EXPORT_FAILED = "Failed to export trajectory"
    IMPORT_FAILED = "Failed to import trajectory"
    INVALID_FORMAT = "Invalid export format"
    ROBOT_NOT_CONNECTED = "Robot not connected"
    MOVEMENT_IN_PROGRESS = "Movement already in progress"






