"""
Robot Movement AI - Plataforma IA de Movimiento Robótico
==========================================================

Sistema avanzado de IA para control y optimización de movimiento robótico
mediante chat, con soporte para múltiples marcas y protocolos.

Características principales:
- Algoritmos de reinforcement learning para optimización de trayectorias
- Redes neuronales convolucionales para procesamiento visual
- Modelos predictivos para cinemática inversa
- Sistema de feedback en tiempo real a 1000Hz
- Integración con ROS (Robot Operating System)
- APIs RESTful para integración
- SDK para Python, C++, y MATLAB
- Soporte para KUKA, ABB, Fanuc, Universal Robots
"""

__version__ = "1.0.0"
__author__ = "Blatam Academy"
__license__ = "Proprietary"
__description__ = "Advanced AI platform for robotic movement control and optimization via chat, with support for multiple brands and protocols"

# Try to import components with error handling and fallbacks
try:
    from .core.robot.movement_engine import RobotMovementEngine
except ImportError:
    RobotMovementEngine = None

try:
    from .core.robot.trajectory_optimizer import TrajectoryOptimizer
except ImportError:
    try:
        from .core.optimization.trajectory_optimizer import TrajectoryOptimizer
    except ImportError:
        TrajectoryOptimizer = None

try:
    from .core.robot.inverse_kinematics import InverseKinematicsSolver
except ImportError:
    try:
        from .core.inverse_kinematics import InverseKinematicsSolver
    except ImportError:
        InverseKinematicsSolver = None

try:
    from .core.robot.visual_processor import VisualProcessor
except ImportError:
    try:
        from .core.visual_processor import VisualProcessor
    except ImportError:
        VisualProcessor = None

try:
    from .core.robot.real_time_feedback import RealTimeFeedbackSystem
except ImportError:
    try:
        from .core.real_time_feedback import RealTimeFeedbackSystem
    except ImportError:
        RealTimeFeedbackSystem = None

try:
    from .api.robot_api import create_robot_app
except ImportError:
    create_robot_app = None

try:
    from .chat.chat_controller import ChatRobotController
except ImportError:
    ChatRobotController = None

__all__ = [
    "RobotMovementEngine",
    "TrajectoryOptimizer",
    "InverseKinematicsSolver",
    "VisualProcessor",
    "RealTimeFeedbackSystem",
    "create_robot_app",
    "ChatRobotController",
]

