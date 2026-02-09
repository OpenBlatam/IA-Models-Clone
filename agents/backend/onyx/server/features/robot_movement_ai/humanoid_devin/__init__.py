"""
Humanoid Devin Robot - Professional Deep Learning Integration (Optimizado)
===========================================================================

Robot humanoide con control mediante chat natural y integración profesional
de Deep Learning, Transformers y modelos de difusión.

Características:
- Control mediante chat natural
- Modelos Transformer para movimiento coordinado
- Diffusion models para movimientos suaves
- Reinforcement Learning para optimización
- Manejo robusto de errores y validación
- Validaciones completas en todos los módulos
- Utilidades y helpers profesionales
"""

__version__ = "1.0.0"
__author__ = "Blatam Academy"

# Importar driver principal
try:
    from .drivers.humanoid_devin_driver import HumanoidDevinDriver, RobotType
    DRIVER_AVAILABLE = True
except ImportError as e:
    DRIVER_AVAILABLE = False
    HumanoidDevinDriver = None
    RobotType = None

# Importar integraciones core
try:
    from .core.ros2_integration import ROS2Integration, ROS2IntegrationError
    from .core.moveit2_integration import MoveIt2Integration, MoveIt2Error
    from .core.vision_processor import VisionProcessor, VisionError
    from .core.ai_models import AIModelManager, TensorFlowModel, PyTorchModel, ModelError
    from .core.point_cloud_processor import PointCloudProcessor, PointCloudError
    from .core.nav2_integration import Nav2Integration, Nav2Error
    from .core.poppy_icub_integration import PoppyIntegration, ICubIntegration, PoppyError, ICubError
    CORE_AVAILABLE = True
except ImportError as e:
    CORE_AVAILABLE = False
    ROS2Integration = None
    MoveIt2Integration = None
    VisionProcessor = None
    AIModelManager = None
    PointCloudProcessor = None
    Nav2Integration = None
    PoppyIntegration = None
    ICubIntegration = None

# Importar excepciones
from .exceptions import (
    HumanoidRobotError,
    RobotConnectionError,
    RobotControlError,
    RobotConfigurationError,
    TrajectoryError,
    ValidationError
)

# Importar utilidades
from .utils import (
    normalize_quaternion,
    quaternion_to_euler,
    euler_to_quaternion,
    clamp,
    normalize_angle,
    validate_joint_positions,
    interpolate_joint_positions,
    calculate_distance,
    smooth_trajectory,
    validate_pose,
    get_joint_velocity
)

# Importar helpers
try:
    from .helpers.performance_monitor import PerformanceMonitor, OperationMetrics
    from .helpers.safety_monitor import SafetyMonitor, SafetyError
    from .helpers.trajectory_planner import TrajectoryPlanner, TrajectoryPlannerError
    from .helpers.motion_sequencer import MotionSequencer, MotionStep, MotionType, MotionSequencerError
    from .helpers.gesture_library import GestureLibrary
    from .helpers.calibration_manager import CalibrationManager, CalibrationError
    from .helpers.diagnostics import SystemDiagnostics, DiagnosticsError
    from .helpers.adaptive_learning import AdaptiveLearningSystem, AdaptiveLearningError
    from .helpers.error_recovery import ErrorRecoverySystem, RecoveryStrategy, ErrorRecoveryError
    from .helpers.energy_optimizer import EnergyOptimizer, EnergyOptimizerError
    from .helpers.telemetry import TelemetrySystem, TelemetryError
    from .helpers.predictive_planner import PredictivePlanner, PredictivePlannerError
    from .helpers.analytics import AnalyticsSystem, AnalyticsError
    from .helpers.voice_control import VoiceControlSystem, VoiceCommandType, VoiceControlError
    HELPERS_AVAILABLE = True
except ImportError:
    HELPERS_AVAILABLE = False
    PerformanceMonitor = None
    OperationMetrics = None
    SafetyMonitor = None
    SafetyError = None
    TrajectoryPlanner = None
    TrajectoryPlannerError = None
    MotionSequencer = None
    MotionStep = None
    MotionType = None
    MotionSequencerError = None
    GestureLibrary = None
    CalibrationManager = None
    CalibrationError = None
    SystemDiagnostics = None
    DiagnosticsError = None
    AdaptiveLearningSystem = None
    AdaptiveLearningError = None
    ErrorRecoverySystem = None
    RecoveryStrategy = None
    ErrorRecoveryError = None
    EnergyOptimizer = None
    EnergyOptimizerError = None
    TelemetrySystem = None
    TelemetryError = None
    PredictivePlanner = None
    PredictivePlannerError = None
    AnalyticsSystem = None
    AnalyticsError = None
    VoiceControlSystem = None
    VoiceCommandType = None
    VoiceControlError = None

# Importar configuración
try:
    from .config.config_loader import ConfigLoader, load_config, ConfigError
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    ConfigLoader = None
    load_config = None
    ConfigError = None

# Intentar importar API si está disponible
try:
    from .api.humanoid_api import create_humanoid_app, HumanoidAPI
    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False
    create_humanoid_app = None
    HumanoidAPI = None

__all__ = [
    # Versión
    "__version__",
    "__author__",
    # Driver principal
    "HumanoidDevinDriver",
    "RobotType",
    "DRIVER_AVAILABLE",
    # Integraciones core
    "ROS2Integration",
    "MoveIt2Integration",
    "VisionProcessor",
    "AIModelManager",
    "TensorFlowModel",
    "PyTorchModel",
    "PointCloudProcessor",
    "Nav2Integration",
    "PoppyIntegration",
    "ICubIntegration",
    "CORE_AVAILABLE",
    # Excepciones
    "HumanoidRobotError",
    "RobotConnectionError",
    "RobotControlError",
    "RobotConfigurationError",
    "ROS2IntegrationError",
    "MoveIt2Error",
    "Nav2Error",
    "VisionError",
    "PointCloudError",
    "ModelError",
    "PoppyError",
    "ICubError",
    "TrajectoryError",
    "ValidationError",
    # Utilidades
    "normalize_quaternion",
    "quaternion_to_euler",
    "euler_to_quaternion",
    "clamp",
    "normalize_angle",
    "validate_joint_positions",
    "interpolate_joint_positions",
    "calculate_distance",
    "smooth_trajectory",
    "validate_pose",
    "get_joint_velocity",
    # Helpers
    "PerformanceMonitor",
    "OperationMetrics",
    "SafetyMonitor",
    "SafetyError",
    "TrajectoryPlanner",
    "TrajectoryPlannerError",
    "MotionSequencer",
    "MotionStep",
    "MotionType",
    "MotionSequencerError",
    "GestureLibrary",
    "CalibrationManager",
    "CalibrationError",
    "SystemDiagnostics",
    "DiagnosticsError",
    "AdaptiveLearningSystem",
    "AdaptiveLearningError",
    "ErrorRecoverySystem",
    "RecoveryStrategy",
    "ErrorRecoveryError",
    "EnergyOptimizer",
    "EnergyOptimizerError",
    "TelemetrySystem",
    "TelemetryError",
    "PredictivePlanner",
    "PredictivePlannerError",
    "AnalyticsSystem",
    "AnalyticsError",
    "VoiceControlSystem",
    "VoiceCommandType",
    "VoiceControlError",
    "HELPERS_AVAILABLE",
    # Configuración
    "ConfigLoader",
    "load_config",
    "ConfigError",
    "CONFIG_AVAILABLE",
    # API (opcional)
    "create_humanoid_app",
    "HumanoidAPI",
    "API_AVAILABLE",
]
