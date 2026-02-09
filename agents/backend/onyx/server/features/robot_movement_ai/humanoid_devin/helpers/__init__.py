"""
Helpers for Humanoid Devin Robot (Optimizado)
==============================================

Helpers y utilidades adicionales para el robot humanoide.
"""

from .performance_monitor import PerformanceMonitor, OperationMetrics
from .safety_monitor import SafetyMonitor, SafetyError
from .trajectory_planner import TrajectoryPlanner, TrajectoryPlannerError
from .motion_sequencer import MotionSequencer, MotionStep, MotionType, MotionSequencerError
from .gesture_library import GestureLibrary
from .calibration_manager import CalibrationManager, CalibrationError
from .diagnostics import SystemDiagnostics, DiagnosticsError
from .adaptive_learning import AdaptiveLearningSystem, AdaptiveLearningError
from .error_recovery import ErrorRecoverySystem, RecoveryStrategy, ErrorRecoveryError
from .energy_optimizer import EnergyOptimizer, EnergyOptimizerError
from .telemetry import TelemetrySystem, TelemetryError
from .predictive_planner import PredictivePlanner, PredictivePlannerError
from .analytics import AnalyticsSystem, AnalyticsError
from .voice_control import VoiceControlSystem, VoiceCommandType, VoiceControlError

__all__ = [
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
    "VoiceControlError"
]

