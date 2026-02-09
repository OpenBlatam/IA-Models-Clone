"""
Robot Maintenance Teaching AI
============================

Sistema de IA para enseñar mantenimiento de robots y máquinas usando OpenRouter,
NLP y Machine Learning con las mejores librerías disponibles.
"""

__version__ = "1.0.0"
__author__ = "Blatam Academy"
__license__ = "Proprietary"
__description__ = "AI system for teaching robot and machine maintenance using OpenRouter, NLP and Machine Learning"

# Try to import components with error handling
try:
    from .core.maintenance_tutor import RobotMaintenanceTutor
except ImportError:
    RobotMaintenanceTutor = None

try:
    from .config.maintenance_config import MaintenanceConfig, OpenRouterConfig, MLConfig, NLPConfig
except ImportError:
    MaintenanceConfig = None
    OpenRouterConfig = None
    MLConfig = None
    NLPConfig = None

__all__ = [
    "RobotMaintenanceTutor",
    "MaintenanceConfig",
    "OpenRouterConfig",
    "MLConfig",
    "NLPConfig",
]
