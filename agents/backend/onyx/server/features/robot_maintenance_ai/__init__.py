"""
Robot Maintenance AI - Sistema de enseñanza de mantenimiento de robots y máquinas
=================================================================================

Utiliza OpenRouter, NLP y ML para proporcionar asistencia inteligente en mantenimiento.
"""

__version__ = "1.0.0"
__author__ = "Blatam Academy"
__license__ = "Proprietary"
__description__ = "AI system for teaching robot and machine maintenance using OpenRouter, NLP and ML"

# Try to import components with error handling
try:
    from .core.maintenance_tutor import RobotMaintenanceTutor
except ImportError:
    RobotMaintenanceTutor = None

try:
    from .config.maintenance_config import MaintenanceConfig, OpenRouterConfig
except ImportError:
    MaintenanceConfig = None
    OpenRouterConfig = None

try:
    from .api.router import router
except ImportError:
    router = None

__all__ = [
    "RobotMaintenanceTutor",
    "MaintenanceConfig",
    "OpenRouterConfig",
    "router",
]
