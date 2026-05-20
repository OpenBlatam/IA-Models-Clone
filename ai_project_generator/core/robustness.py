"""
Robustness Module - Módulo de robustez
======================================

Módulo que integra todas las características de robustez.
"""

from .robust_service import RobustService
from .robust_repository import RobustRepository
from .health_checker import RobustHealthChecker, get_health_checker
from .data_validator import RobustValidator, ProjectDataModel
from .dependency_validator import DependencyValidator, get_dependency_validator
from .fallback_manager import FallbackManager, get_fallback_manager
from .timeout_manager import TimeoutManager, get_timeout_manager

__all__ = [
    "RobustService",
    "RobustRepository",
    "RobustHealthChecker",
    "get_health_checker",
    "RobustValidator",
    "ProjectDataModel",
    "DependencyValidator",
    "get_dependency_validator",
    "FallbackManager",
    "get_fallback_manager",
    "TimeoutManager",
    "get_timeout_manager",
]















