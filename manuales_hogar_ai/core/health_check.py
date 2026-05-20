"""
Comprehensive Health Check (Legacy)
===================================

Este archivo mantiene compatibilidad hacia atrás.
Nuevo código debe usar core.health.health_checker.HealthChecker
"""

from .health.health_checker import HealthChecker

__all__ = ["HealthChecker"]
