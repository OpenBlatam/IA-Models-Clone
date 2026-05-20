"""
Servicio de Analytics Avanzado (Legacy)
========================================

Este archivo mantiene compatibilidad hacia atrás.
Nuevo código debe usar services.analytics.analytics_service.AnalyticsService
"""

from .analytics.analytics_service import AnalyticsService

__all__ = ["AnalyticsService"]

