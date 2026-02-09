"""
Servicio de Cache Persistente (Legacy)
======================================

Este archivo mantiene compatibilidad hacia atrás.
Nuevo código debe usar services.cache.cache_service.CacheService
"""

from .cache.cache_service import CacheService

__all__ = ["CacheService"]

