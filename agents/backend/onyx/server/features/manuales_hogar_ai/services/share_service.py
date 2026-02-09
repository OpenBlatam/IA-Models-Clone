"""
Servicio de Compartir Manuales (Legacy)
========================================

Este archivo mantiene compatibilidad hacia atrás.
Nuevo código debe usar services.share.share_service.ShareService
"""

from .share.share_service import ShareService

__all__ = ["ShareService"]

