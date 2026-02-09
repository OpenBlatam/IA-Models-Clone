"""
Servicio de Plantillas (Legacy)
================================

Este archivo mantiene compatibilidad hacia atrás.
Nuevo código debe usar services.template.template_service.TemplateService
"""

from .template.template_service import TemplateService

__all__ = ["TemplateService"]

