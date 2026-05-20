"""
Servicio de Búsqueda Semántica (Legacy)
=======================================

Este archivo mantiene compatibilidad hacia atrás.
Nuevo código debe usar services.semantic.semantic_search_service.SemanticSearchService
"""

from .semantic.semantic_search_service import SemanticSearchService

__all__ = ["SemanticSearchService"]
