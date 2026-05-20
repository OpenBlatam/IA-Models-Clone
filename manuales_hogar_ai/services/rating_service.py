"""
Servicio de Ratings y Favoritos (Legacy)
========================================

Este archivo mantiene compatibilidad hacia atrás.
Nuevo código debe usar services.rating.rating_service.RatingService
"""

from .rating.rating_service import RatingService
from .rating.favorite_service import FavoriteService

__all__ = ["RatingService", "FavoriteService"]
