"""
Rating Service Module
=====================

Módulo especializado para gestión de ratings y favoritos.
"""

from .rating_service import RatingService
from .favorite_service import FavoriteService
from .rating_repository import RatingRepository

__all__ = [
    "RatingService",
    "FavoriteService",
    "RatingRepository",
]

