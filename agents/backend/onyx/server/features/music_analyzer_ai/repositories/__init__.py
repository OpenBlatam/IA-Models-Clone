"""
Repository Pattern - Data access abstraction
"""

from .repository import IRepository, BaseRepository
from .model_repository import ModelRepository
from .data_repository import DataRepository

__all__ = [
    "IRepository",
    "BaseRepository",
    "ModelRepository",
    "DataRepository"
]








