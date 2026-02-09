"""
Adapters for External Systems

Adapters provide interfaces to external systems like cameras, storage, and ML model loaders.
"""

from .camera_adapter import CameraAdapter
from .ml_model_loader import MLModelLoader
from .storage_adapter import StorageAdapter

__all__ = [
    "CameraAdapter",
    "MLModelLoader",
    "StorageAdapter",
]



