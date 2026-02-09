"""
Infrastructure Layer - External concerns
Database, cache, message brokers, external APIs
Implements domain interfaces (adapters)
"""

from .repositories import *
from .adapters import *

__all__ = [
    "AnalysisRepository",
    "UserRepository",
    "ProductRepository",
    "ImageProcessorAdapter",
    "CacheAdapter",
    "EventPublisherAdapter",
]















