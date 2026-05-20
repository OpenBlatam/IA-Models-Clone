"""
Shared Module
Shared utilities and components used across modules
"""

from .interfaces import IRepository, IUseCase, IController
from .events import BaseEvent, EventBus, EventHandler
from .value_objects import EntityId, Timestamp, Metadata

__all__ = [
    "IRepository",
    "IUseCase",
    "IController",
    "BaseEvent",
    "EventBus",
    "EventHandler",
    "EntityId",
    "Timestamp",
    "Metadata"
]






