"""
Interfaces - Contratos y abstracciones
======================================

Interfaces que definen contratos entre módulos siguiendo principios SOLID.
"""

from .repository import IRepository, IProjectRepository
from .service import IService, IProjectService, IGenerationService
from .cache import ICacheService
from .events import IEventPublisher, IEventSubscriber
from .workers import IWorkerService

__all__ = [
    "IRepository",
    "IProjectRepository",
    "IService",
    "IProjectService",
    "IGenerationService",
    "ICacheService",
    "IEventPublisher",
    "IEventSubscriber",
    "IWorkerService",
]















