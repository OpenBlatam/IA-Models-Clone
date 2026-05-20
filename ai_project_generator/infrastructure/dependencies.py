"""
Dependencies - Dependencies para inyección de dependencias
==========================================================

Dependencies para FastAPI que proporcionan instancias de servicios.
Usa un contenedor de dependencias para mejor gestión y testabilidad.
"""

from functools import lru_cache
from typing import Optional

from ..factories.service_factory import ServiceFactory
from ..factories.infrastructure_factory import InfrastructureFactory
from ..factories.repository_factory import RepositoryFactory
from ..services.project_service import ProjectService
from ..services.generation_service import GenerationService
from ..infrastructure.cache import CacheService, get_cache_service
from ..infrastructure.events import EventPublisher
from ..infrastructure.workers import WorkerService, get_worker_service
from ..core.project_generator import ProjectGenerator
from ..core.continuous_generator import ContinuousGenerator


class DependencyContainer:
    """Contenedor de dependencias para gestión centralizada"""
    
    def __init__(self):
        self._cache_service: Optional[CacheService] = None
        self._worker_service: Optional[WorkerService] = None
        self._event_publisher: Optional[EventPublisher] = None
        self._project_generator: Optional[ProjectGenerator] = None
        self._continuous_generator: Optional[ContinuousGenerator] = None
    
    def get_cache_service(self) -> CacheService:
        """Obtiene cache service (singleton)"""
        if self._cache_service is None:
            self._cache_service = get_cache_service()
        return self._cache_service
    
    def get_worker_service(self) -> WorkerService:
        """Obtiene worker service (singleton)"""
        if self._worker_service is None:
            self._worker_service = get_worker_service()
        return self._worker_service
    
    def get_event_publisher(self) -> EventPublisher:
        """Obtiene event publisher (singleton)"""
        if self._event_publisher is None:
            self._event_publisher = EventPublisher()
        return self._event_publisher
    
    def get_project_generator(self) -> Optional[ProjectGenerator]:
        """Obtiene project generator (singleton, lazy)"""
        if self._project_generator is None:
            try:
                self._project_generator = ProjectGenerator()
            except Exception:
                pass
        return self._project_generator
    
    def get_continuous_generator(self) -> Optional[ContinuousGenerator]:
        """Obtiene continuous generator (singleton, lazy)"""
        if self._continuous_generator is None:
            try:
                self._continuous_generator = ContinuousGenerator()
            except Exception:
                pass
        return self._continuous_generator
    
    def reset(self):
        """Resetea todas las dependencias (útil para testing)"""
        self._cache_service = None
        self._worker_service = None
        self._event_publisher = None
        self._project_generator = None
        self._continuous_generator = None


_container = DependencyContainer()


@lru_cache()
def get_dependency_container() -> DependencyContainer:
    """Obtiene el contenedor de dependencias (singleton)"""
    return _container


def get_cache_service_dependency() -> CacheService:
    """Dependency para cache service"""
    return get_dependency_container().get_cache_service()


def get_worker_service_dependency() -> WorkerService:
    """Dependency para worker service"""
    return get_dependency_container().get_worker_service()


def get_event_publisher_dependency() -> EventPublisher:
    """Dependency para event publisher"""
    return get_dependency_container().get_event_publisher()


def get_project_generator_dependency() -> Optional[ProjectGenerator]:
    """Dependency para project generator"""
    return get_dependency_container().get_project_generator()


def get_continuous_generator_dependency() -> Optional[ContinuousGenerator]:
    """Dependency para continuous generator"""
    return get_dependency_container().get_continuous_generator()


@lru_cache()
def get_project_service() -> ProjectService:
    """Dependency para project service - Usa factory para crear con dependencias"""
    return ServiceFactory.create_project_service(
        cache_service=get_cache_service_dependency(),
        event_publisher=get_event_publisher_dependency(),
        project_generator=get_project_generator_dependency(),
        continuous_generator=get_continuous_generator_dependency()
    )


@lru_cache()
def get_generation_service() -> GenerationService:
    """Dependency para generation service - Usa factory para crear con dependencias"""
    return ServiceFactory.create_generation_service(
        project_generator=get_project_generator_dependency(),
        worker_service=get_worker_service_dependency(),
        event_publisher=get_event_publisher_dependency(),
        cache_service=get_cache_service_dependency()
    )


@lru_cache()
def get_validation_service():
    """Dependency para validation service"""
    from ..services.validation_service import ValidationService
    return ValidationService()


@lru_cache()
def get_export_service():
    """Dependency para export service"""
    from ..services.export_service import ExportService
    return ExportService()


@lru_cache()
def get_deployment_service():
    """Dependency para deployment service"""
    from ..services.deployment_service import DeploymentService
    return DeploymentService()


@lru_cache()
def get_analytics_service():
    """Dependency para analytics service"""
    from ..services.analytics_service import AnalyticsService
    return AnalyticsService()

