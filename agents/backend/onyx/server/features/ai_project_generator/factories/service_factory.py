"""
Service Factory - Factory para servicios
=========================================

Factory que crea instancias de servicios con sus dependencias.
"""

import logging
from typing import Optional

from ..interfaces.service import IProjectService, IGenerationService
from ..services.project_service import ProjectService
from ..services.generation_service import GenerationService
from ..core.refactored_services import (
    RefactoredProjectService,
    RefactoredGenerationService
)
from ..interfaces.repository import IProjectRepository
from ..interfaces.cache import ICacheService
from ..interfaces.events import IEventPublisher
from ..interfaces.workers import IWorkerService
from ..core.project_generator import ProjectGenerator
from ..factories.repository_factory import RepositoryFactory

logger = logging.getLogger(__name__)


class ServiceFactory:
    """Factory para crear servicios"""
    
    @staticmethod
    def create_project_service(
        repository: Optional[IProjectRepository] = None,
        cache_service: Optional[ICacheService] = None,
        event_publisher: Optional[IEventPublisher] = None,
        project_generator: Optional[ProjectGenerator] = None,
        continuous_generator = None,
        use_refactored: bool = True
    ) -> IProjectService:
        """
        Crea servicio de proyectos.
        
        Args:
            repository: Repositorio de proyectos (opcional, se crea automáticamente)
            cache_service: Servicio de cache (opcional)
            event_publisher: Publicador de eventos (opcional)
            project_generator: Generador de proyectos (opcional)
            continuous_generator: Generador continuo (opcional)
            use_refactored: Usar versión refactorizada (recomendado)
        
        Returns:
            Servicio de proyectos
        """
        # Crear repositorio si no se proporciona
        if repository is None:
            repository = RepositoryFactory.create_project_repository_auto(
                continuous_generator=continuous_generator
            )
        
        if use_refactored:
            return RefactoredProjectService(
                repository=repository,
                cache_service=cache_service,
                event_publisher=event_publisher
            )
        else:
            return ProjectService(
                repository=repository,
                project_generator=project_generator,
                continuous_generator=continuous_generator,
                cache_service=cache_service,
                event_publisher=event_publisher
            )
    
    @staticmethod
    def create_generation_service(
        project_generator: Optional[ProjectGenerator] = None,
        worker_service: Optional[IWorkerService] = None,
        event_publisher: Optional[IEventPublisher] = None,
        cache_service: Optional[ICacheService] = None,
        use_refactored: bool = True
    ) -> IGenerationService:
        """
        Crea servicio de generación.
        
        Args:
            project_generator: Generador de proyectos (opcional)
            worker_service: Servicio de workers (opcional)
            event_publisher: Publicador de eventos (opcional)
            cache_service: Servicio de cache (opcional)
            use_refactored: Usar versión refactorizada (recomendado)
        
        Returns:
            Servicio de generación
        """
        if use_refactored:
            return RefactoredGenerationService(
                project_generator=project_generator,
                worker_service=worker_service,
                event_publisher=event_publisher,
                cache_service=cache_service
            )
        else:
            return GenerationService(
                project_generator=project_generator,
                worker_service=worker_service,
                event_publisher=event_publisher,
                cache_service=cache_service
            )

