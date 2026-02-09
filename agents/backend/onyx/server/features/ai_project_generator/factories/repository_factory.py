"""
Repository Factory - Factory para repositorios
==============================================

Factory que crea instancias de repositorios según configuración.
"""

import logging
from typing import Optional
from enum import Enum

from ..interfaces.repository import IProjectRepository
from ..repositories.project_repository import ProjectRepository
from ..repositories.memory_repository import MemoryProjectRepository
from ..core.continuous_generator import ContinuousGenerator

logger = logging.getLogger(__name__)


class RepositoryType(str, Enum):
    """Tipos de repositorio disponibles"""
    CONTINUOUS_GENERATOR = "continuous_generator"
    MEMORY = "memory"


class RepositoryFactory:
    """Factory para crear repositorios"""
    
    @staticmethod
    def create_project_repository(
        repository_type: RepositoryType = RepositoryType.CONTINUOUS_GENERATOR,
        continuous_generator: Optional[ContinuousGenerator] = None
    ) -> IProjectRepository:
        """
        Crea un repositorio de proyectos.
        
        Args:
            repository_type: Tipo de repositorio a crear
            continuous_generator: Instancia de ContinuousGenerator (opcional)
        
        Returns:
            Repositorio de proyectos
        """
        if repository_type == RepositoryType.CONTINUOUS_GENERATOR:
            return ProjectRepository(continuous_generator=continuous_generator)
        elif repository_type == RepositoryType.MEMORY:
            return MemoryProjectRepository()
        else:
            raise ValueError(f"Unknown repository type: {repository_type}")
    
    @staticmethod
    def create_project_repository_auto(
        continuous_generator: Optional[ContinuousGenerator] = None
    ) -> IProjectRepository:
        """
        Crea repositorio automáticamente según disponibilidad.
        
        Args:
            continuous_generator: Instancia de ContinuousGenerator (opcional)
        
        Returns:
            Repositorio de proyectos
        """
        if continuous_generator:
            return RepositoryFactory.create_project_repository(
                RepositoryType.CONTINUOUS_GENERATOR,
                continuous_generator
            )
        else:
            logger.warning("ContinuousGenerator not available, using memory repository")
            return RepositoryFactory.create_project_repository(RepositoryType.MEMORY)















