"""
Dependency Injection Setup
==========================

Configuración y setup del sistema de dependency injection
con todos los componentes de la arquitectura mejorada.
"""

import logging
import os
from typing import Optional, Dict, Any

from .dependency_injection import Container, Lifecycle
from .application_layer import (
    IRobotRepository,
    IMovementRepository,
    MoveRobotUseCase,
    GetRobotStatusUseCase,
    GetMovementHistoryUseCase
)
from .repository_factory import (
    RepositoryFactory,
    RepositoryType,
    create_repository_factory
)
from .error_handling import ErrorHandler, get_error_handler
from .events import EventBus
from .circuit_breaker import CircuitBreakerManager, get_circuit_breaker_manager

logger = logging.getLogger(__name__)


class DISetup:
    """
    Configurador del sistema de dependency injection.
    
    Centraliza la configuración y registro de todos los servicios
    de la arquitectura mejorada.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Inicializar setup de DI.
        
        Args:
            config: Configuración del sistema
                - repository_type: Tipo de repositorio
                - db_connection: Conexión a BD
                - cache_config: Configuración de cache
                - enable_event_bus: Habilitar event bus
        """
        self.config = config or self._load_config()
        self.container = Container()
        self._initialized = False
    
    def _load_config(self) -> Dict[str, Any]:
        """Cargar configuración desde variables de entorno."""
        return {
            'repository_type': os.getenv('REPOSITORY_TYPE', 'in_memory'),
            'db_connection': None,  # Se configura después si es necesario
            'cache_config': {
                'ttl': int(os.getenv('CACHE_TTL', '300')),
                'max_size': int(os.getenv('CACHE_MAX_SIZE', '1000'))
            },
            'enable_event_bus': os.getenv('ENABLE_EVENT_BUS', 'true').lower() == 'true'
        }
    
    async def setup(self):
        """
        Configurar todos los servicios en el contenedor.
        
        Este método debe llamarse antes de usar cualquier servicio.
        """
        if self._initialized:
            logger.warning("DI ya está inicializado, omitiendo setup")
            return
        
        logger.info("Configurando Dependency Injection...")
        
        # 1. Setup de repositorios
        await self._setup_repositories()
        
        # 2. Setup de use cases
        await self._setup_use_cases()
        
        # 3. Setup de servicios auxiliares
        await self._setup_auxiliary_services()
        
        self._initialized = True
        logger.info("Dependency Injection configurado exitosamente")
    
    async def _setup_repositories(self):
        """Configurar repositorios."""
        logger.info("Configurando repositorios...")
        
        # Crear factory de repositorios
        factory = create_repository_factory(
            repository_type=self.config.get('repository_type', 'in_memory'),
            db_connection=self.config.get('db_connection'),
            cache_config=self.config.get('cache_config')
        )
        
        # Crear repositorios
        robot_repo = factory.create_robot_repository()
        movement_repo = factory.create_movement_repository()
        
        # Inicializar repositorios
        await robot_repo.initialize()
        await movement_repo.initialize()
        
        # Registrar en contenedor como singletons
        self.container.register(
            IRobotRepository,
            implementation=robot_repo,
            lifecycle=Lifecycle.SINGLETON
        )
        
        self.container.register(
            IMovementRepository,
            implementation=movement_repo,
            lifecycle=Lifecycle.SINGLETON
        )
        
        logger.info("Repositorios configurados")
    
    async def _setup_use_cases(self):
        """Configurar use cases."""
        logger.info("Configurando use cases...")
        
        # Resolver dependencias
        robot_repo = await self.container.resolve_async(IRobotRepository)
        movement_repo = await self.container.resolve_async(IMovementRepository)
        
        # Event bus (opcional)
        event_publisher = None
        if self.config.get('enable_event_bus', True):
            try:
                event_publisher = await self.container.resolve_async(EventBus)
            except ValueError:
                # EventBus no registrado, crear uno nuevo
                event_publisher = EventBus()
                self.container.register(
                    EventBus,
                    implementation=event_publisher,
                    lifecycle=Lifecycle.SINGLETON
                )
        
        # Crear use cases
        move_robot_use_case = MoveRobotUseCase(
            robot_repository=robot_repo,
            movement_repository=movement_repo,
            event_publisher=event_publisher
        )
        
        get_robot_status_use_case = GetRobotStatusUseCase(
            robot_repository=robot_repo
        )
        
        get_movement_history_use_case = GetMovementHistoryUseCase(
            movement_repository=movement_repo
        )
        
        # Registrar use cases como singletons
        self.container.register(
            MoveRobotUseCase,
            implementation=move_robot_use_case,
            lifecycle=Lifecycle.SINGLETON
        )
        
        self.container.register(
            GetRobotStatusUseCase,
            implementation=get_robot_status_use_case,
            lifecycle=Lifecycle.SINGLETON
        )
        
        self.container.register(
            GetMovementHistoryUseCase,
            implementation=get_movement_history_use_case,
            lifecycle=Lifecycle.SINGLETON
        )
        
        logger.info("Use cases configurados")
    
    async def _setup_auxiliary_services(self):
        """Configurar servicios auxiliares."""
        logger.info("Configurando servicios auxiliares...")
        
        # Error Handler (siempre como singleton)
        error_handler = get_error_handler()
        self.container.register(
            ErrorHandler,
            implementation=error_handler,
            lifecycle=Lifecycle.SINGLETON
        )
        
        # Circuit Breaker Manager (siempre como singleton)
        circuit_manager = get_circuit_breaker_manager()
        self.container.register(
            CircuitBreakerManager,
            implementation=circuit_manager,
            lifecycle=Lifecycle.SINGLETON
        )
        
        logger.info("Servicios auxiliares configurados")
    
    def get_container(self) -> Container:
        """
        Obtener contenedor configurado.
        
        Returns:
            Contenedor con todos los servicios registrados
        """
        if not self._initialized:
            raise RuntimeError("DI no está inicializado. Llama a setup() primero.")
        return self.container
    
    async def resolve(self, service_type):
        """
        Resolver servicio del contenedor.
        
        Args:
            service_type: Tipo del servicio a resolver
            
        Returns:
            Instancia del servicio
        """
        return await self.container.resolve_async(service_type)


# ============================================================================
# Global DI Setup Instance
# ============================================================================

_global_di_setup: Optional[DISetup] = None


def get_di_setup(config: Optional[Dict[str, Any]] = None) -> DISetup:
    """
    Obtener instancia global del DI setup.
    
    Args:
        config: Configuración (solo se usa en primera llamada)
        
    Returns:
        Instancia de DISetup
    """
    global _global_di_setup
    if _global_di_setup is None:
        _global_di_setup = DISetup(config)
    return _global_di_setup


async def setup_dependency_injection(config: Optional[Dict[str, Any]] = None) -> Container:
    """
    Configurar dependency injection globalmente.
    
    Esta función debe llamarse al inicio de la aplicación.
    
    Args:
        config: Configuración del sistema
        
    Returns:
        Contenedor configurado
    """
    di_setup = get_di_setup(config)
    await di_setup.setup()
    return di_setup.get_container()


# ============================================================================
# Helper Functions para Resolver Servicios
# ============================================================================

async def get_robot_repository() -> IRobotRepository:
    """
    Obtener repositorio de robots.
    
    Returns:
        Repositorio de robots configurado
    """
    container = get_di_setup().get_container()
    return await container.resolve_async(IRobotRepository)


async def get_movement_repository() -> IMovementRepository:
    """
    Obtener repositorio de movimientos.
    
    Returns:
        Repositorio de movimientos configurado
    """
    container = get_di_setup().get_container()
    return await container.resolve_async(IMovementRepository)


async def get_move_robot_use_case() -> MoveRobotUseCase:
    """
    Obtener use case de mover robot.
    
    Returns:
        Use case configurado
    """
    container = get_di_setup().get_container()
    return await container.resolve_async(MoveRobotUseCase)


async def get_robot_status_use_case() -> GetRobotStatusUseCase:
    """
    Obtener use case de estado del robot.
    
    Returns:
        Use case configurado
    """
    container = get_di_setup().get_container()
    return await container.resolve_async(GetRobotStatusUseCase)


async def get_movement_history_use_case() -> GetMovementHistoryUseCase:
    """
    Obtener use case de historial de movimientos.
    
    Returns:
        Use case configurado
    """
    container = get_di_setup().get_container()
    return await container.resolve_async(GetMovementHistoryUseCase)


# ============================================================================
# FastAPI Integration
# ============================================================================

def create_dependency(service_type):
    """
    Crear dependency para FastAPI.
    
    Args:
        service_type: Tipo del servicio
        
    Returns:
        Función dependency para FastAPI
    """
    async def dependency():
        container = get_di_setup().get_container()
        return await container.resolve_async(service_type)
    
    return dependency

