"""
Repository Factory
==================

Factory para crear repositorios según configuración.
Soporta múltiples backends: In-Memory, SQL, y con cache.
"""

import logging
from typing import Optional, Dict, Any
from enum import Enum

from .infrastructure_repositories import (
    InMemoryRobotRepository,
    InMemoryMovementRepository,
    SQLRobotRepository,
    SQLMovementRepository
)
from .application_layer import IRobotRepository, IMovementRepository
from .domain_improved import Robot, RobotMovement

logger = logging.getLogger(__name__)


class RepositoryType(Enum):
    """Tipo de repositorio."""
    IN_MEMORY = "in_memory"
    SQL = "sql"
    SQL_WITH_CACHE = "sql_with_cache"


class RepositoryFactory:
    """
    Factory para crear repositorios.
    
    Permite crear repositorios según configuración sin acoplar
    el código a implementaciones específicas.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Inicializar factory.
        
        Args:
            config: Configuración del factory
                - repository_type: Tipo de repositorio (RepositoryType)
                - db_connection: Conexión a BD (para SQL)
                - cache_config: Configuración de cache (opcional)
        """
        self.config = config or {}
        self._robot_repository: Optional[IRobotRepository] = None
        self._movement_repository: Optional[IMovementRepository] = None
    
    def create_robot_repository(self) -> IRobotRepository:
        """
        Crear repositorio de robots.
        
        Returns:
            Repositorio de robots según configuración
        """
        if self._robot_repository:
            return self._robot_repository
        
        repo_type = self.config.get(
            'repository_type',
            RepositoryType.IN_MEMORY.value
        )
        
        if repo_type == RepositoryType.IN_MEMORY.value:
            self._robot_repository = InMemoryRobotRepository()
            logger.info("Repositorio de robots: In-Memory")
        
        elif repo_type == RepositoryType.SQL.value:
            db_connection = self.config.get('db_connection')
            if not db_connection:
                raise ValueError("db_connection requerido para repositorio SQL")
            self._robot_repository = SQLRobotRepository(db_connection)
            logger.info("Repositorio de robots: SQL")
        
        elif repo_type == RepositoryType.SQL_WITH_CACHE.value:
            db_connection = self.config.get('db_connection')
            if not db_connection:
                raise ValueError("db_connection requerido para repositorio SQL con cache")
            cache_config = self.config.get('cache_config', {})
            self._robot_repository = CachedRobotRepository(
                SQLRobotRepository(db_connection),
                cache_config
            )
            logger.info("Repositorio de robots: SQL con Cache")
        
        else:
            raise ValueError(f"Tipo de repositorio no soportado: {repo_type}")
        
        return self._robot_repository
    
    def create_movement_repository(self) -> IMovementRepository:
        """
        Crear repositorio de movimientos.
        
        Returns:
            Repositorio de movimientos según configuración
        """
        if self._movement_repository:
            return self._movement_repository
        
        repo_type = self.config.get(
            'repository_type',
            RepositoryType.IN_MEMORY.value
        )
        
        if repo_type == RepositoryType.IN_MEMORY.value:
            self._movement_repository = InMemoryMovementRepository()
            logger.info("Repositorio de movimientos: In-Memory")
        
        elif repo_type == RepositoryType.SQL.value:
            db_connection = self.config.get('db_connection')
            if not db_connection:
                raise ValueError("db_connection requerido para repositorio SQL")
            self._movement_repository = SQLMovementRepository(db_connection)
            logger.info("Repositorio de movimientos: SQL")
        
        elif repo_type == RepositoryType.SQL_WITH_CACHE.value:
            db_connection = self.config.get('db_connection')
            if not db_connection:
                raise ValueError("db_connection requerido para repositorio SQL con cache")
            cache_config = self.config.get('cache_config', {})
            self._movement_repository = CachedMovementRepository(
                SQLMovementRepository(db_connection),
                cache_config
            )
            logger.info("Repositorio de movimientos: SQL con Cache")
        
        else:
            raise ValueError(f"Tipo de repositorio no soportado: {repo_type}")
        
        return self._movement_repository


# ============================================================================
# Cached Repositories (Decorator Pattern)
# ============================================================================

class CachedRobotRepository(IRobotRepository):
    """
    Decorador para agregar cache a repositorio de robots.
    
    Implementa el patrón Decorator para agregar funcionalidad
    de cache sin modificar el repositorio base.
    """
    
    def __init__(
        self,
        base_repository: IRobotRepository,
        cache_config: Optional[Dict[str, Any]] = None
    ):
        """
        Inicializar repositorio con cache.
        
        Args:
            base_repository: Repositorio base a decorar
            cache_config: Configuración de cache
                - ttl: Time to live en segundos (default: 300)
                - max_size: Tamaño máximo del cache (default: 1000)
        """
        self.base_repository = base_repository
        self.cache_config = cache_config or {}
        self._cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, float] = {}
        self._ttl = self.cache_config.get('ttl', 300)  # 5 minutos por defecto
        self._max_size = self.cache_config.get('max_size', 1000)
    
    async def find_by_id(self, robot_id: str):
        """Encontrar robot por ID (con cache)."""
        import time
        
        # Verificar cache
        if robot_id in self._cache:
            timestamp = self._cache_timestamps.get(robot_id, 0)
            if time.time() - timestamp < self._ttl:
                logger.debug(f"Cache hit para robot {robot_id}")
                return self._cache[robot_id]
            else:
                # Cache expirado
                del self._cache[robot_id]
                del self._cache_timestamps[robot_id]
        
        # Cache miss - obtener de repositorio base
        robot = await self.base_repository.find_by_id(robot_id)
        
        if robot:
            # Guardar en cache
            self._add_to_cache(robot_id, robot)
        
        return robot
    
    async def save(self, robot: Robot) -> None:
        """Guardar robot (invalidar cache)."""
        # Guardar en repositorio base
        await self.base_repository.save(robot)
        
        # Invalidar cache para este robot
        if robot.id in self._cache:
            del self._cache[robot.id]
            del self._cache_timestamps[robot.id]
        
        logger.debug(f"Cache invalidado para robot {robot.id}")
    
    async def find_all(self):
        """Encontrar todos los robots (sin cache por simplicidad)."""
        return await self.base_repository.find_all()
    
    def _add_to_cache(self, robot_id: str, robot: Any):
        """Agregar robot al cache."""
        import time
        
        # Limpiar cache si está lleno
        if len(self._cache) >= self._max_size:
            # Eliminar el más antiguo
            oldest_key = min(self._cache_timestamps.items(), key=lambda x: x[1])[0]
            del self._cache[oldest_key]
            del self._cache_timestamps[oldest_key]
        
        # Agregar al cache
        self._cache[robot_id] = robot
        self._cache_timestamps[robot_id] = time.time()
    
    def clear_cache(self):
        """Limpiar cache (útil para testing)."""
        self._cache.clear()
        self._cache_timestamps.clear()


class CachedMovementRepository(IMovementRepository):
    """
    Decorador para agregar cache a repositorio de movimientos.
    """
    
    def __init__(
        self,
        base_repository: IMovementRepository,
        cache_config: Optional[Dict[str, Any]] = None
    ):
        """
        Inicializar repositorio con cache.
        
        Args:
            base_repository: Repositorio base a decorar
            cache_config: Configuración de cache
        """
        self.base_repository = base_repository
        self.cache_config = cache_config or {}
        self._cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, float] = {}
        self._robot_cache: Dict[str, Any] = {}  # Cache para find_by_robot_id
        self._robot_cache_timestamps: Dict[str, float] = {}
        self._ttl = self.cache_config.get('ttl', 300)
        self._max_size = self.cache_config.get('max_size', 1000)
    
    async def find_by_id(self, movement_id: str):
        """Encontrar movimiento por ID (con cache)."""
        import time
        
        if movement_id in self._cache:
            timestamp = self._cache_timestamps.get(movement_id, 0)
            if time.time() - timestamp < self._ttl:
                logger.debug(f"Cache hit para movimiento {movement_id}")
                return self._cache[movement_id]
            else:
                del self._cache[movement_id]
                del self._cache_timestamps[movement_id]
        
        movement = await self.base_repository.find_by_id(movement_id)
        
        if movement:
            self._add_to_cache(movement_id, movement)
        
        return movement
    
    async def save(self, movement: RobotMovement) -> None:
        """Guardar movimiento (invalidar cache)."""
        await self.base_repository.save(movement)
        
        # Invalidar caches relacionados
        if movement.id in self._cache:
            del self._cache[movement.id]
            del self._cache_timestamps[movement.id]
        
        # Invalidar cache de robot
        if movement.robot_id in self._robot_cache:
            del self._robot_cache[movement.robot_id]
            del self._robot_cache_timestamps[movement.robot_id]
        
        logger.debug(f"Cache invalidado para movimiento {movement.id}")
    
    async def find_by_robot_id(self, robot_id: str, limit: int = 100):
        """Encontrar movimientos por robot ID (con cache)."""
        import time
        
        cache_key = f"{robot_id}:{limit}"
        
        if cache_key in self._robot_cache:
            timestamp = self._robot_cache_timestamps.get(cache_key, 0)
            if time.time() - timestamp < self._ttl:
                logger.debug(f"Cache hit para movimientos del robot {robot_id}")
                return self._robot_cache[cache_key]
            else:
                del self._robot_cache[cache_key]
                del self._robot_cache_timestamps[cache_key]
        
        movements = await self.base_repository.find_by_robot_id(robot_id, limit)
        
        # Guardar en cache
        self._add_to_robot_cache(cache_key, movements)
        
        return movements
    
    def _add_to_cache(self, movement_id: str, movement: Any):
        """Agregar movimiento al cache."""
        import time
        
        if len(self._cache) >= self._max_size:
            oldest_key = min(self._cache_timestamps.items(), key=lambda x: x[1])[0]
            del self._cache[oldest_key]
            del self._cache_timestamps[oldest_key]
        
        self._cache[movement_id] = movement
        self._cache_timestamps[movement_id] = time.time()
    
    def _add_to_robot_cache(self, cache_key: str, movements: Any):
        """Agregar movimientos al cache de robot."""
        import time
        
        if len(self._robot_cache) >= self._max_size:
            oldest_key = min(self._robot_cache_timestamps.items(), key=lambda x: x[1])[0]
            del self._robot_cache[oldest_key]
            del self._robot_cache_timestamps[oldest_key]
        
        self._robot_cache[cache_key] = movements
        self._robot_cache_timestamps[cache_key] = time.time()
    
    def clear_cache(self):
        """Limpiar cache."""
        self._cache.clear()
        self._cache_timestamps.clear()
        self._robot_cache.clear()
        self._robot_cache_timestamps.clear()


# ============================================================================
# Helper Functions
# ============================================================================

def create_repository_factory(
    repository_type: str = "in_memory",
    db_connection: Optional[Any] = None,
    cache_config: Optional[Dict[str, Any]] = None
) -> RepositoryFactory:
    """
    Crear factory de repositorios con configuración.
    
    Args:
        repository_type: Tipo de repositorio ("in_memory", "sql", "sql_with_cache")
        db_connection: Conexión a BD (requerido para SQL)
        cache_config: Configuración de cache
        
    Returns:
        Factory configurado
    """
    config = {
        'repository_type': repository_type,
        'db_connection': db_connection,
        'cache_config': cache_config
    }
    return RepositoryFactory(config)

