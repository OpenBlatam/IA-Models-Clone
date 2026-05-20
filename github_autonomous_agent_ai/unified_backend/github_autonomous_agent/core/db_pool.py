"""
Database Connection Pool - Pool básico de conexiones para SQLite con mejoras.
"""

import aiosqlite
from typing import Optional
from pathlib import Path
from contextlib import asynccontextmanager
from config.logging_config import get_logger
from core.exceptions import StorageError

logger = get_logger(__name__)


class DatabasePool:
    """
    Pool básico de conexiones para SQLite con mejoras.
    
    Attributes:
        _instance: Instancia singleton del pool
        _connection: Conexión activa (opcional)
        _db_path: Ruta a la base de datos
        _initialized: Si el pool ha sido inicializado
    """
    
    _instance: Optional['DatabasePool'] = None
    _connection: Optional[aiosqlite.Connection] = None
    _db_path: Optional[str] = None
    _initialized: bool = False
    
    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def initialize(self, db_path: str) -> None:
        """
        Inicializar el pool con la ruta de la base de datos con validaciones.
        
        Args:
            db_path: Ruta a la base de datos (debe ser string no vacío)
            
        Raises:
            ValueError: Si db_path es inválido o no se puede acceder
        """
        # Validaciones
        if not db_path or not isinstance(db_path, str) or not db_path.strip():
            raise ValueError(f"db_path debe ser un string no vacío, recibido: {db_path}")
        
        db_path = db_path.strip()
        
        # Verificar que el directorio padre existe o puede crearse
        db_file = Path(db_path)
        try:
            db_file.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Error al crear directorio para base de datos: {e}", exc_info=True)
            raise ValueError(f"No se puede crear directorio para base de datos: {e}") from e
        
        self._db_path = db_path
        self._initialized = True
        
        logger.info(f"✅ DatabasePool inicializado: {db_path}")
    
    @asynccontextmanager
    async def get_connection(self):
        """
        Obtener una conexión de la base de datos con validaciones y mejor manejo de errores.
        
        Yields:
            Conexión aiosqlite
            
        Raises:
            StorageError: Si no se puede obtener la conexión o el pool no está inicializado
        """
        if not self._initialized or not self._db_path:
            error_msg = "Database pool not initialized. Call initialize() first."
            logger.error(error_msg)
            raise StorageError(error_msg)
        
        logger.debug(f"Obteniendo conexión de base de datos: {self._db_path}")
        
        try:
            # Para SQLite, creamos una nueva conexión cada vez
            # pero con configuración optimizada
            async with aiosqlite.connect(
                self._db_path,
                timeout=10.0,
                isolation_level=None  # Autocommit mode
            ) as conn:
                # Optimizaciones para SQLite
                try:
                    await conn.execute("PRAGMA journal_mode=WAL")
                    await conn.execute("PRAGMA synchronous=NORMAL")
                    await conn.execute("PRAGMA cache_size=10000")
                    await conn.execute("PRAGMA foreign_keys=ON")
                    logger.debug("Optimizaciones de SQLite aplicadas")
                except Exception as e:
                    logger.warning(f"Error al aplicar optimizaciones de SQLite: {e}")
                    # Continuar sin optimizaciones
                
                logger.debug("✅ Conexión de base de datos obtenida exitosamente")
                yield conn
                logger.debug("Conexión de base de datos cerrada")
        except aiosqlite.Error as e:
            error_msg = f"Error de SQLite al obtener conexión: {e}"
            logger.error(error_msg, exc_info=True)
            raise StorageError(error_msg) from e
        except Exception as e:
            error_msg = f"Error inesperado al obtener conexión de base de datos: {e}"
            logger.error(error_msg, exc_info=True)
            raise StorageError(error_msg) from e
    
    async def close(self) -> None:
        """
        Cerrar todas las conexiones del pool con mejor manejo de errores.
        
        Raises:
            StorageError: Si hay error al cerrar las conexiones
        """
        if self._connection:
            try:
                await self._connection.close()
                self._connection = None
                logger.info("✅ Conexiones del pool cerradas exitosamente")
            except Exception as e:
                error_msg = f"Error al cerrar conexiones del pool: {e}"
                logger.error(error_msg, exc_info=True)
                raise StorageError(error_msg) from e
        else:
            logger.debug("No hay conexiones activas para cerrar")


# Instancia global del pool
db_pool = DatabasePool()

