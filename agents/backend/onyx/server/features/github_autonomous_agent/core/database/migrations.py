"""
Sistema de Migraciones de Base de Datos.
"""

import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import json

from config.logging_config import get_logger
from config.settings import settings
from core.database.connection_pool import get_pool
from core.exceptions import StorageError

logger = get_logger(__name__)


class Migration:
    """Representa una migración."""
    
    def __init__(
        self,
        version: str,
        name: str,
        up_sql: str,
        down_sql: Optional[str] = None
    ):
        """
        Inicializar migración.
        
        Args:
            version: Versión de la migración (ej: "001", "002")
            name: Nombre descriptivo
            up_sql: SQL para aplicar migración
            down_sql: SQL para revertir migración (opcional)
        """
        self.version = version
        self.name = name
        self.up_sql = up_sql
        self.down_sql = down_sql
        self.applied_at: Optional[datetime] = None


class MigrationManager:
    """Manager de migraciones de base de datos."""
    
    def __init__(self, migrations_dir: Optional[Path] = None):
        """
        Inicializar manager de migraciones.
        
        Args:
            migrations_dir: Directorio con archivos de migración
        """
        if migrations_dir is None:
            migrations_dir = Path(settings.STORAGE_PATH) / "migrations"
        self.migrations_dir = migrations_dir
        self.migrations_dir.mkdir(parents=True, exist_ok=True)
        self.migrations: List[Migration] = []
    
    def register_migration(self, migration: Migration) -> None:
        """
        Registrar migración.
        
        Args:
            migration: Migración a registrar
        """
        self.migrations.append(migration)
        self.migrations.sort(key=lambda m: m.version)
        logger.debug(f"Migración registrada: {migration.version} - {migration.name}")
    
    async def init_migrations_table(self) -> None:
        """Inicializar tabla de migraciones."""
        pool = get_pool()
        async with pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS migrations (
                    version TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    applied_at TEXT NOT NULL
                )
            """)
            await conn.commit()
    
    async def get_applied_migrations(self) -> List[str]:
        """
        Obtener lista de migraciones aplicadas.
        
        Returns:
            Lista de versiones aplicadas
        """
        await self.init_migrations_table()
        pool = get_pool()
        async with pool.acquire() as conn:
            async with conn.execute("SELECT version FROM migrations ORDER BY version") as cursor:
                rows = await cursor.fetchall()
                return [row[0] for row in rows]
    
    async def apply_migration(self, migration: Migration) -> bool:
        """
        Aplicar migración.
        
        Args:
            migration: Migración a aplicar
            
        Returns:
            True si se aplicó exitosamente
        """
        try:
            pool = get_pool()
            async with pool.acquire() as conn:
                # Aplicar SQL
                await conn.executescript(migration.up_sql)
                
                # Registrar en tabla de migraciones
                await conn.execute(
                    "INSERT INTO migrations (version, name, applied_at) VALUES (?, ?, ?)",
                    (migration.version, migration.name, datetime.now().isoformat())
                )
                await conn.commit()
            
            logger.info(f"Migración aplicada: {migration.version} - {migration.name}")
            return True
        except Exception as e:
            logger.error(f"Error aplicando migración {migration.version}: {e}", exc_info=True)
            return False
    
    async def rollback_migration(self, migration: Migration) -> bool:
        """
        Revertir migración.
        
        Args:
            migration: Migración a revertir
            
        Returns:
            True si se revirtió exitosamente
        """
        if not migration.down_sql:
            logger.warning(f"Migración {migration.version} no tiene down_sql")
            return False
        
        try:
            pool = get_pool()
            async with pool.acquire() as conn:
                # Revertir SQL
                await conn.executescript(migration.down_sql)
                
                # Eliminar de tabla de migraciones
                await conn.execute("DELETE FROM migrations WHERE version = ?", (migration.version,))
                await conn.commit()
            
            logger.info(f"Migración revertida: {migration.version} - {migration.name}")
            return True
        except Exception as e:
            logger.error(f"Error revirtiendo migración {migration.version}: {e}", exc_info=True)
            return False
    
    async def migrate(self, target_version: Optional[str] = None) -> Dict[str, Any]:
        """
        Aplicar todas las migraciones pendientes.
        
        Args:
            target_version: Versión objetivo (opcional, todas si None)
            
        Returns:
            Resultado de la migración
        """
        await self.init_migrations_table()
        applied = await self.get_applied_migrations()
        
        result = {
            "applied": [],
            "failed": [],
            "skipped": []
        }
        
        for migration in self.migrations:
            if migration.version in applied:
                result["skipped"].append(migration.version)
                continue
            
            if target_version and migration.version > target_version:
                break
            
            success = await self.apply_migration(migration)
            if success:
                result["applied"].append(migration.version)
            else:
                result["failed"].append(migration.version)
                break  # Detener en primer fallo
        
        return result
    
    async def rollback(self, target_version: Optional[str] = None) -> Dict[str, Any]:
        """
        Revertir migraciones.
        
        Args:
            target_version: Versión objetivo (opcional)
            
        Returns:
            Resultado del rollback
        """
        applied = await self.get_applied_migrations()
        
        result = {
            "rolled_back": [],
            "failed": []
        }
        
        # Revertir en orden inverso
        for migration in reversed(self.migrations):
            if migration.version not in applied:
                continue
            
            if target_version and migration.version <= target_version:
                break
            
            success = await self.rollback_migration(migration)
            if success:
                result["rolled_back"].append(migration.version)
            else:
                result["failed"].append(migration.version)
                break
        
        return result


# Migraciones predefinidas
def get_default_migrations() -> List[Migration]:
    """Obtener migraciones por defecto."""
    return [
        Migration(
            version="001",
            name="add_indexes",
            up_sql="""
                CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
                CREATE INDEX IF NOT EXISTS idx_tasks_created_at ON tasks(created_at);
                CREATE INDEX IF NOT EXISTS idx_tasks_repository ON tasks(repository_owner, repository_name);
            """,
            down_sql="""
                DROP INDEX IF EXISTS idx_tasks_status;
                DROP INDEX IF EXISTS idx_tasks_created_at;
                DROP INDEX IF EXISTS idx_tasks_repository;
            """
        ),
        Migration(
            version="002",
            name="add_task_metrics",
            up_sql="""
                ALTER TABLE tasks ADD COLUMN duration_seconds REAL;
                ALTER TABLE tasks ADD COLUMN retry_count INTEGER DEFAULT 0;
            """,
            down_sql="""
                -- SQLite no soporta DROP COLUMN directamente
                -- Se requiere recrear la tabla
            """
        )
    ]



