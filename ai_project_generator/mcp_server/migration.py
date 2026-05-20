"""
MCP Migration Tools - Herramientas de migración
================================================
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Callable
from pydantic import BaseModel, Field
from datetime import datetime

logger = logging.getLogger(__name__)


class Migration(BaseModel):
    """Migración"""
    migration_id: str = Field(..., description="ID único de la migración")
    version: str = Field(..., description="Versión objetivo")
    description: str = Field(..., description="Descripción de la migración")
    up: Any = Field(..., description="Función para aplicar migración (callable)")
    down: Optional[Any] = Field(None, description="Función para revertir migración (callable)")
    applied_at: Optional[datetime] = None


class MigrationManager:
    """
    Gestor de migraciones
    
    Permite migrar datos y configuraciones entre versiones.
    """
    
    def __init__(self):
        self._migrations: List[Migration] = []
        self._applied_migrations: List[str] = []
    
    def register_migration(self, migration: Migration):
        """
        Registra una migración
        
        Args:
            migration: Migración a registrar
        """
        self._migrations.append(migration)
        logger.info(f"Registered migration: {migration.migration_id}")
    
    async def apply_migration(self, migration_id: str) -> bool:
        """
        Aplica una migración
        
        Args:
            migration_id: ID de la migración
            
        Returns:
            True si se aplicó exitosamente
        """
        migration = next((m for m in self._migrations if m.migration_id == migration_id), None)
        
        if not migration:
            logger.error(f"Migration {migration_id} not found")
            return False
        
        if migration_id in self._applied_migrations:
            logger.warning(f"Migration {migration_id} already applied")
            return True
        
        try:
            if asyncio.iscoroutinefunction(migration.up):
                await migration.up()
            else:
                migration.up()
            
            migration.applied_at = datetime.utcnow()
            self._applied_migrations.append(migration_id)
            
            logger.info(f"Applied migration: {migration_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error applying migration {migration_id}: {e}")
            return False
    
    async def rollback_migration(self, migration_id: str) -> bool:
        """
        Revierte una migración
        
        Args:
            migration_id: ID de la migración
            
        Returns:
            True si se revirtió exitosamente
        """
        migration = next((m for m in self._migrations if m.migration_id == migration_id), None)
        
        if not migration:
            logger.error(f"Migration {migration_id} not found")
            return False
        
        if not migration.down:
            logger.error(f"Migration {migration_id} has no rollback function")
            return False
        
        if migration_id not in self._applied_migrations:
            logger.warning(f"Migration {migration_id} not applied")
            return True
        
        try:
            if asyncio.iscoroutinefunction(migration.down):
                await migration.down()
            else:
                migration.down()
            
            self._applied_migrations.remove(migration_id)
            migration.applied_at = None
            
            logger.info(f"Rolled back migration: {migration_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error rolling back migration {migration_id}: {e}")
            return False
    
    async def migrate_to_version(self, target_version: str) -> bool:
        """
        Migra a una versión específica
        
        Args:
            target_version: Versión objetivo
            
        Returns:
            True si se migró exitosamente
        """
        migrations_to_apply = [
            m for m in self._migrations
            if m.version == target_version and m.migration_id not in self._applied_migrations
        ]
        
        for migration in migrations_to_apply:
            success = await self.apply_migration(migration.migration_id)
            if not success:
                return False
        
        return True
    
    def get_pending_migrations(self) -> List[Migration]:
        """
        Obtiene migraciones pendientes
        
        Returns:
            Lista de migraciones pendientes
        """
        return [
            m for m in self._migrations
            if m.migration_id not in self._applied_migrations
        ]

