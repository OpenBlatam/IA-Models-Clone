"""
PDF Variantes Database Migrations
Migraciones de base de datos para el sistema PDF Variantes
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

logger = logging.getLogger(__name__)

class MigrationManager:
    """Gestor de migraciones de base de datos"""
    
    def __init__(self, engine: AsyncEngine, alembic_cfg_path: str = "alembic.ini"):
        self.engine = engine
        self.alembic_cfg_path = alembic_cfg_path
        self.alembic_cfg = None
    
    async def initialize(self):
        """Inicializar gestor de migraciones"""
        try:
            # Configurar Alembic
            self.alembic_cfg = Config(self.alembic_cfg_path)
            
            logger.info("Migration Manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing Migration Manager: {e}")
            raise
    
    async def create_migration(self, message: str) -> bool:
        """Crear nueva migración"""
        try:
            # Crear migración
            command.revision(
                self.alembic_cfg,
                message=message,
                autogenerate=True
            )
            
            logger.info(f"Migration created successfully: {message}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating migration: {e}")
            return False
    
    async def upgrade(self, revision: str = "head") -> bool:
        """Aplicar migraciones"""
        try:
            # Aplicar migraciones
            command.upgrade(self.alembic_cfg, revision)
            
            logger.info(f"Database upgraded to revision: {revision}")
            return True
            
        except Exception as e:
            logger.error(f"Error upgrading database: {e}")
            return False
    
    async def downgrade(self, revision: str) -> bool:
        """Revertir migraciones"""
        try:
            # Revertir migraciones
            command.downgrade(self.alembic_cfg, revision)
            
            logger.info(f"Database downgraded to revision: {revision}")
            return True
            
        except Exception as e:
            logger.error(f"Error downgrading database: {e}")
            return False
    
    async def get_current_revision(self) -> Optional[str]:
        """Obtener revisión actual"""
        try:
            # Obtener revisión actual
            with self.engine.connect() as conn:
                context = MigrationContext.configure(conn)
                current_rev = context.get_current_revision()
                
                return current_rev
                
        except Exception as e:
            logger.error(f"Error getting current revision: {e}")
            return None
    
    async def get_migration_history(self) -> List[Dict[str, Any]]:
        """Obtener historial de migraciones"""
        try:
            # Obtener historial
            script = ScriptDirectory.from_config(self.alembic_cfg)
            revisions = []
            
            for revision in script.walk_revisions():
                revisions.append({
                    "revision": revision.revision,
                    "down_revision": revision.down_revision,
                    "branch_labels": revision.branch_labels,
                    "depends_on": revision.depends_on,
                    "comment": revision.comment,
                    "doc": revision.doc
                })
            
            return revisions
            
        except Exception as e:
            logger.error(f"Error getting migration history: {e}")
            return []
    
    async def check_migration_status(self) -> Dict[str, Any]:
        """Verificar estado de migraciones"""
        try:
            current_rev = await self.get_current_revision()
            history = await self.get_migration_history()
            
            return {
                "current_revision": current_rev,
                "total_migrations": len(history),
                "migration_history": history,
                "is_up_to_date": current_rev == "head" if history else False
            }
            
        except Exception as e:
            logger.error(f"Error checking migration status: {e}")
            return {}

# Factory function
async def create_migration_manager(engine: AsyncEngine, alembic_cfg_path: str = "alembic.ini") -> MigrationManager:
    """Crear gestor de migraciones"""
    manager = MigrationManager(engine, alembic_cfg_path)
    await manager.initialize()
    return manager
