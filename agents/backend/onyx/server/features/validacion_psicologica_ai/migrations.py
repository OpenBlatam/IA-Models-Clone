"""
Sistema de Migraciones de Base de Datos
========================================
Gestión de migraciones de esquema
"""

from typing import Dict, Any, List, Optional
from uuid import UUID, uuid4
from datetime import datetime
from enum import Enum
import structlog
import json
import asyncio

logger = structlog.get_logger()


class MigrationStatus(str, Enum):
    """Estado de migración"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class Migration:
    """Migración de base de datos"""
    
    def __init__(
        self,
        id: UUID,
        version: str,
        name: str,
        up_sql: str,
        down_sql: str,
        description: Optional[str] = None
    ):
        self.id = id
        self.version = version
        self.name = name
        self.up_sql = up_sql
        self.down_sql = down_sql
        self.description = description
        self.status = MigrationStatus.PENDING
        self.created_at = datetime.utcnow()
        self.applied_at: Optional[datetime] = None
        self.rolled_back_at: Optional[datetime] = None
        self.error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
        return {
            "id": str(self.id),
            "version": self.version,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "applied_at": self.applied_at.isoformat() if self.applied_at else None,
            "rolled_back_at": self.rolled_back_at.isoformat() if self.rolled_back_at else None,
            "error": self.error
        }


class MigrationManager:
    """Gestor de migraciones"""
    
    def __init__(self):
        """Inicializar gestor"""
        self._migrations: Dict[str, Migration] = {}
        self._applied_migrations: List[str] = []
        logger.info("MigrationManager initialized")
    
    def register_migration(
        self,
        version: str,
        name: str,
        up_sql: str,
        down_sql: str,
        description: Optional[str] = None
    ) -> Migration:
        """
        Registrar migración
        
        Args:
            version: Versión de migración
            name: Nombre de migración
            up_sql: SQL para aplicar migración
            down_sql: SQL para revertir migración
            description: Descripción (opcional)
            
        Returns:
            Migración registrada
        """
        migration = Migration(
            id=uuid4(),
            version=version,
            name=name,
            up_sql=up_sql,
            down_sql=down_sql,
            description=description
        )
        
        self._migrations[version] = migration
        
        logger.info("Migration registered", version=version, name=name)
        
        return migration
    
    async def apply_migration(
        self,
        version: str
    ) -> bool:
        """
        Aplicar migración
        
        Args:
            version: Versión de migración
            
        Returns:
            True si se aplicó exitosamente
        """
        migration = self._migrations.get(version)
        if not migration:
            raise ValueError(f"Migration {version} not found")
        
        if migration.status == MigrationStatus.COMPLETED:
            logger.warning("Migration already applied", version=version)
            return True
        
        migration.status = MigrationStatus.RUNNING
        
        try:
            # En producción, ejecutar SQL real
            # await db.execute(migration.up_sql)
            
            # Simulación
            await asyncio.sleep(0.1)
            
            migration.status = MigrationStatus.COMPLETED
            migration.applied_at = datetime.utcnow()
            self._applied_migrations.append(version)
            
            logger.info("Migration applied", version=version, name=migration.name)
            return True
            
        except Exception as e:
            migration.status = MigrationStatus.FAILED
            migration.error = str(e)
            logger.error("Migration failed", version=version, error=str(e))
            return False
    
    async def rollback_migration(
        self,
        version: str
    ) -> bool:
        """
        Revertir migración
        
        Args:
            version: Versión de migración
            
        Returns:
            True si se revirtió exitosamente
        """
        migration = self._migrations.get(version)
        if not migration:
            raise ValueError(f"Migration {version} not found")
        
        if migration.status != MigrationStatus.COMPLETED:
            raise ValueError(f"Migration {version} not applied")
        
        migration.status = MigrationStatus.RUNNING
        
        try:
            # En producción, ejecutar SQL real
            # await db.execute(migration.down_sql)
            
            # Simulación
            await asyncio.sleep(0.1)
            
            migration.status = MigrationStatus.ROLLED_BACK
            migration.rolled_back_at = datetime.utcnow()
            
            if version in self._applied_migrations:
                self._applied_migrations.remove(version)
            
            logger.info("Migration rolled back", version=version, name=migration.name)
            return True
            
        except Exception as e:
            migration.status = MigrationStatus.FAILED
            migration.error = str(e)
            logger.error("Migration rollback failed", version=version, error=str(e))
            return False
    
    async def apply_all_pending(self) -> List[str]:
        """
        Aplicar todas las migraciones pendientes
        
        Returns:
            Lista de versiones aplicadas
        """
        pending = [
            version for version, migration in self._migrations.items()
            if migration.status == MigrationStatus.PENDING
        ]
        
        applied = []
        for version in sorted(pending):
            if await self.apply_migration(version):
                applied.append(version)
            else:
                break  # Detener si falla
        
        return applied
    
    def get_migration_status(self, version: Optional[str] = None) -> Dict[str, Any]:
        """
        Obtener estado de migraciones
        
        Args:
            version: Versión específica (opcional)
            
        Returns:
            Estado de migraciones
        """
        if version:
            migration = self._migrations.get(version)
            if not migration:
                return {"error": f"Migration {version} not found"}
            return migration.to_dict()
        
        return {
            "total": len(self._migrations),
            "applied": len(self._applied_migrations),
            "pending": len([
                m for m in self._migrations.values()
                if m.status == MigrationStatus.PENDING
            ]),
            "migrations": [m.to_dict() for m in self._migrations.values()]
        }


# Instancia global del gestor de migraciones
migration_manager = MigrationManager()

# Registrar migraciones iniciales
migration_manager.register_migration(
    version="001",
    name="create_validations_table",
    up_sql="""
    CREATE TABLE IF NOT EXISTS psychological_validations (
        id UUID PRIMARY KEY,
        user_id UUID NOT NULL,
        status VARCHAR(50) NOT NULL,
        created_at TIMESTAMP DEFAULT NOW(),
        updated_at TIMESTAMP DEFAULT NOW()
    );
    """,
    down_sql="DROP TABLE IF EXISTS psychological_validations;",
    description="Create psychological validations table"
)

migration_manager.register_migration(
    version="002",
    name="create_profiles_table",
    up_sql="""
    CREATE TABLE IF NOT EXISTS psychological_profiles (
        id UUID PRIMARY KEY,
        validation_id UUID REFERENCES psychological_validations(id),
        confidence_score FLOAT,
        created_at TIMESTAMP DEFAULT NOW()
    );
    """,
    down_sql="DROP TABLE IF EXISTS psychological_profiles;",
    description="Create psychological profiles table"
)

