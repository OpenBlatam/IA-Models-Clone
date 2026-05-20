"""
Migrations - Sistema de Migraciones
===================================

Sistema para gestionar migraciones de esquemas y configuraciones.
"""

import logging
import json
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


class MigrationStatus(Enum):
    """Estado de migración"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class Migration:
    """Migración"""
    version: str
    name: str
    up_func: Callable
    down_func: Optional[Callable] = None
    description: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    status: MigrationStatus = MigrationStatus.PENDING
    executed_at: Optional[datetime] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class MigrationManager:
    """
    Gestor de migraciones.
    
    Gestiona versionado y ejecución de migraciones.
    """
    
    def __init__(self, migrations_dir: str = "./migrations"):
        self.migrations: Dict[str, Migration] = {}
        self.migrations_dir = Path(migrations_dir)
        self.migrations_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.migrations_dir / "migration_state.json"
        self.applied_migrations: List[str] = []
        self._load_state()
    
    def register(
        self,
        version: str,
        name: str,
        up_func: Callable,
        down_func: Optional[Callable] = None,
        description: Optional[str] = None,
        dependencies: Optional[List[str]] = None,
        **metadata
    ) -> Migration:
        """
        Registrar migración.
        
        Args:
            version: Versión de la migración (ej: "001", "002")
            name: Nombre descriptivo
            up_func: Función para aplicar migración (puede ser async o sync)
            down_func: Función para revertir migración (opcional)
            description: Descripción de la migración
            dependencies: Lista de versiones de las que depende
            **metadata: Metadata adicional
            
        Returns:
            Migration creada
        """
        migration = Migration(
            version=version,
            name=name,
            up_func=up_func,
            down_func=down_func,
            description=description,
            dependencies=dependencies or [],
            metadata=metadata
        )
        
        self.migrations[version] = migration
        logger.info(f"📦 Migration registered: {version} - {name}")
        return migration
    
    async def migrate(self, target_version: Optional[str] = None) -> List[str]:
        """
        Aplicar migraciones pendientes.
        
        Args:
            target_version: Versión objetivo (None = todas las pendientes)
            
        Returns:
            Lista de versiones aplicadas
        """
        pending = self._get_pending_migrations(target_version)
        
        if not pending:
            logger.info("✅ No pending migrations")
            return []
        
        applied = []
        
        for version in pending:
            migration = self.migrations[version]
            
            try:
                logger.info(f"🔄 Applying migration: {version} - {migration.name}")
                migration.status = MigrationStatus.RUNNING
                
                # Ejecutar migración
                if asyncio.iscoroutinefunction(migration.up_func):
                    await migration.up_func()
                else:
                    migration.up_func()
                
                migration.status = MigrationStatus.COMPLETED
                migration.executed_at = datetime.now()
                self.applied_migrations.append(version)
                applied.append(version)
                
                logger.info(f"✅ Migration applied: {version}")
                
            except Exception as e:
                migration.status = MigrationStatus.FAILED
                migration.error = str(e)
                logger.error(f"❌ Migration failed: {version} - {e}")
                raise
        
        self._save_state()
        return applied
    
    async def rollback(self, target_version: Optional[str] = None, steps: int = 1) -> List[str]:
        """
        Revertir migraciones.
        
        Args:
            target_version: Versión objetivo (None = última)
            steps: Número de migraciones a revertir
            
        Returns:
            Lista de versiones revertidas
        """
        if target_version:
            to_rollback = [
                v for v in reversed(self.applied_migrations)
                if v > target_version
            ]
        else:
            to_rollback = list(reversed(self.applied_migrations))[:steps]
        
        rolled_back = []
        
        for version in to_rollback:
            if version not in self.migrations:
                logger.warning(f"Migration {version} not found, skipping")
                continue
            
            migration = self.migrations[version]
            
            if not migration.down_func:
                logger.warning(f"Migration {version} has no rollback function")
                continue
            
            try:
                logger.info(f"🔄 Rolling back migration: {version} - {migration.name}")
                migration.status = MigrationStatus.RUNNING
                
                # Ejecutar rollback
                if asyncio.iscoroutinefunction(migration.down_func):
                    await migration.down_func()
                else:
                    migration.down_func()
                
                migration.status = MigrationStatus.ROLLED_BACK
                self.applied_migrations.remove(version)
                rolled_back.append(version)
                
                logger.info(f"✅ Migration rolled back: {version}")
                
            except Exception as e:
                migration.status = MigrationStatus.FAILED
                migration.error = str(e)
                logger.error(f"❌ Rollback failed: {version} - {e}")
                raise
        
        self._save_state()
        return rolled_back
    
    def _get_pending_migrations(self, target_version: Optional[str] = None) -> List[str]:
        """Obtener migraciones pendientes en orden"""
        all_versions = sorted(self.migrations.keys())
        
        if target_version:
            pending = [v for v in all_versions if v <= target_version and v not in self.applied_migrations]
        else:
            pending = [v for v in all_versions if v not in self.applied_migrations]
        
        # Ordenar considerando dependencias
        ordered = []
        visited = set()
        
        def visit(version: str):
            if version in visited or version not in pending:
                return
            
            migration = self.migrations[version]
            for dep in migration.dependencies:
                if dep in pending:
                    visit(dep)
            
            visited.add(version)
            ordered.append(version)
        
        for version in pending:
            visit(version)
        
        return ordered
    
    def get_status(self) -> Dict[str, Any]:
        """Obtener estado de migraciones"""
        return {
            "total_migrations": len(self.migrations),
            "applied_migrations": len(self.applied_migrations),
            "pending_migrations": len(self._get_pending_migrations()),
            "applied": self.applied_migrations,
            "pending": self._get_pending_migrations()
        }
    
    def _load_state(self) -> None:
        """Cargar estado de migraciones"""
        if not self.state_file.exists():
            return
        
        try:
            data = json.loads(self.state_file.read_text(encoding="utf-8"))
            self.applied_migrations = data.get("applied_migrations", [])
            logger.debug(f"📦 Migration state loaded: {len(self.applied_migrations)} applied")
        except Exception as e:
            logger.error(f"Error loading migration state: {e}")
    
    def _save_state(self) -> None:
        """Guardar estado de migraciones"""
        try:
            data = {
                "applied_migrations": self.applied_migrations,
                "updated_at": datetime.now().isoformat()
            }
            self.state_file.write_text(
                json.dumps(data, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )
        except Exception as e:
            logger.error(f"Error saving migration state: {e}")


# Importar asyncio
import asyncio




