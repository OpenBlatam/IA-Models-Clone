"""
Data Migration Service - Migración de datos
============================================

Sistema para migrar y transformar datos entre versiones.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class MigrationStatus(str, Enum):
    """Estados de migración"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class Migration:
    """Migración"""
    id: str
    name: str
    version: str
    description: str
    up_function: Callable
    down_function: Optional[Callable] = None
    status: MigrationStatus = MigrationStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None


@dataclass
class MigrationResult:
    """Resultado de migración"""
    migration_id: str
    status: MigrationStatus
    records_processed: int = 0
    duration_seconds: float = 0.0
    error_message: Optional[str] = None


class DataMigrationService:
    """Servicio de migración de datos"""
    
    def __init__(self):
        """Inicializar servicio"""
        self.migrations: Dict[str, Migration] = {}
        self.migration_history: List[MigrationResult] = []
        logger.info("DataMigrationService initialized")
    
    def register_migration(
        self,
        name: str,
        version: str,
        description: str,
        up_function: Callable,
        down_function: Optional[Callable] = None
    ) -> Migration:
        """Registrar migración"""
        migration_id = f"migration_{version}_{name.lower().replace(' ', '_')}"
        
        migration = Migration(
            id=migration_id,
            name=name,
            version=version,
            description=description,
            up_function=up_function,
            down_function=down_function,
        )
        
        self.migrations[migration_id] = migration
        
        logger.info(f"Migration registered: {migration_id}")
        return migration
    
    async def run_migration(
        self,
        migration_id: str,
        dry_run: bool = False
    ) -> MigrationResult:
        """Ejecutar migración"""
        import time
        
        migration = self.migrations.get(migration_id)
        if not migration:
            raise ValueError(f"Migration {migration_id} not found")
        
        if migration.status == MigrationStatus.COMPLETED:
            return MigrationResult(
                migration_id=migration_id,
                status=MigrationStatus.COMPLETED,
            )
        
        migration.status = MigrationStatus.RUNNING
        migration.started_at = datetime.now()
        
        start_time = time.time()
        records_processed = 0
        error_message = None
        
        try:
            if not dry_run:
                if asyncio.iscoroutinefunction(migration.up_function):
                    result = await migration.up_function()
                else:
                    result = migration.up_function()
                
                if isinstance(result, dict) and "records_processed" in result:
                    records_processed = result["records_processed"]
                
                migration.status = MigrationStatus.COMPLETED
                migration.completed_at = datetime.now()
            else:
                migration.status = MigrationStatus.PENDING
                logger.info(f"Dry run completed for {migration_id}")
        
        except Exception as e:
            migration.status = MigrationStatus.FAILED
            error_message = str(e)
            migration.error_message = error_message
            logger.error(f"Migration {migration_id} failed: {e}")
        
        duration = time.time() - start_time
        
        result = MigrationResult(
            migration_id=migration_id,
            status=migration.status,
            records_processed=records_processed,
            duration_seconds=duration,
            error_message=error_message,
        )
        
        self.migration_history.append(result)
        
        return result
    
    async def rollback_migration(self, migration_id: str) -> MigrationResult:
        """Revertir migración"""
        import time
        
        migration = self.migrations.get(migration_id)
        if not migration:
            raise ValueError(f"Migration {migration_id} not found")
        
        if not migration.down_function:
            raise ValueError(f"Migration {migration_id} has no rollback function")
        
        migration.status = MigrationStatus.RUNNING
        start_time = time.time()
        error_message = None
        
        try:
            if asyncio.iscoroutinefunction(migration.down_function):
                await migration.down_function()
            else:
                migration.down_function()
            
            migration.status = MigrationStatus.ROLLED_BACK
            migration.completed_at = datetime.now()
        
        except Exception as e:
            migration.status = MigrationStatus.FAILED
            error_message = str(e)
            logger.error(f"Migration rollback {migration_id} failed: {e}")
        
        duration = time.time() - start_time
        
        return MigrationResult(
            migration_id=migration_id,
            status=migration.status,
            duration_seconds=duration,
            error_message=error_message,
        )
    
    def get_migration_history(self) -> List[Dict[str, Any]]:
        """Obtener historial de migraciones"""
        return [
            {
                "migration_id": r.migration_id,
                "status": r.status.value,
                "records_processed": r.records_processed,
                "duration_seconds": r.duration_seconds,
                "error_message": r.error_message,
            }
            for r in self.migration_history
        ]

