"""
Data Migration - Sistema de data migration
===========================================
"""

import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class MigrationStatus(str, Enum):
    """Estados de migración"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class DataMigration:
    """Sistema de migración de datos"""
    
    def __init__(self):
        self.migrations: Dict[str, Dict[str, Any]] = {}
        self.migration_history: List[Dict[str, Any]] = []
    
    def register_migration(self, migration_id: str, name: str,
                         up_function: Callable, down_function: Optional[Callable] = None,
                         description: str = ""):
        """Registra una migración"""
        self.migrations[migration_id] = {
            "id": migration_id,
            "name": name,
            "description": description,
            "up_function": up_function,
            "down_function": down_function,
            "registered_at": datetime.now().isoformat()
        }
        
        logger.info(f"Migración registrada: {migration_id}")
    
    async def run_migration(self, migration_id: str, 
                          context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Ejecuta una migración"""
        migration = self.migrations.get(migration_id)
        if not migration:
            raise ValueError(f"Migración no encontrada: {migration_id}")
        
        history_entry = {
            "migration_id": migration_id,
            "status": MigrationStatus.RUNNING.value,
            "started_at": datetime.now().isoformat(),
            "completed_at": None,
            "error": None
        }
        
        try:
            up_func = migration["up_function"]
            
            if hasattr(up_func, '__await__'):
                await up_func(context or {})
            else:
                up_func(context or {})
            
            history_entry["status"] = MigrationStatus.COMPLETED.value
            history_entry["completed_at"] = datetime.now().isoformat()
            
            logger.info(f"Migración completada: {migration_id}")
        
        except Exception as e:
            history_entry["status"] = MigrationStatus.FAILED.value
            history_entry["error"] = str(e)
            history_entry["completed_at"] = datetime.now().isoformat()
            
            logger.error(f"Migración falló: {migration_id} - {e}")
            raise
        
        finally:
            self.migration_history.append(history_entry)
        
        return history_entry
    
    async def rollback_migration(self, migration_id: str,
                                context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Revierte una migración"""
        migration = self.migrations.get(migration_id)
        if not migration:
            raise ValueError(f"Migración no encontrada: {migration_id}")
        
        if not migration["down_function"]:
            raise ValueError(f"Migración {migration_id} no tiene función de rollback")
        
        history_entry = {
            "migration_id": migration_id,
            "status": MigrationStatus.RUNNING.value,
            "action": "rollback",
            "started_at": datetime.now().isoformat(),
            "completed_at": None,
            "error": None
        }
        
        try:
            down_func = migration["down_function"]
            
            if hasattr(down_func, '__await__'):
                await down_func(context or {})
            else:
                down_func(context or {})
            
            history_entry["status"] = MigrationStatus.ROLLED_BACK.value
            history_entry["completed_at"] = datetime.now().isoformat()
            
            logger.info(f"Rollback completado: {migration_id}")
        
        except Exception as e:
            history_entry["status"] = MigrationStatus.FAILED.value
            history_entry["error"] = str(e)
            history_entry["completed_at"] = datetime.now().isoformat()
            
            logger.error(f"Rollback falló: {migration_id} - {e}")
            raise
        
        finally:
            self.migration_history.append(history_entry)
        
        return history_entry
    
    def get_migration_status(self, migration_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene estado de una migración"""
        migration = self.migrations.get(migration_id)
        if not migration:
            return None
        
        # Buscar en historial
        history = [h for h in self.migration_history if h["migration_id"] == migration_id]
        latest = history[-1] if history else None
        
        return {
            "migration": {
                "id": migration["id"],
                "name": migration["name"],
                "description": migration["description"]
            },
            "latest_run": latest,
            "total_runs": len(history)
        }
    
    def list_migrations(self) -> List[Dict[str, Any]]:
        """Lista todas las migraciones"""
        return [
            {
                "id": m["id"],
                "name": m["name"],
                "description": m["description"],
                "has_rollback": m["down_function"] is not None,
                "registered_at": m["registered_at"]
            }
            for m in self.migrations.values()
        ]




