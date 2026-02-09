"""
Database Migrations - Sistema de migraciones de base de datos
=============================================================
"""

import logging
import os
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json

logger = logging.getLogger(__name__)


@dataclass
class Migration:
    """Migración individual"""
    id: str
    name: str
    version: str
    up_migration: Callable
    down_migration: Callable
    description: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    applied_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario"""
        return {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "applied_at": self.applied_at.isoformat() if self.applied_at else None,
            "is_applied": self.applied_at is not None
        }


class MigrationManager:
    """Gestor de migraciones"""
    
    def __init__(self, migrations_path: str = "./migrations"):
        self.migrations: Dict[str, Migration] = {}
        self.applied_migrations: List[str] = []
        self.migrations_path = migrations_path
        os.makedirs(migrations_path, exist_ok=True)
        self._load_migration_history()
    
    def _load_migration_history(self):
        """Carga el historial de migraciones aplicadas"""
        history_file = os.path.join(self.migrations_path, ".migration_history.json")
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r') as f:
                    data = json.load(f)
                    self.applied_migrations = data.get("applied", [])
            except Exception as e:
                logger.error(f"Error cargando historial de migraciones: {e}")
    
    def _save_migration_history(self):
        """Guarda el historial de migraciones"""
        history_file = os.path.join(self.migrations_path, ".migration_history.json")
        try:
            with open(history_file, 'w') as f:
                json.dump({"applied": self.applied_migrations}, f, indent=2)
        except Exception as e:
            logger.error(f"Error guardando historial de migraciones: {e}")
    
    def register_migration(
        self,
        migration_id: str,
        name: str,
        version: str,
        up_migration: Callable,
        down_migration: Callable,
        description: str = ""
    ) -> Migration:
        """Registra una migración"""
        migration = Migration(
            id=migration_id,
            name=name,
            version=version,
            up_migration=up_migration,
            down_migration=down_migration,
            description=description
        )
        
        self.migrations[migration_id] = migration
        logger.info(f"Migración {migration_id} registrada")
        return migration
    
    def apply_migration(self, migration_id: str) -> bool:
        """Aplica una migración"""
        if migration_id not in self.migrations:
            logger.error(f"Migración {migration_id} no encontrada")
            return False
        
        if migration_id in self.applied_migrations:
            logger.warning(f"Migración {migration_id} ya aplicada")
            return True
        
        migration = self.migrations[migration_id]
        
        try:
            logger.info(f"Aplicando migración {migration_id}: {migration.name}")
            
            # Ejecutar up migration
            if callable(migration.up_migration):
                migration.up_migration()
            
            migration.applied_at = datetime.now()
            self.applied_migrations.append(migration_id)
            self._save_migration_history()
            
            logger.info(f"Migración {migration_id} aplicada exitosamente")
            return True
        except Exception as e:
            logger.error(f"Error aplicando migración {migration_id}: {e}")
            return False
    
    def rollback_migration(self, migration_id: str) -> bool:
        """Revierte una migración"""
        if migration_id not in self.migrations:
            logger.error(f"Migración {migration_id} no encontrada")
            return False
        
        if migration_id not in self.applied_migrations:
            logger.warning(f"Migración {migration_id} no está aplicada")
            return False
        
        migration = self.migrations[migration_id]
        
        try:
            logger.info(f"Revirtiendo migración {migration_id}: {migration.name}")
            
            # Ejecutar down migration
            if callable(migration.down_migration):
                migration.down_migration()
            
            self.applied_migrations.remove(migration_id)
            migration.applied_at = None
            self._save_migration_history()
            
            logger.info(f"Migración {migration_id} revertida exitosamente")
            return True
        except Exception as e:
            logger.error(f"Error revirtiendo migración {migration_id}: {e}")
            return False
    
    def apply_all_pending(self) -> Dict[str, bool]:
        """Aplica todas las migraciones pendientes"""
        results = {}
        
        # Ordenar por versión
        pending = [
            m for m in self.migrations.values()
            if m.id not in self.applied_migrations
        ]
        pending.sort(key=lambda m: m.version)
        
        for migration in pending:
            results[migration.id] = self.apply_migration(migration.id)
        
        return results
    
    def get_pending_migrations(self) -> List[Dict[str, Any]]:
        """Obtiene migraciones pendientes"""
        pending = [
            m for m in self.migrations.values()
            if m.id not in self.applied_migrations
        ]
        pending.sort(key=lambda m: m.version)
        return [m.to_dict() for m in pending]
    
    def get_applied_migrations(self) -> List[Dict[str, Any]]:
        """Obtiene migraciones aplicadas"""
        applied = [
            m for m in self.migrations.values()
            if m.id in self.applied_migrations
        ]
        applied.sort(key=lambda m: m.version)
        return [m.to_dict() for m in applied]
    
    def list_migrations(self) -> List[Dict[str, Any]]:
        """Lista todas las migraciones"""
        migrations = list(self.migrations.values())
        migrations.sort(key=lambda m: m.version)
        return [m.to_dict() for m in migrations]
    
    def get_migration_status(self) -> Dict[str, Any]:
        """Obtiene el estado de las migraciones"""
        total = len(self.migrations)
        applied = len(self.applied_migrations)
        pending = total - applied
        
        return {
            "total_migrations": total,
            "applied_migrations": applied,
            "pending_migrations": pending,
            "migrations": self.list_migrations()
        }




