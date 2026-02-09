"""
Gestor de migraciones de base de datos
"""

import os
import importlib
from typing import List, Optional
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class Migration:
    """Representa una migración"""
    version: str
    name: str
    description: str
    up: callable
    down: Optional[callable] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class MigrationManager:
    """Gestor de migraciones de base de datos"""
    
    def __init__(self, db_connection, migrations_dir: str = "db/migrations"):
        """
        Inicializar gestor de migraciones
        
        Args:
            db_connection: Conexión a la base de datos
            migrations_dir: Directorio donde están las migraciones
        """
        self.db = db_connection
        self.migrations_dir = Path(migrations_dir)
        self.migrations: List[Migration] = []
        self._ensure_migrations_table()
    
    def _ensure_migrations_table(self):
        """Crear tabla de migraciones si no existe"""
        cursor = self.db.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.db.commit()
    
    def register_migration(self, migration: Migration):
        """Registrar una migración"""
        self.migrations.append(migration)
        self.migrations.sort(key=lambda m: m.version)
    
    def get_applied_migrations(self) -> List[str]:
        """Obtener lista de migraciones aplicadas"""
        cursor = self.db.cursor()
        cursor.execute("SELECT version FROM schema_migrations ORDER BY version")
        return [row[0] for row in cursor.fetchall()]
    
    def is_applied(self, version: str) -> bool:
        """Verificar si una migración está aplicada"""
        return version in self.get_applied_migrations()
    
    def apply_migration(self, migration: Migration):
        """Aplicar una migración"""
        if self.is_applied(migration.version):
            print(f"Migration {migration.version} already applied")
            return
        
        print(f"Applying migration {migration.version}: {migration.name}")
        
        try:
            # Ejecutar migración
            migration.up(self.db)
            
            # Registrar migración
            cursor = self.db.cursor()
            cursor.execute(
                "INSERT INTO schema_migrations (version, name) VALUES (?, ?)",
                (migration.version, migration.name)
            )
            self.db.commit()
            
            print(f"✓ Migration {migration.version} applied successfully")
        except Exception as e:
            self.db.rollback()
            print(f"✗ Error applying migration {migration.version}: {e}")
            raise
    
    def rollback_migration(self, migration: Migration):
        """Revertir una migración"""
        if not self.is_applied(migration.version):
            print(f"Migration {migration.version} not applied")
            return
        
        if migration.down is None:
            print(f"Migration {migration.version} has no rollback")
            return
        
        print(f"Rolling back migration {migration.version}: {migration.name}")
        
        try:
            # Ejecutar rollback
            migration.down(self.db)
            
            # Eliminar registro
            cursor = self.db.cursor()
            cursor.execute(
                "DELETE FROM schema_migrations WHERE version = ?",
                (migration.version,)
            )
            self.db.commit()
            
            print(f"✓ Migration {migration.version} rolled back successfully")
        except Exception as e:
            self.db.rollback()
            print(f"✗ Error rolling back migration {migration.version}: {e}")
            raise
    
    def migrate(self, target_version: Optional[str] = None):
        """Aplicar todas las migraciones pendientes"""
        applied = self.get_applied_migrations()
        
        for migration in self.migrations:
            if migration.version not in applied:
                if target_version and migration.version > target_version:
                    break
                self.apply_migration(migration)
    
    def rollback(self, target_version: Optional[str] = None):
        """Revertir migraciones"""
        applied = self.get_applied_migrations()
        
        # Revertir en orden inverso
        for migration in reversed(self.migrations):
            if migration.version in applied:
                if target_version and migration.version <= target_version:
                    break
                self.rollback_migration(migration)
    
    def status(self):
        """Mostrar estado de migraciones"""
        applied = self.get_applied_migrations()
        
        print("Migration Status:")
        print("=" * 60)
        
        for migration in self.migrations:
            status = "✓ Applied" if migration.version in applied else "○ Pending"
            print(f"{status} | {migration.version} | {migration.name}")
        
        print("=" * 60)
        print(f"Total: {len(self.migrations)} | Applied: {len(applied)} | Pending: {len(self.migrations) - len(applied)}")




