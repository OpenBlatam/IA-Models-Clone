"""
Database Migrations
===================

Sistema de migraciones de base de datos.
"""

import logging
import sqlite3
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Migration:
    """Migración de base de datos."""
    version: str
    description: str
    up_sql: str
    down_sql: str
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class MigrationService:
    """Servicio de migraciones."""
    
    def __init__(self, db_path: str):
        """
        Inicializar servicio de migraciones.
        
        Args:
            db_path: Ruta a la base de datos
        """
        self.db_path = db_path
        self._logger = logger
        self._init_migrations_table()
    
    def _init_migrations_table(self):
        """Inicializar tabla de migraciones."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS migrations (
                version TEXT PRIMARY KEY,
                description TEXT,
                applied_at TEXT NOT NULL
            )
        """)
        
        conn.commit()
        conn.close()
    
    def get_applied_migrations(self) -> List[str]:
        """
        Obtener migraciones aplicadas.
        
        Returns:
            Lista de versiones aplicadas
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT version FROM migrations ORDER BY applied_at")
        versions = [row[0] for row in cursor.fetchall()]
        
        conn.close()
        return versions
    
    def apply_migration(self, migration: Migration) -> bool:
        """
        Aplicar migración.
        
        Args:
            migration: Migración a aplicar
        
        Returns:
            True si se aplicó correctamente
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Verificar si ya está aplicada
            cursor.execute("SELECT version FROM migrations WHERE version = ?", (migration.version,))
            if cursor.fetchone():
                self._logger.warning(f"Migration {migration.version} already applied")
                conn.close()
                return False
            
            # Aplicar migración
            cursor.executescript(migration.up_sql)
            
            # Registrar migración
            cursor.execute("""
                INSERT INTO migrations (version, description, applied_at)
                VALUES (?, ?, ?)
            """, (migration.version, migration.description, datetime.now().isoformat()))
            
            conn.commit()
            conn.close()
            
            self._logger.info(f"Applied migration {migration.version}: {migration.description}")
            return True
        
        except Exception as e:
            self._logger.error(f"Error applying migration {migration.version}: {str(e)}")
            return False
    
    def rollback_migration(self, migration: Migration) -> bool:
        """
        Revertir migración.
        
        Args:
            migration: Migración a revertir
        
        Returns:
            True si se revirtió correctamente
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Ejecutar rollback
            cursor.executescript(migration.down_sql)
            
            # Eliminar registro
            cursor.execute("DELETE FROM migrations WHERE version = ?", (migration.version,))
            
            conn.commit()
            conn.close()
            
            self._logger.info(f"Rolled back migration {migration.version}")
            return True
        
        except Exception as e:
            self._logger.error(f"Error rolling back migration {migration.version}: {str(e)}")
            return False




