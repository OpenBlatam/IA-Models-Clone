"""
Migrations Module - Database migrations.

Provides:
- Migration management
- Version tracking
- Rollback support
"""

import logging
import sqlite3
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Migration:
    """Migration definition."""
    version: int
    name: str
    up_sql: str
    down_sql: str
    description: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "version": self.version,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at,
        }


class MigrationManager:
    """Database migration manager."""
    
    def __init__(self, db_path: str):
        """
        Initialize migration manager.
        
        Args:
            db_path: Database path
        """
        self.db_path = Path(db_path)
        self.migrations: List[Migration] = []
        self._init_migrations_table()
    
    def _init_migrations_table(self) -> None:
        """Initialize migrations tracking table."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                applied_at TEXT NOT NULL
            )
        """)
        
        conn.commit()
        conn.close()
    
    def register_migration(self, migration: Migration) -> None:
        """
        Register a migration.
        
        Args:
            migration: Migration object
        """
        self.migrations.append(migration)
        self.migrations.sort(key=lambda m: m.version)
    
    def get_current_version(self) -> int:
        """Get current database version."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT MAX(version) FROM schema_migrations")
        result = cursor.fetchone()
        
        conn.close()
        
        return result[0] if result[0] else 0
    
    def get_applied_migrations(self) -> List[int]:
        """Get list of applied migration versions."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT version FROM schema_migrations ORDER BY version")
        versions = [row[0] for row in cursor.fetchall()]
        
        conn.close()
        
        return versions
    
    def migrate_up(self, target_version: Optional[int] = None) -> None:
        """
        Apply migrations up to target version.
        
        Args:
            target_version: Target version (None = latest)
        """
        current_version = self.get_current_version()
        applied = set(self.get_applied_migrations())
        
        pending = [
            m for m in self.migrations
            if m.version > current_version and m.version not in applied
        ]
        
        if target_version:
            pending = [m for m in pending if m.version <= target_version]
        
        if not pending:
            logger.info("No migrations to apply")
            return
        
        conn = sqlite3.connect(self.db_path)
        
        try:
            for migration in pending:
                logger.info(f"Applying migration {migration.version}: {migration.name}")
                
                cursor = conn.cursor()
                cursor.executescript(migration.up_sql)
                
                cursor.execute("""
                    INSERT INTO schema_migrations (version, name, applied_at)
                    VALUES (?, ?, ?)
                """, (migration.version, migration.name, datetime.now().isoformat()))
                
                conn.commit()
                logger.info(f"Applied migration {migration.version}")
        
        except Exception as e:
            conn.rollback()
            logger.error(f"Migration failed: {e}")
            raise
        finally:
            conn.close()
    
    def migrate_down(self, target_version: int) -> None:
        """
        Rollback migrations down to target version.
        
        Args:
            target_version: Target version
        """
        current_version = self.get_current_version()
        
        if target_version >= current_version:
            logger.info("No migrations to rollback")
            return
        
        applied = self.get_applied_migrations()
        to_rollback = [
            m for m in self.migrations
            if m.version in applied and m.version > target_version
        ]
        to_rollback.sort(key=lambda m: m.version, reverse=True)
        
        conn = sqlite3.connect(self.db_path)
        
        try:
            for migration in to_rollback:
                logger.info(f"Rolling back migration {migration.version}: {migration.name}")
                
                cursor = conn.cursor()
                cursor.executescript(migration.down_sql)
                
                cursor.execute("DELETE FROM schema_migrations WHERE version = ?", (migration.version,))
                
                conn.commit()
                logger.info(f"Rolled back migration {migration.version}")
        
        except Exception as e:
            conn.rollback()
            logger.error(f"Rollback failed: {e}")
            raise
        finally:
            conn.close()
    
    def create_migration(
        self,
        name: str,
        up_sql: str,
        down_sql: str,
        description: str = "",
    ) -> Migration:
        """
        Create a new migration.
        
        Args:
            name: Migration name
            up_sql: SQL for applying migration
            down_sql: SQL for rolling back migration
            description: Migration description
            
        Returns:
            Created migration
        """
        current_version = self.get_current_version()
        new_version = current_version + 1
        
        migration = Migration(
            version=new_version,
            name=name,
            up_sql=up_sql,
            down_sql=down_sql,
            description=description,
        )
        
        self.register_migration(migration)
        logger.info(f"Created migration {new_version}: {name}")
        
        return migration












