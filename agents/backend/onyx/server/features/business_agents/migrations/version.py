"""
Version Management
==================

Database version tracking and management.
"""

import hashlib
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from .types import DatabaseVersion, Migration

logger = logging.getLogger(__name__)

class VersionManager:
    """Manages database version tracking."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self.version_table = "schema_migrations"
    
    async def initialize(self):
        """Initialize the version tracking table."""
        try:
            # Create version tracking table if it doesn't exist
            create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS {self.version_table} (
                version VARCHAR(255) PRIMARY KEY,
                applied_at TIMESTAMP NOT NULL,
                migration_name VARCHAR(255) NOT NULL,
                checksum VARCHAR(64) NOT NULL,
                metadata JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
            
            await self.session.execute(text(create_table_sql))
            await self.session.commit()
            
            logger.info("Version tracking table initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize version tracking: {str(e)}")
            await self.session.rollback()
            raise
    
    async def get_current_version(self) -> Optional[str]:
        """Get the current database version."""
        try:
            query = f"""
            SELECT version FROM {self.version_table}
            ORDER BY applied_at DESC
            LIMIT 1
            """
            
            result = await self.session.execute(text(query))
            row = result.fetchone()
            
            return row[0] if row else None
            
        except Exception as e:
            logger.error(f"Failed to get current version: {str(e)}")
            return None
    
    async def get_all_versions(self) -> List[DatabaseVersion]:
        """Get all applied versions."""
        try:
            query = f"""
            SELECT version, applied_at, migration_name, checksum, metadata
            FROM {self.version_table}
            ORDER BY applied_at ASC
            """
            
            result = await self.session.execute(text(query))
            rows = result.fetchall()
            
            versions = []
            for row in rows:
                version = DatabaseVersion(
                    version=row[0],
                    applied_at=row[1],
                    migration_name=row[2],
                    checksum=row[3],
                    metadata=row[4] or {}
                )
                versions.append(version)
            
            return versions
            
        except Exception as e:
            logger.error(f"Failed to get all versions: {str(e)}")
            return []
    
    async def is_version_applied(self, version: str) -> bool:
        """Check if a version is already applied."""
        try:
            query = f"""
            SELECT COUNT(*) FROM {self.version_table}
            WHERE version = :version
            """
            
            result = await self.session.execute(text(query), {"version": version})
            count = result.scalar()
            
            return count > 0
            
        except Exception as e:
            logger.error(f"Failed to check version status: {str(e)}")
            return False
    
    async def record_version(
        self, 
        migration: Migration, 
        checksum: str,
        metadata: Dict[str, Any] = None
    ):
        """Record a version as applied."""
        try:
            insert_sql = f"""
            INSERT INTO {self.version_table} 
            (version, applied_at, migration_name, checksum, metadata)
            VALUES (:version, :applied_at, :migration_name, :checksum, :metadata)
            """
            
            await self.session.execute(text(insert_sql), {
                "version": migration.version,
                "applied_at": datetime.now(),
                "migration_name": migration.name,
                "checksum": checksum,
                "metadata": metadata or {}
            })
            
            await self.session.commit()
            logger.info(f"Recorded version: {migration.version}")
            
        except Exception as e:
            logger.error(f"Failed to record version {migration.version}: {str(e)}")
            await self.session.rollback()
            raise
    
    async def remove_version(self, version: str):
        """Remove a version record (for rollback)."""
        try:
            delete_sql = f"""
            DELETE FROM {self.version_table}
            WHERE version = :version
            """
            
            await self.session.execute(text(delete_sql), {"version": version})
            await self.session.commit()
            
            logger.info(f"Removed version: {version}")
            
        except Exception as e:
            logger.error(f"Failed to remove version {version}: {str(e)}")
            await self.session.rollback()
            raise
    
    async def get_version_checksum(self, version: str) -> Optional[str]:
        """Get the checksum for a version."""
        try:
            query = f"""
            SELECT checksum FROM {self.version_table}
            WHERE version = :version
            """
            
            result = await self.session.execute(text(query), {"version": version})
            row = result.fetchone()
            
            return row[0] if row else None
            
        except Exception as e:
            logger.error(f"Failed to get checksum for version {version}: {str(e)}")
            return None
    
    def calculate_checksum(self, migration: Migration) -> str:
        """Calculate checksum for a migration."""
        content = f"{migration.version}:{migration.name}:{migration.up_sql}:{migration.down_sql}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    async def validate_checksum(self, migration: Migration) -> bool:
        """Validate migration checksum."""
        try:
            stored_checksum = await self.get_version_checksum(migration.version)
            if not stored_checksum:
                return True  # New migration
            
            current_checksum = self.calculate_checksum(migration)
            return stored_checksum == current_checksum
            
        except Exception as e:
            logger.error(f"Failed to validate checksum: {str(e)}")
            return False
    
    async def get_migration_history(self) -> List[Dict[str, Any]]:
        """Get detailed migration history."""
        try:
            query = f"""
            SELECT version, applied_at, migration_name, checksum, metadata
            FROM {self.version_table}
            ORDER BY applied_at DESC
            """
            
            result = await self.session.execute(text(query))
            rows = result.fetchall()
            
            history = []
            for row in rows:
                history.append({
                    "version": row[0],
                    "applied_at": row[1].isoformat(),
                    "migration_name": row[2],
                    "checksum": row[3],
                    "metadata": row[4] or {}
                })
            
            return history
            
        except Exception as e:
            logger.error(f"Failed to get migration history: {str(e)}")
            return []
