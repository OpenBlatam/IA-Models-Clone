from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import asyncio
import logging
import json
from typing import Optional, Dict, Any, List, Callable, Union
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import hashlib
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text, MetaData, Table, Column, Integer, String, DateTime
from sqlalchemy.exc import SQLAlchemyError, OperationalError
from .async_database import AsyncDatabaseManager, DatabaseType
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
Database Migration System for HeyGen AI API
Supports multiple database backends with version control and rollback capabilities
"""




logger = logging.getLogger(__name__)


class MigrationStatus(Enum):
    """Migration status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class Migration:
    """Migration record."""
    id: int
    version: str
    name: str
    description: str
    sql_up: str
    sql_down: str
    checksum: str
    created_at: datetime
    executed_at: Optional[datetime] = None
    status: MigrationStatus = MigrationStatus.PENDING
    execution_time: Optional[float] = None
    error_message: Optional[str] = None


class MigrationManager:
    """
    Database migration manager with support for multiple backends.
    Handles version control, rollbacks, and schema management.
    """
    
    def __init__(self, db_manager: AsyncDatabaseManager, migrations_dir: str = "migrations"):
        
    """__init__ function."""
self.db_manager = db_manager
        self.migrations_dir = Path(migrations_dir)
        self.migrations_dir.mkdir(exist_ok=True)
        
        # Create migrations table if it doesn't exist
        asyncio.create_task(self._ensure_migrations_table())
    
    async def _ensure_migrations_table(self) -> None:
        """Ensure migrations table exists."""
        try:
            async with self.db_manager.get_session() as session:
                if self.db_manager.config.type == DatabaseType.POSTGRESQL:
                    await session.execute(text("""
                        CREATE TABLE IF NOT EXISTS migrations (
                            id SERIAL PRIMARY KEY,
                            version VARCHAR(50) UNIQUE NOT NULL,
                            name VARCHAR(255) NOT NULL,
                            description TEXT,
                            sql_up TEXT NOT NULL,
                            sql_down TEXT NOT NULL,
                            checksum VARCHAR(64) NOT NULL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            executed_at TIMESTAMP NULL,
                            status VARCHAR(20) DEFAULT 'pending',
                            execution_time FLOAT NULL,
                            error_message TEXT NULL
                        )
                    """))
                elif self.db_manager.config.type == DatabaseType.MYSQL:
                    await session.execute(text("""
                        CREATE TABLE IF NOT EXISTS migrations (
                            id INT AUTO_INCREMENT PRIMARY KEY,
                            version VARCHAR(50) UNIQUE NOT NULL,
                            name VARCHAR(255) NOT NULL,
                            description TEXT,
                            sql_up TEXT NOT NULL,
                            sql_down TEXT NOT NULL,
                            checksum VARCHAR(64) NOT NULL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            executed_at TIMESTAMP NULL,
                            status VARCHAR(20) DEFAULT 'pending',
                            execution_time FLOAT NULL,
                            error_message TEXT NULL
                        )
                    """))
                elif self.db_manager.config.type == DatabaseType.SQLITE:
                    await session.execute(text("""
                        CREATE TABLE IF NOT EXISTS migrations (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            version TEXT UNIQUE NOT NULL,
                            name TEXT NOT NULL,
                            description TEXT,
                            sql_up TEXT NOT NULL,
                            sql_down TEXT NOT NULL,
                            checksum TEXT NOT NULL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            executed_at TIMESTAMP NULL,
                            status TEXT DEFAULT 'pending',
                            execution_time REAL NULL,
                            error_message TEXT NULL
                        )
                    """))
                
                await session.commit()
                logger.info("✓ Migrations table ensured")
                
        except Exception as e:
            logger.error(f"Error ensuring migrations table: {e}")
            raise
    
    def create_migration(
        self, 
        name: str, 
        description: str, 
        sql_up: str, 
        sql_down: str
    ) -> Migration:
        """Create a new migration."""
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
        checksum = hashlib.sha256(f"{sql_up}{sql_down}".encode()).hexdigest()
        
        migration = Migration(
            id=0,
            version=version,
            name=name,
            description=description,
            sql_up=sql_up,
            sql_down=sql_down,
            checksum=checksum,
            created_at=datetime.now()
        )
        
        # Save migration to file
        self._save_migration_file(migration)
        
        return migration
    
    def _save_migration_file(self, migration: Migration) -> None:
        """Save migration to file."""
        migration_file = self.migrations_dir / f"{migration.version}_{migration.name}.json"
        
        migration_data = {
            "version": migration.version,
            "name": migration.name,
            "description": migration.description,
            "sql_up": migration.sql_up,
            "sql_down": migration.sql_down,
            "checksum": migration.checksum,
            "created_at": migration.created_at.isoformat()
        }
        
        with open(migration_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(migration_data, f, indent=2)
        
        logger.info(f"✓ Migration file created: {migration_file}")
    
    def load_migrations_from_files(self) -> List[Migration]:
        """Load migrations from files."""
        migrations = []
        
        for migration_file in sorted(self.migrations_dir.glob("*.json")):
            try:
                with open(migration_file, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    data = json.load(f)
                
                migration = Migration(
                    id=0,
                    version=data["version"],
                    name=data["name"],
                    description=data["description"],
                    sql_up=data["sql_up"],
                    sql_down=data["sql_down"],
                    checksum=data["checksum"],
                    created_at=datetime.fromisoformat(data["created_at"])
                )
                
                migrations.append(migration)
                
            except Exception as e:
                logger.error(f"Error loading migration file {migration_file}: {e}")
        
        return migrations
    
    async def get_executed_migrations(self) -> List[Migration]:
        """Get migrations that have been executed."""
        try:
            async with self.db_manager.get_session() as session:
                result = await session.execute(text(
                    "SELECT * FROM migrations WHERE status = 'completed' ORDER BY version"
                ))
                
                migrations = []
                for row in result:
                    migration = Migration(
                        id=row.id,
                        version=row.version,
                        name=row.name,
                        description=row.description,
                        sql_up=row.sql_up,
                        sql_down=row.sql_down,
                        checksum=row.checksum,
                        created_at=row.created_at,
                        executed_at=row.executed_at,
                        status=MigrationStatus(row.status),
                        execution_time=row.execution_time,
                        error_message=row.error_message
                    )
                    migrations.append(migration)
                
                return migrations
                
        except Exception as e:
            logger.error(f"Error getting executed migrations: {e}")
            return []
    
    async def get_pending_migrations(self) -> List[Migration]:
        """Get migrations that are pending execution."""
        executed_versions = {m.version for m in await self.get_executed_migrations()}
        all_migrations = self.load_migrations_from_files()
        
        return [m for m in all_migrations if m.version not in executed_versions]
    
    async def execute_migration(self, migration: Migration) -> bool:
        """Execute a migration."""
        try:
            start_time = datetime.now()
            
            async with self.db_manager.get_session() as session:
                # Update migration status to running
                await session.execute(text("""
                    INSERT INTO migrations (version, name, description, sql_up, sql_down, checksum, status)
                    VALUES (:version, :name, :description, :sql_up, :sql_down, :checksum, 'running')
                    ON CONFLICT (version) DO UPDATE SET status = 'running'
                """), {
                    "version": migration.version,
                    "name": migration.name,
                    "description": migration.description,
                    "sql_up": migration.sql_up,
                    "sql_down": migration.sql_down,
                    "checksum": migration.checksum
                })
                
                # Execute the migration SQL
                await session.execute(text(migration.sql_up))
                await session.commit()
                
                # Update migration status to completed
                execution_time = (datetime.now() - start_time).total_seconds()
                await session.execute(text("""
                    UPDATE migrations 
                    SET status = 'completed', executed_at = :executed_at, execution_time = :execution_time
                    WHERE version = :version
                """), {
                    "executed_at": datetime.now(),
                    "execution_time": execution_time,
                    "version": migration.version
                })
                await session.commit()
                
                logger.info(f"✓ Migration {migration.version} executed successfully in {execution_time:.3f}s")
                return True
                
        except Exception as e:
            # Update migration status to failed
            try:
                async with self.db_manager.get_session() as session:
                    await session.execute(text("""
                        UPDATE migrations 
                        SET status = 'failed', error_message = :error_message
                        WHERE version = :version
                    """), {
                        "error_message": str(e),
                        "version": migration.version
                    })
                    await session.commit()
            except:
                pass
            
            logger.error(f"✗ Migration {migration.version} failed: {e}")
            return False
    
    async def rollback_migration(self, version: str) -> bool:
        """Rollback a specific migration."""
        try:
            # Get the migration
            async with self.db_manager.get_session() as session:
                result = await session.execute(text(
                    "SELECT * FROM migrations WHERE version = :version AND status = 'completed'"
                ), {"version": version})
                
                row = result.fetchone()
                if not row:
                    logger.error(f"Migration {version} not found or not completed")
                    return False
                
                migration = Migration(
                    id=row.id,
                    version=row.version,
                    name=row.name,
                    description=row.description,
                    sql_up=row.sql_up,
                    sql_down=row.sql_down,
                    checksum=row.checksum,
                    created_at=row.created_at,
                    executed_at=row.executed_at,
                    status=MigrationStatus(row.status),
                    execution_time=row.execution_time,
                    error_message=row.error_message
                )
            
            # Execute rollback
            start_time = datetime.now()
            
            async with self.db_manager.get_session() as session:
                # Execute the rollback SQL
                await session.execute(text(migration.sql_down))
                await session.commit()
                
                # Update migration status to rolled_back
                execution_time = (datetime.now() - start_time).total_seconds()
                await session.execute(text("""
                    UPDATE migrations 
                    SET status = 'rolled_back', execution_time = :execution_time
                    WHERE version = :version
                """), {
                    "execution_time": execution_time,
                    "version": version
                })
                await session.commit()
                
                logger.info(f"✓ Migration {version} rolled back successfully in {execution_time:.3f}s")
                return True
                
        except Exception as e:
            logger.error(f"✗ Failed to rollback migration {version}: {e}")
            return False
    
    async def migrate(self, target_version: Optional[str] = None) -> Dict[str, Any]:
        """Run all pending migrations up to target version."""
        pending_migrations = await self.get_pending_migrations()
        
        if not pending_migrations:
            return {
                "success": True,
                "message": "No pending migrations",
                "executed": 0,
                "failed": 0
            }
        
        # Filter by target version if specified
        if target_version:
            pending_migrations = [m for m in pending_migrations if m.version <= target_version]
        
        executed = 0
        failed = 0
        
        for migration in sorted(pending_migrations, key=lambda m: m.version):
            if await self.execute_migration(migration):
                executed += 1
            else:
                failed += 1
                # Stop on first failure
                break
        
        return {
            "success": failed == 0,
            "executed": executed,
            "failed": failed,
            "total": len(pending_migrations)
        }
    
    async def get_migration_status(self) -> Dict[str, Any]:
        """Get current migration status."""
        executed_migrations = await self.get_executed_migrations()
        pending_migrations = await self.get_pending_migrations()
        
        return {
            "executed_count": len(executed_migrations),
            "pending_count": len(pending_migrations),
            "latest_version": executed_migrations[-1].version if executed_migrations else None,
            "executed_migrations": [
                {
                    "version": m.version,
                    "name": m.name,
                    "executed_at": m.executed_at.isoformat() if m.executed_at else None,
                    "execution_time": m.execution_time
                }
                for m in executed_migrations
            ],
            "pending_migrations": [
                {
                    "version": m.version,
                    "name": m.name,
                    "description": m.description
                }
                for m in pending_migrations
            ]
        }


# Predefined migrations for common schema changes
class CommonMigrations:
    """Common migration templates."""
    
    @staticmethod
    def create_users_table() -> str:
        """Create users table migration."""
        return """
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            username VARCHAR(50) UNIQUE NOT NULL,
            email VARCHAR(100) UNIQUE NOT NULL,
            api_key VARCHAR(255) UNIQUE NOT NULL,
            is_active BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
        CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
        CREATE INDEX IF NOT EXISTS idx_users_api_key ON users(api_key);
        """
    
    @staticmethod
    def create_videos_table() -> str:
        """Create videos table migration."""
        return """
        CREATE TABLE IF NOT EXISTS videos (
            id SERIAL PRIMARY KEY,
            video_id VARCHAR(100) UNIQUE NOT NULL,
            user_id INTEGER NOT NULL REFERENCES users(id),
            script TEXT NOT NULL,
            voice_id VARCHAR(50) NOT NULL,
            language VARCHAR(10) DEFAULT 'en',
            quality VARCHAR(20) DEFAULT 'medium',
            status VARCHAR(20) DEFAULT 'processing',
            file_path VARCHAR(500),
            file_size INTEGER,
            processing_time FLOAT,
            metadata TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_videos_video_id ON videos(video_id);
        CREATE INDEX IF NOT EXISTS idx_videos_user_id ON videos(user_id);
        CREATE INDEX IF NOT EXISTS idx_videos_status ON videos(status);
        CREATE INDEX IF NOT EXISTS idx_videos_created_at ON videos(created_at);
        """
    
    @staticmethod
    def create_model_usage_table() -> str:
        """Create model usage table migration."""
        return """
        CREATE TABLE IF NOT EXISTS model_usage (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users(id),
            video_id INTEGER NOT NULL REFERENCES videos(id),
            model_type VARCHAR(50) NOT NULL,
            processing_time FLOAT NOT NULL,
            memory_usage FLOAT,
            gpu_usage FLOAT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_model_usage_user_id ON model_usage(user_id);
        CREATE INDEX IF NOT EXISTS idx_model_usage_video_id ON model_usage(video_id);
        CREATE INDEX IF NOT EXISTS idx_model_usage_model_type ON model_usage(model_type);
        CREATE INDEX IF NOT EXISTS idx_model_usage_created_at ON model_usage(created_at);
        """
    
    @staticmethod
    def drop_tables() -> str:
        """Drop all tables migration."""
        return """
        DROP TABLE IF EXISTS model_usage CASCADE;
        DROP TABLE IF EXISTS videos CASCADE;
        DROP TABLE IF EXISTS users CASCADE;
        """ 