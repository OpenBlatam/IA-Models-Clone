from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import asyncio
import logging
import os
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
from datetime import datetime
import json
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from alembic.runtime.migration import MigrationContext
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy import text, MetaData
from .sqlalchemy_2_implementation import Base, DatabaseConfig
import asyncio
from logging.config import fileConfig
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config
from alembic import context
from sqlalchemy_2_implementation import Base
from typing import Any, List, Dict, Optional
"""
🗄️ SQLAlchemy 2.0 Migration System
==================================

Production-ready migration system with:
- Alembic integration
- Async migration support
- Data migration utilities
- Rollback capabilities
- Migration testing
- Performance optimization
"""




logger = logging.getLogger(__name__)


class MigrationManager:
    """SQLAlchemy 2.0 migration manager with Alembic integration."""
    
    def __init__(self, db_config: DatabaseConfig, migrations_dir: str = "migrations"):
        
    """__init__ function."""
self.db_config = db_config
        self.migrations_dir = Path(migrations_dir)
        self.alembic_cfg = None
        self.engine = None
        
        # Ensure migrations directory exists
        self.migrations_dir.mkdir(exist_ok=True)
        
        logger.info(f"Migration Manager initialized for {migrations_dir}")
    
    async def initialize(self) -> Any:
        """Initialize migration system."""
        try:
            # Create async engine
            self.engine = create_async_engine(
                self.db_config.url,
                echo=self.db_config.echo
            )
            
            # Setup Alembic configuration
            self._setup_alembic_config()
            
            # Initialize Alembic if not already done
            if not (self.migrations_dir / "versions").exists():
                await self._initialize_alembic()
            
            logger.info("Migration Manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Migration Manager: {e}")
            raise
    
    def _setup_alembic_config(self) -> Any:
        """Setup Alembic configuration."""
        alembic_ini_path = self.migrations_dir / "alembic.ini"
        
        if not alembic_ini_path.exists():
            # Create basic alembic.ini
            self.alembic_cfg = Config()
            self.alembic_cfg.set_main_option("script_location", str(self.migrations_dir))
            self.alembic_cfg.set_main_option("sqlalchemy.url", self.db_config.url)
            self.alembic_cfg.set_main_option("file_template", "%%(year)d%%(month).2d%%(day).2d_%%(hour).2d%%(minute).2d_%%(rev)s_%%(slug)s")
        else:
            self.alembic_cfg = Config(str(alembic_ini_path))
    
    async def _initialize_alembic(self) -> Any:
        """Initialize Alembic for the project."""
        try:
            # Create alembic.ini
            command.init(self.alembic_cfg, str(self.migrations_dir))
            
            # Update env.py for async support
            await self._update_env_py()
            
            # Create initial migration
            await self.create_migration("Initial migration")
            
            logger.info("Alembic initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Alembic: {e}")
            raise
    
    async def _update_env_py(self) -> Any:
        """Update env.py for async SQLAlchemy 2.0 support."""
        env_py_path = self.migrations_dir / "env.py"
        
        if env_py_path.exists():
            # Read current env.py
            with open(env_py_path, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                content = f.read()
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            
            # Update for async support
            async_content = self._get_async_env_py_content()
            
            # Write updated content
            with open(env_py_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                f.write(async_content)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
    
    def _get_async_env_py_content(self) -> str:
        """Get async-enabled env.py content."""
        return '''
"""Async SQLAlchemy 2.0 Alembic Environment Configuration."""


# Import your models

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
target_metadata = Base.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
'''
    
    async def create_migration(self, message: str, autogenerate: bool = True) -> str:
        """Create a new migration."""
        try:
            if autogenerate:
                # Auto-generate migration based on model changes
                command.revision(
                    self.alembic_cfg,
                    message=message,
                    autogenerate=True
                )
            else:
                # Create empty migration
                command.revision(
                    self.alembic_cfg,
                    message=message,
                    autogenerate=False
                )
            
            # Get the latest revision
            script_dir = ScriptDirectory.from_config(self.alembic_cfg)
            latest_revision = script_dir.get_current_head()
            
            logger.info(f"Created migration: {latest_revision}")
            return latest_revision
            
        except Exception as e:
            logger.error(f"Failed to create migration: {e}")
            raise
    
    async def upgrade(self, revision: str = "head") -> bool:
        """Upgrade database to specified revision."""
        try:
            command.upgrade(self.alembic_cfg, revision)
            logger.info(f"Database upgraded to revision: {revision}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upgrade database: {e}")
            return False
    
    async def downgrade(self, revision: str) -> bool:
        """Downgrade database to specified revision."""
        try:
            command.downgrade(self.alembic_cfg, revision)
            logger.info(f"Database downgraded to revision: {revision}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to downgrade database: {e}")
            return False
    
    async def current_revision(self) -> Optional[str]:
        """Get current database revision."""
        try:
            async with self.engine.begin() as conn:
                context = MigrationContext.configure(conn)
                return context.get_current_revision()
                
        except Exception as e:
            logger.error(f"Failed to get current revision: {e}")
            return None
    
    async def migration_history(self) -> List[Dict[str, Any]]:
        """Get migration history."""
        try:
            script_dir = ScriptDirectory.from_config(self.alembic_cfg)
            revisions = []
            
            for revision in script_dir.walk_revisions():
                revisions.append({
                    "revision": revision.revision,
                    "down_revision": revision.down_revision,
                    "message": revision.doc,
                    "date": revision.date
                })
            
            return revisions
            
        except Exception as e:
            logger.error(f"Failed to get migration history: {e}")
            return []
    
    async def check_migrations(self) -> Dict[str, Any]:
        """Check migration status."""
        try:
            current_rev = await self.current_revision()
            script_dir = ScriptDirectory.from_config(self.alembic_cfg)
            head_revision = script_dir.get_current_head()
            
            return {
                "current_revision": current_rev,
                "head_revision": head_revision,
                "is_up_to_date": current_rev == head_revision,
                "pending_migrations": current_rev != head_revision
            }
            
        except Exception as e:
            logger.error(f"Failed to check migrations: {e}")
            return {
                "current_revision": None,
                "head_revision": None,
                "is_up_to_date": False,
                "pending_migrations": False,
                "error": str(e)
            }
    
    async def cleanup(self) -> Any:
        """Cleanup migration manager."""
        if self.engine:
            await self.engine.dispose()


class DataMigration:
    """Data migration utilities."""
    
    def __init__(self, engine: AsyncEngine):
        
    """__init__ function."""
self.engine = engine
    
    async def migrate_data(self, migration_func: Callable, **kwargs) -> bool:
        """Execute data migration function."""
        try:
            async with self.engine.begin() as conn:
                await migration_func(conn, **kwargs)
            return True
            
        except Exception as e:
            logger.error(f"Data migration failed: {e}")
            return False
    
    async def backup_table(self, table_name: str, backup_suffix: str = "_backup") -> str:
        """Create table backup."""
        backup_table = f"{table_name}{backup_suffix}"
        
        try:
            async with self.engine.begin() as conn:
                # Create backup table
                await conn.execute(text(f"CREATE TABLE {backup_table} AS SELECT * FROM {table_name}"))
                logger.info(f"Created backup table: {backup_table}")
                return backup_table
                
        except Exception as e:
            logger.error(f"Failed to create backup table: {e}")
            raise
    
    async def restore_table(self, table_name: str, backup_table: str) -> bool:
        """Restore table from backup."""
        try:
            async with self.engine.begin() as conn:
                # Drop current table
                await conn.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
                
                # Restore from backup
                await conn.execute(text(f"CREATE TABLE {table_name} AS SELECT * FROM {backup_table}"))
                
                logger.info(f"Restored table {table_name} from {backup_table}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to restore table: {e}")
            return False


# ============================================================================
# Migration Utilities
# ============================================================================

async def create_migration_manager(db_config: DatabaseConfig) -> MigrationManager:
    """Create and initialize migration manager."""
    manager = MigrationManager(db_config)
    await manager.initialize()
    return manager


async def run_migrations(db_config: DatabaseConfig, target_revision: str = "head") -> bool:
    """Run database migrations."""
    manager = await create_migration_manager(db_config)
    
    try:
        # Check current status
        status = await manager.check_migrations()
        logger.info(f"Migration status: {status}")
        
        if not status["is_up_to_date"]:
            # Run migrations
            success = await manager.upgrade(target_revision)
            if success:
                logger.info("Migrations completed successfully")
            else:
                logger.error("Migrations failed")
            return success
        else:
            logger.info("Database is up to date")
            return True
            
    finally:
        await manager.cleanup()


async def create_initial_migration(db_config: DatabaseConfig) -> str:
    """Create initial migration for the project."""
    manager = await create_migration_manager(db_config)
    
    try:
        revision = await manager.create_migration("Initial migration")
        logger.info(f"Created initial migration: {revision}")
        return revision
        
    finally:
        await manager.cleanup()


# ============================================================================
# Example Usage
# ============================================================================

async def example_migration_usage():
    """Example usage of migration system."""
    
    # Create database configuration
    db_config = DatabaseConfig(
        url="postgresql+asyncpg://user:password@localhost/nlp_db",
        pool_size=5
    )
    
    # Create migration manager
    manager = await create_migration_manager(db_config)
    
    try:
        # Check migration status
        status = await manager.check_migrations()
        print(f"Migration status: {status}")
        
        if not status["is_up_to_date"]:
            # Run migrations
            success = await manager.upgrade()
            if success:
                print("Database migrated successfully")
            else:
                print("Migration failed")
        
        # Get migration history
        history = await manager.migration_history()
        print(f"Migration history: {len(history)} migrations")
        
    finally:
        await manager.cleanup()


match __name__:
    case "__main__":
    asyncio.run(example_migration_usage()) 