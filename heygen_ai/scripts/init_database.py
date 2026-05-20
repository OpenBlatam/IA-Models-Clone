from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import asyncio
import logging
import os
import sys
from pathlib import Path
from api.core.async_database import (
from api.core.migrations import MigrationManager, CommonMigrations
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
Database Initialization Script for HeyGen AI API
Demonstrates setup and configuration of async database system
"""


# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

    AsyncDatabaseManager,
    DatabaseConnectionPool,
    create_postgresql_config,
    create_mysql_config,
    create_sqlite_config,
    DatabaseType
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def setup_sqlite_database():
    """Setup SQLite database for development."""
    logger.info("Setting up SQLite database...")
    
    config = create_sqlite_config(
        database_path="heygen_ai.db",
        pool_size=10,
        max_overflow=20,
        echo=True  # Enable SQL logging for development
    )
    
    db_manager = AsyncDatabaseManager(config)
    await db_manager.initialize()
    
    # Setup migrations
    migration_manager = MigrationManager(db_manager)
    
    # Create initial schema
    await create_initial_schema(migration_manager)
    
    logger.info("✓ SQLite database setup completed")
    return db_manager


async def setup_postgresql_database():
    """Setup PostgreSQL database for production."""
    logger.info("Setting up PostgreSQL database...")
    
    # Get database configuration from environment
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = int(os.getenv("POSTGRES_PORT", "5432"))
    database = os.getenv("POSTGRES_DB", "heygen_ai")
    username = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD", "")
    
    config = create_postgresql_config(
        host=host,
        port=port,
        database=database,
        username=username,
        password=password,
        pool_size=20,
        max_overflow=30,
        pool_pre_ping=True,
        pool_recycle=3600,
        echo=False  # Disable SQL logging for production
    )
    
    db_manager = AsyncDatabaseManager(config)
    await db_manager.initialize()
    
    # Setup migrations
    migration_manager = MigrationManager(db_manager)
    
    # Create initial schema
    await create_initial_schema(migration_manager)
    
    logger.info("✓ PostgreSQL database setup completed")
    return db_manager


async def setup_mysql_database():
    """Setup MySQL database."""
    logger.info("Setting up MySQL database...")
    
    # Get database configuration from environment
    host = os.getenv("MYSQL_HOST", "localhost")
    port = int(os.getenv("MYSQL_PORT", "3306"))
    database = os.getenv("MYSQL_DB", "heygen_ai")
    username = os.getenv("MYSQL_USER", "root")
    password = os.getenv("MYSQL_PASSWORD", "")
    
    config = create_mysql_config(
        host=host,
        port=port,
        database=database,
        username=username,
        password=password,
        pool_size=20,
        max_overflow=30,
        pool_pre_ping=True,
        pool_recycle=3600,
        echo=False
    )
    
    db_manager = AsyncDatabaseManager(config)
    await db_manager.initialize()
    
    # Setup migrations
    migration_manager = MigrationManager(db_manager)
    
    # Create initial schema
    await create_initial_schema(migration_manager)
    
    logger.info("✓ MySQL database setup completed")
    return db_manager


async def setup_database_pool():
    """Setup database connection pool with multiple databases."""
    logger.info("Setting up database connection pool...")
    
    db_pool = DatabaseConnectionPool()
    
    # Add primary database (PostgreSQL)
    primary_config = create_postgresql_config(
        host=os.getenv("PRIMARY_DB_HOST", "localhost"),
        port=int(os.getenv("PRIMARY_DB_PORT", "5432")),
        database=os.getenv("PRIMARY_DB_NAME", "heygen_ai"),
        username=os.getenv("PRIMARY_DB_USER", "postgres"),
        password=os.getenv("PRIMARY_DB_PASSWORD", ""),
        pool_size=20,
        max_overflow=30
    )
    await db_pool.add_database("primary", primary_config, is_primary=True)
    
    # Add failover database (MySQL)
    failover_config = create_mysql_config(
        host=os.getenv("FAILOVER_DB_HOST", "localhost"),
        port=int(os.getenv("FAILOVER_DB_PORT", "3306")),
        database=os.getenv("FAILOVER_DB_NAME", "heygen_ai"),
        username=os.getenv("FAILOVER_DB_USER", "root"),
        password=os.getenv("FAILOVER_DB_PASSWORD", ""),
        pool_size=15,
        max_overflow=25
    )
    await db_pool.add_database("failover", failover_config, is_primary=False)
    
    logger.info("✓ Database connection pool setup completed")
    return db_pool


async def create_initial_schema(migration_manager: MigrationManager):
    """Create initial database schema."""
    logger.info("Creating initial database schema...")
    
    # Create users table migration
    users_migration = migration_manager.create_migration(
        name="create_users_table",
        description="Create users table for authentication and user management",
        sql_up=CommonMigrations.create_users_table(),
        sql_down=CommonMigrations.drop_tables()
    )
    
    # Create videos table migration
    videos_migration = migration_manager.create_migration(
        name="create_videos_table",
        description="Create videos table for video generation tracking",
        sql_up=CommonMigrations.create_videos_table(),
        sql_down=CommonMigrations.drop_tables()
    )
    
    # Create model usage table migration
    usage_migration = migration_manager.create_migration(
        name="create_model_usage_table",
        description="Create model usage table for analytics and monitoring",
        sql_up=CommonMigrations.create_model_usage_table(),
        sql_down=CommonMigrations.drop_tables()
    )
    
    # Run migrations
    result = await migration_manager.migrate()
    
    if result["success"]:
        logger.info(f"✓ Schema created successfully: {result['executed']} migrations executed")
    else:
        logger.error(f"✗ Schema creation failed: {result['failed']} migrations failed")
        raise RuntimeError("Failed to create database schema")


async def test_database_connection(db_manager: AsyncDatabaseManager):
    """Test database connection and basic operations."""
    logger.info("Testing database connection...")
    
    try:
        # Test basic connectivity
        is_healthy = await db_manager.is_healthy()
        if not is_healthy:
            raise RuntimeError("Database health check failed")
        
        # Test basic query
        result = await db_manager.execute_query("SELECT 1 as test")
        test_value = result.scalar()
        if test_value != 1:
            raise RuntimeError("Basic query test failed")
        
        # Get database info
        info = await db_manager.get_database_info()
        logger.info(f"Database type: {info['type']}")
        logger.info(f"Database version: {info['version']}")
        logger.info(f"Table count: {info['table_count']}")
        
        # Get connection stats
        stats = await db_manager.get_connection_stats()
        logger.info(f"Connection pool: {stats.total_connections} total, {stats.active_connections} active")
        
        logger.info("✓ Database connection test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Database connection test failed: {e}")
        return False


async def create_sample_data(db_manager: AsyncDatabaseManager):
    """Create sample data for testing."""
    logger.info("Creating sample data...")
    
    try:
        # Create sample user
        user_data = {
            "username": "test_user",
            "email": "test@example.com",
            "api_key": "test_api_key_12345",
            "is_active": True
        }
        
        result = await db_manager.execute_query("""
            INSERT INTO users (username, email, api_key, is_active)
            VALUES (:username, :email, :api_key, :is_active)
            RETURNING id
        """, user_data)
        
        user_id = result.scalar()
        logger.info(f"✓ Created sample user with ID: {user_id}")
        
        # Create sample video
        video_data = {
            "video_id": "test_video_001",
            "user_id": user_id,
            "script": "This is a test video script for the HeyGen AI system.",
            "voice_id": "voice_001",
            "language": "en",
            "quality": "medium",
            "status": "completed"
        }
        
        result = await db_manager.execute_query("""
            INSERT INTO videos (video_id, user_id, script, voice_id, language, quality, status)
            VALUES (:video_id, :user_id, :script, :voice_id, :language, :quality, :status)
            RETURNING id
        """, video_data)
        
        video_id = result.scalar()
        logger.info(f"✓ Created sample video with ID: {video_id}")
        
        # Create sample model usage
        usage_data = {
            "user_id": user_id,
            "video_id": video_id,
            "model_type": "transformer",
            "processing_time": 45.2,
            "memory_usage": 1024.5,
            "gpu_usage": 85.3
        }
        
        result = await db_manager.execute_query("""
            INSERT INTO model_usage (user_id, video_id, model_type, processing_time, memory_usage, gpu_usage)
            VALUES (:user_id, :video_id, :model_type, :processing_time, :memory_usage, :gpu_usage)
            RETURNING id
        """, usage_data)
        
        usage_id = result.scalar()
        logger.info(f"✓ Created sample model usage with ID: {usage_id}")
        
        logger.info("✓ Sample data created successfully")
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed to create sample data: {e}")
        return False


async def main():
    """Main initialization function."""
    logger.info("🚀 Starting database initialization...")
    
    # Get database type from environment
    db_type = os.getenv("DATABASE_TYPE", "sqlite").lower()
    
    try:
        if db_type == "postgresql":
            db_manager = await setup_postgresql_database()
        elif db_type == "mysql":
            db_manager = await setup_mysql_database()
        elif db_type == "pool":
            db_pool = await setup_database_pool()
            db_manager = db_pool.databases["primary"]
        else:  # Default to SQLite
            db_manager = await setup_sqlite_database()
        
        # Test database connection
        if not await test_database_connection(db_manager):
            raise RuntimeError("Database connection test failed")
        
        # Create sample data if requested
        if os.getenv("CREATE_SAMPLE_DATA", "false").lower() == "true":
            await create_sample_data(db_manager)
        
        logger.info("✅ Database initialization completed successfully!")
        
        # Keep the connection alive for a moment to show stats
        await asyncio.sleep(2)
        
        # Show final stats
        stats = await db_manager.get_connection_stats()
        logger.info(f"Final connection stats: {stats.total_connections} total, {stats.active_connections} active")
        
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        sys.exit(1)
    finally:
        # Cleanup
        if 'db_manager' in locals():
            await db_manager.close()
        if 'db_pool' in locals():
            await db_pool.close_all()


match __name__:
    case "__main__":
    asyncio.run(main()) 