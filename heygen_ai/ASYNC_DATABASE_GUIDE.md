# Async Database System Guide for HeyGen AI API

A comprehensive guide for implementing and using async database libraries (asyncpg, aiomysql, aiosqlite) in the HeyGen AI FastAPI backend.

## 🚀 Overview

The async database system provides:
- **Multiple Database Support**: PostgreSQL (asyncpg), MySQL (aiomysql), SQLite (aiosqlite)
- **Connection Pooling**: Optimized connection management with health monitoring
- **Repository Pattern**: Clean separation of data access logic
- **Migration System**: Version-controlled schema management
- **Health Monitoring**: Real-time database health checks
- **Failover Support**: Automatic failover between database instances

## 📋 Table of Contents

1. [Installation & Dependencies](#installation--dependencies)
2. [Database Configuration](#database-configuration)
3. [Connection Pooling](#connection-pooling)
4. [Repository Pattern](#repository-pattern)
5. [Migration System](#migration-system)
6. [Health Monitoring](#health-monitoring)
7. [Performance Optimization](#performance-optimization)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)

## 🔧 Installation & Dependencies

### Required Dependencies

```bash
# Core async database drivers
pip install asyncpg>=0.29.0  # PostgreSQL
pip install aiomysql>=0.2.0  # MySQL
pip install aiosqlite>=0.19.0  # SQLite

# SQLAlchemy async support
pip install sqlalchemy>=2.0.23
pip install alembic>=1.13.0

# Additional utilities
pip install redis>=5.0.1
pip install structlog>=23.2.0
```

### Environment Variables

```bash
# Database Configuration
DATABASE_TYPE=postgresql  # postgresql, mysql, sqlite
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/heygen_ai
DATABASE_HOST=localhost
DATABASE_PORT=5432
DATABASE_NAME=heygen_ai
DATABASE_USER=postgres
DATABASE_PASSWORD=your_password

# Connection Pool Settings
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=30
DATABASE_POOL_TIMEOUT=30
DATABASE_POOL_RECYCLE=3600

# Health Monitoring
DATABASE_HEALTH_CHECK_INTERVAL=30
DATABASE_HEALTH_CHECK_ENABLED=true
```

## 🗄️ Database Configuration

### PostgreSQL Configuration

```python
from api.core.async_database import create_postgresql_config, AsyncDatabaseManager

# Basic PostgreSQL configuration
config = create_postgresql_config(
    host="localhost",
    port=5432,
    database="heygen_ai",
    username="postgres",
    password="your_password",
    pool_size=20,
    max_overflow=30
)

# Advanced PostgreSQL configuration
config = create_postgresql_config(
    host="localhost",
    port=5432,
    database="heygen_ai",
    username="postgres",
    password="your_password",
    pool_size=50,
    max_overflow=100,
    pool_pre_ping=True,
    pool_recycle=3600,
    connect_args={
        "server_settings": {
            "application_name": "heygen_ai",
            "timezone": "UTC"
        },
        "command_timeout": 60,
        "statement_timeout": 30000
    }
)
```

### MySQL Configuration

```python
from api.core.async_database import create_mysql_config

# Basic MySQL configuration
config = create_mysql_config(
    host="localhost",
    port=3306,
    database="heygen_ai",
    username="root",
    password="your_password",
    pool_size=20,
    max_overflow=30
)

# Advanced MySQL configuration
config = create_mysql_config(
    host="localhost",
    port=3306,
    database="heygen_ai",
    username="root",
    password="your_password",
    pool_size=50,
    max_overflow=100,
    connect_args={
        "charset": "utf8mb4",
        "autocommit": False,
        "sql_mode": "STRICT_TRANS_TABLES,NO_ZERO_DATE,NO_ZERO_IN_DATE,ERROR_FOR_DIVISION_BY_ZERO"
    }
)
```

### SQLite Configuration

```python
from api.core.async_database import create_sqlite_config

# Basic SQLite configuration
config = create_sqlite_config(
    database_path="heygen_ai.db",
    pool_size=10,
    max_overflow=20
)

# Advanced SQLite configuration
config = create_sqlite_config(
    database_path="heygen_ai.db",
    pool_size=10,
    max_overflow=20,
    connect_args={
        "timeout": 30,
        "check_same_thread": False,
        "isolation_level": None
    }
)
```

## 🔄 Connection Pooling

### Database Manager Setup

```python
from api.core.async_database import AsyncDatabaseManager, DatabaseConnectionPool

# Single database setup
db_manager = AsyncDatabaseManager(config)
await db_manager.initialize()

# Multiple database setup with failover
db_pool = DatabaseConnectionPool()

# Add primary database
primary_config = create_postgresql_config(
    host="primary-db.example.com",
    database="heygen_ai",
    username="user",
    password="pass"
)
await db_pool.add_database("primary", primary_config, is_primary=True)

# Add failover database
failover_config = create_postgresql_config(
    host="failover-db.example.com",
    database="heygen_ai",
    username="user",
    password="pass"
)
await db_pool.add_database("failover", failover_config, is_primary=False)
```

### Connection Pool Monitoring

```python
# Get connection statistics
stats = await db_manager.get_connection_stats()
print(f"Active connections: {stats.active_connections}")
print(f"Idle connections: {stats.idle_connections}")
print(f"Total connections: {stats.total_connections}")

# Check database health
is_healthy = await db_manager.is_healthy()
if not is_healthy:
    logger.warning("Database is unhealthy!")

# Get database information
info = await db_manager.get_database_info()
print(f"Database type: {info['type']}")
print(f"Version: {info['version']}")
print(f"Table count: {info['table_count']}")
```

## 🏗️ Repository Pattern

### Base Repository Usage

```python
from api.core.repositories import BaseRepository
from api.database import User

# Create user repository
user_repo = BaseRepository(db_manager, User)

# Basic CRUD operations
user = await user_repo.create(
    username="john_doe",
    email="john@example.com",
    api_key="api_key_123"
)

user = await user_repo.get_by_id(1)
user = await user_repo.get_by_field("username", "john_doe")

users = await user_repo.get_all(limit=10, offset=0, order_by="created_at")

updated_user = await user_repo.update(1, email="new_email@example.com")

success = await user_repo.delete(1)

# Bulk operations
users_data = [
    {"username": "user1", "email": "user1@example.com", "api_key": "key1"},
    {"username": "user2", "email": "user2@example.com", "api_key": "key2"}
]
users = await user_repo.bulk_create(users_data)

# Advanced queries
count = await user_repo.count(is_active=True)
exists = await user_repo.exists(username="john_doe")
```

### Specialized Repositories

```python
from api.core.repositories import UserRepository, VideoRepository, ModelUsageRepository

# User repository with user-specific methods
user_repo = UserRepository(db_manager, User)
user = await user_repo.get_by_api_key("api_key_123")
user = await user_repo.get_by_email("john@example.com")
active_users = await user_repo.get_active_users(limit=100)
stats = await user_repo.get_user_stats(user_id=1)

# Video repository with video-specific methods
video_repo = VideoRepository(db_manager, Video)
video = await video_repo.get_by_video_id("video_123")
user_videos = await video_repo.get_user_videos(user_id=1, status="completed")
processing_videos = await video_repo.get_processing_videos()
video_stats = await video_repo.get_video_stats(user_id=1)

# Model usage repository for analytics
usage_repo = ModelUsageRepository(db_manager, ModelUsage)
await usage_repo.log_usage({
    "user_id": 1,
    "video_id": 1,
    "model_type": "transformer",
    "processing_time": 45.2,
    "memory_usage": 1024.5,
    "gpu_usage": 85.3
})
usage_stats = await usage_repo.get_usage_stats(user_id=1, days=30)
daily_usage = await usage_repo.get_daily_usage(days=7)
```

## 🔄 Migration System

### Creating Migrations

```python
from api.core.migrations import MigrationManager, CommonMigrations

# Initialize migration manager
migration_manager = MigrationManager(db_manager)

# Create a new migration
migration = migration_manager.create_migration(
    name="add_user_preferences",
    description="Add user preferences table",
    sql_up="""
        CREATE TABLE user_preferences (
            id SERIAL PRIMARY KEY,
            user_id INTEGER REFERENCES users(id),
            theme VARCHAR(20) DEFAULT 'light',
            language VARCHAR(10) DEFAULT 'en',
            notifications_enabled BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX idx_user_preferences_user_id ON user_preferences(user_id);
    """,
    sql_down="""
        DROP TABLE IF EXISTS user_preferences CASCADE;
    """
)
```

### Running Migrations

```python
# Run all pending migrations
result = await migration_manager.migrate()
print(f"Executed: {result['executed']}, Failed: {result['failed']}")

# Run migrations up to specific version
result = await migration_manager.migrate(target_version="20231201_120000")

# Get migration status
status = await migration_manager.get_migration_status()
print(f"Executed: {status['executed_count']}, Pending: {status['pending_count']}")

# Rollback specific migration
success = await migration_manager.rollback_migration("20231201_120000")
```

### Common Migration Templates

```python
# Create users table
users_migration = migration_manager.create_migration(
    name="create_users_table",
    description="Create users table",
    sql_up=CommonMigrations.create_users_table(),
    sql_down=CommonMigrations.drop_tables()
)

# Create videos table
videos_migration = migration_manager.create_migration(
    name="create_videos_table",
    description="Create videos table",
    sql_up=CommonMigrations.create_videos_table(),
    sql_down=CommonMigrations.drop_tables()
)

# Create model usage table
usage_migration = migration_manager.create_migration(
    name="create_model_usage_table",
    description="Create model usage tracking table",
    sql_up=CommonMigrations.create_model_usage_table(),
    sql_down=CommonMigrations.drop_tables()
)
```

## 📊 Health Monitoring

### Health Check Endpoints

```python
from fastapi import APIRouter, Depends
from api.core.async_database import check_database_health

router = APIRouter()

@router.get("/health/database")
async def database_health():
    """Check database health status."""
    health_status = await check_database_health()
    return {
        "status": "healthy" if all(db["healthy"] for db in health_status.values()) else "unhealthy",
        "databases": health_status
    }

@router.get("/health/database/stats")
async def database_stats():
    """Get database connection statistics."""
    stats = {}
    for name, db_manager in db_pool.databases.items():
        stats[name] = await db_manager.get_connection_stats()
    return stats

@router.get("/health/database/info")
async def database_info():
    """Get database information."""
    info = {}
    for name, db_manager in db_pool.databases.items():
        info[name] = await db_manager.get_database_info()
    return info
```

### Health Check Integration

```python
# In your main application
from api.core.async_database import db_pool

async def check_readiness() -> bool:
    """Check if system is ready to receive traffic."""
    try:
        # Check if any database is healthy
        healthy_db = await db_pool.get_healthy_database()
        if not healthy_db:
            return False
        
        # Additional checks...
        return True
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return False

async def check_liveness() -> bool:
    """Check if system is alive."""
    try:
        # Simple database connectivity check
        healthy_db = await db_pool.get_healthy_database()
        return healthy_db is not None
    except Exception as e:
        logger.error(f"Liveness check failed: {e}")
        return False
```

## ⚡ Performance Optimization

### Connection Pool Optimization

```python
# Optimize for high concurrency
config = create_postgresql_config(
    host="localhost",
    database="heygen_ai",
    username="user",
    password="pass",
    pool_size=50,  # Increase for high concurrency
    max_overflow=100,  # Allow more connections during peak
    pool_timeout=30,  # Wait up to 30 seconds for connection
    pool_recycle=1800,  # Recycle connections every 30 minutes
    pool_pre_ping=True  # Validate connections before use
)
```

### Query Optimization

```python
# Use selectinload for related data
async def get_user_with_videos(user_id: int):
    async with db_manager.get_session() as session:
        result = await session.execute(
            select(User)
            .options(selectinload(User.videos))
            .where(User.id == user_id)
        )
        return result.scalar_one_or_none()

# Use bulk operations for multiple records
async def bulk_create_videos(videos_data: List[Dict]):
    async with db_manager.get_session() as session:
        videos = [Video(**data) for data in videos_data]
        session.add_all(videos)
        await session.commit()
        return videos

# Use raw SQL for complex queries
async def get_video_stats():
    async with db_manager.get_session() as session:
        result = await session.execute(text("""
            SELECT 
                status,
                COUNT(*) as count,
                AVG(processing_time) as avg_time
            FROM videos 
            GROUP BY status
        """))
        return result.fetchall()
```

### Caching Integration

```python
import redis.asyncio as redis
from functools import wraps

# Redis cache for database queries
redis_client = redis.Redis(host='localhost', port=6379, db=0)

def cache_result(expire_time: int = 300):
    """Cache decorator for database queries."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Create cache key
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Try to get from cache
            cached_result = await redis_client.get(cache_key)
            if cached_result:
                return json.loads(cached_result)
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            await redis_client.setex(
                cache_key, 
                expire_time, 
                json.dumps(result, default=str)
            )
            return result
        return wrapper
    return decorator

# Use caching with repository methods
@cache_result(expire_time=600)
async def get_user_stats(user_id: int):
    return await user_repo.get_user_stats(user_id)
```

## 🛠️ Best Practices

### 1. Connection Management

```python
# Always use context managers for sessions
async def create_user(user_data: Dict):
    async with db_manager.get_session() as session:
        user = User(**user_data)
        session.add(user)
        await session.commit()
        await session.refresh(user)
        return user

# Handle transactions properly
async def transfer_credits(from_user_id: int, to_user_id: int, amount: int):
    async with db_manager.get_session() as session:
        try:
            # Deduct from source user
            from_user = await session.get(User, from_user_id)
            from_user.credits -= amount
            
            # Add to target user
            to_user = await session.get(User, to_user_id)
            to_user.credits += amount
            
            await session.commit()
        except Exception as e:
            await session.rollback()
            raise
```

### 2. Error Handling

```python
from sqlalchemy.exc import SQLAlchemyError, IntegrityError, OperationalError

async def safe_database_operation():
    try:
        async with db_manager.get_session() as session:
            # Database operations
            pass
    except IntegrityError as e:
        logger.error(f"Integrity constraint violated: {e}")
        raise ValueError("Data validation failed")
    except OperationalError as e:
        logger.error(f"Database operation failed: {e}")
        raise RuntimeError("Database temporarily unavailable")
    except SQLAlchemyError as e:
        logger.error(f"Database error: {e}")
        raise RuntimeError("Database error occurred")
```

### 3. Monitoring and Logging

```python
import structlog

logger = structlog.get_logger()

async def monitored_database_operation():
    start_time = time.time()
    try:
        # Database operation
        result = await db_manager.execute_query("SELECT COUNT(*) FROM users")
        execution_time = time.time() - start_time
        
        logger.info(
            "Database operation completed",
            operation="user_count",
            execution_time=execution_time,
            result=result.scalar()
        )
        return result
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(
            "Database operation failed",
            operation="user_count",
            execution_time=execution_time,
            error=str(e)
        )
        raise
```

### 4. Configuration Management

```python
from pydantic_settings import BaseSettings

class DatabaseSettings(BaseSettings):
    database_type: str = "postgresql"
    database_url: str = "postgresql+asyncpg://user:pass@localhost/heygen_ai"
    pool_size: int = 20
    max_overflow: int = 30
    pool_timeout: int = 30
    pool_recycle: int = 3600
    health_check_interval: int = 30
    
    class Config:
        env_prefix = "DATABASE_"

# Use settings in application
settings = DatabaseSettings()

if settings.database_type == "postgresql":
    config = create_postgresql_config(
        url=settings.database_url,
        pool_size=settings.pool_size,
        max_overflow=settings.max_overflow,
        pool_timeout=settings.pool_timeout,
        pool_recycle=settings.pool_recycle
    )
```

## 🔍 Troubleshooting

### Common Issues

#### 1. Connection Pool Exhausted

```python
# Symptoms: "QueuePool limit of size X overflow Y reached"
# Solution: Increase pool size and max overflow

config = create_postgresql_config(
    pool_size=50,  # Increase from default 20
    max_overflow=100,  # Increase from default 30
    pool_timeout=60  # Increase timeout
)
```

#### 2. Connection Timeouts

```python
# Symptoms: "Connection timeout" errors
# Solution: Adjust connection settings

config = create_postgresql_config(
    connect_args={
        "command_timeout": 60,
        "statement_timeout": 30000,
        "connect_timeout": 10
    }
)
```

#### 3. Memory Leaks

```python
# Symptoms: Memory usage increasing over time
# Solution: Ensure proper session cleanup

async def proper_session_usage():
    async with db_manager.get_session() as session:
        # Use session
        result = await session.execute(query)
        return result.scalars().all()
    # Session automatically closed here
```

#### 4. Slow Queries

```python
# Symptoms: High response times
# Solution: Optimize queries and add indexes

# Add database indexes
await db_manager.execute_query("""
    CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_videos_user_status 
    ON videos(user_id, status);
""")

# Use query optimization
async def optimized_query():
    async with db_manager.get_session() as session:
        # Use selectinload for related data
        result = await session.execute(
            select(User)
            .options(selectinload(User.videos))
            .where(User.is_active == True)
            .limit(100)
        )
        return result.scalars().all()
```

### Debugging Tools

```python
# Enable SQL logging
config = create_postgresql_config(
    echo=True,  # Log all SQL statements
    echo_pool=True  # Log connection pool events
)

# Monitor connection pool
async def monitor_pool():
    stats = await db_manager.get_connection_stats()
    logger.info(
        "Connection pool status",
        total=stats.total_connections,
        active=stats.active_connections,
        idle=stats.idle_connections,
        overflow=stats.overflow_connections
    )

# Health check monitoring
async def monitor_health():
    is_healthy = await db_manager.is_healthy()
    if not is_healthy:
        logger.warning("Database health check failed")
        # Send alert or trigger failover
```

## 📚 Additional Resources

- [SQLAlchemy Async Documentation](https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html)
- [asyncpg Documentation](https://asyncpg.readthedocs.io/)
- [aiomysql Documentation](https://aiomysql.readthedocs.io/)
- [aiosqlite Documentation](https://aiosqlite.readthedocs.io/)
- [FastAPI Database Dependencies](https://fastapi.tiangolo.com/tutorial/dependencies/)

## 🚀 Next Steps

1. **Set up your database configuration** using the provided examples
2. **Initialize the migration system** and run initial migrations
3. **Implement repository pattern** for your data models
4. **Add health monitoring** to your application
5. **Configure connection pooling** based on your load requirements
6. **Set up monitoring and alerting** for database health
7. **Implement caching** for frequently accessed data
8. **Add comprehensive error handling** and logging

This async database system provides a robust foundation for building scalable, high-performance applications with proper connection management, health monitoring, and failover capabilities. 