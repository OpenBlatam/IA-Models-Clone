from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import asyncio
import logging
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager
import asyncpg
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy import text, MetaData
import redis.asyncio as redis
from src.core.config import DatabaseSettings, RedisSettings
from src.core.exceptions import DatabaseException
        import hashlib
from typing import Any, List, Dict, Optional
"""
🗄️ Ultra-Optimized Database Manager
===================================

Production-grade database management with:
- Connection pooling
- Async operations
- Query optimization
- Migration support
- Performance monitoring
"""





Base = declarative_base()


class DatabaseManager:
    """
    Ultra-optimized database manager with connection pooling,
    query optimization, and performance monitoring.
    """
    
    def __init__(self, db_settings: DatabaseSettings):
        
    """__init__ function."""
self.settings = db_settings
        self.logger = logging.getLogger(__name__)
        
        # Database engine and session factory
        self.engine = None
        self.session_factory = None
        
        # Connection pool
        self.pool = None
        
        # Performance metrics
        self.query_count = 0
        self.total_query_time = 0.0
        self.connection_count = 0
        self.pool_size = 0
        
        # Health status
        self.is_healthy = False
        self.last_health_check = None
        
        # Query cache
        self.query_cache = {}
        
        self.logger.info("Database Manager initialized")
    
    async def initialize(self) -> Any:
        """Initialize database connections and pools"""
        
        self.logger.info("Initializing Database Manager...")
        
        try:
            # Create async engine
            self.engine = create_async_engine(
                self.settings.URL,
                pool_size=self.settings.POOL_SIZE,
                max_overflow=self.settings.MAX_OVERFLOW,
                pool_timeout=self.settings.POOL_TIMEOUT,
                pool_recycle=self.settings.POOL_RECYCLE,
                echo=self.settings.ECHO,
                future=True
            )
            
            # Create session factory
            self.session_factory = async_sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Test connection
            await self._test_connection()
            
            # Initialize connection pool
            await self._initialize_pool()
            
            # Run migrations
            await self._run_migrations()
            
            # Set health status
            self.is_healthy = True
            self.last_health_check = asyncio.get_event_loop().time()
            
            self.logger.info("Database Manager initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Database Manager: {e}")
            raise DatabaseException("initialization", reason=str(e))
    
    async def cleanup(self) -> Any:
        """Cleanup database connections"""
        
        self.logger.info("Cleaning up Database Manager...")
        
        try:
            # Close session factory
            if self.session_factory:
                await self.session_factory.close_all()
            
            # Close engine
            if self.engine:
                await self.engine.dispose()
            
            # Close connection pool
            if self.pool:
                await self.pool.close()
            
            self.logger.info("Database Manager cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during database cleanup: {e}")
    
    @asynccontextmanager
    async def get_session(self) -> Optional[Dict[str, Any]]:
        """Get database session with automatic cleanup"""
        
        session = None
        start_time = asyncio.get_event_loop().time()
        
        try:
            session = self.session_factory()
            self.connection_count += 1
            yield session
            
        except Exception as e:
            if session:
                await session.rollback()
            self.logger.error(f"Database session error: {e}")
            raise DatabaseException("session operation", reason=str(e))
            
        finally:
            if session:
                await session.close()
            
            # Update metrics
            query_time = asyncio.get_event_loop().time() - start_time
            self.total_query_time += query_time
    
    async def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute raw SQL query with optimization"""
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            async with self.get_session() as session:
                # Check query cache
                cache_key = self._generate_cache_key(query, params)
                if cache_key in self.query_cache:
                    self.logger.debug("Query cache hit")
                    return self.query_cache[cache_key]
                
                # Execute query
                result = await session.execute(text(query), params or {})
                
                # Convert to list of dictionaries
                rows = []
                for row in result:
                    rows.append(dict(row._mapping))
                
                # Cache result (for read-only queries)
                if query.strip().upper().startswith('SELECT'):
                    self.query_cache[cache_key] = rows
                
                # Update metrics
                self.query_count += 1
                
                return rows
                
        except Exception as e:
            self.logger.error(f"Query execution failed: {e}")
            raise DatabaseException("query execution", query=query, reason=str(e))
    
    async def execute_transaction(self, queries: List[Dict[str, Any]]) -> bool:
        """Execute multiple queries in a transaction"""
        
        try:
            async with self.get_session() as session:
                async with session.begin():
                    for query_data in queries:
                        query = query_data['query']
                        params = query_data.get('params', {})
                        
                        await session.execute(text(query), params)
                    
                    return True
                    
        except Exception as e:
            self.logger.error(f"Transaction failed: {e}")
            raise DatabaseException("transaction", reason=str(e))
    
    async def bulk_insert(self, table: str, data: List[Dict[str, Any]]) -> int:
        """Bulk insert data with optimization"""
        
        if not data:
            return 0
        
        try:
            # Prepare bulk insert query
            columns = list(data[0].keys())
            placeholders = [f":{col}" for col in columns]
            
            query = f"""
            INSERT INTO {table} ({', '.join(columns)})
            VALUES ({', '.join(placeholders)})
            """
            
            async with self.get_session() as session:
                async with session.begin():
                    result = await session.execute(text(query), data)
                    await session.commit()
                    
                    return len(data)
                    
        except Exception as e:
            self.logger.error(f"Bulk insert failed: {e}")
            raise DatabaseException("bulk insert", table=table, reason=str(e))
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform database health check"""
        
        try:
            start_time = asyncio.get_event_loop().time()
            
            # Simple query to test connection
            result = await self.execute_query("SELECT 1 as health_check")
            
            response_time = asyncio.get_event_loop().time() - start_time
            
            # Update health status
            self.is_healthy = True
            self.last_health_check = asyncio.get_event_loop().time()
            
            return {
                "status": "healthy",
                "response_time": response_time,
                "connection_count": self.connection_count,
                "pool_size": self.pool_size,
                "query_count": self.query_count,
                "average_query_time": self._get_average_query_time()
            }
            
        except Exception as e:
            self.is_healthy = False
            self.logger.error(f"Health check failed: {e}")
            
            return {
                "status": "unhealthy",
                "error": str(e),
                "last_health_check": self.last_health_check
            }
    
    async def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """Get table information and statistics"""
        
        try:
            # Get table structure
            structure_query = """
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns
            WHERE table_name = :table_name
            ORDER BY ordinal_position
            """
            
            structure = await self.execute_query(structure_query, {"table_name": table_name})
            
            # Get table statistics
            stats_query = """
            SELECT 
                schemaname,
                tablename,
                attname,
                n_distinct,
                correlation
            FROM pg_stats
            WHERE tablename = :table_name
            """
            
            stats = await self.execute_query(stats_query, {"table_name": table_name})
            
            # Get row count
            count_query = f"SELECT COUNT(*) as row_count FROM {table_name}"
            count_result = await self.execute_query(count_query)
            row_count = count_result[0]['row_count'] if count_result else 0
            
            return {
                "table_name": table_name,
                "structure": structure,
                "statistics": stats,
                "row_count": row_count
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get table info: {e}")
            raise DatabaseException("table info", table=table_name, reason=str(e))
    
    async def optimize_table(self, table_name: str) -> Dict[str, Any]:
        """Optimize table performance"""
        
        try:
            async with self.get_session() as session:
                # Analyze table
                await session.execute(text(f"ANALYZE {table_name}"))
                
                # Vacuum table
                await session.execute(text(f"VACUUM {table_name}"))
                
                # Get optimization results
                info = await self.get_table_info(table_name)
                
                return {
                    "table_name": table_name,
                    "optimization": "completed",
                    "table_info": info
                }
                
        except Exception as e:
            self.logger.error(f"Table optimization failed: {e}")
            raise DatabaseException("table optimization", table=table_name, reason=str(e))
    
    async def backup_database(self, backup_path: str) -> bool:
        """Create database backup"""
        
        try:
            # This is a simplified backup implementation
            # In production, you'd use pg_dump or similar tools
            
            self.logger.info(f"Creating database backup to {backup_path}")
            
            # For now, just log the backup request
            # In real implementation, you'd execute pg_dump command
            
            return True
            
        except Exception as e:
            self.logger.error(f"Database backup failed: {e}")
            return False
    
    async def _test_connection(self) -> Any:
        """Test database connection"""
        
        try:
            async with self.get_session() as session:
                result = await session.execute(text("SELECT 1"))
                await result.fetchone()
                
            self.logger.info("Database connection test successful")
            
        except Exception as e:
            self.logger.error(f"Database connection test failed: {e}")
            raise DatabaseException("connection test", reason=str(e))
    
    async def _initialize_pool(self) -> Any:
        """Initialize connection pool"""
        
        try:
            # Create connection pool
            self.pool = await asyncpg.create_pool(
                self.settings.URL,
                min_size=5,
                max_size=self.settings.POOL_SIZE,
                command_timeout=self.settings.POOL_TIMEOUT
            )
            
            self.pool_size = self.settings.POOL_SIZE
            self.logger.info(f"Connection pool initialized with size {self.pool_size}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize connection pool: {e}")
            raise DatabaseException("pool initialization", reason=str(e))
    
    async def _run_migrations(self) -> Any:
        """Run database migrations"""
        
        try:
            # This is a simplified migration system
            # In production, you'd use Alembic or similar
            
            migrations = [
                """
                CREATE TABLE IF NOT EXISTS users (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    email VARCHAR(255) UNIQUE NOT NULL,
                    username VARCHAR(100),
                    full_name VARCHAR(255),
                    role VARCHAR(50) DEFAULT 'user',
                    credits INTEGER DEFAULT 100,
                    max_credits INTEGER DEFAULT 1000,
                    is_active BOOLEAN DEFAULT true,
                    preferred_language VARCHAR(10) DEFAULT 'en',
                    preferred_tone VARCHAR(50) DEFAULT 'professional',
                    content_preferences JSONB DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS content_requests (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id UUID REFERENCES users(id),
                    content_type VARCHAR(50) NOT NULL,
                    prompt TEXT NOT NULL,
                    title VARCHAR(200),
                    keywords TEXT[],
                    tone VARCHAR(50) DEFAULT 'professional',
                    language VARCHAR(10) DEFAULT 'en',
                    word_count INTEGER,
                    target_audience TEXT,
                    brand_voice TEXT,
                    call_to_action TEXT,
                    seo_optimized BOOLEAN DEFAULT true,
                    tags TEXT[],
                    metadata JSONB DEFAULT '{}',
                    status VARCHAR(20) DEFAULT 'pending',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS generated_content (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    request_id UUID REFERENCES content_requests(id),
                    user_id UUID REFERENCES users(id),
                    title VARCHAR(200) NOT NULL,
                    content TEXT NOT NULL,
                    summary TEXT,
                    word_count INTEGER NOT NULL,
                    reading_time INTEGER,
                    seo_score FLOAT,
                    readability_score FLOAT,
                    model_used VARCHAR(100) NOT NULL,
                    model_version VARCHAR(50),
                    generation_time FLOAT,
                    tokens_used INTEGER,
                    confidence_score FLOAT,
                    plagiarism_score FLOAT,
                    status VARCHAR(20) DEFAULT 'completed',
                    metadata JSONB DEFAULT '{}',
                    versions JSONB DEFAULT '[]',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS content_templates (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    name VARCHAR(255) NOT NULL,
                    description TEXT,
                    user_id UUID REFERENCES users(id),
                    prompt_template TEXT NOT NULL,
                    content_type VARCHAR(50) NOT NULL,
                    default_tone VARCHAR(50) DEFAULT 'professional',
                    default_language VARCHAR(10) DEFAULT 'en',
                    parameters JSONB DEFAULT '[]',
                    default_values JSONB DEFAULT '{}',
                    usage_count INTEGER DEFAULT 0,
                    last_used TIMESTAMP,
                    tags TEXT[],
                    is_public BOOLEAN DEFAULT false,
                    is_active BOOLEAN DEFAULT true,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS usage_metrics (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id UUID REFERENCES users(id),
                    date DATE NOT NULL,
                    total_requests INTEGER DEFAULT 0,
                    successful_requests INTEGER DEFAULT 0,
                    failed_requests INTEGER DEFAULT 0,
                    credits_used INTEGER DEFAULT 0,
                    credits_earned INTEGER DEFAULT 0,
                    content_type_breakdown JSONB DEFAULT '{}',
                    average_generation_time FLOAT,
                    total_generation_time FLOAT DEFAULT 0.0,
                    average_seo_score FLOAT,
                    average_readability_score FLOAT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, date)
                )
                """
            ]
            
            async with self.get_session() as session:
                for migration in migrations:
                    await session.execute(text(migration))
                await session.commit()
            
            self.logger.info("Database migrations completed successfully")
            
        except Exception as e:
            self.logger.error(f"Database migrations failed: {e}")
            raise DatabaseException("migrations", reason=str(e))
    
    def _generate_cache_key(self, query: str, params: Optional[Dict[str, Any]]) -> str:
        """Generate cache key for query"""
        
        
        key_data = {
            "query": query,
            "params": params or {}
        }
        
        key_string = str(key_data)
        return f"query_cache:{hashlib.md5(key_string.encode()).hexdigest()}"
    
    def _get_average_query_time(self) -> float:
        """Get average query execution time"""
        
        if self.query_count == 0:
            return 0.0
        
        return self.total_query_time / self.query_count
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get database performance metrics"""
        
        return {
            "query_count": self.query_count,
            "total_query_time": self.total_query_time,
            "average_query_time": self._get_average_query_time(),
            "connection_count": self.connection_count,
            "pool_size": self.pool_size,
            "is_healthy": self.is_healthy,
            "last_health_check": self.last_health_check,
            "cache_size": len(self.query_cache)
        } 