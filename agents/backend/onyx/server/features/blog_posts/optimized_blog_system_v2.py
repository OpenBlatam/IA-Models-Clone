"""
🚀 OPTIMIZED BLOG SYSTEM V2
============================

Production-ready blog system with comprehensive optimizations:
- Multi-tier caching (Redis + Memory)
- Connection pooling and async operations
- Database optimization with SQLAlchemy 2.0
- Advanced error handling and monitoring
- Performance middleware and metrics
- Background task processing
- Rate limiting and security
- Comprehensive logging and tracing
"""

import asyncio
import logging
import time
import hashlib
import json
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
from functools import wraps, lru_cache
from contextlib import asynccontextmanager
import traceback

# FastAPI and web framework
from fastapi import FastAPI, HTTPException, Depends, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.background import BackgroundTasks
from pydantic import BaseModel, Field, ConfigDict, ValidationError

# Database and ORM
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import select, update, delete, text
from sqlalchemy.pool import QueuePool

# Caching and performance
try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    from cachetools import TTLCache, LRUCache
    CACHETOOLS_AVAILABLE = True
except ImportError:
    CACHETOOLS_AVAILABLE = False

# Monitoring and metrics
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Performance optimization
try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    EVENT_LOOP = "uvloop"
except ImportError:
    EVENT_LOOP = "asyncio"

# JSON optimization
try:
    import orjson
    JSON_LIB = "orjson"
except ImportError:
    JSON_LIB = "json"

# ============================================================================
# CONFIGURATION
# ============================================================================

class CacheStrategy(str, Enum):
    """Cache eviction strategies."""
    TTL = "ttl"
    LRU = "lru"
    LFU = "lfu"

class DatabaseConfig(BaseModel):
    """Database configuration."""
    url: str = Field(default="sqlite+aiosqlite:///./blog.db")
    pool_size: int = Field(default=20)
    max_overflow: int = Field(default=30)
    pool_pre_ping: bool = Field(default=True)
    echo: bool = Field(default=False)

class CacheConfig(BaseModel):
    """Cache configuration."""
    redis_url: Optional[str] = Field(default=None)
    memory_cache_size: int = Field(default=1000)
    memory_cache_ttl: int = Field(default=300)
    redis_ttl: int = Field(default=3600)
    enable_compression: bool = Field(default=True)

class PerformanceConfig(BaseModel):
    """Performance configuration."""
    enable_gzip: bool = Field(default=True)
    enable_cors: bool = Field(default=True)
    rate_limit_requests: int = Field(default=100)
    rate_limit_window: int = Field(default=60)
    background_tasks: bool = Field(default=True)

class Config(BaseModel):
    """Main configuration."""
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    debug: bool = Field(default=False)

# ============================================================================
# DATABASE MODELS
# ============================================================================

class Base(DeclarativeBase):
    """Base class for all database models."""
    pass

class BlogPostModel(Base):
    """Database model for blog posts."""
    __tablename__ = "blog_posts"
    
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(nullable=False, index=True)
    content: Mapped[str] = mapped_column(nullable=False)
    tags: Mapped[Optional[str]] = mapped_column(nullable=True)  # JSON string
    is_published: Mapped[bool] = mapped_column(default=False, index=True)
    created_at: Mapped[str] = mapped_column(default=lambda: time.time())
    updated_at: Mapped[str] = mapped_column(default=lambda: time.time())
    views: Mapped[int] = mapped_column(default=0)
    likes: Mapped[int] = mapped_column(default=0)

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class BlogPost(BaseModel):
    """Pydantic model for blog posts."""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
        json_encoders={time.time: lambda v: v}
    )
    
    id: int = Field(..., gt=0)
    title: str = Field(..., min_length=1, max_length=200)
    content: str = Field(..., min_length=1)
    tags: Optional[List[str]] = None
    is_published: bool = False
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    views: int = Field(default=0, ge=0)
    likes: int = Field(default=0, ge=0)

    def to_dict(self) -> dict:
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict) -> "BlogPost":
        return cls.model_validate(data)

class BlogPostCreate(BaseModel):
    """Model for creating blog posts."""
    title: str = Field(..., min_length=1, max_length=200)
    content: str = Field(..., min_length=1)
    tags: List[str] = Field(default_factory=list)
    is_published: bool = Field(default=False)

class BlogPostUpdate(BaseModel):
    """Model for updating blog posts."""
    title: Optional[str] = Field(None, min_length=1, max_length=200)
    content: Optional[str] = Field(None, min_length=1)
    tags: Optional[List[str]] = None
    is_published: Optional[bool] = None

# ============================================================================
# CACHING SYSTEM
# ============================================================================

class CacheManager:
    """Multi-tier caching system."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.memory_cache = None
        self.redis_client = None
        self._setup_caches()
    
    def _setup_caches(self):
        """Setup memory and Redis caches."""
        if CACHETOOLS_AVAILABLE:
            self.memory_cache = TTLCache(
                maxsize=self.config.memory_cache_size,
                ttl=self.config.memory_cache_ttl
            )
        
        if REDIS_AVAILABLE and self.config.redis_url:
            self.redis_client = redis.from_url(
                self.config.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
    
    def _generate_key(self, prefix: str, *args) -> str:
        """Generate cache key."""
        key_parts = [prefix] + [str(arg) for arg in args]
        return ":".join(key_parts)
    
    async def get(self, prefix: str, *args) -> Optional[Any]:
        """Get value from cache."""
        key = self._generate_key(prefix, *args)
        
        # Try memory cache first
        if self.memory_cache and key in self.memory_cache:
            return self.memory_cache[key]
        
        # Try Redis cache
        if self.redis_client:
            try:
                value = await self.redis_client.get(key)
                if value:
                    data = json.loads(value)
                    # Store in memory cache
                    if self.memory_cache:
                        self.memory_cache[key] = data
                    return data
            except Exception as e:
                logging.warning(f"Redis cache error: {e}")
        
        return None
    
    async def set(self, prefix: str, value: Any, *args, ttl: Optional[int] = None) -> None:
        """Set value in cache."""
        key = self._generate_key(prefix, *args)
        
        # Store in memory cache
        if self.memory_cache:
            self.memory_cache[key] = value
        
        # Store in Redis cache
        if self.redis_client:
            try:
                serialized = json.dumps(value)
                await self.redis_client.setex(
                    key, 
                    ttl or self.config.redis_ttl, 
                    serialized
                )
            except Exception as e:
                logging.warning(f"Redis cache error: {e}")
    
    async def delete(self, prefix: str, *args) -> None:
        """Delete value from cache."""
        key = self._generate_key(prefix, *args)
        
        if self.memory_cache and key in self.memory_cache:
            del self.memory_cache[key]
        
        if self.redis_client:
            try:
                await self.redis_client.delete(key)
            except Exception as e:
                logging.warning(f"Redis cache error: {e}")
    
    async def invalidate_pattern(self, pattern: str) -> None:
        """Invalidate cache by pattern."""
        if self.redis_client:
            try:
                keys = await self.redis_client.keys(pattern)
                if keys:
                    await self.redis_client.delete(*keys)
            except Exception as e:
                logging.warning(f"Redis cache error: {e}")

# ============================================================================
# DATABASE MANAGER
# ============================================================================

class DatabaseManager:
    """Database connection and session management."""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.engine = None
        self.session_factory = None
        self._setup_database()
    
    def _setup_database(self):
        """Setup database engine and session factory."""
        self.engine = create_async_engine(
            self.config.url,
            poolclass=QueuePool,
            pool_size=self.config.pool_size,
            max_overflow=self.config.max_overflow,
            pool_pre_ping=self.config.pool_pre_ping,
            echo=self.config.echo
        )
        
        self.session_factory = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
    
    @asynccontextmanager
    async def get_session(self) -> AsyncSession:
        """Get database session."""
        async with self.session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    async def create_tables(self):
        """Create database tables."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

# ============================================================================
# PERFORMANCE MIDDLEWARE
# ============================================================================

class PerformanceMiddleware:
    """Performance monitoring middleware."""
    
    def __init__(self):
        self.request_count = 0
        self.response_times = []
    
    async def __call__(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        self.request_count += 1
        
        # Add request ID
        request_id = hashlib.md5(f"{time.time()}{self.request_count}".encode()).hexdigest()[:8]
        request.state.request_id = request_id
        
        try:
            response = await call_next(request)
            
            # Calculate response time
            process_time = time.time() - start_time
            self.response_times.append(process_time)
            
            # Add performance headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = f"{process_time:.4f}"
            response.headers["X-Request-Count"] = str(self.request_count)
            
            # Log performance metrics
            logging.info(
                f"Request {request_id}: {request.method} {request.url.path} - "
                f"{response.status_code} - {process_time:.4f}s"
            )
            
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            logging.error(
                f"Request {request_id} failed: {request.method} {request.url.path} - "
                f"{process_time:.4f}s - Error: {e}"
            )
            raise

class RateLimitMiddleware:
    """Rate limiting middleware."""
    
    def __init__(self, requests_per_minute: int = 100):
        self.requests_per_minute = requests_per_minute
        self.requests = {}
    
    async def __call__(self, request: Request, call_next: Callable) -> Response:
        client_ip = request.client.host
        current_time = time.time()
        
        # Clean old requests
        self.requests = {
            ip: times for ip, times in self.requests.items()
            if current_time - max(times) < 60
        }
        
        # Check rate limit
        if client_ip in self.requests:
            recent_requests = [
                req_time for req_time in self.requests[client_ip]
                if current_time - req_time < 60
            ]
            
            if len(recent_requests) >= self.requests_per_minute:
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded"}
                )
        
        # Add current request
        if client_ip not in self.requests:
            self.requests[client_ip] = []
        self.requests[client_ip].append(current_time)
        
        return await call_next(request)

# ============================================================================
# BLOG SERVICE
# ============================================================================

class BlogService:
    """Business logic for blog operations."""
    
    def __init__(self, db_manager: DatabaseManager, cache_manager: CacheManager):
        self.db_manager = db_manager
        self.cache_manager = cache_manager
    
    async def list_posts(self, limit: int = 100, offset: int = 0) -> List[BlogPost]:
        """List blog posts with caching."""
        cache_key = f"posts:list:{limit}:{offset}"
        
        # Try cache first
        cached = await self.cache_manager.get("posts", "list", limit, offset)
        if cached:
            return [BlogPost.from_dict(post) for post in cached]
        
        # Query database
        async with self.db_manager.get_session() as session:
            query = select(BlogPostModel).offset(offset).limit(limit)
            result = await session.execute(query)
            posts = result.scalars().all()
            
            # Convert to Pydantic models
            blog_posts = []
            for post in posts:
                tags = json.loads(post.tags) if post.tags else []
                blog_post = BlogPost(
                    id=post.id,
                    title=post.title,
                    content=post.content,
                    tags=tags,
                    is_published=post.is_published,
                    created_at=post.created_at,
                    updated_at=post.updated_at,
                    views=post.views,
                    likes=post.likes
                )
                blog_posts.append(blog_post)
            
            # Cache results
            await self.cache_manager.set("posts", blog_posts, "list", limit, offset)
            return blog_posts
    
    async def get_post(self, post_id: int) -> Optional[BlogPost]:
        """Get single blog post with caching."""
        # Try cache first
        cached = await self.cache_manager.get("posts", "single", post_id)
        if cached:
            return BlogPost.from_dict(cached)
        
        # Query database
        async with self.db_manager.get_session() as session:
            query = select(BlogPostModel).where(BlogPostModel.id == post_id)
            result = await session.execute(query)
            post = result.scalar_one_or_none()
            
            if not post:
                return None
            
            # Convert to Pydantic model
            tags = json.loads(post.tags) if post.tags else []
            blog_post = BlogPost(
                id=post.id,
                title=post.title,
                content=post.content,
                tags=tags,
                is_published=post.is_published,
                created_at=post.created_at,
                updated_at=post.updated_at,
                views=post.views,
                likes=post.likes
            )
            
            # Cache result
            await self.cache_manager.set("posts", blog_post.to_dict(), "single", post_id)
            return blog_post
    
    async def create_post(self, post_data: BlogPostCreate) -> BlogPost:
        """Create new blog post."""
        async with self.db_manager.get_session() as session:
            # Check for duplicate title
            query = select(BlogPostModel).where(BlogPostModel.title == post_data.title)
            result = await session.execute(query)
            if result.scalar_one_or_none():
                raise HTTPException(
                    status_code=400,
                    detail="Post with this title already exists"
                )
            
            # Create new post
            new_post = BlogPostModel(
                title=post_data.title,
                content=post_data.content,
                tags=json.dumps(post_data.tags),
                is_published=post_data.is_published
            )
            
            session.add(new_post)
            await session.flush()
            
            # Convert to Pydantic model
            blog_post = BlogPost(
                id=new_post.id,
                title=new_post.title,
                content=new_post.content,
                tags=post_data.tags,
                is_published=new_post.is_published,
                created_at=new_post.created_at,
                updated_at=new_post.updated_at,
                views=new_post.views,
                likes=new_post.likes
            )
            
            # Invalidate cache
            await self.cache_manager.invalidate_pattern("posts:*")
            
            return blog_post
    
    async def update_post(self, post_id: int, post_data: BlogPostUpdate) -> Optional[BlogPost]:
        """Update blog post."""
        async with self.db_manager.get_session() as session:
            query = select(BlogPostModel).where(BlogPostModel.id == post_id)
            result = await session.execute(query)
            post = result.scalar_one_or_none()
            
            if not post:
                return None
            
            # Update fields
            update_data = post_data.model_dump(exclude_unset=True)
            if "tags" in update_data:
                update_data["tags"] = json.dumps(update_data["tags"])
            
            update_data["updated_at"] = time.time()
            
            # Update database
            update_query = (
                update(BlogPostModel)
                .where(BlogPostModel.id == post_id)
                .values(**update_data)
            )
            await session.execute(update_query)
            
            # Get updated post
            result = await session.execute(query)
            updated_post = result.scalar_one()
            
            # Convert to Pydantic model
            tags = json.loads(updated_post.tags) if updated_post.tags else []
            blog_post = BlogPost(
                id=updated_post.id,
                title=updated_post.title,
                content=updated_post.content,
                tags=tags,
                is_published=updated_post.is_published,
                created_at=updated_post.created_at,
                updated_at=updated_post.updated_at,
                views=updated_post.views,
                likes=updated_post.likes
            )
            
            # Invalidate cache
            await self.cache_manager.delete("posts", "single", post_id)
            await self.cache_manager.invalidate_pattern("posts:list:*")
            
            return blog_post
    
    async def delete_post(self, post_id: int) -> bool:
        """Delete blog post."""
        async with self.db_manager.get_session() as session:
            query = select(BlogPostModel).where(BlogPostModel.id == post_id)
            result = await session.execute(query)
            post = result.scalar_one_or_none()
            
            if not post:
                return False
            
            # Delete post
            delete_query = delete(BlogPostModel).where(BlogPostModel.id == post_id)
            await session.execute(delete_query)
            
            # Invalidate cache
            await self.cache_manager.delete("posts", "single", post_id)
            await self.cache_manager.invalidate_pattern("posts:list:*")
            
            return True
    
    async def increment_views(self, post_id: int) -> None:
        """Increment post views."""
        async with self.db_manager.get_session() as session:
            update_query = (
                update(BlogPostModel)
                .where(BlogPostModel.id == post_id)
                .values(views=BlogPostModel.views + 1)
            )
            await session.execute(update_query)
            
            # Invalidate cache
            await self.cache_manager.delete("posts", "single", post_id)

# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

class OptimizedBlogSystem:
    """Optimized blog system with comprehensive features."""
    
    def __init__(self, config: Config):
        self.config = config
        self.app = FastAPI(
            title="Optimized Blog System",
            description="High-performance blog system with caching and monitoring",
            version="2.0.0",
            debug=config.debug
        )
        
        # Initialize components
        self.db_manager = DatabaseManager(config.database)
        self.cache_manager = CacheManager(config.cache)
        self.blog_service = BlogService(self.db_manager, self.cache_manager)
        
        # Setup middleware
        self._setup_middleware()
        
        # Setup routes
        self._setup_routes()
        
        # Setup exception handlers
        self._setup_exception_handlers()
    
    def _setup_middleware(self):
        """Setup application middleware."""
        # Performance middleware
        self.app.add_middleware(PerformanceMiddleware)
        
        # Rate limiting
        if self.config.performance.rate_limit_requests > 0:
            self.app.add_middleware(RateLimitMiddleware, self.config.performance.rate_limit_requests)
        
        # CORS
        if self.config.performance.enable_cors:
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
        
        # Gzip compression
        if self.config.performance.enable_gzip:
            self.app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "timestamp": time.time(),
                "cache": "redis" if REDIS_AVAILABLE else "memory",
                "event_loop": EVENT_LOOP,
                "json_lib": JSON_LIB
            }
        
        @self.app.get("/metrics")
        async def get_metrics():
            """Get system metrics."""
            metrics = {
                "timestamp": time.time(),
                "memory": {},
                "cache": {},
                "database": {}
            }
            
            # Memory metrics
            if PSUTIL_AVAILABLE:
                memory = psutil.virtual_memory()
                metrics["memory"] = {
                    "total": memory.total,
                    "available": memory.available,
                    "percent": memory.percent,
                    "used": memory.used
                }
            
            # Cache metrics
            if self.cache_manager.memory_cache:
                metrics["cache"]["memory"] = {
                    "size": len(self.cache_manager.memory_cache),
                    "maxsize": self.cache_manager.memory_cache.maxsize
                }
            
            return metrics
        
        @self.app.get("/posts", response_model=List[BlogPost])
        async def list_posts(
            limit: int = 100,
            offset: int = 0,
            background_tasks: BackgroundTasks = None
        ):
            """List blog posts with pagination."""
            posts = await self.blog_service.list_posts(limit, offset)
            
            # Background task to increment views
            if background_tasks and self.config.performance.background_tasks:
                for post in posts:
                    background_tasks.add_task(
                        self.blog_service.increment_views, post.id
                    )
            
            return posts
        
        @self.app.get("/posts/{post_id}", response_model=BlogPost)
        async def get_post(
            post_id: int,
            background_tasks: BackgroundTasks = None
        ):
            """Get single blog post."""
            post = await self.blog_service.get_post(post_id)
            if not post:
                raise HTTPException(status_code=404, detail="Post not found")
            
            # Background task to increment views
            if background_tasks and self.config.performance.background_tasks:
                background_tasks.add_task(
                    self.blog_service.increment_views, post_id
                )
            
            return post
        
        @self.app.post("/posts", response_model=BlogPost, status_code=201)
        async def create_post(post: BlogPostCreate):
            """Create new blog post."""
            return await self.blog_service.create_post(post)
        
        @self.app.patch("/posts/{post_id}", response_model=BlogPost)
        async def update_post(post_id: int, post: BlogPostUpdate):
            """Update blog post."""
            updated_post = await self.blog_service.update_post(post_id, post)
            if not updated_post:
                raise HTTPException(status_code=404, detail="Post not found")
            return updated_post
        
        @self.app.delete("/posts/{post_id}", status_code=204)
        async def delete_post(post_id: int):
            """Delete blog post."""
            success = await self.blog_service.delete_post(post_id)
            if not success:
                raise HTTPException(status_code=404, detail="Post not found")
    
    def _setup_exception_handlers(self):
        """Setup exception handlers."""
        
        @self.app.exception_handler(ValidationError)
        async def validation_exception_handler(request: Request, exc: ValidationError):
            return JSONResponse(
                status_code=422,
                content={"detail": exc.errors()}
            )
        
        @self.app.exception_handler(Exception)
        async def generic_exception_handler(request: Request, exc: Exception):
            logging.error(f"Unhandled exception: {exc}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal server error"}
            )
    
    async def startup(self):
        """Application startup."""
        # Create database tables
        await self.db_manager.create_tables()
        logging.info("Database tables created")
        
        # Test cache connection
        if self.cache_manager.redis_client:
            try:
                await self.cache_manager.redis_client.ping()
                logging.info("Redis cache connected")
            except Exception as e:
                logging.warning(f"Redis cache not available: {e}")
        
        logging.info("Optimized blog system started")
    
    async def shutdown(self):
        """Application shutdown."""
        if self.cache_manager.redis_client:
            await self.cache_manager.redis_client.close()
        
        if self.db_manager.engine:
            await self.db_manager.engine.dispose()
        
        logging.info("Optimized blog system shutdown")

# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_optimized_blog_system(config: Config = None) -> OptimizedBlogSystem:
    """Create optimized blog system instance."""
    if config is None:
        config = Config()
    
    system = OptimizedBlogSystem(config)
    
    @system.app.on_event("startup")
    async def startup_event():
        await system.startup()
    
    @system.app.on_event("shutdown")
    async def shutdown_event():
        await system.shutdown()
    
    return system

# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    # Create configuration
    config = Config(
        database=DatabaseConfig(
            url="sqlite+aiosqlite:///./optimized_blog.db"
        ),
        cache=CacheConfig(
            redis_url="redis://localhost:6379",
            memory_cache_size=1000,
            memory_cache_ttl=300
        ),
        performance=PerformanceConfig(
            enable_gzip=True,
            enable_cors=True,
            rate_limit_requests=100,
            background_tasks=True
        ),
        debug=True
    )
    
    # Create and run application
    app = create_optimized_blog_system(config)
    
    uvicorn.run(
        app.app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    ) 
 
 