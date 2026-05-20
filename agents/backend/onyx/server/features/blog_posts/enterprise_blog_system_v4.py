"""
🚀 ENTERPRISE BLOG SYSTEM V4
============================

Advanced enterprise blog system with:
- Multi-tenant architecture with tenant isolation
- JWT authentication and role-based access control
- Content versioning and audit trails
- Advanced caching with cache warming
- Microservices architecture with service discovery
- Distributed tracing and advanced monitoring
- Content scheduling and publishing workflows
- Advanced SEO with structured data
- Social media integration
- A/B testing and conversion tracking
"""

import asyncio
import logging
import time
import hashlib
import json
import uuid
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
from functools import wraps, lru_cache
from contextlib import asynccontextmanager
import traceback
from datetime import datetime, timedelta

# FastAPI and web framework
from fastapi import FastAPI, HTTPException, Depends, Request, Response, status, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.background import BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, ConfigDict, ValidationError

# Database and ORM
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import select, update, delete, text, func, desc, asc
from sqlalchemy.pool import QueuePool

# JWT Authentication
try:
    import jwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False

# Search and analytics
try:
    from elasticsearch import AsyncElasticsearch
    ELASTICSEARCH_AVAILABLE = True
except ImportError:
    ELASTICSEARCH_AVAILABLE = False

# AI and ML
try:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

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
# ENTERPRISE CONFIGURATION
# ============================================================================

class SecurityConfig(BaseModel):
    """Security configuration."""
    jwt_secret: str = Field(default="your-secret-key")
    jwt_algorithm: str = Field(default="HS256")
    jwt_expiration_hours: int = Field(default=24)
    enable_rate_limiting: bool = Field(default=True)
    enable_cors: bool = Field(default=True)
    allowed_origins: List[str] = Field(default=["*"])

class TenantConfig(BaseModel):
    """Multi-tenant configuration."""
    enable_multi_tenancy: bool = Field(default=True)
    tenant_header: str = Field(default="X-Tenant-ID")
    default_tenant: str = Field(default="default")
    tenant_isolation_level: str = Field(default="database")  # database, schema, row

class VersioningConfig(BaseModel):
    """Content versioning configuration."""
    enable_versioning: bool = Field(default=True)
    max_versions_per_post: int = Field(default=10)
    auto_version_on_update: bool = Field(default=True)
    enable_audit_trail: bool = Field(default=True)

class MicroserviceConfig(BaseModel):
    """Microservices configuration."""
    enable_service_discovery: bool = Field(default=True)
    service_registry_url: str = Field(default="http://localhost:8500")
    enable_circuit_breaker: bool = Field(default=True)
    enable_load_balancing: bool = Field(default=True)

class EnterpriseConfig(BaseModel):
    """Enterprise configuration."""
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    analytics: AnalyticsConfig = Field(default_factory=AnalyticsConfig)
    ai: AIConfig = Field(default_factory=AIConfig)
    notifications: NotificationConfig = Field(default_factory=NotificationConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    tenant: TenantConfig = Field(default_factory=TenantConfig)
    versioning: VersioningConfig = Field(default_factory=VersioningConfig)
    microservice: MicroserviceConfig = Field(default_factory=MicroserviceConfig)
    debug: bool = Field(default=False)

# ============================================================================
# ENTERPRISE DATABASE MODELS
# ============================================================================

class Base(DeclarativeBase):
    """Base class for all models."""
    pass

class TenantModel(Base):
    """Tenant model for multi-tenancy."""
    __tablename__ = "tenants"
    
    id: Mapped[str] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(nullable=False)
    domain: Mapped[Optional[str]] = mapped_column(nullable=True)
    settings: Mapped[Optional[str]] = mapped_column(nullable=True)  # JSON string
    created_at: Mapped[float] = mapped_column(default=lambda: time.time())
    is_active: Mapped[bool] = mapped_column(default=True)

class UserModel(Base):
    """User model for authentication."""
    __tablename__ = "users"
    
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(nullable=False, index=True)
    username: Mapped[str] = mapped_column(nullable=False, unique=True, index=True)
    email: Mapped[str] = mapped_column(nullable=False, unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(nullable=False)
    role: Mapped[str] = mapped_column(default="user")  # admin, editor, author, user
    permissions: Mapped[Optional[str]] = mapped_column(nullable=True)  # JSON string
    created_at: Mapped[float] = mapped_column(default=lambda: time.time())
    last_login: Mapped[Optional[float]] = mapped_column(nullable=True)
    is_active: Mapped[bool] = mapped_column(default=True)

class BlogPostModel(Base):
    """Enhanced database model for blog posts with multi-tenancy."""
    __tablename__ = "blog_posts"
    
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(nullable=False, index=True)
    author_id: Mapped[int] = mapped_column(nullable=False, index=True)
    title: Mapped[str] = mapped_column(nullable=False, index=True)
    content: Mapped[str] = mapped_column(nullable=False)
    excerpt: Mapped[Optional[str]] = mapped_column(nullable=True)
    tags: Mapped[Optional[str]] = mapped_column(nullable=True)  # JSON string
    category: Mapped[Optional[str]] = mapped_column(nullable=True, index=True)
    status: Mapped[str] = mapped_column(default="draft")  # draft, published, scheduled, archived
    version: Mapped[int] = mapped_column(default=1)
    parent_version_id: Mapped[Optional[int]] = mapped_column(nullable=True)
    created_at: Mapped[float] = mapped_column(default=lambda: time.time(), index=True)
    updated_at: Mapped[float] = mapped_column(default=lambda: time.time())
    published_at: Mapped[Optional[float]] = mapped_column(nullable=True, index=True)
    scheduled_at: Mapped[Optional[float]] = mapped_column(nullable=True, index=True)
    views: Mapped[int] = mapped_column(default=0, index=True)
    likes: Mapped[int] = mapped_column(default=0)
    shares: Mapped[int] = mapped_column(default=0)
    comments_count: Mapped[int] = mapped_column(default=0)
    reading_time: Mapped[Optional[int]] = mapped_column(nullable=True)
    seo_title: Mapped[Optional[str]] = mapped_column(nullable=True)
    seo_description: Mapped[Optional[str]] = mapped_column(nullable=True)
    seo_keywords: Mapped[Optional[str]] = mapped_column(nullable=True)
    featured_image: Mapped[Optional[str]] = mapped_column(nullable=True)
    structured_data: Mapped[Optional[str]] = mapped_column(nullable=True)  # JSON string

class PostVersionModel(Base):
    """Post version model for content versioning."""
    __tablename__ = "post_versions"
    
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    post_id: Mapped[int] = mapped_column(nullable=False, index=True)
    tenant_id: Mapped[str] = mapped_column(nullable=False, index=True)
    version: Mapped[int] = mapped_column(nullable=False)
    title: Mapped[str] = mapped_column(nullable=False)
    content: Mapped[str] = mapped_column(nullable=False)
    excerpt: Mapped[Optional[str]] = mapped_column(nullable=True)
    tags: Mapped[Optional[str]] = mapped_column(nullable=True)
    category: Mapped[Optional[str]] = mapped_column(nullable=True)
    created_by: Mapped[int] = mapped_column(nullable=False)
    created_at: Mapped[float] = mapped_column(default=lambda: time.time())
    change_summary: Mapped[Optional[str]] = mapped_column(nullable=True)

class AuditLogModel(Base):
    """Audit log model for tracking changes."""
    __tablename__ = "audit_logs"
    
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(nullable=False, index=True)
    user_id: Mapped[int] = mapped_column(nullable=False, index=True)
    action: Mapped[str] = mapped_column(nullable=False)  # create, update, delete, publish
    resource_type: Mapped[str] = mapped_column(nullable=False)  # post, user, tenant
    resource_id: Mapped[Optional[int]] = mapped_column(nullable=True)
    old_values: Mapped[Optional[str]] = mapped_column(nullable=True)  # JSON string
    new_values: Mapped[Optional[str]] = mapped_column(nullable=True)  # JSON string
    ip_address: Mapped[Optional[str]] = mapped_column(nullable=True)
    user_agent: Mapped[Optional[str]] = mapped_column(nullable=True)
    timestamp: Mapped[float] = mapped_column(default=lambda: time.time(), index=True)

class ABTestModel(Base):
    """A/B test model for conversion tracking."""
    __tablename__ = "ab_tests"
    
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(nullable=False, index=True)
    name: Mapped[str] = mapped_column(nullable=False)
    description: Mapped[Optional[str]] = mapped_column(nullable=True)
    variant_a: Mapped[str] = mapped_column(nullable=False)  # JSON string
    variant_b: Mapped[str] = mapped_column(nullable=False)  # JSON string
    start_date: Mapped[float] = mapped_column(nullable=False)
    end_date: Mapped[Optional[float]] = mapped_column(nullable=True)
    is_active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[float] = mapped_column(default=lambda: time.time())

# ============================================================================
# ENTERPRISE PYDANTIC MODELS
# ============================================================================

class User(BaseModel):
    """User model."""
    id: int
    tenant_id: str
    username: str
    email: str
    role: str
    permissions: Optional[List[str]] = None
    created_at: float
    last_login: Optional[float] = None
    is_active: bool

class UserCreate(BaseModel):
    """User creation model."""
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., regex=r"^[^@]+@[^@]+\.[^@]+$")
    password: str = Field(..., min_length=8)
    role: str = Field(default="user")
    permissions: Optional[List[str]] = None

class UserLogin(BaseModel):
    """User login model."""
    username: str
    password: str

class Token(BaseModel):
    """JWT token model."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: User

class BlogPost(BaseModel):
    """Enhanced blog post model."""
    id: int
    tenant_id: str
    author_id: int
    title: str
    content: str
    excerpt: Optional[str] = None
    tags: Optional[List[str]] = None
    category: Optional[str] = None
    status: str
    version: int
    parent_version_id: Optional[int] = None
    created_at: float
    updated_at: float
    published_at: Optional[float] = None
    scheduled_at: Optional[float] = None
    views: int
    likes: int
    shares: int
    comments_count: int
    reading_time: Optional[int] = None
    seo_title: Optional[str] = None
    seo_description: Optional[str] = None
    seo_keywords: Optional[str] = None
    featured_image: Optional[str] = None
    structured_data: Optional[Dict[str, Any]] = None

class BlogPostCreate(BaseModel):
    """Blog post creation model."""
    title: str = Field(..., min_length=1, max_length=200)
    content: str = Field(..., min_length=10)
    excerpt: Optional[str] = Field(None, max_length=500)
    tags: Optional[List[str]] = None
    category: Optional[str] = None
    status: str = Field(default="draft")
    scheduled_at: Optional[float] = None
    seo_title: Optional[str] = None
    seo_description: Optional[str] = None
    seo_keywords: Optional[str] = None
    featured_image: Optional[str] = None
    structured_data: Optional[Dict[str, Any]] = None

class BlogPostUpdate(BaseModel):
    """Blog post update model."""
    title: Optional[str] = Field(None, min_length=1, max_length=200)
    content: Optional[str] = Field(None, min_length=10)
    excerpt: Optional[str] = Field(None, max_length=500)
    tags: Optional[List[str]] = None
    category: Optional[str] = None
    status: Optional[str] = None
    scheduled_at: Optional[float] = None
    seo_title: Optional[str] = None
    seo_description: Optional[str] = None
    seo_keywords: Optional[str] = None
    featured_image: Optional[str] = None
    structured_data: Optional[Dict[str, Any]] = None

class PostVersion(BaseModel):
    """Post version model."""
    id: int
    post_id: int
    version: int
    title: str
    content: str
    excerpt: Optional[str] = None
    tags: Optional[List[str]] = None
    category: Optional[str] = None
    created_by: int
    created_at: float
    change_summary: Optional[str] = None

class AuditLog(BaseModel):
    """Audit log model."""
    id: int
    tenant_id: str
    user_id: int
    action: str
    resource_type: str
    resource_id: Optional[int] = None
    old_values: Optional[Dict[str, Any]] = None
    new_values: Optional[Dict[str, Any]] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    timestamp: float

class ABTest(BaseModel):
    """A/B test model."""
    id: int
    tenant_id: str
    name: str
    description: Optional[str] = None
    variant_a: Dict[str, Any]
    variant_b: Dict[str, Any]
    start_date: float
    end_date: Optional[float] = None
    is_active: bool
    created_at: float

# ============================================================================
# ENTERPRISE SERVICES
# ============================================================================

class SecurityService:
    """Security service for authentication and authorization."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.security = HTTPBearer()
    
    def create_access_token(self, data: Dict[str, Any]) -> str:
        """Create JWT access token."""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(hours=self.config.jwt_expiration_hours)
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.config.jwt_secret, algorithm=self.config.jwt_algorithm)
        return encoded_jwt
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify JWT token."""
        try:
            payload = jwt.decode(token, self.config.jwt_secret, algorithms=[self.config.jwt_algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token has expired")
        except jwt.JWTError:
            raise HTTPException(status_code=401, detail="Could not validate credentials")
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt."""
        import bcrypt
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash."""
        import bcrypt
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

class TenantService:
    """Tenant service for multi-tenancy."""
    
    def __init__(self, db_manager: DatabaseManager, config: TenantConfig):
        self.db_manager = db_manager
        self.config = config
    
    async def get_tenant_from_header(self, request: Request) -> str:
        """Get tenant ID from request header."""
        tenant_id = request.headers.get(self.config.tenant_header, self.config.default_tenant)
        return tenant_id
    
    async def validate_tenant(self, tenant_id: str) -> bool:
        """Validate if tenant exists and is active."""
        async with self.db_manager.get_session() as session:
            result = await session.execute(
                select(TenantModel).where(
                    TenantModel.id == tenant_id,
                    TenantModel.is_active == True
                )
            )
            return result.scalar_one_or_none() is not None
    
    async def create_tenant(self, tenant_id: str, name: str, domain: Optional[str] = None) -> TenantModel:
        """Create a new tenant."""
        async with self.db_manager.get_session() as session:
            tenant = TenantModel(
                id=tenant_id,
                name=name,
                domain=domain,
                settings=json.dumps({})
            )
            session.add(tenant)
            await session.commit()
            await session.refresh(tenant)
            return tenant

class VersioningService:
    """Content versioning service."""
    
    def __init__(self, db_manager: DatabaseManager, config: VersioningConfig):
        self.db_manager = db_manager
        self.config = config
    
    async def create_version(self, post: BlogPostModel, user_id: int, change_summary: Optional[str] = None) -> PostVersionModel:
        """Create a new version of a post."""
        async with self.db_manager.get_session() as session:
            # Get the latest version number
            result = await session.execute(
                select(func.max(PostVersionModel.version))
                .where(PostVersionModel.post_id == post.id)
            )
            latest_version = result.scalar_one_or_none() or 0
            new_version = latest_version + 1
            
            # Create version record
            version = PostVersionModel(
                post_id=post.id,
                tenant_id=post.tenant_id,
                version=new_version,
                title=post.title,
                content=post.content,
                excerpt=post.excerpt,
                tags=post.tags,
                category=post.category,
                created_by=user_id,
                change_summary=change_summary
            )
            session.add(version)
            await session.commit()
            await session.refresh(version)
            return version
    
    async def get_post_versions(self, post_id: int, tenant_id: str) -> List[PostVersion]:
        """Get all versions of a post."""
        async with self.db_manager.get_session() as session:
            result = await session.execute(
                select(PostVersionModel)
                .where(
                    PostVersionModel.post_id == post_id,
                    PostVersionModel.tenant_id == tenant_id
                )
                .order_by(desc(PostVersionModel.version))
            )
            versions = result.scalars().all()
            return [PostVersion.model_validate(version) for version in versions]
    
    async def restore_version(self, post_id: int, version: int, user_id: int) -> BlogPostModel:
        """Restore a post to a specific version."""
        async with self.db_manager.get_session() as session:
            # Get the version to restore
            result = await session.execute(
                select(PostVersionModel)
                .where(
                    PostVersionModel.post_id == post_id,
                    PostVersionModel.version == version
                )
            )
            version_data = result.scalar_one_or_none()
            if not version_data:
                raise HTTPException(status_code=404, detail="Version not found")
            
            # Update the post with version data
            post_result = await session.execute(
                select(BlogPostModel).where(BlogPostModel.id == post_id)
            )
            post = post_result.scalar_one_or_none()
            if not post:
                raise HTTPException(status_code=404, detail="Post not found")
            
            post.title = version_data.title
            post.content = version_data.content
            post.excerpt = version_data.excerpt
            post.tags = version_data.tags
            post.category = version_data.category
            post.updated_at = time.time()
            post.version += 1
            
            await session.commit()
            await session.refresh(post)
            return post

class AuditService:
    """Audit service for tracking changes."""
    
    def __init__(self, db_manager: DatabaseManager, config: VersioningConfig):
        self.db_manager = db_manager
        self.config = config
    
    async def log_action(self, tenant_id: str, user_id: int, action: str, 
                        resource_type: str, resource_id: Optional[int] = None,
                        old_values: Optional[Dict[str, Any]] = None,
                        new_values: Optional[Dict[str, Any]] = None,
                        request: Optional[Request] = None) -> None:
        """Log an audit action."""
        if not self.config.enable_audit_trail:
            return
        
        async with self.db_manager.get_session() as session:
            audit_log = AuditLogModel(
                tenant_id=tenant_id,
                user_id=user_id,
                action=action,
                resource_type=resource_type,
                resource_id=resource_id,
                old_values=json.dumps(old_values) if old_values else None,
                new_values=json.dumps(new_values) if new_values else None,
                ip_address=request.client.host if request else None,
                user_agent=request.headers.get("user-agent") if request else None
            )
            session.add(audit_log)
            await session.commit()

class EnterpriseBlogService:
    """Enhanced blog service with enterprise features."""
    
    def __init__(self, db_manager: DatabaseManager, cache_manager: CacheManager,
                 search_service: SearchService, analytics_service: AnalyticsService,
                 ai_service: AIService, notification_service: NotificationService,
                 security_service: SecurityService, tenant_service: TenantService,
                 versioning_service: VersioningService, audit_service: AuditService):
        self.db_manager = db_manager
        self.cache_manager = cache_manager
        self.search_service = search_service
        self.analytics_service = analytics_service
        self.ai_service = ai_service
        self.notification_service = notification_service
        self.security_service = security_service
        self.tenant_service = tenant_service
        self.versioning_service = versioning_service
        self.audit_service = audit_service
    
    async def list_posts(self, tenant_id: str, limit: int = 100, offset: int = 0,
                        category: Optional[str] = None, author: Optional[str] = None,
                        status: Optional[str] = None, sort_by: str = "created_at",
                        sort_order: str = "desc") -> Tuple[List[BlogPost], int]:
        """List posts with tenant isolation."""
        cache_key = f"posts:{tenant_id}:{limit}:{offset}:{category}:{author}:{status}:{sort_by}:{sort_order}"
        
        # Try cache first
        cached_result = await self.cache_manager.get(cache_key)
        if cached_result:
            return cached_result["posts"], cached_result["total"]
        
        async with self.db_manager.get_session() as session:
            # Build query with tenant isolation
            query = select(BlogPostModel).where(BlogPostModel.tenant_id == tenant_id)
            
            # Apply filters
            if category:
                query = query.where(BlogPostModel.category == category)
            if author:
                query = query.where(BlogPostModel.author_id == author)
            if status:
                query = query.where(BlogPostModel.status == status)
            
            # Get total count
            count_query = select(func.count()).select_from(query.subquery())
            total_result = await session.execute(count_query)
            total = total_result.scalar()
            
            # Apply sorting and pagination
            if sort_by == "created_at":
                query = query.order_by(desc(BlogPostModel.created_at) if sort_order == "desc" else asc(BlogPostModel.created_at))
            elif sort_by == "views":
                query = query.order_by(desc(BlogPostModel.views) if sort_order == "desc" else asc(BlogPostModel.views))
            elif sort_by == "title":
                query = query.order_by(desc(BlogPostModel.title) if sort_order == "desc" else asc(BlogPostModel.title))
            
            query = query.offset(offset).limit(limit)
            
            # Execute query
            result = await session.execute(query)
            posts = result.scalars().all()
            
            # Convert to Pydantic models
            blog_posts = []
            for post in posts:
                post_dict = {
                    "id": post.id,
                    "tenant_id": post.tenant_id,
                    "author_id": post.author_id,
                    "title": post.title,
                    "content": post.content,
                    "excerpt": post.excerpt,
                    "tags": json.loads(post.tags) if post.tags else None,
                    "category": post.category,
                    "status": post.status,
                    "version": post.version,
                    "parent_version_id": post.parent_version_id,
                    "created_at": post.created_at,
                    "updated_at": post.updated_at,
                    "published_at": post.published_at,
                    "scheduled_at": post.scheduled_at,
                    "views": post.views,
                    "likes": post.likes,
                    "shares": post.shares,
                    "comments_count": post.comments_count,
                    "reading_time": post.reading_time,
                    "seo_title": post.seo_title,
                    "seo_description": post.seo_description,
                    "seo_keywords": post.seo_keywords,
                    "featured_image": post.featured_image,
                    "structured_data": json.loads(post.structured_data) if post.structured_data else None
                }
                blog_posts.append(BlogPost.model_validate(post_dict))
            
            # Cache result
            await self.cache_manager.set(cache_key, {
                "posts": blog_posts,
                "total": total
            }, ttl=300)  # 5 minutes
            
            return blog_posts, total
    
    async def create_post(self, tenant_id: str, author_id: int, post_data: BlogPostCreate) -> BlogPost:
        """Create a new post with versioning and audit."""
        async with self.db_manager.get_session() as session:
            # Create post
            post = BlogPostModel(
                tenant_id=tenant_id,
                author_id=author_id,
                title=post_data.title,
                content=post_data.content,
                excerpt=post_data.excerpt,
                tags=json.dumps(post_data.tags) if post_data.tags else None,
                category=post_data.category,
                status=post_data.status,
                scheduled_at=post_data.scheduled_at,
                seo_title=post_data.seo_title,
                seo_description=post_data.seo_description,
                seo_keywords=post_data.seo_keywords,
                featured_image=post_data.featured_image,
                structured_data=json.dumps(post_data.structured_data) if post_data.structured_data else None
            )
            session.add(post)
            await session.commit()
            await session.refresh(post)
            
            # Create initial version
            await self.versioning_service.create_version(post, author_id, "Initial version")
            
            # Log audit
            await self.audit_service.log_action(
                tenant_id=tenant_id,
                user_id=author_id,
                action="create",
                resource_type="post",
                resource_id=post.id,
                new_values=post_data.model_dump()
            )
            
            # Invalidate cache
            await self.cache_manager.invalidate_pattern(f"posts:{tenant_id}:*")
            
            # Convert to Pydantic model
            post_dict = {
                "id": post.id,
                "tenant_id": post.tenant_id,
                "author_id": post.author_id,
                "title": post.title,
                "content": post.content,
                "excerpt": post.excerpt,
                "tags": json.loads(post.tags) if post.tags else None,
                "category": post.category,
                "status": post.status,
                "version": post.version,
                "parent_version_id": post.parent_version_id,
                "created_at": post.created_at,
                "updated_at": post.updated_at,
                "published_at": post.published_at,
                "scheduled_at": post.scheduled_at,
                "views": post.views,
                "likes": post.likes,
                "shares": post.shares,
                "comments_count": post.comments_count,
                "reading_time": post.reading_time,
                "seo_title": post.seo_title,
                "seo_description": post.seo_description,
                "seo_keywords": post.seo_keywords,
                "featured_image": post.featured_image,
                "structured_data": json.loads(post.structured_data) if post.structured_data else None
            }
            
            return BlogPost.model_validate(post_dict)

# ============================================================================
# ENTERPRISE BLOG SYSTEM
# ============================================================================

class EnterpriseBlogSystem:
    """Enterprise blog system with advanced features."""
    
    def __init__(self, config: EnterpriseConfig):
        self.config = config
        
        # Initialize services
        self.db_manager = DatabaseManager(config.database)
        self.cache_manager = CacheManager(config.cache)
        self.search_service = SearchService(config.search) if ELASTICSEARCH_AVAILABLE else None
        self.analytics_service = AnalyticsService(self.db_manager, config.analytics)
        self.ai_service = AIService(config.ai) if ML_AVAILABLE else None
        self.notification_service = NotificationService(config.notifications)
        self.security_service = SecurityService(config.security)
        self.tenant_service = TenantService(self.db_manager, config.tenant)
        self.versioning_service = VersioningService(self.db_manager, config.versioning)
        self.audit_service = AuditService(self.db_manager, config.versioning)
        
        # Initialize main service
        self.blog_service = EnterpriseBlogService(
            self.db_manager, self.cache_manager, self.search_service,
            self.analytics_service, self.ai_service, self.notification_service,
            self.security_service, self.tenant_service, self.versioning_service,
            self.audit_service
        )
        
        # Create FastAPI app
        self.app = FastAPI(
            title="Enterprise Blog System V4",
            description="Advanced enterprise blog system with multi-tenancy, security, and versioning",
            version="4.0.0",
            docs_url="/docs" if config.debug else None,
            redoc_url="/redoc" if config.debug else None
        )
        
        # Setup middleware and routes
        self._setup_middleware()
        self._setup_routes()
        self._setup_exception_handlers()
    
    def _setup_middleware(self):
        """Setup middleware."""
        # CORS
        if self.config.security.enable_cors:
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=self.config.security.allowed_origins,
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
        
        # GZip compression
        self.app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    def _setup_routes(self):
        """Setup API routes."""
        
        # Authentication dependency
        async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())) -> User:
            """Get current user from JWT token."""
            payload = self.security_service.verify_token(credentials.credentials)
            user_id = payload.get("sub")
            if user_id is None:
                raise HTTPException(status_code=401, detail="Invalid token")
            
            # Get user from database
            async with self.db_manager.get_session() as session:
                result = await session.execute(
                    select(UserModel).where(UserModel.id == user_id)
                )
                user = result.scalar_one_or_none()
                if user is None:
                    raise HTTPException(status_code=401, detail="User not found")
                return User.model_validate(user)
        
        # Tenant dependency
        async def get_current_tenant(request: Request) -> str:
            """Get current tenant from request."""
            tenant_id = await self.tenant_service.get_tenant_from_header(request)
            if not await self.tenant_service.validate_tenant(tenant_id):
                raise HTTPException(status_code=400, detail="Invalid tenant")
            return tenant_id
        
        @self.app.post("/auth/login", response_model=Token)
        async def login(user_credentials: UserLogin):
            """User login endpoint."""
            async with self.db_manager.get_session() as session:
                result = await session.execute(
                    select(UserModel).where(UserModel.username == user_credentials.username)
                )
                user = result.scalar_one_or_none()
                
                if not user or not self.security_service.verify_password(user_credentials.password, user.password_hash):
                    raise HTTPException(status_code=401, detail="Invalid credentials")
                
                # Update last login
                user.last_login = time.time()
                await session.commit()
                
                # Create access token
                access_token = self.security_service.create_access_token(
                    data={"sub": str(user.id), "tenant_id": user.tenant_id, "role": user.role}
                )
                
                return Token(
                    access_token=access_token,
                    expires_in=self.config.security.jwt_expiration_hours * 3600,
                    user=User.model_validate(user)
                )
        
        @self.app.post("/auth/register", response_model=User)
        async def register(user_data: UserCreate, tenant_id: str = Depends(get_current_tenant)):
            """User registration endpoint."""
            async with self.db_manager.get_session() as session:
                # Check if username or email already exists
                existing_user = await session.execute(
                    select(UserModel).where(
                        (UserModel.username == user_data.username) |
                        (UserModel.email == user_data.email)
                    )
                )
                if existing_user.scalar_one_or_none():
                    raise HTTPException(status_code=400, detail="Username or email already exists")
                
                # Create new user
                hashed_password = self.security_service.hash_password(user_data.password)
                user = UserModel(
                    tenant_id=tenant_id,
                    username=user_data.username,
                    email=user_data.email,
                    password_hash=hashed_password,
                    role=user_data.role,
                    permissions=json.dumps(user_data.permissions) if user_data.permissions else None
                )
                session.add(user)
                await session.commit()
                await session.refresh(user)
                
                return User.model_validate(user)
        
        @self.app.get("/posts", response_model=Dict[str, Any])
        async def list_posts(
            limit: int = 100,
            offset: int = 0,
            category: Optional[str] = None,
            author: Optional[str] = None,
            status: Optional[str] = None,
            sort_by: str = "created_at",
            sort_order: str = "desc",
            current_user: User = Depends(get_current_user),
            tenant_id: str = Depends(get_current_tenant)
        ):
            """List posts with tenant isolation."""
            posts, total = await self.blog_service.list_posts(
                tenant_id=tenant_id,
                limit=limit,
                offset=offset,
                category=category,
                author=author,
                status=status,
                sort_by=sort_by,
                sort_order=sort_order
            )
            
            return {
                "posts": [post.model_dump() for post in posts],
                "total": total,
                "has_more": offset + limit < total
            }
        
        @self.app.post("/posts", response_model=BlogPost)
        async def create_post(
            post_data: BlogPostCreate,
            current_user: User = Depends(get_current_user),
            tenant_id: str = Depends(get_current_tenant)
        ):
            """Create a new post."""
            return await self.blog_service.create_post(tenant_id, current_user.id, post_data)
        
        @self.app.get("/posts/{post_id}/versions")
        async def get_post_versions(
            post_id: int,
            current_user: User = Depends(get_current_user),
            tenant_id: str = Depends(get_current_tenant)
        ):
            """Get all versions of a post."""
            return await self.versioning_service.get_post_versions(post_id, tenant_id)
        
        @self.app.post("/posts/{post_id}/versions/{version}/restore")
        async def restore_post_version(
            post_id: int,
            version: int,
            current_user: User = Depends(get_current_user),
            tenant_id: str = Depends(get_current_tenant)
        ):
            """Restore a post to a specific version."""
            return await self.versioning_service.restore_version(post_id, version, current_user.id)
    
    def _setup_exception_handlers(self):
        """Setup exception handlers."""
        
        @self.app.exception_handler(ValidationError)
        async def validation_exception_handler(request: Request, exc: ValidationError):
            return JSONResponse(
                status_code=422,
                content={"detail": "Validation error", "errors": exc.errors()}
            )
        
        @self.app.exception_handler(Exception)
        async def generic_exception_handler(request: Request, exc: Exception):
            logging.error(f"Unhandled exception: {exc}")
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal server error"}
            )
    
    async def startup(self):
        """Startup event."""
        # Create database tables
        await self.db_manager.create_tables()
        
        # Initialize Elasticsearch index if available
        if self.search_service and ELASTICSEARCH_AVAILABLE:
            try:
                await self.search_service.es.indices.create(
                    index="blog_posts",
                    body={
                        "mappings": {
                            "properties": {
                                "title": {"type": "text"},
                                "content": {"type": "text"},
                                "excerpt": {"type": "text"},
                                "tags": {"type": "keyword"},
                                "category": {"type": "keyword"},
                                "tenant_id": {"type": "keyword"},
                                "author_id": {"type": "integer"},
                                "status": {"type": "keyword"},
                                "created_at": {"type": "date"}
                            }
                        }
                    },
                    ignore=400  # Index already exists
                )
            except Exception as e:
                logging.warning(f"Could not create Elasticsearch index: {e}")
        
        logging.info("Enterprise Blog System V4 started successfully")
    
    async def shutdown(self):
        """Shutdown event."""
        # Close database connections
        await self.db_manager.engine.dispose()
        
        # Close Redis connections
        if REDIS_AVAILABLE:
            await self.cache_manager.redis.close()
        
        # Close Elasticsearch connections
        if self.search_service and ELASTICSEARCH_AVAILABLE:
            await self.search_service.es.close()
        
        logging.info("Enterprise Blog System V4 shutdown complete")

def create_enterprise_blog_system(config: EnterpriseConfig = None) -> EnterpriseBlogSystem:
    """Create and configure enterprise blog system."""
    if config is None:
        config = EnterpriseConfig()
    
    system = EnterpriseBlogSystem(config)
    
    @system.app.on_event("startup")
    async def startup_event():
        await system.startup()
    
    @system.app.on_event("shutdown")
    async def shutdown_event():
        await system.shutdown()
    
    return system

if __name__ == "__main__":
    import uvicorn
    
    # Create system
    system = create_enterprise_blog_system()
    
    # Run server
    uvicorn.run(
        system.app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    ) 
 
 