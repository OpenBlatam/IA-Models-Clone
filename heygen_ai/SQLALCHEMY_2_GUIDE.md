# SQLAlchemy 2.0 Implementation Guide for HeyGen AI API

A comprehensive guide for implementing and using SQLAlchemy 2.0 with modern features, type annotations, and performance optimization in the HeyGen AI FastAPI backend.

## 🚀 Overview

SQLAlchemy 2.0 provides:
- **Modern Type Annotations**: Full type safety with Python type hints
- **Enhanced Performance**: Optimized query execution and connection pooling
- **Improved API**: Cleaner, more intuitive API design
- **Better Error Handling**: More descriptive error messages
- **Async Support**: Native async/await support throughout
- **Migration Tools**: Easy migration from SQLAlchemy 1.x

## 📋 Table of Contents

1. [SQLAlchemy 2.0 Features](#sqlalchemy-20-features)
2. [Model Definitions](#model-definitions)
3. [Session Management](#session-management)
4. [Query Patterns](#query-patterns)
5. [Relationships](#relationships)
6. [Performance Optimization](#performance-optimization)
7. [Migration from 1.x](#migration-from-1x)
8. [Best Practices](#best-practices)
9. [Advanced Features](#advanced-features)
10. [Troubleshooting](#troubleshooting)

## 🔧 SQLAlchemy 2.0 Features

### Key Improvements

```python
# SQLAlchemy 1.x (Old Style)
from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    name = Column(String(50))

# SQLAlchemy 2.0 (New Style)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(50))
```

### Type Annotations

```python
from typing import Optional, List
from sqlalchemy.orm import Mapped, mapped_column, relationship

class User(Base):
    __tablename__ = "users"
    
    # Required fields
    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(50), unique=True)
    
    # Optional fields
    email: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    avatar_url: Mapped[Optional[str]] = mapped_column(String(500))
    
    # Relationships
    videos: Mapped[List["Video"]] = relationship(back_populates="user")
```

## 🏗️ Model Definitions

### Enhanced Base Class

```python
from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.orm import DeclarativeBase
from datetime import datetime

class Base(AsyncAttrs, DeclarativeBase):
    """Enhanced base class with common functionality."""
    
    __abstract__ = True
    
    # Automatic timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
        index=True
    )
    
    def __repr__(self) -> str:
        """Enhanced string representation."""
        attrs = []
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                if isinstance(value, str) and len(value) > 50:
                    value = value[:50] + "..."
                attrs.append(f"{key}={value}")
        return f"{self.__class__.__name__}({', '.join(attrs)})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        result = {}
        for column in self.__table__.columns:
            value = getattr(self, column.name)
            if isinstance(value, datetime):
                value = value.isoformat()
            result[column.name] = value
        return result
```

### Advanced Model with Mixins

```python
from sqlalchemy.orm import declarative_mixin, declared_attr

@declarative_mixin
class TimestampMixin:
    """Mixin for automatic timestamp management."""
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
        index=True
    )

@declarative_mixin
class SoftDeleteMixin:
    """Mixin for soft delete functionality."""
    
    deleted_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True
    )
    is_deleted: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        index=True
    )

class User(Base, TimestampMixin, SoftDeleteMixin):
    __tablename__ = "users"
    
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(50), unique=True, index=True)
    email: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    
    # Relationships
    videos: Mapped[List["Video"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="selectin"
    )
```

### Enum Support

```python
from enum import Enum
from sqlalchemy import Enum as SQLEnum

class VideoStatus(str, Enum):
    """Video processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class Video(Base):
    __tablename__ = "videos"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    status: Mapped[VideoStatus] = mapped_column(
        SQLEnum(VideoStatus),
        default=VideoStatus.PENDING,
        nullable=False,
        index=True
    )
```

## 🔄 Session Management

### Async Session Factory

```python
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

# Create async engine
engine = create_async_engine(
    "postgresql+asyncpg://user:pass@localhost/db",
    echo=True,  # Enable SQL logging
    pool_size=20,
    max_overflow=30,
    pool_pre_ping=True,
    pool_recycle=3600
)

# Create async session factory
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False
)

# Context manager for sessions
@asynccontextmanager
async def get_session() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
```

### Scoped Sessions

```python
from sqlalchemy.ext.asyncio import async_scoped_session
import asyncio

# Create scoped session factory
AsyncScopedSession = async_scoped_session(
    AsyncSessionLocal,
    scopefunc=asyncio.current_task
)

# Use scoped session
async def some_function():
    session = AsyncScopedSession()
    try:
        # Use session
        result = await session.execute(select(User))
        return result.scalars().all()
    finally:
        await session.close()
```

## 🔍 Query Patterns

### Modern Select Statements

```python
from sqlalchemy import select, update, delete

# Basic select
stmt = select(User).where(User.is_active == True)
result = await session.execute(stmt)
users = result.scalars().all()

# Select with joins
stmt = (
    select(User, Video)
    .join(Video, User.id == Video.user_id)
    .where(Video.status == "completed")
)
result = await session.execute(stmt)
for user, video in result:
    print(f"{user.username}: {video.video_id}")

# Select with options
stmt = (
    select(User)
    .options(selectinload(User.videos))
    .where(User.id == user_id)
)
result = await session.execute(stmt)
user = result.scalar_one_or_none()
# user.videos is now loaded
```

### Advanced Queries

```python
# Aggregation queries
stmt = (
    select(
        User.username,
        func.count(Video.id).label('video_count'),
        func.avg(Video.processing_time).label('avg_time')
    )
    .join(Video, User.id == Video.user_id)
    .group_by(User.username)
    .having(func.count(Video.id) > 5)
)
result = await session.execute(stmt)

# Subqueries
subquery = (
    select(Video.user_id)
    .where(Video.status == "completed")
    .group_by(Video.user_id)
    .having(func.count(Video.id) > 10)
    .subquery()
)

stmt = select(User).where(User.id.in_(select(subquery.c.user_id)))
result = await session.execute(stmt)

# Window functions
stmt = (
    select(
        User.username,
        Video.video_id,
        func.row_number().over(
            partition_by=Video.user_id,
            order_by=Video.created_at.desc()
        ).label('row_num')
    )
    .join(Video, User.id == Video.user_id)
)
```

### Bulk Operations

```python
# Bulk insert
users_data = [
    {"username": "user1", "email": "user1@example.com"},
    {"username": "user2", "email": "user2@example.com"},
]
users = [User(**data) for data in users_data]
session.add_all(users)
await session.commit()

# Bulk update
stmt = (
    update(User)
    .where(User.is_active == False)
    .values(last_login_at=datetime.now())
)
result = await session.execute(stmt)
await session.commit()

# Bulk delete
stmt = delete(Video).where(Video.created_at < cutoff_date)
result = await session.execute(stmt)
await session.commit()
```

## 🔗 Relationships

### Relationship Loading Strategies

```python
class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    
    # Lazy loading (default)
    videos: Mapped[List["Video"]] = relationship(back_populates="user")
    
    # Eager loading
    videos_eager: Mapped[List["Video"]] = relationship(
        back_populates="user",
        lazy="selectin"  # Load with separate SELECT
    )
    
    # Joined loading
    videos_joined: Mapped[List["Video"]] = relationship(
        back_populates="user",
        lazy="joined"  # Load with JOIN
    )

# Manual relationship loading
stmt = select(User).options(
    selectinload(User.videos),  # Load videos with separate query
    joinedload(User.profile)    # Load profile with JOIN
)
```

### Relationship Configuration

```python
class Video(Base):
    __tablename__ = "videos"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    
    # Many-to-one relationship
    user: Mapped["User"] = relationship(
        back_populates="videos",
        lazy="selectin"
    )
    
    # One-to-many relationship with cascade
    model_usage: Mapped[List["ModelUsage"]] = relationship(
        back_populates="video",
        cascade="all, delete-orphan",
        lazy="selectin"
    )

class ModelUsage(Base):
    __tablename__ = "model_usage"
    id: Mapped[int] = mapped_column(primary_key=True)
    video_id: Mapped[int] = mapped_column(ForeignKey("videos.id"))
    
    # Many-to-one relationship
    video: Mapped["Video"] = relationship(back_populates="model_usage")
```

## ⚡ Performance Optimization

### Connection Pooling

```python
from sqlalchemy.pool import QueuePool

engine = create_async_engine(
    "postgresql+asyncpg://user:pass@localhost/db",
    poolclass=QueuePool,
    pool_size=20,           # Number of connections to maintain
    max_overflow=30,        # Additional connections when pool is full
    pool_pre_ping=True,     # Validate connections before use
    pool_recycle=3600,      # Recycle connections every hour
    pool_timeout=30,        # Wait time for available connection
    pool_reset_on_return="commit"  # Reset connection state
)
```

### Query Optimization

```python
# Use selectinload for related data
stmt = (
    select(User)
    .options(selectinload(User.videos))
    .where(User.is_active == True)
)

# Use load_only for specific columns
stmt = (
    select(User)
    .options(load_only(User.id, User.username))
    .where(User.is_active == True)
)

# Use contains_eager for manual joins
stmt = (
    select(User)
    .join(Video, User.id == Video.user_id)
    .options(contains_eager(User.videos))
    .where(Video.status == "completed")
)
```

### Indexing Strategy

```python
from sqlalchemy import Index, UniqueConstraint

class User(Base):
    __tablename__ = "users"
    __table_args__ = (
        # Composite indexes
        Index('idx_users_username_active', 'username', 'is_active'),
        Index('idx_users_email_active', 'email', 'is_active'),
        
        # Unique constraints
        UniqueConstraint('username', name='uq_users_username'),
        UniqueConstraint('email', name='uq_users_email'),
        
        # Check constraints
        CheckConstraint("username ~ '^[a-zA-Z0-9_]{3,50}$'", name='ck_users_username_format'),
    )
    
    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(50), unique=True, index=True)
    email: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
```

## 🔄 Migration from SQLAlchemy 1.x

### Key Changes

```python
# SQLAlchemy 1.x
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    name = Column(String(50))

# SQLAlchemy 2.0
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(50))
```

### Query Migration

```python
# SQLAlchemy 1.x
users = session.query(User).filter(User.is_active == True).all()

# SQLAlchemy 2.0
stmt = select(User).where(User.is_active == True)
result = await session.execute(stmt)
users = result.scalars().all()
```

### Session Migration

```python
# SQLAlchemy 1.x
Session = sessionmaker(bind=engine)
session = Session()

# SQLAlchemy 2.0
AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession)
async with AsyncSessionLocal() as session:
    # Use session
    pass
```

## 🛠️ Best Practices

### Model Design

```python
# Good: Use type annotations
class User(Base):
    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(50), unique=True)
    email: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

# Good: Use mixins for common functionality
class User(Base, TimestampMixin, SoftDeleteMixin):
    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(50), unique=True)

# Good: Use enums for status fields
class VideoStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"

class Video(Base):
    status: Mapped[VideoStatus] = mapped_column(SQLEnum(VideoStatus))
```

### Session Management

```python
# Good: Use context managers
@asynccontextmanager
async def get_session() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

# Good: Handle transactions properly
async def create_user_with_videos(user_data: dict, videos_data: list):
    async with get_session() as session:
        try:
            user = User(**user_data)
            session.add(user)
            await session.flush()  # Get user ID
            
            videos = [Video(user_id=user.id, **video_data) for video_data in videos_data]
            session.add_all(videos)
            
            await session.commit()
            return user
        except Exception:
            await session.rollback()
            raise
```

### Query Optimization

```python
# Good: Use appropriate loading strategies
# For single user with videos
stmt = select(User).options(selectinload(User.videos)).where(User.id == user_id)

# For multiple users without videos
stmt = select(User).where(User.is_active == True)

# Good: Use pagination
stmt = select(Video).order_by(Video.created_at.desc()).limit(10).offset(20)

# Good: Use bulk operations for large datasets
async def bulk_create_users(users_data: List[dict]):
    async with get_session() as session:
        users = [User(**data) for data in users_data]
        session.add_all(users)
        await session.commit()
```

## 🔧 Advanced Features

### Event Listeners

```python
from sqlalchemy import event

@event.listens_for(User, 'before_insert')
def set_created_at(mapper, connection, target):
    target.created_at = datetime.now()

@event.listens_for(User, 'before_update')
def set_updated_at(mapper, connection, target):
    target.updated_at = datetime.now()

# Engine events
@event.listens_for(engine, "before_cursor_execute")
def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    conn.info.setdefault('query_start_time', []).append(datetime.now())

@event.listens_for(engine, "after_cursor_execute")
def after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    if not conn.info.get('query_start_time'):
        return
    
    start_time = conn.info['query_start_time'].pop()
    execution_time = (datetime.now() - start_time).total_seconds()
    
    if execution_time > 1.0:
        logger.warning(f"Slow query: {execution_time:.3f}s - {statement[:100]}...")
```

### Custom Types

```python
from sqlalchemy.types import TypeDecorator, String
import json

class JSONType(TypeDecorator):
    impl = String
    
    def process_bind_param(self, value, dialect):
        if value is not None:
            return json.dumps(value)
        return None
    
    def process_result_value(self, value, dialect):
        if value is not None:
            return json.loads(value)
        return None

class User(Base):
    preferences: Mapped[Dict[str, Any]] = mapped_column(JSONType)
```

### Hybrid Properties

```python
from sqlalchemy.ext.hybrid import hybrid_property

class Video(Base):
    __tablename__ = "videos"
    id: Mapped[int] = mapped_column(primary_key=True)
    processing_time: Mapped[Optional[float]] = mapped_column(Float)
    file_size: Mapped[Optional[int]] = mapped_column(BigInteger)
    
    @hybrid_property
    def is_large_file(self) -> bool:
        return self.file_size and self.file_size > 100 * 1024 * 1024  # 100MB
    
    @is_large_file.expression
    def is_large_file(cls):
        return cls.file_size > 100 * 1024 * 1024
```

## 🔍 Troubleshooting

### Common Issues

#### 1. Type Annotation Errors

```python
# Problem: Type annotation not working
class User(Base):
    id = mapped_column(primary_key=True)  # Missing type annotation

# Solution: Add type annotation
class User(Base):
    id: Mapped[int] = mapped_column(primary_key=True)
```

#### 2. Relationship Loading Issues

```python
# Problem: N+1 query problem
users = await session.execute(select(User))
for user in users.scalars():
    print(len(user.videos))  # Triggers additional queries

# Solution: Use selectinload
stmt = select(User).options(selectinload(User.videos))
users = await session.execute(stmt)
for user in users.scalars():
    print(len(user.videos))  # Already loaded
```

#### 3. Session Management Issues

```python
# Problem: Session not properly closed
async def get_user(user_id: int):
    session = AsyncSessionLocal()
    user = await session.get(User, user_id)
    return user  # Session not closed

# Solution: Use context manager
async def get_user(user_id: int):
    async with AsyncSessionLocal() as session:
        user = await session.get(User, user_id)
        return user
```

### Performance Monitoring

```python
import time
from sqlalchemy import event

# Monitor query performance
@event.listens_for(engine, "before_cursor_execute")
def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    conn.info['query_start_time'] = time.time()

@event.listens_for(engine, "after_cursor_execute")
def after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    execution_time = time.time() - conn.info['query_start_time']
    
    if execution_time > 0.1:  # Log queries taking more than 100ms
        logger.info(f"Query took {execution_time:.3f}s: {statement[:100]}...")

# Monitor connection pool
async def get_pool_stats():
    pool = engine.pool
    return {
        "size": pool.size(),
        "checked_in": pool.checkedin(),
        "checked_out": pool.checkedout(),
        "overflow": pool.overflow(),
        "invalid": pool.invalid()
    }
```

## 📚 Additional Resources

- [SQLAlchemy 2.0 Documentation](https://docs.sqlalchemy.org/en/20/)
- [SQLAlchemy 2.0 Migration Guide](https://docs.sqlalchemy.org/en/20/changelog/migration_20.html)
- [SQLAlchemy 2.0 Type Annotations](https://docs.sqlalchemy.org/en/20/orm/declarative_tables.html)
- [SQLAlchemy 2.0 Async Support](https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html)

## 🚀 Next Steps

1. **Upgrade your models** to use SQLAlchemy 2.0 syntax
2. **Implement type annotations** for better type safety
3. **Optimize your queries** with appropriate loading strategies
4. **Set up connection pooling** for better performance
5. **Add monitoring** to track query performance
6. **Use bulk operations** for large datasets
7. **Implement proper session management** with context managers

This SQLAlchemy 2.0 implementation provides a modern, type-safe, and performant database layer for your HeyGen AI API with all the latest features and best practices. 