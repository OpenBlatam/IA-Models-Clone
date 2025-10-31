# 🗄️ SQLAlchemy 2.0 Implementation - Complete Documentation

## 📋 Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Models](#models)
6. [Database Operations](#database-operations)
7. [Migrations](#migrations)
8. [Performance Optimization](#performance-optimization)
9. [Testing](#testing)
10. [Production Deployment](#production-deployment)
11. [Best Practices](#best-practices)
12. [Troubleshooting](#troubleshooting)

## 🎯 Overview

This SQLAlchemy 2.0 implementation provides a modern, production-ready database layer for the Blatam Academy NLP system. It features:

- **Async/await support** with SQLAlchemy 2.0
- **Type-safe ORM** with Pydantic integration
- **Comprehensive migration system** with Alembic
- **Performance monitoring** and optimization
- **Production-ready features** like connection pooling and caching

## ✨ Features

### Core Features
- ✅ **SQLAlchemy 2.0** with modern async patterns
- ✅ **Type-safe models** with Pydantic integration
- ✅ **Connection pooling** with optimized settings
- ✅ **Comprehensive error handling** with custom exceptions
- ✅ **Performance monitoring** with detailed metrics
- ✅ **Migration system** with Alembic integration
- ✅ **Caching support** with Redis integration
- ✅ **Health monitoring** with status checks

### Advanced Features
- ✅ **Bulk operations** for high-performance data processing
- ✅ **Query optimization** with automatic caching
- ✅ **Transaction management** with automatic rollback
- ✅ **Data validation** with Pydantic models
- ✅ **Multi-database support** (PostgreSQL, MySQL, SQLite)
- ✅ **Testing utilities** with pytest integration

## 🚀 Installation

### 1. Install Dependencies

```bash
pip install -r requirements_sqlalchemy_2.txt
```

### 2. Database Setup

#### PostgreSQL (Recommended)
```bash
# Install PostgreSQL
sudo apt-get install postgresql postgresql-contrib

# Create database
sudo -u postgres createdb nlp_db
sudo -u postgres createuser nlp_user
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE nlp_db TO nlp_user;"
```

#### SQLite (Development)
```bash
# No additional setup required for SQLite
```

### 3. Redis Setup (Optional, for Caching)
```bash
# Install Redis
sudo apt-get install redis-server

# Start Redis
sudo systemctl start redis-server
```

## 🏃‍♂️ Quick Start

### 1. Basic Setup

```python
import asyncio
from sqlalchemy_2_implementation import (
    DatabaseConfig, SQLAlchemy2Manager,
    TextAnalysisCreate, AnalysisType, OptimizationTier
)

async def main():
    # Create database configuration
    config = DatabaseConfig(
        url="postgresql+asyncpg://user:password@localhost/nlp_db",
        pool_size=20,
        enable_caching=True
    )
    
    # Create and initialize database manager
    db_manager = SQLAlchemy2Manager(config)
    await db_manager.initialize()
    
    try:
        # Create text analysis
        analysis_data = TextAnalysisCreate(
            text_content="This is a sample text for sentiment analysis.",
            analysis_type=AnalysisType.SENTIMENT,
            optimization_tier=OptimizationTier.STANDARD
        )
        
        analysis = await db_manager.create_text_analysis(analysis_data)
        print(f"Created analysis: {analysis.id}")
        
    finally:
        await db_manager.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
```

### 2. FastAPI Integration

```python
from fastapi import FastAPI, Depends
from sqlalchemy_2_implementation import (
    DatabaseConfig, SQLAlchemy2Manager,
    TextAnalysisCreate, TextAnalysisResponse
)

app = FastAPI()

# Database dependency
async def get_db():
    config = DatabaseConfig(
        url="postgresql+asyncpg://user:password@localhost/nlp_db"
    )
    db_manager = SQLAlchemy2Manager(config)
    await db_manager.initialize()
    try:
        yield db_manager
    finally:
        await db_manager.cleanup()

@app.post("/analyses/", response_model=TextAnalysisResponse)
async def create_analysis(
    data: TextAnalysisCreate,
    db: SQLAlchemy2Manager = Depends(get_db)
):
    analysis = await db.create_text_analysis(data)
    return analysis

@app.get("/analyses/{analysis_id}", response_model=TextAnalysisResponse)
async def get_analysis(
    analysis_id: int,
    db: SQLAlchemy2Manager = Depends(get_db)
):
    analysis = await db.get_text_analysis(analysis_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return analysis
```

## 📊 Models

### Core Models

#### TextAnalysis
```python
class TextAnalysis(Base):
    """Text analysis results model."""
    __tablename__ = "text_analyses"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    text_content: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    text_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    analysis_type: Mapped[AnalysisType] = mapped_column(String(50), nullable=False, index=True)
    status: Mapped[AnalysisStatus] = mapped_column(String(20), nullable=False, default=AnalysisStatus.PENDING)
    
    # Results
    sentiment_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    quality_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    emotion_scores: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    
    # Performance metrics
    processing_time_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    model_used: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
```

#### BatchAnalysis
```python
class BatchAnalysis(Base):
    """Batch analysis management model."""
    __tablename__ = "batch_analyses"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    batch_name: Mapped[str] = mapped_column(String(200), nullable=False)
    batch_size: Mapped[int] = mapped_column(Integer, nullable=False)
    analysis_type: Mapped[AnalysisType] = mapped_column(String(50), nullable=False)
    
    # Status and progress
    status: Mapped[AnalysisStatus] = mapped_column(String(20), nullable=False, default=AnalysisStatus.PENDING)
    completed_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    
    # Relationships
    text_analyses: Mapped[List["TextAnalysis"]] = relationship(
        "TextAnalysis", back_populates="batch_analysis", lazy="selectin"
    )
```

### Pydantic Models

#### TextAnalysisCreate
```python
class TextAnalysisCreate(BaseModel):
    """Pydantic model for creating text analysis."""
    text_content: str = Field(..., min_length=1, max_length=10000)
    analysis_type: AnalysisType
    optimization_tier: OptimizationTier = OptimizationTier.STANDARD
    model_config: Optional[Dict[str, Any]] = None
```

#### TextAnalysisUpdate
```python
class TextAnalysisUpdate(BaseModel):
    """Pydantic model for updating text analysis."""
    status: Optional[AnalysisStatus] = None
    sentiment_score: Optional[float] = Field(None, ge=-1.0, le=1.0)
    quality_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    processing_time_ms: Optional[float] = Field(None, ge=0.0)
    model_used: Optional[str] = None
```

## 🔧 Database Operations

### CRUD Operations

#### Create
```python
# Create single analysis
analysis_data = TextAnalysisCreate(
    text_content="Sample text",
    analysis_type=AnalysisType.SENTIMENT
)
analysis = await db_manager.create_text_analysis(analysis_data)

# Create batch analysis
batch_data = BatchAnalysisCreate(
    batch_name="My Batch",
    analysis_type=AnalysisType.SENTIMENT
)
batch = await db_manager.create_batch_analysis(batch_data)
```

#### Read
```python
# Get by ID
analysis = await db_manager.get_text_analysis(analysis_id)

# Get by hash
analysis = await db_manager.get_text_analysis_by_hash(text_hash, AnalysisType.SENTIMENT)

# List with filtering
analyses = await db_manager.list_text_analyses(
    analysis_type=AnalysisType.SENTIMENT,
    status=AnalysisStatus.COMPLETED,
    limit=100,
    offset=0,
    order_by="created_at",
    order_desc=True
)
```

#### Update
```python
# Update analysis
update_data = TextAnalysisUpdate(
    status=AnalysisStatus.COMPLETED,
    sentiment_score=0.8,
    processing_time_ms=150.5
)
updated = await db_manager.update_text_analysis(analysis_id, update_data)

# Update batch progress
updated_batch = await db_manager.update_batch_progress(
    batch_id, completed_count=10, error_count=1
)
```

#### Delete
```python
# Delete analysis
success = await db_manager.delete_text_analysis(analysis_id)
```

### Advanced Operations

#### Bulk Operations
```python
# Bulk insert (implemented in the manager)
analyses_data = [
    TextAnalysisCreate(text_content=f"Text {i}", analysis_type=AnalysisType.SENTIMENT)
    for i in range(1000)
]

for data in analyses_data:
    await db_manager.create_text_analysis(data)
```

#### Transaction Management
```python
async with db_manager.get_session() as session:
    try:
        # Multiple operations in transaction
        analysis1 = TextAnalysis(...)
        analysis2 = TextAnalysis(...)
        
        session.add(analysis1)
        session.add(analysis2)
        
        await session.commit()
        
    except Exception:
        await session.rollback()
        raise
```

## 🚀 Migrations

### Setup Migrations

```python
from sqlalchemy_migrations import MigrationManager, DatabaseConfig

async def setup_migrations():
    config = DatabaseConfig(
        url="postgresql+asyncpg://user:password@localhost/nlp_db"
    )
    
    manager = MigrationManager(config, migrations_dir="migrations")
    await manager.initialize()
    
    # Create initial migration
    revision = await manager.create_migration("Initial migration")
    
    # Run migrations
    success = await manager.upgrade()
    
    await manager.cleanup()
```

### Migration Commands

```bash
# Create new migration
alembic revision --autogenerate -m "Add new field"

# Run migrations
alembic upgrade head

# Downgrade
alembic downgrade -1

# Check status
alembic current
alembic history
```

### Data Migrations

```python
from sqlalchemy_migrations import DataMigration

async def migrate_data(conn, **kwargs):
    """Custom data migration function."""
    # Update existing data
    await conn.execute(text("""
        UPDATE text_analyses 
        SET optimization_tier = 'standard' 
        WHERE optimization_tier IS NULL
    """))

# Run data migration
data_migration = DataMigration(engine)
success = await data_migration.migrate_data(migrate_data)
```

## ⚡ Performance Optimization

### Connection Pooling

```python
config = DatabaseConfig(
    url="postgresql+asyncpg://user:password@localhost/nlp_db",
    pool_size=20,           # Number of connections in pool
    max_overflow=30,        # Additional connections when pool is full
    pool_timeout=30,        # Timeout for getting connection
    pool_recycle=3600,      # Recycle connections after 1 hour
    pool_pre_ping=True      # Verify connections before use
)
```

### Query Optimization

```python
# Use indexes for frequently queried columns
__table_args__ = (
    Index('idx_text_hash_analysis_type', 'text_hash', 'analysis_type'),
    Index('idx_status_created_at', 'status', 'created_at'),
    Index('idx_sentiment_score', 'sentiment_score'),
)

# Use selectinload for relationships
analyses = await session.execute(
    select(TextAnalysis).options(selectinload(TextAnalysis.batch_analysis))
)
```

### Caching

```python
# Enable Redis caching
config = DatabaseConfig(
    url="postgresql+asyncpg://user:password@localhost/nlp_db",
    enable_caching=True,
    cache_ttl=3600  # Cache for 1 hour
)

# Cache is automatically used for read operations
analysis = await db_manager.get_text_analysis(analysis_id)  # Uses cache if available
```

### Performance Monitoring

```python
# Get performance metrics
metrics = await db_manager.get_performance_metrics()
print(f"Total queries: {metrics['database']['total_queries']}")
print(f"Success rate: {metrics['database']['success_rate']:.2%}")
print(f"Average query time: {metrics['database']['avg_query_time']:.2f}ms")

# Health check
health = await db_manager.health_check()
print(f"Database healthy: {health.is_healthy}")
```

## 🧪 Testing

### Test Setup

```python
import pytest
from sqlalchemy_2_implementation import DatabaseConfig, SQLAlchemy2Manager

@pytest.fixture
async def test_db_config():
    return DatabaseConfig(
        url="sqlite+aiosqlite:///:memory:",
        pool_size=5,
        enable_caching=False
    )

@pytest.fixture
async def db_manager(test_db_config):
    manager = SQLAlchemy2Manager(test_db_config)
    await manager.initialize()
    yield manager
    await manager.cleanup()
```

### Test Examples

```python
@pytest.mark.asyncio
async def test_create_text_analysis(db_manager):
    """Test creating text analysis."""
    data = TextAnalysisCreate(
        text_content="Test content",
        analysis_type=AnalysisType.SENTIMENT
    )
    
    analysis = await db_manager.create_text_analysis(data)
    
    assert analysis.id is not None
    assert analysis.text_content == data.text_content
    assert analysis.analysis_type == data.analysis_type

@pytest.mark.asyncio
async def test_concurrent_operations(db_manager):
    """Test concurrent operations."""
    async def create_analysis(i):
        data = TextAnalysisCreate(
            text_content=f"Content {i}",
            analysis_type=AnalysisType.SENTIMENT
        )
        return await db_manager.create_text_analysis(data)
    
    tasks = [create_analysis(i) for i in range(10)]
    results = await asyncio.gather(*tasks)
    
    assert len(results) == 10
    assert all(r.id is not None for r in results)
```

### Run Tests

```bash
# Run all tests
pytest test_sqlalchemy_2.py -v

# Run specific test class
pytest test_sqlalchemy_2.py::TestDatabaseManager -v

# Run with coverage
pytest test_sqlalchemy_2.py --cov=sqlalchemy_2_implementation --cov-report=html
```

## 🚀 Production Deployment

### Environment Configuration

```python
import os
from sqlalchemy_2_implementation import DatabaseConfig

def get_database_config():
    """Get database configuration from environment."""
    return DatabaseConfig(
        url=os.getenv("DATABASE_URL", "postgresql+asyncpg://localhost/nlp_db"),
        pool_size=int(os.getenv("DB_POOL_SIZE", "20")),
        max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "30")),
        enable_caching=os.getenv("ENABLE_CACHING", "true").lower() == "true",
        cache_ttl=int(os.getenv("CACHE_TTL", "3600"))
    )
```

### Docker Configuration

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements_sqlalchemy_2.txt .
RUN pip install -r requirements_sqlalchemy_2.txt

COPY . .

CMD ["python", "main.py"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  app:
    build: .
    environment:
      - DATABASE_URL=postgresql+asyncpg://postgres:password@db/nlp_db
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis

  db:
    image: postgres:15
    environment:
      - POSTGRES_DB=nlp_db
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

volumes:
  postgres_data:
```

### Health Checks

```python
from fastapi import FastAPI
from sqlalchemy_2_implementation import SQLAlchemy2Manager

app = FastAPI()

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    config = get_database_config()
    db_manager = SQLAlchemy2Manager(config)
    await db_manager.initialize()
    
    try:
        health = await db_manager.health_check()
        metrics = await db_manager.get_performance_metrics()
        
        return {
            "status": "healthy" if health.is_healthy else "unhealthy",
            "database": health.is_healthy,
            "metrics": metrics
        }
    finally:
        await db_manager.cleanup()
```

## 📚 Best Practices

### 1. Connection Management

```python
# ✅ Good: Use context manager
async with db_manager.get_session() as session:
    # Database operations
    pass

# ❌ Bad: Manual session management
session = db_manager.session_factory()
try:
    # Operations
    pass
finally:
    await session.close()
```

### 2. Error Handling

```python
# ✅ Good: Proper error handling
try:
    analysis = await db_manager.create_text_analysis(data)
except ValidationError as e:
    logger.error(f"Validation error: {e}")
    raise HTTPException(status_code=400, detail=str(e))
except DatabaseError as e:
    logger.error(f"Database error: {e}")
    raise HTTPException(status_code=500, detail="Database error")
```

### 3. Performance Optimization

```python
# ✅ Good: Use bulk operations for large datasets
analyses_data = [TextAnalysisCreate(...) for _ in range(1000)]
for data in analyses_data:
    await db_manager.create_text_analysis(data)

# ✅ Good: Use appropriate indexes
__table_args__ = (
    Index('idx_frequently_queried', 'column1', 'column2'),
)

# ✅ Good: Use caching for read-heavy operations
analysis = await db_manager.get_text_analysis(analysis_id)  # Uses cache
```

### 4. Migration Best Practices

```python
# ✅ Good: Always backup before migrations
async def safe_migration():
    data_migration = DataMigration(engine)
    backup_table = await data_migration.backup_table("text_analyses")
    
    try:
        # Run migration
        await migration_manager.upgrade()
    except Exception:
        # Restore from backup
        await data_migration.restore_table("text_analyses", backup_table)
        raise
```

### 5. Testing Best Practices

```python
# ✅ Good: Use in-memory database for tests
@pytest.fixture
async def test_db_config():
    return DatabaseConfig(
        url="sqlite+aiosqlite:///:memory:",
        enable_caching=False
    )

# ✅ Good: Clean up after tests
@pytest.fixture
async def db_manager(test_db_config):
    manager = SQLAlchemy2Manager(test_db_config)
    await manager.initialize()
    yield manager
    await manager.cleanup()
```

## 🔧 Troubleshooting

### Common Issues

#### 1. Connection Pool Exhausted
```
Error: QueuePool limit of size 20 overflow 30 reached
```

**Solution:**
```python
# Increase pool size
config = DatabaseConfig(
    pool_size=50,
    max_overflow=100
)
```

#### 2. Migration Conflicts
```
Error: Can't locate revision identified by 'abc123'
```

**Solution:**
```python
# Check migration status
status = await migration_manager.check_migrations()
print(f"Current: {status['current_revision']}")
print(f"Head: {status['head_revision']}")

# Reset migrations if needed
await migration_manager.downgrade("base")
await migration_manager.upgrade("head")
```

#### 3. Performance Issues
```
Slow query execution
```

**Solution:**
```python
# Check performance metrics
metrics = await db_manager.get_performance_metrics()
print(f"Avg query time: {metrics['database']['avg_query_time']}ms")

# Add indexes
__table_args__ = (
    Index('idx_slow_column', 'slow_column'),
)

# Use query optimization
analyses = await db_manager.list_text_analyses(
    limit=100,  # Limit results
    order_by="created_at"  # Use indexed column
)
```

#### 4. Memory Issues
```
High memory usage
```

**Solution:**
```python
# Monitor memory usage
memory_usage = db_manager.performance_monitor.get_memory_usage()
print(f"Memory usage: {memory_usage:.2f} MB")

# Reduce pool size
config = DatabaseConfig(
    pool_size=10,  # Reduce from 20
    max_overflow=20  # Reduce from 30
)

# Clear cache periodically
db_manager.query_cache.clear()
```

### Debug Mode

```python
# Enable debug logging
config = DatabaseConfig(
    url="postgresql+asyncpg://user:password@localhost/nlp_db",
    echo=True,  # Log SQL queries
    echo_pool=True  # Log connection pool events
)
```

### Monitoring

```python
# Set up monitoring
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Monitor database operations
logger = logging.getLogger('sqlalchemy.engine')
logger.setLevel(logging.DEBUG)
```

## 📖 Additional Resources

- [SQLAlchemy 2.0 Documentation](https://docs.sqlalchemy.org/en/20/)
- [Alembic Documentation](https://alembic.sqlalchemy.org/)
- [AsyncPG Documentation](https://asyncpg.readthedocs.io/)
- [Pydantic Documentation](https://docs.pydantic.dev/)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run the test suite
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details. 