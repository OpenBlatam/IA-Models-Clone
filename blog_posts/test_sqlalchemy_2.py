from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import pytest
import asyncio
import tempfile
import os
from typing import AsyncGenerator, Dict, Any
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from .sqlalchemy_2_implementation import (
from .sqlalchemy_migrations import MigrationManager, DataMigration
        import time
from typing import Any, List, Dict, Optional
import logging
"""
🧪 SQLAlchemy 2.0 Test Suite
============================

Comprehensive test suite for SQLAlchemy 2.0 implementation with:
- Async testing with pytest-asyncio
- Database fixtures and setup
- Performance testing
- Migration testing
- Error handling testing
- Integration testing
"""



    Base, DatabaseConfig, SQLAlchemy2Manager,
    TextAnalysisCreate, TextAnalysisUpdate, BatchAnalysisCreate,
    AnalysisType, AnalysisStatus, OptimizationTier,
    TextAnalysis, BatchAnalysis, ModelPerformance, CacheEntry
)


# ============================================================================
# Test Configuration
# ============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def test_db_config() -> DatabaseConfig:
    """Create test database configuration."""
    # Use SQLite for testing
    return DatabaseConfig(
        url="sqlite+aiosqlite:///:memory:",
        pool_size=5,
        max_overflow=10,
        enable_caching=False,  # Disable Redis for tests
        echo=False
    )


@pytest.fixture
async def db_manager(test_db_config: DatabaseConfig) -> AsyncGenerator[SQLAlchemy2Manager, None]:
    """Create database manager for testing."""
    manager = SQLAlchemy2Manager(test_db_config)
    await manager.initialize()
    
    yield manager
    
    await manager.cleanup()


@pytest.fixture
async def db_session(db_manager: SQLAlchemy2Manager) -> AsyncGenerator[AsyncSession, None]:
    """Create database session for testing."""
    async with db_manager.get_session() as session:
        yield session


@pytest.fixture
async def migration_manager(test_db_config: DatabaseConfig) -> AsyncGenerator[MigrationManager, None]:
    """Create migration manager for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        manager = MigrationManager(test_db_config, migrations_dir=temp_dir)
        await manager.initialize()
        
        yield manager
        
        await manager.cleanup()


# ============================================================================
# Model Tests
# ============================================================================

class TestModels:
    """Test SQLAlchemy models."""
    
    def test_text_analysis_model(self) -> Any:
        """Test TextAnalysis model creation."""
        analysis = TextAnalysis(
            text_content="Test content",
            text_hash="test_hash",
            content_length=12,
            analysis_type=AnalysisType.SENTIMENT,
            optimization_tier=OptimizationTier.STANDARD
        )
        
        assert analysis.text_content == "Test content"
        assert analysis.analysis_type == AnalysisType.SENTIMENT
        assert analysis.status == AnalysisStatus.PENDING
        assert analysis.optimization_tier == OptimizationTier.STANDARD
    
    def test_batch_analysis_model(self) -> Any:
        """Test BatchAnalysis model creation."""
        batch = BatchAnalysis(
            batch_name="Test Batch",
            batch_size=10,
            analysis_type=AnalysisType.SENTIMENT,
            optimization_tier=OptimizationTier.STANDARD
        )
        
        assert batch.batch_name == "Test Batch"
        assert batch.batch_size == 10
        assert batch.completed_count == 0
        assert batch.error_count == 0
    
    def test_model_to_dict(self) -> Any:
        """Test model to_dict method."""
        analysis = TextAnalysis(
            text_content="Test content",
            text_hash="test_hash",
            content_length=12,
            analysis_type=AnalysisType.SENTIMENT
        )
        
        data = analysis.to_dict()
        assert "text_content" in data
        assert "analysis_type" in data
        assert data["text_content"] == "Test content"
    
    def test_model_to_pydantic(self) -> Any:
        """Test model to_pydantic method."""
        analysis = TextAnalysis(
            text_content="Test content",
            text_hash="test_hash",
            content_length=12,
            analysis_type=AnalysisType.SENTIMENT
        )
        
        data = analysis.to_pydantic()
        assert "text_content" in data
        assert "analysis_type" in data
        assert isinstance(data["created_at"], str)


# ============================================================================
# Pydantic Model Tests
# ============================================================================

class TestPydanticModels:
    """Test Pydantic models."""
    
    def test_text_analysis_create(self) -> Any:
        """Test TextAnalysisCreate model."""
        data = TextAnalysisCreate(
            text_content="Test content",
            analysis_type=AnalysisType.SENTIMENT,
            optimization_tier=OptimizationTier.STANDARD
        )
        
        assert data.text_content == "Test content"
        assert data.analysis_type == AnalysisType.SENTIMENT
        assert data.optimization_tier == OptimizationTier.STANDARD
    
    def test_text_analysis_update(self) -> Any:
        """Test TextAnalysisUpdate model."""
        data = TextAnalysisUpdate(
            status=AnalysisStatus.COMPLETED,
            sentiment_score=0.8,
            processing_time_ms=150.5
        )
        
        assert data.status == AnalysisStatus.COMPLETED
        assert data.sentiment_score == 0.8
        assert data.processing_time_ms == 150.5
    
    def test_batch_analysis_create(self) -> Any:
        """Test BatchAnalysisCreate model."""
        data = BatchAnalysisCreate(
            batch_name="Test Batch",
            analysis_type=AnalysisType.SENTIMENT,
            optimization_tier=OptimizationTier.STANDARD
        )
        
        assert data.batch_name == "Test Batch"
        assert data.analysis_type == AnalysisType.SENTIMENT
    
    def test_validation_errors(self) -> Any:
        """Test Pydantic validation errors."""
        with pytest.raises(ValueError):
            TextAnalysisCreate(
                text_content="",  # Empty content should fail
                analysis_type=AnalysisType.SENTIMENT
            )
        
        with pytest.raises(ValueError):
            TextAnalysisUpdate(
                sentiment_score=1.5  # Score > 1 should fail
            )


# ============================================================================
# Database Manager Tests
# ============================================================================

class TestDatabaseManager:
    """Test SQLAlchemy2Manager."""
    
    @pytest.mark.asyncio
    async def test_initialization(self, test_db_config: DatabaseConfig):
        """Test database manager initialization."""
        manager = SQLAlchemy2Manager(test_db_config)
        await manager.initialize()
        
        assert manager.engine is not None
        assert manager.session_factory is not None
        assert manager.health_status.is_healthy
        
        await manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_session_management(self, db_manager: SQLAlchemy2Manager):
        """Test session management."""
        async with db_manager.get_session() as session:
            assert isinstance(session, AsyncSession)
            assert not session.is_active  # Session should be active
    
    @pytest.mark.asyncio
    async def test_create_text_analysis(self, db_manager: SQLAlchemy2Manager):
        """Test creating text analysis."""
        data = TextAnalysisCreate(
            text_content="Test content for sentiment analysis",
            analysis_type=AnalysisType.SENTIMENT,
            optimization_tier=OptimizationTier.STANDARD
        )
        
        analysis = await db_manager.create_text_analysis(data)
        
        assert analysis.id is not None
        assert analysis.text_content == data.text_content
        assert analysis.analysis_type == data.analysis_type
        assert analysis.status == AnalysisStatus.PENDING
        assert analysis.text_hash is not None
    
    @pytest.mark.asyncio
    async def test_get_text_analysis(self, db_manager: SQLAlchemy2Manager):
        """Test getting text analysis by ID."""
        # Create analysis first
        data = TextAnalysisCreate(
            text_content="Test content",
            analysis_type=AnalysisType.SENTIMENT
        )
        created = await db_manager.create_text_analysis(data)
        
        # Get analysis
        analysis = await db_manager.get_text_analysis(created.id)
        
        assert analysis is not None
        assert analysis.id == created.id
        assert analysis.text_content == created.text_content
    
    @pytest.mark.asyncio
    async def test_get_text_analysis_by_hash(self, db_manager: SQLAlchemy2Manager):
        """Test getting text analysis by hash."""
        # Create analysis first
        data = TextAnalysisCreate(
            text_content="Test content",
            analysis_type=AnalysisType.SENTIMENT
        )
        created = await db_manager.create_text_analysis(data)
        
        # Get analysis by hash
        analysis = await db_manager.get_text_analysis_by_hash(
            created.text_hash, 
            AnalysisType.SENTIMENT
        )
        
        assert analysis is not None
        assert analysis.id == created.id
        assert analysis.text_hash == created.text_hash
    
    @pytest.mark.asyncio
    async def test_update_text_analysis(self, db_manager: SQLAlchemy2Manager):
        """Test updating text analysis."""
        # Create analysis first
        data = TextAnalysisCreate(
            text_content="Test content",
            analysis_type=AnalysisType.SENTIMENT
        )
        created = await db_manager.create_text_analysis(data)
        
        # Update analysis
        update_data = TextAnalysisUpdate(
            status=AnalysisStatus.COMPLETED,
            sentiment_score=0.8,
            processing_time_ms=150.5,
            model_used="distilbert-sentiment"
        )
        
        updated = await db_manager.update_text_analysis(created.id, update_data)
        
        assert updated is not None
        assert updated.status == AnalysisStatus.COMPLETED
        assert updated.sentiment_score == 0.8
        assert updated.processed_at is not None
    
    @pytest.mark.asyncio
    async def test_list_text_analyses(self, db_manager: SQLAlchemy2Manager):
        """Test listing text analyses."""
        # Create multiple analyses
        for i in range(5):
            data = TextAnalysisCreate(
                text_content=f"Test content {i}",
                analysis_type=AnalysisType.SENTIMENT
            )
            await db_manager.create_text_analysis(data)
        
        # List analyses
        analyses = await db_manager.list_text_analyses(
            analysis_type=AnalysisType.SENTIMENT,
            limit=10
        )
        
        assert len(analyses) >= 5
        assert all(a.analysis_type == AnalysisType.SENTIMENT for a in analyses)
    
    @pytest.mark.asyncio
    async def test_delete_text_analysis(self, db_manager: SQLAlchemy2Manager):
        """Test deleting text analysis."""
        # Create analysis first
        data = TextAnalysisCreate(
            text_content="Test content",
            analysis_type=AnalysisType.SENTIMENT
        )
        created = await db_manager.create_text_analysis(data)
        
        # Delete analysis
        success = await db_manager.delete_text_analysis(created.id)
        
        assert success is True
        
        # Verify deletion
        analysis = await db_manager.get_text_analysis(created.id)
        assert analysis is None
    
    @pytest.mark.asyncio
    async def test_create_batch_analysis(self, db_manager: SQLAlchemy2Manager):
        """Test creating batch analysis."""
        data = BatchAnalysisCreate(
            batch_name="Test Batch",
            analysis_type=AnalysisType.SENTIMENT,
            optimization_tier=OptimizationTier.STANDARD
        )
        
        batch = await db_manager.create_batch_analysis(data)
        
        assert batch.id is not None
        assert batch.batch_name == data.batch_name
        assert batch.analysis_type == data.analysis_type
        assert batch.status == AnalysisStatus.PENDING
    
    @pytest.mark.asyncio
    async def test_update_batch_progress(self, db_manager: SQLAlchemy2Manager):
        """Test updating batch progress."""
        # Create batch first
        data = BatchAnalysisCreate(
            batch_name="Test Batch",
            analysis_type=AnalysisType.SENTIMENT
        )
        created = await db_manager.create_batch_analysis(data)
        
        # Update progress
        updated = await db_manager.update_batch_progress(
            created.id, 
            completed_count=5, 
            error_count=1
        )
        
        assert updated is not None
        assert updated.completed_count == 5
        assert updated.error_count == 1
    
    @pytest.mark.asyncio
    async def test_performance_metrics(self, db_manager: SQLAlchemy2Manager):
        """Test performance metrics."""
        # Perform some operations
        data = TextAnalysisCreate(
            text_content="Test content",
            analysis_type=AnalysisType.SENTIMENT
        )
        await db_manager.create_text_analysis(data)
        
        metrics = await db_manager.get_performance_metrics()
        
        assert "database" in metrics
        assert "cache" in metrics
        assert "health" in metrics
        assert metrics["database"]["total_queries"] > 0
    
    @pytest.mark.asyncio
    async def test_health_check(self, db_manager: SQLAlchemy2Manager):
        """Test health check."""
        health = await db_manager.health_check()
        
        assert health.is_healthy is True
        assert health.last_check is not None
        assert health.avg_query_time >= 0


# ============================================================================
# Migration Tests
# ============================================================================

class TestMigrations:
    """Test migration system."""
    
    @pytest.mark.asyncio
    async def test_migration_manager_initialization(self, test_db_config: DatabaseConfig):
        """Test migration manager initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = MigrationManager(test_db_config, migrations_dir=temp_dir)
            await manager.initialize()
            
            assert manager.engine is not None
            assert manager.alembic_cfg is not None
            
            await manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_create_migration(self, migration_manager: MigrationManager):
        """Test creating migration."""
        revision = await migration_manager.create_migration("Test migration")
        
        assert revision is not None
        assert isinstance(revision, str)
    
    @pytest.mark.asyncio
    async def test_migration_history(self, migration_manager: MigrationManager):
        """Test getting migration history."""
        # Create a migration first
        await migration_manager.create_migration("Test migration")
        
        history = await migration_manager.migration_history()
        
        assert isinstance(history, list)
        assert len(history) > 0
    
    @pytest.mark.asyncio
    async def test_check_migrations(self, migration_manager: MigrationManager):
        """Test checking migration status."""
        status = await migration_manager.check_migrations()
        
        assert "current_revision" in status
        assert "head_revision" in status
        assert "is_up_to_date" in status
        assert "pending_migrations" in status


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Test performance aspects."""
    
    @pytest.mark.asyncio
    async def test_bulk_operations(self, db_manager: SQLAlchemy2Manager):
        """Test bulk operations performance."""
        
        # Create multiple analyses
        start_time = time.time()
        
        analyses_data = [
            TextAnalysisCreate(
                text_content=f"Test content {i}",
                analysis_type=AnalysisType.SENTIMENT
            )
            for i in range(100)
        ]
        
        created_analyses = []
        for data in analyses_data:
            analysis = await db_manager.create_text_analysis(data)
            created_analyses.append(analysis)
        
        creation_time = time.time() - start_time
        
        # List all analyses
        start_time = time.time()
        analyses = await db_manager.list_text_analyses(limit=1000)
        listing_time = time.time() - start_time
        
        assert len(analyses) >= 100
        assert creation_time < 10.0  # Should complete within 10 seconds
        assert listing_time < 5.0   # Should complete within 5 seconds
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, test_db_config: DatabaseConfig):
        """Test concurrent operations."""
        manager = SQLAlchemy2Manager(test_db_config)
        await manager.initialize()
        
        try:
            # Create multiple concurrent operations
            async def create_analysis(i: int):
                
    """create_analysis function."""
data = TextAnalysisCreate(
                    text_content=f"Concurrent content {i}",
                    analysis_type=AnalysisType.SENTIMENT
                )
                return await manager.create_text_analysis(data)
            
            # Run concurrent operations
            tasks = [create_analysis(i) for i in range(10)]
            results = await asyncio.gather(*tasks)
            
            assert len(results) == 10
            assert all(r.id is not None for r in results)
            
        finally:
            await manager.cleanup()


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestErrorHandling:
    """Test error handling."""
    
    @pytest.mark.asyncio
    async def test_invalid_analysis_id(self, db_manager: SQLAlchemy2Manager):
        """Test handling of invalid analysis ID."""
        analysis = await db_manager.get_text_analysis(99999)
        assert analysis is None
    
    @pytest.mark.asyncio
    async def test_duplicate_text_hash(self, db_manager: SQLAlchemy2Manager):
        """Test handling of duplicate text hash."""
        data = TextAnalysisCreate(
            text_content="Duplicate content",
            analysis_type=AnalysisType.SENTIMENT
        )
        
        # Create first analysis
        first = await db_manager.create_text_analysis(data)
        assert first is not None
        
        # Try to create duplicate
        second = await db_manager.create_text_analysis(data)
        assert second is not None
        assert second.id == first.id  # Should return existing analysis
    
    @pytest.mark.asyncio
    async def test_invalid_update_data(self, db_manager: SQLAlchemy2Manager):
        """Test handling of invalid update data."""
        # Create analysis first
        data = TextAnalysisCreate(
            text_content="Test content",
            analysis_type=AnalysisType.SENTIMENT
        )
        created = await db_manager.create_text_analysis(data)
        
        # Try to update with invalid data
        update_data = TextAnalysisUpdate(
            sentiment_score=2.0  # Invalid score > 1
        )
        
        # Should raise validation error
        with pytest.raises(Exception):
            await db_manager.update_text_analysis(created.id, update_data)


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Test integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_full_analysis_workflow(self, db_manager: SQLAlchemy2Manager):
        """Test complete analysis workflow."""
        # 1. Create batch analysis
        batch_data = BatchAnalysisCreate(
            batch_name="Integration Test Batch",
            analysis_type=AnalysisType.SENTIMENT
        )
        batch = await db_manager.create_batch_analysis(batch_data)
        
        # 2. Create multiple text analyses
        texts = [
            "This is a positive text.",
            "This is a negative text.",
            "This is a neutral text."
        ]
        
        analyses = []
        for text in texts:
            data = TextAnalysisCreate(
                text_content=text,
                analysis_type=AnalysisType.SENTIMENT
            )
            analysis = await db_manager.create_text_analysis(data)
            analyses.append(analysis)
        
        # 3. Update analyses with results
        for i, analysis in enumerate(analyses):
            update_data = TextAnalysisUpdate(
                status=AnalysisStatus.COMPLETED,
                sentiment_score=0.5 + (i * 0.2),
                processing_time_ms=100.0 + i,
                model_used="test-model"
            )
            await db_manager.update_text_analysis(analysis.id, update_data)
        
        # 4. Update batch progress
        await db_manager.update_batch_progress(
            batch.id, 
            completed_count=len(analyses), 
            error_count=0
        )
        
        # 5. Verify results
        updated_batch = await db_manager.get_batch_analysis(batch.id)
        assert updated_batch.completed_count == len(analyses)
        assert updated_batch.status == AnalysisStatus.COMPLETED
        
        # List completed analyses
        completed_analyses = await db_manager.list_text_analyses(
            status=AnalysisStatus.COMPLETED
        )
        assert len(completed_analyses) >= len(analyses)
    
    @pytest.mark.asyncio
    async def test_migration_integration(self, test_db_config: DatabaseConfig):
        """Test migration integration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create migration manager
            migration_manager = MigrationManager(test_db_config, migrations_dir=temp_dir)
            await migration_manager.initialize()
            
            try:
                # Create initial migration
                revision = await migration_manager.create_migration("Initial migration")
                assert revision is not None
                
                # Run migrations
                success = await migration_manager.upgrade()
                assert success is True
                
                # Check status
                status = await migration_manager.check_migrations()
                assert status["is_up_to_date"] is True
                
            finally:
                await migration_manager.cleanup()


# ============================================================================
# Test Utilities
# ============================================================================

def test_database_config():
    """Test database configuration creation."""
    config = DatabaseConfig(
        url="postgresql+asyncpg://user:pass@localhost/db",
        pool_size=20,
        enable_caching=True
    )
    
    assert config.url == "postgresql+asyncpg://user:pass@localhost/db"
    assert config.pool_size == 20
    assert config.enable_caching is True


def test_enum_values():
    """Test enum values."""
    assert AnalysisType.SENTIMENT == "sentiment"
    assert AnalysisStatus.PENDING == "pending"
    assert OptimizationTier.STANDARD == "standard"


match __name__:
    case "__main__":
    pytest.main([__file__, "-v"]) 