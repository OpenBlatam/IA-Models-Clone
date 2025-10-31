"""
Pytest configuration and shared fixtures for the ads feature tests.

This module provides:
- Shared test fixtures
- Test database setup/teardown
- Mock configurations
- Test environment setup
"""

import pytest
import asyncio
import tempfile
import os
import sqlite3
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Any, List, Optional, Generator
from unittest.mock import Mock, AsyncMock, patch, MagicMock

# Import test utilities
from .utils.test_helpers import create_test_data, cleanup_test_data
from .utils.test_mocks import create_mock_repository, create_mock_service
from .fixtures.test_data import (
    sample_ad_data, sample_campaign_data, sample_group_data,
    sample_performance_data, sample_optimization_data
)

# Import domain entities and value objects
from agents.backend.onyx.server.features.ads.domain.entities import (
    Ad, AdCampaign, AdGroup, AdPerformance
)
from agents.backend.onyx.server.features.ads.domain.value_objects import (
    AdStatus, AdType, Platform, Budget, TargetingCriteria, AdMetrics, AdSchedule
)
from agents.backend.onyx.server.features.ads.domain.repositories import (
    AdRepository, CampaignRepository, GroupRepository, PerformanceRepository
)
from agents.backend.onyx.server.features.ads.domain.services import (
    AdService, CampaignService, OptimizationService
)

# Import application layer components
from agents.backend.onyx.server.features.ads.application.dto import (
    CreateAdRequest, CreateAdResponse, ApproveAdRequest, ApproveAdResponse,
    ActivateAdRequest, ActivateAdResponse, PauseAdRequest, PauseAdResponse,
    ArchiveAdRequest, ArchiveAdResponse, CreateCampaignRequest, CreateCampaignResponse,
    ActivateCampaignRequest, ActivateCampaignResponse, PauseCampaignRequest, PauseCampaignResponse,
    OptimizeAdRequest, OptimizeAdResponse, PerformancePredictionRequest, PerformancePredictionResponse,
    ErrorResponse
)
from agents.backend.onyx.server.features.ads.application.use_cases import (
    CreateAdUseCase, ApproveAdUseCase, ActivateAdUseCase, PauseAdUseCase, ArchiveAdUseCase,
    CreateCampaignUseCase, ActivateCampaignUseCase, PauseCampaignUseCase,
    OptimizeAdUseCase, PredictPerformanceUseCase
)

# Import infrastructure components
from agents.backend.onyx.server.features.ads.infrastructure.database import (
    DatabaseConfig, ConnectionPool, DatabaseManager
)
from agents.backend.onyx.server.features.ads.infrastructure.storage import (
    StorageConfig, LocalStorageStrategy
)
from agents.backend.onyx.server.features.ads.infrastructure.cache import (
    CacheConfig, MemoryCacheStrategy
)
from agents.backend.onyx.server.features.ads.infrastructure.repositories import (
    AdsRepositoryImpl, CampaignRepositoryImpl, GroupRepositoryImpl,
    PerformanceRepositoryImpl, AnalyticsRepositoryImpl, OptimizationRepositoryImpl,
    RepositoryFactory
)

# Import optimization components
from agents.backend.onyx.server.features.ads.optimization.base_optimizer import (
    BaseOptimizer, OptimizationStrategy, OptimizationLevel, OptimizationResult, OptimizationContext
)
from agents.backend.onyx.server.features.ads.optimization.factory import (
    OptimizationFactory, OptimizerType
)
from agents.backend.onyx.server.features.ads.optimization.performance_optimizer import (
    PerformanceOptimizer
)

# Import training components
from agents.backend.onyx.server.features.ads.training.base_trainer import (
    BaseTrainer, TrainingPhase, TrainingStatus, TrainingMetrics, TrainingConfig, TrainingResult
)
from agents.backend.onyx.server.features.ads.training.training_factory import (
    TrainingFactory, TrainerType, TrainerConfig
)
from agents.backend.onyx.server.features.ads.training.pytorch_trainer import (
    PyTorchTrainer
)
from agents.backend.onyx.server.features.ads.training.experiment_tracker import (
    ExperimentTracker, ExperimentConfig, ExperimentRun
)


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "async: mark test as async test"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their names and content."""
    for item in items:
        # Mark async tests
        if asyncio.iscoroutinefunction(item.function):
            item.add_marker(pytest.mark.asyncio)
        
        # Mark slow tests based on name
        if "slow" in item.name.lower() or "performance" in item.name.lower():
            item.add_marker(pytest.mark.slow)
        
        # Mark integration tests based on path
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)


# Shared fixtures
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def temp_dir():
    """Create a temporary directory for the test session."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup will be handled by the OS


@pytest.fixture(scope="session")
def test_db_path(temp_dir):
    """Create a test database path for the session."""
    return os.path.join(temp_dir, "test_ads.db")


@pytest.fixture(scope="session")
def test_storage_path(temp_dir):
    """Create a test storage path for the session."""
    storage_path = os.path.join(temp_dir, "storage")
    os.makedirs(storage_path, exist_ok=True)
    return storage_path


# Database fixtures
@pytest.fixture
async def database_config():
    """Create a test database configuration."""
    return DatabaseConfig(
        host="localhost",
        port=5432,
        database="test_ads_db",
        username="test_user",
        password="test_pass",
        pool_size=5,
        max_overflow=10,
        pool_timeout=30,
        pool_recycle=3600
    )


@pytest.fixture
async def mock_connection_pool():
    """Create a mock connection pool."""
    pool = Mock(spec=ConnectionPool)
    pool.get_session.return_value = AsyncMock()
    pool.health_check.return_value = {
        "status": "healthy",
        "message": "All good",
        "timestamp": datetime.now()
    }
    return pool


@pytest.fixture
async def database_manager(mock_connection_pool):
    """Create a database manager with mocked connection pool."""
    return DatabaseManager(connection_pool=mock_connection_pool)


# Storage fixtures
@pytest.fixture
async def storage_config(test_storage_path):
    """Create a test storage configuration."""
    return StorageConfig(
        storage_type="local",
        base_path=test_storage_path,
        max_file_size=10485760,  # 10MB
        allowed_extensions=[".jpg", ".png", ".pdf", ".txt", ".json"],
        compression_enabled=False,
        encryption_enabled=False
    )


@pytest.fixture
async def local_storage_strategy(storage_config):
    """Create a local storage strategy."""
    return LocalStorageStrategy(config=storage_config)


# Cache fixtures
@pytest.fixture
async def cache_config():
    """Create a test cache configuration."""
    return CacheConfig(
        cache_type="memory",
        ttl=300,  # 5 minutes
        max_size=100,
        compression_enabled=False,
        encryption_enabled=False
    )


@pytest.fixture
async def memory_cache_strategy(cache_config):
    """Create a memory cache strategy."""
    return MemoryCacheStrategy(config=cache_config)


# Repository fixtures
@pytest.fixture
async def mock_ads_repository():
    """Create a mock ads repository."""
    return create_mock_repository(AdsRepositoryImpl, "ads")


@pytest.fixture
async def mock_campaign_repository():
    """Create a mock campaign repository."""
    return create_mock_repository(CampaignRepositoryImpl, "campaign")


@pytest.fixture
async def mock_group_repository():
    """Create a mock group repository."""
    return create_mock_repository(GroupRepositoryImpl, "group")


@pytest.fixture
async def mock_performance_repository():
    """Create a mock performance repository."""
    return create_mock_repository(PerformanceRepositoryImpl, "performance")


@pytest.fixture
async def mock_analytics_repository():
    """Create a mock analytics repository."""
    return create_mock_repository(AnalyticsRepositoryImpl, "analytics")


@pytest.fixture
async def mock_optimization_repository():
    """Create a mock optimization repository."""
    return create_mock_repository(OptimizationRepositoryImpl, "optimization")


@pytest.fixture
async def repository_factory(database_manager):
    """Create a repository factory."""
    return RepositoryFactory(database_manager=database_manager)


# Service fixtures
@pytest.fixture
async def mock_ad_service(mock_ads_repository, mock_campaign_repository):
    """Create a mock ad service."""
    return create_mock_service(AdService, {
        "ad_repository": mock_ads_repository,
        "campaign_repository": mock_campaign_repository
    })


@pytest.fixture
async def mock_campaign_service(mock_campaign_repository, mock_group_repository):
    """Create a mock campaign service."""
    return create_mock_service(CampaignService, {
        "campaign_repository": mock_campaign_repository,
        "group_repository": mock_group_repository
    })


@pytest.fixture
async def mock_optimization_service(mock_performance_repository, mock_analytics_repository):
    """Create a mock optimization service."""
    return create_mock_service(OptimizationService, {
        "performance_repository": mock_performance_repository,
        "analytics_repository": mock_analytics_repository
    })


# Use case fixtures
@pytest.fixture
async def create_ad_use_case(mock_ad_service, mock_campaign_service):
    """Create a create ad use case."""
    return CreateAdUseCase(
        ad_service=mock_ad_service,
        campaign_service=mock_campaign_service
    )


@pytest.fixture
async def approve_ad_use_case(mock_ad_service):
    """Create an approve ad use case."""
    return ApproveAdUseCase(ad_service=mock_ad_service)


@pytest.fixture
async def activate_ad_use_case(mock_ad_service):
    """Create an activate ad use case."""
    return ActivateAdUseCase(ad_service=mock_ad_service)


@pytest.fixture
async def create_campaign_use_case(mock_campaign_service):
    """Create a create campaign use case."""
    return CreateCampaignUseCase(campaign_service=mock_campaign_service)


@pytest.fixture
async def optimize_ad_use_case(mock_optimization_service):
    """Create an optimize ad use case."""
    return OptimizeAdUseCase(optimization_service=mock_optimization_service)


# Optimization fixtures
@pytest.fixture
async def performance_optimizer():
    """Create a performance optimizer."""
    return PerformanceOptimizer()


@pytest.fixture
async def optimization_factory():
    """Create an optimization factory."""
    return OptimizationFactory()


@pytest.fixture
async def optimization_context():
    """Create an optimization context."""
    return OptimizationContext(
        current_metrics={
            "cpu_usage": 0.75,
            "memory_usage": 0.80,
            "response_time": 150
        },
        target_metrics={
            "cpu_usage": 0.50,
            "memory_usage": 0.60,
            "response_time": 100
        },
        constraints={
            "max_memory": 8192,
            "max_cpu": 0.90
        },
        optimization_level=OptimizationLevel.STANDARD
    )


# Training fixtures
@pytest.fixture
async def pytorch_trainer():
    """Create a PyTorch trainer."""
    return PyTorchTrainer()


@pytest.fixture
async def training_factory():
    """Create a training factory."""
    return TrainingFactory()


@pytest.fixture
async def experiment_tracker(test_db_path):
    """Create an experiment tracker."""
    return ExperimentTracker(db_path=test_db_path)


@pytest.fixture
async def training_config():
    """Create a training configuration."""
    return TrainingConfig(
        epochs=100,
        batch_size=32,
        learning_rate=0.001,
        optimizer="adam",
        scheduler="cosine",
        early_stopping_patience=10,
        validation_split=0.2,
        checkpoint_dir="./checkpoints"
    )


# Test data fixtures
@pytest.fixture
async def sample_ad():
    """Create a sample ad entity."""
    return Ad(
        id="ad_123",
        campaign_id="campaign_456",
        group_id="group_789",
        name="Test Ad",
        content="This is a test ad",
        ad_type=AdType.TEXT,
        status=AdStatus.DRAFT,
        platform=Platform.FACEBOOK,
        targeting_criteria=TargetingCriteria(
            demographics={"age_range": "25-34"},
            interests=["technology"],
            location={},
            behavior=[]
        ),
        budget=Budget(
            daily_limit=Decimal("50.00"),
            total_limit=Decimal("500.00"),
            currency="USD"
        ),
        created_at=datetime.now()
    )


@pytest.fixture
async def sample_campaign():
    """Create a sample campaign entity."""
    return AdCampaign(
        id="campaign_123",
        name="Test Campaign",
        description="A test advertising campaign",
        objective="awareness",
        status=AdStatus.DRAFT,
        budget=Budget(
            daily_limit=Decimal("200.00"),
            total_limit=Decimal("2000.00"),
            currency="USD"
        ),
        start_date=datetime.now(),
        end_date=datetime.now() + timedelta(days=60),
        created_at=datetime.now()
    )


@pytest.fixture
async def sample_group():
    """Create a sample ad group entity."""
    return AdGroup(
        id="group_123",
        campaign_id="campaign_456",
        name="Test Ad Group",
        description="A test ad group",
        status=AdStatus.DRAFT,
        targeting_criteria=TargetingCriteria(
            demographics={"age_range": "25-34"},
            interests=["technology"],
            location={},
            behavior=[]
        ),
        bid_amount=Decimal("2.50"),
        created_at=datetime.now()
    )


@pytest.fixture
async def sample_performance():
    """Create a sample performance entity."""
    return AdPerformance(
        id="perf_123",
        ad_id="ad_456",
        date=datetime.now().date(),
        metrics=AdMetrics(
            impressions=1000,
            clicks=50,
            conversions=5,
            spend=Decimal("100.00"),
            ctr=0.05,
            cpc=Decimal("2.00"),
            cpm=Decimal("100.00")
        ),
        created_at=datetime.now()
    )


# Request/Response fixtures
@pytest.fixture
async def create_ad_request():
    """Create a create ad request."""
    return CreateAdRequest(
        campaign_id="campaign_123",
        group_id="group_456",
        name="Test Ad",
        content="This is a test ad",
        ad_type=AdType.TEXT,
        platform=Platform.FACEBOOK,
        targeting_criteria={
            "demographics": {"age_range": "25-34"},
            "interests": ["technology"],
            "location": {},
            "behavior": []
        },
        budget={
            "daily_limit": "50.00",
            "total_limit": "500.00",
            "currency": "USD"
        }
    )


@pytest.fixture
async def create_campaign_request():
    """Create a create campaign request."""
    return CreateCampaignRequest(
        name="Test Campaign",
        description="A test advertising campaign",
        objective="awareness",
        budget={
            "daily_limit": "200.00",
            "total_limit": "2000.00",
            "currency": "USD"
        },
        start_date=datetime.now(),
        end_date=datetime.now() + timedelta(days=60)
    )


@pytest.fixture
async def optimize_ad_request():
    """Create an optimize ad request."""
    return OptimizeAdRequest(
        ad_id="ad_123",
        optimization_type="performance",
        parameters={
            "target_ctr": 0.08,
            "max_cpc": "3.00",
            "budget_adjustment": 0.1
        }
    )


# Test utilities
@pytest.fixture
async def test_data_manager():
    """Create a test data manager for setup and cleanup."""
    return {
        "setup": create_test_data,
        "cleanup": cleanup_test_data
    }


# Cleanup fixtures
@pytest.fixture(autouse=True)
async def cleanup_after_test():
    """Cleanup after each test."""
    yield
    # This will run after each test
    # Add any cleanup logic here if needed


# Performance testing fixtures
@pytest.fixture
async def performance_test_config():
    """Create a performance test configuration."""
    return {
        "iterations": 1000,
        "timeout": 30,
        "memory_limit": 1024 * 1024 * 100,  # 100MB
        "cpu_limit": 0.8
    }


# Mock data fixtures
@pytest.fixture
async def mock_redis_client():
    """Create a mock Redis client."""
    redis_mock = AsyncMock()
    redis_mock.get.return_value = None
    redis_mock.set.return_value = True
    redis_mock.delete.return_value = 1
    redis_mock.scan_iter.return_value = []
    return redis_mock


@pytest.fixture
async def mock_http_client():
    """Create a mock HTTP client."""
    http_mock = AsyncMock()
    http_mock.get.return_value = AsyncMock(status_code=200, json=lambda: {"success": True})
    http_mock.post.return_value = AsyncMock(status_code=200, json=lambda: {"success": True})
    return http_mock


# Environment fixtures
@pytest.fixture(autouse=True)
async def test_environment():
    """Set up test environment variables."""
    # Set test environment
    os.environ["TESTING"] = "true"
    os.environ["ENVIRONMENT"] = "test"
    
    yield
    
    # Cleanup environment variables
    if "TESTING" in os.environ:
        del os.environ["TESTING"]
    if "ENVIRONMENT" in os.environ:
        del os.environ["ENVIRONMENT"]


# Database setup/teardown
@pytest.fixture(scope="session", autouse=True)
async def setup_test_database(test_db_path):
    """Set up test database for the session."""
    # Create test database
    conn = sqlite3.connect(test_db_path)
    cursor = conn.cursor()
    
    # Create test tables (simplified for testing)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS test_ads (
            id TEXT PRIMARY KEY,
            name TEXT,
            content TEXT,
            status TEXT,
            created_at TIMESTAMP
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS test_campaigns (
            id TEXT PRIMARY KEY,
            name TEXT,
            description TEXT,
            status TEXT,
            created_at TIMESTAMP
        )
    """)
    
    conn.commit()
    conn.close()
    
    yield
    
    # Cleanup test database
    if os.path.exists(test_db_path):
        os.remove(test_db_path)


# Storage setup/teardown
@pytest.fixture(scope="session", autouse=True)
async def setup_test_storage(test_storage_path):
    """Set up test storage for the session."""
    # Create test storage directory
    os.makedirs(test_storage_path, exist_ok=True)
    
    yield
    
    # Cleanup test storage
    import shutil
    if os.path.exists(test_storage_path):
        shutil.rmtree(test_storage_path)


# Logging configuration for tests
@pytest.fixture(autouse=True)
async def configure_test_logging():
    """Configure logging for tests."""
    import logging
    
    # Set logging level for tests
    logging.basicConfig(
        level=logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Suppress specific logger noise during tests
    logging.getLogger("asyncio").setLevel(logging.ERROR)
    logging.getLogger("sqlalchemy").setLevel(logging.ERROR)
    
    yield
