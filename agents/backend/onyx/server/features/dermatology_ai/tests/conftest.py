"""
Pytest Configuration and Fixtures
Shared test utilities and fixtures
"""

import pytest
import asyncio
from typing import AsyncGenerator
from unittest.mock import Mock, AsyncMock

from core.domain.interfaces import (
    IAnalysisRepository,
    IUserRepository,
    IProductRepository,
    IImageProcessor,
    IAnalysisService,
    IRecommendationService,
    ICacheService,
    IEventPublisher,
)
from core.service_factory import get_service_factory
from core.plugin_system import get_plugin_registry


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def mock_analysis_repository() -> IAnalysisRepository:
    """Mock analysis repository"""
    repo = Mock(spec=IAnalysisRepository)
    repo.create = AsyncMock()
    repo.get_by_id = AsyncMock()
    repo.get_by_user = AsyncMock(return_value=[])
    repo.update = AsyncMock()
    repo.delete = AsyncMock(return_value=True)
    return repo


@pytest.fixture
async def mock_user_repository() -> IUserRepository:
    """Mock user repository"""
    repo = Mock(spec=IUserRepository)
    repo.create = AsyncMock()
    repo.get_by_id = AsyncMock()
    repo.get_by_email = AsyncMock()
    repo.update = AsyncMock()
    return repo


@pytest.fixture
async def mock_image_processor() -> IImageProcessor:
    """Mock image processor"""
    processor = Mock(spec=IImageProcessor)
    processor.process = AsyncMock(return_value={"metrics": {}})
    processor.validate = AsyncMock(return_value=True)
    return processor


@pytest.fixture
async def mock_cache_service() -> ICacheService:
    """Mock cache service"""
    cache = Mock(spec=ICacheService)
    cache.get = AsyncMock(return_value=None)
    cache.set = AsyncMock(return_value=True)
    cache.delete = AsyncMock(return_value=True)
    return cache


@pytest.fixture
async def mock_event_publisher() -> IEventPublisher:
    """Mock event publisher"""
    publisher = Mock(spec=IEventPublisher)
    publisher.publish = AsyncMock(return_value=True)
    return publisher


@pytest.fixture
async def service_factory():
    """Service factory for testing"""
    factory = get_service_factory()
    yield factory
    factory.clear_request_scope()


@pytest.fixture
async def plugin_registry():
    """Plugin registry for testing"""
    registry = get_plugin_registry()
    yield registry
    await registry.shutdown_all()


@pytest.fixture
def sample_analysis_data():
    """Sample analysis data for testing"""
    from core.domain.entities import Analysis, SkinMetrics, Condition, AnalysisStatus, SkinType
    
    metrics = SkinMetrics(
        overall_score=75.0,
        texture_score=80.0,
        hydration_score=70.0,
        elasticity_score=75.0,
        pigmentation_score=80.0,
        pore_size_score=70.0,
        wrinkles_score=75.0,
        redness_score=80.0,
        dark_spots_score=75.0,
    )
    
    conditions = [
        Condition(
            name="acne",
            confidence=0.65,
            severity="moderate",
            description="Mild acne detected"
        )
    ]
    
    return {
        "id": "test-analysis-123",
        "user_id": "test-user-123",
        "metrics": metrics,
        "conditions": conditions,
        "skin_type": SkinType.COMBINATION,
        "status": AnalysisStatus.COMPLETED,
    }


@pytest.fixture
def sample_user_data():
    """Sample user data for testing"""
    from core.domain.entities import User, SkinType
    
    return {
        "id": "test-user-123",
        "email": "test@example.com",
        "name": "Test User",
        "skin_type": SkinType.COMBINATION,
        "preferences": {},
    }


@pytest.fixture
def mock_recommendation_service():
    """Mock recommendation service"""
    from core.domain.interfaces import IRecommendationService
    service = Mock(spec=IRecommendationService)
    service.generate_recommendations = AsyncMock(return_value={
        "routine": {
            "morning": [],
            "evening": [],
            "weekly": []
        },
        "specific_recommendations": [],
        "tips": []
    })
    return service


@pytest.fixture
def mock_analysis_service():
    """Mock analysis service"""
    from core.domain.interfaces import IAnalysisService
    from core.domain.entities import Analysis, AnalysisStatus, SkinMetrics, SkinType
    
    service = Mock(spec=IAnalysisService)
    service.analyze_image = AsyncMock(return_value=Analysis(
        id="test-analysis-123",
        user_id="test-user-123",
        metrics=SkinMetrics(
            overall_score=75.0,
            texture_score=80.0,
            hydration_score=70.0,
            elasticity_score=75.0,
            pigmentation_score=80.0,
            pore_size_score=70.0,
            wrinkles_score=75.0,
            redness_score=80.0,
            dark_spots_score=75.0
        ),
        conditions=[],
        skin_type=SkinType.COMBINATION,
        status=AnalysisStatus.COMPLETED
    ))
    return service


@pytest.fixture
def mock_product_repository():
    """Mock product repository"""
    from core.domain.interfaces import IProductRepository
    repo = Mock(spec=IProductRepository)
    repo.get_by_id = AsyncMock(return_value=None)
    repo.search = AsyncMock(return_value=[])
    repo.create = AsyncMock()
    repo.update = AsyncMock()
    return repo


@pytest.fixture
def sample_image_bytes():
    """Create sample image bytes for testing"""
    from tests.test_helpers import TestDataBuilder
    from PIL import Image
    import io
    
    img = Image.new('RGB', (200, 200), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    return img_bytes.read()


@pytest.fixture
def test_analysis():
    """Create test analysis using builder"""
    from tests.test_helpers import build_analysis
    return build_analysis()


@pytest.fixture
def test_user():
    """Create test user using builder"""
    from tests.test_helpers import build_user
    return build_user()


@pytest.fixture
def test_product():
    """Create test product using builder"""
    from tests.test_helpers import build_product
    return build_product()


@pytest.fixture
def sample_video_bytes():
    """Create sample video bytes for testing"""
    # Return fake video bytes
    return b"fake video content"


@pytest.fixture
def performance_monitor():
    """Create performance monitor for testing"""
    from core.infrastructure.performance_monitor import PerformanceMonitor
    return PerformanceMonitor()


@pytest.fixture
def cache_strategy_manager():
    """Create cache strategy manager for testing"""
    from core.infrastructure.cache_strategies import CacheStrategyManager
    return CacheStrategyManager()


@pytest.fixture
def query_optimizer():
    """Create query optimizer for testing"""
    from core.infrastructure.query_optimizer import QueryOptimizer
    return QueryOptimizer()


@pytest.fixture
def security_validator():
    """Create security validator for testing"""
    from core.infrastructure.security_utils import SecurityValidator
    return SecurityValidator()


@pytest.fixture
def sample_metrics():
    """Sample skin metrics for testing"""
    from core.domain.entities import SkinMetrics
    
    return SkinMetrics(
        overall_score=75.0,
        texture_score=80.0,
        hydration_score=70.0,
        elasticity_score=75.0,
        pigmentation_score=80.0,
        pore_size_score=70.0,
        wrinkles_score=75.0,
        redness_score=80.0,
        dark_spots_score=75.0
    )


@pytest.fixture
def sample_conditions():
    """Sample conditions for testing"""
    from core.domain.entities import Condition
    
    return [
        Condition(
            name="acne",
            confidence=0.65,
            severity="moderate",
            description="Mild acne detected"
        ),
        Condition(
            name="dryness",
            confidence=0.50,
            severity="mild",
            description="Slight dryness"
        )
    ]


@pytest.fixture
def complete_analysis(sample_metrics, sample_conditions):
    """Complete analysis entity for testing"""
    from core.domain.entities import Analysis, AnalysisStatus, SkinType
    
    return Analysis(
        id="test-analysis-123",
        user_id="test-user-123",
        metrics=sample_metrics,
        conditions=sample_conditions,
        skin_type=SkinType.COMBINATION,
        status=AnalysisStatus.COMPLETED
    )














Shared test utilities and fixtures
"""

import pytest
import asyncio
from typing import AsyncGenerator
from unittest.mock import Mock, AsyncMock

from core.domain.interfaces import (
    IAnalysisRepository,
    IUserRepository,
    IProductRepository,
    IImageProcessor,
    IAnalysisService,
    IRecommendationService,
    ICacheService,
    IEventPublisher,
)
from core.service_factory import get_service_factory
from core.plugin_system import get_plugin_registry

# Import test helpers for use in fixtures
from tests.test_helpers import (
    MockFactory,
    TestDataBuilder,
    build_analysis,
    build_user,
    build_product,
    build_metrics
)


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def mock_analysis_repository() -> IAnalysisRepository:
    """Mock analysis repository"""
    repo = Mock(spec=IAnalysisRepository)
    repo.create = AsyncMock()
    repo.get_by_id = AsyncMock()
    repo.get_by_user = AsyncMock(return_value=[])
    repo.update = AsyncMock()
    repo.delete = AsyncMock(return_value=True)
    return repo


@pytest.fixture
async def mock_user_repository() -> IUserRepository:
    """Mock user repository"""
    repo = Mock(spec=IUserRepository)
    repo.create = AsyncMock()
    repo.get_by_id = AsyncMock()
    repo.get_by_email = AsyncMock()
    repo.update = AsyncMock()
    return repo


@pytest.fixture
async def mock_image_processor() -> IImageProcessor:
    """Mock image processor"""
    processor = Mock(spec=IImageProcessor)
    processor.process = AsyncMock(return_value={"metrics": {}})
    processor.validate = AsyncMock(return_value=True)
    return processor


@pytest.fixture
async def mock_cache_service() -> ICacheService:
    """Mock cache service"""
    cache = Mock(spec=ICacheService)
    cache.get = AsyncMock(return_value=None)
    cache.set = AsyncMock(return_value=True)
    cache.delete = AsyncMock(return_value=True)
    return cache


@pytest.fixture
async def mock_event_publisher() -> IEventPublisher:
    """Mock event publisher"""
    publisher = Mock(spec=IEventPublisher)
    publisher.publish = AsyncMock(return_value=True)
    return publisher


@pytest.fixture
async def service_factory():
    """Service factory for testing"""
    factory = get_service_factory()
    yield factory
    factory.clear_request_scope()


@pytest.fixture
async def plugin_registry():
    """Plugin registry for testing"""
    registry = get_plugin_registry()
    yield registry
    await registry.shutdown_all()


@pytest.fixture
def sample_analysis_data():
    """Sample analysis data for testing"""
    from core.domain.entities import Analysis, SkinMetrics, Condition, AnalysisStatus, SkinType
    
    metrics = SkinMetrics(
        overall_score=75.0,
        texture_score=80.0,
        hydration_score=70.0,
        elasticity_score=75.0,
        pigmentation_score=80.0,
        pore_size_score=70.0,
        wrinkles_score=75.0,
        redness_score=80.0,
        dark_spots_score=75.0,
    )
    
    conditions = [
        Condition(
            name="acne",
            confidence=0.65,
            severity="moderate",
            description="Mild acne detected"
        )
    ]
    
    return {
        "id": "test-analysis-123",
        "user_id": "test-user-123",
        "metrics": metrics,
        "conditions": conditions,
        "skin_type": SkinType.COMBINATION,
        "status": AnalysisStatus.COMPLETED,
    }


@pytest.fixture
def sample_user_data():
    """Sample user data for testing"""
    from core.domain.entities import User, SkinType
    
    return {
        "id": "test-user-123",
        "email": "test@example.com",
        "name": "Test User",
        "skin_type": SkinType.COMBINATION,
        "preferences": {},
    }


@pytest.fixture
def mock_recommendation_service():
    """Mock recommendation service"""
    from core.domain.interfaces import IRecommendationService
    service = Mock(spec=IRecommendationService)
    service.generate_recommendations = AsyncMock(return_value={
        "routine": {
            "morning": [],
            "evening": [],
            "weekly": []
        },
        "specific_recommendations": [],
        "tips": []
    })
    return service


@pytest.fixture
def mock_analysis_service():
    """Mock analysis service"""
    from core.domain.interfaces import IAnalysisService
    from core.domain.entities import Analysis, AnalysisStatus, SkinMetrics, SkinType
    
    service = Mock(spec=IAnalysisService)
    service.analyze_image = AsyncMock(return_value=Analysis(
        id="test-analysis-123",
        user_id="test-user-123",
        metrics=SkinMetrics(
            overall_score=75.0,
            texture_score=80.0,
            hydration_score=70.0,
            elasticity_score=75.0,
            pigmentation_score=80.0,
            pore_size_score=70.0,
            wrinkles_score=75.0,
            redness_score=80.0,
            dark_spots_score=75.0
        ),
        conditions=[],
        skin_type=SkinType.COMBINATION,
        status=AnalysisStatus.COMPLETED
    ))
    return service


@pytest.fixture
def mock_product_repository():
    """Mock product repository"""
    from core.domain.interfaces import IProductRepository
    repo = Mock(spec=IProductRepository)
    repo.get_by_id = AsyncMock(return_value=None)
    repo.search = AsyncMock(return_value=[])
    repo.create = AsyncMock()
    repo.update = AsyncMock()
    return repo


@pytest.fixture
def sample_image_bytes():
    """Create sample image bytes for testing"""
    from PIL import Image
    import io
    
    img = Image.new('RGB', (200, 200), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    return img_bytes.read()


@pytest.fixture
def sample_video_bytes():
    """Create sample video bytes for testing"""
    # Return fake video bytes
    return b"fake video content"


@pytest.fixture
def performance_monitor():
    """Create performance monitor for testing"""
    from core.infrastructure.performance_monitor import PerformanceMonitor
    return PerformanceMonitor()


@pytest.fixture
def cache_strategy_manager():
    """Create cache strategy manager for testing"""
    from core.infrastructure.cache_strategies import CacheStrategyManager
    return CacheStrategyManager()


@pytest.fixture
def query_optimizer():
    """Create query optimizer for testing"""
    from core.infrastructure.query_optimizer import QueryOptimizer
    return QueryOptimizer()


@pytest.fixture
def security_validator():
    """Create security validator for testing"""
    from core.infrastructure.security_utils import SecurityValidator
    return SecurityValidator()


@pytest.fixture
def sample_metrics():
    """Sample skin metrics for testing"""
    from core.domain.entities import SkinMetrics
    
    return SkinMetrics(
        overall_score=75.0,
        texture_score=80.0,
        hydration_score=70.0,
        elasticity_score=75.0,
        pigmentation_score=80.0,
        pore_size_score=70.0,
        wrinkles_score=75.0,
        redness_score=80.0,
        dark_spots_score=75.0
    )


@pytest.fixture
def sample_conditions():
    """Sample conditions for testing"""
    from core.domain.entities import Condition
    
    return [
        Condition(
            name="acne",
            confidence=0.65,
            severity="moderate",
            description="Mild acne detected"
        ),
        Condition(
            name="dryness",
            confidence=0.50,
            severity="mild",
            description="Slight dryness"
        )
    ]


@pytest.fixture
def complete_analysis(sample_metrics, sample_conditions):
    """Complete analysis entity for testing"""
    from core.domain.entities import Analysis, AnalysisStatus, SkinType
    
    return Analysis(
        id="test-analysis-123",
        user_id="test-user-123",
        metrics=sample_metrics,
        conditions=sample_conditions,
        skin_type=SkinType.COMBINATION,
        status=AnalysisStatus.COMPLETED
    )













