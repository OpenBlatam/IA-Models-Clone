"""
Test Helpers
Common helper functions and utilities for tests
"""

from typing import Any, Dict, List, Optional
from unittest.mock import Mock, AsyncMock, MagicMock
import uuid
from datetime import datetime

from core.domain.entities import (
    Analysis,
    AnalysisStatus,
    SkinMetrics,
    Condition,
    SkinType,
    User,
    Product
)


class MockFactory:
    """Factory for creating common mocks"""
    
    @staticmethod
    def create_repository_mock(interface_class, methods: Optional[List[str]] = None):
        """Create a repository mock with common methods"""
        mock = Mock(spec=interface_class)
        common_methods = ["create", "get_by_id", "update", "delete", "get_by_user", "search"]
        methods = methods or common_methods
        
        for method in methods:
            if hasattr(interface_class, method):
                setattr(mock, method, AsyncMock())
        
        return mock
    
    @staticmethod
    def create_service_mock(interface_class, methods: Optional[List[str]] = None):
        """Create a service mock with common methods"""
        mock = Mock(spec=interface_class)
        if methods:
            for method in methods:
                if hasattr(interface_class, method):
                    setattr(mock, method, AsyncMock())
        return mock
    
    @staticmethod
    def create_cache_mock():
        """Create a cache service mock"""
        cache = Mock()
        cache.get = AsyncMock(return_value=None)
        cache.set = AsyncMock(return_value=True)
        cache.delete = AsyncMock(return_value=True)
        cache.clear = AsyncMock(return_value=True)
        return cache
    
    @staticmethod
    def create_event_publisher_mock():
        """Create an event publisher mock"""
        publisher = Mock()
        publisher.publish = AsyncMock(return_value=True)
        publisher.publish_batch = AsyncMock(return_value=True)
        return publisher


class TestDataBuilder:
    """Builder for creating test data"""
    
    @staticmethod
    def build_analysis(
        analysis_id: Optional[str] = None,
        user_id: Optional[str] = None,
        status: AnalysisStatus = AnalysisStatus.COMPLETED,
        with_metrics: bool = True,
        with_conditions: bool = True,
        skin_type: SkinType = SkinType.COMBINATION
    ) -> Analysis:
        """Build an Analysis entity for testing"""
        analysis_id = analysis_id or f"analysis-{uuid.uuid4().hex[:8]}"
        user_id = user_id or f"user-{uuid.uuid4().hex[:8]}"
        
        metrics = None
        if with_metrics:
            metrics = SkinMetrics(
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
        
        conditions = []
        if with_conditions:
            conditions = [
                Condition(
                    name="acne",
                    confidence=0.65,
                    severity="moderate",
                    description="Mild acne detected"
                )
            ]
        
        return Analysis(
            id=analysis_id,
            user_id=user_id,
            metrics=metrics,
            conditions=conditions,
            skin_type=skin_type,
            status=status
        )
    
    @staticmethod
    def build_user(
        user_id: Optional[str] = None,
        email: Optional[str] = None,
        skin_type: SkinType = SkinType.COMBINATION
    ) -> User:
        """Build a User entity for testing"""
        user_id = user_id or f"user-{uuid.uuid4().hex[:8]}"
        email = email or f"test-{uuid.uuid4().hex[:8]}@example.com"
        
        return User(
            id=user_id,
            email=email,
            name="Test User",
            skin_type=skin_type,
            preferences={}
        )
    
    @staticmethod
    def build_product(
        product_id: Optional[str] = None,
        name: Optional[str] = None
    ) -> Product:
        """Build a Product entity for testing"""
        product_id = product_id or f"product-{uuid.uuid4().hex[:8]}"
        name = name or f"Test Product {uuid.uuid4().hex[:8]}"
        
        return Product(
            id=product_id,
            name=name,
            brand="Test Brand",
            category="cleanser",
            ingredients=[],
            price=29.99
        )
    
    @staticmethod
    def build_metrics(
        overall_score: float = 75.0,
        **kwargs
    ) -> SkinMetrics:
        """Build SkinMetrics for testing"""
        defaults = {
            "texture_score": 80.0,
            "hydration_score": 70.0,
            "elasticity_score": 75.0,
            "pigmentation_score": 80.0,
            "pore_size_score": 70.0,
            "wrinkles_score": 75.0,
            "redness_score": 80.0,
            "dark_spots_score": 75.0
        }
        defaults.update(kwargs)
        defaults["overall_score"] = overall_score
        
        return SkinMetrics(**defaults)


class AssertionHelpers:
    """Helper functions for assertions"""
    
    @staticmethod
    def assert_analysis_valid(analysis: Analysis):
        """Assert analysis is valid"""
        assert analysis is not None
        assert analysis.id is not None
        assert analysis.user_id is not None
        assert analysis.status in AnalysisStatus
    
    @staticmethod
    def assert_metrics_valid(metrics: SkinMetrics):
        """Assert metrics are valid"""
        assert metrics is not None
        assert 0 <= metrics.overall_score <= 100
        assert 0 <= metrics.texture_score <= 100
        assert 0 <= metrics.hydration_score <= 100
    
    @staticmethod
    def assert_condition_valid(condition: Condition):
        """Assert condition is valid"""
        assert condition is not None
        assert condition.name is not None
        assert 0 <= condition.confidence <= 1
        assert condition.severity in ["mild", "moderate", "severe"]
    
    @staticmethod
    def assert_response_structure(response_data: Dict[str, Any], required_keys: List[str]):
        """Assert response has required structure"""
        for key in required_keys:
            assert key in response_data, f"Missing required key: {key}"
    
    @staticmethod
    def assert_api_response(response, expected_status: int = 200):
        """Assert API response is valid"""
        assert response.status_code == expected_status
        assert response.headers.get("content-type") == "application/json"


class AsyncTestHelpers:
    """Helpers for async testing"""
    
    @staticmethod
    async def wait_for_condition(condition_func, timeout: float = 1.0, interval: float = 0.1):
        """Wait for a condition to become true"""
        import asyncio
        elapsed = 0.0
        while elapsed < timeout:
            if await condition_func() if asyncio.iscoroutinefunction(condition_func) else condition_func():
                return True
            await asyncio.sleep(interval)
            elapsed += interval
        return False
    
    @staticmethod
    async def run_concurrent_tasks(tasks: List, max_concurrent: int = 5):
        """Run tasks concurrently with limit"""
        import asyncio
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def run_with_semaphore(task):
            async with semaphore:
                return await task
        
        return await asyncio.gather(*[run_with_semaphore(task) for task in tasks])


class ResponseHelpers:
    """Helpers for API response testing"""
    
    @staticmethod
    def create_success_response(data: Dict[str, Any], status_code: int = 200) -> Dict[str, Any]:
        """Create a success response structure"""
        return {
            "success": True,
            "status_code": status_code,
            "data": data
        }
    
    @staticmethod
    def create_error_response(message: str, status_code: int = 400, error_code: str = None) -> Dict[str, Any]:
        """Create an error response structure"""
        response = {
            "success": False,
            "status_code": status_code,
            "error": message
        }
        if error_code:
            response["error_code"] = error_code
        return response
    
    @staticmethod
    def create_paginated_response(items: List[Any], page: int = 1, page_size: int = 10, total: int = None) -> Dict[str, Any]:
        """Create a paginated response structure"""
        total = total or len(items)
        return {
            "items": items,
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total": total,
                "pages": (total + page_size - 1) // page_size
            }
        }


class MockHelpers:
    """Additional mock helpers"""
    
    @staticmethod
    def create_async_mock_with_side_effect(side_effects: List[Any]):
        """Create async mock that returns different values on each call"""
        from unittest.mock import AsyncMock
        mock = AsyncMock()
        mock.side_effect = side_effects
        return mock
    
    @staticmethod
    def create_mock_with_side_effect(side_effects: List[Any]):
        """Create mock that returns different values on each call"""
        from unittest.mock import Mock
        mock = Mock()
        mock.side_effect = side_effects
        return mock
    
    @staticmethod
    def create_failing_mock(exception: Exception):
        """Create mock that raises exception"""
        from unittest.mock import AsyncMock
        mock = AsyncMock()
        mock.side_effect = exception
        return mock


# Convenience exports
create_repository_mock = MockFactory.create_repository_mock
create_service_mock = MockFactory.create_service_mock
create_cache_mock = MockFactory.create_cache_mock
create_event_publisher_mock = MockFactory.create_event_publisher_mock

build_analysis = TestDataBuilder.build_analysis
build_user = TestDataBuilder.build_user
build_product = TestDataBuilder.build_product
build_metrics = TestDataBuilder.build_metrics


Common helper functions and utilities for tests
"""

from typing import Any, Dict, List, Optional
from unittest.mock import Mock, AsyncMock, MagicMock
import uuid
from datetime import datetime

from core.domain.entities import (
    Analysis,
    AnalysisStatus,
    SkinMetrics,
    Condition,
    SkinType,
    User,
    Product
)


class MockFactory:
    """Factory for creating common mocks"""
    
    @staticmethod
    def create_repository_mock(interface_class, methods: Optional[List[str]] = None):
        """Create a repository mock with common methods"""
        mock = Mock(spec=interface_class)
        common_methods = ["create", "get_by_id", "update", "delete", "get_by_user", "search"]
        methods = methods or common_methods
        
        for method in methods:
            if hasattr(interface_class, method):
                setattr(mock, method, AsyncMock())
        
        return mock
    
    @staticmethod
    def create_service_mock(interface_class, methods: Optional[List[str]] = None):
        """Create a service mock with common methods"""
        mock = Mock(spec=interface_class)
        if methods:
            for method in methods:
                if hasattr(interface_class, method):
                    setattr(mock, method, AsyncMock())
        return mock
    
    @staticmethod
    def create_cache_mock():
        """Create a cache service mock"""
        cache = Mock()
        cache.get = AsyncMock(return_value=None)
        cache.set = AsyncMock(return_value=True)
        cache.delete = AsyncMock(return_value=True)
        cache.clear = AsyncMock(return_value=True)
        return cache
    
    @staticmethod
    def create_event_publisher_mock():
        """Create an event publisher mock"""
        publisher = Mock()
        publisher.publish = AsyncMock(return_value=True)
        publisher.publish_batch = AsyncMock(return_value=True)
        return publisher


class TestDataBuilder:
    """Builder for creating test data"""
    
    @staticmethod
    def build_analysis(
        analysis_id: Optional[str] = None,
        user_id: Optional[str] = None,
        status: AnalysisStatus = AnalysisStatus.COMPLETED,
        with_metrics: bool = True,
        with_conditions: bool = True,
        skin_type: SkinType = SkinType.COMBINATION
    ) -> Analysis:
        """Build an Analysis entity for testing"""
        analysis_id = analysis_id or f"analysis-{uuid.uuid4().hex[:8]}"
        user_id = user_id or f"user-{uuid.uuid4().hex[:8]}"
        
        metrics = None
        if with_metrics:
            metrics = SkinMetrics(
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
        
        conditions = []
        if with_conditions:
            conditions = [
                Condition(
                    name="acne",
                    confidence=0.65,
                    severity="moderate",
                    description="Mild acne detected"
                )
            ]
        
        return Analysis(
            id=analysis_id,
            user_id=user_id,
            metrics=metrics,
            conditions=conditions,
            skin_type=skin_type,
            status=status
        )
    
    @staticmethod
    def build_user(
        user_id: Optional[str] = None,
        email: Optional[str] = None,
        skin_type: SkinType = SkinType.COMBINATION
    ) -> User:
        """Build a User entity for testing"""
        user_id = user_id or f"user-{uuid.uuid4().hex[:8]}"
        email = email or f"test-{uuid.uuid4().hex[:8]}@example.com"
        
        return User(
            id=user_id,
            email=email,
            name="Test User",
            skin_type=skin_type,
            preferences={}
        )
    
    @staticmethod
    def build_product(
        product_id: Optional[str] = None,
        name: Optional[str] = None
    ) -> Product:
        """Build a Product entity for testing"""
        product_id = product_id or f"product-{uuid.uuid4().hex[:8]}"
        name = name or f"Test Product {uuid.uuid4().hex[:8]}"
        
        return Product(
            id=product_id,
            name=name,
            brand="Test Brand",
            category="cleanser",
            ingredients=[],
            price=29.99
        )
    
    @staticmethod
    def build_metrics(
        overall_score: float = 75.0,
        **kwargs
    ) -> SkinMetrics:
        """Build SkinMetrics for testing"""
        defaults = {
            "texture_score": 80.0,
            "hydration_score": 70.0,
            "elasticity_score": 75.0,
            "pigmentation_score": 80.0,
            "pore_size_score": 70.0,
            "wrinkles_score": 75.0,
            "redness_score": 80.0,
            "dark_spots_score": 75.0
        }
        defaults.update(kwargs)
        defaults["overall_score"] = overall_score
        
        return SkinMetrics(**defaults)


class AssertionHelpers:
    """Helper functions for assertions"""
    
    @staticmethod
    def assert_analysis_valid(analysis: Analysis):
        """Assert analysis is valid"""
        assert analysis is not None
        assert analysis.id is not None
        assert analysis.user_id is not None
        assert analysis.status in AnalysisStatus
    
    @staticmethod
    def assert_metrics_valid(metrics: SkinMetrics):
        """Assert metrics are valid"""
        assert metrics is not None
        assert 0 <= metrics.overall_score <= 100
        assert 0 <= metrics.texture_score <= 100
        assert 0 <= metrics.hydration_score <= 100
    
    @staticmethod
    def assert_condition_valid(condition: Condition):
        """Assert condition is valid"""
        assert condition is not None
        assert condition.name is not None
        assert 0 <= condition.confidence <= 1
        assert condition.severity in ["mild", "moderate", "severe"]
    
    @staticmethod
    def assert_response_structure(response_data: Dict[str, Any], required_keys: List[str]):
        """Assert response has required structure"""
        for key in required_keys:
            assert key in response_data, f"Missing required key: {key}"
    
    @staticmethod
    def assert_api_response(response, expected_status: int = 200):
        """Assert API response is valid"""
        assert response.status_code == expected_status
        assert response.headers.get("content-type") == "application/json"


class AsyncTestHelpers:
    """Helpers for async testing"""
    
    @staticmethod
    async def wait_for_condition(condition_func, timeout: float = 1.0, interval: float = 0.1):
        """Wait for a condition to become true"""
        import asyncio
        elapsed = 0.0
        while elapsed < timeout:
            if await condition_func() if asyncio.iscoroutinefunction(condition_func) else condition_func():
                return True
            await asyncio.sleep(interval)
            elapsed += interval
        return False
    
    @staticmethod
    async def run_concurrent_tasks(tasks: List, max_concurrent: int = 5):
        """Run tasks concurrently with limit"""
        import asyncio
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def run_with_semaphore(task):
            async with semaphore:
                return await task
        
        return await asyncio.gather(*[run_with_semaphore(task) for task in tasks])


# Convenience exports
create_repository_mock = MockFactory.create_repository_mock
create_service_mock = MockFactory.create_service_mock
create_cache_mock = MockFactory.create_cache_mock
create_event_publisher_mock = MockFactory.create_event_publisher_mock

build_analysis = TestDataBuilder.build_analysis
build_user = TestDataBuilder.build_user
build_product = TestDataBuilder.build_product
build_metrics = TestDataBuilder.build_metrics


class TestFixtures:
    """Helper class for creating common test fixtures"""
    
    @staticmethod
    def create_image_bytes(size: tuple = (200, 200), color: str = 'red', format: str = 'JPEG') -> bytes:
        """Create test image bytes"""
        from PIL import Image
        import io
        
        img = Image.new('RGB', size, color=color)
        img_bytes = io.BytesIO()
        img.save(img_bytes, format=format)
        img_bytes.seek(0)
        return img_bytes.read()
    
    @staticmethod
    def create_mock_request_context(user_id: str = "test-user-123", request_id: str = None):
        """Create mock request context"""
        from core.infrastructure.request_context import RequestContext
        import uuid
        
        request_id = request_id or str(uuid.uuid4())
        context = RequestContext()
        context.set_user_id(user_id)
        context.set_request_id(request_id)
        return context
    
    @staticmethod
    def create_mock_composition_root():
        """Create mock composition root"""
        from unittest.mock import Mock
        root = Mock()
        root._initialized = True
        root._use_case_cache = {}
        root.service_factory = Mock()
        root.service_factory.create = AsyncMock()
        root._database_adapter = Mock()
        return root


# Additional convenience exports
create_image_bytes = TestFixtures.create_image_bytes
create_mock_request_context = TestFixtures.create_mock_request_context
create_mock_composition_root = TestFixtures.create_mock_composition_root


# Import extended helpers
try:
    from tests.test_helpers_extended import (
        PerformanceHelpers,
        MockHelpersExtended,
        DataHelpers,
        ValidationHelpers,
        AsyncHelpersExtended,
        ErrorHelpers
    )
except ImportError:
    # Extended helpers not available
    pass

