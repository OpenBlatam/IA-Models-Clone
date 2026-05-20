"""
Test Fixtures Factory
Factory for creating common test fixtures and test data
"""

from typing import Any, Dict, List, Optional, Callable
from unittest.mock import Mock, AsyncMock
import uuid
from datetime import datetime, timedelta

from core.domain.entities import (
    Analysis,
    AnalysisStatus,
    SkinMetrics,
    Condition,
    SkinType,
    User,
    Product,
    Recommendation
)
from tests.test_helpers import build_analysis, build_user, build_product, build_metrics


class FixtureFactory:
    """Factory for creating test fixtures"""
    
    @staticmethod
    def create_analysis_fixture(
        user_id: Optional[str] = None,
        status: AnalysisStatus = AnalysisStatus.COMPLETED,
        with_metrics: bool = True,
        with_conditions: bool = True
    ) -> Analysis:
        """Create analysis fixture"""
        return build_analysis(
            user_id=user_id,
            status=status,
            with_metrics=with_metrics,
            with_conditions=with_conditions
        )
    
    @staticmethod
    def create_user_fixture(
        email: Optional[str] = None,
        skin_type: SkinType = SkinType.COMBINATION
    ) -> User:
        """Create user fixture"""
        return build_user(email=email, skin_type=skin_type)
    
    @staticmethod
    def create_product_fixture(
        name: Optional[str] = None,
        category: str = "cleanser"
    ) -> Product:
        """Create product fixture"""
        product = build_product(name=name)
        product.category = category
        return product
    
    @staticmethod
    def create_analysis_list(count: int, user_id: Optional[str] = None) -> List[Analysis]:
        """Create list of analyses"""
        return [
            FixtureFactory.create_analysis_fixture(
                user_id=user_id or f"user-{i}",
                status=AnalysisStatus.COMPLETED
            )
            for i in range(count)
        ]
    
    @staticmethod
    def create_repository_mock_fixture(interface_class, methods: Optional[List[str]] = None):
        """Create repository mock fixture"""
        from tests.test_helpers import create_repository_mock
        return create_repository_mock(interface_class, methods)
    
    @staticmethod
    def create_service_mock_fixture(interface_class, methods: Optional[List[str]] = None):
        """Create service mock fixture"""
        from tests.test_helpers import create_service_mock
        return create_service_mock(interface_class, methods)
    
    @staticmethod
    def create_cache_mock_fixture():
        """Create cache mock fixture"""
        from tests.test_helpers import create_cache_mock
        return create_cache_mock()
    
    @staticmethod
    def create_event_publisher_mock_fixture():
        """Create event publisher mock fixture"""
        from tests.test_helpers import create_event_publisher_mock
        return create_event_publisher_mock()


class TestScenarioBuilder:
    """Builder for creating test scenarios"""
    
    def __init__(self):
        self.scenario = {
            "user": None,
            "analyses": [],
            "products": [],
            "mocks": {},
            "expected_results": {}
        }
    
    def with_user(self, user: User) -> 'TestScenarioBuilder':
        """Add user to scenario"""
        self.scenario["user"] = user
        return self
    
    def with_analysis(self, analysis: Analysis) -> 'TestScenarioBuilder':
        """Add analysis to scenario"""
        self.scenario["analyses"].append(analysis)
        return self
    
    def with_analyses(self, count: int, user_id: Optional[str] = None) -> 'TestScenarioBuilder':
        """Add multiple analyses to scenario"""
        analyses = FixtureFactory.create_analysis_list(count, user_id)
        self.scenario["analyses"].extend(analyses)
        return self
    
    def with_product(self, product: Product) -> 'TestScenarioBuilder':
        """Add product to scenario"""
        self.scenario["products"].append(product)
        return self
    
    def with_mock(self, name: str, mock: Any) -> 'TestScenarioBuilder':
        """Add mock to scenario"""
        self.scenario["mocks"][name] = mock
        return self
    
    def with_expected_result(self, key: str, value: Any) -> 'TestScenarioBuilder':
        """Add expected result to scenario"""
        self.scenario["expected_results"][key] = value
        return self
    
    def build(self) -> Dict[str, Any]:
        """Build the test scenario"""
        return self.scenario.copy()


class MockBuilder:
    """Builder for creating complex mocks"""
    
    def __init__(self, spec=None):
        self.mock = Mock(spec=spec)
        self.async_methods = []
        self.sync_methods = []
        self.return_values = {}
        self.side_effects = {}
    
    def with_async_method(self, method_name: str, return_value: Any = None, side_effect: Any = None) -> 'MockBuilder':
        """Add async method to mock"""
        self.async_methods.append(method_name)
        if return_value is not None:
            self.return_values[method_name] = return_value
        if side_effect is not None:
            self.side_effects[method_name] = side_effect
        return self
    
    def with_sync_method(self, method_name: str, return_value: Any = None, side_effect: Any = None) -> 'MockBuilder':
        """Add sync method to mock"""
        self.sync_methods.append(method_name)
        if return_value is not None:
            self.return_values[method_name] = return_value
        if side_effect is not None:
            self.side_effects[method_name] = side_effect
        return self
    
    def build(self) -> Mock:
        """Build the mock"""
        for method_name in self.async_methods:
            mock_method = AsyncMock()
            if method_name in self.return_values:
                mock_method.return_value = self.return_values[method_name]
            if method_name in self.side_effects:
                mock_method.side_effect = self.side_effects[method_name]
            setattr(self.mock, method_name, mock_method)
        
        for method_name in self.sync_methods:
            mock_method = Mock()
            if method_name in self.return_values:
                mock_method.return_value = self.return_values[method_name]
            if method_name in self.side_effects:
                mock_method.side_effect = self.side_effects[method_name]
            setattr(self.mock, method_name, mock_method)
        
        return self.mock


class TestDataGenerator:
    """Generator for test data"""
    
    @staticmethod
    def generate_analyses(
        count: int,
        user_id: Optional[str] = None,
        date_range: Optional[tuple[datetime, datetime]] = None
    ) -> List[Analysis]:
        """Generate multiple analyses with optional date range"""
        analyses = []
        start_date = date_range[0] if date_range else datetime.utcnow() - timedelta(days=count)
        end_date = date_range[1] if date_range else datetime.utcnow()
        
        for i in range(count):
            analysis = build_analysis(
                user_id=user_id or f"user-{i}",
                status=AnalysisStatus.COMPLETED
            )
            # Add created_at if entity supports it
            analyses.append(analysis)
        
        return analyses
    
    @staticmethod
    def generate_users(count: int, domain: str = "example.com") -> List[User]:
        """Generate multiple users"""
        return [
            build_user(email=f"user{i}@{domain}")
            for i in range(count)
        ]
    
    @staticmethod
    def generate_products(count: int, category: str = "cleanser") -> List[Product]:
        """Generate multiple products"""
        products = []
        for i in range(count):
            product = build_product(name=f"Product {i}")
            product.category = category
            products.append(product)
        return products
    
    @staticmethod
    def generate_metrics_range(
        min_score: float = 0.0,
        max_score: float = 100.0,
        count: int = 10
    ) -> List[SkinMetrics]:
        """Generate metrics with scores in range"""
        step = (max_score - min_score) / count if count > 1 else 0
        return [
            build_metrics(overall_score=min_score + (step * i))
            for i in range(count)
        ]


# Convenience exports
create_analysis_fixture = FixtureFactory.create_analysis_fixture
create_user_fixture = FixtureFactory.create_user_fixture
create_product_fixture = FixtureFactory.create_product_fixture
create_analysis_list = FixtureFactory.create_analysis_list

