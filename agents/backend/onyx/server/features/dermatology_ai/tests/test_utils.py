"""
Test Utilities
Helper functions and utilities for testing
"""

import pytest
from typing import Dict, Any, List
from datetime import datetime
import uuid

from core.domain.entities import (
    Analysis,
    AnalysisStatus,
    SkinMetrics,
    Condition,
    SkinType,
    User,
    Product
)


class TestDataFactory:
    """Factory for creating test data"""
    
    @staticmethod
    def create_analysis(
        user_id: str = None,
        status: AnalysisStatus = AnalysisStatus.COMPLETED,
        with_metrics: bool = True,
        with_conditions: bool = True
    ) -> Analysis:
        """Create a test analysis"""
        analysis_id = str(uuid.uuid4())
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
            skin_type=SkinType.COMBINATION,
            status=status
        )
    
    @staticmethod
    def create_user(
        email: str = None,
        skin_type: SkinType = SkinType.COMBINATION
    ) -> User:
        """Create a test user"""
        user_id = str(uuid.uuid4())
        email = email or f"test-{uuid.uuid4().hex[:8]}@example.com"
        
        return User(
            id=user_id,
            email=email,
            name="Test User",
            skin_type=skin_type,
            preferences={}
        )
    
    @staticmethod
    def create_product(
        category: str = "moisturizer",
        name: str = None
    ) -> Product:
        """Create a test product"""
        product_id = str(uuid.uuid4())
        name = name or f"Test {category.title()}"
        
        return Product(
            id=product_id,
            name=name,
            category=category,
            description=f"Test {category} product",
            ingredients=["Glycerin", "Hyaluronic Acid"]
        )
    
    @staticmethod
    def create_metrics(
        overall_score: float = 75.0,
        **kwargs
    ) -> SkinMetrics:
        """Create test skin metrics"""
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


class TestAssertions:
    """Custom assertions for tests"""
    
    @staticmethod
    def assert_analysis_valid(analysis: Analysis):
        """Assert that analysis is valid"""
        assert analysis is not None
        assert analysis.id is not None
        assert analysis.user_id is not None
        assert analysis.status is not None
        assert isinstance(analysis.created_at, datetime)
    
    @staticmethod
    def assert_metrics_valid(metrics: SkinMetrics):
        """Assert that metrics are valid"""
        assert metrics is not None
        assert 0 <= metrics.overall_score <= 100
        assert 0 <= metrics.texture_score <= 100
        assert 0 <= metrics.hydration_score <= 100
    
    @staticmethod
    def assert_condition_valid(condition: Condition):
        """Assert that condition is valid"""
        assert condition is not None
        assert condition.name is not None
        assert 0 <= condition.confidence <= 1
        assert condition.severity in ["mild", "moderate", "severe"]


class TestHelpers:
    """Helper functions for tests"""
    
    @staticmethod
    def create_image_bytes(size: int = 1024) -> bytes:
        """Create test image bytes"""
        from PIL import Image
        import io
        
        img = Image.new('RGB', (200, 200), color='red')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        return img_bytes.read()
    
    @staticmethod
    def create_multiple_analyses(count: int, user_id: str = None) -> List[Analysis]:
        """Create multiple test analyses"""
        return [
            TestDataFactory.create_analysis(user_id=user_id)
            for _ in range(count)
        ]
    
    @staticmethod
    def assert_dict_contains(dict_obj: Dict[str, Any], required_keys: List[str]):
        """Assert that dict contains required keys"""
        for key in required_keys:
            assert key in dict_obj, f"Missing required key: {key}"
    
    @staticmethod
    def assert_response_success(response_data: Dict[str, Any]):
        """Assert that API response indicates success"""
        assert "success" in response_data
        assert response_data["success"] is True
    
    @staticmethod
    def assert_response_error(response_data: Dict[str, Any], expected_status: int = None):
        """Assert that API response indicates error"""
        assert "error" in response_data or "message" in response_data
        if expected_status:
            assert "status_code" in response_data or response_data.get("status") == expected_status


# Pytest fixtures using the factory
@pytest.fixture
def test_data_factory():
    """Provide test data factory"""
    return TestDataFactory


@pytest.fixture
def test_assertions():
    """Provide test assertions"""
    return TestAssertions


@pytest.fixture
def test_helpers():
    """Provide test helpers"""
    return TestHelpers


# Convenience fixtures
@pytest.fixture
def sample_analysis_factory(test_data_factory):
    """Create sample analysis using factory"""
    return test_data_factory.create_analysis()


@pytest.fixture
def sample_user_factory(test_data_factory):
    """Create sample user using factory"""
    return test_data_factory.create_user()


@pytest.fixture
def sample_product_factory(test_data_factory):
    """Create sample product using factory"""
    return test_data_factory.create_product()


@pytest.fixture
def sample_metrics_factory(test_data_factory):
    """Create sample metrics using factory"""
    return test_data_factory.create_metrics()



