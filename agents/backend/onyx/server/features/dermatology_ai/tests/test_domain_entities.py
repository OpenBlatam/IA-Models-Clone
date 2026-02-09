"""
Tests for Domain Entities
Tests for domain models and value objects
"""

import pytest
from datetime import datetime
from decimal import Decimal

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
from core.domain.entities.value_objects import (
    QualityScore,
    ConfidenceLevel,
    SeverityLevel
)
from core.domain.exceptions import InvalidValueError


class TestSkinMetrics:
    """Tests for SkinMetrics entity"""
    
    def test_create_valid_metrics(self):
        """Test creating valid skin metrics"""
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
        
        assert metrics.overall_score == 75.0
        assert metrics.texture_score == 80.0
        assert metrics.hydration_score == 70.0
    
    def test_metrics_score_validation(self):
        """Test that scores are within valid range"""
        # Valid scores
        metrics = SkinMetrics(
            overall_score=50.0,
            texture_score=0.0,
            hydration_score=100.0,
            elasticity_score=75.0,
            pigmentation_score=80.0,
            pore_size_score=70.0,
            wrinkles_score=75.0,
            redness_score=80.0,
            dark_spots_score=75.0
        )
        assert metrics.overall_score == 50.0
    
    def test_metrics_calculation(self):
        """Test metrics calculation methods"""
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
        
        # Test that all scores are accessible
        assert hasattr(metrics, 'overall_score')
        assert hasattr(metrics, 'texture_score')
        assert hasattr(metrics, 'hydration_score')


class TestCondition:
    """Tests for Condition entity"""
    
    def test_create_valid_condition(self):
        """Test creating valid condition"""
        condition = Condition(
            name="acne",
            confidence=0.65,
            severity="moderate",
            description="Mild acne detected"
        )
        
        assert condition.name == "acne"
        assert condition.confidence == 0.65
        assert condition.severity == "moderate"
        assert condition.description == "Mild acne detected"
    
    def test_condition_confidence_range(self):
        """Test confidence is within valid range"""
        condition = Condition(
            name="acne",
            confidence=0.0,  # Minimum
            severity="mild"
        )
        assert condition.confidence == 0.0
        
        condition = Condition(
            name="acne",
            confidence=1.0,  # Maximum
            severity="severe"
        )
        assert condition.confidence == 1.0
    
    def test_condition_without_description(self):
        """Test condition can be created without description"""
        condition = Condition(
            name="rosacea",
            confidence=0.75,
            severity="moderate"
        )
        
        assert condition.name == "rosacea"
        assert condition.description is None or condition.description == ""


class TestAnalysis:
    """Tests for Analysis entity"""
    
    def test_create_analysis(self):
        """Test creating analysis entity"""
        analysis = Analysis(
            id="test-123",
            user_id="user-123",
            status=AnalysisStatus.PROCESSING
        )
        
        assert analysis.id == "test-123"
        assert analysis.user_id == "user-123"
        assert analysis.status == AnalysisStatus.PROCESSING
        assert analysis.created_at is not None
    
    def test_analysis_with_metrics(self):
        """Test analysis with metrics"""
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
        
        analysis = Analysis(
            id="test-123",
            user_id="user-123",
            metrics=metrics,
            status=AnalysisStatus.COMPLETED
        )
        
        assert analysis.metrics == metrics
        assert analysis.metrics.overall_score == 75.0
    
    def test_analysis_with_conditions(self):
        """Test analysis with conditions"""
        conditions = [
            Condition(
                name="acne",
                confidence=0.65,
                severity="moderate"
            ),
            Condition(
                name="dryness",
                confidence=0.50,
                severity="mild"
            )
        ]
        
        analysis = Analysis(
            id="test-123",
            user_id="user-123",
            conditions=conditions,
            status=AnalysisStatus.COMPLETED
        )
        
        assert len(analysis.conditions) == 2
        assert analysis.conditions[0].name == "acne"
    
    def test_analysis_status_transitions(self):
        """Test analysis status transitions"""
        analysis = Analysis(
            id="test-123",
            user_id="user-123",
            status=AnalysisStatus.PROCESSING
        )
        
        assert analysis.status == AnalysisStatus.PROCESSING
        
        # Mark as completed
        analysis.status = AnalysisStatus.COMPLETED
        assert analysis.status == AnalysisStatus.COMPLETED
        
        # Mark as failed
        analysis.mark_failed()
        assert analysis.status == AnalysisStatus.FAILED
    
    def test_analysis_metadata(self):
        """Test analysis with metadata"""
        metadata = {
            "filename": "test.jpg",
            "enhanced": True,
            "advanced_analysis": True
        }
        
        analysis = Analysis(
            id="test-123",
            user_id="user-123",
            metadata=metadata,
            status=AnalysisStatus.PROCESSING
        )
        
        assert analysis.metadata == metadata
        assert analysis.metadata["filename"] == "test.jpg"


class TestUser:
    """Tests for User entity"""
    
    def test_create_user(self):
        """Test creating user entity"""
        user = User(
            id="user-123",
            email="test@example.com",
            name="Test User"
        )
        
        assert user.id == "user-123"
        assert user.email == "test@example.com"
        assert user.name == "Test User"
    
    def test_user_with_skin_type(self):
        """Test user with skin type"""
        user = User(
            id="user-123",
            email="test@example.com",
            name="Test User",
            skin_type=SkinType.COMBINATION
        )
        
        assert user.skin_type == SkinType.COMBINATION
    
    def test_user_preferences(self):
        """Test user with preferences"""
        preferences = {
            "notifications_enabled": True,
            "language": "es",
            "theme": "dark"
        }
        
        user = User(
            id="user-123",
            email="test@example.com",
            name="Test User",
            preferences=preferences
        )
        
        assert user.preferences == preferences
        assert user.preferences["notifications_enabled"] is True


class TestProduct:
    """Tests for Product entity"""
    
    def test_create_product(self):
        """Test creating product entity"""
        product = Product(
            id="product-123",
            name="Moisturizer",
            category="moisturizer",
            brand="Test Brand"
        )
        
        assert product.id == "product-123"
        assert product.name == "Moisturizer"
        assert product.category == "moisturizer"
        assert product.brand == "Test Brand"
    
    def test_product_with_ingredients(self):
        """Test product with ingredients"""
        ingredients = ["Glycerin", "Hyaluronic Acid", "Niacinamide"]
        
        product = Product(
            id="product-123",
            name="Serum",
            category="serum",
            brand="Test Brand",
            key_ingredients=ingredients
        )
        
        assert product.key_ingredients == ingredients
        assert len(product.key_ingredients) == 3


class TestRecommendation:
    """Tests for Recommendation entity"""
    
    def test_create_recommendation(self):
        """Test creating recommendation"""
        recommendation = Recommendation(
            name="Cleanser",
            category="cleanser",
            description="Gentle daily cleanser",
            priority=1
        )
        
        assert recommendation.name == "Cleanser"
        assert recommendation.category == "cleanser"
        assert recommendation.description == "Gentle daily cleanser"
        assert recommendation.priority == 1
    
    def test_recommendation_with_product(self):
        """Test recommendation with associated product"""
        product = Product(
            id="product-123",
            name="Moisturizer",
            category="moisturizer"
        )
        
        recommendation = Recommendation(
            name="Moisturizer",
            category="moisturizer",
            product=product,
            priority=1
        )
        
        assert recommendation.product == product
        assert recommendation.product.id == "product-123"


class TestValueObjects:
    """Tests for value objects"""
    
    def test_quality_score(self):
        """Test QualityScore value object"""
        score = QualityScore(value=75.0)
        assert score.value == 75.0
        assert 0 <= score.value <= 100
    
    def test_confidence_level(self):
        """Test ConfidenceLevel value object"""
        confidence = ConfidenceLevel(value=0.65)
        assert confidence.value == 0.65
        assert 0 <= confidence.value <= 1
    
    def test_severity_level(self):
        """Test SeverityLevel value object"""
        severity = SeverityLevel(value="moderate")
        assert severity.value == "moderate"
        assert severity.value in ["mild", "moderate", "severe"]



