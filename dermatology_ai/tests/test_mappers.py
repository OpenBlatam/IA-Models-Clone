"""
Tests for Mappers
Tests for entity-to-dict and dict-to-entity mappers
"""

import pytest
from datetime import datetime
from typing import Dict, Any

from core.infrastructure.mappers.analysis_mapper import AnalysisMapper
from core.infrastructure.mappers.user_mapper import UserMapper
from core.infrastructure.mappers.product_mapper import ProductMapper
from core.domain.entities import (
    Analysis,
    AnalysisStatus,
    SkinMetrics,
    Condition,
    SkinType,
    User,
    Product
)


class TestAnalysisMapper:
    """Tests for AnalysisMapper"""
    
    def test_to_dict_complete_analysis(self):
        """Test mapping complete analysis to dict"""
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
        
        conditions = [
            Condition(
                name="acne",
                confidence=0.65,
                severity="moderate",
                description="Mild acne"
            )
        ]
        
        analysis = Analysis(
            id="test-123",
            user_id="user-123",
            image_url="https://example.com/image.jpg",
            metrics=metrics,
            conditions=conditions,
            skin_type=SkinType.COMBINATION,
            status=AnalysisStatus.COMPLETED,
            created_at=datetime(2024, 1, 1, 12, 0, 0),
            completed_at=datetime(2024, 1, 1, 12, 5, 0),
            metadata={"enhanced": True}
        )
        
        result = AnalysisMapper.to_dict(analysis)
        
        assert result["id"] == "test-123"
        assert result["user_id"] == "user-123"
        assert result["image_url"] == "https://example.com/image.jpg"
        assert result["metrics"] is not None
        assert result["metrics"]["overall_score"] == 75.0
        assert len(result["conditions"]) == 1
        assert result["conditions"][0]["name"] == "acne"
        assert result["skin_type"] == "combination"
        assert result["status"] == "completed"
        assert result["metadata"]["enhanced"] is True
    
    def test_to_dict_minimal_analysis(self):
        """Test mapping minimal analysis to dict"""
        analysis = Analysis(
            id="test-123",
            user_id="user-123",
            status=AnalysisStatus.PROCESSING
        )
        
        result = AnalysisMapper.to_dict(analysis)
        
        assert result["id"] == "test-123"
        assert result["user_id"] == "user-123"
        assert result["metrics"] is None
        assert result["conditions"] == []
        assert result["skin_type"] is None
        assert result["status"] == "processing"
        assert result["completed_at"] is None
    
    def test_to_entity_complete_analysis(self):
        """Test mapping dict to complete analysis entity"""
        data: Dict[str, Any] = {
            "id": "test-123",
            "user_id": "user-123",
            "image_url": "https://example.com/image.jpg",
            "metrics": {
                "overall_score": 75.0,
                "texture_score": 80.0,
                "hydration_score": 70.0,
                "elasticity_score": 75.0,
                "pigmentation_score": 80.0,
                "pore_size_score": 70.0,
                "wrinkles_score": 75.0,
                "redness_score": 80.0,
                "dark_spots_score": 75.0
            },
            "conditions": [
                {
                    "name": "acne",
                    "confidence": 0.65,
                    "severity": "moderate",
                    "description": "Mild acne"
                }
            ],
            "skin_type": "combination",
            "status": "completed",
            "created_at": "2024-01-01T12:00:00",
            "completed_at": "2024-01-01T12:05:00",
            "metadata": {"enhanced": True}
        }
        
        result = AnalysisMapper.to_entity(data)
        
        assert isinstance(result, Analysis)
        assert result.id == "test-123"
        assert result.user_id == "user-123"
        assert result.image_url == "https://example.com/image.jpg"
        assert result.metrics is not None
        assert result.metrics.overall_score == 75.0
        assert len(result.conditions) == 1
        assert result.conditions[0].name == "acne"
        assert result.skin_type == SkinType.COMBINATION
        assert result.status == AnalysisStatus.COMPLETED
        assert result.metadata["enhanced"] is True
    
    def test_to_entity_minimal_analysis(self):
        """Test mapping minimal dict to analysis entity"""
        data: Dict[str, Any] = {
            "id": "test-123",
            "user_id": "user-123",
            "status": "processing",
            "created_at": "2024-01-01T12:00:00"
        }
        
        result = AnalysisMapper.to_entity(data)
        
        assert isinstance(result, Analysis)
        assert result.id == "test-123"
        assert result.metrics is None
        assert result.conditions == []
        assert result.skin_type is None
        assert result.status == AnalysisStatus.PROCESSING
    
    def test_to_update_dict(self):
        """Test mapping analysis to update dict"""
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
            skin_type=SkinType.COMBINATION,
            status=AnalysisStatus.COMPLETED,
            completed_at=datetime(2024, 1, 1, 12, 5, 0)
        )
        
        result = AnalysisMapper.to_update_dict(analysis)
        
        assert "metrics" in result
        assert result["metrics"]["overall_score"] == 75.0
        assert result["skin_type"] == "combination"
        assert result["status"] == "completed"
        assert result["completed_at"] is not None
        # Should not include id or user_id
        assert "id" not in result
        assert "user_id" not in result


class TestUserMapper:
    """Tests for UserMapper"""
    
    def test_to_dict_complete_user(self):
        """Test mapping complete user to dict"""
        user = User(
            id="user-123",
            email="test@example.com",
            name="Test User",
            skin_type=SkinType.COMBINATION,
            preferences={"theme": "dark"}
        )
        
        result = UserMapper.to_dict(user)
        
        assert result["id"] == "user-123"
        assert result["email"] == "test@example.com"
        assert result["name"] == "Test User"
        assert result["skin_type"] == "combination"
        assert result["preferences"]["theme"] == "dark"
    
    def test_to_entity_complete_user(self):
        """Test mapping dict to complete user entity"""
        data: Dict[str, Any] = {
            "id": "user-123",
            "email": "test@example.com",
            "name": "Test User",
            "skin_type": "combination",
            "preferences": {"theme": "dark"}
        }
        
        result = UserMapper.to_entity(data)
        
        assert isinstance(result, User)
        assert result.id == "user-123"
        assert result.email == "test@example.com"
        assert result.skin_type == SkinType.COMBINATION


class TestProductMapper:
    """Tests for ProductMapper"""
    
    def test_to_dict_complete_product(self):
        """Test mapping complete product to dict"""
        product = Product(
            id="product-123",
            name="Moisturizer",
            category="moisturizer",
            description="A great moisturizer",
            ingredients=["Glycerin", "Hyaluronic Acid"]
        )
        
        result = ProductMapper.to_dict(product)
        
        assert result["id"] == "product-123"
        assert result["name"] == "Moisturizer"
        assert result["category"] == "moisturizer"
        assert len(result.get("ingredients", [])) == 2
        assert result["description"] == "A great moisturizer"
    
    def test_to_entity_complete_product(self):
        """Test mapping dict to complete product entity"""
        data: Dict[str, Any] = {
            "id": "product-123",
            "name": "Moisturizer",
            "category": "moisturizer",
            "ingredients": ["Glycerin", "Hyaluronic Acid"],
            "description": "A great moisturizer"
        }
        
        result = ProductMapper.to_entity(data)
        
        assert isinstance(result, Product)
        assert result.id == "product-123"
        assert result.name == "Moisturizer"
        assert len(result.ingredients) == 2

