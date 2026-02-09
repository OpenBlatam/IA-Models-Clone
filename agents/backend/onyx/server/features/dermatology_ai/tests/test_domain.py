"""
Domain Layer Tests
Pure unit tests for domain entities and logic
"""

import pytest
from datetime import datetime

from core.domain.entities import (
    Analysis,
    User,
    SkinMetrics,
    Condition,
    AnalysisStatus,
    SkinType,
)


class TestAnalysis:
    """Tests for Analysis entity"""
    
    def test_create_analysis(self):
        """Test creating analysis"""
        analysis = Analysis(
            id="test-1",
            user_id="user-1",
            status=AnalysisStatus.PENDING
        )
        
        assert analysis.id == "test-1"
        assert analysis.user_id == "user-1"
        assert analysis.status == AnalysisStatus.PENDING
        assert not analysis.is_completed()
    
    def test_mark_completed(self):
        """Test marking analysis as completed"""
        analysis = Analysis(
            id="test-1",
            user_id="user-1",
            status=AnalysisStatus.PENDING
        )
        
        metrics = SkinMetrics(
            overall_score=80.0,
            texture_score=85.0,
            hydration_score=75.0,
            elasticity_score=80.0,
            pigmentation_score=85.0,
            pore_size_score=75.0,
            wrinkles_score=80.0,
            redness_score=85.0,
            dark_spots_score=80.0,
        )
        
        conditions = [
            Condition(
                name="acne",
                confidence=0.65,
                severity="moderate"
            )
        ]
        
        analysis.mark_completed(metrics, conditions)
        
        assert analysis.status == AnalysisStatus.COMPLETED
        assert analysis.metrics == metrics
        assert len(analysis.conditions) == 1
        assert analysis.is_completed()
        assert analysis.completed_at is not None
    
    def test_mark_failed(self):
        """Test marking analysis as failed"""
        analysis = Analysis(
            id="test-1",
            user_id="user-1",
            status=AnalysisStatus.PROCESSING
        )
        
        analysis.mark_failed()
        
        assert analysis.status == AnalysisStatus.FAILED
        assert analysis.completed_at is not None


class TestSkinMetrics:
    """Tests for SkinMetrics value object"""
    
    def test_to_dict(self):
        """Test converting metrics to dictionary"""
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
        
        result = metrics.to_dict()
        
        assert result["overall_score"] == 75.0
        assert result["texture_score"] == 80.0
        assert len(result) == 9


class TestUser:
    """Tests for User entity"""
    
    def test_update_preferences(self):
        """Test updating user preferences"""
        user = User(
            id="user-1",
            email="test@example.com",
            preferences={"theme": "dark"}
        )
        
        original_updated_at = user.updated_at
        
        user.update_preferences({"language": "es", "notifications": True})
        
        assert user.preferences["theme"] == "dark"
        assert user.preferences["language"] == "es"
        assert user.preferences["notifications"] is True
        assert user.updated_at > original_updated_at















