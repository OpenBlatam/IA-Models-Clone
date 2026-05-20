"""
Tests for Domain Services
Tests for analysis and recommendation services
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
import numpy as np
from PIL import Image
import io

from core.domain.services.analysis_service import AnalysisService
from core.domain.services.recommendation_service import RecommendationService
from core.domain.entities import (
    Analysis,
    AnalysisStatus,
    SkinMetrics,
    Condition,
    SkinType,
    Recommendation,
    Product
)
from core.domain.interfaces import IImageProcessor
from tests.test_base import BaseServiceTest
from tests.test_helpers import build_analysis, build_metrics, create_image_bytes


class TestAnalysisService(BaseServiceTest):
    """Tests for AnalysisService"""
    
    @pytest.fixture
    def mock_image_processor(self):
        """Mock image processor"""
        processor = Mock(spec=IImageProcessor)
        processor.process = AsyncMock(return_value={
            "metrics": {
                "overall_score": 75.0,
                "texture_score": 80.0,
                "hydration_score": 70.0
            }
        })
        return processor
    
    @pytest.mark.asyncio
    async def test_analyze_image_success(self, mock_image_processor):
        """Test successful image analysis"""
        service = AnalysisService(image_processor=mock_image_processor)
        
        # Use helper to create image bytes
        image_data = create_image_bytes()
        
        result = await service.analyze_image(
            user_id="user-123",
            image_data=image_data,
            metadata={}
        )
        
        self.assert_service_result_valid(result, Analysis)
        assert result.metrics is not None
        assert result.status == AnalysisStatus.COMPLETED
        mock_image_processor.process.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_analyze_image_with_conditions(self, mock_image_processor):
        """Test image analysis that detects conditions"""
        # Mock processor to return conditions
        mock_image_processor.process = AsyncMock(return_value={
            "metrics": {
                "overall_score": 65.0,
                "texture_score": 60.0,
                "hydration_score": 70.0
            },
            "conditions": [
                {
                    "name": "acne",
                    "confidence": 0.65,
                    "severity": "moderate"
                }
            ]
        })
        
        service = AnalysisService(image_processor=mock_image_processor)
        
        img = Image.new('RGB', (200, 200), color='red')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        image_data = img_bytes.read()
        
        result = await service.analyze_image(
            user_id="user-123",
            image_data=image_data,
            metadata={}
        )
        
        assert result is not None
        assert len(result.conditions) > 0
        assert result.conditions[0].name == "acne"
    
    @pytest.mark.asyncio
    async def test_analyze_image_determines_skin_type(self, mock_image_processor):
        """Test that analysis determines skin type"""
        service = AnalysisService(image_processor=mock_image_processor)
        
        img = Image.new('RGB', (200, 200), color='red')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        image_data = img_bytes.read()
        
        result = await service.analyze_image(
            user_id="user-123",
            image_data=image_data,
            metadata={}
        )
        
        assert result.skin_type is not None
        assert isinstance(result.skin_type, SkinType)


class TestRecommendationService(BaseServiceTest):
    """Tests for RecommendationService"""
    
    @pytest.fixture
    def sample_analysis(self):
        """Create sample analysis for recommendations"""
        return Analysis(
            id="test-123",
            user_id="user-123",
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
            conditions=[
                Condition(
                    name="acne",
                    confidence=0.65,
                    severity="moderate"
                )
            ],
            skin_type=SkinType.COMBINATION,
            status=AnalysisStatus.COMPLETED
        )
    
    @pytest.mark.asyncio
    async def test_generate_recommendations_success(self, sample_analysis):
        """Test successful recommendations generation"""
        service = RecommendationService()
        
        result = await service.generate_recommendations(
            analysis=sample_analysis,
            include_routine=True
        )
        
        assert result is not None
        assert "routine" in result
        assert "morning" in result["routine"]
        assert "evening" in result["routine"]
        assert "weekly" in result["routine"]
        assert "specific_recommendations" in result
        assert "tips" in result
    
    @pytest.mark.asyncio
    async def test_generate_recommendations_for_acne(self, sample_analysis):
        """Test recommendations for acne condition"""
        service = RecommendationService()
        
        result = await service.generate_recommendations(
            analysis=sample_analysis,
            include_routine=True
        )
        
        # Should include acne-specific recommendations
        assert result is not None
        # Check that recommendations are generated
        assert len(result["routine"]["morning"]) > 0 or len(result["specific_recommendations"]) > 0
    
    @pytest.mark.asyncio
    async def test_generate_recommendations_for_dry_skin(self):
        """Test recommendations for dry skin"""
        analysis = Analysis(
            id="test-123",
            user_id="user-123",
            metrics=SkinMetrics(
                overall_score=70.0,
                texture_score=65.0,
                hydration_score=50.0,  # Low hydration
                elasticity_score=70.0,
                pigmentation_score=75.0,
                pore_size_score=80.0,
                wrinkles_score=70.0,
                redness_score=75.0,
                dark_spots_score=75.0
            ),
            conditions=[
                Condition(
                    name="dryness",
                    confidence=0.80,
                    severity="moderate"
                )
            ],
            skin_type=SkinType.DRY,
            status=AnalysisStatus.COMPLETED
        )
        
        service = RecommendationService()
        
        result = await service.generate_recommendations(
            analysis=analysis,
            include_routine=True
        )
        
        assert result is not None
        # Should prioritize hydration
        assert len(result["routine"]["morning"]) > 0
    
    @pytest.mark.asyncio
    async def test_generate_recommendations_without_routine(self, sample_analysis):
        """Test recommendations without routine"""
        service = RecommendationService()
        
        result = await service.generate_recommendations(
            analysis=sample_analysis,
            include_routine=False
        )
        
        assert result is not None
        # Routine might be empty or not included
        assert "specific_recommendations" in result
        assert "tips" in result
    
    @pytest.mark.asyncio
    async def test_recommendations_prioritization(self, sample_analysis):
        """Test that recommendations are properly prioritized"""
        service = RecommendationService()
        
        result = await service.generate_recommendations(
            analysis=sample_analysis,
            include_routine=True
        )
        
        assert result is not None
        
        # Check that recommendations have priority
        if result["routine"]["morning"]:
            recommendations = result["routine"]["morning"]
            # Should be sorted by priority or have priority field
            assert all(hasattr(r, 'priority') or isinstance(r, dict) for r in recommendations)


class TestServiceIntegration:
    """Integration tests for services"""
    
    @pytest.mark.asyncio
    async def test_analysis_to_recommendations_flow(self):
        """Test complete flow from analysis to recommendations"""
        # Create mock image processor
        mock_processor = Mock(spec=IImageProcessor)
        mock_processor.process = AsyncMock(return_value={
            "metrics": {
                "overall_score": 75.0,
                "texture_score": 80.0,
                "hydration_score": 70.0
            },
            "conditions": [
                {
                    "name": "acne",
                    "confidence": 0.65,
                    "severity": "moderate"
                }
            ]
        })
        
        # Run analysis
        analysis_service = AnalysisService(image_processor=mock_processor)
        
        img = Image.new('RGB', (200, 200), color='red')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        image_data = img_bytes.read()
        
        analysis = await analysis_service.analyze_image(
            user_id="user-123",
            image_data=image_data,
            metadata={}
        )
        
        # Generate recommendations
        recommendation_service = RecommendationService()
        recommendations = await recommendation_service.generate_recommendations(
            analysis=analysis,
            include_routine=True
        )
        
        # Verify complete flow
        assert analysis is not None
        assert analysis.status == AnalysisStatus.COMPLETED
        assert recommendations is not None
        assert "routine" in recommendations

