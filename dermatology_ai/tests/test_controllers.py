"""
Tests for Controllers
Tests for API controllers
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from fastapi import UploadFile
import io

from api.controllers.analysis_controller import AnalysisController
from api.controllers.recommendation_controller import RecommendationController
from core.application import AnalyzeImageUseCase, GetAnalysisHistoryUseCase, GetRecommendationsUseCase
from core.domain.entities import Analysis, AnalysisStatus, SkinMetrics, SkinType
from tests.test_base import BaseAPITest
from tests.test_helpers import build_analysis, build_metrics


class TestAnalysisController(BaseAPITest):
    """Tests for AnalysisController"""
    
    @pytest.fixture
    def mock_use_cases(self):
        """Create mock use cases"""
        analyze_use_case = Mock(spec=AnalyzeImageUseCase)
        analyze_use_case.execute = AsyncMock()
        
        history_use_case = Mock(spec=GetAnalysisHistoryUseCase)
        history_use_case.execute = AsyncMock(return_value=[])
        
        return {
            "analyze": analyze_use_case,
            "history": history_use_case
        }
    
    @pytest.fixture
    def controller(self, mock_use_cases):
        """Create analysis controller"""
        return AnalysisController(
            analyze_image_use_case=mock_use_cases["analyze"],
            get_history_use_case=mock_use_cases["history"]
        )
    
    def test_controller_initialization(self, controller, mock_use_cases):
        """Test controller initialization"""
        assert controller.analyze_image_use_case == mock_use_cases["analyze"]
        assert controller.get_history_use_case == mock_use_cases["history"]
        assert controller.router is not None
    
    @pytest.mark.asyncio
    async def test_analyze_image_endpoint(self, controller, mock_use_cases):
        """Test analyze image endpoint"""
        # Use builder to create analysis
        analysis = build_analysis(
            analysis_id="test-123",
            user_id="user-123",
            status=AnalysisStatus.COMPLETED
        )
        analysis.metrics = build_metrics(overall_score=75.0)
        
        mock_use_cases["analyze"].execute = AsyncMock(return_value=analysis)
        
        # Create test image
        image_bytes = b"fake_image_data"
        file = UploadFile(
            filename="test.jpg",
            file=io.BytesIO(image_bytes)
        )
        
        # Mock current user
        with patch('api.controllers.analysis_controller.get_current_user') as mock_user:
            mock_user.return_value = {"id": "user-123"}
            
            # Call endpoint (would need FastAPI test client in real scenario)
            # For now, test the controller method directly
            result = await controller.analyze_image(
                file=file,
                current_user=mock_user.return_value
            )
            
            # Verify use case was called
            mock_use_cases["analyze"].execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_history_endpoint(self, controller, mock_use_cases):
        """Test get history endpoint"""
        analyses = [
            Analysis(
                id=f"test-{i}",
                user_id="user-123",
                status=AnalysisStatus.COMPLETED
            )
            for i in range(3)
        ]
        
        mock_use_cases["history"].execute = AsyncMock(return_value=analyses)
        
        with patch('api.controllers.analysis_controller.get_current_user') as mock_user:
            mock_user.return_value = {"id": "user-123"}
            
            result = await controller.get_history(
                current_user=mock_user.return_value,
                limit=10,
                offset=0
            )
            
            assert len(result) == 3
            mock_use_cases["history"].execute.assert_called_once()
    
    def test_serialize_conditions(self, controller):
        """Test serializing conditions"""
        from core.domain.entities import Condition
        
        conditions = [
            Condition(
                name="acne",
                confidence=0.65,
                severity="moderate",
                description="Mild acne"
            )
        ]
        
        serialized = controller._serialize_conditions(conditions)
        
        assert len(serialized) == 1
        assert serialized[0]["name"] == "acne"
        assert serialized[0]["confidence"] == 0.65


class TestRecommendationController(BaseAPITest):
    """Tests for RecommendationController"""
    
    @pytest.fixture
    def mock_use_case(self):
        """Create mock use case"""
        use_case = Mock(spec=GetRecommendationsUseCase)
        use_case.execute = AsyncMock(return_value={
            "routine": {
                "morning": [],
                "evening": []
            }
        })
        return use_case
    
    @pytest.fixture
    def controller(self, mock_use_case):
        """Create recommendation controller"""
        return RecommendationController(
            get_recommendations_use_case=mock_use_case
        )
    
    def test_controller_initialization(self, controller, mock_use_case):
        """Test controller initialization"""
        assert controller.get_recommendations_use_case == mock_use_case
        assert controller.router is not None
    
    @pytest.mark.asyncio
    async def test_get_recommendations_endpoint(self, controller, mock_use_case):
        """Test get recommendations endpoint"""
        with patch('api.controllers.recommendation_controller.get_current_user') as mock_user:
            mock_user.return_value = {"id": "user-123"}
            
            result = await controller.get_recommendations(
                analysis_id="test-123",
                current_user=mock_user.return_value,
                include_routine=True
            )
            
            assert "routine" in result
            mock_use_case.execute.assert_called_once()

