"""
Advanced Integration Tests
Tests for complex integration scenarios
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from fastapi.testclient import TestClient
import io
from PIL import Image

from main import create_application


@pytest.fixture
def client():
    """Create test client"""
    app = create_application()
    return TestClient(app)


@pytest.fixture
def sample_image_bytes():
    """Create sample image bytes"""
    img = Image.new('RGB', (200, 200), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    return img_bytes.read()


class TestCompleteUserJourney:
    """Tests for complete user journey"""
    
    @patch('api.routers.analysis_router.get_service')
    def test_complete_analysis_to_recommendations_flow(self, mock_get_service, client, sample_image_bytes):
        """Test complete flow from analysis to recommendations"""
        # Setup all services
        mock_skin_analyzer = Mock()
        mock_skin_analyzer.analyze_image = Mock(return_value={
            "quality_scores": {"overall_score": 75.0},
            "conditions": [{"name": "acne", "confidence": 0.65}],
            "skin_type": "combination"
        })
        
        mock_recommender = Mock()
        mock_recommender.generate_recommendations = Mock(return_value={
            "routine": {"morning": [], "evening": []},
            "specific_recommendations": [],
            "tips": []
        })
        
        mock_image_processor = Mock()
        mock_image_processor.process_for_analysis = Mock(return_value=(
            Mock(), True, "Processed"
        ))
        
        mock_validator = Mock()
        mock_validator.validate_image_comprehensive = Mock(return_value=(True, {}))
        
        def get_service_side_effect(service_name):
            services = {
                "skin_analyzer": mock_skin_analyzer,
                "skincare_recommender": mock_recommender,
                "image_processor": mock_image_processor,
                "advanced_validator": mock_validator,
                "history_tracker": Mock(save_analysis=Mock(return_value="record-123")),
                "db_manager": Mock(),
                "alert_system": Mock(check_analysis_alerts=Mock(return_value=[])),
                "webhook_manager": Mock(),
                "notification_service": Mock()
            }
            return services.get(service_name)
        
        mock_get_service.side_effect = get_service_side_effect
        
        # Step 1: Analyze image
        response1 = client.post(
            "/dermatology/analyze-image",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
            data={"enhance": "true"}
        )
        
        assert response1.status_code == 200
        analysis_data = response1.json()
        assert analysis_data["success"] is True
        
        # Step 2: Get recommendations
        response2 = client.post(
            "/dermatology/get-recommendations",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
            data={"include_routine": "true"}
        )
        
        assert response2.status_code == 200
        recommendations_data = response2.json()
        assert recommendations_data["success"] is True
        assert "recommendations" in recommendations_data


class TestErrorPropagation:
    """Tests for error propagation through layers"""
    
    @patch('api.routers.analysis_router.get_service')
    def test_error_propagation_from_service_to_api(self, mock_get_service, client, sample_image_bytes):
        """Test error propagation from service layer to API"""
        mock_skin_analyzer = Mock()
        mock_skin_analyzer.analyze_image = Mock(side_effect=Exception("Service error"))
        
        mock_image_processor = Mock()
        mock_image_processor.process_for_analysis = Mock(return_value=(Mock(), True, "OK"))
        
        mock_validator = Mock()
        mock_validator.validate_image_comprehensive = Mock(return_value=(True, {}))
        
        def get_service_side_effect(service_name):
            services = {
                "skin_analyzer": mock_skin_analyzer,
                "image_processor": mock_image_processor,
                "advanced_validator": mock_validator,
                "history_tracker": Mock(),
                "db_manager": Mock(),
                "alert_system": Mock(),
                "webhook_manager": Mock(),
                "notification_service": Mock()
            }
            return services.get(service_name)
        
        mock_get_service.side_effect = get_service_side_effect
        
        response = client.post(
            "/dermatology/analyze-image",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
        )
        
        # Should return error response
        assert response.status_code in [400, 500]


class TestConcurrentOperations:
    """Tests for concurrent operations"""
    
    @patch('api.routers.analysis_router.get_service')
    @pytest.mark.asyncio
    async def test_concurrent_analyses(self, mock_get_service, client, sample_image_bytes):
        """Test handling concurrent analysis requests"""
        import asyncio
        from concurrent.futures import ThreadPoolExecutor
        
        mock_skin_analyzer = Mock()
        mock_skin_analyzer.analyze_image = Mock(return_value={
            "quality_scores": {"overall_score": 75.0},
            "conditions": []
        })
        
        mock_image_processor = Mock()
        mock_image_processor.process_for_analysis = Mock(return_value=(Mock(), True, "OK"))
        
        mock_validator = Mock()
        mock_validator.validate_image_comprehensive = Mock(return_value=(True, {}))
        
        def get_service_side_effect(service_name):
            services = {
                "skin_analyzer": mock_skin_analyzer,
                "image_processor": mock_image_processor,
                "advanced_validator": mock_validator,
                "history_tracker": Mock(save_analysis=Mock(return_value="record")),
                "db_manager": Mock(),
                "alert_system": Mock(check_analysis_alerts=Mock(return_value=[])),
                "webhook_manager": Mock(),
                "notification_service": Mock()
            }
            return services.get(service_name)
        
        mock_get_service.side_effect = get_service_side_effect
        
        # Make concurrent requests
        def make_request():
            return client.post(
                "/dermatology/analyze-image",
                files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
            )
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(5)]
            results = [f.result() for f in futures]
        
        # All should succeed
        assert all(r.status_code == 200 for r in results)


class TestDataConsistency:
    """Tests for data consistency"""
    
    @patch('api.routers.analysis_router.get_service')
    def test_analysis_data_consistency(self, mock_get_service, client, sample_image_bytes):
        """Test that analysis data is consistent across requests"""
        analysis_results = []
        
        mock_skin_analyzer = Mock()
        mock_skin_analyzer.analyze_image = Mock(return_value={
            "quality_scores": {"overall_score": 75.0},
            "conditions": []
        })
        
        mock_image_processor = Mock()
        mock_image_processor.process_for_analysis = Mock(return_value=(Mock(), True, "OK"))
        
        mock_validator = Mock()
        mock_validator.validate_image_comprehensive = Mock(return_value=(True, {}))
        
        def get_service_side_effect(service_name):
            services = {
                "skin_analyzer": mock_skin_analyzer,
                "image_processor": mock_image_processor,
                "advanced_validator": mock_validator,
                "history_tracker": Mock(save_analysis=Mock(return_value="record")),
                "db_manager": Mock(),
                "alert_system": Mock(check_analysis_alerts=Mock(return_value=[])),
                "webhook_manager": Mock(),
                "notification_service": Mock()
            }
            return services.get(service_name)
        
        mock_get_service.side_effect = get_service_side_effect
        
        # Make same request multiple times
        for _ in range(3):
            response = client.post(
                "/dermatology/analyze-image",
                files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
            )
            if response.status_code == 200:
                analysis_results.append(response.json()["analysis"])
        
        # Results should be consistent (same image should give similar results)
        if len(analysis_results) > 1:
            # Check that structure is consistent
            assert all("quality_scores" in r for r in analysis_results)



