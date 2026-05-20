"""
Integration Tests
End-to-end tests for the application
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
import numpy as np
from PIL import Image

from main import create_application
from tests.test_base import BaseIntegrationTest
from tests.test_helpers import create_image_bytes, build_analysis


class TestImageAnalysisFlow(BaseIntegrationTest):
    """Integration tests for image analysis flow"""
    
    @patch('api.routers.analysis_router.get_service')
    def test_complete_image_analysis_flow(self, mock_get_service, client, sample_image_bytes):
        """Test complete image analysis flow from API to response"""
        # Setup all required services
        mock_skin_analyzer = Mock()
        mock_skin_analyzer.use_advanced = True
        mock_skin_analyzer.analyze_image = Mock(return_value={
            "quality_scores": {
                "overall_score": 75.5,
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
                    "description": "Mild acne detected"
                }
            ],
            "skin_type": "combination",
            "recommendations_priority": ["hydration", "texture"]
        })
        
        mock_image_processor = Mock()
        mock_image_processor.process_for_analysis = Mock(return_value=(
            np.array(Image.new('RGB', (200, 200))),
            True,
            "Image processed successfully"
        ))
        mock_image_processor.enhance_for_analysis = Mock(return_value=np.array(Image.new('RGB', (200, 200))))
        
        mock_validator = Mock()
        mock_validator.validate_image_comprehensive = Mock(return_value=(True, {}))
        
        mock_history = Mock()
        mock_history.save_analysis = Mock(return_value="record-123")
        
        mock_db = Mock()
        mock_db.save_analysis = Mock()
        
        mock_alert = Mock()
        mock_alert.check_analysis_alerts = Mock(return_value=[])
        
        def get_service_side_effect(service_name):
            services = {
                "skin_analyzer": mock_skin_analyzer,
                "image_processor": mock_image_processor,
                "history_tracker": mock_history,
                "db_manager": mock_db,
                "alert_system": mock_alert,
                "advanced_validator": mock_validator,
                "webhook_manager": Mock(),
                "notification_service": Mock()
            }
            return services.get(service_name)
        
        mock_get_service.side_effect = get_service_side_effect
        
        # Make API request
        response = client.post(
            "/dermatology/analyze-image",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
            data={
                "enhance": "true",
                "use_advanced": "true",
                "use_cache": "true"
            }
        )
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "analysis" in data
        assert "quality_scores" in data["analysis"]
        assert "conditions" in data["analysis"]
        assert "processing_time" in data
        
        # Verify services were called
        mock_validator.validate_image_comprehensive.assert_called_once()
        mock_image_processor.process_for_analysis.assert_called_once()
        mock_skin_analyzer.analyze_image.assert_called_once()
        mock_history.save_analysis.assert_called_once()


class TestRecommendationsFlow:
    """Integration tests for recommendations flow"""
    
    @patch('api.routers.recommendations_router.get_service')
    def test_complete_recommendations_flow(self, mock_get_service, client, sample_image_bytes):
        """Test complete recommendations flow"""
        # Setup services
        mock_skin_analyzer = Mock()
        mock_skin_analyzer.analyze_image = Mock(return_value={
            "quality_scores": {
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
            ],
            "skin_type": "combination"
        })
        
        mock_recommender = Mock()
        mock_recommender.generate_recommendations = Mock(return_value={
            "routine": {
                "morning": [
                    {
                        "name": "Cleanser",
                        "category": "cleanser",
                        "description": "Gentle daily cleanser",
                        "priority": 1
                    }
                ],
                "evening": [],
                "weekly": []
            },
            "specific_recommendations": [
                {
                    "condition": "acne",
                    "recommendation": "Use salicylic acid cleanser"
                }
            ],
            "tips": [
                "Keep skin hydrated",
                "Avoid harsh products"
            ]
        })
        
        mock_image_processor = Mock()
        mock_image_processor.process_for_analysis = Mock(return_value=(
            np.array(Image.new('RGB', (200, 200))),
            True,
            "Image processed"
        ))
        
        mock_validator = Mock()
        mock_validator.validate_image_comprehensive = Mock(return_value=(True, {}))
        
        def get_service_side_effect(service_name):
            services = {
                "skincare_recommender": mock_recommender,
                "skin_analyzer": mock_skin_analyzer,
                "image_processor": mock_image_processor,
                "advanced_validator": mock_validator
            }
            return services.get(service_name)
        
        mock_get_service.side_effect = get_service_side_effect
        
        # Make API request
        response = client.post(
            "/dermatology/get-recommendations",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
            data={"include_routine": "true"}
        )
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "recommendations" in data
        assert "routine" in data["recommendations"]
        assert "specific_recommendations" in data["recommendations"]
        assert "tips" in data["recommendations"]


class TestErrorHandling:
    """Integration tests for error handling"""
    
    @patch('api.routers.analysis_router.get_service')
    def test_invalid_image_error_handling(self, mock_get_service, client):
        """Test error handling for invalid image"""
        mock_validator = Mock()
        mock_validator.validate_image_comprehensive = Mock(return_value=(
            False,
            {"errors": ["Invalid image format"]}
        ))
        
        def get_service_side_effect(service_name):
            if service_name == "advanced_validator":
                return mock_validator
            return Mock()
        
        mock_get_service.side_effect = get_service_side_effect
        
        response = client.post(
            "/dermatology/analyze-image",
            files={"file": ("test.txt", b"not an image", "text/plain")},
            data={"enhance": "true"}
        )
        
        assert response.status_code == 400
    
    @patch('api.routers.analysis_router.get_service')
    def test_processing_error_handling(self, mock_get_service, client, sample_image_bytes):
        """Test error handling for processing errors"""
        mock_skin_analyzer = Mock()
        mock_skin_analyzer.analyze_image = Mock(side_effect=Exception("Processing failed"))
        
        mock_image_processor = Mock()
        mock_image_processor.process_for_analysis = Mock(return_value=(
            np.array(Image.new('RGB', (200, 200))),
            True,
            "Image processed"
        ))
        
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
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
            data={"enhance": "true"}
        )
        
        assert response.status_code == 500


class TestHealthChecks:
    """Integration tests for health checks"""
    
    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert "version" in data


class TestPerformance:
    """Integration tests for performance"""
    
    @patch('api.routers.analysis_router.get_service')
    def test_response_includes_processing_time(self, mock_get_service, client, sample_image_bytes):
        """Test that response includes processing time"""
        mock_skin_analyzer = Mock()
        mock_skin_analyzer.use_advanced = True
        mock_skin_analyzer.analyze_image = Mock(return_value={
            "quality_scores": {"overall_score": 75.0},
            "conditions": []
        })
        
        mock_image_processor = Mock()
        mock_image_processor.process_for_analysis = Mock(return_value=(
            np.array(Image.new('RGB', (200, 200))),
            True,
            "Processed"
        ))
        
        mock_validator = Mock()
        mock_validator.validate_image_comprehensive = Mock(return_value=(True, {}))
        
        def get_service_side_effect(service_name):
            services = {
                "skin_analyzer": mock_skin_analyzer,
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
        
        response = client.post(
            "/dermatology/analyze-image",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
            data={"enhance": "true"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "processing_time" in data
        assert isinstance(data["processing_time"], (int, float))

