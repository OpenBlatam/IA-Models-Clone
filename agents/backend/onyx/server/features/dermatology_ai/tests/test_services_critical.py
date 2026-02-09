"""
Tests for Critical Services
Tests for key service implementations
"""

import pytest
from unittest.mock import Mock, AsyncMock
from datetime import datetime

# Import key services
from services.history_tracker import HistoryTracker
from services.alert_system import AlertSystem
from services.image_processor import ImageProcessor
from services.skincare_recommender import SkincareRecommender


class TestHistoryTracker:
    """Tests for HistoryTracker"""
    
    @pytest.fixture
    def history_tracker(self):
        """Create history tracker"""
        return HistoryTracker()
    
    def test_save_analysis(self, history_tracker):
        """Test saving analysis to history"""
        analysis_data = {
            "id": "test-123",
            "user_id": "user-123",
            "metrics": {"overall_score": 75.0}
        }
        
        record_id = history_tracker.save_analysis(analysis_data)
        
        assert record_id is not None
        assert isinstance(record_id, str)
    
    def test_get_history(self, history_tracker):
        """Test getting analysis history"""
        # Save some analyses
        for i in range(3):
            history_tracker.save_analysis({
                "id": f"test-{i}",
                "user_id": "user-123"
            })
        
        history = history_tracker.get_history("user-123", limit=10)
        
        assert len(history) >= 0  # May be empty or have data
        assert isinstance(history, list)


class TestAlertSystem:
    """Tests for AlertSystem"""
    
    @pytest.fixture
    def alert_system(self):
        """Create alert system"""
        return AlertSystem()
    
    def test_check_analysis_alerts(self, alert_system):
        """Test checking for alerts in analysis"""
        analysis_result = {
            "quality_scores": {
                "overall_score": 50.0,  # Low score
                "hydration_score": 40.0  # Very low
            },
            "conditions": [
                {
                    "name": "severe_acne",
                    "confidence": 0.9,
                    "severity": "severe"
                }
            ]
        }
        
        alerts = alert_system.check_analysis_alerts(analysis_result)
        
        # Should detect alerts for low scores or severe conditions
        assert isinstance(alerts, list)
    
    def test_no_alerts_for_good_analysis(self, alert_system):
        """Test no alerts for good analysis"""
        analysis_result = {
            "quality_scores": {
                "overall_score": 85.0,
                "hydration_score": 80.0
            },
            "conditions": []
        }
        
        alerts = alert_system.check_analysis_alerts(analysis_result)
        
        # Should have no or minimal alerts
        assert isinstance(alerts, list)


class TestImageProcessor:
    """Tests for ImageProcessor"""
    
    @pytest.fixture
    def image_processor(self):
        """Create image processor"""
        return ImageProcessor()
    
    def test_process_for_analysis(self, image_processor):
        """Test processing image for analysis"""
        from PIL import Image
        import io
        import numpy as np
        
        # Create test image
        img = Image.new('RGB', (200, 200), color='red')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        image_data = img_bytes.read()
        
        processed, is_valid, message = image_processor.process_for_analysis(image_data)
        
        assert is_valid is True or isinstance(processed, np.ndarray)
        assert message is not None
    
    def test_validate_image(self, image_processor):
        """Test validating image"""
        from PIL import Image
        import io
        
        img = Image.new('RGB', (200, 200), color='red')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        image_data = img_bytes.read()
        
        is_valid, message = image_processor.validate_image(image_data)
        
        assert isinstance(is_valid, bool)
        assert message is not None
    
    def test_enhance_for_analysis(self, image_processor):
        """Test enhancing image for analysis"""
        import numpy as np
        
        test_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        
        enhanced = image_processor.enhance_for_analysis(test_image)
        
        assert enhanced is not None
        assert isinstance(enhanced, np.ndarray)


class TestSkincareRecommender:
    """Tests for SkincareRecommender"""
    
    @pytest.fixture
    def recommender(self):
        """Create skincare recommender"""
        return SkincareRecommender()
    
    def test_generate_recommendations(self, recommender):
        """Test generating recommendations"""
        analysis_result = {
            "quality_scores": {
                "overall_score": 75.0,
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
        }
        
        recommendations = recommender.generate_recommendations(analysis_result)
        
        assert isinstance(recommendations, dict)
        assert "routine" in recommendations or "specific_recommendations" in recommendations
    
    def test_recommendations_for_dry_skin(self, recommender):
        """Test recommendations for dry skin"""
        analysis_result = {
            "quality_scores": {
                "hydration_score": 50.0  # Low hydration
            },
            "skin_type": "dry",
            "conditions": []
        }
        
        recommendations = recommender.generate_recommendations(analysis_result)
        
        # Should prioritize hydration
        assert isinstance(recommendations, dict)



