"""
Tests comprehensivos para Dermatology AI
"""

import pytest
import numpy as np
from PIL import Image
import io

from ..core.skin_analyzer import SkinAnalyzer
from ..services.image_processor import ImageProcessor
from ..services.skincare_recommender import SkincareRecommender
from ..services.history_tracker import HistoryTracker
from ..services.product_database import ProductDatabase
from ..utils.advanced_validator import AdvancedImageValidator
from ..utils.rate_limiter import RateLimiter


class TestSkinAnalyzer:
    """Tests para SkinAnalyzer"""
    
    def test_analyze_image_basic(self):
        """Test análisis básico de imagen"""
        analyzer = SkinAnalyzer(use_advanced=False)
        test_image = np.random.randint(100, 200, (400, 400, 3), dtype=np.uint8)
        
        result = analyzer.analyze_image(test_image)
        
        assert "quality_scores" in result
        assert "conditions" in result
        assert "skin_type" in result
        assert 0 <= result["quality_scores"]["overall_score"] <= 100
    
    def test_analyze_image_advanced(self):
        """Test análisis avanzado"""
        analyzer = SkinAnalyzer(use_advanced=True)
        test_image = np.random.randint(100, 200, (400, 400, 3), dtype=np.uint8)
        
        result = analyzer.analyze_image(test_image)
        
        assert "quality_scores" in result
        if "detailed_metrics" in result:
            assert "texture" in result["detailed_metrics"]
    
    def test_analyze_video(self):
        """Test análisis de video"""
        analyzer = SkinAnalyzer()
        frames = [
            np.random.randint(100, 200, (400, 400, 3), dtype=np.uint8)
            for _ in range(5)
        ]
        
        result = analyzer.analyze_video(frames)
        
        assert "quality_scores" in result
        assert "analysis_frames" in result


class TestImageProcessor:
    """Tests para ImageProcessor"""
    
    def test_preprocess_image(self):
        """Test preprocesamiento de imagen"""
        processor = ImageProcessor()
        
        # Crear imagen de prueba
        img = Image.new('RGB', (400, 400), color='rgb(200, 180, 160)')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        processed = processor.preprocess_image(img_bytes.read())
        
        assert processed.shape[:2] == processor.target_size
        assert len(processed.shape) == 3
    
    def test_validate_image_quality(self):
        """Test validación de calidad"""
        processor = ImageProcessor()
        
        # Imagen válida
        valid_image = np.random.randint(50, 200, (400, 400, 3), dtype=np.uint8)
        is_valid, message = processor.validate_image_quality(valid_image)
        
        assert is_valid
        assert "válida" in message.lower()


class TestSkincareRecommender:
    """Tests para SkincareRecommender"""
    
    def test_generate_recommendations(self):
        """Test generación de recomendaciones"""
        recommender = SkincareRecommender()
        
        analysis_result = {
            "quality_scores": {
                "overall_score": 65.0,
                "hydration_score": 45.0
            },
            "conditions": [{"name": "dryness", "severity": "moderate"}],
            "skin_type": "dry",
            "recommendations_priority": ["hydration"]
        }
        
        recommendations = recommender.generate_recommendations(analysis_result)
        
        assert "routine" in recommendations
        assert "morning" in recommendations["routine"]
        assert "evening" in recommendations["routine"]
        assert "tips" in recommendations


class TestHistoryTracker:
    """Tests para HistoryTracker"""
    
    def test_save_and_retrieve(self):
        """Test guardar y recuperar análisis"""
        tracker = HistoryTracker()
        
        analysis_result = {
            "quality_scores": {"overall_score": 75.0},
            "conditions": [],
            "skin_type": "normal",
            "recommendations_priority": []
        }
        
        record_id = tracker.save_analysis(analysis_result, user_id="test_user")
        
        history = tracker.get_user_history("test_user")
        assert len(history) > 0
        assert history[0].id == record_id


class TestProductDatabase:
    """Tests para ProductDatabase"""
    
    def test_search_products(self):
        """Test búsqueda de productos"""
        db = ProductDatabase()
        
        products = db.search_products(
            category="serum",
            min_rating=4.0,
            limit=5
        )
        
        assert isinstance(products, list)
        for product in products:
            assert product.category.value == "serum"
            assert product.rating >= 4.0


class TestAdvancedValidator:
    """Tests para AdvancedImageValidator"""
    
    def test_validate_image(self):
        """Test validación de imagen"""
        validator = AdvancedImageValidator()
        
        # Crear imagen válida
        img = Image.new('RGB', (400, 400), color='rgb(200, 180, 160)')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        is_valid, info = validator.validate_image_comprehensive(img_bytes.read())
        
        assert isinstance(is_valid, bool)
        assert "metadata" in info


class TestRateLimiter:
    """Tests para RateLimiter"""
    
    def test_rate_limiting(self):
        """Test rate limiting"""
        limiter = RateLimiter(max_requests=5, window_seconds=60)
        
        # Hacer 5 requests (deberían pasar)
        for i in range(5):
            allowed, info = limiter.is_allowed("test_user")
            assert allowed
        
        # El 6to debería ser bloqueado
        allowed, info = limiter.is_allowed("test_user")
        assert not allowed
        assert "retry_after" in info


if __name__ == "__main__":
    pytest.main([__file__, "-v"])






