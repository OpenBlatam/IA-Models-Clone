"""
Comprehensive Tests for Skin Analysis Components
Tests for SkinAnalyzer, AdvancedSkinAnalyzer, SkinQualityMetrics, SkinConditionsDetector
"""

import pytest
from unittest.mock import Mock, patch
import numpy as np
from PIL import Image
import io

from core.skin_analyzer import SkinAnalyzer
from core.advanced_skin_analyzer import AdvancedSkinAnalyzer
from core.skin_quality_metrics import SkinQualityMetrics
from core.skin_conditions_detector import SkinConditionsDetector


class TestSkinQualityMetrics:
    """Tests for SkinQualityMetrics"""
    
    @pytest.fixture
    def quality_metrics(self):
        """Create skin quality metrics"""
        return SkinQualityMetrics()
    
    def test_calculate_texture_score(self, quality_metrics):
        """Test calculating texture score"""
        test_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        
        score = quality_metrics.calculate_texture_score(test_image)
        
        assert 0 <= score <= 100
        assert isinstance(score, (int, float))
    
    def test_calculate_hydration_score(self, quality_metrics):
        """Test calculating hydration score"""
        test_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        
        score = quality_metrics.calculate_hydration_score(test_image)
        
        assert 0 <= score <= 100
        assert isinstance(score, (int, float))
    
    def test_calculate_all_metrics(self, quality_metrics):
        """Test calculating all quality metrics"""
        test_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        
        metrics = quality_metrics.calculate_all_metrics(test_image)
        
        assert "texture_score" in metrics
        assert "hydration_score" in metrics
        assert "elasticity_score" in metrics
        assert "overall_score" in metrics
        assert all(0 <= v <= 100 for v in metrics.values() if isinstance(v, (int, float)))


class TestSkinConditionsDetector:
    """Tests for SkinConditionsDetector"""
    
    @pytest.fixture
    def conditions_detector(self):
        """Create skin conditions detector"""
        return SkinConditionsDetector()
    
    def test_detect_acne(self, conditions_detector):
        """Test detecting acne"""
        test_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        
        conditions = conditions_detector.detect_conditions(test_image)
        
        assert isinstance(conditions, list)
        # May or may not detect acne depending on image
        assert all(isinstance(c, dict) or hasattr(c, 'name') for c in conditions)
    
    def test_detect_multiple_conditions(self, conditions_detector):
        """Test detecting multiple conditions"""
        test_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        
        conditions = conditions_detector.detect_conditions(test_image)
        
        assert isinstance(conditions, list)
        # Should return list of conditions (may be empty)
    
    def test_condition_confidence(self, conditions_detector):
        """Test condition confidence levels"""
        test_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        
        conditions = conditions_detector.detect_conditions(test_image)
        
        # If conditions detected, check confidence
        for condition in conditions:
            if isinstance(condition, dict):
                assert "confidence" in condition
                assert 0 <= condition["confidence"] <= 1
            elif hasattr(condition, 'confidence'):
                assert 0 <= condition.confidence <= 1


class TestAdvancedSkinAnalyzer:
    """Tests for AdvancedSkinAnalyzer"""
    
    @pytest.fixture
    def advanced_analyzer(self):
        """Create advanced skin analyzer"""
        return AdvancedSkinAnalyzer()
    
    def test_advanced_analysis(self, advanced_analyzer):
        """Test advanced analysis"""
        test_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        
        metrics = advanced_analyzer.calculate_all_metrics(test_image)
        
        assert "overall_score" in metrics
        assert all(0 <= v <= 100 for k, v in metrics.items() 
                  if isinstance(v, (int, float)) and "score" in k)
    
    def test_advanced_vs_basic(self, advanced_analyzer):
        """Test that advanced analyzer provides more detailed metrics"""
        test_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        
        basic_metrics = SkinQualityMetrics().calculate_all_metrics(test_image)
        advanced_metrics = advanced_analyzer.calculate_all_metrics(test_image)
        
        # Advanced should have at least as many metrics as basic
        assert len(advanced_metrics) >= len(basic_metrics)


class TestSkinAnalyzerComprehensive:
    """Comprehensive tests for SkinAnalyzer"""
    
    @pytest.fixture
    def skin_analyzer(self):
        """Create skin analyzer"""
        return SkinAnalyzer(use_advanced=True, use_cache=True)
    
    def test_analyze_numpy_array(self, skin_analyzer):
        """Test analyzing numpy array"""
        test_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        
        result = skin_analyzer.analyze_image(test_image)
        
        assert "quality_scores" in result
        assert "conditions" in result
        assert "skin_type" in result
        assert "recommendations_priority" in result
    
    def test_analyze_pil_image(self, skin_analyzer):
        """Test analyzing PIL Image"""
        test_image = Image.new('RGB', (200, 200), color='red')
        
        result = skin_analyzer.analyze_image(test_image)
        
        assert "quality_scores" in result
        assert "conditions" in result
    
    def test_analyze_image_bytes(self, skin_analyzer):
        """Test analyzing image bytes"""
        img = Image.new('RGB', (200, 200), color='blue')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        image_bytes = img_bytes.read()
        
        result = skin_analyzer.analyze_image(image_bytes)
        
        assert "quality_scores" in result
        assert "conditions" in result
    
    def test_analyze_with_cache(self, skin_analyzer):
        """Test analysis with cache enabled"""
        test_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        
        # First analysis
        result1 = skin_analyzer.analyze_image(test_image, use_cache=True)
        
        # Second analysis (should use cache)
        result2 = skin_analyzer.analyze_image(test_image, use_cache=True)
        
        # Results should be similar (may be identical if cached)
        assert "quality_scores" in result1
        assert "quality_scores" in result2
    
    def test_analyze_without_cache(self, skin_analyzer):
        """Test analysis without cache"""
        test_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        
        result = skin_analyzer.analyze_image(test_image, use_cache=False)
        
        assert "quality_scores" in result
        assert "conditions" in result
    
    def test_analyze_video(self, skin_analyzer):
        """Test analyzing video frames"""
        frames = [
            np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
            for _ in range(5)
        ]
        
        result = skin_analyzer.analyze_video(frames)
        
        assert "quality_scores" in result
        assert "analysis_frames" in result
        assert result["analysis_frames"] == 5
    
    def test_basic_vs_advanced_mode(self):
        """Test basic vs advanced analysis mode"""
        basic_analyzer = SkinAnalyzer(use_advanced=False)
        advanced_analyzer = SkinAnalyzer(use_advanced=True)
        
        test_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        
        basic_result = basic_analyzer.analyze_image(test_image)
        advanced_result = advanced_analyzer.analyze_image(test_image)
        
        # Both should return valid results
        assert "quality_scores" in basic_result
        assert "quality_scores" in advanced_result
        # Advanced may have more detailed analysis
        assert isinstance(basic_result["quality_scores"]["overall_score"], (int, float))
        assert isinstance(advanced_result["quality_scores"]["overall_score"], (int, float))
    
    def test_skin_type_detection(self, skin_analyzer):
        """Test skin type detection"""
        test_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        
        result = skin_analyzer.analyze_image(test_image)
        
        assert "skin_type" in result
        assert result["skin_type"] in ["dry", "oily", "combination", "normal", "sensitive"] or result["skin_type"] is not None
    
    def test_recommendations_priority(self, skin_analyzer):
        """Test recommendations priority calculation"""
        test_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        
        result = skin_analyzer.analyze_image(test_image)
        
        assert "recommendations_priority" in result
        assert isinstance(result["recommendations_priority"], list)



