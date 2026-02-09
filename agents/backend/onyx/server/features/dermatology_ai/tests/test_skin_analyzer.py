"""
Tests para SkinAnalyzer
"""

import pytest
import numpy as np
from PIL import Image

from ..core.skin_analyzer import SkinAnalyzer


def test_analyze_image():
    """Test análisis de imagen básico"""
    analyzer = SkinAnalyzer()
    
    # Crear imagen de prueba
    test_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    
    result = analyzer.analyze_image(test_image)
    
    assert "quality_scores" in result
    assert "conditions" in result
    assert "skin_type" in result
    assert "recommendations_priority" in result
    
    assert 0 <= result["quality_scores"]["overall_score"] <= 100


def test_analyze_video():
    """Test análisis de video"""
    analyzer = SkinAnalyzer()
    
    # Crear frames de prueba
    frames = [
        np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        for _ in range(5)
    ]
    
    result = analyzer.analyze_video(frames)
    
    assert "quality_scores" in result
    assert "analysis_frames" in result
    assert result["analysis_frames"] == 5






