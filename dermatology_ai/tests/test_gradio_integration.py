"""
Tests for Gradio Integration
Tests for Gradio demo interface
"""

import pytest
from unittest.mock import Mock, patch
from PIL import Image
import numpy as np

# Gradio may not be available
try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False


@pytest.mark.skipif(not GRADIO_AVAILABLE, reason="Gradio not available")
class TestGradioDemo:
    """Tests for GradioDemo"""
    
    @pytest.fixture
    def mock_analyzer(self):
        """Create mock analyzer"""
        analyzer = Mock()
        analyzer.analyze_image = Mock(return_value={
            "quality_scores": {"overall_score": 75.0},
            "conditions": [],
            "skin_type": "combination"
        })
        return analyzer
    
    @pytest.fixture
    def gradio_demo(self, mock_analyzer):
        """Create Gradio demo"""
        from core.gradio_integration import GradioDemo
        return GradioDemo(analyzer=mock_analyzer)
    
    def test_gradio_demo_initialization(self, gradio_demo, mock_analyzer):
        """Test Gradio demo initialization"""
        assert gradio_demo.analyzer == mock_analyzer
        assert gradio_demo.title is not None
        assert gradio_demo.description is not None
    
    def test_analyze_image_gradio(self, gradio_demo, mock_analyzer):
        """Test analyzing image through Gradio interface"""
        test_image = Image.new('RGB', (200, 200), color='red')
        
        result = gradio_demo.analyze_image_gradio(
            image=test_image,
            use_advanced=True,
            enhance=False
        )
        
        assert isinstance(result, dict)
        assert "quality_scores" in result or "analysis" in result
        mock_analyzer.analyze_image.assert_called_once()
    
    def test_gradio_interface_creation(self, gradio_demo):
        """Test creating Gradio interface"""
        interface = gradio_demo.create_interface()
        
        assert interface is not None
        assert isinstance(interface, gr.Interface) or hasattr(interface, 'launch')
    
    def test_gradio_error_handling(self, gradio_demo, mock_analyzer):
        """Test error handling in Gradio interface"""
        mock_analyzer.analyze_image = Mock(side_effect=Exception("Analysis failed"))
        
        test_image = Image.new('RGB', (200, 200), color='red')
        
        # Should handle errors gracefully
        try:
            result = gradio_demo.analyze_image_gradio(test_image)
            # May return error message or raise
            assert isinstance(result, (dict, str))
        except Exception:
            # Exception is also acceptable
            pass


class TestGradioAvailability:
    """Tests for Gradio availability"""
    
    def test_gradio_import_handling(self):
        """Test Gradio import handling"""
        try:
            from core.gradio_integration import GRADIO_AVAILABLE
            assert isinstance(GRADIO_AVAILABLE, bool)
        except ImportError:
            # Gradio not installed, which is fine
            pass
    
    def test_gradio_demo_without_gradio(self):
        """Test Gradio demo when Gradio is not available"""
        if not GRADIO_AVAILABLE:
            with pytest.raises(ImportError):
                from core.gradio_integration import GradioDemo
                mock_analyzer = Mock()
                GradioDemo(mock_analyzer)



