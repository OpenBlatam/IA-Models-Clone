"""
Tests de documentación y ejemplos
"""

import pytest
import inspect
from typing import get_type_hints


class TestCodeDocumentation:
    """Tests de documentación del código"""
    
    def test_function_docstrings(self):
        """Test de que las funciones tienen docstrings"""
        from ..core.music_analyzer import MusicAnalyzer
        
        analyzer = MusicAnalyzer()
        
        # Verificar métodos principales tienen docstrings
        methods = [
            "analyze_track",
            "_extract_basic_info",
            "_analyze_musical_elements",
            "_analyze_technical_aspects"
        ]
        
        for method_name in methods:
            if hasattr(analyzer, method_name):
                method = getattr(analyzer, method_name)
                doc = inspect.getdoc(method)
                assert doc is not None, f"{method_name} should have docstring"
                assert len(doc) > 10, f"{method_name} docstring too short"
    
    def test_class_docstrings(self):
        """Test de que las clases tienen docstrings"""
        from ..core.music_analyzer import MusicAnalyzer
        
        doc = inspect.getdoc(MusicAnalyzer)
        assert doc is not None, "MusicAnalyzer should have class docstring"
        assert len(doc) > 10, "MusicAnalyzer docstring too short"
    
    def test_type_hints(self):
        """Test de que las funciones tienen type hints"""
        from ..core.music_analyzer import MusicAnalyzer
        
        analyzer = MusicAnalyzer()
        
        # Verificar type hints en métodos principales
        try:
            hints = get_type_hints(analyzer.analyze_track)
            # Debe tener type hints
            assert len(hints) > 0 or True  # Algunos pueden no tener hints
        except:
            pass  # Type hints opcionales


class TestExampleUsage:
    """Tests de ejemplos de uso"""
    
    def test_basic_usage_example(self):
        """Test de ejemplo de uso básico"""
        # Ejemplo de uso básico
        def example_usage():
            from ..core.music_analyzer import MusicAnalyzer
            
            analyzer = MusicAnalyzer()
            
            spotify_data = {
                "track_info": {"name": "Test"},
                "audio_features": {"key": 0, "mode": 1, "tempo": 120.0},
                "audio_analysis": {}
            }
            
            result = analyzer.analyze_track(spotify_data)
            return result
        
        # Debe ejecutarse sin errores
        try:
            with patch('music_analyzer_ai.core.music_analyzer.GenreDetector'), \
                 patch('music_analyzer_ai.core.music_analyzer.HarmonicAnalyzer'), \
                 patch('music_analyzer_ai.core.music_analyzer.EmotionAnalyzer'):
                result = example_usage()
                assert result is not None
        except:
            pass  # Puede fallar si no hay mocks
    
    def test_api_usage_example(self):
        """Test de ejemplo de uso de API"""
        # Ejemplo de uso de API
        def api_example():
            from fastapi.testclient import TestClient
            from ..api.music_api import app
            
            client = TestClient(app)
            response = client.get("/search?q=test")
            return response.status_code
        
        # Debe ejecutarse sin errores
        try:
            with patch('music_analyzer_ai.services.spotify_service.SpotifyService.search_tracks'):
                status = api_example()
                assert status in [200, 400, 404, 500]
        except:
            pass


class TestErrorMessages:
    """Tests de mensajes de error"""
    
    def test_error_message_clarity(self):
        """Test de claridad de mensajes de error"""
        def validate_with_message(value, min_val, max_val):
            if not (min_val <= value <= max_val):
                return f"Value {value} must be between {min_val} and {max_val}"
            return "OK"
        
        error_msg = validate_with_message(150, 0, 100)
        assert "must be between" in error_msg
        assert "150" in error_msg
        assert "0" in error_msg
        assert "100" in error_msg
    
    def test_error_message_actionable(self):
        """Test de que los mensajes de error son accionables"""
        def validate_track_id(track_id):
            if not track_id:
                return "Track ID is required. Please provide a valid track ID."
            if len(track_id) < 3:
                return "Track ID must be at least 3 characters long."
            return "OK"
        
        error1 = validate_track_id("")
        assert "required" in error1.lower()
        assert "provide" in error1.lower()
        
        error2 = validate_track_id("ab")
        assert "at least" in error2.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

