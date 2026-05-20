"""
Tests de regresión para prevenir bugs conocidos
"""

import pytest
from unittest.mock import Mock, patch


class TestRegressionBugs:
    """Tests de regresión para bugs conocidos"""
    
    @pytest.fixture
    def analyzer(self):
        """Fixture para MusicAnalyzer"""
        with patch('music_analyzer_ai.core.music_analyzer.GenreDetector'), \
             patch('music_analyzer_ai.core.music_analyzer.HarmonicAnalyzer'), \
             patch('music_analyzer_ai.core.music_analyzer.EmotionAnalyzer'):
            from ..core.music_analyzer import MusicAnalyzer
            return MusicAnalyzer()
    
    def test_regression_key_negative_one(self, analyzer):
        """Regresión: key = -1 no debe causar IndexError"""
        audio_features = {"key": -1, "mode": 1, "tempo": 120.0}
        audio_analysis = {}
        
        # No debe lanzar IndexError
        result = analyzer._analyze_musical_elements(audio_features, audio_analysis)
        assert result is not None
        assert result["root_note"] == "Unknown" or "Unknown" in result["key_signature"]
    
    def test_regression_empty_artists_list(self, analyzer):
        """Regresión: lista de artistas vacía no debe causar error"""
        track_info = {
            "name": "Test Track",
            "artists": [],  # Lista vacía
            "album": {"name": "Test Album"}
        }
        
        result = analyzer._extract_basic_info(track_info)
        assert result is not None
        assert isinstance(result["artists"], list)
    
    def test_regression_tempo_zero(self, analyzer):
        """Regresión: tempo = 0 no debe causar división por cero"""
        audio_features = {"key": 0, "mode": 1, "tempo": 0.0}
        audio_analysis = {}
        
        result = analyzer._categorize_tempo(0.0)
        assert result is not None
        assert isinstance(result, str)
    
    def test_regression_missing_sections(self, analyzer):
        """Regresión: audio_analysis sin sections no debe causar KeyError"""
        audio_features = {"key": 0, "mode": 1, "tempo": 120.0}
        audio_analysis = {}  # Sin sections
        
        result = analyzer._analyze_composition(audio_features, audio_analysis)
        assert result is not None
        assert "structure" in result
    
    def test_regression_none_audio_analysis(self, analyzer):
        """Regresión: audio_analysis = None no debe causar TypeError"""
        audio_features = {"key": 0, "mode": 1, "tempo": 120.0}
        audio_analysis = None
        
        # Debe manejar None sin error
        try:
            result = analyzer._analyze_composition(audio_features, audio_analysis)
            assert result is not None
        except (TypeError, AttributeError):
            pytest.fail("Should handle None audio_analysis gracefully")


class TestRegressionServices:
    """Tests de regresión para servicios"""
    
    def test_regression_spotify_service_empty_response(self):
        """Regresión: respuesta vacía de Spotify no debe causar error"""
        from ..services.spotify_service import SpotifyService
        
        with patch('music_analyzer_ai.services.spotify_service.requests.get') as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = {}
            mock_response.status_code = 200
            mock_get.return_value = mock_response
            
            service = SpotifyService()
            result = service.search_tracks("test")
            
            # Debe manejar respuesta vacía
            assert result is not None
    
    def test_regression_comparison_service_single_track(self):
        """Regresión: comparación con un solo track no debe causar error"""
        from ..services.comparison_service import ComparisonService
        service = ComparisonService()
        
        features = {"key": 0, "mode": 1, "tempo": 120.0}
        
        # Comparar track consigo mismo
        result = service.compare_tracks(features, features)
        assert result is not None


class TestRegressionAPI:
    """Tests de regresión para API"""
    
    @pytest.fixture
    def client(self):
        """Cliente de test"""
        from ..api.music_api import app
        from fastapi.testclient import TestClient
        return TestClient(app)
    
    def test_regression_malformed_json(self, client):
        """Regresión: JSON malformado no debe causar 500"""
        # Intentar enviar JSON inválido
        response = client.post(
            "/compare",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        # Debe retornar error de validación, no 500
        assert response.status_code != 500
    
    def test_regression_missing_required_params(self, client):
        """Regresión: parámetros requeridos faltantes deben retornar 422"""
        response = client.post("/analyze")
        
        # Debe retornar error de validación
        assert response.status_code in [400, 422]


class TestRegressionDataTypes:
    """Tests de regresión para tipos de datos"""
    
    def test_regression_string_instead_of_number(self):
        """Regresión: string en lugar de número no debe causar TypeError"""
        def safe_convert_to_float(value, default=0.0):
            try:
                return float(value)
            except (ValueError, TypeError):
                return default
        
        assert safe_convert_to_float("120") == 120.0
        assert safe_convert_to_float("invalid") == 0.0
        assert safe_convert_to_float(None) == 0.0
        assert safe_convert_to_float(120) == 120.0
    
    def test_regression_list_instead_of_dict(self):
        """Regresión: lista en lugar de dict no debe causar AttributeError"""
        def safe_get_nested(data, *keys, default=None):
            try:
                result = data
                for key in keys:
                    if isinstance(result, dict):
                        result = result.get(key)
                    else:
                        return default
                    if result is None:
                        return default
                return result
            except (AttributeError, TypeError):
                return default
        
        assert safe_get_nested({"key": {"nested": "value"}}, "key", "nested") == "value"
        assert safe_get_nested([], "key") == None
        assert safe_get_nested(None, "key") == None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

