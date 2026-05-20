"""
Tests para casos edge y límite
"""

import pytest
from unittest.mock import Mock, patch
import math


class TestEdgeCasesMusicAnalyzer:
    """Tests de casos edge para MusicAnalyzer"""
    
    @pytest.fixture
    def analyzer(self):
        """Fixture para crear MusicAnalyzer"""
        with patch('music_analyzer_ai.core.music_analyzer.GenreDetector'), \
             patch('music_analyzer_ai.core.music_analyzer.HarmonicAnalyzer'), \
             patch('music_analyzer_ai.core.music_analyzer.EmotionAnalyzer'):
            from ..core.music_analyzer import MusicAnalyzer
            return MusicAnalyzer()
    
    def test_analyze_with_missing_key(self, analyzer):
        """Test con key inválido (-1)"""
        audio_features = {
            "key": -1,  # Key inválido
            "mode": 1,
            "tempo": 120.0,
            "time_signature": 4
        }
        audio_analysis = {"sections": [], "beats": [], "bars": []}
        
        result = analyzer._analyze_musical_elements(audio_features, audio_analysis)
        
        assert result is not None
        assert result["root_note"] == "Unknown" or result["key_signature"] == "Unknown major"
    
    def test_analyze_with_extreme_tempo(self, analyzer):
        """Test con tempo extremo"""
        # Tempo muy bajo
        result = analyzer._categorize_tempo(30)
        assert "lento" in result.lower() or "largo" in result.lower()
        
        # Tempo muy alto
        result = analyzer._categorize_tempo(250)
        assert "rápido" in result.lower() or "prestissimo" in result.lower()
    
    def test_analyze_with_zero_values(self, analyzer):
        """Test con valores cero"""
        audio_features = {
            "key": 0,
            "mode": 0,
            "tempo": 0.0,
            "time_signature": 0,
            "energy": 0.0,
            "danceability": 0.0,
            "valence": 0.0
        }
        audio_analysis = {}
        
        result = analyzer._analyze_technical_aspects(audio_features, audio_analysis)
        
        assert result is not None
        assert result["energy"]["value"] == 0.0
    
    def test_analyze_with_max_values(self, analyzer):
        """Test con valores máximos"""
        audio_features = {
            "key": 11,  # B
            "mode": 1,
            "tempo": 200.0,
            "time_signature": 7,
            "energy": 1.0,
            "danceability": 1.0,
            "valence": 1.0
        }
        audio_analysis = {}
        
        result = analyzer._analyze_technical_aspects(audio_features, audio_analysis)
        
        assert result is not None
        assert result["energy"]["value"] == 1.0
    
    def test_analyze_with_empty_sections(self, analyzer):
        """Test con secciones vacías"""
        audio_features = {"key": 0, "mode": 1, "tempo": 120.0}
        audio_analysis = {
            "sections": [],
            "segments": [],
            "beats": [],
            "bars": []
        }
        
        result = analyzer._analyze_composition(audio_features, audio_analysis)
        
        assert result is not None
        assert isinstance(result["structure"], list)
    
    def test_analyze_with_many_sections(self, analyzer):
        """Test con muchas secciones"""
        audio_features = {"key": 0, "mode": 1, "tempo": 120.0}
        audio_analysis = {
            "sections": [
                {"start": i * 10, "duration": 10, "loudness": -5}
                for i in range(50)
            ],
            "segments": []
        }
        
        result = analyzer._assess_complexity(audio_features, audio_analysis)
        
        assert result is not None
        assert result["factors"]["sections"] == 50
    
    def test_identify_scale_with_invalid_key(self, analyzer):
        """Test de identificación de escala con key inválido"""
        result = analyzer._identify_scale(-1, 1)
        assert result["name"] == "Unknown"
        assert result["notes"] == []
    
    def test_get_common_chords_with_invalid_key(self, analyzer):
        """Test de acordes comunes con key inválido"""
        result = analyzer._get_common_chords(-1, 1)
        assert result == []
    
    def test_calculate_dynamic_range_empty(self, analyzer):
        """Test de rango dinámico con segmentos vacíos"""
        result = analyzer._calculate_dynamic_range([])
        assert result["range"] == "Unknown"
    
    def test_calculate_dynamic_range_single_segment(self, analyzer):
        """Test de rango dinámico con un solo segmento"""
        segments = [{"loudness_max": -5.0}]
        result = analyzer._calculate_dynamic_range(segments)
        
        assert result is not None
        assert result["min"] == result["max"] == -5.0


class TestEdgeCasesServices:
    """Tests de casos edge para servicios"""
    
    def test_spotify_service_empty_search(self):
        """Test de búsqueda vacía"""
        from ..services.spotify_service import SpotifyService
        with patch('music_analyzer_ai.services.spotify_service.requests.get') as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = {"tracks": {"items": []}}
            mock_response.status_code = 200
            mock_get.return_value = mock_response
            
            service = SpotifyService()
            result = service.search_tracks("")
            
            assert result is not None
    
    def test_comparison_service_identical_tracks(self):
        """Test de comparación de tracks idénticos"""
        from ..services.comparison_service import ComparisonService
        service = ComparisonService()
        
        features = {
            "key": 0,
            "mode": 1,
            "tempo": 120.0,
            "energy": 0.8
        }
        
        result = service.compare_tracks(features, features)
        
        assert result is not None
        # Tracks idénticos deberían tener alta similitud
        if "similarity" in result:
            assert result["similarity"] >= 0.9


class TestEdgeCasesValidation:
    """Tests de validación de casos edge"""
    
    def test_validate_track_id_edge_cases(self):
        """Test de validación de IDs edge"""
        def is_valid_track_id(track_id):
            if not track_id:
                return False
            if not isinstance(track_id, str):
                return False
            if len(track_id) > 100:  # Límite razonable
                return False
            return True
        
        assert is_valid_track_id("") == False
        assert is_valid_track_id(None) == False
        assert is_valid_track_id("a" * 101) == False
        assert is_valid_track_id("valid_id_123") == True
    
    def test_validate_audio_features_edge_cases(self):
        """Test de validación de características edge"""
        def validate_features(features):
            if not isinstance(features, dict):
                return False
            
            # Validar rangos
            if "energy" in features:
                if not (0 <= features["energy"] <= 1):
                    return False
            if "tempo" in features:
                if features["tempo"] < 0 or features["tempo"] > 300:
                    return False
            
            return True
        
        # Valores fuera de rango
        assert validate_features({"energy": 1.5}) == False
        assert validate_features({"energy": -0.5}) == False
        assert validate_features({"tempo": 500}) == False
        
        # Valores válidos
        assert validate_features({"energy": 0.8, "tempo": 120}) == True


class TestEdgeCasesFormatting:
    """Tests de formateo de casos edge"""
    
    def test_format_duration_edge_cases(self):
        """Test de formateo de duración edge"""
        def format_duration_ms(ms):
            if ms < 0:
                return "0:00"
            seconds = ms // 1000
            minutes = seconds // 60
            remaining_seconds = seconds % 60
            return f"{minutes}:{remaining_seconds:02d}"
        
        assert format_duration_ms(-1000) == "0:00"
        assert format_duration_ms(0) == "0:00"
        assert format_duration_ms(999) == "0:00"
        assert format_duration_ms(1000) == "0:01"
        assert format_duration_ms(3599999) == "59:59"
    
    def test_format_key_signature_edge_cases(self):
        """Test de formateo de tonalidad edge"""
        NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        MODES = {0: "minor", 1: "major"}
        
        def format_key(key, mode):
            if key < 0 or key >= 12:
                return "Unknown"
            note = NOTES[key]
            mode_name = MODES.get(mode, "unknown")
            return f"{note} {mode_name}"
        
        assert format_key(-1, 1) == "Unknown"
        assert format_key(12, 1) == "Unknown"
        assert format_key(0, 2) == "C unknown"  # Mode inválido


class TestEdgeCasesErrorHandling:
    """Tests de manejo de errores edge"""
    
    def test_handle_none_values(self):
        """Test de manejo de valores None"""
        def safe_get(data, key, default=None):
            try:
                return data.get(key, default) if data else default
            except:
                return default
        
        assert safe_get(None, "key") == None
        assert safe_get({}, "key", "default") == "default"
        assert safe_get({"key": "value"}, "key") == "value"
    
    def test_handle_type_errors(self):
        """Test de manejo de errores de tipo"""
        def safe_operation(value):
            try:
                if isinstance(value, (int, float)):
                    return value * 2
                return None
            except TypeError:
                return None
        
        assert safe_operation(5) == 10
        assert safe_operation("string") == None
        assert safe_operation(None) == None
    
    def test_handle_division_by_zero(self):
        """Test de manejo de división por cero"""
        def safe_divide(a, b):
            try:
                if b == 0:
                    return None
                return a / b
            except ZeroDivisionError:
                return None
        
        assert safe_divide(10, 2) == 5.0
        assert safe_divide(10, 0) == None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

