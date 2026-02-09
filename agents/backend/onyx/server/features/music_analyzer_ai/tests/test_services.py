"""
Tests para servicios de música
"""

import pytest
from unittest.mock import Mock, patch, MagicMock


class TestGenreDetector:
    """Tests para GenreDetector"""
    
    @pytest.fixture
    def genre_detector(self):
        """Fixture para crear GenreDetector"""
        from ..services.genre_detector import GenreDetector
        return GenreDetector()
    
    def test_detect_genre(self, genre_detector):
        """Test de detección de género"""
        audio_features = {
            "energy": 0.8,
            "danceability": 0.7,
            "acousticness": 0.2,
            "valence": 0.6
        }
        audio_analysis = {}
        
        result = genre_detector.detect_genre(audio_features, audio_analysis)
        
        assert "genre" in result or "genres" in result or "primary_genre" in result


class TestEmotionAnalyzer:
    """Tests para EmotionAnalyzer"""
    
    @pytest.fixture
    def emotion_analyzer(self):
        """Fixture para crear EmotionAnalyzer"""
        from ..services.emotion_analyzer import EmotionAnalyzer
        return EmotionAnalyzer()
    
    def test_analyze_emotions(self, emotion_analyzer):
        """Test de análisis de emociones"""
        audio_features = {
            "valence": 0.7,
            "energy": 0.8,
            "danceability": 0.6
        }
        
        result = emotion_analyzer.analyze_emotions(audio_features)
        
        assert result is not None
        assert isinstance(result, dict)


class TestHarmonicAnalyzer:
    """Tests para HarmonicAnalyzer"""
    
    @pytest.fixture
    def harmonic_analyzer(self):
        """Fixture para crear HarmonicAnalyzer"""
        from ..services.harmonic_analyzer import HarmonicAnalyzer
        return HarmonicAnalyzer()
    
    def test_analyze_harmonic_progression(self, harmonic_analyzer):
        """Test de análisis de progresión armónica"""
        audio_analysis = {
            "segments": [
                {"start": 0.0, "pitches": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2]}
            ]
        }
        key = 0  # C
        mode = 1  # Major
        
        result = harmonic_analyzer.analyze_harmonic_progression(audio_analysis, key, mode)
        
        assert result is not None
        assert isinstance(result, dict)


class TestSpotifyService:
    """Tests para SpotifyService"""
    
    @pytest.fixture
    def spotify_service(self):
        """Fixture para crear SpotifyService"""
        from ..services.spotify_service import SpotifyService
        return SpotifyService()
    
    @patch('requests.get')
    def test_search_tracks(self, mock_get, spotify_service):
        """Test de búsqueda de tracks"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "tracks": {
                "items": [
                    {
                        "id": "123",
                        "name": "Test Track",
                        "artists": [{"name": "Test Artist"}]
                    }
                ]
            }
        }
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        result = spotify_service.search_tracks("test query")
        
        assert result is not None
    
    @patch('requests.get')
    def test_get_track_features(self, mock_get, spotify_service):
        """Test de obtención de características de audio"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "key": 0,
            "mode": 1,
            "tempo": 120.0
        }
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        result = spotify_service.get_track_features("track_id")
        
        assert result is not None
        assert "key" in result


class TestComparisonService:
    """Tests para ComparisonService"""
    
    @pytest.fixture
    def comparison_service(self):
        """Fixture para crear ComparisonService"""
        from ..services.comparison_service import ComparisonService
        return ComparisonService()
    
    def test_compare_tracks(self, comparison_service):
        """Test de comparación de tracks"""
        track1_features = {
            "key": 0,
            "mode": 1,
            "tempo": 120.0,
            "energy": 0.8
        }
        track2_features = {
            "key": 0,
            "mode": 1,
            "tempo": 125.0,
            "energy": 0.7
        }
        
        result = comparison_service.compare_tracks(
            track1_features, track2_features
        )
        
        assert result is not None
        assert "similarities" in result or "differences" in result


class TestRecommendationService:
    """Tests para servicios de recomendación"""
    
    @pytest.fixture
    def recommender(self):
        """Fixture para crear recomendador"""
        from ..services.enhanced_recommender import EnhancedRecommender
        return EnhancedRecommender()
    
    def test_get_recommendations(self, recommender):
        """Test de obtención de recomendaciones"""
        track_features = {
            "key": 0,
            "mode": 1,
            "tempo": 120.0,
            "energy": 0.8
        }
        
        result = recommender.get_recommendations(track_features, limit=10)
        
        assert result is not None
        assert isinstance(result, list) or isinstance(result, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

