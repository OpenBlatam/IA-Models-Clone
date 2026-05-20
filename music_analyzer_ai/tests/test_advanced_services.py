"""
Tests para servicios avanzados
"""

import pytest
from unittest.mock import Mock, patch, MagicMock


class TestTemporalAnalyzer:
    """Tests para TemporalAnalyzer"""
    
    @pytest.fixture
    def temporal_analyzer(self):
        """Fixture para crear TemporalAnalyzer"""
        from ..services.temporal_analyzer import TemporalAnalyzer
        with patch('music_analyzer_ai.services.temporal_analyzer.SpotifyService'):
            return TemporalAnalyzer()
    
    def test_analyze_temporal_structure(self, temporal_analyzer, sample_audio_analysis):
        """Test de análisis de estructura temporal"""
        result = temporal_analyzer.analyze_temporal_structure(sample_audio_analysis)
        
        assert result is not None
        assert isinstance(result, dict)
        assert "temporal_progression" in result or "sections" in result
    
    def test_analyze_temporal_structure_empty(self, temporal_analyzer):
        """Test con datos vacíos"""
        result = temporal_analyzer.analyze_temporal_structure({})
        
        assert "error" in result or result == {}
    
    def test_analyze_energy_progression(self, temporal_analyzer, sample_audio_analysis):
        """Test de análisis de progresión de energía"""
        if hasattr(temporal_analyzer, 'analyze_energy_progression'):
            result = temporal_analyzer.analyze_energy_progression(sample_audio_analysis)
            assert result is not None


class TestQualityAnalyzer:
    """Tests para QualityAnalyzer"""
    
    @pytest.fixture
    def quality_analyzer(self):
        """Fixture para crear QualityAnalyzer"""
        from ..services.quality_analyzer import QualityAnalyzer
        with patch('music_analyzer_ai.services.quality_analyzer.SpotifyService') as mock_spotify:
            mock_spotify.return_value.get_track = Mock(return_value={})
            mock_spotify.return_value.get_track_audio_features = Mock(return_value={})
            mock_spotify.return_value.get_track_audio_analysis = Mock(return_value={})
            return QualityAnalyzer()
    
    @patch('music_analyzer_ai.services.quality_analyzer.SpotifyService')
    def test_analyze_production_quality(self, mock_spotify, sample_audio_features, sample_audio_analysis):
        """Test de análisis de calidad de producción"""
        mock_service = Mock()
        mock_service.get_track = Mock(return_value={"name": "Test Track"})
        mock_service.get_track_audio_features = Mock(return_value=sample_audio_features)
        mock_service.get_track_audio_analysis = Mock(return_value=sample_audio_analysis)
        mock_spotify.return_value = mock_service
        
        from ..services.quality_analyzer import QualityAnalyzer
        analyzer = QualityAnalyzer()
        
        result = analyzer.analyze_production_quality("track_id")
        
        assert result is not None
        assert isinstance(result, dict)
        assert "quality_score" in result or "quality_level" in result or "factors" in result
    
    def test_analyze_audio_quality(self, quality_analyzer, sample_audio_features, sample_audio_analysis):
        """Test de análisis de calidad de audio"""
        if hasattr(quality_analyzer, '_analyze_audio_quality'):
            result = quality_analyzer._analyze_audio_quality(sample_audio_features, sample_audio_analysis)
            assert result is not None


class TestMusicCoach:
    """Tests para MusicCoach"""
    
    @pytest.fixture
    def music_coach(self):
        """Fixture para crear MusicCoach"""
        from ..services.music_coach import MusicCoach
        return MusicCoach()
    
    @pytest.fixture
    def sample_music_analysis(self):
        """Análisis de música de ejemplo"""
        return {
            "track_basic_info": {
                "name": "Test Track",
                "artists": ["Test Artist"]
            },
            "musical_analysis": {
                "key_signature": "C major",
                "tempo": {"bpm": 120.0},
                "time_signature": "4/4"
            },
            "technical_analysis": {
                "energy": {"value": 0.8, "description": "Alta energía"},
                "danceability": {"value": 0.7}
            },
            "composition_analysis": {
                "composition_style": "Rock"
            }
        }
    
    def test_generate_coaching_analysis(self, music_coach, sample_music_analysis):
        """Test de generación de análisis de coaching"""
        result = music_coach.generate_coaching_analysis(sample_music_analysis)
        
        assert result is not None
        assert isinstance(result, dict)
        assert "overview" in result
        assert "technical_breakdown" in result
        assert "learning_path" in result
        assert "practice_exercises" in result
    
    def test_generate_overview(self, music_coach, sample_music_analysis):
        """Test de generación de resumen"""
        result = music_coach._generate_overview(sample_music_analysis)
        
        assert result is not None
        assert "summary" in result
        assert "key_findings" in result
    
    def test_generate_technical_breakdown(self, music_coach, sample_music_analysis):
        """Test de generación de desglose técnico"""
        result = music_coach._generate_technical_breakdown(sample_music_analysis)
        
        assert result is not None
        assert isinstance(result, dict)


class TestMLService:
    """Tests para MLService"""
    
    @pytest.fixture
    def ml_service(self):
        """Fixture para crear MLService"""
        with patch('music_analyzer_ai.services.ml_service.ML_AVAILABLE', True), \
             patch('music_analyzer_ai.services.ml_service.get_deep_analyzer'), \
             patch('music_analyzer_ai.services.ml_service.get_ml_analyzer'), \
             patch('music_analyzer_ai.services.ml_service.get_transformer_analyzer'), \
             patch('music_analyzer_ai.services.ml_service.AudioFeatureExtractor'), \
             patch('music_analyzer_ai.services.ml_service.create_default_pipeline'):
            try:
                from ..services.ml_service import MLService
                return MLService()
            except ImportError:
                pytest.skip("ML components not available")
    
    def test_analyze_track_comprehensive(self, ml_service, sample_audio_features):
        """Test de análisis comprehensivo con ML"""
        if ml_service is None:
            pytest.skip("ML service not available")
        
        result = ml_service.analyze_track_comprehensive(
            audio_features=sample_audio_features
        )
        
        assert result is not None
    
    def test_predict_genre(self, ml_service, sample_audio_features):
        """Test de predicción de género"""
        if ml_service is None:
            pytest.skip("ML service not available")
        
        if hasattr(ml_service, 'predict_genre'):
            result = ml_service.predict_genre(sample_audio_features)
            assert result is not None


class TestTrendsAnalyzer:
    """Tests para TrendsAnalyzer"""
    
    @pytest.fixture
    def trends_analyzer(self):
        """Fixture para crear TrendsAnalyzer"""
        from ..services.trends_analyzer import TrendsAnalyzer
        with patch('music_analyzer_ai.services.trends_analyzer.SpotifyService'):
            return TrendsAnalyzer()
    
    def test_analyze_trends(self, trends_analyzer):
        """Test de análisis de tendencias"""
        if trends_analyzer is None:
            pytest.skip("TrendsAnalyzer not available")
        
        result = trends_analyzer.analyze_trends(genre="Rock", time_range="short_term")
        
        assert result is not None


class TestPlaylistAnalyzer:
    """Tests para PlaylistAnalyzer"""
    
    @pytest.fixture
    def playlist_analyzer(self):
        """Fixture para crear PlaylistAnalyzer"""
        from ..services.playlist_analyzer import PlaylistAnalyzer
        with patch('music_analyzer_ai.services.playlist_analyzer.SpotifyService'):
            return PlaylistAnalyzer()
    
    def test_analyze_playlist(self, playlist_analyzer):
        """Test de análisis de playlist"""
        if playlist_analyzer is None:
            pytest.skip("PlaylistAnalyzer not available")
        
        mock_tracks = [
            {"id": "1", "audio_features": {}},
            {"id": "2", "audio_features": {}}
        ]
        
        result = playlist_analyzer.analyze_playlist(mock_tracks)
        
        assert result is not None


class TestLyricsAnalyzer:
    """Tests para LyricsAnalyzer"""
    
    @pytest.fixture
    def lyrics_analyzer(self):
        """Fixture para crear LyricsAnalyzer"""
        from ..services.lyrics_analyzer import LyricsAnalyzer
        return LyricsAnalyzer()
    
    def test_analyze_lyrics(self, lyrics_analyzer):
        """Test de análisis de letras"""
        if lyrics_analyzer is None:
            pytest.skip("LyricsAnalyzer not available")
        
        lyrics = "This is a test song with meaningful lyrics"
        
        result = lyrics_analyzer.analyze_lyrics(lyrics)
        
        assert result is not None


class TestContextualRecommender:
    """Tests para ContextualRecommender"""
    
    @pytest.fixture
    def contextual_recommender(self):
        """Fixture para crear ContextualRecommender"""
        from ..services.contextual_recommender import ContextualRecommender
        with patch('music_analyzer_ai.services.contextual_recommender.SpotifyService'):
            return ContextualRecommender()
    
    def test_get_recommendations_by_mood(self, contextual_recommender):
        """Test de recomendaciones por mood"""
        if contextual_recommender is None:
            pytest.skip("ContextualRecommender not available")
        
        result = contextual_recommender.get_recommendations_by_mood("happy", limit=10)
        
        assert result is not None
    
    def test_get_recommendations_by_activity(self, contextual_recommender):
        """Test de recomendaciones por actividad"""
        if contextual_recommender is None:
            pytest.skip("ContextualRecommender not available")
        
        result = contextual_recommender.get_recommendations_by_activity("workout", limit=10)
        
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

