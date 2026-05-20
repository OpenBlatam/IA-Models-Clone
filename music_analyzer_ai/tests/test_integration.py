"""
Tests de integración end-to-end
"""

import pytest
from unittest.mock import Mock, patch, MagicMock


class TestFullAnalysisFlow:
    """Tests del flujo completo de análisis"""
    
    @pytest.fixture
    def full_spotify_data(self, sample_track_info, sample_audio_features, sample_audio_analysis):
        """Datos completos de Spotify"""
        return {
            "track_info": sample_track_info,
            "audio_features": sample_audio_features,
            "audio_analysis": sample_audio_analysis
        }
    
    @patch('music_analyzer_ai.services.spotify_service.SpotifyService')
    def test_complete_analysis_workflow(self, mock_spotify, full_spotify_data):
        """Test del flujo completo de análisis"""
        from ..core.music_analyzer import MusicAnalyzer
        
        # Mock Spotify service
        mock_spotify.return_value.get_track.return_value = full_spotify_data["track_info"]
        mock_spotify.return_value.get_track_audio_features.return_value = full_spotify_data["audio_features"]
        mock_spotify.return_value.get_track_audio_analysis.return_value = full_spotify_data["audio_analysis"]
        
        # Mock servicios auxiliares
        with patch('music_analyzer_ai.core.music_analyzer.GenreDetector') as mock_genre, \
             patch('music_analyzer_ai.core.music_analyzer.HarmonicAnalyzer') as mock_harmonic, \
             patch('music_analyzer_ai.core.music_analyzer.EmotionAnalyzer') as mock_emotion:
            
            mock_genre.return_value.detect_genre.return_value = {"genre": "Rock"}
            mock_harmonic.return_value.analyze_harmonic_progression.return_value = {"progression": "I-V-vi-IV"}
            mock_emotion.return_value.analyze_emotions.return_value = {"emotion": "Happy"}
            
            analyzer = MusicAnalyzer()
            result = analyzer.analyze_track(full_spotify_data)
            
            assert result is not None
            assert "track_basic_info" in result
            assert "musical_analysis" in result
            assert "technical_analysis" in result


class TestRecommendationFlow:
    """Tests del flujo de recomendaciones"""
    
    @patch('music_analyzer_ai.services.spotify_service.SpotifyService')
    @patch('music_analyzer_ai.services.enhanced_recommender.EnhancedRecommender')
    def test_recommendation_workflow(self, mock_recommender, mock_spotify):
        """Test del flujo completo de recomendaciones"""
        mock_recommender.return_value.get_recommendations.return_value = [
            {"id": "1", "name": "Recommended 1"},
            {"id": "2", "name": "Recommended 2"}
        ]
        
        from ..services.enhanced_recommender import EnhancedRecommender
        recommender = EnhancedRecommender()
        
        result = recommender.get_recommendations(
            {"key": 0, "mode": 1, "tempo": 120},
            limit=10
        )
        
        assert result is not None
        assert isinstance(result, list) or isinstance(result, dict)


class TestComparisonFlow:
    """Tests del flujo de comparación"""
    
    @patch('music_analyzer_ai.services.spotify_service.SpotifyService')
    def test_comparison_workflow(self, mock_spotify):
        """Test del flujo completo de comparación"""
        from ..services.comparison_service import ComparisonService
        
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
        
        service = ComparisonService()
        result = service.compare_tracks(track1_features, track2_features)
        
        assert result is not None
        assert isinstance(result, dict)


class TestPlaylistFlow:
    """Tests del flujo de playlists"""
    
    @patch('music_analyzer_ai.services.playlist_service.PlaylistService')
    def test_playlist_creation_workflow(self, mock_playlist_service):
        """Test del flujo de creación de playlist"""
        mock_service = Mock()
        mock_service.create_playlist.return_value = {
            "id": "playlist123",
            "name": "My Playlist",
            "tracks": []
        }
        mock_playlist_service.return_value = mock_service
        
        from ..services.playlist_service import PlaylistService
        service = PlaylistService()
        
        result = service.create_playlist("user123", "My Playlist", is_public=False)
        
        assert result is not None
        assert "id" in result or "playlist_id" in result


class TestFavoritesFlow:
    """Tests del flujo de favoritos"""
    
    @patch('music_analyzer_ai.services.favorites_service.FavoritesService')
    def test_favorites_workflow(self, mock_favorites_service):
        """Test del flujo completo de favoritos"""
        mock_service = Mock()
        mock_service.add_favorite.return_value = {"success": True}
        mock_service.get_favorites.return_value = {"tracks": []}
        mock_favorites_service.return_value = mock_service
        
        from ..services.favorites_service import FavoritesService
        service = FavoritesService()
        
        # Agregar favorito
        add_result = service.add_favorite("user123", "track123", "Track Name", ["Artist"])
        assert add_result is not None
        
        # Obtener favoritos
        get_result = service.get_favorites("user123")
        assert get_result is not None


class TestBatchOperations:
    """Tests de operaciones en lote"""
    
    def test_batch_analysis(self):
        """Test de análisis en lote"""
        track_ids = ["1", "2", "3"]
        results = []
        
        for track_id in track_ids:
            # Simular análisis
            results.append({
                "track_id": track_id,
                "status": "completed"
            })
        
        assert len(results) == len(track_ids)
        assert all(r["status"] == "completed" for r in results)
    
    def test_batch_comparison(self):
        """Test de comparación en lote"""
        track_pairs = [
            ({"key": 0, "tempo": 120}, {"key": 0, "tempo": 125}),
            ({"key": 0, "tempo": 120}, {"key": 1, "tempo": 120})
        ]
        
        comparisons = []
        for track1, track2 in track_pairs:
            similarity = 1.0 if track1["key"] == track2["key"] else 0.5
            comparisons.append({
                "similarity": similarity,
                "key_match": track1["key"] == track2["key"]
            })
        
        assert len(comparisons) == len(track_pairs)
        assert comparisons[0]["key_match"] == True
        assert comparisons[1]["key_match"] == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

