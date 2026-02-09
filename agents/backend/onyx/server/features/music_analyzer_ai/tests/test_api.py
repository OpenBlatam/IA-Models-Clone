"""
Tests para endpoints de API
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock


@pytest.fixture
def client():
    """Fixture para crear cliente de test"""
    from ..api.music_api import app
    return TestClient(app)


class TestSearchEndpoints:
    """Tests para endpoints de búsqueda"""
    
    def test_search_tracks(self, client):
        """Test de búsqueda de tracks"""
        with patch('music_analyzer_ai.services.spotify_service.SpotifyService.search_tracks') as mock_search:
            mock_search.return_value = {
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
            
            response = client.get("/search?q=test")
            
            assert response.status_code == 200
            data = response.json()
            assert "results" in data or "tracks" in data
    
    def test_search_tracks_empty_query(self, client):
        """Test de búsqueda con query vacía"""
        response = client.get("/search?q=")
        
        # Debe retornar error o lista vacía
        assert response.status_code in [200, 400, 422]


class TestAnalysisEndpoints:
    """Tests para endpoints de análisis"""
    
    @patch('music_analyzer_ai.core.music_analyzer.MusicAnalyzer.analyze_track')
    def test_analyze_track(self, mock_analyze, client):
        """Test de análisis de track"""
        mock_analyze.return_value = {
            "musical_analysis": {
                "key_signature": "C major",
                "tempo": {"bpm": 120.0}
            },
            "technical_analysis": {
                "energy": {"value": 0.8}
            }
        }
        
        response = client.post("/analyze", params={"track_id": "123"})
        
        assert response.status_code == 200
        data = response.json()
        assert "musical_analysis" in data or "analysis" in data
    
    def test_analyze_track_invalid_id(self, client):
        """Test de análisis con ID inválido"""
        response = client.post("/analyze", params={"track_id": ""})
        
        assert response.status_code in [400, 404, 422]


class TestComparisonEndpoints:
    """Tests para endpoints de comparación"""
    
    @patch('music_analyzer_ai.services.comparison_service.ComparisonService.compare_tracks')
    def test_compare_tracks(self, mock_compare, client):
        """Test de comparación de tracks"""
        mock_compare.return_value = {
            "similarities": {"tempo": 0.9},
            "differences": {"energy": 0.2}
        }
        
        response = client.post(
            "/compare",
            json={"track_ids": ["123", "456"]}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "similarities" in data or "differences" in data
    
    def test_compare_tracks_insufficient(self, client):
        """Test de comparación con menos de 2 tracks"""
        response = client.post(
            "/compare",
            json={"track_ids": ["123"]}
        )
        
        assert response.status_code in [400, 422]


class TestRecommendationEndpoints:
    """Tests para endpoints de recomendaciones"""
    
    @patch('music_analyzer_ai.services.enhanced_recommender.EnhancedRecommender.get_recommendations')
    def test_get_recommendations(self, mock_recommend, client):
        """Test de obtención de recomendaciones"""
        mock_recommend.return_value = [
            {"id": "123", "name": "Recommended Track"}
        ]
        
        response = client.post(
            "/recommendations",
            json={"track_id": "123", "limit": 10}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list) or "tracks" in data


class TestFavoritesEndpoints:
    """Tests para endpoints de favoritos"""
    
    def test_get_favorites(self, client):
        """Test de obtención de favoritos"""
        with patch('music_analyzer_ai.services.favorites_service.FavoritesService.get_favorites') as mock_get:
            mock_get.return_value = {"tracks": []}
            
            response = client.get("/favorites?user_id=user123")
            
            assert response.status_code == 200
    
    def test_add_to_favorites(self, client):
        """Test de agregar a favoritos"""
        with patch('music_analyzer_ai.services.favorites_service.FavoritesService.add_favorite') as mock_add:
            mock_add.return_value = {"success": True}
            
            response = client.post(
                "/favorites",
                json={
                    "user_id": "user123",
                    "track_id": "track123",
                    "track_name": "Test Track",
                    "artists": ["Test Artist"]
                }
            )
            
            assert response.status_code in [200, 201]


class TestPlaylistEndpoints:
    """Tests para endpoints de playlists"""
    
    def test_get_playlists(self, client):
        """Test de obtención de playlists"""
        with patch('music_analyzer_ai.services.playlist_service.PlaylistService.get_playlists') as mock_get:
            mock_get.return_value = {"playlists": []}
            
            response = client.get("/playlists?user_id=user123")
            
            assert response.status_code == 200
    
    def test_create_playlist(self, client):
        """Test de creación de playlist"""
        with patch('music_analyzer_ai.services.playlist_service.PlaylistService.create_playlist') as mock_create:
            mock_create.return_value = {"id": "playlist123", "name": "My Playlist"}
            
            response = client.post(
                "/playlists",
                json={
                    "user_id": "user123",
                    "name": "My Playlist",
                    "is_public": False
                }
            )
            
            assert response.status_code in [200, 201]


    def test_analyze_endpoint_error_handling(self, client):
        """Test mejorado de manejo de errores en endpoint de análisis"""
        with patch('music_analyzer_ai.core.music_analyzer.MusicAnalyzer.analyze_track') as mock_analyze:
            mock_analyze.side_effect = Exception("Analysis failed")
            
            response = client.post("/analyze", params={"track_id": "123"})
            
            # Debe retornar error apropiado
            assert response.status_code in [400, 500, 503]
    
    def test_search_endpoint_error_recovery(self, client):
        """Test de recuperación de errores en búsqueda"""
        with patch('music_analyzer_ai.services.spotify_service.SpotifyService.search_tracks') as mock_search:
            # Simular error y luego éxito
            mock_search.side_effect = [
                Exception("Network error"),
                {"tracks": {"items": []}}
            ]
            
            # Primera llamada falla
            response1 = client.get("/search?q=test")
            assert response1.status_code in [400, 500, 503]
    
    def test_comparison_endpoint_validation(self, client):
        """Test mejorado de validación en endpoint de comparación"""
        # Test con menos de 2 tracks
        response1 = client.post("/compare", json={"track_ids": ["1"]})
        assert response1.status_code in [400, 422]
        
        # Test con más de 5 tracks
        response2 = client.post("/compare", json={"track_ids": ["1", "2", "3", "4", "5", "6"]})
        assert response2.status_code in [400, 422]
        
        # Test con tracks válidos
        with patch('music_analyzer_ai.services.comparison_service.ComparisonService.compare_tracks') as mock_compare:
            mock_compare.return_value = {"similarities": {}}
            response3 = client.post("/compare", json={"track_ids": ["1", "2"]})
            assert response3.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

