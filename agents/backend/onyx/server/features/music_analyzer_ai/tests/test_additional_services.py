"""
Tests para servicios adicionales
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import json


class TestDiscoveryService:
    """Tests para DiscoveryService"""
    
    @pytest.fixture
    def discovery_service(self):
        """Fixture para crear DiscoveryService"""
        from ..services.discovery_service import DiscoveryService
        with patch('music_analyzer_ai.services.discovery_service.SpotifyService') as mock_spotify, \
             patch('music_analyzer_ai.services.discovery_service.GenreDetector'), \
             patch('music_analyzer_ai.services.discovery_service.EmotionAnalyzer'), \
             patch('music_analyzer_ai.services.discovery_service.IntelligentRecommender'):
            mock_spotify.return_value.search_tracks.return_value = [
                {"id": "1", "name": "Track 1"}
            ]
            mock_spotify.return_value.get_track_audio_features.return_value = {
                "key": 0, "mode": 1, "tempo": 120.0
            }
            return DiscoveryService()
    
    def test_discover_similar_artists(self, discovery_service):
        """Test de descubrimiento de artistas similares"""
        if discovery_service is None:
            pytest.skip("DiscoveryService not available")
        
        result = discovery_service.discover_similar_artists("Test Artist", limit=10)
        
        assert result is not None
        assert isinstance(result, dict)
    
    def test_discover_similar_artists_not_found(self, discovery_service):
        """Test con artista no encontrado"""
        if discovery_service is None:
            pytest.skip("DiscoveryService not available")
        
        discovery_service.spotify.search_tracks.return_value = []
        
        result = discovery_service.discover_similar_artists("Unknown Artist")
        
        assert "error" in result or result == {}
    
    def test_discover_underground_music(self, discovery_service):
        """Test de descubrimiento de música underground"""
        if discovery_service is None:
            pytest.skip("DiscoveryService not available")
        
        if hasattr(discovery_service, 'discover_underground_music'):
            result = discovery_service.discover_underground_music(limit=10)
            assert result is not None


class TestExportService:
    """Tests para ExportService"""
    
    @pytest.fixture
    def export_service(self):
        """Fixture para crear ExportService"""
        from ..services.export_service import ExportService
        return ExportService()
    
    @pytest.fixture
    def sample_analysis(self):
        """Análisis de ejemplo"""
        return {
            "track_basic_info": {
                "name": "Test Track",
                "artists": ["Test Artist"],
                "album": "Test Album"
            },
            "musical_analysis": {
                "key_signature": "C major",
                "tempo": {"bpm": 120.0}
            },
            "coaching": {
                "overview": {"summary": "Test summary"}
            }
        }
    
    def test_export_to_json(self, export_service, sample_analysis):
        """Test de exportación a JSON"""
        result = export_service.export_to_json(sample_analysis, include_coaching=True)
        
        assert result is not None
        assert isinstance(result, str)
        
        # Verificar que es JSON válido
        parsed = json.loads(result)
        assert "export_date" in parsed
        assert "analysis" in parsed
        assert "coaching" in parsed
    
    def test_export_to_json_without_coaching(self, export_service, sample_analysis):
        """Test de exportación a JSON sin coaching"""
        result = export_service.export_to_json(sample_analysis, include_coaching=False)
        
        parsed = json.loads(result)
        assert "coaching" not in parsed or parsed.get("coaching") is None
    
    def test_export_to_text(self, export_service, sample_analysis):
        """Test de exportación a texto"""
        result = export_service.export_to_text(sample_analysis, include_coaching=True)
        
        assert result is not None
        assert isinstance(result, str)
        assert "Test Track" in result
        assert "Test Artist" in result
    
    def test_export_to_markdown(self, export_service, sample_analysis):
        """Test de exportación a Markdown"""
        if hasattr(export_service, 'export_to_markdown'):
            result = export_service.export_to_markdown(sample_analysis)
            assert result is not None
            assert isinstance(result, str)
            assert "#" in result or "**" in result  # Formato markdown


class TestHistoryService:
    """Tests para HistoryService"""
    
    @pytest.fixture
    def temp_storage(self):
        """Directorio temporal para storage"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def history_service(self, temp_storage):
        """Fixture para crear HistoryService"""
        from ..services.history_service import HistoryService
        return HistoryService(storage_path=temp_storage)
    
    def test_add_analysis(self, history_service):
        """Test de agregar análisis al historial"""
        analysis = {
            "musical_analysis": {"key_signature": "C major"},
            "technical_analysis": {"energy": {"value": 0.8}}
        }
        
        entry_id = history_service.add_analysis(
            track_id="123",
            track_name="Test Track",
            artists=["Test Artist"],
            analysis=analysis,
            user_id="user123"
        )
        
        assert entry_id is not None
        assert len(history_service.history) == 1
    
    def test_get_analysis_history(self, history_service):
        """Test de obtener historial de análisis"""
        # Agregar algunos análisis
        for i in range(3):
            history_service.add_analysis(
                track_id=f"track_{i}",
                track_name=f"Track {i}",
                artists=["Artist"],
                analysis={},
                user_id="user123"
            )
        
        history = history_service.get_analysis_history(user_id="user123", limit=10)
        
        assert len(history) == 3
    
    def test_get_analysis_by_track_id(self, history_service):
        """Test de obtener análisis por track ID"""
        analysis = {"test": "data"}
        entry_id = history_service.add_analysis(
            track_id="track123",
            track_name="Test Track",
            artists=["Artist"],
            analysis=analysis,
            user_id="user123"
        )
        
        result = history_service.get_analysis_by_track_id("track123", user_id="user123")
        
        assert result is not None
        assert result["track_id"] == "track123"
    
    def test_delete_analysis(self, history_service):
        """Test de eliminar análisis"""
        entry_id = history_service.add_analysis(
            track_id="track123",
            track_name="Test Track",
            artists=["Artist"],
            analysis={},
            user_id="user123"
        )
        
        success = history_service.delete_analysis(entry_id, user_id="user123")
        
        assert success == True
        assert len(history_service.history) == 0


class TestTaggingService:
    """Tests para TaggingService"""
    
    @pytest.fixture
    def tagging_service(self):
        """Fixture para crear TaggingService"""
        from ..services.tagging_service import TaggingService
        return TaggingService()
    
    def test_add_tags(self, tagging_service):
        """Test de agregar tags"""
        if tagging_service is None:
            pytest.skip("TaggingService not available")
        
        result = tagging_service.add_tags("resource123", "track", ["rock", "energetic"])
        
        assert result is not None
    
    def test_get_tags(self, tagging_service):
        """Test de obtener tags"""
        if tagging_service is None:
            pytest.skip("TaggingService not available")
        
        result = tagging_service.get_tags("resource123", "track")
        
        assert result is not None
        assert isinstance(result, dict) or isinstance(result, list)
    
    def test_remove_tags(self, tagging_service):
        """Test de remover tags"""
        if tagging_service is None:
            pytest.skip("TaggingService not available")
        
        result = tagging_service.remove_tags("resource123", "track", ["rock"])
        
        assert result is not None


class TestNotificationService:
    """Tests para NotificationService"""
    
    @pytest.fixture
    def notification_service(self):
        """Fixture para crear NotificationService"""
        from ..services.notification_service import NotificationService
        return NotificationService()
    
    def test_send_notification(self, notification_service):
        """Test de envío de notificación"""
        if notification_service is None:
            pytest.skip("NotificationService not available")
        
        result = notification_service.send_notification(
            user_id="user123",
            message="Test notification",
            type="info"
        )
        
        assert result is not None
    
    def test_get_notifications(self, notification_service):
        """Test de obtener notificaciones"""
        if notification_service is None:
            pytest.skip("NotificationService not available")
        
        result = notification_service.get_notifications("user123", limit=10)
        
        assert result is not None
        assert isinstance(result, list) or isinstance(result, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

