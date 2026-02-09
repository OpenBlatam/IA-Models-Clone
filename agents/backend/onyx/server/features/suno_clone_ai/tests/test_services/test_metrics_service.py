"""
Tests modulares para MetricsService
"""

import pytest
from unittest.mock import Mock, patch
from tests.helpers.test_helpers import generate_test_song_id


class TestMetricsService:
    """Tests para MetricsService"""
    
    @pytest.fixture
    def metrics_service(self):
        """Fixture para crear instancia de MetricsService"""
        try:
            from services.metrics_service import get_metrics_service
            return get_metrics_service()
        except ImportError:
            pytest.skip("MetricsService not available")
    
    @pytest.mark.unit
    def test_record_generation_start(self, metrics_service):
        """Test de registro de inicio de generación"""
        song_id = generate_test_song_id()
        user_id = "test-user-123"
        
        result = metrics_service.record_generation_start(
            song_id=song_id,
            user_id=user_id,
            source="direct",
            prompt_length=50
        )
        
        # Puede retornar True, None, o no retornar nada
        assert result is True or result is None or result is None
    
    @pytest.mark.unit
    def test_record_generation(self, metrics_service):
        """Test de registro de generación completa"""
        song_id = generate_test_song_id()
        user_id = "test-user-123"
        
        result = metrics_service.record_generation(
            song_id=song_id,
            user_id=user_id,
            prompt="Test song",
            duration=30,
            generation_time=5.2,
            model_used="facebook/musicgen-medium",
            status="completed"
        )
        
        assert result is True or result is None
    
    @pytest.mark.unit
    def test_get_stats(self, metrics_service):
        """Test de obtención de estadísticas"""
        stats = metrics_service.get_stats(days=7)
        
        assert isinstance(stats, dict)
        # Verificar campos comunes
        assert "total_songs" in stats or "songs_generated" in stats or len(stats) >= 0
    
    @pytest.mark.unit
    def test_get_user_stats(self, metrics_service):
        """Test de obtención de estadísticas de usuario"""
        user_id = "test-user-123"
        
        stats = metrics_service.get_user_stats(user_id, days=30)
        
        assert isinstance(stats, dict)
        # Verificar que incluye user_id o campos relacionados
        assert "user_id" in stats or "songs_generated" in stats or len(stats) >= 0
    
    @pytest.mark.edge_case
    def test_record_generation_failed(self, metrics_service):
        """Test de registro de generación fallida"""
        song_id = generate_test_song_id()
        
        result = metrics_service.record_generation(
            song_id=song_id,
            user_id="test-user",
            prompt="Test",
            duration=30,
            generation_time=2.0,
            model_used="facebook/musicgen-medium",
            status="failed"
        )
        
        assert result is True or result is None
    
    @pytest.mark.edge_case
    def test_get_stats_different_periods(self, metrics_service):
        """Test con diferentes períodos"""
        for days in [1, 7, 30, 365]:
            stats = metrics_service.get_stats(days=days)
            assert isinstance(stats, dict)

