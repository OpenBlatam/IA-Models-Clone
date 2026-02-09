"""
Tests para utilidades de filtrado
"""

import pytest
from datetime import datetime, timedelta
from api.filters import SongFilters, apply_filters


@pytest.mark.unit
@pytest.mark.api
class TestSongFilters:
    """Tests para SongFilters"""
    
    def test_song_filters_empty(self):
        """Test de filtros vacíos"""
        filters = SongFilters()
        
        assert filters.user_id is None
        assert filters.genre is None
        assert filters.status is None
    
    def test_song_filters_with_values(self):
        """Test de filtros con valores"""
        filters = SongFilters(
            user_id="user-123",
            genre="pop",
            status="completed"
        )
        
        assert filters.user_id == "user-123"
        assert filters.genre == "pop"
        assert filters.status == "completed"
    
    def test_song_filters_with_dates(self):
        """Test de filtros con fechas"""
        date_from = datetime.now() - timedelta(days=7)
        date_to = datetime.now()
        
        filters = SongFilters(
            date_from=date_from,
            date_to=date_to
        )
        
        assert filters.date_from == date_from
        assert filters.date_to == date_to
    
    def test_song_filters_with_duration(self):
        """Test de filtros con duración"""
        filters = SongFilters(
            min_duration=30,
            max_duration=300
        )
        
        assert filters.min_duration == 30
        assert filters.max_duration == 300
    
    def test_song_filters_to_dict(self):
        """Test de conversión a diccionario"""
        filters = SongFilters(
            user_id="user-123",
            genre="pop",
            status="completed"
        )
        
        result = filters.to_dict()
        
        assert result["user_id"] == "user-123"
        assert result["genre"] == "pop"
        assert result["status"] == "completed"
    
    def test_song_filters_to_dict_empty(self):
        """Test de conversión a diccionario vacío"""
        filters = SongFilters()
        result = filters.to_dict()
        
        assert result == {}


@pytest.mark.unit
@pytest.mark.api
class TestApplyFilters:
    """Tests para apply_filters"""
    
    @pytest.fixture
    def sample_items(self):
        """Items de ejemplo"""
        return [
            {
                "id": "1",
                "user_id": "user-1",
                "status": "completed",
                "created_at": "2024-01-01T00:00:00",
                "metadata": {"genre": "pop", "duration": 180}
            },
            {
                "id": "2",
                "user_id": "user-2",
                "status": "pending",
                "created_at": "2024-01-02T00:00:00",
                "metadata": {"genre": "rock", "duration": 240}
            },
            {
                "id": "3",
                "user_id": "user-1",
                "status": "completed",
                "created_at": "2024-01-03T00:00:00",
                "metadata": {"genre": "pop", "duration": 120}
            }
        ]
    
    def test_apply_filters_by_user_id(self, sample_items):
        """Test de filtrado por user_id"""
        filters = SongFilters(user_id="user-1")
        result = apply_filters(sample_items, filters)
        
        assert len(result) == 2
        assert all(item["user_id"] == "user-1" for item in result)
    
    def test_apply_filters_by_genre(self, sample_items):
        """Test de filtrado por género"""
        filters = SongFilters(genre="pop")
        result = apply_filters(sample_items, filters)
        
        assert len(result) == 2
        assert all(item["metadata"]["genre"] == "pop" for item in result)
    
    def test_apply_filters_by_status(self, sample_items):
        """Test de filtrado por estado"""
        filters = SongFilters(status="completed")
        result = apply_filters(sample_items, filters)
        
        assert len(result) == 2
        assert all(item["status"] == "completed" for item in result)
    
    def test_apply_filters_by_date_from(self, sample_items):
        """Test de filtrado por fecha desde"""
        date_from = datetime.fromisoformat("2024-01-02T00:00:00")
        filters = SongFilters(date_from=date_from)
        result = apply_filters(sample_items, filters)
        
        assert len(result) == 2
    
    def test_apply_filters_by_date_to(self, sample_items):
        """Test de filtrado por fecha hasta"""
        date_to = datetime.fromisoformat("2024-01-02T00:00:00")
        filters = SongFilters(date_to=date_to)
        result = apply_filters(sample_items, filters)
        
        assert len(result) == 2
    
    def test_apply_filters_by_min_duration(self, sample_items):
        """Test de filtrado por duración mínima"""
        filters = SongFilters(min_duration=180)
        result = apply_filters(sample_items, filters)
        
        assert len(result) == 2
        assert all(item["metadata"]["duration"] >= 180 for item in result)
    
    def test_apply_filters_by_max_duration(self, sample_items):
        """Test de filtrado por duración máxima"""
        filters = SongFilters(max_duration=180)
        result = apply_filters(sample_items, filters)
        
        assert len(result) == 2
        assert all(item["metadata"]["duration"] <= 180 for item in result)
    
    def test_apply_filters_multiple(self, sample_items):
        """Test de filtrado con múltiples filtros"""
        filters = SongFilters(
            user_id="user-1",
            genre="pop",
            status="completed"
        )
        result = apply_filters(sample_items, filters)
        
        assert len(result) == 1
        assert result[0]["id"] == "1"
    
    def test_apply_filters_no_matches(self, sample_items):
        """Test de filtrado sin coincidencias"""
        filters = SongFilters(user_id="user-999")
        result = apply_filters(sample_items, filters)
        
        assert len(result) == 0
    
    def test_apply_filters_empty_list(self):
        """Test de filtrado con lista vacía"""
        filters = SongFilters(user_id="user-1")
        result = apply_filters([], filters)
        
        assert len(result) == 0



