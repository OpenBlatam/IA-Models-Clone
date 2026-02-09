"""
Tests refactorizados para utilidades de filtrado
Usando clases base y helpers para eliminar duplicación
"""

import pytest
from datetime import datetime, timedelta
from api.filters import SongFilters, apply_filters
from test_helpers import BaseServiceTestCase, StandardTestMixin


class TestSongFiltersRefactored(BaseServiceTestCase, StandardTestMixin):
    """Tests refactorizados para SongFilters"""
    
    def test_song_filters_empty(self):
        """Test de filtros vacíos"""
        filters = SongFilters()
        
        assert filters.user_id is None
        assert filters.genre is None
        assert filters.status is None
    
    @pytest.mark.parametrize("user_id,genre,status", [
        ("user-123", "pop", "completed"),
        ("user-456", "rock", "pending"),
        (None, "jazz", None)
    ])
    def test_song_filters_with_values(self, user_id, genre, status):
        """Test de filtros con diferentes valores"""
        filters = SongFilters(
            user_id=user_id,
            genre=genre,
            status=status
        )
        
        assert filters.user_id == user_id
        assert filters.genre == genre
        assert filters.status == status
    
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


class TestApplyFiltersRefactored(BaseServiceTestCase, StandardTestMixin):
    """Tests refactorizados para apply_filters"""
    
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
    
    @pytest.mark.parametrize("filter_value,filter_key,expected_count", [
        ("user-1", "user_id", 2),
        ("user-2", "user_id", 1),
        ("pop", "genre", 2),
        ("rock", "genre", 1),
        ("completed", "status", 2),
        ("pending", "status", 1)
    ])
    def test_apply_filters_single(self, sample_items, filter_value, filter_key, expected_count):
        """Test de filtrado con un solo filtro"""
        filters_dict = {filter_key: filter_value}
        filters = SongFilters(**filters_dict)
        result = apply_filters(sample_items, filters)
        
        assert len(result) == expected_count
    
    def test_apply_filters_by_date_from(self, sample_items):
        """Test de filtrado por fecha desde"""
        date_from = datetime.fromisoformat("2024-01-02T00:00:00")
        filters = SongFilters(date_from=date_from)
        result = apply_filters(sample_items, filters)
        
        assert len(result) == 2
    
    def test_apply_filters_by_duration_range(self, sample_items):
        """Test de filtrado por rango de duración"""
        filters = SongFilters(min_duration=150, max_duration=200)
        result = apply_filters(sample_items, filters)
        
        assert len(result) == 1
        assert result[0]["id"] == "1"
    
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



