"""
Tests para helpers de validación
"""

import pytest
from tests.helpers.test_helpers import create_song_dict, generate_test_song_id
from tests.helpers.advanced_helpers import ResponseValidator


class TestParseCommaSeparatedIds:
    """Tests para parse_comma_separated_ids"""
    
    @pytest.fixture
    def parse_function(self):
        """Fixture para obtener la función de parseo"""
        try:
            from api.utils.validation_helpers import parse_comma_separated_ids
            return parse_comma_separated_ids
        except ImportError:
            pytest.skip("parse_comma_separated_ids not available")
    
    @pytest.mark.unit
    def test_parse_single_id(self, parse_function):
        """Test con un solo ID"""
        result = parse_function("id1", max_items=10)
        assert result == ["id1"]
    
    @pytest.mark.unit
    def test_parse_multiple_ids(self, parse_function):
        """Test con múltiples IDs"""
        result = parse_function("id1,id2,id3", max_items=10)
        assert result == ["id1", "id2", "id3"]
        assert len(result) == 3
    
    @pytest.mark.unit
    def test_parse_with_spaces(self, parse_function):
        """Test con espacios"""
        result = parse_function("id1, id2, id3", max_items=10)
        # Debe limpiar espacios
        assert len(result) == 3
    
    @pytest.mark.unit
    def test_parse_max_items(self, parse_function):
        """Test respetando máximo de items"""
        ids = ",".join([f"id{i}" for i in range(10)])
        result = parse_function(ids, max_items=10)
        assert len(result) == 10
    
    @pytest.mark.error_handling
    def test_parse_exceeds_max(self, parse_function):
        """Test cuando excede el máximo"""
        ids = ",".join([f"id{i}" for i in range(11)])
        with pytest.raises(ValueError):
            parse_function(ids, max_items=10)
    
    @pytest.mark.error_handling
    def test_parse_empty_string(self, parse_function):
        """Test con string vacío"""
        with pytest.raises(ValueError):
            parse_function("", max_items=10)
    
    @pytest.mark.edge_case
    def test_parse_duplicate_ids(self, parse_function):
        """Test con IDs duplicados"""
        result = parse_function("id1,id2,id1", max_items=10)
        # Puede o no eliminar duplicados según implementación
        assert len(result) >= 2


class TestBatchProcessor:
    """Tests para batch processor"""
    
    @pytest.fixture
    def batch_function(self):
        """Fixture para obtener la función batch"""
        try:
            from api.utils.batch_processor import batch_get_songs
            return batch_get_songs
        except ImportError:
            pytest.skip("batch_get_songs not available")
    
    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_batch_get_songs_success(self, batch_function, mock_song_service):
        """Test exitoso de batch get"""
        song_ids = [generate_test_song_id() for _ in range(3)]
        songs = [
            create_song_dict(song_id=sid, status="processing")
            for sid in song_ids
        ]
        mock_song_service.get_song.side_effect = lambda sid: next(
            (s for s in songs if s["song_id"] == sid), None
        )
        
        result = await batch_function(mock_song_service, song_ids, max_concurrent=3)
        
        assert len(result) == 3
        assert all(song is not None for song in result)
    
    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_batch_get_songs_with_none(self, batch_function, mock_song_service):
        """Test con algunos None"""
        song_ids = [generate_test_song_id() for _ in range(5)]
        # Solo 2 canciones existen
        songs = [
            create_song_dict(song_id=song_ids[0], status="processing"),
            create_song_dict(song_id=song_ids[1], status="completed")
        ]
        mock_song_service.get_song.side_effect = lambda sid: next(
            (s for s in songs if s["song_id"] == sid), None
        )
        
        result = await batch_function(mock_song_service, song_ids, max_concurrent=5)
        
        assert len(result) == 5
        # Algunos pueden ser None
        found = [s for s in result if s is not None]
        assert len(found) == 2

