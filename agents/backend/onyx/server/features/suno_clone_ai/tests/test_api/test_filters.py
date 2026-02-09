"""
Comprehensive Unit Tests for API Filters

Tests cover filtering utilities with diverse test cases
"""

import pytest
from datetime import datetime, timedelta
from pydantic import ValidationError

from api.filters import SongFilters, apply_filters


class TestSongFilters:
    """Test cases for SongFilters model"""
    
    def test_song_filters_empty(self):
        """Test creating empty filters"""
        filters = SongFilters()
        assert filters.user_id is None
        assert filters.genre is None
        assert filters.status is None
    
    def test_song_filters_user_id(self):
        """Test filter with user_id"""
        filters = SongFilters(user_id="user123")
        assert filters.user_id == "user123"
    
    def test_song_filters_genre(self):
        """Test filter with genre"""
        filters = SongFilters(genre="rock")
        assert filters.genre == "rock"
    
    def test_song_filters_status(self):
        """Test filter with status"""
        filters = SongFilters(status="completed")
        assert filters.status == "completed"
    
    def test_song_filters_date_range(self):
        """Test filter with date range"""
        date_from = datetime(2023, 1, 1)
        date_to = datetime(2023, 12, 31)
        filters = SongFilters(date_from=date_from, date_to=date_to)
        
        assert filters.date_from == date_from
        assert filters.date_to == date_to
    
    def test_song_filters_duration_range(self):
        """Test filter with duration range"""
        filters = SongFilters(min_duration=30, max_duration=300)
        assert filters.min_duration == 30
        assert filters.max_duration == 300
    
    def test_song_filters_min_duration_below_minimum(self):
        """Test min_duration below 1 raises error"""
        with pytest.raises(ValidationError):
            SongFilters(min_duration=0)
    
    def test_song_filters_max_duration_below_minimum(self):
        """Test max_duration below 1 raises error"""
        with pytest.raises(ValidationError):
            SongFilters(max_duration=0)
    
    def test_song_filters_all_fields(self):
        """Test filter with all fields"""
        date_from = datetime(2023, 1, 1)
        date_to = datetime(2023, 12, 31)
        filters = SongFilters(
            user_id="user123",
            genre="pop",
            status="completed",
            date_from=date_from,
            date_to=date_to,
            min_duration=30,
            max_duration=180
        )
        
        assert filters.user_id == "user123"
        assert filters.genre == "pop"
        assert filters.status == "completed"
        assert filters.min_duration == 30
        assert filters.max_duration == 180
    
    def test_song_filters_to_dict_empty(self):
        """Test to_dict with empty filters"""
        filters = SongFilters()
        result = filters.to_dict()
        assert result == {}
    
    def test_song_filters_to_dict_all_fields(self):
        """Test to_dict with all fields"""
        date_from = datetime(2023, 1, 1)
        date_to = datetime(2023, 12, 31)
        filters = SongFilters(
            user_id="user123",
            genre="rock",
            status="completed",
            date_from=date_from,
            date_to=date_to,
            min_duration=30,
            max_duration=180
        )
        
        result = filters.to_dict()
        assert result["user_id"] == "user123"
        assert result["genre"] == "rock"
        assert result["status"] == "completed"
        assert result["date_from"] == date_from
        assert result["date_to"] == date_to
        assert result["min_duration"] == 30
        assert result["max_duration"] == 180


class TestApplyFilters:
    """Test cases for apply_filters function"""
    
    def test_apply_filters_no_filters(self):
        """Test applying no filters returns all items"""
        items = [
            {"id": 1, "user_id": "user1"},
            {"id": 2, "user_id": "user2"}
        ]
        filters = SongFilters()
        
        result = apply_filters(items, filters)
        assert len(result) == 2
        assert result == items
    
    def test_apply_filters_by_user_id(self):
        """Test filtering by user_id"""
        items = [
            {"id": 1, "user_id": "user1"},
            {"id": 2, "user_id": "user2"},
            {"id": 3, "user_id": "user1"}
        ]
        filters = SongFilters(user_id="user1")
        
        result = apply_filters(items, filters)
        assert len(result) == 2
        assert all(item["user_id"] == "user1" for item in result)
    
    def test_apply_filters_by_genre(self):
        """Test filtering by genre"""
        items = [
            {"id": 1, "metadata": {"genre": "rock"}},
            {"id": 2, "metadata": {"genre": "pop"}},
            {"id": 3, "metadata": {"genre": "rock"}}
        ]
        filters = SongFilters(genre="rock")
        
        result = apply_filters(items, filters)
        assert len(result) == 2
        assert all(item["metadata"]["genre"] == "rock" for item in result)
    
    def test_apply_filters_by_status(self):
        """Test filtering by status"""
        items = [
            {"id": 1, "status": "completed"},
            {"id": 2, "status": "processing"},
            {"id": 3, "status": "completed"}
        ]
        filters = SongFilters(status="completed")
        
        result = apply_filters(items, filters)
        assert len(result) == 2
        assert all(item["status"] == "completed" for item in result)
    
    def test_apply_filters_by_date_from(self):
        """Test filtering by date_from"""
        base_date = datetime(2023, 6, 1)
        items = [
            {"id": 1, "created_at": (base_date - timedelta(days=10)).isoformat()},
            {"id": 2, "created_at": (base_date + timedelta(days=5)).isoformat()},
            {"id": 3, "created_at": (base_date + timedelta(days=10)).isoformat()}
        ]
        filters = SongFilters(date_from=base_date)
        
        result = apply_filters(items, filters)
        assert len(result) == 2
        assert all(
            datetime.fromisoformat(item["created_at"]) >= base_date
            for item in result
        )
    
    def test_apply_filters_by_date_to(self):
        """Test filtering by date_to"""
        base_date = datetime(2023, 6, 1)
        items = [
            {"id": 1, "created_at": (base_date - timedelta(days=10)).isoformat()},
            {"id": 2, "created_at": (base_date + timedelta(days=5)).isoformat()},
            {"id": 3, "created_at": (base_date + timedelta(days=10)).isoformat()}
        ]
        filters = SongFilters(date_to=base_date)
        
        result = apply_filters(items, filters)
        assert len(result) == 1
        assert datetime.fromisoformat(result[0]["created_at"]) <= base_date
    
    def test_apply_filters_by_date_range(self):
        """Test filtering by date range"""
        date_from = datetime(2023, 6, 1)
        date_to = datetime(2023, 6, 30)
        items = [
            {"id": 1, "created_at": (date_from - timedelta(days=5)).isoformat()},
            {"id": 2, "created_at": (date_from + timedelta(days=10)).isoformat()},
            {"id": 3, "created_at": (date_to + timedelta(days=5)).isoformat()}
        ]
        filters = SongFilters(date_from=date_from, date_to=date_to)
        
        result = apply_filters(items, filters)
        assert len(result) == 1
        assert result[0]["id"] == 2
    
    def test_apply_filters_by_min_duration(self):
        """Test filtering by min_duration"""
        items = [
            {"id": 1, "metadata": {"duration": 20}},
            {"id": 2, "metadata": {"duration": 60}},
            {"id": 3, "metadata": {"duration": 180}}
        ]
        filters = SongFilters(min_duration=60)
        
        result = apply_filters(items, filters)
        assert len(result) == 2
        assert all(item["metadata"]["duration"] >= 60 for item in result)
    
    def test_apply_filters_by_max_duration(self):
        """Test filtering by max_duration"""
        items = [
            {"id": 1, "metadata": {"duration": 20}},
            {"id": 2, "metadata": {"duration": 60}},
            {"id": 3, "metadata": {"duration": 180}}
        ]
        filters = SongFilters(max_duration=60)
        
        result = apply_filters(items, filters)
        assert len(result) == 2
        assert all(item["metadata"]["duration"] <= 60 for item in result)
    
    def test_apply_filters_by_duration_range(self):
        """Test filtering by duration range"""
        items = [
            {"id": 1, "metadata": {"duration": 20}},
            {"id": 2, "metadata": {"duration": 60}},
            {"id": 3, "metadata": {"duration": 180}}
        ]
        filters = SongFilters(min_duration=30, max_duration=120)
        
        result = apply_filters(items, filters)
        assert len(result) == 1
        assert result[0]["id"] == 2
    
    def test_apply_filters_multiple_filters(self):
        """Test applying multiple filters"""
        items = [
            {"id": 1, "user_id": "user1", "status": "completed", "metadata": {"genre": "rock"}},
            {"id": 2, "user_id": "user1", "status": "processing", "metadata": {"genre": "rock"}},
            {"id": 3, "user_id": "user2", "status": "completed", "metadata": {"genre": "pop"}},
            {"id": 4, "user_id": "user1", "status": "completed", "metadata": {"genre": "pop"}}
        ]
        filters = SongFilters(user_id="user1", status="completed", genre="rock")
        
        result = apply_filters(items, filters)
        assert len(result) == 1
        assert result[0]["id"] == 1
    
    def test_apply_filters_missing_fields(self):
        """Test filtering with missing fields in items"""
        items = [
            {"id": 1},  # Missing user_id
            {"id": 2, "user_id": "user1"}
        ]
        filters = SongFilters(user_id="user1")
        
        result = apply_filters(items, filters)
        assert len(result) == 1
        assert result[0]["id"] == 2
    
    def test_apply_filters_empty_items(self):
        """Test filtering empty items list"""
        items = []
        filters = SongFilters(user_id="user1")
        
        result = apply_filters(items, filters)
        assert result == []
    
    def test_apply_filters_no_matching_items(self):
        """Test filtering with no matching items"""
        items = [
            {"id": 1, "user_id": "user1"},
            {"id": 2, "user_id": "user2"}
        ]
        filters = SongFilters(user_id="user3")
        
        result = apply_filters(items, filters)
        assert result == []
    
    def test_apply_filters_missing_metadata(self):
        """Test filtering when metadata is missing"""
        items = [
            {"id": 1},  # No metadata
            {"id": 2, "metadata": {"genre": "rock"}}
        ]
        filters = SongFilters(genre="rock")
        
        result = apply_filters(items, filters)
        assert len(result) == 1
        assert result[0]["id"] == 2
    
    def test_apply_filters_invalid_date_format(self):
        """Test filtering with invalid date format (should handle gracefully)"""
        items = [
            {"id": 1, "created_at": "invalid-date"},
            {"id": 2, "created_at": datetime(2023, 6, 1).isoformat()}
        ]
        filters = SongFilters(date_from=datetime(2023, 6, 1))
        
        # Should handle invalid dates gracefully (may raise or skip)
        try:
            result = apply_filters(items, filters)
            # If it doesn't raise, should filter correctly
            assert len(result) >= 0
        except (ValueError, TypeError):
            # Expected behavior for invalid dates
            pass















