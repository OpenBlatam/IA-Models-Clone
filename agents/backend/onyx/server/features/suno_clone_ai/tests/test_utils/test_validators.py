"""
Comprehensive Unit Tests for Utils Validators

Tests cover InputValidators class with diverse test cases
"""

import pytest
import re

from utils.validators import InputValidators


class TestInputValidatorsPrompt:
    """Test cases for validate_prompt function"""
    
    def test_validate_prompt_valid(self):
        """Test validating valid prompt"""
        result = InputValidators.validate_prompt("Create a happy pop song")
        assert result == "Create a happy pop song"
    
    def test_validate_prompt_empty(self):
        """Test validating empty prompt raises error"""
        with pytest.raises(ValueError, match="cannot be empty"):
            InputValidators.validate_prompt("")
    
    def test_validate_prompt_whitespace_only(self):
        """Test validating whitespace-only prompt"""
        with pytest.raises(ValueError, match="cannot be empty"):
            InputValidators.validate_prompt("   ")
    
    def test_validate_prompt_too_long(self):
        """Test validating prompt that exceeds max length"""
        long_prompt = "A" * 501
        with pytest.raises(ValueError, match="too long"):
            InputValidators.validate_prompt(long_prompt, max_length=500)
    
    def test_validate_prompt_removes_dangerous_chars(self):
        """Test that dangerous characters are removed"""
        prompt = 'Test<script>alert("xss")</script>'
        result = InputValidators.validate_prompt(prompt)
        assert "<" not in result
        assert ">" not in result
        assert '"' not in result
    
    def test_validate_prompt_strips_whitespace(self):
        """Test that whitespace is stripped"""
        prompt = "  Create a song  "
        result = InputValidators.validate_prompt(prompt)
        assert result == "Create a song"
    
    def test_validate_prompt_at_max_length(self):
        """Test prompt at maximum length"""
        max_prompt = "A" * 500
        result = InputValidators.validate_prompt(max_prompt, max_length=500)
        assert len(result) == 500


class TestInputValidatorsDuration:
    """Test cases for validate_duration function"""
    
    def test_validate_duration_valid(self):
        """Test validating valid duration"""
        assert InputValidators.validate_duration(30) == 30
        assert InputValidators.validate_duration(180) == 180
    
    def test_validate_duration_none_returns_default(self):
        """Test None duration returns default"""
        assert InputValidators.validate_duration(None) == 30
    
    def test_validate_duration_below_minimum(self):
        """Test duration below minimum raises error"""
        with pytest.raises(ValueError, match="at least 1 second"):
            InputValidators.validate_duration(0)
        
        with pytest.raises(ValueError, match="at least 1 second"):
            InputValidators.validate_duration(-10)
    
    def test_validate_duration_above_maximum(self):
        """Test duration above maximum raises error"""
        with pytest.raises(ValueError, match="cannot exceed"):
            InputValidators.validate_duration(301, max_duration=300)
    
    def test_validate_duration_at_maximum(self):
        """Test duration at maximum boundary"""
        assert InputValidators.validate_duration(300, max_duration=300) == 300
    
    def test_validate_duration_custom_max(self):
        """Test with custom maximum duration"""
        assert InputValidators.validate_duration(60, max_duration=120) == 60


class TestInputValidatorsGenre:
    """Test cases for validate_genre function"""
    
    def test_validate_genre_valid_rock(self):
        """Test validating valid rock genre"""
        assert InputValidators.validate_genre("rock") == "rock"
    
    def test_validate_genre_valid_pop(self):
        """Test validating valid pop genre"""
        assert InputValidators.validate_genre("pop") == "pop"
    
    def test_validate_genre_case_insensitive(self):
        """Test genre validation is case insensitive"""
        assert InputValidators.validate_genre("ROCK") == "rock"
        assert InputValidators.validate_genre("Pop") == "pop"
    
    def test_validate_genre_none(self):
        """Test None genre returns None"""
        assert InputValidators.validate_genre(None) is None
    
    def test_validate_genre_strips_whitespace(self):
        """Test genre strips whitespace"""
        assert InputValidators.validate_genre("  rock  ") == "rock"
    
    def test_validate_genre_unknown_allowed(self):
        """Test unknown genre is allowed but lowercased"""
        result = InputValidators.validate_genre("UnknownGenre")
        assert result == "unknowngenre"
    
    def test_validate_genre_all_valid_genres(self):
        """Test all valid genres"""
        valid_genres = [
            "rock", "pop", "jazz", "classical", "electronic", "hip hop", "rap",
            "country", "blues", "reggae", "metal", "folk", "latin", "r&b",
            "soul", "funk", "disco", "techno", "house", "ambient", "indie"
        ]
        for genre in valid_genres:
            result = InputValidators.validate_genre(genre)
            assert result == genre.lower()


class TestInputValidatorsSongIds:
    """Test cases for validate_song_ids function"""
    
    def test_validate_song_ids_valid(self):
        """Test validating valid song IDs"""
        valid_ids = [
            "12345678-1234-1234-1234-123456789012",
            "abcdefab-cdef-cdef-cdef-abcdefabcdef"
        ]
        result = InputValidators.validate_song_ids(valid_ids)
        assert result == valid_ids
    
    def test_validate_song_ids_empty_list(self):
        """Test empty list raises error"""
        with pytest.raises(ValueError, match="At least one song ID"):
            InputValidators.validate_song_ids([])
    
    def test_validate_song_ids_exceeds_max(self):
        """Test too many song IDs raises error"""
        many_ids = [f"12345678-1234-1234-1234-{i:012d}" for i in range(11)]
        with pytest.raises(ValueError, match="Cannot mix more than"):
            InputValidators.validate_song_ids(many_ids, max_count=10)
    
    def test_validate_song_ids_invalid_format(self):
        """Test invalid UUID format raises error"""
        invalid_ids = ["not-a-uuid", "12345"]
        with pytest.raises(ValueError, match="Invalid song ID format"):
            InputValidators.validate_song_ids(invalid_ids)
    
    def test_validate_song_ids_mixed_valid_invalid(self):
        """Test mixed valid and invalid IDs"""
        mixed_ids = [
            "12345678-1234-1234-1234-123456789012",
            "invalid-id"
        ]
        with pytest.raises(ValueError, match="Invalid song ID format"):
            InputValidators.validate_song_ids(mixed_ids)
    
    def test_validate_song_ids_at_max_count(self):
        """Test at maximum count"""
        max_ids = [f"12345678-1234-1234-1234-{i:012d}" for i in range(10)]
        result = InputValidators.validate_song_ids(max_ids, max_count=10)
        assert len(result) == 10


class TestInputValidatorsVolumes:
    """Test cases for validate_volumes function"""
    
    def test_validate_volumes_valid(self):
        """Test validating valid volumes"""
        volumes = [0.5, 1.0, 1.5]
        result = InputValidators.validate_volumes(volumes, count=3)
        assert result == volumes
    
    def test_validate_volumes_none(self):
        """Test None volumes returns None"""
        assert InputValidators.validate_volumes(None, count=3) is None
    
    def test_validate_volumes_wrong_count(self):
        """Test volumes count mismatch raises error"""
        volumes = [0.5, 1.0]
        with pytest.raises(ValueError, match="must match number of songs"):
            InputValidators.validate_volumes(volumes, count=3)
    
    def test_validate_volumes_negative(self):
        """Test negative volume raises error"""
        volumes = [-0.5, 1.0]
        with pytest.raises(ValueError, match="must be between 0 and 2.0"):
            InputValidators.validate_volumes(volumes, count=2)
    
    def test_validate_volumes_above_maximum(self):
        """Test volume above 2.0 raises error"""
        volumes = [1.0, 2.5]
        with pytest.raises(ValueError, match="must be between 0 and 2.0"):
            InputValidators.validate_volumes(volumes, count=2)
    
    def test_validate_volumes_at_boundaries(self):
        """Test volumes at boundaries"""
        volumes = [0.0, 2.0]
        result = InputValidators.validate_volumes(volumes, count=2)
        assert result == volumes
    
    def test_validate_volumes_zero(self):
        """Test zero volume is valid"""
        volumes = [0.0, 1.0]
        result = InputValidators.validate_volumes(volumes, count=2)
        assert result == volumes


class TestInputValidatorsFadeTime:
    """Test cases for validate_fade_time function"""
    
    def test_validate_fade_time_valid(self):
        """Test validating valid fade time"""
        assert InputValidators.validate_fade_time(1.0) == 1.0
        assert InputValidators.validate_fade_time(5.5) == 5.5
    
    def test_validate_fade_time_none(self):
        """Test None fade time returns None"""
        assert InputValidators.validate_fade_time(None) is None
    
    def test_validate_fade_time_negative(self):
        """Test negative fade time raises error"""
        with pytest.raises(ValueError, match="cannot be negative"):
            InputValidators.validate_fade_time(-1.0)
    
    def test_validate_fade_time_above_maximum(self):
        """Test fade time above maximum raises error"""
        with pytest.raises(ValueError, match="cannot exceed"):
            InputValidators.validate_fade_time(11.0, max_time=10.0)
    
    def test_validate_fade_time_at_maximum(self):
        """Test fade time at maximum boundary"""
        assert InputValidators.validate_fade_time(10.0, max_time=10.0) == 10.0
    
    def test_validate_fade_time_zero(self):
        """Test zero fade time is valid"""
        assert InputValidators.validate_fade_time(0.0) == 0.0
    
    def test_validate_fade_time_custom_max(self):
        """Test with custom maximum time"""
        assert InputValidators.validate_fade_time(5.0, max_time=15.0) == 5.0















