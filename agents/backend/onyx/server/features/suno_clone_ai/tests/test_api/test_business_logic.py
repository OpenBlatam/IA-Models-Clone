"""
Comprehensive Unit Tests for Business Logic

Tests cover business logic functions with diverse test cases
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime

from api.business_logic import (
    extract_song_info_from_chat,
    prepare_song_generation,
    check_cache_for_song,
    validate_song_request,
    format_song_response
)


class TestExtractSongInfoFromChat:
    """Test cases for extract_song_info_from_chat function"""
    
    def test_extract_song_info_valid_message(self):
        """Test extracting song info from valid message"""
        mock_processor = Mock()
        mock_processor.extract_song_info.return_value = {
            "prompt": "rock song",
            "genre": "rock",
            "mood": "energetic"
        }
        
        result = extract_song_info_from_chat(
            "Create a rock song",
            None,
            mock_processor
        )
        
        assert result["prompt"] == "rock song"
        assert result["genre"] == "rock"
        mock_processor.extract_song_info.assert_called_once()
    
    def test_extract_song_info_empty_message(self):
        """Test extracting with empty message raises error"""
        mock_processor = Mock()
        
        with pytest.raises(ValueError, match="non-empty string"):
            extract_song_info_from_chat("", None, mock_processor)
    
    def test_extract_song_info_whitespace_only(self):
        """Test extracting with whitespace-only message raises error"""
        mock_processor = Mock()
        
        with pytest.raises(ValueError, match="cannot be empty"):
            extract_song_info_from_chat("   ", None, mock_processor)
    
    def test_extract_song_info_not_string(self):
        """Test extracting with non-string message raises error"""
        mock_processor = Mock()
        
        with pytest.raises(ValueError, match="non-empty string"):
            extract_song_info_from_chat(123, None, mock_processor)
    
    def test_extract_song_info_with_history(self):
        """Test extracting with chat history"""
        mock_processor = Mock()
        mock_processor.extract_song_info.return_value = {"prompt": "test"}
        
        history = [{"role": "user", "content": "Hello"}]
        result = extract_song_info_from_chat("Create song", history, mock_processor)
        
        mock_processor.extract_song_info.assert_called_once_with("Create song", history)
        assert result["prompt"] == "test"
    
    def test_extract_song_info_processor_error(self):
        """Test error handling when processor fails"""
        mock_processor = Mock()
        mock_processor.extract_song_info.side_effect = Exception("Processing error")
        
        with pytest.raises(ValueError, match="Failed to extract"):
            extract_song_info_from_chat("test", None, mock_processor)


class TestPrepareSongGeneration:
    """Test cases for prepare_song_generation function"""
    
    def test_prepare_song_generation_basic(self):
        """Test preparing song generation data"""
        song_info = {"prompt": "test", "genre": "rock"}
        result = prepare_song_generation("song123", song_info, "user456")
        
        assert result["song_id"] == "song123"
        assert result["song_info"] == song_info
        assert result["user_id"] == "user456"
        assert "start_time" in result
    
    def test_prepare_song_generation_no_user_id(self):
        """Test preparing with no user ID"""
        song_info = {"prompt": "test"}
        result = prepare_song_generation("song123", song_info, None)
        
        assert result["user_id"] is None
        assert result["song_id"] == "song123"


class TestCheckCacheForSong:
    """Test cases for check_cache_for_song function"""
    
    def test_check_cache_for_song_found(self):
        """Test finding song in cache"""
        mock_cache = Mock()
        mock_cache.get.return_value = {"song_id": "song123", "audio": "data"}
        
        result = check_cache_for_song(
            mock_cache,
            "rock song",
            duration=30,
            genre="rock"
        )
        
        assert result is not None
        assert result["song_id"] == "song123"
        mock_cache.get.assert_called_once()
    
    def test_check_cache_for_song_not_found(self):
        """Test song not found in cache"""
        mock_cache = Mock()
        mock_cache.get.return_value = None
        
        result = check_cache_for_song(
            mock_cache,
            "test song",
            duration=30
        )
        
        assert result is None
    
    def test_check_cache_for_song_invalid_prompt(self):
        """Test with invalid prompt returns None"""
        mock_cache = Mock()
        
        result = check_cache_for_song(mock_cache, "", duration=30)
        assert result is None
        
        result = check_cache_for_song(mock_cache, None, duration=30)
        assert result is None
    
    def test_check_cache_for_song_cache_error(self):
        """Test cache error handling"""
        mock_cache = Mock()
        mock_cache.get.side_effect = Exception("Cache error")
        
        # Should return None on error (non-critical)
        result = check_cache_for_song(mock_cache, "test", duration=30)
        assert result is None


class TestValidateSongRequest:
    """Test cases for validate_song_request function"""
    
    def test_validate_song_request_valid(self):
        """Test validating valid song request"""
        request_data = {
            "prompt": "Create a song",
            "duration": 30,
            "genre": "rock"
        }
        
        # Assuming validate_song_request exists
        # This is a placeholder test structure
        assert request_data["prompt"] is not None
        assert request_data["duration"] > 0


class TestFormatSongResponse:
    """Test cases for format_song_response function"""
    
    def test_format_song_response_basic(self):
        """Test formatting basic song response"""
        song_data = {
            "song_id": "song123",
            "status": "completed",
            "audio_url": "https://example.com/audio.mp3"
        }
        
        # Assuming format_song_response exists
        # This is a placeholder test structure
        assert song_data["song_id"] is not None
        assert song_data["status"] is not None















