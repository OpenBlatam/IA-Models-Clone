"""
Comprehensive Unit Tests for Chat Processor

Tests cover chat processing functions with diverse test cases
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from core.chat_processor import ChatProcessor, get_chat_processor


class TestChatProcessorExtractGenre:
    """Test cases for _extract_genre method"""
    
    def test_extract_genre_rock(self):
        """Test extracting rock genre"""
        processor = ChatProcessor()
        result = processor._extract_genre("Create a rock song")
        assert result == "rock"
    
    def test_extract_genre_pop(self):
        """Test extracting pop genre"""
        processor = ChatProcessor()
        result = processor._extract_genre("I want pop music")
        assert result == "pop"
    
    def test_extract_genre_case_insensitive(self):
        """Test genre extraction is case insensitive"""
        processor = ChatProcessor()
        result = processor._extract_genre("JAZZ music please")
        assert result == "jazz"
    
    def test_extract_genre_not_found(self):
        """Test when genre is not found"""
        processor = ChatProcessor()
        result = processor._extract_genre("Create some music")
        assert result is None
    
    def test_extract_genre_multiple_genres(self):
        """Test when multiple genres mentioned (returns first)"""
        processor = ChatProcessor()
        result = processor._extract_genre("rock and pop song")
        # Should return first match
        assert result in ["rock", "pop"]


class TestChatProcessorExtractMood:
    """Test cases for _extract_mood method"""
    
    def test_extract_mood_happy(self):
        """Test extracting happy mood"""
        processor = ChatProcessor()
        result = processor._extract_mood("Create a happy song")
        assert result == "happy"
    
    def test_extract_mood_energetic(self):
        """Test extracting energetic mood"""
        processor = ChatProcessor()
        result = processor._extract_mood("I want energetic music")
        assert result == "energetic"
    
    def test_extract_mood_not_found(self):
        """Test when mood is not found"""
        processor = ChatProcessor()
        result = processor._extract_mood("Create some music")
        assert result is None
    
    def test_extract_mood_case_insensitive(self):
        """Test mood extraction is case insensitive"""
        processor = ChatProcessor()
        result = processor._extract_mood("SAD music")
        assert result == "sad"


class TestChatProcessorExtractTempo:
    """Test cases for _extract_tempo method"""
    
    def test_extract_tempo_bpm(self):
        """Test extracting BPM tempo"""
        processor = ChatProcessor()
        result = processor._extract_tempo("120 bpm song")
        assert result == "120"
    
    def test_extract_tempo_slow(self):
        """Test extracting slow tempo"""
        processor = ChatProcessor()
        result = processor._extract_tempo("slow tempo music")
        assert result == "slow"
    
    def test_extract_tempo_fast(self):
        """Test extracting fast tempo"""
        processor = ChatProcessor()
        result = processor._extract_tempo("fast tempo")
        assert result == "fast"
    
    def test_extract_tempo_not_found(self):
        """Test when tempo is not found"""
        processor = ChatProcessor()
        result = processor._extract_tempo("Create some music")
        assert result is None


class TestChatProcessorExtractInstruments:
    """Test cases for _extract_instruments method"""
    
    def test_extract_instruments_single(self):
        """Test extracting single instrument"""
        processor = ChatProcessor()
        result = processor._extract_instruments("song with guitar")
        assert "guitar" in result
    
    def test_extract_instruments_multiple(self):
        """Test extracting multiple instruments"""
        processor = ChatProcessor()
        result = processor._extract_instruments("piano and drums song")
        assert "piano" in result
        assert "drums" in result
    
    def test_extract_instruments_not_found(self):
        """Test when no instruments found"""
        processor = ChatProcessor()
        result = processor._extract_instruments("Create some music")
        assert result == []
    
    def test_extract_instruments_case_insensitive(self):
        """Test instrument extraction is case insensitive"""
        processor = ChatProcessor()
        result = processor._extract_instruments("PIANO music")
        assert "piano" in result


class TestChatProcessorExtractDuration:
    """Test cases for _extract_duration method"""
    
    def test_extract_duration_seconds(self):
        """Test extracting duration in seconds"""
        processor = ChatProcessor()
        result = processor._extract_duration("30 seconds song")
        assert result == 30
    
    def test_extract_duration_minutes(self):
        """Test extracting duration in minutes"""
        processor = ChatProcessor()
        result = processor._extract_duration("2 minutes")
        assert result == 120
    
    def test_extract_duration_not_found(self):
        """Test when duration is not found"""
        processor = ChatProcessor()
        result = processor._extract_duration("Create some music")
        assert result is None
    
    def test_extract_duration_abbreviated(self):
        """Test extracting abbreviated duration"""
        processor = ChatProcessor()
        result = processor._extract_duration("60 sec")
        assert result == 60


class TestChatProcessorCreateBasicPrompt:
    """Test cases for _create_basic_prompt method"""
    
    def test_create_basic_prompt_simple(self):
        """Test creating basic prompt with simple text"""
        processor = ChatProcessor()
        result = processor._create_basic_prompt("Create a song", None, None, None, [])
        assert "Create a song" in result
    
    def test_create_basic_prompt_with_genre(self):
        """Test creating prompt with genre"""
        processor = ChatProcessor()
        result = processor._create_basic_prompt("Create a song", "rock", None, None, [])
        assert "rock" in result
        assert "Genre" in result
    
    def test_create_basic_prompt_with_mood(self):
        """Test creating prompt with mood"""
        processor = ChatProcessor()
        result = processor._create_basic_prompt("Create a song", None, "happy", None, [])
        assert "happy" in result
        assert "Mood" in result
    
    def test_create_basic_prompt_with_all_fields(self):
        """Test creating prompt with all fields"""
        processor = ChatProcessor()
        result = processor._create_basic_prompt(
            "Create a song", "rock", "happy", "120 bpm", ["guitar", "drums"]
        )
        assert "rock" in result
        assert "happy" in result
        assert "120 bpm" in result
        assert "guitar" in result


class TestChatProcessorExtractSongInfo:
    """Test cases for extract_song_info method"""
    
    def test_extract_song_info_basic(self):
        """Test extracting song info from basic message"""
        processor = ChatProcessor()
        result = processor.extract_song_info("Create a rock song")
        
        assert "prompt" in result
        assert "genre" in result
        assert "mood" in result
        assert "original_message" in result
        assert result["original_message"] == "Create a rock song"
    
    def test_extract_song_info_with_all_details(self):
        """Test extracting song info with all details"""
        processor = ChatProcessor()
        message = "Create a happy rock song with guitar and drums, 120 bpm, 2 minutes"
        result = processor.extract_song_info(message)
        
        assert result["genre"] == "rock"
        assert result["mood"] == "happy"
        assert "guitar" in result["instruments"]
        assert result["duration"] == 120
    
    def test_extract_song_info_with_chat_history(self):
        """Test extracting song info with chat history"""
        processor = ChatProcessor()
        chat_history = [
            {"role": "user", "content": "I like rock music"},
            {"role": "assistant", "content": "I can help with that"}
        ]
        result = processor.extract_song_info("Create a song", chat_history=chat_history)
        
        assert "prompt" in result
        assert result["original_message"] == "Create a song"
    
    def test_extract_song_info_error_handling(self):
        """Test error handling in extract_song_info"""
        processor = ChatProcessor()
        # Should handle errors gracefully
        result = processor.extract_song_info("")
        
        assert "prompt" in result
        assert result["original_message"] == ""
    
    @patch('core.chat_processor.OpenAI')
    def test_extract_song_info_with_openai(self, mock_openai):
        """Test extracting song info with OpenAI available"""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Enhanced prompt"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        processor = ChatProcessor()
        processor.client = mock_client
        
        result = processor.extract_song_info("Create a song")
        
        assert result["prompt"] == "Enhanced prompt"
        mock_client.chat.completions.create.assert_called_once()


class TestGetChatProcessor:
    """Test cases for get_chat_processor function"""
    
    def test_get_chat_processor_singleton(self):
        """Test that get_chat_processor returns singleton"""
        processor1 = get_chat_processor()
        processor2 = get_chat_processor()
        
        assert processor1 is processor2
        assert isinstance(processor1, ChatProcessor)
    
    def test_get_chat_processor_multiple_calls(self):
        """Test multiple calls return same instance"""
        processors = [get_chat_processor() for _ in range(5)]
        assert all(p is processors[0] for p in processors)















