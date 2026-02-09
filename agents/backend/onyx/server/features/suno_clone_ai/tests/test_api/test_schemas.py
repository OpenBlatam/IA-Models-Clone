"""
Comprehensive Unit Tests for API Schemas

Tests cover Pydantic schema validation with diverse test cases
"""

import pytest
from pydantic import ValidationError

from api.schemas import (
    ChatMessage,
    SongGenerationRequest,
    SongResponse,
    AudioEditRequest,
    AudioMixRequest,
    SongListResponse,
    SongAnalysisResponse,
    StatusResponse,
    BatchStatusResponse
)


class TestChatMessage:
    """Test cases for ChatMessage schema"""
    
    def test_chat_message_valid(self):
        """Test creating valid chat message"""
        message = ChatMessage(message="Create a song")
        assert message.message == "Create a song"
        assert message.user_id is None
        assert message.chat_history is None
    
    def test_chat_message_with_user_id(self):
        """Test chat message with user ID"""
        message = ChatMessage(message="Create a song", user_id="user123")
        assert message.user_id == "user123"
    
    def test_chat_message_with_history(self):
        """Test chat message with chat history"""
        history = [{"role": "user", "content": "Hello"}]
        message = ChatMessage(message="Create a song", chat_history=history)
        assert message.chat_history == history
    
    def test_chat_message_empty_message(self):
        """Test empty message raises error"""
        with pytest.raises(ValidationError):
            ChatMessage(message="")
    
    def test_chat_message_too_long(self):
        """Test message too long raises error"""
        long_message = "A" * 501
        with pytest.raises(ValidationError):
            ChatMessage(message=long_message)
    
    def test_chat_message_whitespace_only(self):
        """Test whitespace-only message"""
        # Should be validated and sanitized
        message = ChatMessage(message="   ")
        assert message.message is not None


class TestSongGenerationRequest:
    """Test cases for SongGenerationRequest schema"""
    
    def test_song_generation_request_valid(self):
        """Test creating valid generation request"""
        request = SongGenerationRequest(prompt="Create a rock song")
        assert request.prompt == "Create a rock song"
        assert request.duration is None
        assert request.genre is None
    
    def test_song_generation_request_with_all_fields(self):
        """Test request with all fields"""
        request = SongGenerationRequest(
            prompt="Create a song",
            duration=60,
            genre="rock",
            mood="happy",
            user_id="user123"
        )
        assert request.duration == 60
        assert request.genre == "rock"
        assert request.mood == "happy"
        assert request.user_id == "user123"
    
    def test_song_generation_request_empty_prompt(self):
        """Test empty prompt raises error"""
        with pytest.raises(ValidationError):
            SongGenerationRequest(prompt="")
    
    def test_song_generation_request_duration_below_minimum(self):
        """Test duration below 1 raises error"""
        with pytest.raises(ValidationError):
            SongGenerationRequest(prompt="test", duration=0)
    
    def test_song_generation_request_duration_above_maximum(self):
        """Test duration above maximum raises error"""
        with pytest.raises(ValidationError):
            SongGenerationRequest(prompt="test", duration=301)
    
    def test_song_generation_request_valid_duration_range(self):
        """Test valid duration range"""
        request = SongGenerationRequest(prompt="test", duration=180)
        assert request.duration == 180


class TestSongResponse:
    """Test cases for SongResponse schema"""
    
    def test_song_response_valid(self):
        """Test creating valid song response"""
        response = SongResponse(
            song_id="song123",
            status="completed",
            message="Song generated successfully"
        )
        assert response.song_id == "song123"
        assert response.status == "completed"
        assert response.audio_url is None
    
    def test_song_response_with_audio_url(self):
        """Test response with audio URL"""
        response = SongResponse(
            song_id="song123",
            status="completed",
            message="Success",
            audio_url="https://example.com/audio.mp3"
        )
        assert response.audio_url == "https://example.com/audio.mp3"
    
    def test_song_response_with_metadata(self):
        """Test response with metadata"""
        metadata = {"duration": 60, "genre": "rock"}
        response = SongResponse(
            song_id="song123",
            status="completed",
            message="Success",
            metadata=metadata
        )
        assert response.metadata == metadata


class TestAudioEditRequest:
    """Test cases for AudioEditRequest schema"""
    
    def test_audio_edit_request_valid(self):
        """Test creating valid edit request"""
        request = AudioEditRequest(song_id="song123")
        assert request.song_id == "song123"
        assert request.operations == []
        assert request.normalize is True
    
    def test_audio_edit_request_with_operations(self):
        """Test request with operations"""
        operations = [{"type": "fade", "duration": 1.0}]
        request = AudioEditRequest(
            song_id="song123",
            operations=operations
        )
        assert len(request.operations) == 1
    
    def test_audio_edit_request_empty_song_id(self):
        """Test empty song ID raises error"""
        with pytest.raises(ValidationError):
            AudioEditRequest(song_id="")
    
    def test_audio_edit_request_whitespace_song_id(self):
        """Test whitespace song ID is trimmed"""
        request = AudioEditRequest(song_id="  song123  ")
        assert request.song_id == "song123"
    
    def test_audio_edit_request_fade_in(self):
        """Test fade in value"""
        request = AudioEditRequest(song_id="song123", fade_in=2.5)
        assert request.fade_in == 2.5
    
    def test_audio_edit_request_fade_in_above_maximum(self):
        """Test fade in above maximum raises error"""
        with pytest.raises(ValidationError):
            AudioEditRequest(song_id="song123", fade_in=11.0)
    
    def test_audio_edit_request_fade_out(self):
        """Test fade out value"""
        request = AudioEditRequest(song_id="song123", fade_out=1.5)
        assert request.fade_out == 1.5
    
    def test_audio_edit_request_negative_fade(self):
        """Test negative fade raises error"""
        with pytest.raises(ValidationError):
            AudioEditRequest(song_id="song123", fade_in=-1.0)


class TestAudioMixRequest:
    """Test cases for AudioMixRequest schema"""
    
    def test_audio_mix_request_valid(self):
        """Test creating valid mix request"""
        request = AudioMixRequest(song_ids=["song1", "song2"])
        assert len(request.song_ids) == 2
        assert request.volumes is None
    
    def test_audio_mix_request_with_volumes(self):
        """Test request with volumes"""
        request = AudioMixRequest(
            song_ids=["song1", "song2"],
            volumes=[0.5, 1.0]
        )
        assert request.volumes == [0.5, 1.0]
    
    def test_audio_mix_request_empty_song_ids(self):
        """Test empty song IDs raises error"""
        with pytest.raises(ValidationError):
            AudioMixRequest(song_ids=[])
    
    def test_audio_mix_request_too_many_songs(self):
        """Test too many songs raises error"""
        song_ids = [f"song{i}" for i in range(11)]
        with pytest.raises(ValidationError):
            AudioMixRequest(song_ids=song_ids)
    
    def test_audio_mix_request_volumes_mismatch(self):
        """Test volumes count mismatch raises error"""
        with pytest.raises(ValidationError):
            AudioMixRequest(
                song_ids=["song1", "song2"],
                volumes=[0.5]  # Only one volume for two songs
            )
    
    def test_audio_mix_request_invalid_song_id_format(self):
        """Test invalid song ID format raises error"""
        with pytest.raises(ValidationError):
            AudioMixRequest(song_ids=["invalid-id-format"])


class TestSongListResponse:
    """Test cases for SongListResponse schema"""
    
    def test_song_list_response_valid(self):
        """Test creating valid list response"""
        response = SongListResponse(
            songs=[{"id": "song1"}, {"id": "song2"}],
            total=2
        )
        assert len(response.songs) == 2
        assert response.total == 2
    
    def test_song_list_response_with_pagination(self):
        """Test response with pagination info"""
        response = SongListResponse(
            songs=[{"id": "song1"}],
            total=10,
            limit=5,
            offset=0,
            has_more=True
        )
        assert response.limit == 5
        assert response.offset == 0
        assert response.has_more is True


class TestSongAnalysisResponse:
    """Test cases for SongAnalysisResponse schema"""
    
    def test_song_analysis_response_valid(self):
        """Test creating valid analysis response"""
        response = SongAnalysisResponse(
            song_id="song123",
            analysis={"tempo": 120, "key": "C"},
            metadata={"duration": 60}
        )
        assert response.song_id == "song123"
        assert response.analysis["tempo"] == 120


class TestStatusResponse:
    """Test cases for StatusResponse schema"""
    
    def test_status_response_valid(self):
        """Test creating valid status response"""
        response = StatusResponse(
            status="processing",
            song_id="song123",
            message="Generating song"
        )
        assert response.status == "processing"
        assert response.progress is None
    
    def test_status_response_with_progress(self):
        """Test response with progress info"""
        progress = {"percentage": 50, "stage": "generating"}
        response = StatusResponse(
            status="processing",
            song_id="song123",
            message="Processing",
            progress=progress
        )
        assert response.progress == progress


class TestBatchStatusResponse:
    """Test cases for BatchStatusResponse schema"""
    
    def test_batch_status_response_valid(self):
        """Test creating valid batch status response"""
        response = BatchStatusResponse(
            total_requested=10,
            found=8,
            not_found=2,
            statuses={"song1": {"status": "completed"}}
        )
        assert response.total_requested == 10
        assert response.found == 8
        assert response.not_found == 2















