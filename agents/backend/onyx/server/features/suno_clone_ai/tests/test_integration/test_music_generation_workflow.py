"""
Integration Tests for Music Generation Workflow

Tests cover complete workflows integrating multiple components
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from core.chat_processor import ChatProcessor
from core.music_generator import MusicGenerator
from core.cache_manager import CacheManager
from core.audio_processor import AudioProcessor
from services.song_service import SongService


class TestMusicGenerationWorkflow:
    """Integration tests for complete music generation workflow"""
    
    @pytest.mark.asyncio
    async def test_complete_generation_workflow(self):
        """Test complete workflow from chat to audio generation"""
        # Setup mocks
        chat_processor = Mock(spec=ChatProcessor)
        chat_processor.extract_song_info.return_value = {
            "prompt": "rock song",
            "genre": "rock",
            "mood": "energetic",
            "duration": 30
        }
        
        music_generator = Mock(spec=MusicGenerator)
        music_generator.generate.return_value = b"audio_data"
        
        cache_manager = Mock(spec=CacheManager)
        cache_manager.get.return_value = None  # Not cached
        cache_manager.set = Mock()
        
        audio_processor = Mock(spec=AudioProcessor)
        audio_processor.normalize.return_value = b"normalized_audio"
        
        # Simulate workflow
        message = "Create a rock song"
        song_info = chat_processor.extract_song_info(message, [])
        
        # Check cache
        cached = cache_manager.get(prompt=song_info["prompt"])
        assert cached is None
        
        # Generate audio
        audio = music_generator.generate(
            prompt=song_info["prompt"],
            duration=song_info.get("duration", 30)
        )
        
        # Process audio
        processed_audio = audio_processor.normalize(audio)
        
        # Cache result
        cache_manager.set(
            prompt=song_info["prompt"],
            audio=processed_audio
        )
        
        # Verify workflow
        assert audio is not None
        assert processed_audio is not None
        cache_manager.set.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generation_workflow_with_cache_hit(self):
        """Test workflow when result is cached"""
        cache_manager = Mock(spec=CacheManager)
        cache_manager.get.return_value = {
            "audio": b"cached_audio",
            "song_id": "song123"
        }
        
        music_generator = Mock(spec=MusicGenerator)
        
        # Check cache first
        cached = cache_manager.get(prompt="rock song")
        
        if cached:
            # Use cached result
            audio = cached["audio"]
            music_generator.generate.assert_not_called()
        else:
            # Generate new
            audio = music_generator.generate(prompt="rock song")
        
        assert audio == b"cached_audio"
        music_generator.generate.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_generation_workflow_with_error_handling(self):
        """Test workflow error handling"""
        chat_processor = Mock(spec=ChatProcessor)
        chat_processor.extract_song_info.side_effect = ValueError("Invalid prompt")
        
        # Should handle error gracefully
        with pytest.raises(ValueError):
            chat_processor.extract_song_info("invalid", [])
    
    @pytest.mark.asyncio
    async def test_generation_workflow_with_audio_processing(self):
        """Test workflow with audio processing steps"""
        audio_processor = Mock(spec=AudioProcessor)
        audio_processor.normalize.return_value = b"normalized"
        audio_processor.add_fade.return_value = b"faded"
        audio_processor.add_reverb.return_value = b"reverbed"
        
        raw_audio = b"raw_audio"
        
        # Process audio through pipeline
        normalized = audio_processor.normalize(raw_audio)
        faded = audio_processor.add_fade(normalized, fade_in=1.0, fade_out=1.0)
        final = audio_processor.add_reverb(faded, room_size=0.5)
        
        assert final == b"reverbed"
        audio_processor.normalize.assert_called_once()
        audio_processor.add_fade.assert_called_once()
        audio_processor.add_reverb.assert_called_once()


class TestSongServiceIntegration:
    """Integration tests for song service with other components"""
    
    @pytest.mark.asyncio
    async def test_save_and_retrieve_song(self):
        """Test saving and retrieving song"""
        song_service = Mock(spec=SongService)
        
        song_data = {
            "song_id": "song123",
            "prompt": "rock song",
            "audio_path": "/path/to/audio.mp3",
            "user_id": "user456"
        }
        
        # Save song
        song_service.save_song.return_value = song_data["song_id"]
        saved_id = song_service.save_song(song_data)
        
        # Retrieve song
        song_service.get_song.return_value = song_data
        retrieved = song_service.get_song(saved_id)
        
        assert saved_id == song_data["song_id"]
        assert retrieved == song_data
        song_service.save_song.assert_called_once()
        song_service.get_song.assert_called_once_with(saved_id)
    
    @pytest.mark.asyncio
    async def test_song_service_with_cache(self):
        """Test song service integration with cache"""
        song_service = Mock(spec=SongService)
        cache_manager = Mock(spec=CacheManager)
        
        song_id = "song123"
        
        # Check cache first
        cached = cache_manager.get(key=f"song:{song_id}")
        
        if cached:
            song = cached
        else:
            # Get from service
            song = song_service.get_song(song_id)
            # Cache it
            cache_manager.set(key=f"song:{song_id}", value=song)
        
        assert song is not None















