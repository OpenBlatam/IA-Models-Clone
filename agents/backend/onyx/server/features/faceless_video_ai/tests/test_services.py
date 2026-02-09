"""
Unit tests for services
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from ..services.script_processor import ScriptProcessor
from ..services.video_generator import VideoGenerator
from ..services.audio_generator import AudioGenerator


class TestScriptProcessor:
    """Tests for ScriptProcessor"""
    
    def test_segment_script(self):
        """Test script segmentation"""
        processor = ScriptProcessor()
        script = "First sentence. Second sentence. Third sentence."
        
        segments = processor.segment_script(script, max_length=50)
        assert len(segments) > 0
        assert all(len(s) <= 50 for s in segments)
    
    def test_empty_script(self):
        """Test empty script handling"""
        processor = ScriptProcessor()
        segments = processor.segment_script("", max_length=50)
        assert len(segments) == 0


class TestVideoGenerator:
    """Tests for VideoGenerator"""
    
    @pytest.mark.asyncio
    async def test_generate_images(self):
        """Test image generation"""
        with patch('..services.ai_providers.image_providers.get_image_provider') as mock_get:
            mock_provider = Mock()
            mock_provider.generate_image = AsyncMock(return_value=Path("/tmp/test.jpg"))
            mock_get.return_value = mock_provider
            
            generator = VideoGenerator(
                output_dir="/tmp",
                image_provider=mock_provider
            )
            
            prompts = ["Test prompt"]
            images = await generator.generate_images(prompts)
            
            assert len(images) == 1
            mock_provider.generate_image.assert_called_once()


class TestAudioGenerator:
    """Tests for AudioGenerator"""
    
    @pytest.mark.asyncio
    async def test_generate_audio(self):
        """Test audio generation"""
        with patch('..services.ai_providers.audio_providers.get_audio_provider') as mock_get:
            mock_provider = Mock()
            mock_provider.generate_speech = AsyncMock(return_value=Path("/tmp/test.mp3"))
            mock_get.return_value = mock_provider
            
            generator = AudioGenerator(
                output_dir="/tmp",
                audio_provider=mock_provider
            )
            
            text = "Test text"
            audio_path = await generator.generate_tts(text, "en", "neutral")
            
            assert audio_path.exists() or str(audio_path) == "/tmp/test.mp3"
            mock_provider.generate_speech.assert_called_once()

