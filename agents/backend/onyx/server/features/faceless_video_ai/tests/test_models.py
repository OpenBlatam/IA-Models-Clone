"""
Unit tests for models
"""

import pytest
from uuid import UUID
from datetime import datetime

from ..core.models import (
    VideoGenerationRequest,
    VideoScript,
    VideoConfig,
    AudioConfig,
    SubtitleConfig,
)


class TestVideoGenerationRequest:
    """Tests for VideoGenerationRequest"""
    
    def test_valid_request(self):
        """Test valid request creation"""
        script = VideoScript(
            text="Test script",
            language="es"
        )
        request = VideoGenerationRequest(script=script)
        assert request.script.text == "Test script"
        assert request.script.language == "es"
    
    def test_request_with_config(self):
        """Test request with custom config"""
        script = VideoScript(text="Test", language="es")
        video_config = VideoConfig(resolution="1920x1080", fps=30)
        request = VideoGenerationRequest(
            script=script,
            video_config=video_config
        )
        assert request.video_config.resolution == "1920x1080"
        assert request.video_config.fps == 30


class TestVideoScript:
    """Tests for VideoScript"""
    
    def test_script_creation(self):
        """Test script creation"""
        script = VideoScript(
            text="Hello world",
            language="en"
        )
        assert script.text == "Hello world"
        assert script.language == "en"
    
    def test_script_segmentation(self):
        """Test script segmentation"""
        script = VideoScript(
            text="First sentence. Second sentence.",
            language="en"
        )
        # Segmentation would be tested here
        assert len(script.text) > 0


class TestVideoConfig:
    """Tests for VideoConfig"""
    
    def test_default_config(self):
        """Test default video config"""
        config = VideoConfig()
        assert config.resolution == "1920x1080"
        assert config.fps == 30
    
    def test_custom_config(self):
        """Test custom video config"""
        config = VideoConfig(
            resolution="1280x720",
            fps=24
        )
        assert config.resolution == "1280x720"
        assert config.fps == 24


class TestAudioConfig:
    """Tests for AudioConfig"""
    
    def test_default_audio_config(self):
        """Test default audio config"""
        config = AudioConfig()
        assert config.voice == "neutral"
        assert config.speed == 1.0
    
    def test_custom_audio_config(self):
        """Test custom audio config"""
        config = AudioConfig(
            voice="male_1",
            speed=1.2
        )
        assert config.voice == "male_1"
        assert config.speed == 1.2


class TestSubtitleConfig:
    """Tests for SubtitleConfig"""
    
    def test_default_subtitle_config(self):
        """Test default subtitle config"""
        config = SubtitleConfig()
        assert config.style == "modern"
        assert config.position == "bottom"
    
    def test_custom_subtitle_config(self):
        """Test custom subtitle config"""
        config = SubtitleConfig(
            style="bold",
            position="top"
        )
        assert config.style == "bold"
        assert config.position == "top"

