"""
Service Tests for Social Video Transcriber AI
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path

from ..services.video_downloader import VideoDownloader
from ..services.transcription_service import TranscriptionService
from ..core.models import SupportedPlatform, TranscriptionSegment


class TestVideoDownloader:
    """Tests for VideoDownloader service"""
    
    def test_detect_platform_youtube(self):
        """Test YouTube URL detection"""
        downloader = VideoDownloader()
        
        youtube_urls = [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtu.be/dQw4w9WgXcQ",
            "https://www.youtube.com/shorts/abc123",
        ]
        
        for url in youtube_urls:
            platform = downloader.detect_platform(url)
            assert platform == SupportedPlatform.YOUTUBE
    
    def test_detect_platform_tiktok(self):
        """Test TikTok URL detection"""
        downloader = VideoDownloader()
        
        tiktok_urls = [
            "https://www.tiktok.com/@user/video/1234567890",
            "https://vm.tiktok.com/abc123",
        ]
        
        for url in tiktok_urls:
            platform = downloader.detect_platform(url)
            assert platform == SupportedPlatform.TIKTOK
    
    def test_detect_platform_instagram(self):
        """Test Instagram URL detection"""
        downloader = VideoDownloader()
        
        instagram_urls = [
            "https://www.instagram.com/reel/abc123",
            "https://www.instagram.com/p/abc123",
            "https://www.instagram.com/tv/abc123",
        ]
        
        for url in instagram_urls:
            platform = downloader.detect_platform(url)
            assert platform == SupportedPlatform.INSTAGRAM
    
    def test_detect_platform_unsupported(self):
        """Test unsupported URL detection"""
        downloader = VideoDownloader()
        
        with pytest.raises(ValueError):
            downloader.detect_platform("https://example.com/video")


class TestTranscriptionService:
    """Tests for TranscriptionService"""
    
    def test_format_timestamp(self):
        """Test timestamp formatting"""
        service = TranscriptionService()
        
        assert service.format_timestamp(0) == "00:00"
        assert service.format_timestamp(65) == "01:05"
        assert service.format_timestamp(3661) == "01:01:01"
    
    def test_segments_to_srt(self):
        """Test SRT format conversion"""
        service = TranscriptionService()
        
        segments = [
            TranscriptionSegment(id=0, start_time=0.0, end_time=5.0, text="Hello"),
            TranscriptionSegment(id=1, start_time=5.0, end_time=10.0, text="World"),
        ]
        
        srt = service.segments_to_srt(segments)
        
        assert "1" in srt
        assert "00:00:00,000 --> 00:00:05,000" in srt
        assert "Hello" in srt
        assert "2" in srt
        assert "World" in srt
    
    def test_segments_to_vtt(self):
        """Test WebVTT format conversion"""
        service = TranscriptionService()
        
        segments = [
            TranscriptionSegment(id=0, start_time=0.0, end_time=5.0, text="Hello"),
        ]
        
        vtt = service.segments_to_vtt(segments)
        
        assert "WEBVTT" in vtt
        assert "00:00:00.000 --> 00:00:05.000" in vtt
        assert "Hello" in vtt


class TestTranscriptionSegment:
    """Tests for TranscriptionSegment model"""
    
    def test_duration_property(self):
        """Test duration calculation"""
        segment = TranscriptionSegment(
            id=0,
            start_time=5.0,
            end_time=10.5,
            text="Test"
        )
        
        assert segment.duration == 5.5
    
    def test_formatted_timestamp(self):
        """Test formatted timestamp"""
        segment = TranscriptionSegment(
            id=0,
            start_time=65.0,
            end_time=130.0,
            text="Test"
        )
        
        assert segment.formatted_timestamp == "[01:05 -> 02:10]"
    
    def test_to_dict(self):
        """Test dict conversion"""
        segment = TranscriptionSegment(
            id=0,
            start_time=0.0,
            end_time=5.0,
            text="Hello",
            confidence=0.95,
        )
        
        d = segment.to_dict()
        
        assert d["id"] == 0
        assert d["start_time"] == 0.0
        assert d["end_time"] == 5.0
        assert d["text"] == "Hello"
        assert d["confidence"] == 0.95
        assert "formatted_timestamp" in d












