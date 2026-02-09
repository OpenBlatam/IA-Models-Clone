"""
Comprehensive Unit Tests for Lyrics Generator

Tests cover lyrics generation functionality with diverse test cases
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from services.lyrics_generator import LyricsGenerator, Lyrics


class TestLyrics:
    """Test cases for Lyrics dataclass"""
    
    def test_lyrics_creation(self):
        """Test creating lyrics object"""
        lyrics = Lyrics(
            title="Test Song",
            verses=["Verse 1", "Verse 2"]
        )
        assert lyrics.title == "Test Song"
        assert len(lyrics.verses) == 2
        assert lyrics.chorus is None
        assert lyrics.language == "en"
    
    def test_lyrics_with_chorus(self):
        """Test lyrics with chorus"""
        lyrics = Lyrics(
            title="Test Song",
            verses=["Verse 1"],
            chorus="Chorus line"
        )
        assert lyrics.chorus == "Chorus line"
    
    def test_lyrics_with_bridge(self):
        """Test lyrics with bridge"""
        lyrics = Lyrics(
            title="Test Song",
            verses=["Verse 1"],
            bridge="Bridge line"
        )
        assert lyrics.bridge == "Bridge line"
    
    def test_lyrics_custom_language(self):
        """Test lyrics with custom language"""
        lyrics = Lyrics(
            title="Test Song",
            verses=["Verse 1"],
            language="es"
        )
        assert lyrics.language == "es"
    
    def test_lyrics_with_style_and_theme(self):
        """Test lyrics with style and theme"""
        lyrics = Lyrics(
            title="Test Song",
            verses=["Verse 1"],
            style="rock",
            theme="love"
        )
        assert lyrics.style == "rock"
        assert lyrics.theme == "love"


class TestLyricsGenerator:
    """Test cases for LyricsGenerator class"""
    
    def test_lyrics_generator_init_without_transformers(self):
        """Test initializing generator without transformers"""
        with patch('services.lyrics_generator.TRANSFORMERS_AVAILABLE', False):
            generator = LyricsGenerator()
            assert generator.generator is None
            assert generator.model_name == "gpt2"
    
    def test_lyrics_generator_init_with_transformers(self):
        """Test initializing generator with transformers"""
        with patch('services.lyrics_generator.TRANSFORMERS_AVAILABLE', True):
            mock_pipeline = Mock()
            with patch('services.lyrics_generator.pipeline', return_value=mock_pipeline):
                generator = LyricsGenerator()
                assert generator.generator == mock_pipeline
    
    def test_generate_lyrics_mock_mode(self):
        """Test generating lyrics in mock mode (no transformers)"""
        with patch('services.lyrics_generator.TRANSFORMERS_AVAILABLE', False):
            generator = LyricsGenerator()
            lyrics = generator.generate_lyrics(
                theme="love",
                style="pop",
                num_verses=3,
                include_chorus=True
            )
            
            assert isinstance(lyrics, Lyrics)
            assert lyrics.theme == "love"
            assert lyrics.style == "pop"
            assert len(lyrics.verses) == 3
            assert lyrics.chorus is not None
    
    def test_generate_lyrics_without_chorus(self):
        """Test generating lyrics without chorus"""
        with patch('services.lyrics_generator.TRANSFORMERS_AVAILABLE', False):
            generator = LyricsGenerator()
            lyrics = generator.generate_lyrics(
                theme="adventure",
                include_chorus=False
            )
            
            assert lyrics.chorus is None
    
    def test_generate_lyrics_custom_language(self):
        """Test generating lyrics in custom language"""
        with patch('services.lyrics_generator.TRANSFORMERS_AVAILABLE', False):
            generator = LyricsGenerator()
            lyrics = generator.generate_lyrics(
                theme="amor",
                language="es"
            )
            
            assert lyrics.language == "es"
    
    def test_generate_lyrics_custom_num_verses(self):
        """Test generating lyrics with custom number of verses"""
        with patch('services.lyrics_generator.TRANSFORMERS_AVAILABLE', False):
            generator = LyricsGenerator()
            lyrics = generator.generate_lyrics(
                theme="test",
                num_verses=5
            )
            
            assert len(lyrics.verses) == 5
    
    def test_generate_lyrics_with_transformers(self):
        """Test generating lyrics with transformers available"""
        with patch('services.lyrics_generator.TRANSFORMERS_AVAILABLE', True):
            mock_generator = Mock()
            mock_generator.return_value = [{"generated_text": "Generated lyrics text"}]
            
            generator = LyricsGenerator()
            generator.generator = mock_generator
            
            lyrics = generator.generate_lyrics(theme="test")
            
            assert isinstance(lyrics, Lyrics)
            mock_generator.assert_called()
    
    def test_generate_lyrics_transformers_error_handling(self):
        """Test error handling when transformers fail"""
        with patch('services.lyrics_generator.TRANSFORMERS_AVAILABLE', True):
            mock_generator = Mock()
            mock_generator.side_effect = Exception("Generation error")
            
            generator = LyricsGenerator()
            generator.generator = mock_generator
            
            # Should fallback to mock mode
            lyrics = generator.generate_lyrics(theme="test")
            assert isinstance(lyrics, Lyrics)
            assert lyrics.theme == "test"
    
    def test_generate_lyrics_empty_theme(self):
        """Test generating lyrics with empty theme"""
        with patch('services.lyrics_generator.TRANSFORMERS_AVAILABLE', False):
            generator = LyricsGenerator()
            lyrics = generator.generate_lyrics(theme="")
            
            assert isinstance(lyrics, Lyrics)
            assert "about" in lyrics.title.lower() or lyrics.title == ""
    
    def test_generate_lyrics_title_generation(self):
        """Test that title is generated from theme"""
        with patch('services.lyrics_generator.TRANSFORMERS_AVAILABLE', False):
            generator = LyricsGenerator()
            lyrics = generator.generate_lyrics(theme="adventure")
            
            assert "adventure" in lyrics.title.lower()















