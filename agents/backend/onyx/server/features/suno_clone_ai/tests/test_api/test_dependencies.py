"""
Comprehensive Unit Tests for API Dependencies

Tests cover FastAPI dependency injection with diverse test cases
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from api.dependencies import (
    get_song_service,
    get_song_service_async,
    get_music_gen,
    get_chat_proc,
    get_cache_mgr,
    get_audio_proc,
    get_metrics_svc,
    get_notification_svc
)


class TestGetSongService:
    """Test cases for get_song_service function"""
    
    def test_get_song_service_singleton(self):
        """Test that get_song_service returns singleton"""
        # Clear cache
        get_song_service.cache_clear()
        
        service1 = get_song_service()
        service2 = get_song_service()
        
        assert service1 is service2
        assert service1 is not None
    
    def test_get_song_service_cached(self):
        """Test that service is cached"""
        get_song_service.cache_clear()
        
        service1 = get_song_service()
        service2 = get_song_service()
        
        # Should be same instance due to lru_cache
        assert service1 is service2


class TestGetSongServiceAsync:
    """Test cases for get_song_service_async function"""
    
    @pytest.mark.asyncio
    async def test_get_song_service_async_available(self):
        """Test getting async service when available"""
        mock_async_service = Mock()
        
        with patch('api.dependencies.get_song_service_async') as mock_get:
            mock_get.return_value = mock_async_service
            
            result = await get_song_service_async()
            
            assert result == mock_async_service
    
    @pytest.mark.asyncio
    async def test_get_song_service_async_fallback(self):
        """Test fallback to sync service when async not available"""
        with patch('api.dependencies.get_song_service_async', side_effect=ImportError):
            result = await get_song_service_async()
            
            # Should fallback to sync service
            assert result is not None


class TestGetMusicGen:
    """Test cases for get_music_gen function"""
    
    def test_get_music_gen_singleton(self):
        """Test that get_music_gen returns singleton"""
        get_music_gen.cache_clear()
        
        gen1 = get_music_gen()
        gen2 = get_music_gen()
        
        assert gen1 is gen2
    
    @patch('api.dependencies.get_music_generator')
    def test_get_music_gen_calls_get_music_generator(self, mock_get):
        """Test that get_music_gen calls get_music_generator"""
        mock_generator = Mock()
        mock_get.return_value = mock_generator
        
        get_music_gen.cache_clear()
        result = get_music_gen()
        
        assert result == mock_generator
        mock_get.assert_called_once()


class TestGetChatProc:
    """Test cases for get_chat_proc function"""
    
    def test_get_chat_proc_singleton(self):
        """Test that get_chat_proc returns singleton"""
        get_chat_proc.cache_clear()
        
        proc1 = get_chat_proc()
        proc2 = get_chat_proc()
        
        assert proc1 is proc2
    
    @patch('api.dependencies.get_chat_processor')
    def test_get_chat_proc_calls_get_chat_processor(self, mock_get):
        """Test that get_chat_proc calls get_chat_processor"""
        mock_processor = Mock()
        mock_get.return_value = mock_processor
        
        get_chat_proc.cache_clear()
        result = get_chat_proc()
        
        assert result == mock_processor
        mock_get.assert_called_once()


class TestGetCacheMgr:
    """Test cases for get_cache_mgr function"""
    
    def test_get_cache_mgr_singleton(self):
        """Test that get_cache_mgr returns singleton"""
        get_cache_mgr.cache_clear()
        
        mgr1 = get_cache_mgr()
        mgr2 = get_cache_mgr()
        
        assert mgr1 is mgr2
    
    @patch('api.dependencies.get_cache_manager')
    def test_get_cache_mgr_calls_get_cache_manager(self, mock_get):
        """Test that get_cache_mgr calls get_cache_manager"""
        mock_manager = Mock()
        mock_get.return_value = mock_manager
        
        get_cache_mgr.cache_clear()
        result = get_cache_mgr()
        
        assert result == mock_manager
        mock_get.assert_called_once()


class TestGetAudioProc:
    """Test cases for get_audio_proc function"""
    
    def test_get_audio_proc_singleton(self):
        """Test that get_audio_proc returns singleton"""
        get_audio_proc.cache_clear()
        
        proc1 = get_audio_proc()
        proc2 = get_audio_proc()
        
        assert proc1 is proc2
    
    @patch('api.dependencies.get_audio_processor')
    def test_get_audio_proc_calls_get_audio_processor(self, mock_get):
        """Test that get_audio_proc calls get_audio_processor"""
        mock_processor = Mock()
        mock_get.return_value = mock_processor
        
        get_audio_proc.cache_clear()
        result = get_audio_proc()
        
        assert result == mock_processor
        mock_get.assert_called_once()


class TestGetMetricsSvc:
    """Test cases for get_metrics_svc function"""
    
    def test_get_metrics_svc_singleton(self):
        """Test that get_metrics_svc returns singleton"""
        get_metrics_svc.cache_clear()
        
        svc1 = get_metrics_svc()
        svc2 = get_metrics_svc()
        
        assert svc1 is svc2
    
    @patch('api.dependencies.get_metrics_service')
    def test_get_metrics_svc_calls_get_metrics_service(self, mock_get):
        """Test that get_metrics_svc calls get_metrics_service"""
        mock_service = Mock()
        mock_get.return_value = mock_service
        
        get_metrics_svc.cache_clear()
        result = get_metrics_svc()
        
        assert result == mock_service
        mock_get.assert_called_once()


class TestGetNotificationSvc:
    """Test cases for get_notification_svc function"""
    
    @patch('api.dependencies.get_notification_service')
    def test_get_notification_svc_available(self, mock_get):
        """Test getting notification service when available"""
        mock_service = Mock()
        mock_get.return_value = mock_service
        
        result = get_notification_svc()
        
        assert result == mock_service
        mock_get.assert_called_once()
    
    @patch('api.dependencies.get_notification_service')
    def test_get_notification_svc_returns_none_on_error(self, mock_get):
        """Test that service returns None on error (non-critical)"""
        mock_get.side_effect = Exception("Service unavailable")
        
        result = get_notification_svc()
        
        # Should return None for non-critical service
        assert result is None















