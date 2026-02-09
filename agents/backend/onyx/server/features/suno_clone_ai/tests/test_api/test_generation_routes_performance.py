"""
Tests de performance y optimización para generation routes
"""

import pytest
from fastapi import status
from unittest.mock import patch, Mock
from tests.helpers.test_helpers import create_song_dict, generate_test_song_id
from tests.helpers.advanced_helpers import PerformanceHelper, AsyncTestHelper


class TestGenerationPerformance:
    """Tests de performance para endpoints de generación"""
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_create_song_from_chat_performance(
        self,
        test_client
    ):
        """Test de performance para create_song_from_chat"""
        request = {
            "message": "Create a song",
            "user_id": "test-user"
        }
        
        with patch('api.routes.generation.extract_song_info_from_chat') as mock_extract, \
             patch('api.routes.generation.generate_song_background'), \
             patch('api.routes.generation.get_request_metadata') as mock_metadata, \
             patch('api.routes.generation.measure_time') as mock_measure:
            
            mock_extract.return_value = {"prompt": "song"}
            mock_metadata.return_value = {}
            
            # Mock measure_time como context manager
            mock_measure.return_value.__enter__ = Mock()
            mock_measure.return_value.__exit__ = Mock(return_value=None)
            
            async def create_song():
                response = test_client.post(
                    "/suno/generate/chat/create-song",
                    json=request
                )
                assert response.status_code == 202
            
            execution_time = await PerformanceHelper.measure_async_execution_time(create_song)
            
            # Debe responder rápidamente (< 500ms)
            assert execution_time < 0.5, f"Too slow: {execution_time}s"
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_generate_song_performance(
        self,
        test_client
    ):
        """Test de performance para generate_song"""
        request = {
            "prompt": "A song",
            "duration": 30
        }
        
        with patch('api.routes.generation.generate_song_background'), \
             patch('api.routes.generation.get_request_metadata') as mock_metadata, \
             patch('api.routes.generation.measure_time') as mock_measure:
            
            mock_metadata.return_value = {}
            mock_measure.return_value.__enter__ = Mock()
            mock_measure.return_value.__exit__ = Mock(return_value=None)
            
            async def generate_song():
                response = test_client.post("/suno/generate", json=request)
                assert response.status_code == 202
            
            execution_time = await PerformanceHelper.measure_async_execution_time(generate_song)
            
            assert execution_time < 0.5, f"Too slow: {execution_time}s"
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_get_status_performance(
        self,
        test_client,
        mock_song_service
    ):
        """Test de performance para get_generation_status"""
        task_id = generate_test_song_id()
        song = create_song_dict(song_id=task_id, status="processing")
        mock_song_service.get_song.return_value = song
        
        with patch('api.routes.generation.validate_song_id'), \
             patch('api.routes.generation.get_song_async_or_sync') as mock_get, \
             patch('api.routes.generation.add_cache_headers') as mock_cache:
            
            mock_get.return_value = song
            
            async def get_status():
                response = test_client.get(f"/suno/generate/status/{task_id}")
                assert response.status_code == 200
            
            execution_time = await PerformanceHelper.measure_async_execution_time(get_status)
            
            # Status check debe ser muy rápido (< 200ms)
            assert execution_time < 0.2, f"Too slow: {execution_time}s"
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_batch_status_performance(
        self,
        test_client,
        mock_song_service
    ):
        """Test de performance para batch_status"""
        task_ids = [generate_test_song_id() for _ in range(10)]
        songs = [
            create_song_dict(song_id=tid, status="processing")
            for tid in task_ids
        ]
        
        with patch('api.routes.generation.parse_comma_separated_ids') as mock_parse, \
             patch('api.routes.generation.batch_get_songs') as mock_batch:
            
            mock_parse.return_value = task_ids
            mock_batch.return_value = songs
            
            async def get_batch_status():
                response = test_client.get(
                    "/suno/generate/batch-status",
                    params={"task_ids": ",".join(task_ids)}
                )
                assert response.status_code == 200
            
            execution_time = await PerformanceHelper.measure_async_execution_time(get_batch_status)
            
            # Batch de 10 items debe ser rápido (< 1s)
            assert execution_time < 1.0, f"Too slow: {execution_time}s"


class TestGenerationOptimization:
    """Tests de optimización para generation routes"""
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_request_metadata_logging(
        self,
        test_client
    ):
        """Test que verifica logging de metadata"""
        request = {
            "message": "Create a song",
            "user_id": "test-user"
        }
        
        with patch('api.routes.generation.extract_song_info_from_chat') as mock_extract, \
             patch('api.routes.generation.generate_song_background'), \
             patch('api.routes.generation.get_request_metadata') as mock_metadata, \
             patch('api.routes.generation.logger') as mock_logger:
            
            mock_extract.return_value = {"prompt": "song"}
            mock_metadata.return_value = {"method": "POST", "path": "/generate/chat/create-song"}
            
            response = test_client.post(
                "/suno/generate/chat/create-song",
                json=request
            )
            
            assert response.status_code == 202
            # Verificar que se llamó get_request_metadata
            mock_metadata.assert_called_once()
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_measure_time_usage(
        self,
        test_client
    ):
        """Test que verifica uso de measure_time"""
        request = {
            "prompt": "A song",
            "duration": 30
        }
        
        with patch('api.routes.generation.generate_song_background'), \
             patch('api.routes.generation.get_request_metadata') as mock_metadata, \
             patch('api.routes.generation.measure_time') as mock_measure:
            
            mock_metadata.return_value = {}
            # Mock measure_time como context manager
            mock_measure.return_value.__enter__ = Mock()
            mock_measure.return_value.__exit__ = Mock(return_value=None)
            
            response = test_client.post("/suno/generate", json=request)
            
            assert response.status_code == 202
            # Verificar que se usó measure_time
            assert mock_measure.called
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_cache_headers_optimization(
        self,
        test_client,
        mock_song_service
    ):
        """Test que verifica optimización de cache headers"""
        task_id = generate_test_song_id()
        song = create_song_dict(song_id=task_id, status="processing")
        mock_song_service.get_song.return_value = song
        
        with patch('api.routes.generation.validate_song_id'), \
             patch('api.routes.generation.get_song_async_or_sync') as mock_get, \
             patch('api.routes.generation.add_cache_headers') as mock_cache:
            
            mock_get.return_value = song
            
            response = test_client.get(f"/suno/generate/status/{task_id}")
            
            assert response.status_code == 200
            # Verificar que se agregaron headers de cache
            assert "Cache-Control" in response.headers or mock_cache.called

