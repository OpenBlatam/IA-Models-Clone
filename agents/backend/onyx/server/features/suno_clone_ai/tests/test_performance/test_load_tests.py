"""
Tests de rendimiento y carga
"""

import pytest
import time
from unittest.mock import Mock, patch
from fastapi import status
from fastapi.testclient import TestClient
import concurrent.futures

from api.routes.generation import router as generation_router
from api.routes.playlists import router as playlists_router


@pytest.fixture
def performance_client():
    """Cliente para tests de rendimiento"""
    from fastapi import FastAPI
    from api.dependencies import SongServiceDep
    
    app = FastAPI()
    app.include_router(generation_router)
    app.include_router(playlists_router)
    
    mock_service = Mock()
    mock_service.save_song = Mock(return_value=True)
    mock_service.get_song = Mock(return_value=None)
    mock_service.list_songs = Mock(return_value=[])
    
    def get_song_service():
        return mock_service
    
    app.dependency_overrides[SongServiceDep] = get_song_service
    
    with patch('api.routes.generation.get_music_generator'):
        with patch('api.routes.generation.get_chat_processor'):
            with patch('api.routes.generation.get_current_user', return_value={"user_id": "test_user"}):
                with patch('api.routes.playlists.get_current_user', return_value={"user_id": "test_user"}):
                    yield TestClient(app)
    
    app.dependency_overrides.clear()


@pytest.mark.performance
@pytest.mark.slow
class TestResponseTime:
    """Tests de tiempo de respuesta"""
    
    def test_playlist_creation_response_time(self, performance_client):
        """Test de tiempo de respuesta para creación de playlist"""
        start_time = time.time()
        
        response = performance_client.post(
            "/playlists",
            params={
                "name": "Performance Test",
                "user_id": "perf-user-123"
            }
        )
        
        end_time = time.time()
        response_time = end_time - start_time
        
        assert response.status_code == status.HTTP_200_OK
        assert response_time < 1.0, f"Response time {response_time}s exceeds 1s threshold"
    
    def test_multiple_playlist_creation_time(self, performance_client):
        """Test de tiempo para múltiples creaciones"""
        num_requests = 10
        start_time = time.time()
        
        for i in range(num_requests):
            response = performance_client.post(
                "/playlists",
                params={
                    "name": f"Playlist {i}",
                    "user_id": "perf-user-123"
                }
            )
            assert response.status_code == status.HTTP_200_OK
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / num_requests
        
        assert avg_time < 0.5, f"Average response time {avg_time}s exceeds 0.5s threshold"


@pytest.mark.performance
@pytest.mark.slow
class TestConcurrentRequests:
    """Tests de requests concurrentes"""
    
    def test_concurrent_playlist_creation(self, performance_client):
        """Test de creación concurrente de playlists"""
        num_concurrent = 5
        
        def create_playlist(index):
            response = performance_client.post(
                "/playlists",
                params={
                    "name": f"Concurrent Playlist {index}",
                    "user_id": f"user-{index}"
                }
            )
            return response.status_code == status.HTTP_200_OK
        
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [
                executor.submit(create_playlist, i)
                for i in range(num_concurrent)
            ]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        end_time = time.time()
        total_time = end_time - start_time
        
        assert all(results), "Not all concurrent requests succeeded"
        assert total_time < 2.0, f"Concurrent requests took {total_time}s, exceeds 2s threshold"
    
    def test_concurrent_lyrics_generation(self, performance_client):
        """Test de generación concurrente de letras"""
        from api.routes.lyrics import router as lyrics_router
        from fastapi import FastAPI
        
        app = FastAPI()
        app.include_router(lyrics_router)
        
        with patch('api.routes.lyrics.get_lyrics_generator') as mock_gen:
            lyrics_obj = Mock()
            lyrics_obj.title = "Test"
            lyrics_obj.verses = ["Verse 1"]
            lyrics_obj.chorus = "Chorus"
            lyrics_obj.bridge = None
            lyrics_obj.language = "en"
            lyrics_obj.style = "pop"
            lyrics_obj.theme = "test"
            mock_gen.return_value.generate_lyrics.return_value = lyrics_obj
            
            with patch('api.routes.lyrics.get_current_user', return_value={"user_id": "test_user"}):
                client = TestClient(app)
                
                num_concurrent = 3
                
                def generate_lyrics():
                    response = client.post(
                        "/lyrics/generate",
                        json={"theme": "test"}
                    )
                    return response.status_code == status.HTTP_200_OK
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent) as executor:
                    futures = [
                        executor.submit(generate_lyrics)
                        for _ in range(num_concurrent)
                    ]
                    results = [future.result() for future in concurrent.futures.as_completed(futures)]
                
                assert all(results), "Not all concurrent lyrics generations succeeded"


@pytest.mark.performance
@pytest.mark.slow
class TestThroughput:
    """Tests de throughput"""
    
    def test_playlist_creation_throughput(self, performance_client):
        """Test de throughput para creación de playlists"""
        num_requests = 20
        start_time = time.time()
        
        successful = 0
        for i in range(num_requests):
            response = performance_client.post(
                "/playlists",
                params={
                    "name": f"Throughput Test {i}",
                    "user_id": "throughput-user"
                }
            )
            if response.status_code == status.HTTP_200_OK:
                successful += 1
        
        end_time = time.time()
        total_time = end_time - start_time
        throughput = successful / total_time if total_time > 0 else 0
        
        assert successful == num_requests, f"Only {successful}/{num_requests} requests succeeded"
        assert throughput > 10, f"Throughput {throughput} req/s is below 10 req/s threshold"


@pytest.mark.performance
@pytest.mark.slow
class TestResourceUsage:
    """Tests de uso de recursos"""
    
    def test_memory_efficient_playlist_creation(self, performance_client):
        """Test de uso eficiente de memoria"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Crear múltiples playlists
        for i in range(50):
            response = performance_client.post(
                "/playlists",
                params={
                    "name": f"Memory Test {i}",
                    "user_id": "memory-user"
                }
            )
            assert response.status_code == status.HTTP_200_OK
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # El aumento de memoria no debería ser excesivo
        assert memory_increase < 100, \
            f"Memory increase {memory_increase}MB exceeds 100MB threshold"



