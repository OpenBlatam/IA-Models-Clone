"""
Tests de carga y stress
"""

import pytest
import asyncio
from concurrent.futures import ThreadPoolExecutor
from tests.helpers.test_helpers import generate_test_song_id, create_song_dict
from tests.helpers.advanced_helpers import DataFactory, PerformanceHelper


class TestLoadGeneration:
    """Tests de carga para generación"""
    
    @pytest.mark.load
    @pytest.mark.asyncio
    async def test_concurrent_generation_requests(
        self,
        test_client
    ):
        """Test de requests concurrentes de generación"""
        requests = DataFactory.create_song_requests(count=50)
        
        async def create_song(req):
            with patch('api.routes.generation.generate_song_background'):
                response = test_client.post("/suno/generate", json=req)
                return response.status_code == 202
        
        # Crear canciones concurrentemente
        results = await asyncio.gather(*[
            create_song(req) for req in requests
        ])
        
        # Al menos el 80% debe tener éxito
        success_rate = sum(results) / len(results)
        assert success_rate >= 0.8, f"Success rate too low: {success_rate}"
    
    @pytest.mark.load
    @pytest.mark.asyncio
    async def test_high_volume_status_checks(
        self,
        test_client,
        mock_song_service
    ):
        """Test de alto volumen de verificaciones de estado"""
        task_ids = [generate_test_song_id() for _ in range(100)]
        songs = [
            create_song_dict(song_id=tid, status="processing")
            for tid in task_ids
        ]
        
        mock_song_service.get_song.side_effect = lambda sid: next(
            (s for s in songs if s["song_id"] == sid), None
        )
        
        async def check_status(task_id):
            with patch('api.routes.generation.validate_song_id'), \
                 patch('api.routes.generation.get_song_async_or_sync') as mock_get:
                mock_get.return_value = next(
                    (s for s in songs if s["song_id"] == task_id), None
                )
                response = test_client.get(f"/suno/generate/status/{task_id}")
                return response.status_code == 200
        
        # Verificar estados concurrentemente
        results = await asyncio.gather(*[
            check_status(tid) for tid in task_ids
        ])
        
        # Al menos el 90% debe tener éxito
        success_rate = sum(results) / len(results)
        assert success_rate >= 0.9, f"Success rate too low: {success_rate}"
    
    @pytest.mark.load
    @pytest.mark.asyncio
    async def test_batch_status_high_volume(
        self,
        test_client,
        mock_song_service
    ):
        """Test de batch status con alto volumen"""
        task_ids = [generate_test_song_id() for _ in range(50)]  # Máximo
        songs = [
            create_song_dict(song_id=tid, status="processing")
            for tid in task_ids
        ]
        
        with patch('api.routes.generation.parse_comma_separated_ids') as mock_parse, \
             patch('api.routes.generation.batch_get_songs') as mock_batch:
            
            mock_parse.return_value = task_ids
            mock_batch.return_value = songs
            
            response = test_client.get(
                "/suno/generate/batch-status",
                params={"task_ids": ",".join(task_ids)}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["total_requested"] == 50


class TestStressTests:
    """Tests de stress"""
    
    @pytest.mark.stress
    @pytest.mark.asyncio
    async def test_sustained_load(
        self,
        test_client
    ):
        """Test de carga sostenida"""
        request = {
            "prompt": "A song",
            "user_id": "test-user"
        }
        
        async def make_request():
            with patch('api.routes.generation.generate_song_background'):
                response = test_client.post("/suno/generate", json=request)
                return response.status_code
        
        # Hacer 100 requests en ráfagas
        results = []
        for _ in range(10):
            batch_results = await asyncio.gather(*[
                make_request() for _ in range(10)
            ])
            results.extend(batch_results)
            await asyncio.sleep(0.1)  # Pequeña pausa entre ráfagas
        
        # Verificar que la mayoría fueron exitosas
        success_count = sum(1 for code in results if code == 202)
        assert success_count >= len(results) * 0.8
    
    @pytest.mark.stress
    @pytest.mark.asyncio
    async def test_memory_usage_under_load(
        self,
        test_client
    ):
        """Test de uso de memoria bajo carga"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        request = {
            "prompt": "A song",
            "user_id": "test-user"
        }
        
        with patch('api.routes.generation.generate_song_background'):
            # Hacer muchas requests
            for _ in range(100):
                test_client.post("/suno/generate", json=request)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # El aumento de memoria no debe ser excesivo (< 100MB)
        assert memory_increase < 100, f"Memory increase too high: {memory_increase}MB"

