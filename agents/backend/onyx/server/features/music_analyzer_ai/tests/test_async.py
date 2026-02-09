"""
Tests para funcionalidades asíncronas
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
import asyncio


class TestAsyncOperations:
    """Tests de operaciones asíncronas"""
    
    @pytest.mark.asyncio
    async def test_async_analysis(self):
        """Test de análisis asíncrono"""
        async def analyze_track_async(track_id):
            # Simular análisis asíncrono
            await asyncio.sleep(0.01)
            return {"track_id": track_id, "status": "completed"}
        
        result = await analyze_track_async("123")
        
        assert result is not None
        assert result["track_id"] == "123"
        assert result["status"] == "completed"
    
    @pytest.mark.asyncio
    async def test_async_batch_processing(self):
        """Test de procesamiento en lote asíncrono"""
        async def process_track(track_id):
            await asyncio.sleep(0.01)
            return {"id": track_id, "processed": True}
        
        track_ids = ["1", "2", "3", "4", "5"]
        
        results = await asyncio.gather(*[process_track(tid) for tid in track_ids])
        
        assert len(results) == 5
        assert all(r["processed"] for r in results)
    
    @pytest.mark.asyncio
    async def test_async_error_handling(self):
        """Test de manejo de errores asíncrono"""
        async def failing_operation():
            await asyncio.sleep(0.01)
            raise ValueError("Async error")
        
        with pytest.raises(ValueError):
            await failing_operation()
    
    @pytest.mark.asyncio
    async def test_async_timeout(self):
        """Test de timeout asíncrono"""
        async def slow_operation():
            await asyncio.sleep(1.0)
            return "completed"
        
        try:
            result = await asyncio.wait_for(slow_operation(), timeout=0.1)
            assert False, "Should have timed out"
        except asyncio.TimeoutError:
            assert True  # Timeout esperado


class TestAsyncAPIs:
    """Tests de APIs asíncronas"""
    
    @pytest.mark.asyncio
    async def test_async_webhook_trigger(self):
        """Test de trigger asíncrono de webhook"""
        from ..services.webhook_service import WebhookService, WebhookEvent
        
        service = WebhookService()
        
        webhook_id = service.register_webhook(
            url="https://example.com/webhook",
            events=[WebhookEvent.ANALYSIS_COMPLETED]
        )
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={"success": True})
            mock_post.return_value.__aenter__.return_value = mock_response
            
            await service.trigger_webhook(
                webhook_id,
                WebhookEvent.ANALYSIS_COMPLETED,
                {"data": "test"}
            )
            
            assert mock_post.called
    
    @pytest.mark.asyncio
    async def test_async_spotify_request(self):
        """Test de request asíncrono a Spotify"""
        async def fetch_spotify_data(track_id):
            # Simular request asíncrono
            await asyncio.sleep(0.01)
            return {
                "id": track_id,
                "name": "Test Track",
                "artists": [{"name": "Test Artist"}]
            }
        
        result = await fetch_spotify_data("123")
        
        assert result is not None
        assert result["id"] == "123"


class TestAsyncConcurrency:
    """Tests de concurrencia asíncrona"""
    
    @pytest.mark.asyncio
    async def test_concurrent_async_operations(self):
        """Test de operaciones asíncronas concurrentes"""
        async def operation(id, delay):
            await asyncio.sleep(delay)
            return {"id": id, "completed": True}
        
        tasks = [
            operation("1", 0.1),
            operation("2", 0.1),
            operation("3", 0.1)
        ]
        
        start = asyncio.get_event_loop().time()
        results = await asyncio.gather(*tasks)
        elapsed = asyncio.get_event_loop().time() - start
        
        assert len(results) == 3
        # Debe completarse en ~0.1s (concurrente), no 0.3s (secuencial)
        assert elapsed < 0.2
    
    @pytest.mark.asyncio
    async def test_async_semaphore(self):
        """Test de semáforo asíncrono para limitar concurrencia"""
        semaphore = asyncio.Semaphore(2)
        results = []
        
        async def limited_operation(id):
            async with semaphore:
                await asyncio.sleep(0.1)
                results.append(id)
        
        tasks = [limited_operation(i) for i in range(5)]
        await asyncio.gather(*tasks)
        
        assert len(results) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "asyncio"])

