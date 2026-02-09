"""
Tests de integración para flujos completos end-to-end
"""

import pytest
from unittest.mock import patch, AsyncMock
from tests.helpers.test_helpers import create_song_dict, generate_test_song_id, create_mock_audio, save_test_audio
from tests.helpers.advanced_helpers import AsyncTestHelper, ResponseValidator, DataFactory


class TestFullGenerationWorkflow:
    """Tests para el flujo completo de generación"""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_complete_song_generation_workflow(
        self,
        test_client,
        mock_song_service,
        temp_audio_dir
    ):
        """
        Test del flujo completo:
        1. Crear canción desde chat
        2. Verificar estado (processing)
        3. Simular completado
        4. Verificar estado (completed)
        5. Descargar canción
        """
        # 1. Crear canción desde chat
        create_request = {
            "message": "Create a happy pop song",
            "user_id": "test-user-123"
        }
        
        with patch('api.routes.generation.extract_song_info_from_chat') as mock_extract, \
             patch('api.routes.generation.generate_song_background'):
            
            mock_extract.return_value = {
                "prompt": "happy pop song",
                "genre": "pop",
                "duration": 30
            }
            
            create_response = test_client.post(
                "/suno/generate/chat/create-song",
                json=create_request
            )
            
            assert create_response.status_code == 202
            song_id = create_response.json()["song_id"]
        
        # 2. Verificar estado inicial (processing)
        song_processing = create_song_dict(
            song_id=song_id,
            status="processing",
            message="Generating..."
        )
        mock_song_service.get_song.return_value = song_processing
        
        with patch('api.routes.generation.validate_song_id'), \
             patch('api.routes.generation.get_song_async_or_sync') as mock_get:
            mock_get.return_value = song_processing
            
            status_response = test_client.get(f"/suno/generate/status/{song_id}")
            
            assert status_response.status_code == 200
            assert status_response.json()["status"] == "processing"
        
        # 3. Simular completado
        audio_file = temp_audio_dir / f"{song_id}.wav"
        audio = create_mock_audio()
        save_test_audio(audio, audio_file)
        
        song_completed = create_song_dict(
            song_id=song_id,
            status="completed",
            message="Song generated successfully",
            file_path=str(audio_file)
        )
        mock_song_service.get_song.return_value = song_completed
        
        # 4. Verificar estado final (completed)
        with patch('api.routes.generation.validate_song_id'), \
             patch('api.routes.generation.get_song_async_or_sync') as mock_get:
            mock_get.return_value = song_completed
            
            status_response = test_client.get(f"/suno/generate/status/{song_id}")
            
            assert status_response.status_code == 200
            assert status_response.json()["status"] == "completed"
        
        # 5. Obtener información de la canción
        with patch('api.routes.songs.validate_song_id'), \
             patch('api.routes.songs.get_song_async_or_sync') as mock_get, \
             patch('api.routes.songs.ensure_song_exists') as mock_ensure:
            mock_get.return_value = song_completed
            mock_ensure.return_value = song_completed
            
            get_response = test_client.get(f"/suno/songs/{song_id}")
            
            assert get_response.status_code == 200
            ResponseValidator.validate_response_structure(
                get_response.json(),
                required_fields=["song_id", "status"]
            )


class TestBatchOperations:
    """Tests para operaciones en lote"""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_create_multiple_songs(
        self,
        test_client
    ):
        """Test de creación de múltiples canciones"""
        requests = DataFactory.create_song_requests(count=5)
        
        song_ids = []
        
        with patch('api.routes.generation.generate_song_background'):
            for request in requests:
                response = test_client.post("/suno/generate", json=request)
                
                if response.status_code == 202:
                    song_ids.append(response.json()["song_id"])
        
        assert len(song_ids) == 5
        # Verificar que todos los IDs son únicos
        assert len(set(song_ids)) == 5
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_list_and_filter_songs(
        self,
        test_client,
        mock_song_service
    ):
        """Test de listado y filtrado de canciones"""
        user_id = "test-user-123"
        songs = [
            create_song_dict(song_id=f"song-{i}", user_id=user_id)
            for i in range(10)
        ]
        mock_song_service.list_songs.return_value = songs
        
        with patch('api.routes.songs.get_song_async_or_sync') as mock_get:
            mock_get.return_value = songs
            
            # Listar todas
            all_response = test_client.get("/suno/songs")
            assert all_response.status_code == 200
            
            # Filtrar por usuario
            filtered_response = test_client.get(
                "/suno/songs",
                params={"user_id": user_id}
            )
            assert filtered_response.status_code == 200
            
            filtered_songs = filtered_response.json()["songs"]
            assert all(song.get("user_id") == user_id for song in filtered_songs)


class TestErrorRecovery:
    """Tests de recuperación de errores"""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_generation_failure_recovery(
        self,
        test_client,
        mock_song_service
    ):
        """Test de recuperación cuando falla la generación"""
        song_id = generate_test_song_id()
        
        # Simular fallo
        song_failed = create_song_dict(
            song_id=song_id,
            status="failed",
            message="Generation failed: timeout"
        )
        mock_song_service.get_song.return_value = song_failed
        
        with patch('api.routes.generation.validate_song_id'), \
             patch('api.routes.generation.get_song_async_or_sync') as mock_get:
            mock_get.return_value = song_failed
            
            status_response = test_client.get(f"/suno/generate/status/{song_id}")
            
            assert status_response.status_code == 200
            assert status_response.json()["status"] == "failed"
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_retry_after_failure(
        self,
        test_client
    ):
        """Test de reintento después de fallo"""
        # Crear nueva canción después de un fallo
        request = {
            "prompt": "Retry song",
            "user_id": "test-user"
        }
        
        with patch('api.routes.generation.generate_song_background'):
            response = test_client.post("/suno/generate", json=request)
            
            assert response.status_code == 202
            # Debe poder crear una nueva canción
            assert "song_id" in response.json()


class TestConcurrentOperations:
    """Tests de operaciones concurrentes"""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_concurrent_song_creation(
        self,
        test_client
    ):
        """Test de creación concurrente de canciones"""
        import asyncio
        
        async def create_song(request):
            with patch('api.routes.generation.generate_song_background'):
                response = test_client.post("/suno/generate", json=request)
                return response.status_code == 202
        
        requests = DataFactory.create_song_requests(count=10)
        
        # Crear canciones concurrentemente
        results = await asyncio.gather(*[
            create_song(req) for req in requests
        ])
        
        # Verificar que todas se crearon exitosamente
        assert all(results)
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_concurrent_status_checks(
        self,
        test_client,
        mock_song_service
    ):
        """Test de verificaciones concurrentes de estado"""
        import asyncio
        
        song_id = generate_test_song_id()
        song = create_song_dict(song_id=song_id, status="processing")
        mock_song_service.get_song.return_value = song
        
        async def check_status():
            with patch('api.routes.generation.validate_song_id'), \
                 patch('api.routes.generation.get_song_async_or_sync') as mock_get:
                mock_get.return_value = song
                response = test_client.get(f"/suno/generate/status/{song_id}")
                return response.status_code == 200
        
        # Verificar estado concurrentemente
        results = await asyncio.gather(*[
            check_status() for _ in range(10)
        ])
        
        assert all(results)

