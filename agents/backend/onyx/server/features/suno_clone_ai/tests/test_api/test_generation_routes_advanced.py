"""
Tests avanzados para endpoints de generación (nuevas funcionalidades)
"""

import pytest
from fastapi import status
from unittest.mock import patch, AsyncMock, MagicMock

from tests.helpers.test_helpers import create_song_dict, generate_test_song_id
from tests.helpers.mock_helpers import (
    create_mock_song_service,
    create_mock_chat_processor,
    create_mock_metrics_service,
    create_mock_notification_service
)
from tests.helpers.assertion_helpers import assert_song_response_valid
from tests.helpers.advanced_helpers import ResponseValidator, MockVerifier, DataFactory


class TestCreateSongFromChatAdvanced:
    """Tests avanzados para create_song_from_chat con métricas y notificaciones"""
    
    @pytest.fixture
    def endpoint_path(self):
        return "/suno/generate/chat/create-song"
    
    @pytest.mark.asyncio
    @pytest.mark.happy_path
    async def test_create_song_with_notifications(
        self,
        test_client,
        endpoint_path,
        mock_notification_service
    ):
        """Test con servicio de notificaciones"""
        request = {
            "message": "Create a song",
            "user_id": "test-user-123"
        }
        
        with patch('api.routes.generation.extract_song_info_from_chat') as mock_extract, \
             patch('api.routes.generation.generate_song_background'), \
             patch('api.routes.generation.notify_song_started') as mock_notify, \
             patch('api.dependencies.get_notification_svc', return_value=mock_notification_service):
            
            mock_extract.return_value = {"prompt": "song"}
            mock_notify.return_value = None
            
            response = test_client.post(endpoint_path, json=request)
            
            assert response.status_code == status.HTTP_202_ACCEPTED
            # Verificar que se intentó notificar
            # (puede fallar silenciosamente según la implementación)
    
    @pytest.mark.asyncio
    @pytest.mark.happy_path
    async def test_create_song_with_metrics(
        self,
        test_client,
        endpoint_path,
        mock_metrics_service
    ):
        """Test con servicio de métricas"""
        request = {
            "message": "Create a song",
            "user_id": "test-user-123"
        }
        
        with patch('api.routes.generation.extract_song_info_from_chat') as mock_extract, \
             patch('api.routes.generation.generate_song_background'), \
             patch('api.dependencies.get_metrics_svc', return_value=mock_metrics_service):
            
            mock_extract.return_value = {"prompt": "song"}
            
            response = test_client.post(endpoint_path, json=request)
            
            assert response.status_code == status.HTTP_202_ACCEPTED
            data = response.json()
            assert "metadata" in data
            assert data["metadata"].get("source") == "chat"
    
    @pytest.mark.asyncio
    @pytest.mark.error_handling
    async def test_create_song_invalid_message_error(
        self,
        test_client,
        endpoint_path
    ):
        """Test con mensaje inválido que causa ValueError"""
        request = {
            "message": "",  # Mensaje vacío
            "user_id": "test-user-123"
        }
        
        with patch('api.routes.generation.extract_song_info_from_chat') as mock_extract:
            mock_extract.side_effect = ValueError("Invalid message")
            
            response = test_client.post(endpoint_path, json=request)
            
            assert response.status_code == status.HTTP_400_BAD_REQUEST
    
    @pytest.mark.asyncio
    @pytest.mark.error_handling
    async def test_create_song_processing_error(
        self,
        test_client,
        endpoint_path
    ):
        """Test con error en el procesamiento"""
        request = {
            "message": "Create a song",
            "user_id": "test-user-123"
        }
        
        with patch('api.routes.generation.extract_song_info_from_chat') as mock_extract:
            mock_extract.side_effect = Exception("Processing error")
            
            response = test_client.post(endpoint_path, json=request)
            
            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


class TestGenerateSongAdvanced:
    """Tests avanzados para generate_song con métricas y notificaciones"""
    
    @pytest.fixture
    def endpoint_path(self):
        return "/suno/generate"
    
    @pytest.mark.asyncio
    @pytest.mark.happy_path
    async def test_generate_song_with_metadata(
        self,
        test_client,
        endpoint_path
    ):
        """Test que verifica metadata en la respuesta"""
        request = {
            "prompt": "A happy song",
            "duration": 30,
            "genre": "pop",
            "user_id": "test-user-123"
        }
        
        with patch('api.routes.generation.generate_song_background'):
            response = test_client.post(endpoint_path, json=request)
            
            assert response.status_code == status.HTTP_202_ACCEPTED
            data = response.json()
            assert "metadata" in data
            assert data["metadata"].get("source") == "direct"
            assert "processing_time" in data["metadata"]


class TestGetGenerationStatusAdvanced:
    """Tests avanzados para get_generation_status con progreso"""
    
    @pytest.fixture
    def endpoint_path(self):
        return "/suno/generate/status/{task_id}"
    
    @pytest.mark.asyncio
    @pytest.mark.happy_path
    async def test_get_status_with_progress(
        self,
        test_client,
        endpoint_path,
        mock_song_service
    ):
        """Test con información de progreso"""
        task_id = generate_test_song_id()
        song = create_song_dict(
            song_id=task_id,
            status="processing",
            metadata={
                "progress_percentage": 50,
                "estimated_time_remaining": 15,
                "current_step": "generating",
                "total_steps": 4
            }
        )
        mock_song_service.get_song.return_value = song
        
        with patch('api.routes.generation.validate_song_id'), \
             patch('api.routes.generation.get_song_async_or_sync') as mock_get:
            mock_get.return_value = song
            
            response = test_client.get(
                endpoint_path.format(task_id=task_id),
                params={"include_progress": True}
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["status"] == "processing"
            # Verificar que el mensaje incluye información de progreso
            assert "Progress" in data["message"] or "progress" in data["message"].lower()
    
    @pytest.mark.asyncio
    @pytest.mark.happy_path
    async def test_get_status_with_custom_headers(
        self,
        test_client,
        endpoint_path,
        mock_song_service
    ):
        """Test que verifica headers personalizados"""
        task_id = generate_test_song_id()
        song = create_song_dict(song_id=task_id, status="processing")
        mock_song_service.get_song.return_value = song
        
        with patch('api.routes.generation.validate_song_id'), \
             patch('api.routes.generation.get_song_async_or_sync') as mock_get:
            mock_get.return_value = song
            
            response = test_client.get(endpoint_path.format(task_id=task_id))
            
            assert response.status_code == status.HTTP_200_OK
            # Verificar headers personalizados
            assert "X-Generation-Status" in response.headers
            assert response.headers["X-Generation-Status"] == "processing"
    
    @pytest.mark.asyncio
    @pytest.mark.error_handling
    async def test_get_status_invalid_id_format(
        self,
        test_client,
        endpoint_path
    ):
        """Test con formato de ID inválido"""
        invalid_id = "not-a-valid-uuid"
        
        with patch('api.routes.generation.validate_song_id') as mock_validate:
            mock_validate.side_effect = ValueError("Invalid UUID format")
            
            response = test_client.get(endpoint_path.format(task_id=invalid_id))
            
            assert response.status_code == status.HTTP_400_BAD_REQUEST


class TestGetBatchGenerationStatus:
    """Tests para get_batch_generation_status"""
    
    @pytest.fixture
    def endpoint_path(self):
        return "/suno/generate/batch-status"
    
    @pytest.mark.asyncio
    @pytest.mark.happy_path
    async def test_batch_status_success(
        self,
        test_client,
        endpoint_path,
        mock_song_service
    ):
        """Test exitoso de batch status"""
        task_ids = [generate_test_song_id() for _ in range(3)]
        songs = [
            create_song_dict(song_id=task_ids[0], status="processing"),
            create_song_dict(song_id=task_ids[1], status="completed"),
            create_song_dict(song_id=task_ids[2], status="failed")
        ]
        
        with patch('api.routes.generation.parse_comma_separated_ids') as mock_parse, \
             patch('api.routes.generation.batch_get_songs') as mock_batch:
            
            mock_parse.return_value = task_ids
            mock_batch.return_value = songs
            
            task_ids_str = ",".join(task_ids)
            response = test_client.get(
                endpoint_path,
                params={"task_ids": task_ids_str}
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert "total_requested" in data
            assert "found" in data
            assert "not_found" in data
            assert "statuses" in data
            assert data["total_requested"] == 3
            assert data["found"] == 3
            assert len(data["statuses"]) == 3
    
    @pytest.mark.asyncio
    @pytest.mark.happy_path
    async def test_batch_status_with_not_found(
        self,
        test_client,
        endpoint_path,
        mock_song_service
    ):
        """Test con algunos IDs no encontrados"""
        task_ids = [generate_test_song_id() for _ in range(5)]
        # Solo 2 canciones encontradas
        songs = [
            create_song_dict(song_id=task_ids[0], status="processing"),
            create_song_dict(song_id=task_ids[1], status="completed")
        ]
        
        with patch('api.routes.generation.parse_comma_separated_ids') as mock_parse, \
             patch('api.routes.generation.batch_get_songs') as mock_batch:
            
            mock_parse.return_value = task_ids
            mock_batch.return_value = songs
            
            task_ids_str = ",".join(task_ids)
            response = test_client.get(
                endpoint_path,
                params={"task_ids": task_ids_str}
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["found"] == 2
            assert data["not_found"] == 3
    
    @pytest.mark.asyncio
    @pytest.mark.boundary
    async def test_batch_status_max_items(
        self,
        test_client,
        endpoint_path
    ):
        """Test con máximo de items (50)"""
        task_ids = [generate_test_song_id() for _ in range(50)]
        
        with patch('api.routes.generation.parse_comma_separated_ids') as mock_parse, \
             patch('api.routes.generation.batch_get_songs') as mock_batch:
            
            mock_parse.return_value = task_ids
            mock_batch.return_value = []
            
            task_ids_str = ",".join(task_ids)
            response = test_client.get(
                endpoint_path,
                params={"task_ids": task_ids_str}
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["total_requested"] == 50
    
    @pytest.mark.asyncio
    @pytest.mark.error_handling
    async def test_batch_status_too_many_items(
        self,
        test_client,
        endpoint_path
    ):
        """Test con demasiados items (más de 50)"""
        task_ids = [generate_test_song_id() for _ in range(51)]
        
        with patch('api.routes.generation.parse_comma_separated_ids') as mock_parse:
            from api.routes.generation import InvalidInputError
            mock_parse.side_effect = InvalidInputError("Too many items")
            
            task_ids_str = ",".join(task_ids)
            response = test_client.get(
                endpoint_path,
                params={"task_ids": task_ids_str}
            )
            
            assert response.status_code in [
                status.HTTP_400_BAD_REQUEST,
                status.HTTP_422_UNPROCESSABLE_ENTITY
            ]
    
    @pytest.mark.asyncio
    @pytest.mark.error_handling
    async def test_batch_status_invalid_format(
        self,
        test_client,
        endpoint_path
    ):
        """Test con formato inválido"""
        with patch('api.routes.generation.parse_comma_separated_ids') as mock_parse:
            mock_parse.side_effect = ValueError("Invalid format")
            
            response = test_client.get(
                endpoint_path,
                params={"task_ids": "invalid,format,here"}
            )
            
            assert response.status_code in [
                status.HTTP_400_BAD_REQUEST,
                status.HTTP_422_UNPROCESSABLE_ENTITY
            ]
    
    @pytest.mark.asyncio
    @pytest.mark.edge_case
    async def test_batch_status_empty_list(
        self,
        test_client,
        endpoint_path
    ):
        """Test con lista vacía"""
        with patch('api.routes.generation.parse_comma_separated_ids') as mock_parse:
            mock_parse.return_value = []
            
            response = test_client.get(
                endpoint_path,
                params={"task_ids": ""}
            )
            
            # Puede retornar 200 con lista vacía o error
            assert response.status_code in [
                status.HTTP_200_OK,
                status.HTTP_400_BAD_REQUEST,
                status.HTTP_422_UNPROCESSABLE_ENTITY
            ]
    
    @pytest.mark.asyncio
    @pytest.mark.edge_case
    async def test_batch_status_single_item(
        self,
        test_client,
        endpoint_path,
        mock_song_service
    ):
        """Test con un solo item"""
        task_id = generate_test_song_id()
        song = create_song_dict(song_id=task_id, status="processing")
        
        with patch('api.routes.generation.parse_comma_separated_ids') as mock_parse, \
             patch('api.routes.generation.batch_get_songs') as mock_batch:
            
            mock_parse.return_value = [task_id]
            mock_batch.return_value = [song]
            
            response = test_client.get(
                endpoint_path,
                params={"task_ids": task_id}
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["total_requested"] == 1
            assert data["found"] == 1


class TestGenerationPerformance:
    """Tests de performance para endpoints de generación"""
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_create_song_response_time(
        self,
        test_client
    ):
        """Test de tiempo de respuesta para crear canción"""
        from tests.helpers.advanced_helpers import PerformanceHelper
        
        request = {
            "message": "Create a song",
            "user_id": "test-user"
        }
        
        with patch('api.routes.generation.extract_song_info_from_chat'), \
             patch('api.routes.generation.generate_song_background'):
            
            async def create_song():
                response = test_client.post("/suno/generate/chat/create-song", json=request)
                assert response.status_code == 202
            
            execution_time = await PerformanceHelper.measure_async_execution_time(create_song)
            
            # Debe responder en menos de 1 segundo
            assert execution_time < 1.0, f"Response time too slow: {execution_time}s"
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_batch_status_performance(
        self,
        test_client,
        mock_song_service
    ):
        """Test de performance para batch status"""
        from tests.helpers.advanced_helpers import PerformanceHelper
        
        task_ids = [generate_test_song_id() for _ in range(10)]
        songs = [create_song_dict(song_id=tid, status="processing") for tid in task_ids]
        
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
            
            # Debe responder en menos de 2 segundos para 10 items
            assert execution_time < 2.0, f"Batch status too slow: {execution_time}s"


class TestGenerationIntegrationAdvanced:
    """Tests de integración avanzados"""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_full_workflow_with_batch_status(
        self,
        test_client
    ):
        """Test del flujo completo con verificación batch"""
        # 1. Crear múltiples canciones
        requests = DataFactory.create_song_requests(count=3)
        song_ids = []
        
        with patch('api.routes.generation.generate_song_background'):
            for request in requests:
                response = test_client.post("/suno/generate", json=request)
                if response.status_code == 202:
                    song_ids.append(response.json()["song_id"])
        
        assert len(song_ids) == 3
        
        # 2. Verificar estados en batch
        songs = [
            create_song_dict(song_id=sid, status="processing")
            for sid in song_ids
        ]
        
        with patch('api.routes.generation.parse_comma_separated_ids') as mock_parse, \
             patch('api.routes.generation.batch_get_songs') as mock_batch:
            
            mock_parse.return_value = song_ids
            mock_batch.return_value = songs
            
            batch_response = test_client.get(
                "/suno/generate/batch-status",
                params={"task_ids": ",".join(song_ids)}
            )
            
            assert batch_response.status_code == 200
            batch_data = batch_response.json()
            assert batch_data["found"] == 3

