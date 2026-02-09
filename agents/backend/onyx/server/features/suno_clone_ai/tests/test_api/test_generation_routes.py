"""
Tests modulares mejorados para endpoints de generación (routes/generation.py)
"""

import pytest
from fastapi import status
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient

from tests.helpers.test_helpers import create_song_dict, generate_test_song_id
from tests.helpers.mock_helpers import (
    create_mock_song_service,
    create_mock_chat_processor
)
from tests.helpers.assertion_helpers import assert_song_response_valid


class TestCreateSongFromChat:
    """Tests mejorados para create_song_from_chat"""
    
    @pytest.fixture
    def endpoint_path(self):
        return "/suno/generate/chat/create-song"
    
    @pytest.fixture
    def valid_request(self):
        return {
            "message": "Create a happy pop song with electric guitar",
            "user_id": "test-user-123",
            "chat_history": []
        }
    
    @pytest.mark.asyncio
    @pytest.mark.happy_path
    async def test_create_song_from_chat_success(
        self,
        test_client,
        endpoint_path,
        valid_request,
        mock_chat_processor,
        mock_song_service
    ):
        """Test exitoso con request completo"""
        with patch('api.routes.generation.extract_song_info_from_chat') as mock_extract, \
             patch('api.routes.generation.generate_song_background') as mock_bg:
            
            mock_extract.return_value = {
                "prompt": "happy pop song",
                "genre": "pop",
                "mood": "happy",
                "duration": 30
            }
            
            response = test_client.post(endpoint_path, json=valid_request)
            
            assert response.status_code == status.HTTP_202_ACCEPTED
            data = response.json()
            assert_song_response_valid(data)
            assert data["status"] == "processing"
            assert "song_id" in data
            assert "metadata" in data
    
    @pytest.mark.asyncio
    @pytest.mark.happy_path
    async def test_create_song_from_chat_with_history(
        self,
        test_client,
        endpoint_path,
        mock_chat_processor
    ):
        """Test con historial de chat completo"""
        request_with_history = {
            "message": "Make it faster and add drums",
            "user_id": "test-user-123",
            "chat_history": [
                {"role": "user", "content": "Create a song"},
                {"role": "assistant", "content": "I'll create a song for you"},
                {"role": "user", "content": "Make it pop style"}
            ]
        }
        
        with patch('api.routes.generation.extract_song_info_from_chat') as mock_extract:
            mock_extract.return_value = {
                "prompt": "fast pop song with drums",
                "genre": "pop",
                "duration": 30
            }
            
            response = test_client.post(endpoint_path, json=request_with_history)
            assert response.status_code == status.HTTP_202_ACCEPTED
            mock_extract.assert_called_once()
    
    @pytest.mark.asyncio
    @pytest.mark.error_handling
    async def test_create_song_from_chat_empty_message(
        self,
        test_client,
        endpoint_path
    ):
        """Test con mensaje vacío"""
        invalid_request = {
            "message": "",
            "user_id": "test-user-123"
        }
        
        response = test_client.post(endpoint_path, json=invalid_request)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    @pytest.mark.asyncio
    @pytest.mark.error_handling
    async def test_create_song_from_chat_missing_message(
        self,
        test_client,
        endpoint_path
    ):
        """Test sin mensaje"""
        invalid_request = {
            "user_id": "test-user-123"
        }
        
        response = test_client.post(endpoint_path, json=invalid_request)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    @pytest.mark.asyncio
    @pytest.mark.boundary
    async def test_create_song_from_chat_long_message(
        self,
        test_client,
        endpoint_path
    ):
        """Test con mensaje muy largo (límite)"""
        long_message = "A" * 500  # Máximo permitido
        request = {
            "message": long_message,
            "user_id": "test-user-123"
        }
        
        with patch('api.routes.generation.extract_song_info_from_chat'):
            response = test_client.post(endpoint_path, json=request)
            # Debe aceptar el límite máximo
            assert response.status_code in [status.HTTP_202_ACCEPTED, status.HTTP_422_UNPROCESSABLE_ENTITY]
    
    @pytest.mark.asyncio
    @pytest.mark.edge_case
    async def test_create_song_from_chat_no_user_id(
        self,
        test_client,
        endpoint_path
    ):
        """Test sin user_id (debe funcionar)"""
        request = {
            "message": "Create a song"
        }
        
        with patch('api.routes.generation.extract_song_info_from_chat') as mock_extract:
            mock_extract.return_value = {"prompt": "song"}
            response = test_client.post(endpoint_path, json=request)
            assert response.status_code == status.HTTP_202_ACCEPTED


class TestGenerateSong:
    """Tests mejorados para generate_song"""
    
    @pytest.fixture
    def endpoint_path(self):
        return "/suno/generate"
    
    @pytest.fixture
    def valid_request(self):
        return {
            "prompt": "A happy pop song with electric guitar and drums",
            "duration": 30,
            "genre": "pop",
            "mood": "happy",
            "user_id": "test-user-123"
        }
    
    @pytest.mark.asyncio
    @pytest.mark.happy_path
    async def test_generate_song_success(
        self,
        test_client,
        endpoint_path,
        valid_request
    ):
        """Test exitoso con todos los parámetros"""
        with patch('api.routes.generation.generate_song_background'):
            response = test_client.post(endpoint_path, json=valid_request)
            
            assert response.status_code == status.HTTP_202_ACCEPTED
            data = response.json()
            assert_song_response_valid(data)
            assert data["status"] == "processing"
            assert "metadata" in data
    
    @pytest.mark.asyncio
    @pytest.mark.happy_path
    async def test_generate_song_minimal(
        self,
        test_client,
        endpoint_path
    ):
        """Test con request mínimo (solo prompt)"""
        minimal_request = {
            "prompt": "A song"
        }
        
        with patch('api.routes.generation.generate_song_background'):
            response = test_client.post(endpoint_path, json=minimal_request)
            assert response.status_code == status.HTTP_202_ACCEPTED
    
    @pytest.mark.asyncio
    @pytest.mark.boundary
    async def test_generate_song_duration_min(
        self,
        test_client,
        endpoint_path
    ):
        """Test con duración mínima"""
        request = {
            "prompt": "A song",
            "duration": 1
        }
        
        with patch('api.routes.generation.generate_song_background'):
            response = test_client.post(endpoint_path, json=request)
            assert response.status_code == status.HTTP_202_ACCEPTED
    
    @pytest.mark.asyncio
    @pytest.mark.boundary
    async def test_generate_song_duration_max(
        self,
        test_client,
        endpoint_path
    ):
        """Test con duración máxima"""
        request = {
            "prompt": "A song",
            "duration": 300
        }
        
        with patch('api.routes.generation.generate_song_background'):
            response = test_client.post(endpoint_path, json=request)
            assert response.status_code == status.HTTP_202_ACCEPTED
    
    @pytest.mark.asyncio
    @pytest.mark.error_handling
    async def test_generate_song_duration_too_high(
        self,
        test_client,
        endpoint_path
    ):
        """Test con duración excesiva"""
        request = {
            "prompt": "A song",
            "duration": 500  # Excede máximo
        }
        
        response = test_client.post(endpoint_path, json=request)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    @pytest.mark.asyncio
    @pytest.mark.error_handling
    async def test_generate_song_duration_zero(
        self,
        test_client,
        endpoint_path
    ):
        """Test con duración cero"""
        request = {
            "prompt": "A song",
            "duration": 0
        }
        
        response = test_client.post(endpoint_path, json=request)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    @pytest.mark.asyncio
    @pytest.mark.error_handling
    async def test_generate_song_empty_prompt(
        self,
        test_client,
        endpoint_path
    ):
        """Test con prompt vacío"""
        request = {
            "prompt": ""
        }
        
        response = test_client.post(endpoint_path, json=request)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    @pytest.mark.asyncio
    @pytest.mark.edge_case
    async def test_generate_song_special_characters(
        self,
        test_client,
        endpoint_path
    ):
        """Test con caracteres especiales en prompt"""
        request = {
            "prompt": "A song with émojis 🎵🎶 and spéciál chàracters!"
        }
        
        with patch('api.routes.generation.generate_song_background'):
            response = test_client.post(endpoint_path, json=request)
            assert response.status_code == status.HTTP_202_ACCEPTED


class TestGetGenerationStatus:
    """Tests mejorados para get_generation_status"""
    
    @pytest.fixture
    def endpoint_path(self):
        return "/suno/generate/status/{task_id}"
    
    @pytest.mark.asyncio
    @pytest.mark.happy_path
    async def test_get_generation_status_processing(
        self,
        test_client,
        endpoint_path,
        mock_song_service
    ):
        """Test con estado processing"""
        task_id = generate_test_song_id()
        song = create_song_dict(
            song_id=task_id,
            status="processing",
            message="Generating..."
        )
        mock_song_service.get_song.return_value = song
        
        with patch('api.routes.generation.validate_song_id'), \
             patch('api.dependencies.get_song_service', return_value=mock_song_service):
            
            response = test_client.get(endpoint_path.format(task_id=task_id))
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["status"] == "processing"
            assert data["song_id"] == task_id
            # Verificar headers de cache
            assert "Cache-Control" in response.headers
    
    @pytest.mark.asyncio
    @pytest.mark.happy_path
    async def test_get_generation_status_completed(
        self,
        test_client,
        endpoint_path,
        mock_song_service
    ):
        """Test con estado completed"""
        task_id = generate_test_song_id()
        song = create_song_dict(
            song_id=task_id,
            status="completed",
            message="Song generated successfully"
        )
        mock_song_service.get_song.return_value = song
        
        with patch('api.routes.generation.validate_song_id'), \
             patch('api.dependencies.get_song_service', return_value=mock_song_service):
            
            response = test_client.get(endpoint_path.format(task_id=task_id))
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["status"] == "completed"
    
    @pytest.mark.asyncio
    @pytest.mark.happy_path
    async def test_get_generation_status_failed(
        self,
        test_client,
        endpoint_path,
        mock_song_service
    ):
        """Test con estado failed"""
        task_id = generate_test_song_id()
        song = create_song_dict(
            song_id=task_id,
            status="failed",
            message="Generation failed"
        )
        mock_song_service.get_song.return_value = song
        
        with patch('api.routes.generation.validate_song_id'), \
             patch('api.dependencies.get_song_service', return_value=mock_song_service):
            
            response = test_client.get(endpoint_path.format(task_id=task_id))
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["status"] == "failed"
    
    @pytest.mark.asyncio
    @pytest.mark.error_handling
    async def test_get_generation_status_not_found(
        self,
        test_client,
        endpoint_path,
        mock_song_service
    ):
        """Test cuando la tarea no existe"""
        task_id = generate_test_song_id()
        mock_song_service.get_song.return_value = None
        
        with patch('api.routes.generation.validate_song_id'), \
             patch('api.dependencies.get_song_service', return_value=mock_song_service):
            
            response = test_client.get(endpoint_path.format(task_id=task_id))
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["status"] == "not_found"
    
    @pytest.mark.asyncio
    @pytest.mark.error_handling
    async def test_get_generation_status_invalid_id(
        self,
        test_client,
        endpoint_path
    ):
        """Test con ID inválido"""
        invalid_id = "not-a-valid-uuid"
        
        with patch('api.routes.generation.validate_song_id') as mock_validate:
            mock_validate.side_effect = ValueError("Invalid song ID")
            
            response = test_client.get(endpoint_path.format(task_id=invalid_id))
            # Debe validar y retornar error
            assert response.status_code in [status.HTTP_400_BAD_REQUEST, status.HTTP_422_UNPROCESSABLE_ENTITY]
    
    @pytest.mark.asyncio
    @pytest.mark.edge_case
    async def test_get_generation_status_unknown_status(
        self,
        test_client,
        endpoint_path,
        mock_song_service
    ):
        """Test con estado desconocido"""
        task_id = generate_test_song_id()
        song = create_song_dict(
            song_id=task_id,
            status="unknown_status"
        )
        mock_song_service.get_song.return_value = song
        
        with patch('api.routes.generation.validate_song_id'), \
             patch('api.dependencies.get_song_service', return_value=mock_song_service):
            
            response = test_client.get(endpoint_path.format(task_id=task_id))
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["status"] == "unknown_status"
    
    @pytest.mark.asyncio
    @pytest.mark.edge_case
    async def test_get_generation_status_cache_headers(
        self,
        test_client,
        endpoint_path,
        mock_song_service
    ):
        """Test que verifica headers de cache"""
        task_id = generate_test_song_id()
        song = create_song_dict(song_id=task_id, status="processing")
        mock_song_service.get_song.return_value = song
        
        with patch('api.routes.generation.validate_song_id'), \
             patch('api.dependencies.get_song_service', return_value=mock_song_service):
            
            response = test_client.get(endpoint_path.format(task_id=task_id))
            
            assert response.status_code == status.HTTP_200_OK
            # Verificar que tiene headers de cache
            cache_control = response.headers.get("Cache-Control", "")
            assert "max-age=5" in cache_control or "public" in cache_control


class TestGenerationIntegration:
    """Tests de integración para flujo completo de generación"""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_full_generation_flow(
        self,
        test_client,
        mock_song_service,
        mock_chat_processor
    ):
        """Test del flujo completo: crear -> verificar estado -> completar"""
        # 1. Crear canción desde chat
        create_request = {
            "message": "Create a happy song",
            "user_id": "test-user-123"
        }
        
        with patch('api.routes.generation.extract_song_info_from_chat') as mock_extract, \
             patch('api.routes.generation.generate_song_background'):
            
            mock_extract.return_value = {"prompt": "happy song"}
            
            create_response = test_client.post(
                "/suno/generate/chat/create-song",
                json=create_request
            )
            
            assert create_response.status_code == status.HTTP_202_ACCEPTED
            song_id = create_response.json()["song_id"]
            
            # 2. Verificar estado
            song = create_song_dict(song_id=song_id, status="processing")
            mock_song_service.get_song.return_value = song
            
            with patch('api.routes.generation.validate_song_id'), \
                 patch('api.dependencies.get_song_service', return_value=mock_song_service):
                
                status_response = test_client.get(
                    f"/suno/generate/status/{song_id}"
                )
                
                assert status_response.status_code == status.HTTP_200_OK
                assert status_response.json()["status"] == "processing"

