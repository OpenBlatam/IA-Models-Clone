"""
Tests modulares para endpoints de generación de canciones
"""

import pytest
from fastapi import status
from unittest.mock import patch, AsyncMock
from tests.helpers.test_helpers import create_song_dict, generate_test_song_id
from tests.helpers.mock_helpers import (
    create_mock_song_service,
    create_mock_music_generator,
    create_mock_chat_processor
)
from tests.helpers.assertion_helpers import assert_song_response_valid


class TestCreateSongFromChat:
    """Tests para el endpoint create_song_from_chat"""
    
    @pytest.fixture
    def endpoint_path(self):
        return "/suno/chat/create-song"
    
    @pytest.fixture
    def valid_request(self):
        return {
            "message": "Create a happy pop song",
            "user_id": "test-user-123",
            "chat_history": []
        }
    
    @pytest.mark.asyncio
    async def test_create_song_from_chat_success(
        self,
        test_client,
        endpoint_path,
        valid_request,
        mock_chat_processor,
        mock_song_service
    ):
        """Test exitoso de creación de canción desde chat"""
        with patch('api.dependencies.get_chat_proc', return_value=mock_chat_processor), \
             patch('api.dependencies.get_song_service', return_value=mock_song_service):
            
            response = test_client.post(endpoint_path, json=valid_request)
            
            assert response.status_code == status.HTTP_202_ACCEPTED
            data = response.json()
            assert_song_response_valid(data)
            assert data["status"] == "processing"
            assert "song_id" in data
    
    @pytest.mark.asyncio
    async def test_create_song_from_chat_invalid_message(
        self,
        test_client,
        endpoint_path
    ):
        """Test con mensaje inválido"""
        invalid_request = {
            "message": "",  # Mensaje vacío
            "user_id": "test-user-123"
        }
        
        response = test_client.post(endpoint_path, json=invalid_request)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    @pytest.mark.asyncio
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
    async def test_create_song_from_chat_with_history(
        self,
        test_client,
        endpoint_path,
        mock_chat_processor,
        mock_song_service
    ):
        """Test con historial de chat"""
        request_with_history = {
            "message": "Make it faster",
            "user_id": "test-user-123",
            "chat_history": [
                {"role": "user", "content": "Create a song"},
                {"role": "assistant", "content": "I'll create a song for you"}
            ]
        }
        
        with patch('api.dependencies.get_chat_proc', return_value=mock_chat_processor), \
             patch('api.dependencies.get_song_service', return_value=mock_song_service):
            
            response = test_client.post(endpoint_path, json=request_with_history)
            assert response.status_code == status.HTTP_202_ACCEPTED


class TestGenerateSong:
    """Tests para el endpoint generate_song"""
    
    @pytest.fixture
    def endpoint_path(self):
        return "/suno/generate"
    
    @pytest.fixture
    def valid_request(self):
        return {
            "prompt": "A happy pop song",
            "duration": 30,
            "genre": "pop",
            "mood": "happy",
            "user_id": "test-user-123"
        }
    
    @pytest.mark.asyncio
    async def test_generate_song_success(
        self,
        test_client,
        endpoint_path,
        valid_request
    ):
        """Test exitoso de generación de canción"""
        response = test_client.post(endpoint_path, json=valid_request)
        
        assert response.status_code == status.HTTP_202_ACCEPTED
        data = response.json()
        assert_song_response_valid(data)
        assert data["status"] == "processing"
    
    @pytest.mark.asyncio
    async def test_generate_song_minimal_request(
        self,
        test_client,
        endpoint_path
    ):
        """Test con request mínimo (solo prompt)"""
        minimal_request = {
            "prompt": "A song"
        }
        
        response = test_client.post(endpoint_path, json=minimal_request)
        assert response.status_code == status.HTTP_202_ACCEPTED
    
    @pytest.mark.asyncio
    async def test_generate_song_invalid_duration(
        self,
        test_client,
        endpoint_path
    ):
        """Test con duración inválida"""
        invalid_request = {
            "prompt": "A song",
            "duration": 500  # Excede el máximo
        }
        
        response = test_client.post(endpoint_path, json=invalid_request)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    @pytest.mark.asyncio
    async def test_generate_song_empty_prompt(
        self,
        test_client,
        endpoint_path
    ):
        """Test con prompt vacío"""
        invalid_request = {
            "prompt": ""
        }
        
        response = test_client.post(endpoint_path, json=invalid_request)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    @pytest.mark.asyncio
    async def test_generate_song_boundary_duration(
        self,
        test_client,
        endpoint_path
    ):
        """Test con duración en límites"""
        # Duración mínima
        request_min = {
            "prompt": "A song",
            "duration": 1
        }
        response = test_client.post(endpoint_path, json=request_min)
        assert response.status_code == status.HTTP_202_ACCEPTED
        
        # Duración máxima
        request_max = {
            "prompt": "A song",
            "duration": 300
        }
        response = test_client.post(endpoint_path, json=request_max)
        assert response.status_code == status.HTTP_202_ACCEPTED

