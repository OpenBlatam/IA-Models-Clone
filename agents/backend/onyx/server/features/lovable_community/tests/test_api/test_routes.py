"""
Tests modulares para endpoints de la API
"""

import pytest
from fastapi import status
from unittest.mock import patch, Mock
from tests.helpers.test_helpers import (
    create_publish_request,
    create_remix_request,
    create_vote_request,
    create_search_request,
    create_chat_dict
)
from tests.helpers.mock_helpers import create_mock_chat_service
from tests.helpers.assertion_helpers import (
    assert_chat_response_valid,
    assert_chat_list_valid,
    assert_pagination_valid,
    assert_vote_response_valid,
    assert_remix_response_valid
)


class TestPublishChat:
    """Tests para publish_chat endpoint"""
    
    @pytest.fixture
    def endpoint_path(self):
        return "/community/publish"
    
    @pytest.mark.asyncio
    @pytest.mark.api
    @pytest.mark.happy_path
    async def test_publish_chat_success(
        self,
        test_client,
        endpoint_path,
        mock_chat_service,
        sample_user_id
    ):
        """Test exitoso de publicación"""
        chat_data = create_chat_dict(user_id=sample_user_id)
        mock_chat_service.publish_chat.return_value = Mock(**chat_data)
        
        request_data = create_publish_request()
        
        with patch('api.routes.get_user_id', return_value=sample_user_id), \
             patch('api.routes.get_chat_service', return_value=mock_chat_service), \
             patch('api.routes.chat_to_response') as mock_to_response:
            
            mock_to_response.return_value = chat_data
            
            response = test_client.post(
                endpoint_path,
                json=request_data,
                headers={"X-User-ID": sample_user_id}
            )
            
            assert response.status_code == status.HTTP_201_CREATED
            data = response.json()
            assert_chat_response_valid(data)
    
    @pytest.mark.asyncio
    @pytest.mark.api
    @pytest.mark.error_handling
    async def test_publish_chat_invalid_data(
        self,
        test_client,
        endpoint_path,
        sample_user_id
    ):
        """Test con datos inválidos"""
        invalid_request = {
            "title": "",  # Título vacío
            "chat_content": "{}"
        }
        
        response = test_client.post(
            endpoint_path,
            json=invalid_request,
            headers={"X-User-ID": sample_user_id}
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestListChats:
    """Tests para list_chats endpoint"""
    
    @pytest.fixture
    def endpoint_path(self):
        return "/community/chats"
    
    @pytest.mark.asyncio
    @pytest.mark.api
    @pytest.mark.happy_path
    async def test_list_chats_success(
        self,
        test_client,
        endpoint_path,
        mock_chat_service
    ):
        """Test exitoso de listado"""
        chats = [
            create_chat_dict(chat_id=f"chat-{i}", title=f"Chat {i}")
            for i in range(5)
        ]
        mock_chat_service.get_top_chats.return_value = [Mock(**chat) for chat in chats]
        
        with patch('api.routes.get_chat_service', return_value=mock_chat_service), \
             patch('api.routes.chats_to_responses') as mock_to_responses:
            
            mock_to_responses.return_value = chats
            
            response = test_client.get(endpoint_path)
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert_pagination_valid(data)
            assert_chat_list_valid(data.get("chats", []), min_count=0)


class TestGetChat:
    """Tests para get_chat endpoint"""
    
    @pytest.fixture
    def endpoint_path(self):
        return "/community/chats/{chat_id}"
    
    @pytest.mark.asyncio
    @pytest.mark.api
    @pytest.mark.happy_path
    async def test_get_chat_success(
        self,
        test_client,
        endpoint_path,
        mock_chat_service,
        sample_chat_id
    ):
        """Test exitoso de obtención"""
        chat_data = create_chat_dict(chat_id=sample_chat_id)
        mock_chat_service.get_chat.return_value = Mock(**chat_data)
        
        with patch('api.routes.get_chat_service', return_value=mock_chat_service), \
             patch('api.routes.chat_to_response') as mock_to_response:
            
            mock_to_response.return_value = chat_data
            
            response = test_client.get(endpoint_path.format(chat_id=sample_chat_id))
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert_chat_response_valid(data)
    
    @pytest.mark.asyncio
    @pytest.mark.api
    @pytest.mark.error_handling
    async def test_get_chat_not_found(
        self,
        test_client,
        endpoint_path,
        mock_chat_service
    ):
        """Test cuando el chat no existe"""
        mock_chat_service.get_chat.return_value = None
        
        with patch('api.routes.get_chat_service', return_value=mock_chat_service):
            response = test_client.get(endpoint_path.format(chat_id="nonexistent"))
            
            assert response.status_code == status.HTTP_404_NOT_FOUND


class TestVoteChat:
    """Tests para vote_chat endpoint"""
    
    @pytest.fixture
    def endpoint_path(self):
        return "/community/chats/{chat_id}/vote"
    
    @pytest.mark.asyncio
    @pytest.mark.api
    @pytest.mark.happy_path
    async def test_vote_chat_upvote_success(
        self,
        test_client,
        endpoint_path,
        mock_chat_service,
        sample_chat_id,
        sample_user_id
    ):
        """Test exitoso de upvote"""
        vote_data = {
            "id": "vote-123",
            "chat_id": sample_chat_id,
            "user_id": sample_user_id,
            "vote_type": "upvote"
        }
        mock_chat_service.vote_chat.return_value = Mock(**vote_data)
        
        request_data = create_vote_request(sample_chat_id, "upvote")
        
        with patch('api.routes.get_user_id', return_value=sample_user_id), \
             patch('api.routes.get_chat_service', return_value=mock_chat_service), \
             patch('api.routes.vote_to_response') as mock_to_response:
            
            mock_to_response.return_value = vote_data
            
            response = test_client.post(
                endpoint_path.format(chat_id=sample_chat_id),
                json=request_data,
                headers={"X-User-ID": sample_user_id}
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert_vote_response_valid(data)


class TestRemixChat:
    """Tests para remix_chat endpoint"""
    
    @pytest.fixture
    def endpoint_path(self):
        return "/community/chats/{chat_id}/remix"
    
    @pytest.mark.asyncio
    @pytest.mark.api
    @pytest.mark.happy_path
    async def test_remix_chat_success(
        self,
        test_client,
        endpoint_path,
        mock_chat_service,
        sample_chat_id,
        sample_user_id
    ):
        """Test exitoso de remix"""
        remix_chat_data = create_chat_dict(title="Remix")
        remix_data = {
            "id": "remix-123",
            "original_chat_id": sample_chat_id,
            "remix_chat_id": remix_chat_data["id"],
            "user_id": sample_user_id
        }
        
        mock_chat_service.remix_chat.return_value = (
            Mock(**remix_chat_data),
            Mock(**remix_data)
        )
        
        request_data = create_remix_request(sample_chat_id)
        
        with patch('api.routes.get_user_id', return_value=sample_user_id), \
             patch('api.routes.get_chat_service', return_value=mock_chat_service), \
             patch('api.routes.remix_to_response') as mock_to_response:
            
            mock_to_response.return_value = remix_data
            
            response = test_client.post(
                endpoint_path.format(chat_id=sample_chat_id),
                json=request_data,
                headers={"X-User-ID": sample_user_id}
            )
            
            assert response.status_code == status.HTTP_201_CREATED
            data = response.json()
            assert_remix_response_valid(data)


class TestSearchChats:
    """Tests para search_chats endpoint"""
    
    @pytest.fixture
    def endpoint_path(self):
        return "/community/search"
    
    @pytest.mark.asyncio
    @pytest.mark.api
    @pytest.mark.happy_path
    async def test_search_chats_success(
        self,
        test_client,
        endpoint_path,
        mock_chat_service
    ):
        """Test exitoso de búsqueda"""
        chats = [create_chat_dict(title="AI Chat")]
        mock_chat_service.search_chats.return_value = (chats, 1)
        
        with patch('api.routes.get_chat_service', return_value=mock_chat_service), \
             patch('api.routes.chats_to_responses') as mock_to_responses:
            
            mock_to_responses.return_value = chats
            
            response = test_client.get(
                endpoint_path,
                params={"query": "AI"}
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert_pagination_valid(data)


class TestGetTopChats:
    """Tests para get_top_chats endpoint"""
    
    @pytest.fixture
    def endpoint_path(self):
        return "/community/top"
    
    @pytest.mark.asyncio
    @pytest.mark.api
    @pytest.mark.happy_path
    async def test_get_top_chats_success(
        self,
        test_client,
        endpoint_path,
        mock_chat_service
    ):
        """Test exitoso de obtener top chats"""
        chats = [
            create_chat_dict(chat_id=f"chat-{i}", score=100.0 - i)
            for i in range(10)
        ]
        mock_chat_service.get_top_chats.return_value = [Mock(**chat) for chat in chats]
        
        with patch('api.routes.get_chat_service', return_value=mock_chat_service), \
             patch('api.routes.chats_to_responses') as mock_to_responses:
            
            mock_to_responses.return_value = chats
            
            response = test_client.get(endpoint_path)
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert_chat_list_valid(data.get("chats", []), min_count=0)

