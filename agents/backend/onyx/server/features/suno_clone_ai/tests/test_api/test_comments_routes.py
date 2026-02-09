"""
Tests para las rutas de comentarios
"""

import pytest
from unittest.mock import Mock, patch
from fastapi import status
from fastapi.testclient import TestClient
from datetime import datetime

from api.routes.comments import router
from services.song_service import SongService


@pytest.fixture
def mock_song_service():
    """Mock del servicio de canciones"""
    service = Mock(spec=SongService)
    service.get_song = Mock(return_value={
        "song_id": "song-123",
        "user_id": "user-456",
        "prompt": "Test song",
        "status": "completed",
        "metadata": {
            "comments": []
        }
    })
    service.save_song = Mock(return_value=True)
    return service


@pytest.fixture
def client(mock_song_service):
    """Cliente de prueba con mocks"""
    from fastapi import FastAPI
    from api.dependencies import SongServiceDep
    
    app = FastAPI()
    app.include_router(router)
    
    def get_song_service():
        return mock_song_service
    
    app.dependency_overrides[SongServiceDep] = get_song_service
    
    yield TestClient(app)
    
    app.dependency_overrides.clear()


@pytest.mark.unit
@pytest.mark.api
class TestAddComment:
    """Tests para agregar comentarios"""
    
    def test_add_comment_success(self, client, mock_song_service):
        """Test de agregar comentario exitosamente"""
        response = client.post(
            "/songs/song-123/comments",
            params={
                "comment": "Great song!",
                "user_id": "user-789"
            }
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["message"] == "Comment added successfully"
        assert data["song_id"] == "song-123"
        assert "comment_id" in data
    
    def test_add_comment_with_parent(self, client, mock_song_service):
        """Test de agregar comentario como respuesta"""
        # Primero agregar comentario padre
        parent_response = client.post(
            "/songs/song-123/comments",
            params={
                "comment": "Original comment",
                "user_id": "user-789"
            }
        )
        assert parent_response.status_code == status.HTTP_200_OK
        parent_comment_id = parent_response.json()["comment_id"]
        
        # Agregar respuesta
        reply_response = client.post(
            "/songs/song-123/comments",
            params={
                "comment": "Reply to comment",
                "user_id": "user-999",
                "parent_comment_id": parent_comment_id
            }
        )
        
        assert reply_response.status_code == status.HTTP_200_OK
        data = reply_response.json()
        assert data.get("parent_comment_id") == parent_comment_id
    
    def test_add_comment_validation(self, client):
        """Test de validación de comentarios"""
        # Comentario vacío
        response = client.post(
            "/songs/song-123/comments",
            params={"comment": "", "user_id": "user-789"}
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        
        # Comentario muy largo
        long_comment = "a" * 501
        response = client.post(
            "/songs/song-123/comments",
            params={"comment": long_comment, "user_id": "user-789"}
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_add_comment_song_not_found(self, client, mock_song_service):
        """Test cuando la canción no existe"""
        mock_song_service.get_song.return_value = None
        
        response = client.post(
            "/songs/nonexistent/comments",
            params={"comment": "Test", "user_id": "user-789"}
        )
        
        assert response.status_code == status.HTTP_404_NOT_FOUND


@pytest.mark.unit
@pytest.mark.api
class TestGetComments:
    """Tests para obtener comentarios"""
    
    def test_get_comments_success(self, client, mock_song_service):
        """Test de obtención exitosa de comentarios"""
        response = client.get("/songs/song-123/comments")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "comments" in data
        assert "total" in data
        assert isinstance(data["comments"], list)
    
    def test_get_comments_pagination(self, client, mock_song_service):
        """Test con paginación"""
        response = client.get("/songs/song-123/comments?limit=10&offset=0")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "limit" in data
        assert "offset" in data
        assert data["limit"] == 10
        assert data["offset"] == 0
    
    def test_get_comments_sorting(self, client, mock_song_service):
        """Test con ordenamiento"""
        response = client.get("/songs/song-123/comments?sort=date_desc")
        
        assert response.status_code == status.HTTP_200_OK


@pytest.mark.unit
@pytest.mark.api
class TestDeleteComment:
    """Tests para eliminar comentarios"""
    
    def test_delete_comment_success(self, client, mock_song_service):
        """Test de eliminación exitosa"""
        # Primero agregar comentario
        add_response = client.post(
            "/songs/song-123/comments",
            params={"comment": "To be deleted", "user_id": "user-789"}
        )
        assert add_response.status_code == status.HTTP_200_OK
        comment_id = add_response.json()["comment_id"]
        
        # Eliminar comentario
        delete_response = client.delete(
            f"/songs/song-123/comments/{comment_id}",
            params={"user_id": "user-789"}
        )
        
        assert delete_response.status_code == status.HTTP_200_OK
        assert delete_response.json()["message"] == "Comment deleted successfully"
    
    def test_delete_comment_not_found(self, client):
        """Test cuando el comentario no existe"""
        response = client.delete(
            "/songs/song-123/comments/nonexistent",
            params={"user_id": "user-789"}
        )
        
        assert response.status_code == status.HTTP_404_NOT_FOUND


@pytest.mark.integration
@pytest.mark.api
class TestCommentsIntegration:
    """Tests de integración para comentarios"""
    
    def test_full_comments_workflow(self, client, mock_song_service):
        """Test del flujo completo de comentarios"""
        # 1. Agregar comentario
        add_response = client.post(
            "/songs/song-123/comments",
            params={"comment": "First comment", "user_id": "user-789"}
        )
        assert add_response.status_code == status.HTTP_200_OK
        comment_id = add_response.json()["comment_id"]
        
        # 2. Obtener comentarios
        get_response = client.get("/songs/song-123/comments")
        assert get_response.status_code == status.HTTP_200_OK
        
        # 3. Agregar respuesta
        reply_response = client.post(
            "/songs/song-123/comments",
            params={
                "comment": "Reply",
                "user_id": "user-999",
                "parent_comment_id": comment_id
            }
        )
        assert reply_response.status_code == status.HTTP_200_OK
        
        # 4. Eliminar comentario
        delete_response = client.delete(
            f"/songs/song-123/comments/{comment_id}",
            params={"user_id": "user-789"}
        )
        assert delete_response.status_code == status.HTTP_200_OK



