"""
Tests para las rutas de colaboración
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from fastapi import status
from fastapi.testclient import TestClient

from api.routes.collaboration import router
from services.collaboration import CollaborationService


@pytest.fixture
def mock_collaboration_service():
    """Mock del servicio de colaboración"""
    service = Mock(spec=CollaborationService)
    
    # Mock de sesión
    session = Mock()
    session.session_id = "session-123"
    session.project_id = "project-456"
    session.owner_id = "user-789"
    session.participants = {"user-789"}
    session.created_at = Mock()
    session.created_at.isoformat = Mock(return_value="2024-01-01T00:00:00")
    
    service.create_session = Mock(return_value=session)
    service.join_session = Mock(return_value=True)
    service.leave_session = Mock(return_value=True)
    service.send_event = Mock(return_value=True)
    service.get_history = Mock(return_value=[])
    
    return service


@pytest.fixture
def client(mock_collaboration_service):
    """Cliente de prueba con mocks"""
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)
    
    with patch('api.routes.collaboration.get_collaboration_service', return_value=mock_collaboration_service):
        with patch('api.routes.collaboration.get_current_user', return_value={"user_id": "test_user"}):
            yield TestClient(app)


@pytest.mark.unit
@pytest.mark.api
class TestCreateSession:
    """Tests para crear sesión de colaboración"""
    
    def test_create_session_success(self, client, mock_collaboration_service):
        """Test de creación exitosa de sesión"""
        response = client.post(
            "/collaboration/sessions",
            json={"project_id": "project-456"}
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "session_id" in data
        assert data["project_id"] == "project-456"
        assert "participants" in data
        assert "created_at" in data
    
    def test_create_session_error_handling(self, client, mock_collaboration_service):
        """Test de manejo de errores"""
        mock_collaboration_service.create_session.side_effect = Exception("Service error")
        
        response = client.post(
            "/collaboration/sessions",
            json={"project_id": "project-456"}
        )
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Error creating session" in response.json()["detail"]


@pytest.mark.unit
@pytest.mark.api
class TestJoinSession:
    """Tests para unirse a sesión"""
    
    def test_join_session_success(self, client, mock_collaboration_service):
        """Test de unirse exitosamente a sesión"""
        response = client.post(
            "/collaboration/sessions/session-123/join",
            json={"permission": "editor"}
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["message"] == "Joined session"
        assert data["session_id"] == "session-123"
    
    def test_join_session_with_viewer_permission(self, client, mock_collaboration_service):
        """Test con permiso de viewer"""
        response = client.post(
            "/collaboration/sessions/session-123/join",
            json={"permission": "viewer"}
        )
        
        assert response.status_code == status.HTTP_200_OK
    
    def test_join_session_not_found(self, client, mock_collaboration_service):
        """Test cuando la sesión no existe"""
        mock_collaboration_service.join_session.return_value = False
        
        response = client.post(
            "/collaboration/sessions/nonexistent/join",
            json={"permission": "editor"}
        )
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "not found" in response.json()["detail"].lower()


@pytest.mark.unit
@pytest.mark.api
class TestLeaveSession:
    """Tests para salir de sesión"""
    
    def test_leave_session_success(self, client, mock_collaboration_service):
        """Test de salir exitosamente de sesión"""
        response = client.post("/collaboration/sessions/session-123/leave")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["message"] == "Left session"


@pytest.mark.unit
@pytest.mark.api
class TestSendEvent:
    """Tests para enviar eventos"""
    
    def test_send_event_success(self, client, mock_collaboration_service):
        """Test de envío exitoso de evento"""
        response = client.post(
            "/collaboration/sessions/session-123/events",
            json={
                "event_type": "edit",
                "data": {"field": "value"}
            }
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["message"] == "Event sent"


@pytest.mark.unit
@pytest.mark.api
class TestGetHistory:
    """Tests para obtener historial"""
    
    def test_get_history_success(self, client, mock_collaboration_service):
        """Test de obtención exitosa de historial"""
        mock_collaboration_service.get_history.return_value = [
            {"event": "edit", "timestamp": "2024-01-01T00:00:00"},
            {"event": "comment", "timestamp": "2024-01-01T00:01:00"}
        ]
        
        response = client.get("/collaboration/sessions/session-123/history")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "history" in data
        assert len(data["history"]) == 2


@pytest.mark.integration
@pytest.mark.api
class TestCollaborationIntegration:
    """Tests de integración para colaboración"""
    
    def test_full_collaboration_workflow(self, client, mock_collaboration_service):
        """Test del flujo completo de colaboración"""
        # 1. Crear sesión
        create_response = client.post(
            "/collaboration/sessions",
            json={"project_id": "project-456"}
        )
        assert create_response.status_code == status.HTTP_200_OK
        session_id = create_response.json()["session_id"]
        
        # 2. Unirse a sesión
        join_response = client.post(
            f"/collaboration/sessions/{session_id}/join",
            json={"permission": "editor"}
        )
        assert join_response.status_code == status.HTTP_200_OK
        
        # 3. Enviar evento
        event_response = client.post(
            f"/collaboration/sessions/{session_id}/events",
            json={"event_type": "edit", "data": {}}
        )
        assert event_response.status_code == status.HTTP_200_OK
        
        # 4. Obtener historial
        history_response = client.get(f"/collaboration/sessions/{session_id}/history")
        assert history_response.status_code == status.HTTP_200_OK
        
        # 5. Salir de sesión
        leave_response = client.post(f"/collaboration/sessions/{session_id}/leave")
        assert leave_response.status_code == status.HTTP_200_OK



