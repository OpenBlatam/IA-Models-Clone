"""
Tests para las rutas de administración
"""

import pytest
from unittest.mock import Mock, patch
from fastapi import status
from fastapi.testclient import TestClient

from api.routes.admin import router
from services.task_queue import TaskQueue, TaskStatus


@pytest.fixture
def mock_task_queue():
    """Mock de la cola de tareas"""
    queue = Mock(spec=TaskQueue)
    queue.get_queue_stats = Mock(return_value={"pending": 5, "completed": 100})
    queue.get_tasks_by_status = Mock(return_value=[])
    queue.tasks = {}
    queue.cancel_task = Mock(return_value=True)
    return queue


@pytest.fixture
def mock_notification_service():
    """Mock del servicio de notificaciones"""
    service = Mock()
    service.get_stats = Mock(return_value={"total": 50})
    return service


@pytest.fixture
def mock_cache():
    """Mock del caché distribuido"""
    cache = Mock()
    cache.get_stats = Mock(return_value={"hits": 100, "misses": 20})
    return cache


@pytest.fixture
def mock_alert_manager():
    """Mock del gestor de alertas"""
    manager = Mock()
    manager.get_stats = Mock(return_value={"active": 2})
    return manager


@pytest.fixture
def client(mock_task_queue, mock_notification_service, mock_cache, mock_alert_manager):
    """Cliente de prueba con mocks"""
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)
    
    with patch('api.routes.admin.get_task_queue', return_value=mock_task_queue):
        with patch('api.routes.admin.get_notification_service', return_value=mock_notification_service):
            with patch('api.routes.admin.get_distributed_cache', return_value=mock_cache):
                with patch('api.routes.admin.get_alert_manager', return_value=mock_alert_manager):
                    with patch('api.routes.admin.require_role', return_value=lambda: None):
                        yield TestClient(app)


@pytest.mark.unit
@pytest.mark.api
class TestAdminStats:
    """Tests para estadísticas de administración"""
    
    def test_get_admin_stats_success(self, client, mock_task_queue, mock_notification_service, mock_cache, mock_alert_manager):
        """Test de obtención exitosa de estadísticas"""
        response = client.get("/admin/stats")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "timestamp" in data
        assert "task_queue" in data
        assert "notifications" in data
        assert "cache" in data
        assert "alerts" in data


@pytest.mark.unit
@pytest.mark.api
class TestListTasks:
    """Tests para listar tareas"""
    
    def test_list_tasks_success(self, client, mock_task_queue):
        """Test de listado exitoso"""
        response = client.get("/admin/tasks")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "tasks" in data or isinstance(data, list)
    
    def test_list_tasks_with_status_filter(self, client, mock_task_queue):
        """Test con filtro de estado"""
        response = client.get("/admin/tasks?status_filter=pending")
        
        assert response.status_code == status.HTTP_200_OK
    
    def test_list_tasks_invalid_status(self, client):
        """Test con estado inválido"""
        response = client.get("/admin/tasks?status_filter=invalid")
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST


@pytest.mark.unit
@pytest.mark.api
class TestCancelTask:
    """Tests para cancelar tareas"""
    
    def test_cancel_task_success(self, client, mock_task_queue):
        """Test de cancelación exitosa"""
        response = client.post("/admin/tasks/task-123/cancel")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "message" in data


@pytest.mark.integration
@pytest.mark.api
class TestAdminIntegration:
    """Tests de integración para administración"""
    
    def test_full_admin_workflow(self, client, mock_task_queue, mock_notification_service, mock_cache, mock_alert_manager):
        """Test del flujo completo de administración"""
        # 1. Obtener estadísticas
        stats_response = client.get("/admin/stats")
        assert stats_response.status_code == status.HTTP_200_OK
        
        # 2. Listar tareas
        tasks_response = client.get("/admin/tasks")
        assert tasks_response.status_code == status.HTTP_200_OK
        
        # 3. Cancelar tarea
        cancel_response = client.post("/admin/tasks/task-123/cancel")
        assert cancel_response.status_code == status.HTTP_200_OK



