"""
Tests para las rutas de backup
"""

import pytest
from unittest.mock import Mock, patch
from fastapi import status
from fastapi.testclient import TestClient

from api.routes.backup import router
from utils.backup_recovery import BackupManager


@pytest.fixture
def mock_backup_manager():
    """Mock del gestor de backups"""
    manager = Mock(spec=BackupManager)
    manager.create_backup = Mock(return_value="backup-123")
    manager.list_backups = Mock(return_value=[
        {"id": "backup-1", "type": "full", "created_at": "2024-01-01T00:00:00"},
        {"id": "backup-2", "type": "incremental", "created_at": "2024-01-02T00:00:00"}
    ])
    manager.restore_backup = Mock(return_value=True)
    manager.verify_backup = Mock(return_value={"valid": True, "size": 1024})
    manager.delete_backup = Mock(return_value=True)
    return manager


@pytest.fixture
def client(mock_backup_manager):
    """Cliente de prueba con mocks"""
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)
    
    with patch('api.routes.backup.get_backup_manager', return_value=mock_backup_manager):
        with patch('api.routes.backup.require_role', return_value=lambda: None):
            yield TestClient(app)


@pytest.mark.unit
@pytest.mark.api
class TestCreateBackup:
    """Tests para crear backup"""
    
    def test_create_backup_success(self, client, mock_backup_manager):
        """Test de creación exitosa de backup"""
        response = client.post(
            "/backup/create",
            json={"backup_type": "full", "description": "Test backup"}
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "backup_id" in data
        assert data["message"] == "Backup created successfully"
        assert data["type"] == "full"
    
    def test_create_backup_incremental(self, client, mock_backup_manager):
        """Test de backup incremental"""
        response = client.post(
            "/backup/create",
            json={"backup_type": "incremental"}
        )
        
        assert response.status_code == status.HTTP_200_OK
        assert response.json()["type"] == "incremental"


@pytest.mark.unit
@pytest.mark.api
class TestListBackups:
    """Tests para listar backups"""
    
    def test_list_backups_success(self, client, mock_backup_manager):
        """Test de listado exitoso"""
        response = client.get("/backup/list")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "backups" in data or isinstance(data, list)


@pytest.mark.unit
@pytest.mark.api
class TestRestoreBackup:
    """Tests para restaurar backup"""
    
    def test_restore_backup_success(self, client, mock_backup_manager):
        """Test de restauración exitosa"""
        response = client.post("/backup/backup-123/restore")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "message" in data


@pytest.mark.unit
@pytest.mark.api
class TestVerifyBackup:
    """Tests para verificar backup"""
    
    def test_verify_backup_success(self, client, mock_backup_manager):
        """Test de verificación exitosa"""
        response = client.get("/backup/backup-123/verify")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "valid" in data or "status" in data


@pytest.mark.integration
@pytest.mark.api
class TestBackupIntegration:
    """Tests de integración para backup"""
    
    def test_full_backup_workflow(self, client, mock_backup_manager):
        """Test del flujo completo de backup"""
        # 1. Crear backup
        create_response = client.post(
            "/backup/create",
            json={"backup_type": "full"}
        )
        assert create_response.status_code == status.HTTP_200_OK
        backup_id = create_response.json()["backup_id"]
        
        # 2. Listar backups
        list_response = client.get("/backup/list")
        assert list_response.status_code == status.HTTP_200_OK
        
        # 3. Verificar backup
        verify_response = client.get(f"/backup/{backup_id}/verify")
        assert verify_response.status_code == status.HTTP_200_OK
        
        # 4. Restaurar backup
        restore_response = client.post(f"/backup/{backup_id}/restore")
        assert restore_response.status_code == status.HTTP_200_OK



