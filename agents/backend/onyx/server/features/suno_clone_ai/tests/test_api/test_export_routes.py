"""
Tests para las rutas de exportación
"""

import pytest
from unittest.mock import Mock, patch
from fastapi import status
from fastapi.testclient import TestClient
import json
import csv
import io

from api.routes.export import router
from services.song_service import SongService


@pytest.fixture
def mock_song_service():
    """Mock del servicio de canciones"""
    service = Mock(spec=SongService)
    service.get_song = Mock(return_value={
        "song_id": "song-123",
        "user_id": "user-456",
        "prompt": "A happy pop song",
        "genre": "pop",
        "mood": "happy",
        "duration": 30,
        "status": "completed",
        "file_path": "/tmp/song.wav",
        "metadata": {
            "model_used": "facebook/musicgen-medium",
            "generation_time": 5.2,
            "tags": ["happy", "pop", "upbeat"]
        }
    })
    service.list_songs = Mock(return_value=[
        {
            "song_id": "song-1",
            "user_id": "user-456",
            "prompt": "Song 1",
            "status": "completed"
        },
        {
            "song_id": "song-2",
            "user_id": "user-456",
            "prompt": "Song 2",
            "status": "completed"
        }
    ])
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
class TestExportSongMetadata:
    """Tests para exportar metadatos de canción"""
    
    def test_export_json_success(self, client, mock_song_service):
        """Test de exportación JSON exitosa"""
        response = client.get("/songs/song-123/export?format=json")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "song_id" in data
        assert data["song_id"] == "song-123"
        assert "metadata" in data
    
    def test_export_json_download(self, client, mock_song_service):
        """Test de exportación JSON como descarga"""
        response = client.get("/songs/song-123/export?format=json&download=true")
        
        assert response.status_code == status.HTTP_200_OK
        assert "attachment" in response.headers.get("content-disposition", "").lower()
        assert "json" in response.headers.get("content-type", "").lower()
    
    def test_export_xml_success(self, client, mock_song_service):
        """Test de exportación XML exitosa"""
        response = client.get("/songs/song-123/export?format=xml")
        
        assert response.status_code == status.HTTP_200_OK
        # XML puede ser texto o JSON con estructura XML
        assert response.headers.get("content-type") in [
            "application/xml",
            "text/xml",
            "application/json"
        ]
    
    def test_export_xml_download(self, client, mock_song_service):
        """Test de exportación XML como descarga"""
        response = client.get("/songs/song-123/export?format=xml&download=true")
        
        assert response.status_code == status.HTTP_200_OK
        assert "attachment" in response.headers.get("content-disposition", "").lower()
    
    def test_export_csv_success(self, client, mock_song_service):
        """Test de exportación CSV exitosa"""
        response = client.get("/songs/song-123/export?format=csv")
        
        assert response.status_code == status.HTTP_200_OK
        # CSV puede ser texto o JSON
        content_type = response.headers.get("content-type", "")
        assert "csv" in content_type.lower() or "text" in content_type.lower()
    
    def test_export_csv_download(self, client, mock_song_service):
        """Test de exportación CSV como descarga"""
        response = client.get("/songs/song-123/export?format=csv&download=true")
        
        assert response.status_code == status.HTTP_200_OK
        assert "attachment" in response.headers.get("content-disposition", "").lower()
        assert "csv" in response.headers.get("content-disposition", "").lower()
    
    def test_export_invalid_format(self, client):
        """Test con formato inválido"""
        response = client.get("/songs/song-123/export?format=invalid")
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "format" in response.json()["detail"].lower()
    
    def test_export_song_not_found(self, client, mock_song_service):
        """Test cuando la canción no existe"""
        mock_song_service.get_song.return_value = None
        
        response = client.get("/songs/nonexistent/export")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
    
    def test_export_invalid_song_id(self, client):
        """Test con song_id inválido"""
        response = client.get("/songs/invalid-id/export")
        
        assert response.status_code in [
            status.HTTP_400_BAD_REQUEST,
            status.HTTP_404_NOT_FOUND,
            status.HTTP_422_UNPROCESSABLE_ENTITY
        ]


@pytest.mark.unit
@pytest.mark.api
class TestExportBatch:
    """Tests para exportación en lote"""
    
    def test_export_batch_json_success(self, client, mock_song_service):
        """Test de exportación en lote JSON exitosa"""
        response = client.get("/songs/export/batch?format=json&user_id=user-456")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "songs" in data or isinstance(data, list)
    
    def test_export_batch_csv_success(self, client, mock_song_service):
        """Test de exportación en lote CSV exitosa"""
        response = client.get("/songs/export/batch?format=csv&user_id=user-456")
        
        assert response.status_code == status.HTTP_200_OK
        content_type = response.headers.get("content-type", "")
        assert "csv" in content_type.lower() or "text" in content_type.lower()
    
    def test_export_batch_xml_success(self, client, mock_song_service):
        """Test de exportación en lote XML exitosa"""
        response = client.get("/songs/export/batch?format=xml&user_id=user-456")
        
        assert response.status_code == status.HTTP_200_OK
    
    def test_export_batch_with_filters(self, client, mock_song_service):
        """Test de exportación en lote con filtros"""
        response = client.get(
            "/songs/export/batch?format=json&user_id=user-456&status=completed"
        )
        
        assert response.status_code == status.HTTP_200_OK
    
    def test_export_batch_invalid_format(self, client):
        """Test con formato inválido"""
        response = client.get("/songs/export/batch?format=invalid")
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
    
    def test_export_batch_empty_results(self, client, mock_song_service):
        """Test cuando no hay resultados"""
        mock_song_service.list_songs.return_value = []
        
        response = client.get("/songs/export/batch?format=json&user_id=user-456")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        # Puede ser lista vacía o dict con lista vacía
        if isinstance(data, dict):
            assert len(data.get("songs", [])) == 0
        else:
            assert len(data) == 0


@pytest.mark.integration
@pytest.mark.api
class TestExportIntegration:
    """Tests de integración para exportación"""
    
    def test_full_export_workflow(self, client, mock_song_service):
        """Test del flujo completo de exportación"""
        formats = ["json", "xml", "csv"]
        
        for fmt in formats:
            # Exportar individual
            individual_response = client.get(f"/songs/song-123/export?format={fmt}")
            assert individual_response.status_code == status.HTTP_200_OK
            
            # Exportar en lote
            batch_response = client.get(f"/songs/export/batch?format={fmt}&user_id=user-456")
            assert batch_response.status_code == status.HTTP_200_OK
    
    def test_export_with_download_workflow(self, client, mock_song_service):
        """Test de exportación con descarga"""
        formats = ["json", "xml", "csv"]
        
        for fmt in formats:
            response = client.get(f"/songs/song-123/export?format={fmt}&download=true")
            assert response.status_code == status.HTTP_200_OK
            assert "attachment" in response.headers.get("content-disposition", "").lower()



