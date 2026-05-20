"""
Tests for PDF Variantes API Endpoints
=====================================
Comprehensive tests for all API endpoints including upload, variants, topics, etc.
"""

import pytest
from fastapi.testclient import TestClient
from fastapi import UploadFile
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path
import tempfile
import io
from typing import Dict, Any

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from routers.pdf_router import router
from routers.analytics_router import router as analytics_router
from api.endpoints.health import router as health_router
from fastapi import FastAPI


@pytest.fixture
def app():
    """Create FastAPI app for testing."""
    app = FastAPI()
    app.include_router(router)
    app.include_router(analytics_router)
    app.include_router(health_router)
    return app


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_pdf_service():
    """Mock PDF service."""
    service = Mock()
    service.upload_handler = Mock()
    service.variant_generator = Mock()
    service.topic_extractor = Mock()
    
    # Mock upload handler
    mock_metadata = Mock()
    mock_metadata.file_id = "test_file_123"
    service.upload_handler.upload_pdf = AsyncMock(return_value=(mock_metadata, "extracted text"))
    service.upload_handler.get_pdf_preview = AsyncMock(return_value={"preview": "base64_image"})
    service.upload_handler.get_file_path = Mock(return_value=Path("/tmp/test.pdf"))
    service.upload_handler.delete_pdf = AsyncMock(return_value=True)
    
    # Mock variant generator
    service.variant_generator.generate = AsyncMock(return_value={
        "variant_type": "summary",
        "content": "Summary content",
        "generated_at": "2024-01-01T00:00:00"
    })
    
    # Mock topic extractor
    mock_topic = Mock()
    mock_topic.to_dict = Mock(return_value={"topic": "AI", "relevance": 0.9})
    service.topic_extractor.extract_topics = AsyncMock(return_value=[mock_topic])
    
    return service


@pytest.fixture
def mock_current_user():
    """Mock current user."""
    return {"user_id": "test_user", "username": "testuser"}


@pytest.fixture
def sample_pdf_content():
    """Sample PDF content for testing."""
    return b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n3 0 obj\n<<\n/Type /Page\n/Parent 2 0 R\n/MediaBox [0 0 612 792]\n/Contents 4 0 R\n>>\nendobj\n4 0 obj\n<<\n/Length 44\n>>\nstream\nBT\n/F1 12 Tf\n72 720 Td\n(Hello World) Tj\nET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000204 00000 n \ntrailer\n<<\n/Size 5\n/Root 1 0 R\n>>\nstartxref\n297\n%%EOF"


class TestHealthEndpoint:
    """Tests for health check endpoint."""
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["api_ready"] is True


class TestPDFUpload:
    """Tests for PDF upload endpoint."""
    
    @patch('routers.pdf_router.get_pdf_service')
    @patch('routers.pdf_router.get_current_user')
    @patch('routers.pdf_router.validate_pdf_file')
    @patch('routers.pdf_router.extract_metadata')
    @patch('routers.pdf_router.sanitize_filename')
    @patch('routers.pdf_router.validate_file_size')
    def test_upload_pdf_success(
        self, 
        mock_validate_size,
        mock_sanitize,
        mock_extract_meta,
        mock_validate_pdf,
        mock_get_user,
        mock_get_service,
        client,
        mock_pdf_service,
        sample_pdf_content
    ):
        """Test successful PDF upload."""
        # Setup mocks
        mock_get_service.return_value = mock_pdf_service
        mock_get_user.return_value = {"user_id": "test_user"}
        mock_validate_pdf.return_value = {"valid": True}
        mock_extract_meta.return_value = {"pages": 1, "title": "Test PDF"}
        mock_sanitize.return_value = "test.pdf"
        mock_validate_size.return_value = None
        
        # Make request
        files = {"file": ("test.pdf", io.BytesIO(sample_pdf_content), "application/pdf")}
        response = client.post("/pdf/upload?auto_process=true&extract_text=true", files=files)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "file_id" in data
        assert data["processing_started"] is True
    
    def test_upload_invalid_file_type(self, client):
        """Test upload with invalid file type."""
        files = {"file": ("test.txt", io.BytesIO(b"not a pdf"), "text/plain")}
        response = client.post("/pdf/upload", files=files)
        assert response.status_code == 400
    
    @patch('routers.pdf_router.validate_pdf_file')
    def test_upload_invalid_pdf(self, mock_validate_pdf, client, sample_pdf_content):
        """Test upload with invalid PDF."""
        mock_validate_pdf.return_value = {"valid": False, "error": "Invalid PDF format"}
        files = {"file": ("test.pdf", io.BytesIO(sample_pdf_content), "application/pdf")}
        response = client.post("/pdf/upload", files=files)
        assert response.status_code == 400


class TestPDFPreview:
    """Tests for PDF preview endpoint."""
    
    @patch('routers.pdf_router.get_pdf_service')
    def test_get_preview_success(self, mock_get_service, client, mock_pdf_service):
        """Test successful preview retrieval."""
        mock_get_service.return_value = mock_pdf_service
        
        response = client.get("/pdf/test_file_123/preview?page_number=1")
        assert response.status_code == 200
        data = response.json()
        assert data["file_id"] == "test_file_123"
        assert data["page_number"] == 1
        assert "preview" in data
    
    @patch('routers.pdf_router.get_pdf_service')
    def test_get_preview_not_found(self, mock_get_service, client, mock_pdf_service):
        """Test preview for non-existent file."""
        mock_get_service.return_value = mock_pdf_service
        mock_pdf_service.upload_handler.get_pdf_preview = AsyncMock(return_value=None)
        
        response = client.get("/pdf/nonexistent/preview")
        assert response.status_code == 404


class TestVariantGeneration:
    """Tests for variant generation endpoint."""
    
    @patch('routers.pdf_router.get_pdf_service')
    def test_generate_variant_success(self, mock_get_service, client, mock_pdf_service):
        """Test successful variant generation."""
        mock_get_service.return_value = mock_pdf_service
        
        # Mock file exists
        mock_path = Mock()
        mock_path.exists.return_value = True
        mock_pdf_service.upload_handler.get_file_path.return_value = mock_path
        
        variant_data = {
            "variant_type": "summary",
            "options": {
                "max_length": 500,
                "style": "academic"
            }
        }
        
        response = client.post("/pdf/test_file_123/variants", json=variant_data)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "variant" in data
    
    @patch('routers.pdf_router.get_pdf_service')
    def test_generate_variant_file_not_found(self, mock_get_service, client, mock_pdf_service):
        """Test variant generation for non-existent file."""
        mock_get_service.return_value = mock_pdf_service
        
        # Mock file doesn't exist
        mock_path = Mock()
        mock_path.exists.return_value = False
        mock_pdf_service.upload_handler.get_file_path.return_value = mock_path
        
        variant_data = {
            "variant_type": "summary",
            "options": {}
        }
        
        response = client.post("/pdf/nonexistent/variants", json=variant_data)
        assert response.status_code == 404


class TestTopicExtraction:
    """Tests for topic extraction endpoint."""
    
    @patch('routers.pdf_router.get_pdf_service')
    def test_extract_topics_success(self, mock_get_service, client, mock_pdf_service):
        """Test successful topic extraction."""
        mock_get_service.return_value = mock_pdf_service
        
        response = client.get("/pdf/test_file_123/topics?min_relevance=0.5&max_topics=50")
        assert response.status_code == 200
        data = response.json()
        assert "topics" in data
        assert "total_count" in data
        assert isinstance(data["topics"], list)
    
    @patch('routers.pdf_router.get_pdf_service')
    def test_extract_topics_with_params(self, mock_get_service, client, mock_pdf_service):
        """Test topic extraction with custom parameters."""
        mock_get_service.return_value = mock_pdf_service
        
        response = client.get("/pdf/test_file_123/topics?min_relevance=0.8&max_topics=10")
        assert response.status_code == 200
        mock_pdf_service.topic_extractor.extract_topics.assert_called_once()
        call_args = mock_pdf_service.topic_extractor.extract_topics.call_args
        assert call_args[1]["min_relevance"] == 0.8
        assert call_args[1]["max_topics"] == 10


class TestPDFDeletion:
    """Tests for PDF deletion endpoint."""
    
    @patch('routers.pdf_router.get_pdf_service')
    @patch('routers.pdf_router.get_current_user')
    def test_delete_pdf_success(
        self, 
        mock_get_user,
        mock_get_service,
        client,
        mock_pdf_service
    ):
        """Test successful PDF deletion."""
        mock_get_service.return_value = mock_pdf_service
        mock_get_user.return_value = {"user_id": "test_user"}
        
        response = client.delete("/pdf/test_file_123")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
    
    @patch('routers.pdf_router.get_pdf_service')
    @patch('routers.pdf_router.get_current_user')
    def test_delete_pdf_not_found(
        self,
        mock_get_user,
        mock_get_service,
        client,
        mock_pdf_service
    ):
        """Test deletion of non-existent PDF."""
        mock_get_service.return_value = mock_pdf_service
        mock_get_user.return_value = {"user_id": "test_user"}
        mock_pdf_service.upload_handler.delete_pdf = AsyncMock(return_value=False)
        
        response = client.delete("/pdf/nonexistent")
        assert response.status_code == 404


class TestValidation:
    """Tests for input validation."""
    
    def test_preview_invalid_page_number(self, client):
        """Test preview with invalid page number."""
        response = client.get("/pdf/test_file/preview?page_number=0")
        assert response.status_code == 422  # Validation error
    
    def test_topics_invalid_min_relevance(self, client):
        """Test topics with invalid min_relevance."""
        response = client.get("/pdf/test_file/topics?min_relevance=1.5")
        assert response.status_code == 422
    
    def test_topics_invalid_max_topics(self, client):
        """Test topics with invalid max_topics."""
        response = client.get("/pdf/test_file/topics?max_topics=300")
        assert response.status_code == 422

