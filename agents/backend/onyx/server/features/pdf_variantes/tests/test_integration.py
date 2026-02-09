"""
Integration Tests for PDF Variantes
====================================
End-to-end tests for complete workflows.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path
import tempfile
import io
from typing import Dict, Any

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI
from routers.pdf_router import router
from api.endpoints.health import router as health_router


@pytest.fixture
def app():
    """Create FastAPI app for integration testing."""
    app = FastAPI()
    app.include_router(router)
    app.include_router(health_router)
    return app


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def sample_pdf_content():
    """Sample PDF content for testing."""
    return b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n3 0 obj\n<<\n/Type /Page\n/Parent 2 0 R\n/MediaBox [0 0 612 792]\n/Contents 4 0 R\n>>\nendobj\n4 0 obj\n<<\n/Length 44\n>>\nstream\nBT\n/F1 12 Tf\n72 720 Td\n(Hello World) Tj\nET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000204 00000 n \ntrailer\n<<\n/Size 5\n/Root 1 0 R\n>>\nstartxref\n297\n%%EOF"


class TestCompleteWorkflow:
    """Tests for complete user workflows."""
    
    @patch('routers.pdf_router.get_pdf_service')
    @patch('routers.pdf_router.get_current_user')
    @patch('routers.pdf_router.validate_pdf_file')
    @patch('routers.pdf_router.extract_metadata')
    @patch('routers.pdf_router.sanitize_filename')
    @patch('routers.pdf_router.validate_file_size')
    def test_upload_and_generate_variant_workflow(
        self,
        mock_validate_size,
        mock_sanitize,
        mock_extract_meta,
        mock_validate_pdf,
        mock_get_user,
        mock_get_service,
        client,
        sample_pdf_content
    ):
        """Test complete workflow: upload PDF and generate variant."""
        # Setup mocks
        mock_service = Mock()
        mock_service.upload_handler = Mock()
        mock_service.variant_generator = Mock()
        
        mock_metadata = Mock()
        mock_metadata.file_id = "test_file_123"
        mock_service.upload_handler.upload_pdf = AsyncMock(
            return_value=(mock_metadata, "extracted text")
        )
        mock_service.upload_handler.get_file_path = Mock(
            return_value=Path("/tmp/test.pdf")
        )
        
        mock_path = Mock()
        mock_path.exists.return_value = True
        mock_service.upload_handler.get_file_path.return_value = mock_path
        
        mock_service.variant_generator.generate = AsyncMock(return_value={
            "variant_type": "summary",
            "content": "Summary content",
            "generated_at": "2024-01-01T00:00:00"
        })
        
        mock_get_service.return_value = mock_service
        mock_get_user.return_value = {"user_id": "test_user"}
        mock_validate_pdf.return_value = {"valid": True}
        mock_extract_meta.return_value = {"pages": 1, "title": "Test PDF"}
        mock_sanitize.return_value = "test.pdf"
        mock_validate_size.return_value = None
        
        # Step 1: Upload PDF
        files = {"file": ("test.pdf", io.BytesIO(sample_pdf_content), "application/pdf")}
        upload_response = client.post(
            "/pdf/upload?auto_process=true&extract_text=true",
            files=files
        )
        
        assert upload_response.status_code == 200
        upload_data = upload_response.json()
        file_id = upload_data["file_id"]
        assert file_id == "test_file_123"
        
        # Step 2: Generate variant
        variant_data = {
            "variant_type": "summary",
            "options": {
                "max_length": 500,
                "style": "academic"
            }
        }
        variant_response = client.post(f"/pdf/{file_id}/variants", json=variant_data)
        
        assert variant_response.status_code == 200
        variant_data_response = variant_response.json()
        assert variant_data_response["success"] is True
        assert "variant" in variant_data_response
    
    @patch('routers.pdf_router.get_pdf_service')
    @patch('routers.pdf_router.get_current_user')
    @patch('routers.pdf_router.validate_pdf_file')
    @patch('routers.pdf_router.extract_metadata')
    @patch('routers.pdf_router.sanitize_filename')
    @patch('routers.pdf_router.validate_file_size')
    def test_upload_preview_and_extract_topics_workflow(
        self,
        mock_validate_size,
        mock_sanitize,
        mock_extract_meta,
        mock_validate_pdf,
        mock_get_user,
        mock_get_service,
        client,
        sample_pdf_content
    ):
        """Test workflow: upload, preview, and extract topics."""
        # Setup mocks
        mock_service = Mock()
        mock_service.upload_handler = Mock()
        mock_service.topic_extractor = Mock()
        
        mock_metadata = Mock()
        mock_metadata.file_id = "test_file_456"
        mock_service.upload_handler.upload_pdf = AsyncMock(
            return_value=(mock_metadata, "extracted text")
        )
        mock_service.upload_handler.get_pdf_preview = AsyncMock(
            return_value={"preview": "base64_image_data"}
        )
        
        mock_topic = Mock()
        mock_topic.to_dict = Mock(return_value={"topic": "Machine Learning", "relevance": 0.95})
        mock_service.topic_extractor.extract_topics = AsyncMock(
            return_value=[mock_topic]
        )
        
        mock_get_service.return_value = mock_service
        mock_get_user.return_value = {"user_id": "test_user"}
        mock_validate_pdf.return_value = {"valid": True}
        mock_extract_meta.return_value = {"pages": 1, "title": "Test PDF"}
        mock_sanitize.return_value = "test.pdf"
        mock_validate_size.return_value = None
        
        # Step 1: Upload PDF
        files = {"file": ("test.pdf", io.BytesIO(sample_pdf_content), "application/pdf")}
        upload_response = client.post("/pdf/upload", files=files)
        
        assert upload_response.status_code == 200
        file_id = upload_response.json()["file_id"]
        
        # Step 2: Get preview
        preview_response = client.get(f"/pdf/{file_id}/preview?page_number=1")
        assert preview_response.status_code == 200
        preview_data = preview_response.json()
        assert "preview" in preview_data
        
        # Step 3: Extract topics
        topics_response = client.get(f"/pdf/{file_id}/topics?min_relevance=0.5&max_topics=10")
        assert topics_response.status_code == 200
        topics_data = topics_response.json()
        assert "topics" in topics_data
        assert len(topics_data["topics"]) > 0


class TestErrorScenarios:
    """Tests for error scenarios in workflows."""
    
    @patch('routers.pdf_router.get_pdf_service')
    @patch('routers.pdf_router.get_current_user')
    def test_workflow_with_missing_file(self, mock_get_user, mock_get_service, client):
        """Test workflow when file doesn't exist."""
        mock_service = Mock()
        mock_service.upload_handler = Mock()
        mock_service.upload_handler.get_file_path = Mock(return_value=Path("/nonexistent.pdf"))
        
        mock_path = Mock()
        mock_path.exists.return_value = False
        mock_service.upload_handler.get_file_path.return_value = mock_path
        
        mock_get_service.return_value = mock_service
        mock_get_user.return_value = {"user_id": "test_user"}
        
        # Try to generate variant for non-existent file
        variant_data = {"variant_type": "summary", "options": {}}
        response = client.post("/pdf/nonexistent/variants", json=variant_data)
        
        assert response.status_code == 404
    
    @patch('routers.pdf_router.get_pdf_service')
    @patch('routers.pdf_router.get_current_user')
    def test_workflow_with_invalid_operations(self, mock_get_user, mock_get_service, client):
        """Test workflow with invalid operations."""
        mock_service = Mock()
        mock_service.upload_handler = Mock()
        mock_service.upload_handler.get_pdf_preview = AsyncMock(return_value=None)
        
        mock_get_service.return_value = mock_service
        mock_get_user.return_value = {"user_id": "test_user"}
        
        # Try to get preview for non-existent file
        response = client.get("/pdf/nonexistent/preview")
        assert response.status_code == 404


class TestConcurrentOperations:
    """Tests for concurrent operations."""
    
    @patch('routers.pdf_router.get_pdf_service')
    @patch('routers.pdf_router.get_current_user')
    @patch('routers.pdf_router.validate_pdf_file')
    @patch('routers.pdf_router.extract_metadata')
    @patch('routers.pdf_router.sanitize_filename')
    @patch('routers.pdf_router.validate_file_size')
    def test_multiple_variant_generation(
        self,
        mock_validate_size,
        mock_sanitize,
        mock_extract_meta,
        mock_validate_pdf,
        mock_get_user,
        mock_get_service,
        client,
        sample_pdf_content
    ):
        """Test generating multiple variants for the same file."""
        # Setup mocks
        mock_service = Mock()
        mock_service.upload_handler = Mock()
        mock_service.variant_generator = Mock()
        
        mock_metadata = Mock()
        mock_metadata.file_id = "test_file_789"
        mock_service.upload_handler.upload_pdf = AsyncMock(
            return_value=(mock_metadata, "extracted text")
        )
        
        mock_path = Mock()
        mock_path.exists.return_value = True
        mock_service.upload_handler.get_file_path.return_value = mock_path
        
        mock_service.variant_generator.generate = AsyncMock(side_effect=[
            {"variant_type": "summary", "content": "Summary"},
            {"variant_type": "outline", "content": "Outline"},
            {"variant_type": "highlights", "content": "Highlights"}
        ])
        
        mock_get_service.return_value = mock_service
        mock_get_user.return_value = {"user_id": "test_user"}
        mock_validate_pdf.return_value = {"valid": True}
        mock_extract_meta.return_value = {"pages": 1}
        mock_sanitize.return_value = "test.pdf"
        mock_validate_size.return_value = None
        
        # Upload file
        files = {"file": ("test.pdf", io.BytesIO(sample_pdf_content), "application/pdf")}
        upload_response = client.post("/pdf/upload", files=files)
        file_id = upload_response.json()["file_id"]
        
        # Generate multiple variants
        variant_types = ["summary", "outline", "highlights"]
        for variant_type in variant_types:
            variant_data = {"variant_type": variant_type, "options": {}}
            response = client.post(f"/pdf/{file_id}/variants", json=variant_data)
            assert response.status_code == 200
            assert response.json()["variant"]["variant_type"] == variant_type


class TestDataPersistence:
    """Tests for data persistence across operations."""
    
    @patch('routers.pdf_router.get_pdf_service')
    @patch('routers.pdf_router.get_current_user')
    @patch('routers.pdf_router.validate_pdf_file')
    @patch('routers.pdf_router.extract_metadata')
    @patch('routers.pdf_router.sanitize_filename')
    @patch('routers.pdf_router.validate_file_size')
    def test_file_persistence_across_operations(
        self,
        mock_validate_size,
        mock_sanitize,
        mock_extract_meta,
        mock_validate_pdf,
        mock_get_user,
        mock_get_service,
        client,
        sample_pdf_content
    ):
        """Test that file persists across multiple operations."""
        # Setup mocks
        mock_service = Mock()
        mock_service.upload_handler = Mock()
        mock_service.variant_generator = Mock()
        mock_service.topic_extractor = Mock()
        
        mock_metadata = Mock()
        mock_metadata.file_id = "persistent_file"
        mock_service.upload_handler.upload_pdf = AsyncMock(
            return_value=(mock_metadata, "extracted text")
        )
        
        mock_path = Mock()
        mock_path.exists.return_value = True
        mock_service.upload_handler.get_file_path.return_value = mock_path
        
        mock_service.variant_generator.generate = AsyncMock(return_value={
            "variant_type": "summary",
            "content": "Summary"
        })
        
        mock_topic = Mock()
        mock_topic.to_dict = Mock(return_value={"topic": "Test", "relevance": 0.8})
        mock_service.topic_extractor.extract_topics = AsyncMock(return_value=[mock_topic])
        
        mock_get_service.return_value = mock_service
        mock_get_user.return_value = {"user_id": "test_user"}
        mock_validate_pdf.return_value = {"valid": True}
        mock_extract_meta.return_value = {"pages": 1}
        mock_sanitize.return_value = "test.pdf"
        mock_validate_size.return_value = None
        
        # Upload file
        files = {"file": ("test.pdf", io.BytesIO(sample_pdf_content), "application/pdf")}
        upload_response = client.post("/pdf/upload", files=files)
        file_id = upload_response.json()["file_id"]
        
        # Perform multiple operations on the same file
        operations = [
            ("preview", lambda: client.get(f"/pdf/{file_id}/preview")),
            ("variant", lambda: client.post(f"/pdf/{file_id}/variants", json={
                "variant_type": "summary",
                "options": {}
            })),
            ("topics", lambda: client.get(f"/pdf/{file_id}/topics"))
        ]
        
        for op_name, op_func in operations:
            response = op_func()
            assert response.status_code in [200, 201], f"{op_name} operation failed"
            
            # Verify file_id is consistent
            if response.status_code == 200:
                data = response.json()
                if "file_id" in data:
                    assert data["file_id"] == file_id

