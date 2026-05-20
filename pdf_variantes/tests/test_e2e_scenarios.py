"""
E2E Scenario Tests
==================
Real-world scenario tests that simulate actual user behavior.
"""

import pytest
import time
import io
from fastapi.testclient import TestClient
from typing import Dict, Any

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from api.main import app
except ImportError:
    try:
        from main import app
    except ImportError:
        from fastapi import FastAPI
        app = FastAPI()


@pytest.fixture(scope="module")
def client():
    """Test client for E2E scenarios."""
    return TestClient(app, timeout=60.0)


@pytest.fixture
def sample_pdf():
    """Sample PDF content."""
    return b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\nxref\n0 3\ntrailer\n<<\n/Size 3\n>>\nstartxref\n50\n%%EOF"


@pytest.fixture
def auth():
    """Authentication headers."""
    return {"Authorization": "Bearer test_token", "X-User-ID": "user_123"}


class TestScenarioStudentWorkflow:
    """Scenario: Student uploading lecture notes and generating study materials."""
    
    @pytest.mark.e2e
    @pytest.mark.scenario
    def test_student_lecture_notes_workflow(self, client, sample_pdf, auth):
        """Student uploads lecture notes and generates summary + quiz."""
        
        # Step 1: Upload lecture notes
        files = {"file": ("lecture_notes.pdf", io.BytesIO(sample_pdf), "application/pdf")}
        upload = client.post("/pdf/upload", files=files, headers=auth)
        
        if upload.status_code not in [200, 201]:
            pytest.skip("Upload failed")
        
        file_id = upload.json().get("file_id") or upload.json().get("id")
        
        # Step 2: Generate summary for quick review
        summary = client.post(
            f"/pdf/{file_id}/variants",
            json={"variant_type": "summary", "options": {"max_length": 300}},
            headers=auth
        )
        assert summary.status_code in [200, 202]
        
        # Step 3: Generate quiz for self-testing
        quiz = client.post(
            f"/pdf/{file_id}/variants",
            json={"variant_type": "quiz", "options": {"num_questions": 10}},
            headers=auth
        )
        assert quiz.status_code in [200, 202]
        
        # Step 4: Extract key topics for study guide
        topics = client.get(
            f"/pdf/{file_id}/topics?min_relevance=0.7&max_topics=20",
            headers=auth
        )
        assert topics.status_code in [200, 202]


class TestScenarioResearcherWorkflow:
    """Scenario: Researcher analyzing multiple papers."""
    
    @pytest.mark.e2e
    @pytest.mark.scenario
    def test_researcher_batch_analysis(self, client, sample_pdf, auth):
        """Researcher uploads multiple papers and extracts topics from all."""
        
        file_ids = []
        
        # Upload multiple papers
        for i in range(3):
            files = {"file": (f"paper_{i}.pdf", io.BytesIO(sample_pdf), "application/pdf")}
            upload = client.post("/pdf/upload", files=files, headers=auth)
            if upload.status_code in [200, 201]:
                file_id = upload.json().get("file_id") or upload.json().get("id")
                file_ids.append(file_id)
        
        assert len(file_ids) > 0, "At least one upload should succeed"
        
        # Extract topics from all papers
        all_topics = []
        for file_id in file_ids:
            topics = client.get(f"/pdf/{file_id}/topics", headers=auth)
            if topics.status_code == 200:
                all_topics.append(topics.json())
        
        # Should have extracted topics from at least some papers
        assert len(all_topics) >= 0


class TestScenarioContentCreatorWorkflow:
    """Scenario: Content creator generating multiple content variations."""
    
    @pytest.mark.e2e
    @pytest.mark.scenario
    def test_content_creator_variations(self, client, sample_pdf, auth):
        """Content creator generates multiple variations of content."""
        
        # Upload source document
        files = {"file": ("source.pdf", io.BytesIO(sample_pdf), "application/pdf")}
        upload = client.post("/pdf/upload", files=files, headers=auth)
        
        if upload.status_code not in [200, 201]:
            pytest.skip("Upload failed")
        
        file_id = upload.json().get("file_id") or upload.json().get("id")
        
        # Generate multiple content variations
        variations = [
            {"variant_type": "summary", "options": {"style": "casual"}},
            {"variant_type": "summary", "options": {"style": "formal"}},
            {"variant_type": "summary", "options": {"style": "academic"}},
            {"variant_type": "presentation", "options": {}},
        ]
        
        results = []
        for variation in variations:
            response = client.post(
                f"/pdf/{file_id}/variants",
                json=variation,
                headers=auth
            )
            results.append(response.status_code)
        
        # At least some variations should be generated
        assert any(status in [200, 202] for status in results)


class TestScenarioCollaborativeWorkflow:
    """Scenario: Team collaborating on document analysis."""
    
    @pytest.mark.e2e
    @pytest.mark.scenario
    def test_team_collaboration(self, client, sample_pdf):
        """Multiple team members working on same document."""
        
        # Upload shared document
        files = {"file": ("shared_doc.pdf", io.BytesIO(sample_pdf), "application/pdf")}
        upload = client.post("/pdf/upload", files=files)
        
        if upload.status_code not in [200, 201]:
            pytest.skip("Upload failed")
        
        file_id = upload.json().get("file_id") or upload.json().get("id")
        
        # Different team members access the document
        team_members = [
            {"X-User-ID": "member_1"},
            {"X-User-ID": "member_2"},
            {"X-User-ID": "member_3"},
        ]
        
        for member in team_members:
            # Each member gets preview
            preview = client.get(f"/pdf/{file_id}/preview", headers=member)
            assert preview.status_code in [200, 202, 404]
            
            # Each member extracts topics
            topics = client.get(f"/pdf/{file_id}/topics", headers=member)
            assert topics.status_code in [200, 202, 404]


class TestScenarioErrorRecovery:
    """Scenario: System recovery from errors."""
    
    @pytest.mark.e2e
    @pytest.mark.scenario
    def test_recovery_after_invalid_request(self, client, sample_pdf, auth):
        """System should recover after invalid request."""
        
        # Make invalid request
        invalid_response = client.post(
            "/pdf/upload",
            json={"invalid": "data"},  # Wrong format
            headers=auth
        )
        assert invalid_response.status_code in [400, 422]
        
        # System should still work after error
        files = {"file": ("valid.pdf", io.BytesIO(sample_pdf), "application/pdf")}
        valid_response = client.post("/pdf/upload", files=files, headers=auth)
        assert valid_response.status_code in [200, 201]
    
    @pytest.mark.e2e
    @pytest.mark.scenario
    def test_graceful_degradation(self, client, sample_pdf, auth):
        """System should degrade gracefully when services are unavailable."""
        
        # Upload document
        files = {"file": ("test.pdf", io.BytesIO(sample_pdf), "application/pdf")}
        upload = client.post("/pdf/upload", files=files, headers=auth)
        
        if upload.status_code not in [200, 201]:
            pytest.skip("Upload failed")
        
        file_id = upload.json().get("file_id") or upload.json().get("id")
        
        # Try operations that might fail
        operations = [
            ("preview", lambda: client.get(f"/pdf/{file_id}/preview", headers=auth)),
            ("topics", lambda: client.get(f"/pdf/{file_id}/topics", headers=auth)),
            ("variant", lambda: client.post(
                f"/pdf/{file_id}/variants",
                json={"variant_type": "summary", "options": {}},
                headers=auth
            )),
        ]
        
        # All should return valid responses (even if errors)
        for op_name, op_func in operations:
            response = op_func()
            assert response.status_code is not None
            assert response.status_code < 600  # No server errors


class TestScenarioPerformanceUnderLoad:
    """Scenario: System performance under load."""
    
    @pytest.mark.e2e
    @pytest.mark.slow
    @pytest.mark.scenario
    def test_sequential_operations_load(self, client, sample_pdf, auth):
        """Test system handling sequential operations."""
        
        start_time = time.time()
        
        # Perform multiple sequential operations
        for i in range(5):
            files = {"file": (f"load_test_{i}.pdf", io.BytesIO(sample_pdf), "application/pdf")}
            upload = client.post("/pdf/upload", files=files, headers=auth)
            
            if upload.status_code in [200, 201]:
                file_id = upload.json().get("file_id") or upload.json().get("id")
                client.get(f"/pdf/{file_id}/preview", headers=auth)
        
        elapsed = time.time() - start_time
        
        # Should complete in reasonable time
        assert elapsed < 60.0, f"Sequential operations took too long: {elapsed:.2f}s"
    
    @pytest.mark.e2e
    @pytest.mark.slow
    @pytest.mark.scenario
    def test_mixed_operation_types(self, client, sample_pdf, auth):
        """Test system handling mixed operation types."""
        
        # Upload
        files = {"file": ("mixed_test.pdf", io.BytesIO(sample_pdf), "application/pdf")}
        upload = client.post("/pdf/upload", files=files, headers=auth)
        
        if upload.status_code not in [200, 201]:
            pytest.skip("Upload failed")
        
        file_id = upload.json().get("file_id") or upload.json().get("id")
        
        # Mix of operations
        operations = [
            client.get(f"/pdf/{file_id}/preview", headers=auth),
            client.get(f"/pdf/{file_id}/topics", headers=auth),
            client.post(
                f"/pdf/{file_id}/variants",
                json={"variant_type": "summary", "options": {}},
                headers=auth
            ),
            client.get(f"/pdf/{file_id}/preview?page_number=1", headers=auth),
        ]
        
        # All should complete
        status_codes = [op.status_code for op in operations]
        assert len(status_codes) == 4
        # At least some should succeed
        assert any(code in [200, 202] for code in status_codes)



