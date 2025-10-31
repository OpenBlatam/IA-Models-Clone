"""
Advanced tests for the Content Redundancy Detector system
"""

import pytest
import asyncio
import time
import json
from typing import Dict, Any

from fastapi.testclient import TestClient
from app import app

# Test client
client = TestClient(app)


class TestBatchProcessing:
    """Test batch processing functionality"""
    
    def test_batch_processing_analyze(self):
        """Test batch processing with content analysis"""
        batch_data = {
            "jobs": [
                {
                    "operation": "analyze",
                    "input": {
                        "content": "This is a test content for analysis. It contains multiple sentences to test redundancy detection."
                    }
                },
                {
                    "operation": "analyze",
                    "input": {
                        "content": "Another test content with different words and structure for comparison."
                    }
                }
            ]
        }
        
        response = client.post("/batch/process", json=batch_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "batch_id" in data
        assert data["status"] == "completed"
        assert data["total_jobs"] == 2
        assert data["completed_jobs"] == 2
        assert data["failed_jobs"] == 0
        assert data["progress"] == 100.0
    
    def test_batch_processing_similarity(self):
        """Test batch processing with similarity detection"""
        batch_data = {
            "jobs": [
                {
                    "operation": "similarity",
                    "input": {
                        "text1": "This is the first text for similarity comparison.",
                        "text2": "This is the second text for similarity comparison.",
                        "threshold": 0.5
                    }
                }
            ]
        }
        
        response = client.post("/batch/process", json=batch_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "completed"
        assert data["total_jobs"] == 1
        assert data["completed_jobs"] == 1
    
    def test_batch_processing_quality(self):
        """Test batch processing with quality assessment"""
        batch_data = {
            "jobs": [
                {
                    "operation": "quality",
                    "input": {
                        "content": "This is a well-written text with proper grammar and structure for quality assessment."
                    }
                }
            ]
        }
        
        response = client.post("/batch/process", json=batch_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "completed"
        assert data["total_jobs"] == 1
        assert data["completed_jobs"] == 1
    
    def test_batch_processing_mixed_operations(self):
        """Test batch processing with mixed operations"""
        batch_data = {
            "jobs": [
                {
                    "operation": "analyze",
                    "input": {
                        "content": "Test content for analysis."
                    }
                },
                {
                    "operation": "similarity",
                    "input": {
                        "text1": "First text for comparison.",
                        "text2": "Second text for comparison.",
                        "threshold": 0.5
                    }
                },
                {
                    "operation": "quality",
                    "input": {
                        "content": "Test content for quality assessment."
                    }
                }
            ]
        }
        
        response = client.post("/batch/process", json=batch_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "completed"
        assert data["total_jobs"] == 3
        assert data["completed_jobs"] == 3
    
    def test_batch_processing_empty_jobs(self):
        """Test batch processing with empty jobs list"""
        batch_data = {"jobs": []}
        
        response = client.post("/batch/process", json=batch_data)
        assert response.status_code == 400
        assert "No jobs provided" in response.json()["detail"]
    
    def test_batch_processing_invalid_operation(self):
        """Test batch processing with invalid operation"""
        batch_data = {
            "jobs": [
                {
                    "operation": "invalid_operation",
                    "input": {
                        "content": "Test content."
                    }
                }
            ]
        }
        
        response = client.post("/batch/process", json=batch_data)
        assert response.status_code == 400
        assert "Unknown operation" in response.json()["detail"]
    
    def test_get_batch_status(self):
        """Test getting batch status"""
        # First create a batch
        batch_data = {
            "jobs": [
                {
                    "operation": "analyze",
                    "input": {
                        "content": "Test content for batch status."
                    }
                }
            ]
        }
        
        response = client.post("/batch/process", json=batch_data)
        assert response.status_code == 200
        
        batch_id = response.json()["batch_id"]
        
        # Get batch status
        response = client.get(f"/batch/{batch_id}")
        assert response.status_code == 200
        
        data = response.json()
        assert data["batch_id"] == batch_id
        assert data["status"] == "completed"
        assert data["total_jobs"] == 1
    
    def test_get_batch_status_not_found(self):
        """Test getting status for non-existent batch"""
        response = client.get("/batch/non_existent_batch")
        assert response.status_code == 404
        assert "Batch not found" in response.json()["detail"]
    
    def test_get_all_batches(self):
        """Test getting all batches"""
        response = client.get("/batch")
        assert response.status_code == 200
        
        data = response.json()
        assert "batches" in data
        assert isinstance(data["batches"], list)


class TestWebhooks:
    """Test webhook functionality"""
    
    def test_register_webhook(self):
        """Test webhook registration"""
        webhook_data = {
            "id": "test_webhook",
            "url": "https://example.com/webhook",
            "events": ["analysis_completed", "similarity_completed"],
            "secret": "test_secret",
            "timeout": 30,
            "retry_count": 3
        }
        
        response = client.post("/webhooks/register", json=webhook_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["message"] == "Webhook registered successfully"
        assert data["endpoint_id"] == "test_webhook"
    
    def test_get_webhooks(self):
        """Test getting all webhooks"""
        response = client.get("/webhooks")
        assert response.status_code == 200
        
        data = response.json()
        assert "endpoints" in data
        assert isinstance(data["endpoints"], list)
    
    def test_get_webhook_stats(self):
        """Test getting webhook statistics"""
        response = client.get("/webhooks/stats")
        assert response.status_code == 200
        
        data = response.json()
        assert "total_deliveries" in data
        assert "successful_deliveries" in data
        assert "failed_deliveries" in data
        assert "active_endpoints" in data
    
    def test_unregister_webhook(self):
        """Test webhook unregistration"""
        # First register a webhook
        webhook_data = {
            "id": "test_webhook_unregister",
            "url": "https://example.com/webhook",
            "events": ["analysis_completed"]
        }
        
        response = client.post("/webhooks/register", json=webhook_data)
        assert response.status_code == 200
        
        # Then unregister it
        response = client.delete("/webhooks/test_webhook_unregister")
        assert response.status_code == 200
        
        data = response.json()
        assert data["message"] == "Webhook unregistered successfully"
    
    def test_unregister_webhook_not_found(self):
        """Test unregistering non-existent webhook"""
        response = client.delete("/webhooks/non_existent_webhook")
        assert response.status_code == 404
        assert "Webhook endpoint not found" in response.json()["detail"]


class TestExport:
    """Test export functionality"""
    
    def test_export_json(self):
        """Test JSON export"""
        export_data = {
            "data": [
                {
                    "content": "Test content 1",
                    "analysis": {"word_count": 2, "redundancy_score": 0.1}
                },
                {
                    "content": "Test content 2",
                    "analysis": {"word_count": 2, "redundancy_score": 0.2}
                }
            ],
            "format": "json",
            "filename": "test_export.json"
        }
        
        response = client.post("/export", json=export_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "export_id" in data
        assert data["format"] == "json"
        assert data["filename"] == "test_export.json"
        assert data["file_size"] > 0
    
    def test_export_csv(self):
        """Test CSV export"""
        export_data = {
            "data": [
                {
                    "content": "Test content 1",
                    "word_count": 2,
                    "redundancy_score": 0.1
                },
                {
                    "content": "Test content 2",
                    "word_count": 2,
                    "redundancy_score": 0.2
                }
            ],
            "format": "csv"
        }
        
        response = client.post("/export", json=export_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["format"] == "csv"
        assert data["file_size"] > 0
    
    def test_export_xml(self):
        """Test XML export"""
        export_data = {
            "data": [
                {
                    "content": "Test content",
                    "analysis": {"word_count": 2}
                }
            ],
            "format": "xml"
        }
        
        response = client.post("/export", json=export_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["format"] == "xml"
        assert data["file_size"] > 0
    
    def test_export_txt(self):
        """Test TXT export"""
        export_data = {
            "data": [
                {
                    "content": "Test content",
                    "analysis": {"word_count": 2}
                }
            ],
            "format": "txt"
        }
        
        response = client.post("/export", json=export_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["format"] == "txt"
        assert data["file_size"] > 0
    
    def test_export_zip(self):
        """Test ZIP export"""
        export_data = {
            "data": [
                {
                    "content": "Test content",
                    "analysis": {"word_count": 2}
                }
            ],
            "format": "zip"
        }
        
        response = client.post("/export", json=export_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["format"] == "zip"
        assert data["file_size"] > 0
    
    def test_export_empty_data(self):
        """Test export with empty data"""
        export_data = {
            "data": [],
            "format": "json"
        }
        
        response = client.post("/export", json=export_data)
        assert response.status_code == 400
        assert "No data provided" in response.json()["detail"]
    
    def test_get_export(self):
        """Test getting export by ID"""
        # First create an export
        export_data = {
            "data": [{"content": "Test content", "analysis": {"word_count": 2}}],
            "format": "json"
        }
        
        response = client.post("/export", json=export_data)
        assert response.status_code == 200
        
        export_id = response.json()["export_id"]
        
        # Get export
        response = client.get(f"/export/{export_id}")
        assert response.status_code == 200
        
        data = response.json()
        assert data["export_id"] == export_id
        assert data["format"] == "json"
    
    def test_get_export_not_found(self):
        """Test getting non-existent export"""
        response = client.get("/export/non_existent_export")
        assert response.status_code == 404
        assert "Export not found" in response.json()["detail"]
    
    def test_get_all_exports(self):
        """Test getting all exports"""
        response = client.get("/export")
        assert response.status_code == 200
        
        data = response.json()
        assert "exports" in data
        assert isinstance(data["exports"], list)


class TestAnalytics:
    """Test analytics functionality"""
    
    def test_performance_analytics(self):
        """Test performance analytics"""
        response = client.get("/analytics/performance")
        assert response.status_code == 200
        
        data = response.json()
        assert "summary" in data
        assert "endpoint_performance" in data
        assert "performance_trends" in data
        assert "recommendations" in data
    
    def test_content_analytics(self):
        """Test content analytics"""
        response = client.get("/analytics/content")
        assert response.status_code == 200
        
        data = response.json()
        # Should return either analytics data or message about no data
        assert isinstance(data, dict)
    
    def test_similarity_analytics(self):
        """Test similarity analytics"""
        response = client.get("/analytics/similarity")
        assert response.status_code == 200
        
        data = response.json()
        # Should return either analytics data or message about no data
        assert isinstance(data, dict)
    
    def test_quality_analytics(self):
        """Test quality analytics"""
        response = client.get("/analytics/quality")
        assert response.status_code == 200
        
        data = response.json()
        # Should return either analytics data or message about no data
        assert isinstance(data, dict)
    
    def test_get_all_analytics_reports(self):
        """Test getting all analytics reports"""
        response = client.get("/analytics/reports")
        assert response.status_code == 200
        
        data = response.json()
        assert "reports" in data
        assert isinstance(data["reports"], list)


class TestIntegration:
    """Test integration between different systems"""
    
    def test_full_workflow(self):
        """Test complete workflow with all systems"""
        # 1. Analyze content
        content_data = {
            "content": "This is a comprehensive test content for the full workflow integration test."
        }
        
        response = client.post("/analyze", json=content_data)
        assert response.status_code == 200
        
        # 2. Check similarity
        similarity_data = {
            "text1": "This is the first text for similarity comparison.",
            "text2": "This is the second text for similarity comparison.",
            "threshold": 0.5
        }
        
        response = client.post("/similarity", json=similarity_data)
        assert response.status_code == 200
        
        # 3. Assess quality
        quality_data = {
            "content": "This is a well-written text with proper grammar and structure."
        }
        
        response = client.post("/quality", json=quality_data)
        assert response.status_code == 200
        
        # 4. Get performance analytics
        response = client.get("/analytics/performance")
        assert response.status_code == 200
        
        # 5. Get system metrics
        response = client.get("/metrics")
        assert response.status_code == 200
        
        # 6. Get health status
        response = client.get("/health")
        assert response.status_code == 200
    
    def test_batch_with_analytics(self):
        """Test batch processing with analytics integration"""
        # Create batch
        batch_data = {
            "jobs": [
                {
                    "operation": "analyze",
                    "input": {
                        "content": "Test content for batch analytics integration."
                    }
                }
            ]
        }
        
        response = client.post("/batch/process", json=batch_data)
        assert response.status_code == 200
        
        # Check analytics after batch processing
        response = client.get("/analytics/content")
        assert response.status_code == 200
    
    def test_webhook_with_analysis(self):
        """Test webhook integration with analysis"""
        # Register webhook
        webhook_data = {
            "id": "integration_test_webhook",
            "url": "https://example.com/webhook",
            "events": ["analysis_completed"]
        }
        
        response = client.post("/webhooks/register", json=webhook_data)
        assert response.status_code == 200
        
        # Perform analysis (should trigger webhook)
        content_data = {
            "content": "Test content for webhook integration."
        }
        
        response = client.post("/analyze", json=content_data)
        assert response.status_code == 200
        
        # Check webhook stats
        response = client.get("/webhooks/stats")
        assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


