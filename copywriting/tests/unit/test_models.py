"""
Unit tests for copywriting models and data structures.
"""
import pytest
from typing import Dict, Any, List
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from models import (
    CopywritingInput,
    CopywritingOutput,
    Feedback,
    SectionFeedback,
    CopyVariantHistory,
    get_settings
)


class TestCopywritingInput:
    """Test cases for CopywritingInput model."""
    
    def test_valid_request_creation(self):
        """Test creating a valid copywriting request."""
        request_data = {
            "product_description": "Zapatos deportivos de alta gama",
            "target_platform": "instagram",
            "tone": "inspirational",
            "target_audience": "Jóvenes activos",
            "key_points": ["Comodidad", "Estilo", "Durabilidad"],
            "instructions": "Enfatiza la innovación",
            "restrictions": ["no mencionar precio"],
            "creativity_level": "balanced",
            "language": "es",
            "use_case": "product_launch",
            "content_type": "social_post"
        }

        request = CopywritingInput(**request_data)
        
        assert request.product_description == "Zapatos deportivos de alta gama"
        assert request.target_platform == "Instagram"
        assert request.tone == "inspirational"
        assert request.creativity_level == 0.8
        assert request.language == "es"
    
    def test_request_with_minimal_data(self):
        """Test creating request with only required fields."""
        request_data = {
            "product_description": "Producto de prueba",
            "target_platform": "Facebook",
            "tone": "informative",
            "language": "es"
        }
        
        request = CopywritingRequest(**request_data)
        
        assert request.product_description == "Producto de prueba"
        assert request.target_audience is None
        assert request.key_points is None
        assert request.creativity_level == 0.5  # default value
    
    def test_invalid_creativity_level(self):
        """Test validation of creativity level bounds."""
        with pytest.raises(ValueError):
            CopywritingRequest(
                product_description="Test",
                target_platform="Instagram",
                tone="inspirational",
                language="es",
                creativity_level=1.5  # Invalid: > 1.0
            )
        
        with pytest.raises(ValueError):
            CopywritingRequest(
                product_description="Test",
                target_platform="Instagram",
                tone="inspirational",
                language="es",
                creativity_level=-0.1  # Invalid: < 0.0
            )
    
    def test_invalid_language_code(self):
        """Test validation of language codes."""
        with pytest.raises(ValueError):
            CopywritingRequest(
                product_description="Test",
                target_platform="Instagram",
                tone="inspirational",
                language="invalid_lang"
            )


class TestCopywritingResponse:
    """Test cases for CopywritingResponse model."""
    
    def test_response_creation(self):
        """Test creating a valid copywriting response."""
        variants = [
            {
                "headline": "¡Descubre la Comodidad Perfecta!",
                "primary_text": "Zapatos deportivos diseñados para tu máximo rendimiento",
                "call_to_action": "Compra ahora",
                "hashtags": ["#deportes", "#comodidad"]
            }
        ]
        
        response_data = {
            "variants": variants,
            "model_used": "gpt-3.5-turbo",
            "generation_time": 2.5,
            "extra_metadata": {"tokens_used": 150}
        }
        
        response = CopywritingResponse(**response_data)
        
        assert len(response.variants) == 1
        assert response.model_used == "gpt-3.5-turbo"
        assert response.generation_time == 2.5
        assert response.extra_metadata["tokens_used"] == 150


class TestBatchCopywritingRequest:
    """Test cases for BatchCopywritingRequest model."""
    
    def test_batch_request_creation(self):
        """Test creating a valid batch request."""
        requests = [
            {
                "product_description": "Zapatos deportivos",
                "target_platform": "Instagram",
                "tone": "inspirational",
                "language": "es"
            },
            {
                "product_description": "Reloj inteligente",
                "target_platform": "Facebook",
                "tone": "informative",
                "language": "es"
            }
        ]
        
        batch_request = BatchCopywritingRequest(requests=requests)
        
        assert len(batch_request.requests) == 2
        assert batch_request.requests[0].product_description == "Zapatos deportivos"
        assert batch_request.requests[1].product_description == "Reloj inteligente"
    
    def test_batch_request_validation(self):
        """Test batch request size validation."""
        # Test maximum batch size
        requests = [
            {
                "product_description": f"Producto {i}",
                "target_platform": "Instagram",
                "tone": "inspirational",
                "language": "es"
            }
            for i in range(20)  # Within limit
        ]
        
        batch_request = BatchCopywritingRequest(requests=requests)
        assert len(batch_request.requests) == 20
        
        # Test exceeding maximum batch size
        with pytest.raises(ValueError):
            requests = [
                {
                    "product_description": f"Producto {i}",
                    "target_platform": "Instagram",
                    "tone": "inspirational",
                    "language": "es"
                }
                for i in range(25)  # Exceeds limit
            ]
            BatchCopywritingRequest(requests=requests)


class TestFeedbackRequest:
    """Test cases for FeedbackRequest model."""
    
    def test_feedback_creation(self):
        """Test creating a valid feedback request."""
        feedback_data = {
            "type": "human",
            "score": 0.9,
            "comments": "Muy buen copy",
            "user_id": "user123",
            "timestamp": "2024-06-01T12:00:00Z"
        }
        
        request = FeedbackRequest(
            variant_id="variant_1",
            feedback=feedback_data
        )
        
        assert request.variant_id == "variant_1"
        assert request.feedback["type"] == "human"
        assert request.feedback["score"] == 0.9
        assert request.feedback["comments"] == "Muy buen copy"
    
    def test_feedback_score_validation(self):
        """Test feedback score validation."""
        with pytest.raises(ValueError):
            FeedbackRequest(
                variant_id="variant_1",
                feedback={
                    "type": "human",
                    "score": 1.5,  # Invalid: > 1.0
                    "comments": "Test",
                    "user_id": "user123"
                }
            )


class TestTaskStatus:
    """Test cases for TaskStatus model."""
    
    def test_task_status_creation(self):
        """Test creating task status responses."""
        # Success status
        success_status = TaskStatus(
            status="SUCCESS",
            result={"variants": []},
            error=None
        )
        
        assert success_status.status == "SUCCESS"
        assert success_status.result == {"variants": []}
        assert success_status.error is None
        
        # Failure status
        failure_status = TaskStatus(
            status="FAILURE",
            result=None,
            error="Test error message"
        )
        
        assert failure_status.status == "FAILURE"
        assert failure_status.result is None
        assert failure_status.error == "Test error message"
    
    def test_task_status_validation(self):
        """Test task status validation."""
        with pytest.raises(ValueError):
            TaskStatus(
                status="INVALID_STATUS",
                result=None,
                error=None
            )





