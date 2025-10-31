"""
Simple unit tests for copywriting models and data structures.
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
        assert request.target_platform == "instagram"
        assert request.tone == "inspirational"
        assert request.creativity_level == "balanced"
        assert request.language == "es"
        assert request.use_case == "product_launch"
        assert request.content_type == "social_post"
    
    def test_request_with_minimal_data(self):
        """Test creating request with only required fields."""
        request_data = {
            "product_description": "Producto de prueba",
            "target_platform": "facebook",
            "tone": "informative",
            "language": "es",
            "use_case": "product_launch",
            "content_type": "social_post"
        }

        request = CopywritingInput(**request_data)
        
        assert request.product_description == "Producto de prueba"
        assert request.target_platform == "facebook"
        assert request.tone == "informative"
        assert request.language == "es"
        assert request.use_case == "product_launch"
        assert request.content_type == "social_post"
    
    def test_invalid_creativity_level(self):
        """Test validation of creativity level bounds."""
        with pytest.raises(Exception):  # Pydantic validation error
            CopywritingInput(
                product_description="Test",
                target_platform="instagram",
                tone="inspirational",
                language="es",
                use_case="product_launch",
                content_type="social_post",
                creativity_level="invalid_level"  # Invalid creativity level
            )
    
    def test_invalid_language_code(self):
        """Test validation of language codes."""
        with pytest.raises(Exception):  # Pydantic validation error
            CopywritingInput(
                product_description="Test",
                target_platform="instagram",
                tone="inspirational",
                language="invalid_lang",
                use_case="product_launch",
                content_type="social_post"
            )
    
    def test_invalid_platform(self):
        """Test validation of platform values."""
        with pytest.raises(Exception):  # Pydantic validation error
            CopywritingInput(
                product_description="Test",
                target_platform="invalid_platform",
                tone="inspirational",
                language="es",
                use_case="product_launch",
                content_type="social_post"
            )
    
    def test_invalid_tone(self):
        """Test validation of tone values."""
        with pytest.raises(Exception):  # Pydantic validation error
            CopywritingInput(
                product_description="Test",
                target_platform="instagram",
                tone="invalid_tone",
                language="es",
                use_case="product_launch",
                content_type="social_post"
            )
    
    def test_serialization(self):
        """Test model serialization to JSON."""
        request = CopywritingInput(
            product_description="Test product",
            target_platform="instagram",
            tone="inspirational",
            language="es",
            use_case="product_launch",
            content_type="social_post"
        )
        
        json_str = request.model_dump_json()
        assert isinstance(json_str, str)
        assert "Test product" in json_str
        assert "instagram" in json_str
    
    def test_deserialization(self):
        """Test model deserialization from JSON."""
        request_data = {
            "product_description": "Test product",
            "target_platform": "instagram",
            "tone": "inspirational",
            "language": "es",
            "use_case": "product_launch",
            "content_type": "social_post"
        }
        
        request = CopywritingInput(**request_data)
        json_str = request.model_dump_json()
        parsed_request = CopywritingInput.model_validate_json(json_str)
        
        assert parsed_request.product_description == request.product_description
        assert parsed_request.target_platform == request.target_platform
        assert parsed_request.tone == request.tone


class TestCopywritingOutput:
    """Test cases for CopywritingOutput model."""
    
    def test_response_creation(self):
        """Test creating a valid copywriting response."""
        variants = [
            {
                "variant_id": "variant_1",
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
            "tokens_used": 150
        }

        response = CopywritingOutput(**response_data)
        
        assert len(response.variants) == 1
        assert response.variants[0].headline == "¡Descubre la Comodidad Perfecta!"
        assert response.model_used == "gpt-3.5-turbo"
        assert response.generation_time == 2.5
        assert response.tokens_used == 150
    
    def test_response_serialization(self):
        """Test response serialization."""
        response = CopywritingOutput(
            variants=[{"variant_id": "test_1", "headline": "Test", "primary_text": "Content"}],
            model_used="gpt-3.5-turbo",
            generation_time=1.0,
            tokens_used=50
        )
        
        json_str = response.model_dump_json()
        assert isinstance(json_str, str)
        assert "Test" in json_str


class TestFeedback:
    """Test cases for Feedback model."""
    
    def test_feedback_creation(self):
        """Test creating a valid feedback."""
        feedback_data = {
            "type": "human",
            "score": 0.9,
            "comments": "Muy buen copy",
            "user_id": "user123"
        }

        feedback = Feedback(**feedback_data)
        
        assert feedback.type == "human"
        assert feedback.score == 0.9
        assert feedback.comments == "Muy buen copy"
        assert feedback.user_id == "user123"
    
    def test_feedback_score_validation(self):
        """Test feedback score validation."""
        with pytest.raises(Exception):  # Pydantic validation error
            Feedback(
                type="human",
                score=1.5,  # Invalid: > 1.0
                comments="Test",
                user_id="user123"
            )
    
    def test_feedback_serialization(self):
        """Test feedback serialization."""
        feedback = Feedback(
            type="model",
            score=0.8,
            comments="Good copy",
            user_id="user456"
        )
        
        json_str = feedback.model_dump_json()
        assert isinstance(json_str, str)
        assert "Good copy" in json_str


class TestSectionFeedback:
    """Test cases for SectionFeedback model."""
    
    def test_section_feedback_creation(self):
        """Test creating section feedback."""
        base_feedback = Feedback(
            type="human",
            score=0.9,
            comments="Great headline",
            user_id="user123"
        )
        
        section_feedback = SectionFeedback(
            section="headline",
            feedback=base_feedback,
            suggestions=["Add more emotion", "Use action words"]
        )
        
        assert section_feedback.section == "headline"
        assert section_feedback.feedback.type == "human"
        assert section_feedback.feedback.score == 0.9
        assert len(section_feedback.suggestions) == 2
        assert "Add more emotion" in section_feedback.suggestions
    
    def test_section_feedback_serialization(self):
        """Test section feedback serialization."""
        base_feedback = Feedback(
            type="model",
            score=0.8,
            comments="Good section",
            user_id="user456"
        )
        
        section_feedback = SectionFeedback(
            section="body",
            feedback=base_feedback,
            suggestions=["Improve flow"]
        )
        
        json_str = section_feedback.model_dump_json()
        assert isinstance(json_str, str)
        assert "body" in json_str


class TestCopyVariantHistory:
    """Test cases for CopyVariantHistory model."""
    
    def test_variant_history_creation(self):
        """Test creating variant history."""
        from datetime import datetime
        
        history = CopyVariantHistory(
            variant_id="variant_123",
            previous_versions=["v1.0", "v1.1"],
            change_log=["Updated headline", "Modified CTA"],
            created_at=datetime.now()
        )
        
        assert history.variant_id == "variant_123"
        assert len(history.previous_versions) == 2
        assert len(history.change_log) == 2
        assert history.created_at is not None
    
    def test_variant_history_serialization(self):
        """Test variant history serialization."""
        from datetime import datetime
        
        history = CopyVariantHistory(
            variant_id="variant_456",
            previous_versions=["v1.0"],
            change_log=["Initial version"],
            created_at=datetime.now()
        )
        
        json_str = history.model_dump_json()
        assert isinstance(json_str, str)
        assert "variant_456" in json_str


class TestSettings:
    """Test cases for settings management."""
    
    def test_get_settings(self):
        """Test getting application settings."""
        settings = get_settings()
        
        # Check that settings object is returned
        assert settings is not None
        # Add more specific assertions based on your settings structure
        # assert hasattr(settings, 'some_setting')
    
    def test_settings_serialization(self):
        """Test settings serialization."""
        settings = get_settings()
        
        # Test that settings can be serialized
        if hasattr(settings, 'model_dump_json'):
            json_str = settings.model_dump_json()
            assert isinstance(json_str, str)


class TestModelValidation:
    """Test cases for model validation."""
    
    def test_required_fields_validation(self):
        """Test that required fields are validated."""
        # Test missing required field
        with pytest.raises(Exception):
            CopywritingInput(
                product_description="Test",
                # Missing required fields
            )
    
    def test_enum_validation(self):
        """Test enum field validation."""
        # Test valid enum values
        valid_request = CopywritingInput(
            product_description="Test",
            target_platform="instagram",
            tone="inspirational",
            language="es",
            use_case="product_launch",
            content_type="social_post"
        )
        assert valid_request.target_platform == "instagram"
        
        # Test invalid enum values
        with pytest.raises(Exception):
            CopywritingInput(
                product_description="Test",
                target_platform="invalid_platform",
                tone="inspirational",
                language="es",
                use_case="product_launch",
                content_type="social_post"
            )
    
    def test_string_length_validation(self):
        """Test string length validation."""
        # Test with a string that's within the limit (2000 characters max)
        long_description = "x" * 1000  # Within the 2000 character limit
        
        # This should work as the model allows descriptions up to 2000 characters
        request = CopywritingInput(
            product_description=long_description,
            target_platform="instagram",
            tone="inspirational",
            language="es",
            use_case="product_launch",
            content_type="social_post"
        )
        assert len(request.product_description) == 1000
        
        # Test with a string that exceeds the limit
        with pytest.raises(Exception):  # Pydantic validation error
            CopywritingInput(
                product_description="x" * 3000,  # Exceeds 2000 character limit
                target_platform="instagram",
                tone="inspirational",
                language="es",
                use_case="product_launch",
                content_type="social_post"
            )
    
    def test_list_validation(self):
        """Test list field validation."""
        # Test valid list
        request = CopywritingInput(
            product_description="Test",
            target_platform="instagram",
            tone="inspirational",
            language="es",
            use_case="product_launch",
            content_type="social_post",
            key_points=["Point 1", "Point 2"]
        )
        assert len(request.key_points) == 2
        
        # Test empty list (should be valid)
        request_empty = CopywritingInput(
            product_description="Test",
            target_platform="instagram",
            tone="inspirational",
            language="es",
            use_case="product_launch",
            content_type="social_post",
            key_points=[]
        )
        assert len(request_empty.key_points) == 0
