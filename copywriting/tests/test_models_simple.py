"""
Simple tests for copywriting models.
"""
import pytest
import sys
import os
from typing import Dict, Any, List

# Add the parent directory to the path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_import_models():
    """Test that we can import the main models."""
    try:
        from models import (
            CopywritingInput,
            CopywritingOutput,
            Feedback,
            SectionFeedback,
            CopyVariantHistory,
            get_settings
        )
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import models: {e}")

def test_copywriting_input_creation():
    """Test creating a CopywritingInput object."""
    from models import CopywritingInput
    
    input_data = {
        "product_description": "Zapatos deportivos de alta gama",
        "target_platform": "instagram",
        "content_type": "social_post",
        "tone": "inspirational",
        "use_case": "product_launch"
    }
    
    input_obj = CopywritingInput(**input_data)
    
    assert input_obj.product_description == "Zapatos deportivos de alta gama"
    assert input_obj.target_platform == "instagram"
    assert input_obj.content_type == "social_post"
    assert input_obj.tone == "inspirational"
    assert input_obj.use_case == "product_launch"

def test_copywriting_input_with_optional_fields():
    """Test creating a CopywritingInput with optional fields."""
    from models import CopywritingInput
    
    input_data = {
        "product_description": "Laptop gaming profesional",
        "target_platform": "facebook",
        "content_type": "ad_copy",
        "tone": "professional",
        "use_case": "sales_conversion",
        "target_audience": "Profesionales tech",
        "key_points": ["Rendimiento", "Calidad", "Innovación"],
        "instructions": "Destaca las especificaciones técnicas",
        "restrictions": ["no mencionar precio"],
        "creativity_level": "balanced",
        "language": "es"
    }
    
    input_obj = CopywritingInput(**input_data)
    
    assert input_obj.product_description == "Laptop gaming profesional"
    assert input_obj.target_platform == "facebook"
    assert input_obj.content_type == "ad_copy"
    assert input_obj.tone == "professional"
    assert input_obj.use_case == "sales_conversion"
    assert input_obj.target_audience == "Profesionales tech"
    assert input_obj.key_points == ["Rendimiento", "Calidad", "Innovación"]
    assert input_obj.instructions == "Destaca las especificaciones técnicas"
    assert input_obj.restrictions == ["no mencionar precio"]
    assert input_obj.creativity_level == "balanced"
    assert input_obj.language == "es"

def test_feedback_creation():
    """Test creating a Feedback object."""
    from models import Feedback, FeedbackType
    
    feedback_data = {
        "type": FeedbackType.human,
        "comments": "Excelente copy, muy atractivo",
        "score": 0.9
    }
    
    feedback = Feedback(**feedback_data)
    
    assert feedback.type == FeedbackType.human
    assert feedback.comments == "Excelente copy, muy atractivo"
    assert feedback.score == 0.9

def test_section_feedback_creation():
    """Test creating a SectionFeedback object."""
    from models import SectionFeedback, Feedback, FeedbackType
    
    feedback = Feedback(
        type=FeedbackType.model,
        comments="Podría ser más conciso",
        score=0.6
    )
    
    section_feedback_data = {
        "section": "headline",
        "feedback": feedback,
        "suggestions": ["Usar palabras más impactantes", "Reducir longitud"]
    }
    
    section_feedback = SectionFeedback(**section_feedback_data)
    
    assert section_feedback.section == "headline"
    assert section_feedback.feedback.type == FeedbackType.model
    assert section_feedback.feedback.comments == "Podría ser más conciso"
    assert section_feedback.suggestions == ["Usar palabras más impactantes", "Reducir longitud"]

def test_copy_variant_history_creation():
    """Test creating a CopyVariantHistory object."""
    from models import CopyVariantHistory
    from datetime import datetime
    
    history_data = {
        "variant_id": "var_123",
        "previous_versions": ["v1", "v2"],
        "change_log": ["Updated headline", "Modified CTA"],
        "created_at": datetime.now()
    }
    
    history = CopyVariantHistory(**history_data)
    
    assert history.variant_id == "var_123"
    assert history.previous_versions == ["v1", "v2"]
    assert history.change_log == ["Updated headline", "Modified CTA"]
    assert isinstance(history.created_at, datetime)

def test_get_settings():
    """Test getting settings."""
    from models import get_settings
    
    settings = get_settings()
    
    assert settings is not None
    assert hasattr(settings, 'api_key')
    assert hasattr(settings, 'max_concurrent_requests')
    assert hasattr(settings, 'request_timeout')

def test_model_validation():
    """Test model validation."""
    from models import CopywritingInput
    
    # Test with invalid data
    with pytest.raises(Exception):  # Should raise validation error
        CopywritingInput(
            product_description="",  # Empty description should fail
            target_platform="invalid_platform",  # Invalid platform
            content_type="invalid_type",  # Invalid content type
            tone="invalid_tone"  # Invalid tone
        )

def test_model_serialization():
    """Test model serialization."""
    from models import CopywritingInput
    import json
    
    input_data = {
        "product_description": "Test product",
        "target_platform": "instagram",
        "content_type": "social_post",
        "tone": "inspirational",
        "use_case": "product_launch"
    }
    
    input_obj = CopywritingInput(**input_data)
    
    # Test JSON serialization
    json_str = input_obj.model_dump_json()
    assert isinstance(json_str, str)
    
    # Test deserialization
    parsed_data = json.loads(json_str)
    assert parsed_data["product_description"] == "Test product"
    assert parsed_data["target_platform"] == "instagram"

def test_model_computed_fields():
    """Test computed fields."""
    from models import CopywritingInput
    
    input_data = {
        "product_description": "Test product",
        "target_platform": "instagram",
        "content_type": "social_post",
        "tone": "inspirational",
        "use_case": "product_launch",
        "creativity_level": "creative"
    }
    
    input_obj = CopywritingInput(**input_data)
    
    # Test computed fields
    assert hasattr(input_obj, 'effective_creativity_score')
    assert hasattr(input_obj, 'effective_max_variants')
    
    # Test that computed fields return reasonable values
    assert isinstance(input_obj.effective_creativity_score, float)
    assert 0.0 <= input_obj.effective_creativity_score <= 1.0
    assert isinstance(input_obj.effective_max_variants, int)
    assert input_obj.effective_max_variants > 0
