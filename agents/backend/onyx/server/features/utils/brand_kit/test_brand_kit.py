from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
TIMEOUT_SECONDS = 60

import pytest
from datetime import datetime
from typing import Dict, List, Any
from .model import BrandKit
from .color import BrandKitColor
from .typography import BrandKitTypography
from .voice import BrandKitVoice
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Brand Kit Tests - Onyx Integration
Comprehensive test suite for BrandKit with advanced features.
"""

# Test Data
TEST_COLORS = [
    {
        "name": "Primary Blue",
        "hex": "#005A9C",
        "category": "primary",
        "description": "Main brand color",
        "rgb": {"r": 0, "g": 90, "b": 156},
        "hsl": {"h": 207, "s": 1.0, "l": 0.31},
        "opacity": 1.0,
        "is_dark": True,
        "contrast_ratio": 4.5,
        "usage_guidelines": "Use for primary actions and key elements",
        "semantic_meaning": "Trust and professionalism",
        "accessibility_level": "AA",
        "color_blind_safe": True
    },
    {
        "name": "Accent Orange",
        "hex": "#FF6B35",
        "category": "accent",
        "description": "Call-to-action color",
        "rgb": {"r": 255, "g": 107, "b": 53},
        "hsl": {"h": 15, "s": 1.0, "l": 0.6},
        "opacity": 1.0,
        "is_dark": False,
        "contrast_ratio": 3.0,
        "usage_guidelines": "Use for CTAs and important elements",
        "semantic_meaning": "Energy and action",
        "accessibility_level": "AA",
        "color_blind_safe": True
    }
]

TEST_TYPOGRAPHY = {
    "heading": {
        "name": "Montserrat",
        "font_family": "Montserrat",
        "style": "heading",
        "category": "primary",
        "weights": [400, 600, 700],
        "sizes": {
            "h1": 2.5,
            "h2": 2.0,
            "h3": 1.75,
            "h4": 1.5,
            "h5": 1.25,
            "h6": 1.0
        },
        "line_heights": {
            "tight": 1.2,
            "normal": 1.5
        },
        "letter_spacing": {
            "normal": 0,
            "wide": 0.025
        }
    },
    "body": {
        "name": "Open Sans",
        "font_family": "Open Sans",
        "style": "body",
        "category": "primary",
        "weights": [400, 500, 600],
        "sizes": {
            "xs": 0.75,
            "sm": 0.875,
            "base": 1.0,
            "lg": 1.125,
            "xl": 1.25
        },
        "line_heights": {
            "normal": 1.5,
            "relaxed": 1.75
        },
        "letter_spacing": {
            "normal": 0,
            "tight": -0.025
        }
    }
}

TEST_VOICE = {
    "name": "Professional Voice",
    "tone": "professional",
    "style": "conversational",
    "personality_traits": ["professional", "trustworthy", "innovative"],
    "industry_terms": ["technology", "innovation", "solutions"],
    "vocabulary_level": "intermediate",
    "sentence_structure": "mixed",
    "formality_level": "neutral",
    "emotional_tone": ["confident", "positive", "engaging"],
    "cultural_references": ["global", "modern"],
    "description": "Professional yet approachable voice for tech company",
    "usage_guidelines": "Maintain professional tone while being engaging",
    "examples": [
        {
            "context": "Product description",
            "text": "Our innovative solution streamlines your workflow"
        }
    ]
}

TEST_VALUES = [
    "Innovation",
    "Excellence",
    "Customer Focus",
    "Integrity",
    "Collaboration"
]

TEST_AUDIENCE = {
    "primary": {
        "age_range": "25-45",
        "occupation": "Tech professionals",
        "interests": ["technology", "innovation", "professional development"],
        "pain_points": ["time management", "workflow efficiency"],
        "goals": ["career growth", "professional success"]
    },
    "secondary": {
        "age_range": "18-24",
        "occupation": "Students",
        "interests": ["learning", "technology", "future career"],
        "pain_points": ["skill development", "career guidance"],
        "goals": ["skill acquisition", "career preparation"]
    }
}

@pytest.fixture
def brand_kit_data() -> Dict[str, Any]:
    """Fixture for brand kit test data"""
    return {
        "id": "test-brand-kit-1",
        "name": "Test Brand Kit",
        "description": "Test brand kit for unit testing",
        "colors": TEST_COLORS,
        "typography": TEST_TYPOGRAPHY,
        "voice": TEST_VOICE,
        "values": TEST_VALUES,
        "target_audience": TEST_AUDIENCE,
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat()
    }

@pytest.fixture
def brand_kit(brand_kit_data: Dict[str, Any]) -> BrandKit:
    """Fixture for BrandKit instance"""
    return BrandKit.from_data(brand_kit_data)

def test_brand_kit_initialization(brand_kit: BrandKit):
    """Test BrandKit initialization"""
    assert brand_kit.id == "test-brand-kit-1"
    assert brand_kit.name == "Test Brand Kit"
    assert brand_kit.description == "Test brand kit for unit testing"
    assert len(brand_kit.colors) == 2
    assert len(brand_kit.typography) == 2
    assert brand_kit.voice is not None
    assert len(brand_kit.values) == 5
    assert brand_kit.target_audience is not None

def test_brand_kit_colors(brand_kit: BrandKit):
    """Test BrandKit colors parsing and validation"""
    colors = brand_kit.colors
    assert len(colors) == 2
    
    primary_color = colors[0]
    assert primary_color["name"] == "Primary Blue"
    assert primary_color["hex"] == "#005A9C"
    assert primary_color["category"] == "primary"
    assert primary_color["rgb"] == {"r": 0, "g": 90, "b": 156}
    assert primary_color["hsl"] == {"h": 207, "s": 1.0, "l": 0.31}
    assert primary_color["is_dark"] is True
    assert primary_color["accessibility_level"] == "AA"
    
    accent_color = colors[1]
    assert accent_color["name"] == "Accent Orange"
    assert accent_color["hex"] == "#FF6B35"
    assert accent_color["category"] == "accent"
    assert accent_color["is_dark"] is False

def test_brand_kit_typography(brand_kit: BrandKit):
    """Test BrandKit typography parsing and validation"""
    typography = brand_kit.typography
    assert len(typography) == 2
    
    heading = typography["heading"]
    assert heading["name"] == "Montserrat"
    assert heading["font_family"] == "Montserrat"
    assert heading["style"] == "heading"
    assert heading["weights"] == [400, 600, 700]
    assert "h1" in heading["sizes"]
    assert "tight" in heading["line_heights"]
    
    body = typography["body"]
    assert body["name"] == "Open Sans"
    assert body["font_family"] == "Open Sans"
    assert body["style"] == "body"
    assert body["weights"] == [400, 500, 600]
    assert "base" in body["sizes"]
    assert "normal" in body["line_heights"]

def test_brand_kit_voice(brand_kit: BrandKit):
    """Test BrandKit voice parsing and validation"""
    voice = brand_kit.voice
    assert voice is not None
    assert voice["name"] == "Professional Voice"
    assert voice["tone"] == "professional"
    assert voice["style"] == "conversational"
    assert len(voice["personality_traits"]) == 3
    assert "professional" in voice["personality_traits"]
    assert voice["vocabulary_level"] == "intermediate"
    assert voice["formality_level"] == "neutral"
    assert len(voice["emotional_tone"]) == 3
    assert "confident" in voice["emotional_tone"]

def test_brand_kit_values(brand_kit: BrandKit):
    """Test BrandKit values parsing and validation"""
    values = brand_kit.values
    assert len(values) == 5
    assert "Innovation" in values
    assert "Excellence" in values
    assert "Customer Focus" in values
    assert "Integrity" in values
    assert "Collaboration" in values

def test_brand_kit_audience(brand_kit: BrandKit):
    """Test BrandKit audience parsing and validation"""
    audience = brand_kit.target_audience
    assert audience is not None
    assert "primary" in audience
    assert "secondary" in audience
    
    primary = audience["primary"]
    assert primary["age_range"] == "25-45"
    assert primary["occupation"] == "Tech professionals"
    assert len(primary["interests"]) == 3
    assert "technology" in primary["interests"]
    
    secondary = audience["secondary"]
    assert secondary["age_range"] == "18-24"
    assert secondary["occupation"] == "Students"
    assert len(secondary["interests"]) == 3
    assert "learning" in secondary["interests"]

def test_brand_kit_get_data(brand_kit: BrandKit):
    """Test BrandKit get_data method"""
    data = brand_kit.get_data()
    assert data["id"] == "test-brand-kit-1"
    assert data["name"] == "Test Brand Kit"
    assert len(data["colors"]) == 2
    assert len(data["typography"]) == 2
    assert data["voice"] is not None
    assert len(data["values"]) == 5
    assert data["target_audience"] is not None

@pytest.mark.asyncio
async def test_brand_kit_aget_data(brand_kit: BrandKit):
    """Test BrandKit aget_data method"""
    data = await brand_kit.aget_data()
    assert data["id"] == "test-brand-kit-1"
    assert data["name"] == "Test Brand Kit"
    assert len(data["colors"]) == 2
    assert len(data["typography"]) == 2
    assert data["voice"] is not None
    assert len(data["values"]) == 5
    assert data["target_audience"] is not None

def test_brand_kit_validation():
    """Test BrandKit validation"""
    # Test with invalid data
    invalid_data = {
        "id": "test-brand-kit-2",
        "name": "Invalid Brand Kit",
        "colors": [{"invalid": "color"}],
        "typography": {"invalid": "typography"},
        "voice": {"invalid": "voice"},
        "values": [123],  # Invalid type
        "target_audience": "invalid"  # Invalid type
    }
    
    brand_kit = BrandKit.from_data(invalid_data)
    assert len(brand_kit.colors) == 0  # Invalid colors should be filtered out
    assert len(brand_kit.typography) == 0  # Invalid typography should be filtered out
    assert brand_kit.voice is None  # Invalid voice should be None
    assert len(brand_kit.values) == 0  # Invalid values should be filtered out
    assert brand_kit.target_audience == {}  # Invalid audience should be empty dict

def test_brand_kit_component_filling():
    """Test BrandKit component filling"""
    # Create components
    color = BrandKitColor(
        name="Test Color",
        hex="#000000",
        category="primary"
    )
    
    typography = BrandKitTypography(
        name="Test Font",
        font_family="Test Font Family",
        style="body"
    )
    
    voice = BrandKitVoice(
        name="Test Voice",
        tone="professional",
        style="conversational"
    )
    
    # Create brand kit with components
    brand_kit = BrandKit(
        id="test-brand-kit-3",
        name="Component Test Brand Kit",
        colors=[color],
        typography={"body": typography},
        voice=voice,
        values=["Test Value"],
        target_audience={"test": {"age": "25-45"}}
    )
    
    # Test component filling
    assert len(brand_kit.colors) == 1
    assert brand_kit.colors[0]["name"] == "Test Color"
    assert brand_kit.colors[0]["hex"] == "#000000"
    
    assert len(brand_kit.typography) == 1
    assert "body" in brand_kit.typography
    assert brand_kit.typography["body"]["name"] == "Test Font"
    
    assert brand_kit.voice is not None
    assert brand_kit.voice["name"] == "Test Voice"
    assert brand_kit.voice["tone"] == "professional"
    
    assert len(brand_kit.values) == 1
    assert brand_kit.values[0] == "Test Value"
    
    assert brand_kit.target_audience is not None
    assert "test" in brand_kit.target_audience
    assert brand_kit.target_audience["test"]["age"] == "25-45" 