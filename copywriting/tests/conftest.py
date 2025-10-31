"""
Test configuration for copywriting service tests.
"""
from typing import Any, List, Dict, Optional
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
import sys
import os

# Add the parent directory to the path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Create a simple test app without complex dependencies
@pytest.fixture(scope="module")
def client():
    """Create test client for copywriting API."""
    app = FastAPI()
    
    # Add a simple test endpoint
    @app.get("/test")
    def test_endpoint():
        return {"status": "ok"}
    
    with TestClient(app) as c:
        yield c

@pytest.fixture
def sample_copywriting_request():
    """Sample copywriting request for testing."""
    return {
        "product_description": "Zapatos deportivos de alta gama",
        "target_platform": "Instagram",
        "tone": "inspirational",
        "target_audience": "Jóvenes activos",
        "key_points": ["Comodidad", "Estilo", "Durabilidad"],
        "instructions": "Enfatiza la innovación",
        "restrictions": ["no mencionar precio"],
        "creativity_level": 0.8,
        "language": "es"
    }

@pytest.fixture
def sample_batch_request():
    """Sample batch copywriting request for testing."""
    return [
        {
            "product_description": "Zapatos deportivos de alta gama",
            "target_platform": "Instagram",
            "tone": "inspirational",
            "target_audience": "Jóvenes activos",
            "key_points": ["Comodidad", "Estilo", "Durabilidad"],
            "instructions": "Enfatiza la innovación",
            "restrictions": ["no mencionar precio"],
            "creativity_level": 0.8,
            "language": "es"
        },
        {
            "product_description": "Laptop gaming profesional",
            "target_platform": "Facebook",
            "tone": "professional",
            "target_audience": "Profesionales tech",
            "key_points": ["Rendimiento", "Calidad", "Innovación"],
            "instructions": "Destaca las especificaciones técnicas",
            "restrictions": ["no mencionar precio"],
            "creativity_level": 0.7,
            "language": "es"
        }
    ]