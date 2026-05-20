import os
import sys
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from key_messages.api import router as key_messages_router, CacheMiddleware
from key_messages.services import KeyMessageService

@pytest.fixture
def app():
    """Create a FastAPI application for testing."""
    app = FastAPI()
    
    # Add middleware
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"]
    )
    
    app.add_middleware(CacheMiddleware)
    
    # Include router
    app.include_router(key_messages_router)
    
    return app

@pytest.fixture
def client(app):
    """Create a test client for the FastAPI application."""
    return TestClient(app)

import pytest

@pytest.fixture(autouse=True)
async def setup_teardown():
    """Setup and teardown for each test."""
    # Setup
    service = KeyMessageService()
    await service.clear_cache()
    
    yield
    
    # Teardown
    await service.clear_cache() 