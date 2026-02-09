"""
Test Helpers
============

Helper functions for testing.
"""

import asyncio
from typing import Any, Callable
from fastapi.testclient import TestClient
from fastapi import FastAPI


class TestHelpers:
    """Test helper functions."""
    
    @staticmethod
    def run_async(coro: Callable) -> Any:
        """Run async function in sync context."""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro)
    
    @staticmethod
    def create_test_client(app: FastAPI) -> TestClient:
        """Create test client for FastAPI app."""
        return TestClient(app)
    
    @staticmethod
    def assert_response(response: Any, status_code: int = 200, **kwargs):
        """Assert response properties."""
        assert response.status_code == status_code
        if kwargs:
            data = response.json()
            for key, value in kwargs.items():
                assert data.get(key) == value















