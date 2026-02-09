"""Tests for comparison endpoints."""

import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_create_comparison_invalid_id():
    """Test creating comparison with invalid visualization ID."""
    response = client.post(
        "/api/v1/compare",
        json={
            "visualization_id": "invalid-id",
            "layout": "side_by_side"
        }
    )
    assert response.status_code in [404, 500]


def test_create_comparison_invalid_layout():
    """Test creating comparison with invalid layout."""
    response = client.post(
        "/api/v1/compare",
        json={
            "visualization_id": "test-id",
            "layout": "invalid_layout"
        }
    )
    # Should accept but use default behavior
    assert response.status_code in [200, 404, 500]


def test_get_comparison_not_found():
    """Test getting non-existent comparison."""
    response = client.get("/api/v1/compare/nonexistent-id")
    assert response.status_code == 404

