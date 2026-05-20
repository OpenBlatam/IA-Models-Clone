"""
Tests for quote endpoints
"""

import pytest
from fastapi import status


def test_create_quote(client, sample_quote_request):
    """Test creating a quote"""
    response = client.post("/forwarding/quotes", json=sample_quote_request)
    assert response.status_code == status.HTTP_201_CREATED
    data = response.json()
    assert "quote_id" in data
    assert data["transportation_mode"] == "maritime"
    assert "options" in data


def test_get_quote_not_found(client):
    """Test getting non-existent quote"""
    response = client.get("/forwarding/quotes/NONEXISTENT")
    assert response.status_code == status.HTTP_404_NOT_FOUND


def test_create_quote_invalid_data(client):
    """Test creating quote with invalid data"""
    invalid_request = {
        "origin": {
            "country": "Mexico"
        }
    }
    response = client.post("/forwarding/quotes", json=invalid_request)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_create_quote_validation(client):
    """Test quote validation"""
    invalid_request = {
        "origin": {
            "country": "",
            "city": "Veracruz",
            "port_code": "MXVER"
        },
        "destination": {
            "country": "Honduras",
            "city": "Comayagua",
            "port_code": "HNCMY"
        },
        "cargo": {
            "description": "Electronics",
            "weight_kg": -100,
            "volume_m3": 5.0,
            "quantity": 10,
            "unit_type": "CTN",
            "value_usd": 50000
        },
        "transportation_mode": "invalid_mode"
    }
    response = client.post("/forwarding/quotes", json=invalid_request)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

