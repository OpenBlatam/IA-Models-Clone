"""
Pytest configuration and fixtures for Logistics AI Platform tests
"""

import pytest
from fastapi.testclient import TestClient
from typing import AsyncGenerator

from main import app
from utils.cache import cache_service
from repositories.quote_repository import QuoteRepository
from repositories.booking_repository import BookingRepository
from repositories.shipment_repository import ShipmentRepository


@pytest.fixture(scope="session")
def client():
    """Create test client"""
    return TestClient(app)


@pytest.fixture(scope="function")
async def cache_cleanup():
    """Clean cache before and after test"""
    await cache_service.clear_pattern("*")
    yield
    await cache_service.clear_pattern("*")


@pytest.fixture
def quote_repository():
    """Create quote repository instance"""
    return QuoteRepository()


@pytest.fixture
def booking_repository():
    """Create booking repository instance"""
    return BookingRepository()


@pytest.fixture
def shipment_repository():
    """Create shipment repository instance"""
    return ShipmentRepository()


@pytest.fixture
def sample_quote_request():
    """Sample quote request data"""
    return {
        "origin": {
            "country": "Mexico",
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
            "weight_kg": 1000,
            "volume_m3": 5.0,
            "quantity": 10,
            "unit_type": "CTN",
            "value_usd": 50000
        },
        "transportation_mode": "maritime",
        "insurance_required": True
    }


@pytest.fixture
def sample_booking_request():
    """Sample booking request data"""
    return {
        "quote_id": "Q12345678",
        "selected_option_id": "option_1",
        "shipper_info": {
            "name": "Shipper Company",
            "email": "shipper@example.com",
            "phone": "+1234567890"
        },
        "consignee_info": {
            "name": "Consignee Company",
            "email": "consignee@example.com",
            "phone": "+0987654321"
        },
        "payment_terms": "NET 30"
    }

