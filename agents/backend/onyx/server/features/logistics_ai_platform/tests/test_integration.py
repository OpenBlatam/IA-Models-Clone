"""
Integration tests for complete workflows

These tests verify end-to-end workflows across multiple endpoints.
"""

import pytest
from fastapi import status


def test_quote_to_booking_workflow(client, sample_quote_request):
    """Test complete workflow: create quote -> create booking"""
    # Step 1: Create a quote
    quote_response = client.post("/forwarding/quotes", json=sample_quote_request)
    assert quote_response.status_code == status.HTTP_201_CREATED
    quote_data = quote_response.json()
    quote_id = quote_data["quote_id"]
    assert quote_id is not None
    
    # Step 2: Get the quote
    get_quote_response = client.get(f"/forwarding/quotes/{quote_id}")
    assert get_quote_response.status_code == status.HTTP_200_OK
    retrieved_quote = get_quote_response.json()
    assert retrieved_quote["quote_id"] == quote_id
    
    # Step 3: Create a booking from the quote
    if retrieved_quote.get("options") and len(retrieved_quote["options"]) > 0:
        option_id = retrieved_quote["options"][0].get("quote_id", quote_id)
        booking_request = {
            "quote_id": quote_id,
            "selected_option_id": option_id,
            "shipper_info": {
                "name": "Test Shipper",
                "email": "shipper@test.com",
                "phone": "+1234567890"
            },
            "consignee_info": {
                "name": "Test Consignee",
                "email": "consignee@test.com",
                "phone": "+0987654321"
            },
            "payment_terms": "NET 30"
        }
        
        booking_response = client.post("/forwarding/bookings", json=booking_request)
        assert booking_response.status_code == status.HTTP_201_CREATED
        booking_data = booking_response.json()
        assert booking_data["booking_id"] is not None
        assert booking_data["quote_id"] == quote_id


def test_shipment_tracking_workflow(client, sample_quote_request):
    """Test workflow: create shipment -> track shipment"""
    # Step 1: Create a quote
    quote_response = client.post("/forwarding/quotes", json=sample_quote_request)
    assert quote_response.status_code == status.HTTP_201_CREATED
    quote_data = quote_response.json()
    quote_id = quote_data["quote_id"]
    
    # Step 2: Create a shipment
    shipment_request = {
        "origin": sample_quote_request["origin"],
        "destination": sample_quote_request["destination"],
        "cargo": sample_quote_request["cargo"],
        "transportation_mode": sample_quote_request["transportation_mode"],
        "booking_id": None
    }
    
    shipment_response = client.post("/forwarding/shipments", json=shipment_request)
    assert shipment_response.status_code == status.HTTP_201_CREATED
    shipment_data = shipment_response.json()
    shipment_id = shipment_data["shipment_id"]
    
    # Step 3: Track the shipment
    tracking_response = client.get(f"/tracking/shipment/{shipment_id}")
    assert tracking_response.status_code == status.HTTP_200_OK
    tracking_data = tracking_response.json()
    assert tracking_data["shipment_id"] == shipment_id
    assert "status" in tracking_data
    assert "events" in tracking_data


def test_health_check_workflow(client):
    """Test health check endpoints"""
    # Test root endpoint
    root_response = client.get("/")
    assert root_response.status_code == status.HTTP_200_OK
    root_data = root_response.json()
    assert root_data["service"] == "Logistics AI Platform"
    
    # Test health endpoint
    health_response = client.get("/health")
    assert health_response.status_code in [status.HTTP_200_OK, status.HTTP_503_SERVICE_UNAVAILABLE]
    health_data = health_response.json()
    assert "status" in health_data
    assert "services" in health_data
    
    # Test readiness endpoint
    ready_response = client.get("/ready")
    assert ready_response.status_code in [status.HTTP_200_OK, status.HTTP_503_SERVICE_UNAVAILABLE]
    ready_data = ready_response.json()
    assert "status" in ready_data


def test_metrics_endpoints(client):
    """Test metrics endpoints"""
    # Test metrics info
    metrics_info_response = client.get("/metrics/info")
    assert metrics_info_response.status_code == status.HTTP_200_OK
    metrics_info = metrics_info_response.json()
    assert "enabled" in metrics_info
    
    # Test Prometheus metrics
    metrics_response = client.get("/metrics")
    assert metrics_response.status_code == status.HTTP_200_OK
    assert metrics_response.headers.get("content-type") is not None


def test_error_handling_workflow(client):
    """Test error handling across endpoints"""
    # Test 404 for non-existent quote
    not_found_response = client.get("/forwarding/quotes/NONEXISTENT123")
    assert not_found_response.status_code == status.HTTP_404_NOT_FOUND
    error_data = not_found_response.json()
    assert "error" in error_data
    
    # Test validation error
    invalid_request = {
        "origin": {"country": ""},
        "destination": {"country": "Honduras"},
        "cargo": {"description": "Test"}
    }
    validation_response = client.post("/forwarding/quotes", json=invalid_request)
    assert validation_response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    validation_data = validation_response.json()
    assert "error" in validation_data

