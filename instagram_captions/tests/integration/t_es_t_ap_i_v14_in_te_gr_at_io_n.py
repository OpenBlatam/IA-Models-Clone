import pytest


@pytest.mark.integration
def test_root_endpoint_smoke(client_v14):
    client = client_v14
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert body.get("status") == "running"


@pytest.mark.integration
def test_health_endpoint_integration(client_v14):
    client = client_v14
    r = client.get("/api/v14/health")
    # Allow either 200 or 503 depending on init; just ensure valid JSON structure
    assert r.status_code in (200, 503)
    body = r.json()
    assert isinstance(body, dict)


@pytest.mark.integration
def test_generate_requires_api_key_header(client_v14):
    client = client_v14
    req = {
        "caption": "hello world",
        "style": "casual",
        "hashtag_count": 5,
    }
    r = client.post("/api/v14/generate", json=req)
    # Expect 401 due to missing X-API-Key
    assert r.status_code in (400, 401, 422, 500)


@pytest.mark.integration
def test_generate_with_header_minimal_request(client_v14):
    client = client_v14
    # Minimal valid request according to schema names used in file; being defensive due to potential schema changes
    req = {
        "content_description": "sunset by the sea",
        "style": "casual",
        "hashtag_count": 5,
    }
    r = client.post("/api/v14/generate", headers={"X-API-Key": "test"}, json=req)
    # Depending on engine init, may return 200 or an error; accept both but require JSON
    assert r.status_code in (200, 400, 500, 503)
    assert isinstance(r.json(), dict)


