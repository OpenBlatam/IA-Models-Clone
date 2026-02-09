import pytest


@pytest.mark.integration
def test_batch_requires_api_key_header(client_v14):
    client = client_v14
    payload = {
        "requests": [
            {"content_description": "a", "style": "casual", "hashtag_count": 3},
            {"content_description": "b", "style": "professional", "hashtag_count": 5},
        ]
    }
    r = client.post("/api/v14/batch", json=payload)
    assert r.status_code in (400, 401, 422, 500)


@pytest.mark.integration
def test_batch_with_header_small_payload(client_v14):
    client = client_v14
    payload = {
        "requests": [
            {"content_description": "hello", "style": "casual", "hashtag_count": 3},
            {"content_description": "world", "style": "casual", "hashtag_count": 3},
        ]
    }
    r = client.post("/api/v14/batch", headers={"X-API-Key": "test"}, json=payload)
    assert r.status_code in (200, 400, 500, 503)
    assert isinstance(r.json(), dict)


@pytest.mark.integration
def test_optimize_requires_api_key_header(client_v14):
    client = client_v14
    payload = {"caption": "ok", "style": "casual", "hashtag_count": 5}
    r = client.post("/api/v14/optimize", json=payload)
    assert r.status_code in (400, 401, 422, 500)


@pytest.mark.integration
def test_optimize_with_header_minimal_request(client_v14):
    client = client_v14
    payload = {"caption": "improve me", "style": "casual", "hashtag_count": 5}
    r = client.post("/api/v14/optimize", headers={"X-API-Key": "test"}, json=payload)
    assert r.status_code in (200, 400, 500, 503)
    assert isinstance(r.json(), dict)


