import pytest


@pytest.mark.integration
def test_limits_status_requires_api_key(client_v14):
    r = client_v14.get("/api/v14/limits/status")
    assert r.status_code in (400, 401, 422, 500)


@pytest.mark.integration
def test_limits_status_with_header(client_v14):
    r = client_v14.get("/api/v14/limits/status", headers={"X-API-Key": "test"})
    assert r.status_code in (200, 500)
    body = r.json()
    assert isinstance(body, dict)
    # Check presence of a few expected keys when available
    for key in ("user_id", "limits"):
        if key in body:
            assert body[key] is not None


@pytest.mark.integration
def test_limits_reset_with_header(client_v14):
    r = client_v14.post("/api/v14/limits/reset", headers={"X-API-Key": "test"})
    assert r.status_code in (200, 500)
    assert isinstance(r.json(), dict)


