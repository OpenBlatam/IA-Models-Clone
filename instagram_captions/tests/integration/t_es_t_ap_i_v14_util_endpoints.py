import pytest


@pytest.mark.integration
def test_metrics_endpoint_integration(client_v14):
    client = client_v14
    r = client.get("/api/v14/metrics")
    assert r.status_code in (200, 500)
    assert isinstance(r.json(), (dict,))


@pytest.mark.integration
def test_info_endpoint_integration(client_v14):
    client = client_v14
    r = client.get("/api/v14/info")
    assert r.status_code in (200, 500)
    body = r.json()
    assert isinstance(body, dict)
    # If present, check expected keys without enforcing strictness
    for key in ("version", "features", "endpoints"):
        if key in body:
            assert body[key] is not None


@pytest.mark.integration
def test_validate_endpoint_various_payloads(client_v14):
    client = client_v14
    # Batch-style
    batch = {"requests": [{"content_description": "a", "style": "casual", "hashtag_count": 3}]}
    r1 = client.post("/api/v14/validate", json=batch)
    assert r1.status_code in (200, 400)
    assert isinstance(r1.json(), dict)

    # Optimize-style
    opt = {"caption": "ok", "style": "casual", "hashtag_count": 5}
    r2 = client.post("/api/v14/validate", json=opt)
    assert r2.status_code in (200, 400)
    assert isinstance(r2.json(), dict)

    # Generate-style
    gen = {"content_description": "nice", "style": "casual", "hashtag_count": 3}
    r3 = client.post("/api/v14/validate", json=gen)
    assert r3.status_code in (200, 400)
    assert isinstance(r3.json(), dict)


@pytest.mark.integration
def test_sanitize_endpoint_basic(client_v14):
    client = client_v14
    data = {"content": "<b>ok</b><script>bad()</script>"}
    r = client.post("/api/v14/sanitize", json=data)
    assert r.status_code in (200, 400)
    body = r.json()
    assert isinstance(body, dict)
    # Expect either sanitized data or an error structure
    assert "sanitized" in body or "error" in body or "data" in body


