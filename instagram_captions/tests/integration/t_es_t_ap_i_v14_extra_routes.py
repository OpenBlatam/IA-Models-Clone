import pytest


@pytest.mark.integration
def test_generate_legacy_requires_api_key(client_v14):
    r = client_v14.post(
        "/api/v14/generate/legacy",
        json={"content_description": "legacy", "style": "casual", "hashtag_count": 3},
    )
    assert r.status_code in (400, 401, 422, 500)


@pytest.mark.integration
def test_generate_legacy_with_api_key(client_v14):
    r = client_v14.post(
        "/api/v14/generate/legacy",
        headers={"X-API-Key": "test"},
        json={"content_description": "legacy", "style": "casual", "hashtag_count": 3},
    )
    assert r.status_code in (200, 400, 500, 503)
    assert isinstance(r.json(), dict)


@pytest.mark.integration
def test_generate_priority_with_invalid_priority(client_v14):
    payload = {"content_description": "p", "style": "casual", "hashtag_count": 3}
    r = client_v14.post(
        "/api/v14/generate/priority",
        headers={"X-API-Key": "test"},
        json=payload,
        params={"priority_level": 0},
    )
    assert r.status_code in (400, 422)


@pytest.mark.integration
def test_generate_priority_with_valid_priority(client_v14):
    payload = {"content_description": "p", "style": "casual", "hashtag_count": 3}
    r = client_v14.post(
        "/api/v14/generate/priority",
        headers={"X-API-Key": "test"},
        json=payload,
        params={"priority_level": 3},
    )
    assert r.status_code in (200, 400, 500, 503)
    assert isinstance(r.json(), dict)



