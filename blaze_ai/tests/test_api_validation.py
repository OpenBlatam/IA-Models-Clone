from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from agents.backend.onyx.server.features.blaze_ai.api.router import router as blaze_router


class _StubAI:
    async def health_check(self):
        return {"ok": True}

    def get_unified_stats(self):
        return {"engines": {"registered": 1}}

    async def process(self, payload: dict):  # noqa: ANN001
        # echo payload for testing
        return {"ok": True, "echo": payload}


def _build_app(monkeypatch) -> TestClient:  # type: ignore[no-untyped-def]
    # Patch internal _get_ai reference to avoid heavy model loading
    from agents.backend.onyx.server.features.blaze_ai.api import router as router_module
    monkeypatch.setattr(
        router_module, "_get_ai", lambda: _StubAI(), raising=True
    )
    app = FastAPI()
    app.include_router(blaze_router)
    return TestClient(app)


def test_brand_train_requires_two_samples(monkeypatch):  # type: ignore[no-untyped-def]
    client = _build_app(monkeypatch)
    resp = client.post("/blaze/brand/train", json={"brand_name": "X", "samples": ["only one"]})
    assert resp.status_code == 422


def test_diffusion_requires_multiple_of_8_dimensions(monkeypatch):  # type: ignore[no-untyped-def]
    client = _build_app(monkeypatch)
    # width invalid
    resp = client.post(
        "/blaze/diffusion/generate",
        json={"prompt": "p", "width": 66, "height": 64},
    )
    assert resp.status_code == 422
    # height invalid
    resp2 = client.post(
        "/blaze/diffusion/generate",
        json={"prompt": "p", "width": 64, "height": 65},
    )
    assert resp2.status_code == 422


def test_health_ok(monkeypatch):  # type: ignore[no-untyped-def]
    client = _build_app(monkeypatch)
    resp = client.get("/blaze/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("ok") is True


