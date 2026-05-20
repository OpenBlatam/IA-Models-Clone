from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from agents.backend.onyx.server.features.blaze_ai.api.router import router as blaze_router


class _StubAI:
    async def health_check(self):
        return {"ok": True}

    def get_unified_stats(self):
        return {"engines": {"registered": 1}, "services": {}}

    async def process(self, payload: dict):  # noqa: ANN001
        return {"ok": True, "engine": payload.get("_engine"), "elapsed_ms": 1.0, "echo": payload}


def _client(monkeypatch) -> TestClient:  # type: ignore[no-untyped-def]
    from agents.backend.onyx.server.features.blaze_ai.api import router as router_module

    monkeypatch.setattr(router_module, "_get_ai", lambda: _StubAI(), raising=True)
    app = FastAPI()
    app.include_router(blaze_router)
    return TestClient(app)


def test_email_create(monkeypatch):  # type: ignore[no-untyped-def]
    c = _client(monkeypatch)
    resp = c.post(
        "/blaze/email/create",
        json={"subject": "Hi", "points": ["a", "b"], "brand_name": "X"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True and body["engine"] == "email.create"


def test_blog_outline(monkeypatch):  # type: ignore[no-untyped-def]
    c = _client(monkeypatch)
    resp = c.post(
        "/blaze/blog/outline",
        json={"title": "T", "sections": ["s1", "s2"], "brand_name": "B"},
    )
    assert resp.status_code == 200
    assert resp.json()["engine"] == "blog.outline"


def test_seo_meta(monkeypatch):  # type: ignore[no-untyped-def]
    c = _client(monkeypatch)
    resp = c.post(
        "/blaze/seo/meta",
        json={"title": "t", "summary": "s"},
    )
    assert resp.status_code == 200
    assert resp.json()["engine"] == "seo.meta"


def test_diffusion_ok_dims(monkeypatch):  # type: ignore[no-untyped-def]
    c = _client(monkeypatch)
    resp = c.post(
        "/blaze/diffusion/generate",
        json={"prompt": "p", "width": 64, "height": 64},
    )
    assert resp.status_code == 200
    assert resp.json()["engine"] == "diffusion.generate"


