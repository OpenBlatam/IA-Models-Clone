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
        # emulate engine-provided fields
        return {
            "ok": True,
            "engine": payload.get("_engine"),
            "elapsed_ms": 1.23,
            "text": "hello",
        }


def _build_app(monkeypatch) -> TestClient:  # type: ignore[no-untyped-def]
    from agents.backend.onyx.server.features.blaze_ai.api import router as router_module

    monkeypatch.setattr(router_module, "_get_ai", lambda: _StubAI(), raising=True)
    app = FastAPI()
    app.include_router(blaze_router)
    return TestClient(app)


def test_llm_generate_response_model(monkeypatch):  # type: ignore[no-untyped-def]
    client = _build_app(monkeypatch)
    resp = client.post("/blaze/llm/generate", json={"prompt": "p", "overrides": {"max_new_tokens": 9999}})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["engine"] == "llm.generate"
    assert isinstance(body["elapsed_ms"], (int, float))
    assert body["text"] == "hello"


