from fastapi import FastAPI
from fastapi.testclient import TestClient

from agents.backend.onyx.server.features.ads.api import router as router_alias, main_router


def test_router_alias_points_to_main_router() -> None:
    # Identity check
    assert router_alias is main_router


def test_router_alias_works_in_app() -> None:
    app = FastAPI()
    app.include_router(router_alias)
    client = TestClient(app)
    r = client.get("/ads/health")
    assert r.status_code == 200







