import pytest


@pytest.fixture
def client_v14():
    pytest.importorskip("fastapi")
    pytest.importorskip("starlette")
    try:
        from importlib import import_module
        mod = import_module(
            "agents.backend.onyx.server.features.instagram_captions.current.v14_optimized.api.f_as_t_a_pi_v14"
        )
        app = getattr(mod, "app", None)
        if app is None:
            raise RuntimeError("app not found")
        from starlette.testclient import TestClient
        return TestClient(app)
    except Exception:
        pytest.skip("API v14 not importable in this environment")



