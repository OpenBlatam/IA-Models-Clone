from __future__ import annotations

import pytest

from agents.backend.onyx.server.features.blaze_ai.engines.diffusion import DiffusionEngines


class _StubDiffGen:
    def __init__(self, *_: object, **__: object) -> None:
        pass

    def generate(self, prompt: str, **overrides):  # type: ignore[no-untyped-def]
        # Return a PIL-like obj signature; engine converts to base64 and returns
        class _Img:
            def save(self, *_: object, **__: object) -> None:
                pass
        return {"image": _Img(), "width": overrides.get("width", 512), "height": overrides.get("height", 512)}


@pytest.mark.asyncio
async def test_diffusion_engine_caching(monkeypatch):  # type: ignore[no-untyped-def]
    from agents.backend.onyx.server.features.blaze_ai.engines import diffusion as diff_mod

    monkeypatch.setattr(diff_mod, "StableDiffusionGenerator", _StubDiffGen, raising=True)

    eng = DiffusionEngines(model_id="stub-model")
    payload = {"prompt": "landscape", "overrides": {"width": 512, "height": 512}}

    r1 = await eng.engine_diffusion_generate(payload)
    assert r1.get("image_base64")
    assert r1.get("cached") is None

    r2 = await eng.engine_diffusion_generate(payload)
    assert r2.get("image_base64")
    assert r2.get("cached") is True


