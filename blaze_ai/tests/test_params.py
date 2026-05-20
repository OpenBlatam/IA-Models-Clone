from __future__ import annotations

from agents.backend.onyx.server.features.blaze_ai.utils.params import (
    sanitize_llm_overrides,
    sanitize_diffusion_overrides,
)


def test_sanitize_llm_overrides_clamps_and_defaults():
    out = sanitize_llm_overrides({
        "max_new_tokens": -5,
        "temperature": 9.9,
        "top_p": -1,
        "top_k": 99999,
        "repetition_penalty": 99,
    })
    assert 1 <= out["max_new_tokens"] <= 2048
    assert 0.0 <= out["temperature"] <= 2.0
    assert 0.0 <= out["top_p"] <= 1.0
    assert 0 <= out["top_k"] <= 1000
    assert 0.5 <= out["repetition_penalty"] <= 2.0


def test_sanitize_diffusion_overrides_clamps_and_multiples():
    out = sanitize_diffusion_overrides({
        "width": 9999,  # too large
        "height": 63,   # below and not multiple of 8
        "guidance_scale": 99,
        "num_inference_steps": 0,
        "scheduler": "unknown",
        "pipeline": "weird",
        "eta": 2.0,
    })
    assert 64 <= out["width"] <= 2048 and out["width"] % 8 == 0
    assert 64 <= out["height"] <= 2048 and out["height"] % 8 == 0
    assert 0.0 <= out["guidance_scale"] <= 20.0
    assert 1 <= out["num_inference_steps"] <= 100
    assert out["scheduler"] in {"dpmpp_2m", "euler_a", "euler", "ddim", "pndm", "lms"}
    assert out["pipeline"] in {"auto", "sd15", "sdxl"}
    assert 0.0 <= out["eta"] <= 1.0


