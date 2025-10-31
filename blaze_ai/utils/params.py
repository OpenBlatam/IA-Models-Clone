from __future__ import annotations

from typing import Any, Dict, Optional


def _as_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return default


def _as_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return default


def _clamp_int(x: Optional[int], lo: int, hi: int) -> Optional[int]:
    if x is None:
        return None
    return max(lo, min(hi, x))


def _clamp_float(x: Optional[float], lo: float, hi: float) -> Optional[float]:
    if x is None:
        return None
    return max(lo, min(hi, x))


def _multiple_of(x: Optional[int], m: int) -> Optional[int]:
    if x is None:
        return None
    return (x // m) * m


def sanitize_llm_overrides(overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Normalize & clamp LLM generation parameters.
    Keys supported: max_new_tokens, temperature, top_p, top_k, repetition_penalty
    """
    src = overrides or {}
    max_new_tokens = _as_int(src.get("max_new_tokens"), 128)
    temperature = _as_float(src.get("temperature"), 0.8)
    top_p = _as_float(src.get("top_p"), 0.95)
    top_k = _as_int(src.get("top_k"), 0)
    repetition_penalty = _as_float(src.get("repetition_penalty"), 1.0)

    max_new_tokens = _clamp_int(max_new_tokens, 1, 2048)
    temperature = _clamp_float(temperature, 0.0, 2.0)
    top_p = _clamp_float(top_p, 0.0, 1.0)
    top_k = _clamp_int(top_k, 0, 1000)
    repetition_penalty = _clamp_float(repetition_penalty, 0.5, 2.0)

    out: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
    }
    return {k: v for k, v in out.items() if v is not None}


def sanitize_diffusion_overrides(overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Normalize & clamp Diffusion parameters.
    Keys supported: width, height, guidance_scale, num_inference_steps, scheduler, pipeline, eta
    - width/height multiples of 8 (>=64, <=2048)
    - steps 1..100
    - guidance 0..20
    """
    src = overrides or {}
    width = _as_int(src.get("width"), 512)
    height = _as_int(src.get("height"), 512)
    guidance_scale = _as_float(src.get("guidance_scale"), 7.5)
    num_inference_steps = _as_int(src.get("num_inference_steps"), 30)
    scheduler = str(src.get("scheduler") or "dpmpp_2m").lower()
    pipeline = str(src.get("pipeline") or "auto").lower()
    eta = _as_float(src.get("eta"), None)

    width = _multiple_of(_clamp_int(width, 64, 2048), 8)
    height = _multiple_of(_clamp_int(height, 64, 2048), 8)
    guidance_scale = _clamp_float(guidance_scale, 0.0, 20.0)
    num_inference_steps = _clamp_int(num_inference_steps, 1, 100)

    allowed_schedulers = {"dpmpp_2m", "euler_a", "euler", "ddim", "pndm", "lms"}
    allowed_pipelines = {"auto", "sd15", "sdxl"}
    if scheduler not in allowed_schedulers:
        scheduler = "dpmpp_2m"
    if pipeline not in allowed_pipelines:
        pipeline = "auto"

    out: Dict[str, Any] = {
        "width": width,
        "height": height,
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps,
        "scheduler": scheduler,
        "pipeline": pipeline,
    }
    if eta is not None:
        out["eta"] = _clamp_float(eta, 0.0, 1.0)
    return {k: v for k, v in out.items() if v is not None}


