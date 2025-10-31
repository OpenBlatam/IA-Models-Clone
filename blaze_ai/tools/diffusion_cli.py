from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

from ..models.diffusion import StableDiffusionGenerator as Generator, DiffusionConfig


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stable Diffusion CLI (SD 1.x / SDXL)")
    p.add_argument("prompt", help="Text prompt")
    p.add_argument("--model-id", default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--pipeline", choices=["auto", "sd15", "sdxl"], default="auto")
    p.add_argument("--scheduler", default="dpmpp_2m")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--guidance", type=float, default=7.5)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--negative", default=None)
    p.add_argument("--out", default="output.png")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    cfg = DiffusionConfig(model_id=args.model_id)
    cfg.pipeline = args.pipeline
    generator = Generator(cfg)
    overrides: Dict[str, Any] = {
        "scheduler": args.scheduler,
        "num_inference_steps": int(args.steps),
        "guidance_scale": float(args.guidance),
        "width": int(args.width),
        "height": int(args.height),
        "negative_prompt": args.negative,
        "pipeline": args.pipeline,
    }
    result = generator.generate(args.prompt, **overrides)
    img = result.pop("image")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="PNG")
    print(json.dumps({"ok": True, "saved": str(out_path), **result}, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


