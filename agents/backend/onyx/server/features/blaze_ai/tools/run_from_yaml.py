from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any, Dict, Optional

from .. import create_modular_ai
from ..core import SystemMode
from ..utils.config import load_yaml_config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Modular Blaze AI from a YAML config")
    parser.add_argument("config", help="Path to YAML config")
    parser.add_argument("--llm-prompt", dest="llm_prompt", default=None, help="Optional prompt to test LLM engine")
    parser.add_argument("--diffusion-prompt", dest="diff_prompt", default=None, help="Optional prompt to test Diffusion engine")
    parser.add_argument("--print-stats", action="store_true", help="Print engine and service stats")
    return parser.parse_args()


async def _run(cfg: Dict[str, Any], llm_prompt: Optional[str], diff_prompt: Optional[str], print_stats: bool) -> Dict[str, Any]:
    ai = await create_modular_ai(system_mode=SystemMode.PRODUCTION, custom_configs=cfg)
    health = await ai.health_check()
    out: Dict[str, Any] = {"health": health}
    if print_stats:
        out["stats"] = ai.get_unified_stats()
    if llm_prompt:
        out["llm"] = await ai.process({"_engine": "llm.generate", "prompt": llm_prompt})
    if diff_prompt:
        out["diffusion"] = await ai.process({"_engine": "diffusion.generate", "prompt": diff_prompt})
    return out


def main() -> int:
    args = _parse_args()
    cfg = load_yaml_config(args.config)
    result = asyncio.run(_run(cfg, args.llm_prompt, args.diff_prompt, args.print_stats))
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


