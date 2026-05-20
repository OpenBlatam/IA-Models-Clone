from __future__ import annotations

import json
from pathlib import Path

from blaze_ai.tools.run_from_yaml import main as cli_main


def test_config_loader_cli_smoke(tmp_path: Path, monkeypatch) -> None:
    cfg = {
        "engines": {"timeout_seconds": 1},
        "llm": {"model_name": "gpt2"},
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("""
engines:
  timeout_seconds: 1
llm:
  model_name: gpt2
""", encoding="utf-8")

    # Prevent real model loading by monkeypatching create_modular_ai
    from blaze_ai import __init__ as blaze_init

    async def fake_create_modular_ai(*args, **kwargs):  # type: ignore[no-redef]
        class FakeAI:
            async def health_check(self):
                return {"initialized": True, "engine_count": 1, "status": "healthy"}

            def get_unified_stats(self):
                return {"engines": {"registered": 1}, "services": {}}

            async def process(self, payload):  # type: ignore[no-redef]
                return {"ok": True, "engine": payload.get("_engine", "default")}

        return FakeAI()

    monkeypatch.setattr(blaze_init, "create_modular_ai", fake_create_modular_ai)

    # Run CLI with the temp config
    def fake_parse_args():
        class A:
            config = str(cfg_path)
            llm_prompt = None
            diff_prompt = None
            print_stats = True
        return A()

    import blaze_ai.tools.run_from_yaml as cli
    monkeypatch.setattr(cli, "_parse_args", fake_parse_args)

    rc = cli_main()
    assert rc == 0


