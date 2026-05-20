from __future__ import annotations

import argparse
import contextlib
import json
import time
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# -----------------------------
# Config
# -----------------------------


@dataclass
class DocsConfig:
    pytorch_version: str = "2.3"
    transformers_version: str = "4.42.0"
    diffusers_version: str = "0.29.0"
    gradio_version: str = "4.36.1"
    cache_dir: str = ".cache/docs"
    cache_ttl_seconds: int = 60 * 60 * 12
    http_timeout_seconds: float = 15.0
    concurrency: int = 8


# -----------------------------
# URLs / References / Tips
# -----------------------------


def build_doc_urls(cfg: DocsConfig) -> Dict[str, str]:
    return {
        "pytorch": f"https://pytorch.org/docs/{cfg.pytorch_version}/",
        "transformers": f"https://huggingface.co/docs/transformers/v{cfg.transformers_version}",
        "diffusers": f"https://huggingface.co/docs/diffusers/v{cfg.diffusers_version}",
        "gradio": f"https://www.gradio.app/docs/{cfg.gradio_version}",
    }


API_REFERENCES: Dict[str, Dict[str, str]] = {
    "pytorch": {
        "nn.Module": "https://pytorch.org/docs/stable/generated/torch.nn.Module.html",
        "autograd": "https://pytorch.org/docs/stable/autograd.html",
        "optim": "https://pytorch.org/docs/stable/optim.html",
        "DataLoader": "https://pytorch.org/docs/stable/data.html",
        "DDP": "https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html",
        "amp": "https://pytorch.org/docs/stable/amp.html",
    },
    "transformers": {
        "AutoModel": "https://huggingface.co/docs/transformers/main/en/model_doc/auto",
        "AutoTokenizer": "https://huggingface.co/docs/transformers/main/en/main_classes/tokenizer",
        "pipeline": "https://huggingface.co/docs/transformers/main/en/main_classes/pipelines",
        "Trainer": "https://huggingface.co/docs/transformers/main/en/main_classes/trainer",
        "PEFT": "https://huggingface.co/docs/peft/index",
    },
    "diffusers": {
        "StableDiffusionPipeline": "https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion",
        "DDIMScheduler": "https://huggingface.co/docs/diffusers/main/en/api/schedulers/ddim",
        "UNet2DConditionModel": "https://huggingface.co/docs/diffusers/main/en/api/models/unet2d",
        "ControlNet": "https://huggingface.co/docs/diffusers/main/en/api/pipelines/controlnet",
    },
    "gradio": {
        "Interface": "https://www.gradio.app/docs/interface",
        "Blocks": "https://www.gradio.app/docs/blocks",
        "Components": "https://www.gradio.app/docs/components",
        "Queue": "https://www.gradio.app/docs/queue",
    },
}


BEST_PRACTICES: Dict[str, List[str]] = {
    "pytorch": [
        "Use torch.compile when available.",
        "Use torch.cuda.amp.autocast for mixed precision.",
        "Prefer DistributedDataParallel for multi-GPU.",
        "Pin memory and non_blocking transfers for DataLoader.",
        "Profile with torch.profiler.",
    ],
    "transformers": [
        "Use from_pretrained(..., device_map='auto') for large models.",
        "Enable gradient checkpointing for memory savings.",
        "Use PEFT (LoRA/Adapters) for efficient fine-tuning.",
        "Batch generation and optionally attention slicing.",
    ],
    "diffusers": [
        "Choose scheduler per task (DDIM/Euler/DPM-Solver).",
        "Enable attention/vae slicing to reduce memory.",
        "Use torch.compile on UNet where supported.",
        "Reuse pipelines/tokenizers across requests.",
    ],
    "gradio": [
        "Use gr.Queue for heavy inference.",
        "Validate inputs and handle errors gracefully.",
        "Stream outputs for better UX when possible.",
        "Keep UI responsive; avoid blocking the event loop.",
    ],
}


# -----------------------------
# Cache
# -----------------------------


class FileCache:
    def __init__(self, root: Path, ttl_seconds: int) -> None:
        self.root = root
        self.ttl = ttl_seconds
        self.root.mkdir(parents=True, exist_ok=True)

    def _p(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        p = self._p(key)
        if not p.exists():
            return None
        if time.time() - p.stat().st_mtime > self.ttl:
            with contextlib.suppress(Exception):
                p.unlink()
            return None
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)

    def set(self, key: str, value: Dict[str, Any]) -> None:
        p = self._p(key)
        with p.open("w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False)


# -----------------------------
# HTTP Client (async capable if httpx present)
# -----------------------------


class HttpClient:
    def __init__(self, timeout: float) -> None:
        self.timeout = timeout
        self._client = None
        self._use_httpx = False
        try:
            import httpx  # type: ignore

            self._use_httpx = True
        except Exception:
            self._use_httpx = False

    async def __aenter__(self):
        if self._use_httpx:
            import httpx  # type: ignore

            self._client = httpx.AsyncClient(http2=True, timeout=self.timeout, follow_redirects=True)
        return self

    async def __aexit__(self, *_):
        if self._client is not None:
            await self._client.aclose()

    async def get_text(self, url: str) -> Tuple[int, str]:
        if self._use_httpx and self._client is not None:
            resp = await self._client.get(url)
            return resp.status_code, resp.text
        import urllib.request

        try:
            with urllib.request.urlopen(url, timeout=self.timeout) as r:  # type: ignore
                code = getattr(r, "status", 200)
                return code, r.read().decode("utf-8", errors="ignore")
        except Exception:
            return 599, ""


# -----------------------------
# Manager
# -----------------------------


class DocsManager:
    def __init__(self, cfg: Optional[DocsConfig] = None) -> None:
        self.cfg = cfg or DocsConfig()
        self.urls = build_doc_urls(self.cfg)
        self.cache = FileCache(Path(self.cfg.cache_dir), self.cfg.cache_ttl_seconds)

    def check_versions(self) -> Dict[str, str]:
        versions: Dict[str, str] = {}
        for pkg in ("torch", "transformers", "diffusers", "gradio"):
            with contextlib.suppress(Exception):
                mod = __import__(pkg)
                versions[pkg] = getattr(mod, "__version__", "unknown")
        return versions

    def requirements(self) -> str:
        return (
            f"torch>={self.cfg.pytorch_version}\n"
            f"transformers>={self.cfg.transformers_version}\n"
            f"diffusers>={self.cfg.diffusers_version}\n"
            f"gradio>={self.cfg.gradio_version}\n"
        )

    def open_in_browser(self, lib: str) -> None:
        url = self.urls.get(lib)
        if url:
            webbrowser.open(url)

    async def fetch_one(self, lib: str) -> Dict[str, Any]:
        url = self.urls[lib]
        key = f"{lib}-index"
        cached = self.cache.get(key)
        if cached:
            return cached

        async with HttpClient(timeout=self.cfg.http_timeout_seconds) as cli:
            code, text = await cli.get_text(url)
            payload = {"library": lib, "url": url, "status": code, "length": len(text)}
            self.cache.set(key, payload)
            return payload

    async def fetch_all(self) -> List[Dict[str, Any]]:
        libs = list(self.urls.keys())
        # Lightweight semaphore for basic throttling
        import asyncio

        sem = asyncio.Semaphore(self.cfg.concurrency)

        async def _task(name: str) -> Dict[str, Any]:
            async with sem:
                return await self.fetch_one(name)

        return await asyncio.gather(*[_task(n) for n in libs])

    def api_ref(self, lib: str, component: str) -> str:
        return API_REFERENCES.get(lib, {}).get(component, "")

    def best_practices(self, lib: str) -> List[str]:
        return BEST_PRACTICES.get(lib, [])

    def validate_snippet(self, lib: str, component: str, snippet: str) -> Dict[str, Any]:
        score = 100
        warnings: List[str] = []
        recs: List[str] = []

        if lib == "pytorch" and component.lower() in {"nn.module", "module"}:
            if "class " in snippet and "forward(" not in snippet:
                score -= 25
                warnings.append("forward() ausente")
            if "super().__init__()" not in snippet:
                score -= 10
                recs.append("Agregar super().__init__() en __init__")

        if lib == "transformers" and "from_pretrained" in snippet and "device_map" not in snippet:
            score -= 10
            recs.append("Usar device_map='auto' para optimizar memoria")

        return {"valid": score >= 80, "score": score, "warnings": warnings, "recommendations": recs}


# -----------------------------
# CLI (argparse)
# -----------------------------


def _print_pairs(pairs: List[Tuple[str, str]]) -> None:
    for k, v in pairs:
        print(f"{k}: {v}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="docs-manager", add_help=True)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("versions")
    sub.add_parser("requirements")

    p_open = sub.add_parser("open")
    p_open.add_argument("lib", choices=["pytorch", "transformers", "diffusers", "gradio"])

    sub.add_parser("fetch")

    p_api = sub.add_parser("api")
    p_api.add_argument("lib", choices=["pytorch", "transformers", "diffusers", "gradio"])
    p_api.add_argument("component")

    p_best = sub.add_parser("best")
    p_best.add_argument("lib", choices=["pytorch", "transformers", "diffusers", "gradio"])

    p_val = sub.add_parser("validate")
    p_val.add_argument("lib", choices=["pytorch", "transformers", "diffusers", "gradio"])
    p_val.add_argument("component")
    p_val.add_argument("file")

    args = parser.parse_args()
    mgr = DocsManager()

    if args.cmd == "versions":
        vers = mgr.check_versions()
        _print_pairs(list(vers.items()))
        return

    if args.cmd == "requirements":
        print(mgr.requirements())
        return

    if args.cmd == "open":
        mgr.open_in_browser(args.lib)
        return

    if args.cmd == "fetch":
        import asyncio

        res = asyncio.run(mgr.fetch_all())
        pairs = [(x["library"], f"{x['status']} ({x['length']} bytes)") for x in res]
        _print_pairs(pairs)
        return

    if args.cmd == "api":
        print(mgr.api_ref(args.lib, args.component))
        return

    if args.cmd == "best":
        for tip in mgr.best_practices(args.lib):
            print(f"- {tip}")
        return

    if args.cmd == "validate":
        snippet = Path(args.file).read_text(encoding="utf-8")
        res = mgr.validate_snippet(args.lib, args.component, snippet)
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return


if __name__ == "__main__":
    main()




