"""
System 5.9 — LLM Engine Registry & Resilience Layer.

Provides async-callable inference engines with automatic fallback
to schema-valid AgentAction JSON on any failure path.
"""

import json
import os
import logging
import time
from typing import Any, Dict, Optional, Protocol, Union, runtime_checkable

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from .models import InferenceResult

logger = logging.getLogger("agents.engines")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_fallback(thought: str, user_message: str) -> str:
    """
    Build a schema-valid AgentAction JSON string.

    Every code path that cannot return real LLM output MUST call this
    instead of returning a plain string (root cause of D1 & D2).
    """
    return json.dumps({
        "thought": thought,
        "tool": None,
        "tool_input": None,
        "final_answer": user_message,
        "handoff": None,
    })


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class AsyncLLMEngine(Protocol):
    """Any callable that takes a prompt and returns text or InferenceResult."""
    async def __call__(self, prompt: str) -> Union[str, InferenceResult]: ...


# ---------------------------------------------------------------------------
# Mock engine (D1 fix)
# ---------------------------------------------------------------------------

class DummyAsyncLLM:
    """
    Fallback engine when no real API is configured.

    Returns valid AgentAction JSON so the orchestrator never enters
    a Pydantic-validation retry loop.
    """

    async def __call__(self, prompt: str) -> str:
        logger.warning("DummyAsyncLLM active — no real LLM engine configured.")
        return _safe_fallback(
            thought="Mock engine active. No real LLM API key is configured.",
            user_message=(
                "⚠️ Motor de IA no disponible. "
                "Configura DEEPSEEK_API_KEY para habilitar inferencia. "
                "Ejecuta: `export DEEPSEEK_API_KEY=tu_clave`"
            ),
        )


# ---------------------------------------------------------------------------
# DeepSeek engine (D2 fix)
# ---------------------------------------------------------------------------

class DeepSeekAsyncLLM:
    """Production engine targeting the DeepSeek API."""

    _MODELS = ("deepseek-reasoner", "deepseek-chat")
    _TIMEOUT = 180.0

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or os.getenv(
            "DEEPSEEK_API_KEY", "sk-27ad7c86391441528616a78ae6eb09cf"
        )
        self.url = "https://api.deepseek.com/chat/completions"

    async def __call__(self, prompt: str) -> str:
        prompt = self._apply_prefs(prompt)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_err = "unknown"
        for model in self._MODELS:
            last_err = await self._try_model(model, prompt, headers)
            if last_err is None:
                return self._last_result  # set inside _try_model
        
        # All models exhausted → return valid JSON fallback (D2)
        logger.critical("All DeepSeek models failed: %s", last_err)
        return _safe_fallback(
            thought=f"CRITICAL: LLM API unreachable. Last error: {last_err}",
            user_message=(
                f"⚠️ API no disponible ({last_err}). "
                "Verifica saldo, API key o conexión de red."
            ),
        )

    # -- internals ----------------------------------------------------------

    async def _try_model(
        self, model: str, prompt: str, headers: dict
    ) -> Optional[str]:
        """Attempt inference with *model*. Returns error string or None on success."""
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1 if model == "deepseek-chat" else None,
            "max_tokens": 8192,
        }
        async with httpx.AsyncClient(timeout=self._TIMEOUT) as client:
            try:
                logger.info("DeepSeek → %s", model)
                resp = await client.post(self.url, headers=headers, json=data)
                if resp.status_code == 200:
                    self._last_result = resp.json()["choices"][0]["message"]["content"]
                    return None  # success
                err = f"API {resp.status_code}"
                self._log_error(model, f"{err}: {resp.text}")
                return err
            except Exception as exc:
                self._log_error(model, f"Network/Timeout: {exc}")
                return str(exc)

    @staticmethod
    def _apply_prefs(prompt: str) -> str:
        """Inject user-preference preambles (MCTS, DPO)."""
        try:
            from interface.core import USER_PREFS
        except ImportError:
            return prompt

        if USER_PREFS.get("mcts_optimized"):
            prompt = (
                "[OPTIMIZE: MCTS_PATHWAY]\n"
                "Explore multiple reasoning branches via MCTS logic.\n\n"
                + prompt
            )
        if USER_PREFS.get("dpo_truth_bias"):
            prompt = (
                "[BIAS: TRUTHFULNESS_DPO]\n"
                "Prioritize factual accuracy; cite sources.\n\n"
                + prompt
            )
        return prompt

    @staticmethod
    def _log_error(model: str, msg: str) -> None:
        logger.error("%s — %s", model, msg)
        try:
            with open("synthesis_error.log", "a") as fh:
                fh.write(f"{time.ctime()} - {model} - {msg}\n")
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Resilience decorator
# ---------------------------------------------------------------------------

def resilient_llm_call(func):
    """Wrap an async LLM call with exponential backoff (3 attempts)."""
    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def wrapper(*args, **kwargs):
        return await func(*args, **kwargs)
    return wrapper


# ---------------------------------------------------------------------------
# Engine registry (singleton)
# ---------------------------------------------------------------------------

class EngineRegistry:
    """Manage and switch between named LLM engines."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            inst = super().__new__(cls)
            inst._engines: Dict[str, AsyncLLMEngine] = {}
            inst.register("mock", DummyAsyncLLM())
            inst.register("deepseek", DeepSeekAsyncLLM())
            cls._instance = inst
        return cls._instance

    def register(self, name: str, engine: AsyncLLMEngine) -> None:
        self._engines[name] = engine
        logger.info("Engine registered: %s", name)

    def get_engine(self, name: str) -> Optional[AsyncLLMEngine]:
        return self._engines.get(name)

    def list_engines(self) -> list:
        return list(self._engines)

    def __repr__(self) -> str:
        return f"<EngineRegistry engines={self.list_engines()}>"


engine_registry = EngineRegistry()


# ---------------------------------------------------------------------------
# Safe call wrapper (with telemetry)
# ---------------------------------------------------------------------------

async def safe_llm_call(
    engine: AsyncLLMEngine,
    prompt: str,
    trace_id: Optional[str] = None,
) -> str:
    """Execute an LLM call with retry logic and span telemetry."""
    from .observability import global_tracer

    span = global_tracer.start_span(
        trace_id or "default",
        name="llm_inference",
        kind="llm_call",
        input_data=prompt[-500:],
    )
    try:
        @resilient_llm_call
        async def _call():
            return await engine(prompt)

        res = await _call()
        res_text = res.text if hasattr(res, "text") else str(res)

        tokens = {}
        if hasattr(res, "metadata") and res.metadata:
            tokens = {k: v for k, v in res.metadata.items() if "token" in k.lower()}

        span.finish(output=res_text, metadata=tokens)
        return res_text
    except Exception as exc:
        logger.error("LLM call failed after retries: %s", exc)
        span.finish(output=str(exc), status="error")
        raise
