import asyncio
import json
import logging
import os
import ssl
import time
import traceback
import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Coroutine, Union



# --- Nuevas importaciones modulares ---
from agents.utils.semantic_cache import get_cached_response, set_cached_response
from agents.utils.config_utils import _get_user_prefs, _normalize_engine_key, _resolve_api_key, _load_api_keys_from_prefs
from agents.providers.base import AsyncLLMEngine, BaseProvider, DummyAsyncLLM
from agents.providers.deepseek import DeepSeekProvider
from agents.providers.google import GoogleGeminiProvider
from agents.providers.openai import OpenAIProvider
from agents.providers.anthropic import AnthropicProvider
from agents.providers.openrouter import OpenRouterProvider

logger = logging.getLogger(__name__)

# Fallback strategies: ordered list of engine keys to try
fallback_order = [
    "deepseek",
    "openai",
    "anthropic",
    "google",
    "openrouter"
]

CC_AVAILABLE = False
try:
    from interface import cc_style
    CC_AVAILABLE = True
except ImportError:
    pass

_benchmark_run_stats = {}
try:
    from agents.ensemble import run_ensemble, ALL_ENSEMBLE_MODES, MULTI_ENGINE_MODES as _MULTI_ENSEMBLE_MODES
except ImportError:
    from .ensemble import run_ensemble, ALL_ENSEMBLE_MODES, MULTI_ENGINE_MODES as _MULTI_ENSEMBLE_MODES

OVERDRIVE_TRIGGERED = False


def _record_benchmark_run(engine_key: str, model_name: str, elapsed: float, tokens: int) -> None:
    _benchmark_run_stats[_normalize_engine_key(engine_key)] = {
        "model": model_name,
        "elapsed": elapsed,
        "tokens": tokens,
        "ts": time.time(),
    }

class EngineRegistry:
    """Refactored Singleton Registry for LLM Providers."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            inst = super().__new__(cls)
            inst._providers: Dict[str, BaseProvider] = {}
            # Defaults are registered as classes for lazy instantiation
            inst._default_providers = {
                "deepseek": DeepSeekProvider,
                "google": GoogleGeminiProvider,
                "chatgpt": OpenAIProvider,
                "openai": OpenAIProvider,
                "claude": AnthropicProvider,
                "anthropic": AnthropicProvider,
                "openrouter": OpenRouterProvider
            }
            cls._instance = inst
        return cls._instance

    def register(self, name: str, provider: Union[BaseProvider, Type[BaseProvider]]):
        self._providers[name] = provider
        if not inspect.isclass(provider):
            logger.info(f"LLM Provider registered: {name}")

    def _refresh_stale_providers(self):
        """Re-instantiate any cached providers that lack an API key (they may have been created before keys were loaded)."""
        for name, provider in list(self._providers.items()):
            if isinstance(provider, BaseProvider) and not provider.api_key:
                if name in self._default_providers:
                    fresh = self._default_providers[name]()
                    if fresh.api_key:
                        self._providers[name] = fresh
                        logger.info(f"Refreshed stale provider: {name}")

    def list_engines(self) -> List[str]:
        """Returns list of registered and default engine names."""
        names = set(self._default_providers.keys()) | set(self._providers.keys())
        return sorted(list(names))

    def _resolve_provider(self, name: Optional[str]) -> tuple[Optional[BaseProvider], str]:
        """Resolve a provider instance and canonical engine key."""
        _load_api_keys_from_prefs()
        self._refresh_stale_providers()

        fallback_order = ["deepseek", "claude", "anthropic", "openai", "chatgpt", "google", "openrouter"]
        resolved_name = name or ""
        if "," in resolved_name:
            resolved_name = resolved_name.split(",")[0].strip()
        base_provider = resolved_name.split(":")[0] if ":" in resolved_name else resolved_name
        base_provider = _normalize_engine_key(base_provider)

        if not resolved_name or base_provider not in self._default_providers:
            for f_name in fallback_order:
                norm = _normalize_engine_key(f_name)
                if norm in self._default_providers:
                    p_inst = self._default_providers[norm]()
                    if p_inst.api_key:
                        resolved_name = norm
                        break
            if not resolved_name:
                resolved_name = "claude"

        resolved_name = _normalize_engine_key(resolved_name.split(":")[0] if ":" in resolved_name else resolved_name)

        if resolved_name in self._default_providers and resolved_name not in self._providers:
            provider_cls = self._default_providers[resolved_name]
            self.register(resolved_name, provider_cls())

        provider = self._providers.get(resolved_name)
        if not provider and name and ":" in name:
            p_name, _, m_name = name.partition(":")
            p_name = _normalize_engine_key(p_name)
            if p_name == "google":
                self.register(name, GoogleGeminiProvider(model=m_name))
            elif p_name == "deepseek":
                self.register(name, DeepSeekProvider(model=m_name))
            elif p_name in ["chatgpt", "openai"]:
                self.register(name, OpenAIProvider(model=m_name))
            elif p_name in ["claude", "anthropic"]:
                self.register(name, AnthropicProvider(model=m_name))
            elif p_name == "openrouter":
                self.register(name, OpenRouterProvider(model=m_name))
            provider = self._providers.get(name)

        if provider and not provider.api_key:
            for f_name in fallback_order:
                norm = _normalize_engine_key(f_name)
                if norm == resolved_name:
                    continue
                if norm in self._default_providers and norm not in self._providers:
                    self._providers[norm] = self._default_providers[norm]()
                f_provider = self._providers.get(norm)
                if f_provider and f_provider.api_key:
                    logger.warning(
                        f"Preferred engine '{resolved_name}' lacks API key. Falling back to '{norm}'"
                    )
                    provider = f_provider
                    resolved_name = norm
                    break

        if not provider:
            for f_name in fallback_order:
                norm = _normalize_engine_key(f_name)
                if norm in self._default_providers and norm not in self._providers:
                    self._providers[norm] = self._default_providers[norm]()
                f_provider = self._providers.get(norm)
                if f_provider and f_provider.api_key:
                    provider = f_provider
                    resolved_name = norm
                    break

        return provider, resolved_name

    def get_active_engines(self) -> List[Dict[str, str]]:
        """Engines listed in preferred_engine that currently have an API key."""
        prefs = _get_user_prefs()
        preferred_raw = prefs.get("preferred_engine", "deepseek")
        keys = [_normalize_engine_key(x) for x in preferred_raw.split(",") if x.strip()]
        if not keys:
            keys = ["deepseek"]

        active: List[Dict[str, str]] = []
        seen: set[str] = set()
        for key in keys:
            if key in seen:
                continue
            seen.add(key)
            provider, resolved = self._resolve_provider(key)
            if provider and provider.api_key:
                active.append({
                    "key": resolved,
                    "label": key,
                    "model": provider.model,
                })
        if not active:
            provider, resolved = self._resolve_provider(None)
            if provider and provider.api_key:
                active.append({
                    "key": resolved,
                    "label": resolved,
                    "model": provider.model,
                })
        return active

    def _get_single_engine_callable(self, name: Optional[str]) -> Optional[AsyncLLMEngine]:
        """Single-provider callable (no ensemble wrapper)."""
        provider, resolved_name = self._resolve_provider(name)
        if not provider:
            return None

        async def _call(prompt: str, **kwargs) -> str:
            merged_kwargs = getattr(_call, "default_kwargs", {}).copy()
            merged_kwargs.update(kwargs)
            return await provider.generate(prompt, **merged_kwargs)

        _call.model_name = provider.model
        _call.provider_name = resolved_name
        _call.is_ensemble = False
        _call.default_kwargs = {}
        return _call

    def _build_ensemble_engine(
        self, active: List[Dict[str, str]], mode: str
    ) -> AsyncLLMEngine:
        """Run all active engines and merge outputs (consensus, parallel, etc.)."""
        registry = self
        mode = (mode or "consensus").lower().strip()

        async def _run_engine_key(
            key: str, eng: Dict[str, str], prompt: str, **kw
        ) -> tuple:
            sub = registry._get_single_engine_callable(key)
            if not sub:
                return key, eng["model"], "", 0.0, 0
            t0 = time.time()
            try:
                text = await sub(prompt, **kw)
            except Exception as exc:
                logger.error(f"Ensemble engine '{key}' failed: {exc}")
                text = json.dumps({
                    "thought": f"[{key}] inference error",
                    "final_answer": f"Error ({key}): {type(exc).__name__}: {str(exc)[:200]}",
                })
            elapsed = time.time() - t0
            tokens = max(1, len(str(text)) // 4)
            model = getattr(sub, "model_name", None) or eng["model"]
            return key, model, text, elapsed, tokens

        async def _ensemble_call(prompt: str, **kwargs) -> str:
            def _record(key: str, model: str, elapsed: float, tokens: int) -> None:
                _record_benchmark_run(key, model, elapsed, tokens)

            return await run_ensemble(
                mode,
                active,
                prompt,
                _run_engine_key,
                record_run=_record if mode != "race" else _record,
                **kwargs,
            )

        _ensemble_call.is_ensemble = True
        _ensemble_call.ensemble_mode = mode
        _ensemble_call.model_name = " + ".join(e["model"] for e in active)
        _ensemble_call.provider_name = ",".join(e["key"] for e in active)
        return _ensemble_call

    def get_engine(self, name: Optional[str] = None) -> Optional[AsyncLLMEngine]:
        """Returns a callable; uses ensemble when multiple engines are configured."""
        prefs = _get_user_prefs()
        ensemble_mode = str(prefs.get("ensemble_mode", "race")).lower()
        if ensemble_mode not in ALL_ENSEMBLE_MODES:
            ensemble_mode = "consensus"
        active = self.get_active_engines()

        if len(active) > 1:
            logger.info(
                f"Ensemble [{ensemble_mode}]: {[e['key'] for e in active]}"
            )
            return self._build_ensemble_engine(active, ensemble_mode)

        single_name = name
        if single_name and "," in single_name:
            single_name = single_name.split(",")[0].strip()
        elif not single_name and active:
            single_name = active[0]["key"]

        engine = self._get_single_engine_callable(single_name)
        if not engine:
            logger.error(
                "ENGINE RESOLUTION FAILED: No LLM provider has a valid API key. "
                "Configure at least one of: DEEPSEEK_API_KEY, ANTHROPIC_API_KEY, "
                "OPENAI_API_KEY, GOOGLE_API_KEY, OPENROUTER_API_KEY. "
                "Falling back to DummyAsyncLLM."
            )
            return DummyAsyncLLM()

        provider, resolved = self._resolve_provider(single_name)
        if provider and provider.api_key:
            logger.info(
                f"Engine resolved: {provider.__class__.__name__} (model={provider.model})"
            )
        return engine

from .telemetry import compute_benchmark_metrics as _compute_benchmark_metrics, render_engine_benchmark_block as _render_engine_benchmark_block

async def _display_truthgpt_benchmark(
    elapsed_time: float,
    model_name: str | None = None,
    tokens: int | None = None,
    engine_key: str | None = None,
):
    """Calculate and display benchmark stats for every active engine in preferred_engine."""
    prefs = _get_user_prefs()

    opts = {
        "MCTS": prefs.get("mcts_optimized", False),
        "Speculative Decoding": prefs.get("speculative_decoding", False),
        "KV-Cache (4-bit)": prefs.get("kv_quantization", False),
        "DPO Truthfulness": prefs.get("dpo_truth_bias", False),
        "RAG Fusion": prefs.get("rag_fusion_opt", False),
        "Swarm Pruning": True,  # Auto-enabled
        "CoVe Verification": prefs.get("cove_hallucination_control", False),
        "Math Formalizer": prefs.get("math_formalizer", False),
        "arXiv SOTA": prefs.get("sota_injection", False),
        "Self-Refinement": prefs.get("self_refinement", False),
        "Flash Attention v3": prefs.get("flash_attention_v3", False),
        "Dynamic LoRA": prefs.get("dynamic_lora", False),
        "Forensic Audit": prefs.get("forensic_audit", False),
        "Cross-Model MoE": prefs.get("cross_model_moe", False),
        "Cache Warming": prefs.get("cache_warming", False),
    }

    active_list = [k for k, v in opts.items() if v]
    active_str = ", ".join(active_list)

    live_key = _normalize_engine_key(engine_key) if engine_key else None
    if not live_key and model_name:
        for eng in engine_registry.get_active_engines():
            if eng["model"] == model_name:
                live_key = eng["key"]
                break

    ensemble_mode = str(prefs.get("ensemble_mode", "race")).lower()
    if ensemble_mode not in ALL_ENSEMBLE_MODES:
        ensemble_mode = "consensus"
    now = time.time()

    try:
        from interface.cc_style import _console
        active_engines = engine_registry.get_active_engines()
        if not active_engines:
            active_engines = [{"key": "unknown", "label": "engine", "model": model_name or "unknown"}]

        engine_names = ", ".join(e["label"] for e in active_engines)
        mode_label = f" · [magenta]{ensemble_mode}[/magenta]" if ensemble_mode in _MULTI_ENSEMBLE_MODES else ""
        _console.print(
            f"     [dim]⎿[/dim]  [bold yellow]NEURAL OVERDRIVE BENCHMARK[/bold yellow] "
            f"[dim]({len(active_engines)} engine(s): {engine_names}{mode_label})[/dim]"
        )

        for eng in active_engines:
            key = eng["key"]
            eng_model = eng["model"]
            cached = _benchmark_run_stats.get(key)
            is_live = False
            if cached and (now - cached.get("ts", 0)) < 8.0:
                is_live = True
            elif live_key == key:
                is_live = True
            elif not live_key and eng_model == model_name:
                is_live = True

            if is_live and cached:
                run_elapsed = cached["elapsed"]
                run_tokens = cached["tokens"]
                eng_model = cached.get("model", eng_model)
            elif is_live:
                run_elapsed = elapsed_time
                run_tokens = tokens
            else:
                run_elapsed = 0.0
                run_tokens = 0
                if cached:
                    run_elapsed = cached["elapsed"]
                    run_tokens = cached["tokens"]
                    eng_model = cached.get("model", eng_model)
                else:
                    continue

            metrics = _compute_benchmark_metrics(run_elapsed, run_tokens, opts)
            _render_engine_benchmark_block(
                _console,
                eng["label"],
                eng_model,
                metrics,
                is_live=is_live,
            )

        _console.print(f"        [dim]Active Layers: {active_str}[/dim]")

        # Render Button to Tune Overdrive
        _console.print("\n        [bold yellow]⚡ [O] Tune Overdrive Layers (Direct Improvement Portal)[/bold yellow]  [dim]│  Press 'o' anytime to open[/dim]")

        global OVERDRIVE_TRIGGERED
        if OVERDRIVE_TRIGGERED:
            OVERDRIVE_TRIGGERED = False
            _console.print("\n        [bold magenta]🚀 Opening Overdrive Portal...[/bold magenta]")
            from interface.overdrive_menu import handle_overdrive_menu
            await handle_overdrive_menu()
            
    except Exception as err:
        logger.debug(f"Could not render benchmark table: {err}")

engine_registry = EngineRegistry()

async def safe_llm_call(engine: AsyncLLMEngine, prompt: str, trace_id: str | None = None, **kwargs) -> str:
    """High-fidelity wrapper for LLM calls with TUI integration and per-call tracing."""
    if engine is None:
        logger.warning("safe_llm_call received None engine. Resolving from registry...")
        engine = engine_registry.get_engine()

    t0 = time.time()
    label = "LLM_INFERENCE"

    # Try to extract model name and provider key from engine callable
    model_name = getattr(engine, "model_name", None)
    engine_key = getattr(engine, "provider_name", None)
    if not model_name:
        model_name = getattr(engine, "model", None)
    if not model_name:
        self_obj = getattr(engine, "__self__", None)
        if self_obj:
            model_name = getattr(self_obj, "model", None) or getattr(self_obj, "model_name", None)
    is_ensemble = getattr(engine, "is_ensemble", False)

    # --- Intelligent API Cost Routing ---
    prefs = _get_user_prefs()
    if prefs.get("auto_cost_routing", True) and not is_ensemble:
        if len(prompt) < 250 and not any(kw in prompt.lower() for kw in ["code", "analyze", "explain in detail", "architect", "refactor"]):
            try:
                cheaper_engine = engine_registry._get_single_engine_callable("google")
                if cheaper_engine and getattr(cheaper_engine, "provider_name", None) != engine_key:
                    engine = cheaper_engine
                    model_name = getattr(engine, "model_name", getattr(engine, "model", None))
                    engine_key = getattr(engine, "provider_name", None)
                    logger.info(f"🪙 API Optimizer: Auto-Routed to {engine_key} ({model_name}) due to low complexity. Savings: ~80%")
            except Exception:
                pass

    # Apply Latency Optimizations (Chain of Draft / Elastic Reasoning)
    if prefs.get("chain_of_draft", True) or prefs.get("elastic_reasoning", True):
        try:
            import sys
            sys.path.insert(0, "C:/blatam-academy")
            from latency_optimizations import apply_chain_of_draft, apply_elastic_reasoning
            if prefs.get("chain_of_draft", True):
                prompt = apply_chain_of_draft(prompt)
            if prefs.get("elastic_reasoning", True):
                prompt = apply_elastic_reasoning(prompt, 30, 200)
        except Exception:
            pass

    # --- Open a tracing span for this LLM call so traces_history.json shows
    # individual llm_inference children again (recent traces had only the root).
    llm_span = None
    if trace_id:
        try:
            try:
                from agents.observability import global_tracer
            except ImportError:
                from .observability import global_tracer
            llm_span = global_tracer.start_span(
                trace_id=trace_id,
                name="llm_inference",
                kind="llm_call",
                input_data=prompt,
                metadata={
                    "model": model_name or "",
                    "engine": engine_key or "",
                    "ensemble": is_ensemble,
                    "prompt_chars": len(prompt),
                },
            )
        except Exception:
            llm_span = None  # tracing must never break inference

    def _finish_span(output: str, status: str, elapsed: float, tokens: int) -> None:
        if llm_span is None:
            return
        try:
            llm_span.finish(
                output=output,
                status=status,
                metadata={
                    "elapsed_ms": round(elapsed * 1000, 2),
                    "approx_tokens": tokens,
                },
            )
        except Exception:
            pass

    # Check cache early to bypass inference entirely if possible
    cached_result = get_cached_response(prompt)
    if cached_result:
        logger.info(f"🟢 Semantic Cache HIT! Bypassing {engine_key} inference. Savings: 100%")
        _finish_span(str(cached_result), "cache_hit", 0.05, len(str(cached_result)) // 4)
        return cached_result

    async def _run_and_record() -> str:
        nonlocal model_name, engine_key
        try:
            result = await engine(prompt, **kwargs)
            elapsed = time.time() - t0
            tokens = max(1, len(str(result)) // 4)
            if engine_key and not is_ensemble:
                _record_benchmark_run(engine_key, model_name or "", elapsed, tokens)
            _finish_span(str(result), "ok", elapsed, tokens)
            set_cached_response(prompt, str(result))
            return result
        except Exception as primary_error:
            logger.warning(f"Primary engine '{engine_key or 'unknown'}' failed: {primary_error}. Attempting fallback cascade...")
            # Fetch fallback engines
            fallback_engines = []
            exclude_providers = set()
            if engine_key:
                exclude_providers.update([k.strip() for k in engine_key.split(",")])
            for name in ["deepseek", "claude", "openai", "google", "openrouter"]:
                try:
                    provider, resolved = engine_registry._resolve_provider(name)
                    if provider and provider.api_key and resolved not in exclude_providers:
                        callable_engine = engine_registry._get_single_engine_callable(resolved)
                        if callable_engine:
                            fallback_engines.append((resolved, provider.model, callable_engine))
                except Exception:
                    pass
            
            for fb_key, fb_model, fb_engine in fallback_engines:
                try:
                    logger.info(f"Fallback cascade: attempting {fb_key} ({fb_model})...")
                    result = await fb_engine(prompt, **kwargs)
                    elapsed = time.time() - t0
                    tokens = max(1, len(str(result)) // 4)
                    _record_benchmark_run(fb_key, fb_model, elapsed, tokens)
                    _finish_span(str(result), "ok", elapsed, tokens)
                    # Modify model_name and engine_key dynamically so metrics show the fallback
                    model_name = fb_model
                    engine_key = fb_key
                    return result
                except Exception as fb_error:
                    logger.warning(f"Fallback engine '{fb_key}' failed: {fb_error}")
            
            # If all fallbacks fail, try DummyAsyncLLM
            logger.error("All fallback engines failed. Executing DummyAsyncLLM fallback.")
            dummy = DummyAsyncLLM()
            result = await dummy(prompt, **kwargs)
            elapsed = time.time() - t0
            model_name = dummy.model_name
            engine_key = dummy.provider_name
            _finish_span(str(result), "dummy_fallback", elapsed, len(str(result)) // 4)
            return result

    if CC_AVAILABLE:
        from interface.cc_style import cc_spinner, cc_result, _fmt_elapsed, _fmt_tokens
        with cc_spinner(label) as sp:
            try:
                result = await _run_and_record()
                elapsed = time.time() - t0
                tokens = len(str(result)) // 4
                from interface import cc_style
                if not cc_style.SUPPRESS_SPINNERS:
                    sp.add_tokens(tokens)
                    note = f"{_fmt_elapsed(elapsed)} · ~{_fmt_tokens(tokens)} tkn"
                    if is_ensemble:
                        mode = getattr(engine, "ensemble_mode", "ensemble")
                        note += f" · {mode}"
                    cc_result(label, note=note)

                    try:
                        from interface.cc_style import _console
                        from rich.panel import Panel
                        from rich.markdown import Markdown
                        from rich.padding import Padding
                        parsed = json.loads(result)
                        if "thought" in parsed and parsed["thought"]:
                            thought_text = parsed["thought"].strip()
                            panel = Panel(
                                Markdown(thought_text),
                                title="[bold cyan]🧠 White Box Trace[/bold cyan]",
                                title_align="left",
                                border_style="dim cyan",
                                expand=False,
                                padding=(0, 2)
                            )
                            # 8 spaces indent to align cleanly with the tree view
                            _console.print(Padding(panel, (0, 0, 0, 8)))
                    except Exception:
                        pass

                    await _display_truthgpt_benchmark(
                        elapsed,
                        model_name,
                        tokens,
                        engine_key=None if is_ensemble else engine_key,
                    )
                if cc_style.REASONING_CALLBACK:
                    cc_style.REASONING_CALLBACK(f"LLM_INFERENCE completed in {_fmt_elapsed(elapsed)} (~{_fmt_tokens(tokens)} tkn)")
                return result
            except Exception as e:
                tb = traceback.format_exc()
                logger.error(f"Inference crash [{type(e).__name__}]: {e}\n{tb}")
                
                try:
                    from interface.cc_style import ssl_error_hint
                    hint = ssl_error_hint(e)
                except ImportError:
                    hint = None
                
                extra = f" {hint}" if hint else " Check API key validity and network connectivity."
                _finish_span(f"{type(e).__name__}: {str(e)[:200]}", "error", time.time() - t0, 0)
                return json.dumps({
                    "thought": f"LLM inference failed: [{type(e).__name__}] {str(e)[:300]}",
                    "final_answer": f"Inference error: {type(e).__name__}: {str(e)[:200]}.{extra}"
                })
    else:
        try:
            return await _run_and_record()
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"Inference crash [{type(e).__name__}]: {e}\n{tb}")
            
            try:
                from interface.cc_style import ssl_error_hint
                hint = ssl_error_hint(e)
            except ImportError:
                hint = None
                
            extra = f" {hint}" if hint else " Check API key validity and network connectivity."
            _finish_span(f"{type(e).__name__}: {str(e)[:200]}", "error", time.time() - t0, 0)
            return json.dumps({
                "thought": f"LLM inference failed: [{type(e).__name__}] {str(e)[:300]}",
                "final_answer": f"Inference error: {type(e).__name__}: {str(e)[:200]}.{extra}"
            })