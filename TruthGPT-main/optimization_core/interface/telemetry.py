"""
System Telemetry & API Balance Tracking for TruthGPT Interface.

Provides:
- TelemetryProvider: Background thread for CPU/memory metrics with caching
- Budget tracking: Reads .budget.json for session cost data
- API balance fetching: Async background fetch for DeepSeek & OpenRouter balances
"""

import os
import json
import time
import threading
from typing import Dict, Any, Optional


# ─── Budget / Cost Tracking ──────────────────────────────────────────────

_CACHED_BUDGET_STATS: Optional[Dict[str, Any]] = None
_CACHED_BUDGET_MTIME: float = 0
_LAST_BUDGET_UPDATE: float = 0


def get_real_budget_stats() -> Dict[str, Any]:
    """Read actual API budget data from .budget.json with 2-second cache + mtime check."""
    global _CACHED_BUDGET_STATS, _CACHED_BUDGET_MTIME, _LAST_BUDGET_UPDATE

    now = time.time()
    path = ".budget.json"

    mtime: float = 0
    if os.path.exists(path):
        try:
            mtime = os.path.getmtime(path)
        except Exception:
            pass

    if (
        _CACHED_BUDGET_STATS is not None
        and mtime == _CACHED_BUDGET_MTIME
        and (now - _LAST_BUDGET_UPDATE) < 2.0
    ):
        return _CACHED_BUDGET_STATS

    stats: Dict[str, Any] = {"total_usd": 0.0, "savings_usd": 0.0, "limit": 2.0}
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                data = json.load(f)
                stats["total_usd"] = data.get("metrics", {}).get("total_usd", 0.0)
                stats["savings_usd"] = data.get("savings_usd", 0.0)
        except Exception:
            pass

    _CACHED_BUDGET_STATS = stats
    _CACHED_BUDGET_MTIME = mtime
    _LAST_BUDGET_UPDATE = now
    return stats


# ─── Async API Balance Fetching ──────────────────────────────────────────

async def fetch_balances_background() -> None:
    """Background task to fetch real API balances without blocking the TUI."""
    import httpx
    import asyncio
    import logging
    import warnings

    # Enforce silencing of HTTP library logs to prevent TUI canvas overlap
    warnings.filterwarnings("ignore")
    for logger_name in ["httpx", "httpcore", "urllib3"]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    provider = TelemetryProvider
    if provider._BALANCE_FETCHING:
        return
    provider._BALANCE_FETCHING = True

    try:
        from interface.preferences import load_user_prefs

        prefs = load_user_prefs()
        api_keys = prefs.get("api_keys", {})

        # 1. Fetch DeepSeek Balance
        deepseek_key = api_keys.get("deepseek") or os.getenv("DEEPSEEK_API_KEY")
        if deepseek_key:
            try:
                async with httpx.AsyncClient(timeout=3.0) as client:
                    resp = await client.get(
                        "https://api.deepseek.com/user/balance",
                        headers={"Authorization": f"Bearer {deepseek_key}"},
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        if data.get("is_available"):
                            infos = data.get("balance_infos", [])
                            if infos:
                                val = float(infos[0].get("total_balance", 0.0))
                                provider._CACHED_BALANCES["deepseek"] = {"val": val, "type": "API"}
            except Exception:
                pass

        # 2. Fetch OpenRouter Balance
        openrouter_key = api_keys.get("openrouter") or os.getenv("OPENROUTER_API_KEY")
        if openrouter_key:
            try:
                async with httpx.AsyncClient(timeout=3.0) as client:
                    resp = await client.get(
                        "https://openrouter.ai/api/v1/credits",
                        headers={"Authorization": f"Bearer {openrouter_key}"},
                    )
                    if resp.status_code == 200:
                        data = resp.json().get("data", {})
                        total_credits = data.get("total_credits")
                        total_usage = data.get("total_usage")
                        if total_credits is not None and total_usage is not None:
                            val = max(0.0, float(total_credits) - float(total_usage))
                            provider._CACHED_BALANCES["openrouter"] = {
                                "val": val, "type": "API", "usage": float(total_usage)
                            }
                        else:
                            await _fetch_openrouter_key_fallback(client, openrouter_key, provider)
                    else:
                        await _fetch_openrouter_key_fallback(client, openrouter_key, provider)
            except Exception:
                pass

        provider._LAST_BALANCE_UPDATE = time.time()
    except Exception:
        pass
    finally:
        provider._BALANCE_FETCHING = False


async def _fetch_openrouter_key_fallback(client, openrouter_key: str, provider) -> None:
    """Fallback: fetch OpenRouter balance via /auth/key endpoint."""
    try:
        resp_key = await client.get(
            "https://openrouter.ai/api/v1/auth/key",
            headers={"Authorization": f"Bearer {openrouter_key}"},
        )
        if resp_key.status_code == 200:
            d_key = resp_key.json().get("data", {})
            limit = d_key.get("limit")
            usage = d_key.get("usage", 0.0)
            if limit is not None and float(limit) > 0.0:
                val = max(0.0, float(limit) - float(usage))
            else:
                val = None
            provider._CACHED_BALANCES["openrouter"] = {
                "val": val, "type": "API", "usage": float(usage)
            }
    except Exception:
        pass


# ─── TelemetryProvider ───────────────────────────────────────────────────

class TelemetryProvider:
    """Encapsulates system telemetry gathering with background-thread caching."""

    _SESSION_ID: Optional[str] = None
    _LAST_CPU_VAL: float = 14.0
    _CACHED_STATS: Optional[Dict[str, Any]] = None
    _LAST_UPDATE: float = 0

    # Live API balance cache
    _CACHED_BALANCES: Dict[str, Dict[str, Any]] = {
        "deepseek": {"val": None, "type": "API"},
        "openrouter": {"val": None, "type": "API"},
        "claude": {"val": None, "type": "Est"},
        "openai": {"val": None, "type": "Est"},
        "google": {"val": None, "type": "Est"},
    }
    _LAST_BALANCE_UPDATE: float = 0
    _BALANCE_FETCHING: bool = False
    _TELEMETRY_UPDATER_STARTED: bool = False

    @classmethod
    def get_session_id(cls) -> str:
        if cls._SESSION_ID is None:
            import uuid
            cls._SESSION_ID = str(uuid.uuid4()).upper()[:5]
        return cls._SESSION_ID

    @classmethod
    def _start_telemetry_updater(cls) -> None:
        def updater():
            while True:
                try:
                    import psutil
                    cpu = psutil.cpu_percent()
                    if cpu > 0.0:
                        cls._LAST_CPU_VAL = cpu
                    mem = psutil.virtual_memory()
                    mem_val = mem.percent
                except (ImportError, Exception):
                    cpu = cls._LAST_CPU_VAL
                    mem_val = 32.0

                cls._CACHED_STATS = {
                    "load": cpu if cpu > 0.0 else cls._LAST_CPU_VAL,
                    "mem": mem_val,
                    "session_id": cls.get_session_id(),
                    "version": "TruthGPT v2.4.1",
                }
                time.sleep(2.0)

        t = threading.Thread(target=updater, daemon=True)
        t.start()

    @classmethod
    def get_stats(cls) -> Dict[str, Any]:
        """Gather metrics with a background thread to prevent UI stutter."""
        if not cls._TELEMETRY_UPDATER_STARTED:
            cls._TELEMETRY_UPDATER_STARTED = True
            cls._start_telemetry_updater()

        if cls._CACHED_STATS is None:
            cls._CACHED_STATS = {
                "load": cls._LAST_CPU_VAL,
                "mem": 32.0,
                "session_id": cls.get_session_id(),
                "version": "TruthGPT v2.4.1",
            }
        return cls._CACHED_STATS

    @classmethod
    def get_api_balances(cls) -> Dict[str, tuple]:
        """Returns cached API credit balances, triggering a background fetch if stale."""
        import asyncio

        now = time.time()

        # Trigger non-blocking fetch if stale (60s cache)
        if (now - cls._LAST_BALANCE_UPDATE) > 60.0 and not cls._BALANCE_FETCHING:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(fetch_balances_background())
            except RuntimeError:
                # No active event loop — run in a background thread
                def run_thread():
                    try:
                        asyncio.run(fetch_balances_background())
                    except Exception:
                        pass
                threading.Thread(target=run_thread, daemon=True).start()

        from interface.preferences import load_user_prefs

        prefs = load_user_prefs()
        credits = prefs.get("api_credits", {"claude": 10.00, "openai": 10.00, "google": 10.00})

        # Deduct session usage from the preferred/active engine
        budget_stats = get_real_budget_stats()
        session_cost = budget_stats.get("total_usd", 0.0)
        pref_engine = prefs.get("preferred_engine", "deepseek").split(",")[0].strip()

        res: Dict[str, tuple] = {}

        # 1. DeepSeek
        deepseek_key = prefs.get("api_keys", {}).get("deepseek") or os.getenv("DEEPSEEK_API_KEY")
        ds_cached = cls._CACHED_BALANCES.get("deepseek", {})
        if deepseek_key and ds_cached.get("val") is not None:
            res["DeepSeek"] = (ds_cached["val"], "API")
        else:
            res["DeepSeek"] = (max(0.0, 5.00 - (session_cost if pref_engine == "deepseek" else 0.0)), "Est")

        # 2. OpenRouter
        openrouter_key = prefs.get("api_keys", {}).get("openrouter") or os.getenv("OPENROUTER_API_KEY")
        or_cached = cls._CACHED_BALANCES.get("openrouter", {})
        if openrouter_key and or_cached.get("val") is not None:
            res["OpenRouter"] = (or_cached["val"], "API")
        else:
            res["OpenRouter"] = (max(0.0, 10.00 - (session_cost if "openrouter" in pref_engine else 0.0)), "Est")

        # 3. Claude (Anthropic)
        anthropic_key = prefs.get("api_keys", {}).get("anthropic") or os.getenv("ANTHROPIC_API_KEY")
        claude_start = float(credits.get("claude", 10.00))
        claude_val = max(0.0, claude_start - (session_cost if "claude" in pref_engine or "anthropic" in pref_engine else 0.0))
        res["Claude"] = (claude_val, "API" if anthropic_key else "Est")

        # 4. OpenAI
        openai_key = prefs.get("api_keys", {}).get("openai") or os.getenv("OPENAI_API_KEY")
        openai_start = float(credits.get("openai", 10.00))
        openai_val = max(0.0, openai_start - (session_cost if "openai" in pref_engine or "chatgpt" in pref_engine else 0.0))
        res["OpenAI"] = (openai_val, "API" if openai_key else "Est")

        # 5. Google (Gemini)
        google_key = prefs.get("api_keys", {}).get("google") or os.getenv("GOOGLE_API_KEY")
        google_start = float(credits.get("google", 10.00))
        google_val = max(0.0, google_start - (session_cost if "google" in pref_engine else 0.0))
        res["Gemini"] = (google_val, "API" if google_key else "Est")

        return res


def get_system_telemetry() -> Dict[str, Any]:
    """Proxy for TelemetryProvider — backward compatibility."""
    return TelemetryProvider.get_stats()
