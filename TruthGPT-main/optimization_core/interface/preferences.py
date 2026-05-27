"""
User Preferences Management for TruthGPT Interface.

Handles loading, saving, and caching of user preferences with:
- Atomic file writes (temp + rename)
- Corruption recovery (auto-backup corrupt files)
- mtime-based cache invalidation
- Environment variable population from stored API keys
"""

import os
import sys
import json
import time
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

# --- Path Initialization ---
_INTERFACE_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _INTERFACE_DIR.parent
CONFIG_PATH = _PROJECT_DIR / "user_preferences.json"

class ApiKeys(BaseModel):
    telegram: str = ""
    discord: str = ""
    slack: str = ""
    whatsapp: str = ""
    openai: str = ""
    deepseek: str = ""
    anthropic: str = ""
    google: str = ""
    openrouter: str = ""

class ApiCredits(BaseModel):
    claude: float = 10.00
    openai: float = 10.00
    google: float = 10.00

class UserPreferences(BaseModel):
    user_name: str = "Explorer"
    preferred_engine: str = "deepseek"
    theme: str = "claude"
    continuous_mode: bool = False
    mcp_servers: List[str] = Field(default_factory=lambda: ["http://localhost:8000"])
    api_keys: ApiKeys = Field(default_factory=ApiKeys)
    api_credits: ApiCredits = Field(default_factory=ApiCredits)
    ensemble_mode: str = "race"
    google_access_token: str = ""
    google_service_account: str = ""

# Environment variable mapping: pref key -> env var name
_API_KEY_ENV_MAP = {
    "openai": "OPENAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}

# --- Cache State ---
_prefs_lock = threading.Lock()
_cached_user_prefs: Optional[Dict[str, Any]] = None
_cached_user_prefs_mtime: float = 0


def load_user_prefs() -> Dict[str, Any]:
    """Load user preferences with mtime-based caching and corruption recovery."""
    global _cached_user_prefs, _cached_user_prefs_mtime

    with _prefs_lock:
        mtime: float = 0
        if CONFIG_PATH.exists():
            try:
                mtime = os.path.getmtime(CONFIG_PATH)
            except Exception:
                pass

        if _cached_user_prefs is not None and mtime == _cached_user_prefs_mtime:
            return _cached_user_prefs

        defaults = UserPreferences().model_dump()

        if CONFIG_PATH.exists():
            try:
                loaded = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    if "api_keys" in loaded and isinstance(loaded["api_keys"], dict):
                        defaults["api_keys"].update(loaded["api_keys"])
                    if "api_credits" in loaded and isinstance(loaded["api_credits"], dict):
                        defaults["api_credits"].update(loaded["api_credits"])
                    defaults.update(loaded)
                    
                # Validate through Pydantic to ensure all types are correct
                validated = UserPreferences(**defaults)
                defaults = validated.model_dump()
            except Exception:
                # Corruption detected — rename to .corrupt and boot defaults
                try:
                    corrupt_backup = CONFIG_PATH.with_suffix(".corrupt")
                    if CONFIG_PATH.exists():
                        if corrupt_backup.exists():
                            corrupt_backup.unlink()
                        CONFIG_PATH.rename(corrupt_backup)
                except Exception:
                    pass

        _cached_user_prefs = defaults
        _cached_user_prefs_mtime = mtime
        return defaults


def save_user_prefs(prefs: Dict[str, Any]) -> None:
    """Persist user preferences with atomic write (temp + replace)."""
    global _cached_user_prefs, _cached_user_prefs_mtime
    
    with _prefs_lock:
        try:
            # Validate before saving
            validated = UserPreferences(**prefs)
            clean_prefs = validated.model_dump()
        except Exception:
            clean_prefs = prefs
            
        _cached_user_prefs = clean_prefs
        try:
            temp_path = CONFIG_PATH.with_suffix(".tmp")
            temp_path.write_text(json.dumps(clean_prefs, indent=4), encoding="utf-8")
            if temp_path.exists():
                os.replace(temp_path, CONFIG_PATH)
        except Exception:
            # Fallback to direct write if replace fails
            try:
                CONFIG_PATH.write_text(json.dumps(clean_prefs, indent=4), encoding="utf-8")
            except Exception as e:
                import logging
                logging.getLogger().error(f"Failed to write user preferences: {e}")

        if CONFIG_PATH.exists():
            try:
                _cached_user_prefs_mtime = os.path.getmtime(CONFIG_PATH)
            except Exception:
                pass


def populate_env_from_prefs(prefs: Dict[str, Any]) -> None:
    """Set environment variables from stored API keys and sync Overdrive options."""
    api_keys = prefs.get("api_keys", {})
    for pref_key, env_key in _API_KEY_ENV_MAP.items():
        val = api_keys.get(pref_key)
        if val and not os.environ.get(env_key):
            os.environ[env_key] = val

    # Sync Overdrive settings with TruthGPT config
    try:
        from interface.overdrive_menu import OPTIONS
        overdrive_map = {
            "mcts_optimized": "TRUTHGPT_USE_MCTS_REASONING",
            "speculative_decoding": "TRUTHGPT_USE_SPECULATIVE_DECODING",
            "self_refinement": "TRUTHGPT_USE_SELF_CONSISTENCY",
            "math_formalizer": "TRUTHGPT_USE_FP16_STABILITY",
            "cove_hallucination_control": "TRUTHGPT_USE_ELASTIC_REASONING",
            "sota_injection": "TRUTHGPT_USE_CHAIN_OF_DRAFT",
        }
        
        for opt in OPTIONS:
            key = opt["key"]
            val = prefs.get(key, False)
            env_key = overdrive_map.get(key, f"TRUTHGPT_USE_{key.upper()}")
            os.environ[env_key] = str(val).lower()
            
        # Hot-reload the settings instance if it's already imported
        try:
            from agents.razonamiento_planificacion.config import settings
            for k, v in os.environ.items():
                if k.startswith("TRUTHGPT_"):
                    attr_name = k.replace("TRUTHGPT_", "")
                    if hasattr(settings, attr_name):
                        if v.lower() == 'true':
                            setattr(settings, attr_name, True)
                        elif v.lower() == 'false':
                            setattr(settings, attr_name, False)
        except Exception:
            pass
    except ImportError:
        pass


def invalidate_cache() -> None:
    """Force the next load_user_prefs() to re-read from disk."""
    global _cached_user_prefs, _cached_user_prefs_mtime
    _cached_user_prefs = None
    _cached_user_prefs_mtime = 0


# ─── Engine Metadata (shared by personalization UI) ─────────────────────

ENGINE_LIST = ["deepseek", "google", "openrouter", "chatgpt", "claude"]

ENGINE_METADATA = {
    "deepseek": ("DeepSeek", "deepseek-v4-pro", "deepseek", "DEEPSEEK_API_KEY"),
    "google": ("Google Gemini", "gemini-3.5-flash", "google", "GOOGLE_API_KEY"),
    "openrouter": ("OpenRouter Unified", "google/gemini-3.5-flash", "openrouter", "OPENROUTER_API_KEY"),
    "chatgpt": ("OpenAI (ChatGPT)", "gpt-5.5", "openai", "OPENAI_API_KEY"),
    "claude": ("Anthropic Claude", "claude-opus-4-7", "anthropic", "ANTHROPIC_API_KEY"),
}

OPENROUTER_MODELS = [
    "perplexity/sonar-reasoning-pro",
    "perplexity/sonar-pro-search",
    "openrouter/owl-alpha",
    "openrouter/hunter-alpha",
    "openrouter/healer-alpha",
    "xai/grok-4.3",
    "xai/grok-build-0.1",
    "nvidia/llama-3.1-nemotron-70b-instruct",
    "cohere/command-r7-plus",
    "deepseek/deepseek-r1",
    "anthropic/claude-3.7-sonnet",
    "google/gemini-3.5-flash",
    "google/gemini-3.5-pro",
    "openai/gpt-5.5",
    "openai/gpt-5.5-instant",
    "openai/gpt-4.5-preview",
    "deepseek/deepseek-chat",
    "anthropic/claude-3.5-sonnet",
    "openai/gpt-4o",
    "meta-llama/llama-3.3-70b-instruct",
    "qwen/qwen-2.5-72b-instruct",
]

OPENROUTER_MODEL_NAMES = {
    "perplexity/sonar-reasoning-pro": "Perplexity Sonar Reasoning Pro (Agentic Search)",
    "perplexity/sonar-pro-search": "Perplexity Sonar Pro Search",
    "openrouter/owl-alpha": "OpenRouter Owl Alpha (Tool Use & Long Context)",
    "openrouter/hunter-alpha": "OpenRouter Hunter Alpha (1T+ Planning Agent)",
    "openrouter/healer-alpha": "OpenRouter Healer Alpha (Omni-modal Reasoning)",
    "xai/grok-4.3": "xAI Grok 4.3 (Exclusive reasoning/search)",
    "xai/grok-build-0.1": "xAI Grok Build 0.1 (Coding)",
    "nvidia/llama-3.1-nemotron-70b-instruct": "Nvidia Nemotron 70B Instruct (High Precision)",
    "cohere/command-r7-plus": "Cohere Command R7+ (Agentic & RAG)",
    "deepseek/deepseek-r1": "DeepSeek R1 (Reasoning)",
    "anthropic/claude-3.7-sonnet": "Claude 3.7 Sonnet (Recommended)",
    "google/gemini-3.5-flash": "Gemini 3.5 Flash",
    "google/gemini-3.5-pro": "Gemini 3.5 Pro",
    "openai/gpt-5.5": "GPT 5.5",
    "openai/gpt-5.5-instant": "GPT 5.5 Instant",
    "openai/gpt-4.5-preview": "GPT-4.5 (Research Preview)",
    "deepseek/deepseek-chat": "DeepSeek V3 (Chat)",
    "anthropic/claude-3.5-sonnet": "Claude 3.5 Sonnet",
    "openai/gpt-4o": "GPT-4o (Omni)",
    "meta-llama/llama-3.3-70b-instruct": "Llama 3.3 70B Instruct",
    "qwen/qwen-2.5-72b-instruct": "Qwen 2.5 72B Instruct",
}


# ─── Personalization TUI Handler ─────────────────────────────────────────

async def handle_personalize() -> None:
    """Interactive personalization menu."""
    from rich.panel import Panel
    from rich.table import Table

    # Lazy imports from sibling modules to avoid circular deps
    from interface.input_handler import get_input
    from interface.theming import clear_screen

    # Use a lazy console proxy
    from interface.core import console

    prefs = load_user_prefs()

    while True:
        clear_screen()
        console.print(Panel("[bold yellow]👤 Personalization & Settings[/bold yellow]", border_style="yellow"))

        engines = prefs.get("preferred_engine", "deepseek").split(",")
        engine_str = ", ".join([f"[cyan]{e.strip()}[/cyan]" for e in engines])

        table = Table(show_header=False, box=None)
        table.add_row("1. Change Name", f"[dim]Current: {prefs['user_name']}[/dim]")
        table.add_row("2. Set Engines (Multi-Engine Support)", f"[dim]Active: {engine_str}[/dim]")
        table.add_row("3. Ensemble Mode", f"[dim]Mode: {prefs.get('ensemble_mode', 'race')}[/dim]")
        table.add_row("4. UI Theme", f"[dim]Theme: {prefs.get('theme', 'industrial')}[/dim]")
        table.add_row("5. Google OAuth Token", f"[dim]Token: {'SET' if prefs.get('google_access_token') else 'EMPTY'}[/dim]")
        table.add_row("6. Google Service Account", f"[dim]Path: {prefs.get('google_service_account', 'EMPTY')}[/dim]")
        table.add_row("7. Set API Credit Balances (Claude/OpenAI/Gemini)", "[dim]Adjust offline starting estimates[/dim]")
        table.add_row("0. Back", "")
        console.print(table)

        choice = get_input("Select setting", choices=["0", "1", "2", "3", "4", "5", "6", "7"])
        if choice == "0":
            break
        elif choice == "1":
            prefs["user_name"] = get_input("Enter your name", default=prefs["user_name"])
        elif choice == "2":
            _handle_engine_selection(prefs, engines, console, get_input)
        elif choice == "3":
            prefs["ensemble_mode"] = get_input(
                "Ensemble Mode",
                choices=["consensus", "parallel", "race", "majority", "debate", "bayesian"],
                default=prefs.get("ensemble_mode", "race"),
            )
        elif choice == "4":
            prefs["theme"] = get_input(
                "Select Theme",
                choices=["industrial", "claude", "minimalist"],
                default=prefs.get("theme", "industrial"),
            )
        elif choice == "5":
            prefs["google_access_token"] = get_input(
                "Paste Google OAuth Token", default=prefs.get("google_access_token", "")
            )
        elif choice == "6":
            prefs["google_service_account"] = get_input(
                "Enter Service Account Path", default=prefs.get("google_service_account", "")
            )
        elif choice == "7":
            _handle_credit_balances(prefs, console, get_input)

        save_user_prefs(prefs)
        console.print("[green]✓ Settings updated.[/green]")
        time.sleep(0.5)


def _handle_engine_selection(prefs, engines, console, get_input):
    """Sub-handler for engine selection in the personalization menu."""
    from rich.table import Table

    table = Table(
        title="🧠 [bold cyan]Neural Reasoning Engines[/bold cyan]",
        border_style="cyan",
        header_style="bold magenta",
        show_lines=True,
    )
    table.add_column("#", justify="center", style="bold cyan")
    table.add_column("Engine Name", style="bold white")
    table.add_column("Provider / Brand", style="dim")
    table.add_column("Default Model", style="green")
    table.add_column("API Key Status", justify="center")

    for idx, eng in enumerate(ENGINE_LIST, 1):
        brand, model, pref_key, env_key = ENGINE_METADATA[eng]
        key_configured = bool(
            prefs.get("api_keys", {}).get(pref_key) or os.getenv(env_key)
        )
        status = "[bold green]Active[/bold green]" if key_configured else "[dim yellow]Key Missing[/dim yellow]"
        table.add_row(str(idx), eng, brand, model, status)

    console.print("\n[bold cyan]Select engines (comma-separated for ensemble):[/bold cyan]")
    console.print(table)

    selection = get_input("Engines", default=",".join(engines))
    parts = [p.strip() for p in selection.split(",")]
    resolved = []
    for p in parts:
        if p.isdigit():
            idx = int(p)
            if 1 <= idx <= len(ENGINE_LIST):
                resolved.append(ENGINE_LIST[idx - 1])
        else:
            resolved.append(p)

    # OpenRouter model sub-menu
    if "openrouter" in resolved:
        _handle_openrouter_model_selection(resolved, console, get_input)

    prefs["preferred_engine"] = ",".join(resolved)


def _handle_openrouter_model_selection(resolved, console, get_input):
    """Sub-handler for OpenRouter model selection."""
    from rich.table import Table

    console.print("\n[bold yellow]⚡ OpenRouter Model Selection:[/bold yellow]")

    model_table = Table(
        title="🌐 [bold cyan]Available OpenRouter Models[/bold cyan]",
        border_style="yellow",
        header_style="bold magenta",
        show_lines=True,
    )
    model_table.add_column("#", justify="center", style="bold yellow")
    model_table.add_column("Model ID", style="white")
    model_table.add_column("Friendly Name", style="dim")

    for idx, model_id in enumerate(OPENROUTER_MODELS, 1):
        model_table.add_row(str(idx), model_id, OPENROUTER_MODEL_NAMES[model_id])

    console.print(model_table)

    model_choice = get_input("Select model # or enter custom model ID", default="1")
    if model_choice.isdigit():
        m_idx = int(model_choice)
        if 1 <= m_idx <= len(OPENROUTER_MODELS):
            selected_model = OPENROUTER_MODELS[m_idx - 1]
        else:
            selected_model = OPENROUTER_MODELS[0]
    else:
        selected_model = model_choice.strip() or OPENROUTER_MODELS[0]

    for i, r in enumerate(resolved):
        if r == "openrouter":
            resolved[i] = f"openrouter:{selected_model}"

    console.print(f"[bold green]✓[/bold green] Selected OpenRouter model: [bold white]{selected_model}[/bold white]")


def _handle_credit_balances(prefs, console, get_input):
    """Sub-handler for API credit balance adjustment."""
    console.print("\n[bold yellow]💰 Adjust API Credit Balances[/bold yellow]")
    credits = prefs.setdefault("api_credits", {"claude": 10.00, "openai": 10.00, "google": 10.00})

    try:
        credits["claude"] = float(get_input("Claude starting credits ($USD)", default=str(credits.get("claude", 10.00))))
        credits["openai"] = float(get_input("OpenAI starting credits ($USD)", default=str(credits.get("openai", 10.00))))
        credits["google"] = float(get_input("Google starting credits ($USD)", default=str(credits.get("google", 10.00))))
    except ValueError:
        console.print("[red]❌ Invalid input. Please enter numbers only.[/red]")
        time.sleep(1.0)
