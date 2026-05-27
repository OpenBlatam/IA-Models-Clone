"""
Core Utilities & Shared State for TruthGPT Interface.

╔═══════════════════════════════════════════════════════════════════╗
║  This module is now a thin re-export facade.                     ║
║  All logic has been decomposed into focused submodules:          ║
║                                                                   ║
║  • interface.preferences   — user prefs, env keys, personalize  ║
║  • interface.telemetry     — TelemetryProvider, budget, balances ║
║  • interface.theming       — headers, boot, themes, logging     ║
║  • interface.input_handler — get_input, get_choice, wait_for_user║
║  • interface.export_utils  — export/save mission, code-block    ║
╚═══════════════════════════════════════════════════════════════════╝

All imports remain backward-compatible. Direct imports from this
module will continue to work, but new code should import from the
dedicated submodules for clarity.
"""

from __future__ import annotations

import os
import sys

# ─── .env Loading (must happen early, before any prefs) ──────────────────

try:
    from dotenv import load_dotenv
    from pathlib import Path as _Path

    _current = _Path(__file__).resolve().parent
    for _ in range(15):
        _env_path = _current / ".env"
        if _env_path.exists():
            load_dotenv(_env_path, override=True)
            break
        if _current.parent == _current:
            break
        _current = _current.parent
except Exception:
    pass

# ─── Lazy Console Proxy (kept here as the canonical console instance) ────

_console = None


def get_console():
    global _console
    if _console is None:
        from rich.console import Console
        _console = Console()
    return _console


class LazyConsole:
    def __getattr__(self, name):
        return getattr(get_console(), name)

    def __repr__(self):
        return repr(get_console())


console = LazyConsole()

# ─── Path Initialization ─────────────────────────────────────────────────

from pathlib import Path
current_dir = Path(__file__).resolve().parent.parent

# ─── Initialize: QuickEdit fix ───────────────────────────────────────────

from interface.input_handler import disable_quick_edit
disable_quick_edit()

# ═══════════════════════════════════════════════════════════════════════════
# Backward-compatible re-exports from submodules
# ═══════════════════════════════════════════════════════════════════════════

# --- Preferences ---
from interface.preferences import (
    CONFIG_PATH,
    load_user_prefs,
    save_user_prefs,
    populate_env_from_prefs,
    handle_personalize,
)

USER_PREFS = load_user_prefs()
populate_env_from_prefs(USER_PREFS)

# --- Telemetry ---
from interface.telemetry import (
    TelemetryProvider,
    get_real_budget_stats,
    get_system_telemetry,
    fetch_balances_background,
)

# --- Theming / Visual ---
from interface.theming import (
    clear_screen,
    get_header,
    get_claude_header,
    get_theme_color,
    get_theme_panel,
    linux_boot_sequence,
    show_main_dashboard,
    log_event,
    log_activity,
    claude_log_event,
    START_TIME,
    SYSTEM_LOGS,
    system_history,
    background_missions,
    BLOCKCHAIN_READY,
    _CACHED_PAPER_COUNT,
    _LAST_PAPER_SCAN,
    _LAST_SWARM_LATENCY,
    _LAST_FRONTIER_LATENCY,
    _LAST_ACTIVITY_TIME,
    _SPARKLINE_HISTORY,
)

# --- Input Handling ---
from interface.input_handler import (
    get_input,
    get_choice,
    wait_for_user,
)

# --- Export Utilities ---
from interface.export_utils import (
    export_mission_result,
    save_mission_output,
    extract_target_directory,
)
