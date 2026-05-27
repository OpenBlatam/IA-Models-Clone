"""
Theming & Visual Chrome for TruthGPT Interface.

Contains:
- Theme-aware header rendering (industrial / claude / minimalist)
- Boot sequence animation
- Theme utility functions (get_theme_color, get_theme_panel)
- Screen management (clear_screen)
- Log event rendering with theme dispatch
"""

import os
import time
from typing import Any, Optional, List


# ─── Screen Management ───────────────────────────────────────────────────

def clear_screen() -> None:
    """Clear the terminal screen."""
    os.system("cls" if os.name == "nt" else "clear")


# ─── Global State ────────────────────────────────────────────────────────

START_TIME = time.time()
SYSTEM_LOGS: List[dict] = []
system_history: List[dict] = []
background_missions: List[Any] = []
BLOCKCHAIN_READY = False

# Cached values for UI responsiveness
_CACHED_PAPER_COUNT = None
_LAST_PAPER_SCAN = 0

_LAST_SWARM_LATENCY = None
_LAST_FRONTIER_LATENCY = None
_LAST_ACTIVITY_TIME = time.time()
_SPARKLINE_HISTORY = list("⎵⎶⎷▂▃▅▇█▆▅▃ ")


# ─── Theme Utilities ─────────────────────────────────────────────────────

def get_theme_color() -> str:
    """Return the primary accent color for the current theme."""
    from interface.preferences import load_user_prefs
    theme = load_user_prefs().get("theme", "industrial")
    if theme in ["claude", "anthropic", "minimalist"]:
        return "plum1"
    return "orange3"


def get_theme_panel(content: Any, title: Optional[str] = None, border_style: Optional[str] = None):
    """Create a Rich Panel styled according to the current theme."""
    from rich.panel import Panel
    from interface.preferences import load_user_prefs

    theme = load_user_prefs().get("theme", "industrial")
    if not border_style:
        border_style = get_theme_color()

    if theme in ["claude", "anthropic", "minimalist"]:
        return Panel(content, title=title, border_style=border_style, padding=(1, 2))
    else:
        return Panel(content, title=title, border_style=border_style)


# ─── Headers ─────────────────────────────────────────────────────────────

def get_header():
    """Return the appropriate header panel based on the current theme."""
    from interface.preferences import load_user_prefs
    theme = load_user_prefs().get("theme", "industrial")
    if theme in ["claude", "anthropic", "minimalist"]:
        return get_claude_header()
    return _get_industrial_header()


def _get_industrial_header():
    """Classic TruthGPT industrial-style ASCII banner header."""
    from rich.panel import Panel
    from rich.text import Text

    banner = r"""
   _____                      _      _____  _____  _______
  |_   _| __ _   _  | |_  | |_   / ____||  __ \|__   __|
    | |  |  __| | | | | __| | __| | |  __ | |__) |  | |
    | |  | |  | |_| | | |_  | |_  | |__  ||  ___/   | |
    |_|  |_|   \__,_|  \__|  \__|  \_____||_|       |_|
    """
    return Panel(
        Text(banner, style="bold orange3", justify="center"),
        title="[bold purple] TruthGPT Industrial OS [/bold purple]",
        subtitle="[bold orange3] truthgpt@kernel [/bold orange3]  [bold red][R] Reboot[/bold red]",
        border_style="orange3",
        padding=(1, 2),
    )


def get_claude_header(updates: Optional[List[str]] = None):
    """Claude Code-style ultra-minimal header: one title line + thin divider."""
    from rich.text import Text
    import shutil

    from interface.preferences import load_user_prefs
    from interface.telemetry import TelemetryProvider, get_real_budget_stats

    w = shutil.get_terminal_size().columns or 100

    budget_stats = get_real_budget_stats()
    cost_str = f"${budget_stats['total_usd']:.4f}"
    tel = TelemetryProvider.get_stats()
    ts = time.strftime("%H:%M")
    prefs = load_user_prefs()
    user = prefs.get("user_name", "Explorer")
    engine = prefs.get("preferred_engine", "deepseek").split(",")[0].strip()
    uptime_s = int(time.time() - START_TIME)
    h, r = divmod(uptime_s, 3600)
    m, s = divmod(r, 60)
    uptime = f"{h:02d}:{m:02d}:{s:02d}"

    line = Text()
    line.append("\n  TruthGPT", style="bold white")
    line.append(" ✦", style="bold orange1")
    line.append(
        f"  {engine}  ·  {user}  ·  cost {cost_str}  ·  up {uptime}  ·  {ts}  ·  {tel['session_id']}",
        style="dim",
    )
    line.append("\n")
    line.append("  " + "─" * (w - 4), style="dim")
    line.append("\n")

    return line


# ─── Boot Sequence ───────────────────────────────────────────────────────

def linux_boot_sequence() -> None:
    """Claude Code-style boot: clean checklist, no logo, minimal chrome."""
    import shutil

    from interface.preferences import load_user_prefs
    from interface.telemetry import get_real_budget_stats

    # Lazy console
    _console = _get_console()

    clear_screen()
    w = shutil.get_terminal_size().columns or 100

    stages = [
        ("Neural Fabric", "reasoning substrate online"),
        ("Swarm Mesh", "agent cluster linked"),
        ("Expert Matrices", "47 MoE layers loaded"),
        ("SOTA Optimizer", "adaptive core v5.9 active"),
        ("Vault Sync", "knowledge layer decrypted"),
        ("Overdrive", "max-throughput pipeline unlocked"),
    ]

    budget = get_real_budget_stats()
    prefs = load_user_prefs()
    engine = prefs.get("preferred_engine", "deepseek").split(",")[0].strip()
    user = prefs.get("user_name", "Explorer")

    # Title line
    _console.print()
    t = time.strftime("%H:%M:%S")
    _console.print(f"  [bold white]TruthGPT[/bold white] [bold orange1]✦[/bold orange1]  [dim]v5.9-GOLD  ·  starting up  ·  {t}[/dim]")
    _console.print(f"  [dim]{'─' * (w - 4)}[/dim]")
    _console.print()

    for label, detail in stages:
        _console.print(f"  [bold green]✔[/bold green]  [white]{label:<18}[/white]  [dim]{detail}[/dim]")
        time.sleep(0.08)

    _console.print()
    _console.print(f"  [dim]{'─' * (w - 4)}[/dim]")
    _console.print(f"  [bold white]TruthGPT[/bold white] [bold orange1]✦[/bold orange1]  [dim]ready  ·  engine: {engine}  ·  user: {user}  ·  cost: ${budget['total_usd']:.4f}[/dim]")
    _console.print(f"  [dim]{'─' * (w - 4)}[/dim]")
    _console.print()
    time.sleep(0.15)


# ─── Dashboard ───────────────────────────────────────────────────────────

async def show_main_dashboard(extended: bool = False) -> None:
    """Render the main command dashboard."""
    from interface.preferences import load_user_prefs

    _console = _get_console()
    prefs = load_user_prefs()
    theme = prefs.get("theme", "industrial")

    if theme in ["claude", "anthropic", "minimalist"]:
        from interface.cc_style import cc_divider, cc_action, cc_prompt_footer
        clear_screen()
        print(get_header())
        core_layers = [
            ("K", "🛡️ Kernel"), ("1", "🐝 Swarm"), ("2", "🚀 Frontier"), ("3", "🔍 Research"),
        ]
        for lid, name in core_layers:
            _console.print(f"  [bold orange3]{lid:>2}[/bold orange3]  [white]{name}[/white]")

        if extended:
            _console.print()
            cc_action("ADVANCED & EXTERNAL LAYERS", status="INFO")
            advanced_layers = [
                ("4", "⚙️ Opts"), ("5", "🧠 Labs"), ("6", "📱 Comm"),
                ("9", "⛓️ Web3"), ("10", "🖥️ Node"), ("11", "📜 Tasks"),
                ("13", "📊 Market"), ("15", "🤖 RL"), ("16", "⚡ Overdrive"), ("P", "👤 Settings"),
            ]
            from rich.columns import Columns
            cols = [f"  [bold cyan]{lid:>2}[/bold cyan] [white]{name}[/white]" for lid, name in advanced_layers]
            _console.print(Columns(cols, equal=True, expand=True))
        else:
            _console.print(f"\n [dim] (Type '99' or '+' to toggle Extended View) [/dim]")

        cc_prompt_footer(context_hint="TruthGPT OS v5.9", interrupt_hint="Type command ID")
        return

    clear_screen()
    _console.print(get_header())

    core_layers = [
        ("K", "🛡️ Kernel"), ("1", "🐝 Swarm"), ("2", "🚀 Frontier"), ("3", "🔍 Research"),
    ]
    for lid, name in core_layers:
        _console.print(f"  [bold orange3]{lid:>2}[/bold orange3]  [white]{name}[/white]")

    if extended:
        _console.print(f"\n [bold white]ADVANCED & EXTERNAL LAYERS[/bold white]\n")
        advanced_layers = [
            ("4", "⚙️ Opts"), ("5", "🧠 Labs"), ("6", "📱 Comm"),
            ("9", "⛓️ Web3"), ("10", "🖥️ Node"), ("11", "📜 Tasks"),
            ("13", "📊 Market"), ("15", "🤖 RL"), ("16", "⚡ Overdrive"), ("P", "👤 Settings"),
        ]
        from rich.columns import Columns
        cols = [f"  [bold cyan]{lid:>2}[/bold cyan] [white]{name}[/white]" for lid, name in advanced_layers]
        _console.print(Columns(cols, equal=True, expand=True))
    else:
        _console.print(f"\n [dim] (Type '99' or '+' to toggle Extended View) [/dim]")

    _console.print(f"\n [bold white]Type command ID or 'help' to begin.[/bold white]")


# ─── Log Event Dispatchers ───────────────────────────────────────────────

def log_event(layer: str, event: str, status: str = "DONE") -> None:
    """Log an event with theme-aware rendering."""
    timestamp = time.strftime("%H:%M:%S")
    SYSTEM_LOGS.append({"time": timestamp, "layer": layer, "event": event, "status": status})

    from interface.preferences import load_user_prefs
    theme = load_user_prefs().get("theme", "industrial")
    if theme in ["claude", "anthropic", "minimalist"]:
        from interface.cc_style import cc_log_event
        cc_log_event(layer, event, status)
    else:
        _console = _get_console()
        _console.print(
            f"[dim]{timestamp}[/dim] [[bold orange3]{layer.upper()}[/bold orange3]] "
            f"[white]{event}[/white] -> [bold green]{status}[/bold green]"
        )


def log_activity(module: str, task: str, status: str = "Completed") -> None:
    """Log an activity entry with theme dispatch."""
    timestamp = time.strftime("%H:%M:%S")
    system_history.append({"time": timestamp, "module": module, "task": task, "status": status})
    if len(system_history) > 20:
        system_history.pop(0)

    from interface.preferences import load_user_prefs
    theme = load_user_prefs().get("theme", "industrial")
    if theme in ["claude", "anthropic", "minimalist"]:
        from interface.cc_style import cc_log_activity
        cc_log_activity(module, task, status)


def claude_log_event(layer: str, event: str, status: str = "DONE") -> None:
    """Claude-style log entry: clean and minimal."""
    colors = {"DONE": "green", "RUNNING": "cyan", "ERROR": "red", "PENDING": "dim"}
    color = colors.get(status, "white")
    timestamp = time.strftime("%H:%M:%S")
    _console = _get_console()
    _console.print(
        f"[dim]{timestamp}[/dim] [bold plum1]│[/bold plum1] "
        f"[white]{layer.upper():<8}[/white] [dim]➔[/dim] [{color}]{event}[/{color}]"
    )


# ─── Console Helper ──────────────────────────────────────────────────────

def _get_console():
    """Get a lazy Rich Console instance."""
    from interface.core import console
    return console
