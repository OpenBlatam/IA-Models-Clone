"""
Core Utilities & Shared State for TruthGPT Interface
"""
import os
import sys
import time
import json
import re
from pathlib import Path
from typing import Optional, Dict, Any, List

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, IntPrompt, FloatPrompt, Confirm
from rich.text import Text

# Initialize Rich Console
console = Console()

# --- Path Initialization ---
current_dir = Path(__file__).resolve().parent.parent
CONFIG_PATH = current_dir / "user_preferences.json"

def load_user_prefs() -> Dict[str, Any]:
    defaults = {
        "user_name": "Explorer", 
        "preferred_engine": "deepseek", 
        "theme": "blue",
        "continuous_mode": False,
        "mcp_servers": ["http://localhost:8000"],
        "api_keys": {
            "telegram": "",
            "discord": "",
            "slack": "",
            "whatsapp": "",
            "openai": "",
            "deepseek": ""
        }
    }
    if CONFIG_PATH.exists():
        try:
            loaded = json.loads(CONFIG_PATH.read_text())
            if isinstance(loaded, dict):
                if "api_keys" in loaded and isinstance(loaded["api_keys"], dict):
                    defaults["api_keys"].update(loaded["api_keys"])
                    del loaded["api_keys"]
                defaults.update(loaded)
        except:
            pass
    return defaults

def save_user_prefs(prefs: Dict[str, Any]):
    CONFIG_PATH.write_text(json.dumps(prefs, indent=4))

USER_PREFS = load_user_prefs()

# --- Global System State ---
SYSTEM_LOGS = []
system_history = []
background_missions = []
BLOCKCHAIN_READY = False


def log_event(layer: str, event: str, status: str = "DONE"):
    timestamp = time.strftime("%H:%M:%S")
    SYSTEM_LOGS.append({"time": timestamp, "layer": layer, "event": event, "status": status})

def log_activity(module: str, task: str, status: str = "Completed"):
    system_history.append({
        "time": time.strftime('%H:%M:%S'),
        "module": module,
        "task": task,
        "status": status
    })
    if len(system_history) > 20:
        system_history.pop(0)

async def handle_personalize():
    while True:
        clear_screen()
        console.print(Panel("[bold yellow]👤 Personalization & Settings[/bold yellow]", border_style="yellow"))
        table = Table(show_header=False, box=None)
        table.add_row("1. Change Name", f"[dim]Current: {USER_PREFS['user_name']}[/dim]")
        table.add_row("2. Change Engine", f"[dim]Current: {USER_PREFS['preferred_engine']}[/dim]")
        table.add_row("0. Back", "")
        console.print(table)
        choice = Prompt.ask("Select setting", choices=["0", "1", "2"])
        if choice == "0": break
        elif choice == "1":
            USER_PREFS["user_name"] = Prompt.ask("Enter your name", default=USER_PREFS["user_name"])
        elif choice == "2":
            USER_PREFS["preferred_engine"] = Prompt.ask("Preferred LLM Engine", choices=["deepseek", "mock"], default=USER_PREFS["preferred_engine"])
        save_user_prefs(USER_PREFS)
        console.print("[green]✓ Settings updated.[/green]")
        time.sleep(0.5)

def linux_boot_sequence():
    clear_screen()
    console.print("[bold green]Booting TruthGPT Industrial OS...[/bold green]")
    time.sleep(0.5)

async def show_main_dashboard(extended: bool = False):
    clear_screen()
    console.print(get_header())
    
    console.print(f" [bold white]CORE SYSTEM LAYERS[/bold white]\n")
    core_layers = [
        ("0", "🛡️ Kernel"), ("1", "🐝 Swarm"), ("2", "🚀 Frontier"), ("3", "🔍 Research")
    ]
    for lid, name in core_layers:
        console.print(f"  [bold orange3]{lid:>2}[/bold orange3]  [white]{name}[/white]")
    
    if extended:
        console.print(f"\n [bold white]ADVANCED & EXTERNAL LAYERS[/bold white]\n")
        advanced_layers = [
            ("4", "⚙️ Opts"), ("5", "🧠 Labs"), ("6", "📱 Comm"), 
            ("9", "⛓️ Web3"), ("10", "🖥️ Node"), ("11", "📜 Tasks"),
            ("13", "📊 Market"), ("15", "🤖 RL"), ("16", "⚡ Overdrive"), ("P", "👤 Settings")
        ]
        # Print in a 3-column grid for compactness
        from rich.columns import Columns
        cols = [f"  [bold cyan]{lid:>2}[/bold cyan] [white]{name}[/white]" for lid, name in advanced_layers]
        console.print(Columns(cols, equal=True, expand=True))
    else:
        console.print(f"\n [dim] (Type '99' or '+' to toggle Extended View) [/dim]")

    console.print(f"\n [bold white]Type command ID or 'help' to begin.[/bold white]")


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def wait_for_user(force: bool = False):
    if force or not USER_PREFS.get("continuous_mode", False):
        console.input("\n[dim]Press Enter to continue...[/dim]")
    else:
        time.sleep(1)

def get_header() -> Panel:
    banner = r"""
   _____                      _      _____  _____  _______
  |_   _| __ _   _  | |_  | |_   / ____||  __ \|__   __|
    | |  |  __| | | | | __| | __| | |  __ | |__) |  | |
    | |  | |  | |_| | | |_  | |_  | |__  ||  ___/   | |
    |_|  |_|   \__,_|  \__|  \__|  \_____||_|       |_|
    """
    user_name = USER_PREFS.get("user_name", "Explorer")
    engine = USER_PREFS.get("preferred_engine", "deepseek")
    return Panel(
        Text(banner, style="bold orange3", justify="center"),
        title="[bold purple] TruthGPT Industrial OS [/bold purple]",
        subtitle=f"[bold orange3] truthgpt@kernel [/bold orange3]",
        border_style="orange3",
        padding=(1, 2)
    )

def export_mission_result(content: str, mission_name: str = "Mission_Result"):
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mission_name = mission_name.replace(" ", "_")
    console.print("\n[bold cyan]📤 Export & Reporting Engine[/bold cyan]")
    fmt = Prompt.ask("Export format", choices=["MD", "PDF", "Word"], default="MD").upper()
    filename = f"{mission_name}_{timestamp}"
    try:
        if fmt == "MD":
            path = Path(f"exports/{filename}.md")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            console.print(f"[bold green]✓ Exported to {path}[/bold green]")
    except Exception as e:
        console.print(f"[red]Export Error: {e}[/red]")

def save_mission_output(content: str, mission_name: str = "Mission"):
    report_dir = current_dir / "reports"
    report_dir.mkdir(exist_ok=True)
    filename = f"{mission_name}_{time.strftime('%Y%m%d_%H%M%S')}.md"
    filepath = report_dir / filename
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    console.print(f"[bold green]✓ Output exported to {filepath}[/bold green]")
