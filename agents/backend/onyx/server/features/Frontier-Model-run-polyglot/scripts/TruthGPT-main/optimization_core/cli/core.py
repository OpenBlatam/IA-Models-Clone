import os
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

console = Console()

def _fix_param(val, default_val):
    """Helper to unwrap Typer OptionInfo if called directly."""
    if hasattr(val, "default"):
        return val.default
    return val if val is not None else default_val

def safe_int(val, default=10):
    """Aggressively convert to int to fix slicing errors."""
    try:
        if hasattr(val, "default"):
            return int(val.default)
        return int(val)
    except:
        return default

def get_root_dirs():
    ROOT_DIR = Path(__file__).resolve().parent.parent.parent
    CURRENT_DIR = Path(__file__).resolve().parent.parent
    return ROOT_DIR, CURRENT_DIR

def setup_paths():
    ROOT_DIR, CURRENT_DIR = get_root_dirs()
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))
    if str(CURRENT_DIR) not in sys.path:
        sys.path.insert(0, str(CURRENT_DIR))
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())
