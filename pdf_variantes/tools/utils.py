"""
Tool Utilities
==============
Common utility functions for tools.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def format_response_time(ms: float) -> str:
    """Format response time in human-readable format."""
    if ms < 1:
        return f"{ms*1000:.2f}μs"
    elif ms < 1000:
        return f"{ms:.2f}ms"
    else:
        return f"{ms/1000:.2f}s"


def format_bytes(bytes: int) -> str:
    """Format bytes in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024.0:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.2f} TB"


def validate_json(file_path: Path) -> bool:
    """Validate JSON file."""
    try:
        with open(file_path, "r") as f:
            json.load(f)
        return True
    except json.JSONDecodeError:
        return False
    except Exception:
        return False


def merge_json_files(files: List[Path], output_file: Path) -> bool:
    """Merge multiple JSON files."""
    try:
        merged_data = {
            "sources": [],
            "data": {}
        }
        
        for file_path in files:
            if not file_path.exists():
                continue
            
            with open(file_path, "r") as f:
                data = json.load(f)
            
            merged_data["sources"].append(str(file_path))
            merged_data["data"][file_path.stem] = data
        
        with open(output_file, "w") as f:
            json.dump(merged_data, f, indent=2)
        
        return True
    except Exception:
        return False


def clean_old_files(directory: Path, pattern: str = "*.json", days: int = 7) -> int:
    """Clean old files in directory."""
    from datetime import datetime, timedelta
    
    cutoff_date = datetime.now() - timedelta(days=days)
    deleted = 0
    
    for file_path in directory.glob(pattern):
        if file_path.is_file():
            file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
            if file_time < cutoff_date:
                file_path.unlink()
                deleted += 1
    
    return deleted


def print_colored(text: str, color: str = ""):
    """Print colored text."""
    colors = {
        "green": "\033[92m",
        "red": "\033[91m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "cyan": "\033[96m",
        "reset": "\033[0m"
    }
    
    color_code = colors.get(color.lower(), "")
    reset = colors["reset"]
    
    print(f"{color_code}{text}{reset}")


def print_success(text: str):
    """Print success message."""
    print_colored(f"✅ {text}", "green")


def print_error(text: str):
    """Print error message."""
    print_colored(f"❌ {text}", "red")


def print_warning(text: str):
    """Print warning message."""
    print_colored(f"⚠️  {text}", "yellow")


def print_info(text: str):
    """Print info message."""
    print_colored(f"ℹ️  {text}", "blue")



