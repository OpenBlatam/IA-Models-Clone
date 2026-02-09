"""Development utilities."""

import os
import sys
from pathlib import Path
from typing import Optional


def is_development() -> bool:
    """Check if running in development mode."""
    return os.getenv("ENVIRONMENT", "development").lower() == "development"


def is_production() -> bool:
    """Check if running in production mode."""
    return os.getenv("ENVIRONMENT", "development").lower() == "production"


def get_project_root() -> Path:
    """Get project root directory."""
    current = Path(__file__).resolve()
    
    # Look for markers of project root
    markers = [".git", "pyproject.toml", "setup.py", "requirements.txt"]
    
    for parent in current.parents:
        if any((parent / marker).exists() for marker in markers):
            return parent
    
    # Fallback to current directory
    return current.parent


def setup_dev_environment():
    """Setup development environment."""
    if not is_development():
        return
    
    # Add project root to path
    project_root = get_project_root()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # Enable debug mode
    os.environ.setdefault("DEBUG", "1")
    os.environ.setdefault("LOG_LEVEL", "DEBUG")


def print_debug_info():
    """Print debug information."""
    if not is_development():
        return
    
    print("=" * 50)
    print("Development Mode")
    print("=" * 50)
    print(f"Python: {sys.version}")
    print(f"Project Root: {get_project_root()}")
    print(f"Environment: {os.getenv('ENVIRONMENT', 'development')}")
    print("=" * 50)

