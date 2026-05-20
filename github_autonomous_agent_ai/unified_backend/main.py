"""
Unified Backend - Main Application
Backend built around Unified AI Model as the core

Usage:
    uvicorn main:app --host 0.0.0.0 --port 8080
    
    Or:
    python main.py
"""

import os
import sys

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the Unified AI Model application
from unified_ai_model.main import app, create_app, get_app, run, __version__

# Re-export for easy access
__all__ = ["app", "create_app", "get_app", "run", "__version__"]


if __name__ == "__main__":
    run()
