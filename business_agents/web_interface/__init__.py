"""
Web Interface Module
====================

Web interface components for the Business Agents system.
"""

import os
from pathlib import Path

# Get the directory of this module
WEB_INTERFACE_DIR = Path(__file__).parent

def get_web_interface_path():
    """Get the path to the web interface directory."""
    return str(WEB_INTERFACE_DIR)

def get_static_files_path():
    """Get the path to the static files directory."""
    return str(WEB_INTERFACE_DIR / "static")

def get_index_html_path():
    """Get the path to the index.html file."""
    return str(WEB_INTERFACE_DIR / "index.html")





























