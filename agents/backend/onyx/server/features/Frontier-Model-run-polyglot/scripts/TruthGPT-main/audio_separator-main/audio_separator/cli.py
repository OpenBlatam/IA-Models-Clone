"""
Command-line interface for audio separator.

DEPRECATED: This module is kept for backward compatibility.
New code should use audio_separator.cli.main instead.
"""

from .cli.main import main, create_parser

__all__ = ["main", "create_parser"]
