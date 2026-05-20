"""
Gradio interface for Music Analyzer AI
"""

from .music_analyzer_ui import (
    MusicAnalyzerGradioUI,
    create_gradio_app
)

__all__ = [
    "MusicAnalyzerGradioUI",
    "create_gradio_app",
]

