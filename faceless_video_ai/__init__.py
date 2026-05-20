"""
Faceless Video AI - Sistema de generación de videos sin rostro con IA
======================================================================

Sistema completo para generar videos completamente con IA a partir de scripts,
incluyendo generación de imágenes, animación, subtítulos y audio.

Versión: 1.0.0
Autor: Blatam Academy
Licencia: Propietaria
"""

__version__ = "1.0.0"
__author__ = "Blatam Academy"
__license__ = "Proprietary"
__description__ = "AI system for generating faceless videos from scripts"

from .core.models import (
    VideoScript,
    VideoGenerationRequest,
    VideoGenerationResponse,
    SubtitleConfig,
    VideoConfig,
    AudioConfig,
)

from .services.script_processor import ScriptProcessor
from .services.video_generator import VideoGenerator
from .services.subtitle_generator import SubtitleGenerator
from .services.audio_generator import AudioGenerator

__all__ = [
    "VideoScript",
    "VideoGenerationRequest",
    "VideoGenerationResponse",
    "SubtitleConfig",
    "VideoConfig",
    "AudioConfig",
    "ScriptProcessor",
    "VideoGenerator",
    "SubtitleGenerator",
    "AudioGenerator",
]

