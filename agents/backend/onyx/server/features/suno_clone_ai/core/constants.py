"""
Application constants

Centralized constants for the Suno Clone AI application.
"""

from typing import Dict, List

# API Endpoints
API_ENDPOINTS: Dict[str, str] = {
    "songs": "/suno/songs",
    "generate": "/suno/generate",
    "search": "/suno/search/songs",
    "websocket": "/suno/ws",
    "batch": "/suno/batch/generate",
    "health": "/suno/health",
    "health_detailed": "/suno/health/detailed",
    "metrics": "/metrics",
    "versions": "/suno/versions",
    "transcription": "/suno/transcription",
    "sentiment": "/suno/sentiment",
    "lyrics": "/suno/lyrics",
    "distributed": "/suno/distributed",
    "scaling": "/suno/scaling",
    "streaming": "/suno/streaming",
    "audio_analysis": "/suno/audio-analysis",
    "remix": "/suno/remix",
    "karaoke": "/suno/karaoke",
    "collaboration": "/suno/collaboration",
    "marketplace": "/suno/marketplace",
    "monetization": "/suno/monetization",
    "auto_dj": "/suno/auto-dj",
    "trends": "/suno/trends"
}

# Audio Configuration
DEFAULT_SAMPLE_RATE: int = 32000
DEFAULT_DURATION: int = 30
MAX_AUDIO_LENGTH: int = 300
SUPPORTED_AUDIO_FORMATS: List[str] = [".wav", ".mp3", ".flac", ".ogg", ".m4a"]

# Generation Parameters
DEFAULT_TOP_K: int = 250
DEFAULT_TOP_P: float = 0.0
DEFAULT_TEMPERATURE: float = 1.0
DEFAULT_CFG_COEF: float = 3.0

# Cache Configuration
DEFAULT_CACHE_TTL: int = 3600
DEFAULT_RESPONSE_CACHE_TTL: int = 300

# Rate Limiting
DEFAULT_RATE_LIMIT_REQUESTS: int = 50
DEFAULT_RATE_LIMIT_WINDOW: int = 60

# File Limits
DEFAULT_MAX_FILE_SIZE_MB: int = 50







