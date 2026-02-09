"""
Social Video Transcriber AI
===========================

A powerful AI-powered tool for transcribing videos from TikTok, Instagram, and YouTube.

Features:
- Video transcription with optional timestamps
- AI-powered framework and structure analysis
- Text variant generation maintaining context, length, and structure
- OpenRouter integration for AI capabilities
"""

__version__ = "1.0.0"
__author__ = "Blatam Academy"
__license__ = "Proprietary"
__description__ = "AI-powered video transcription tool for TikTok, Instagram, and YouTube with structure analysis and variant generation"

# Try to import components with error handling
try:
    from .core.models import (
        TranscriptionRequest,
        TranscriptionResponse,
        VariantRequest,
        VariantResponse,
        AnalysisResponse,
        SupportedPlatform,
        TranscriptionStatus,
    )
except ImportError:
    TranscriptionRequest = None
    TranscriptionResponse = None
    VariantRequest = None
    VariantResponse = None
    AnalysisResponse = None
    SupportedPlatform = None
    TranscriptionStatus = None

__all__ = [
    "TranscriptionRequest",
    "TranscriptionResponse",
    "VariantRequest",
    "VariantResponse",
    "AnalysisResponse",
    "SupportedPlatform",
    "TranscriptionStatus",
]
