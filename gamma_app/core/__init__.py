"""
Gamma App - Core Module
AI-Powered Content Generation System

This module provides the core functionality for generating presentations,
documents, and web pages using advanced AI models.
"""

from .content_generator import ContentGenerator
from .design_engine import DesignEngine
from .collaboration_engine import CollaborationEngine

__all__ = [
    'ContentGenerator',
    'DesignEngine',
    'CollaborationEngine'
]

__version__ = "1.0.0"
