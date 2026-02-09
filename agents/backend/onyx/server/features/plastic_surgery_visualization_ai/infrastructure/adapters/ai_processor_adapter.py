"""Adapter for AIProcessor to implement IAProcessor interface."""

from core.interfaces import IAProcessor
from core.services.ai_processor import AIProcessor

# Type alias for backward compatibility
AIProcessorAdapter = AIProcessor

