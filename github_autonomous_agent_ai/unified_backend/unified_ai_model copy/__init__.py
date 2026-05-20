"""
Unified AI Model - Combined best practices from autonomous agents, bulk processing, and LLM services.

This module provides a unified interface for AI/LLM functionality with:
- OpenRouter integration for multiple AI models (DeepSeek, GPT-4, Claude, etc.)
- Streaming and parallel generation
- Intelligent caching with circuit breaker
- Performance monitoring and analytics
- Chat functionality with conversation memory
- Rate limiting and resilience patterns
"""

__version__ = "1.0.0"

from .config import UnifiedAIConfig

# Lazy imports to avoid circular import issues
def create_app(*args, **kwargs):
    from .main import create_app as _create_app
    return _create_app(*args, **kwargs)

def get_app():
    from .main import get_app as _get_app
    return _get_app()

__all__ = ["UnifiedAIConfig", "create_app", "get_app", "__version__"]



