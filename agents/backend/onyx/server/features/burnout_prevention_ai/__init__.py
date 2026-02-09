"""
Burnout Prevention AI
====================

AI-powered burnout prevention and wellness assistant.

This module provides a comprehensive system for:
- Burnout risk assessment
- Wellness checks
- Coping strategies
- Conversational AI chat
- Progress tracking
- Trend analysis
- Educational resources
- Personalized prevention plans

Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Blatam Academy"
__license__ = "Proprietary"
__description__ = "AI-powered burnout prevention and wellness assistant"

# Try to import components with error handling
try:
    from .services import BurnoutPreventionService, ContinuousBurnoutService
except ImportError:
    BurnoutPreventionService = None
    ContinuousBurnoutService = None

try:
    from .api import router
except ImportError:
    router = None

__all__ = [
    "BurnoutPreventionService",
    "ContinuousBurnoutService",
    "router",
]
