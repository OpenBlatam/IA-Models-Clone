"""
AI History Comparison System

A comprehensive system for analyzing, comparing, and tracking AI model outputs over time.

This package provides:
- Content analysis and quality metrics
- Historical comparison capabilities
- Trend analysis and predictions
- Quality reporting and insights
- Content clustering and pattern recognition
- Bulk processing capabilities

Main Components:
- ai_history_analyzer: Core analysis engine
- api_endpoints: REST API endpoints
- config: Configuration management
- models: Database models
- main: Application entry point
"""

__version__ = "1.0.0"
__author__ = "Blatam Academy"
__license__ = "Proprietary"
__description__ = "Comprehensive system for analyzing and comparing AI model outputs over time"

from .ai_history_analyzer import AIHistoryAnalyzer, ComparisonType, MetricType
from .config import get_config, Config
from .models import ModelUtils, ModelSerializer

# Try to import components with error handling
try:
    from .ai_history_analyzer import AIHistoryAnalyzer, ComparisonType, MetricType
except ImportError:
    AIHistoryAnalyzer = None
    ComparisonType = None
    MetricType = None

try:
    from .config import get_config, Config
except ImportError:
    get_config = None
    Config = None

try:
    from .models import ModelUtils, ModelSerializer
except ImportError:
    ModelUtils = None
    ModelSerializer = None

__all__ = [
    "AIHistoryAnalyzer",
    "ComparisonType", 
    "MetricType",
    "get_config",
    "Config",
    "ModelUtils",
    "ModelSerializer"
]

















