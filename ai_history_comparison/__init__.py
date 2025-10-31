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
__author__ = "AI History Comparison Team"
__email__ = "support@ai-history.com"

from .ai_history_analyzer import AIHistoryAnalyzer, ComparisonType, MetricType
from .config import get_config, Config
from .models import ModelUtils, ModelSerializer

__all__ = [
    "AIHistoryAnalyzer",
    "ComparisonType", 
    "MetricType",
    "get_config",
    "Config",
    "ModelUtils",
    "ModelSerializer"
]



























