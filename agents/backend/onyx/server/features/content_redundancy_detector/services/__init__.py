"""
Services Module - Domain-specific service functions
Refactored from monolithic services.py into modular structure
"""

from .analysis import analyze_content
from .similarity import detect_similarity
from .quality import assess_quality
from .ai_ml import (
    analyze_sentiment,
    detect_language,
    extract_topics,
    calculate_semantic_similarity,
    detect_plagiarism,
    extract_entities,
    generate_summary,
    analyze_readability_advanced,
    comprehensive_analysis,
    batch_analyze_content
)
from .system import get_system_stats, get_health_status

__all__ = [
    "analyze_content",
    "detect_similarity",
    "assess_quality",
    "analyze_sentiment",
    "detect_language",
    "extract_topics",
    "calculate_semantic_similarity",
    "detect_plagiarism",
    "extract_entities",
    "generate_summary",
    "analyze_readability_advanced",
    "comprehensive_analysis",
    "batch_analyze_content",
    "get_system_stats",
    "get_health_status",
]






