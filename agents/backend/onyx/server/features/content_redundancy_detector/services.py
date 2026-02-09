"""
Enhanced Service functions for advanced content analysis
REFACTORED: This module now re-exports from the new modular services structure
for backward compatibility. New code should import directly from services.* modules.
"""

# Re-export all services from the new modular structure
from services.analysis import analyze_content
from services.similarity import detect_similarity
from services.quality import assess_quality
from services.ai_ml import (
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
from services.system import get_system_stats, get_health_status

# Maintain backward compatibility - export all functions
__all__ = [
    "analyze_content",
    "detect_similarity",
    "assess_quality",
    "get_system_stats",
    "get_health_status",
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
]
