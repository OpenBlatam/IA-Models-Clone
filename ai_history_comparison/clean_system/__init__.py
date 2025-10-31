"""
Clean AI History Comparison System
=================================

Sistema simple, limpio y funcional para comparar historial de IA.
Solo lo esencial, sin complejidad innecesaria.
"""

__version__ = "1.0.0"
__author__ = "AI History Team"

# Core
from .models import HistoryEntry, ComparisonResult, AnalysisJob
from .services import ContentAnalyzer, ModelComparator, QualityAssessor
from .repositories import HistoryRepository, ComparisonRepository
from .api import create_app

__all__ = [
    "HistoryEntry",
    "ComparisonResult", 
    "AnalysisJob",
    "ContentAnalyzer",
    "ModelComparator",
    "QualityAssessor",
    "HistoryRepository",
    "ComparisonRepository",
    "create_app"
]




