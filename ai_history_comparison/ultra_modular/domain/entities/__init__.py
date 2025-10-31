"""
Domain Entities
==============

This module contains the core domain entities, each in its own focused module
following the Single Responsibility Principle.
"""

from .history_entry import HistoryEntry
from .comparison_result import ComparisonResult
from .trend_analysis import TrendAnalysis
from .quality_report import QualityReport
from .analysis_job import AnalysisJob
from .user_feedback import UserFeedback

__all__ = [
    "HistoryEntry",
    "ComparisonResult",
    "TrendAnalysis", 
    "QualityReport",
    "AnalysisJob",
    "UserFeedback"
]




