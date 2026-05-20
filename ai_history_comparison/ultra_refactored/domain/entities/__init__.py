"""
Domain Entities
==============

Core business entities with single responsibilities.
"""

from .history_entry import HistoryEntry
from .comparison_result import ComparisonResult
from .analysis_job import AnalysisJob
from .quality_report import QualityReport

__all__ = [
    "HistoryEntry",
    "ComparisonResult",
    "AnalysisJob",
    "QualityReport"
]




