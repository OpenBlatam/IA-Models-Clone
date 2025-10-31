"""
Core Domain Layer
================

This module contains the core business logic and domain models for the AI History Comparison system.

The core layer is independent of external frameworks and contains:
- Domain entities and value objects
- Business rules and domain services
- Domain events and specifications
- Core business logic interfaces
"""

from .domain import *
from .services import *
from .events import *
from .specifications import *

__all__ = [
    # Domain entities
    "HistoryEntry",
    "ComparisonResult",
    "TrendAnalysis", 
    "QualityReport",
    "ModelDefinition",
    "PerformanceMetric",
    "AnalysisJob",
    "UserFeedback",
    
    # Domain services
    "HistoryAnalysisService",
    "ModelComparisonService",
    "TrendAnalysisService",
    "QualityAssessmentService",
    "ContentAnalysisService",
    
    # Domain events
    "AnalysisCompletedEvent",
    "ModelComparisonEvent",
    "TrendDetectedEvent",
    "QualityAlertEvent",
    
    # Specifications
    "QualityThresholdSpecification",
    "TrendSignificanceSpecification",
    "ModelComparisonSpecification"
]




