"""
Models Module - Data Models and Schemas
======================================

Pydantic models for data validation and serialization.
"""

from .document import Document, DocumentMetadata, DocumentType
from .processing import ProcessingResult, ProcessingStatus, ProcessingConfig
from .ai import AIResponse, AIConfig, ModelInfo
from .errors import ErrorResult, ErrorType, ErrorSeverity
from .cache import CacheEntry, CacheStats
from .metrics import Metrics, PerformanceMetrics, SystemMetrics

__all__ = [
    # Document models
    "Document",
    "DocumentMetadata", 
    "DocumentType",
    
    # Processing models
    "ProcessingResult",
    "ProcessingStatus",
    "ProcessingConfig",
    
    # AI models
    "AIResponse",
    "AIConfig",
    "ModelInfo",
    
    # Error models
    "ErrorResult",
    "ErrorType",
    "ErrorSeverity",
    
    # Cache models
    "CacheEntry",
    "CacheStats",
    
    # Metrics models
    "Metrics",
    "PerformanceMetrics",
    "SystemMetrics",
]

















