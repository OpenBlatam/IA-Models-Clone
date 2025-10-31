"""
AI Document Processor - Refactored Architecture
==============================================

A modern, well-architected AI document processing system with clean separation of concerns.
"""

__version__ = "3.0.0"
__author__ = "AI Document Processor Team"
__description__ = "Ultra-fast AI document processing with modern architecture"

# Core modules
from .core import (
    DocumentProcessor,
    ProcessingEngine,
    CacheManager,
    PerformanceMonitor
)

# Services
from .services import (
    DocumentService,
    AIService,
    TransformService,
    ValidationService
)

# Models
from .models import (
    Document,
    ProcessingResult,
    ProcessingConfig,
    ErrorResult
)

# Utils
from .utils import (
    FileHandler,
    TextExtractor,
    FormatConverter,
    Validator
)

__all__ = [
    # Core
    "DocumentProcessor",
    "ProcessingEngine", 
    "CacheManager",
    "PerformanceMonitor",
    
    # Services
    "DocumentService",
    "AIService",
    "TransformService",
    "ValidationService",
    
    # Models
    "Document",
    "ProcessingResult",
    "ProcessingConfig",
    "ErrorResult",
    
    # Utils
    "FileHandler",
    "TextExtractor",
    "FormatConverter",
    "Validator",
]

















