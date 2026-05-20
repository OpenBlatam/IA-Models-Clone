"""
Export IA - Refactored AI Document Processing System
==================================================

A modular, professional-grade document export system with AI-powered quality enhancement.
"""

__version__ = "2.0.0"
__author__ = "Blatam Academy"
__license__ = "Proprietary"
__description__ = "AI-powered document export system with quality enhancement"

# Try to import components with error handling
try:
    from .core.engine import ExportIAEngine
except ImportError:
    ExportIAEngine = None

try:
    from .core.config import ExportConfig, QualityLevel, ExportFormat, DocumentType
except ImportError:
    ExportConfig = None
    QualityLevel = None
    ExportFormat = None
    DocumentType = None

try:
    from .core.models import ExportTask, ExportResult
except ImportError:
    ExportTask = None
    ExportResult = None

__all__ = [
    "ExportIAEngine",
    "ExportConfig", 
    "QualityLevel",
    "ExportFormat",
    "DocumentType",
    "ExportTask",
    "ExportResult"
]

