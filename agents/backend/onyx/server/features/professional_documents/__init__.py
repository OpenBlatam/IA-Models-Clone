"""
Professional Document Generation System
======================================

This module provides a comprehensive document generation system that creates
professional documents based on user queries and exports them in multiple formats
(PDF, MD, Word) with high-quality formatting and styling.

Features:
- AI-powered content generation
- Professional document templates
- Multiple export formats (PDF, MD, Word)
- Customizable styling and branding
- Document management and history
"""

__version__ = "1.0.0"
__author__ = "Blatam Academy"
__license__ = "Proprietary"
__description__ = "AI-powered professional document generation system with multiple export formats"

# Try to import components with error handling
try:
    from .core.models import (
        DocumentGenerationRequest,
        DocumentGenerationResponse,
        DocumentExportRequest,
        DocumentExportResponse,
        DocumentTemplate,
        DocumentStyle,
        ProfessionalDocument
    )
except ImportError:
    DocumentGenerationRequest = None
    DocumentGenerationResponse = None
    DocumentExportRequest = None
    DocumentExportResponse = None
    DocumentTemplate = None
    DocumentStyle = None
    ProfessionalDocument = None

try:
    from .core.services import (
        DocumentGenerationService,
        DocumentExportService,
        TemplateService
    )
except ImportError:
    DocumentGenerationService = None
    DocumentExportService = None
    TemplateService = None

try:
    from .core.api import router as professional_documents_router
except ImportError:
    professional_documents_router = None

__all__ = [
    "DocumentGenerationRequest",
    "DocumentGenerationResponse", 
    "DocumentExportRequest",
    "DocumentExportResponse",
    "DocumentTemplate",
    "DocumentStyle",
    "ProfessionalDocument",
    "DocumentGenerationService",
    "DocumentExportService",
    "TemplateService",
    "professional_documents_router"
]






