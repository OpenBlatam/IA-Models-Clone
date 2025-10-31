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

from .models import (
    DocumentGenerationRequest,
    DocumentGenerationResponse,
    DocumentExportRequest,
    DocumentExportResponse,
    DocumentTemplate,
    DocumentStyle,
    ProfessionalDocument
)

from .services import (
    DocumentGenerationService,
    DocumentExportService,
    TemplateService
)

from .api import router as professional_documents_router

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




























