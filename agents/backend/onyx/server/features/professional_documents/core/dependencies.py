"""
Dependency injection for professional documents module.

Centralized service instances and dependency providers.
"""

from functools import lru_cache
from .services import DocumentGenerationService, DocumentExportService, TemplateService
from .storage import InMemoryDocumentStorage


@lru_cache()
def get_document_generation_service() -> DocumentGenerationService:
    """Get or create document generation service instance."""
    return DocumentGenerationService(storage=InMemoryDocumentStorage())


@lru_cache()
def get_document_export_service() -> DocumentExportService:
    """Get or create document export service instance."""
    return DocumentExportService()


@lru_cache()
def get_template_service() -> TemplateService:
    """Get or create template service instance."""
    return TemplateService()






