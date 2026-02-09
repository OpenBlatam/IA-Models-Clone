"""
Professional Document Services
==============================

Core services for document generation, processing, and export functionality.
"""

import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from contextlib import contextmanager

from .models import (
    DocumentGenerationRequest,
    DocumentGenerationResponse,
    DocumentExportRequest,
    DocumentExportResponse,
    DocumentTemplate,
    ProfessionalDocument,
    DocumentSection,
    DocumentStyle,
    ExportFormat,
    DocumentType
)
from .templates import template_manager
from .ai_service import AIDocumentGenerator
from .storage import DocumentStorage, InMemoryDocumentStorage
from .constants import WORDS_PER_PAGE, DOCUMENT_TYPE_SUFFIXES, EXPORT_FORMAT_EXTENSIONS, DEFAULT_EXPORT_DIR
from .exceptions import DocumentNotFoundError, DocumentGenerationError, DocumentExportError
from .exporter_factory import ExporterFactory
from .helpers import sanitize_title, calculate_word_count, process_sections_data, generate_document_title
from .formatters import format_file_size, format_duration
from .logging_utils import log_document_operation, log_performance, log_error_with_context

logger = logging.getLogger(__name__)


class DocumentGenerationService:
    """Service for generating professional documents using AI."""
    
    def __init__(self, storage: Optional[DocumentStorage] = None):
        self.ai_generator = AIDocumentGenerator()
        self.storage = storage or InMemoryDocumentStorage()
    
    async def generate_document(self, request: DocumentGenerationRequest) -> DocumentGenerationResponse:
        """Generate a professional document based on the request."""
        start_time = time.time()
        log_document_operation(
            "generate_document",
            document_type=request.document_type.value,
            query_length=len(request.query)
        )
        
        try:
            # Get template
            if request.template_id:
                template = template_manager.get_template(request.template_id)
            else:
                template = template_manager.get_default_template(request.document_type)
            
            # Create document structure
            document = ProfessionalDocument(
                title=request.title or generate_document_title(request.query, request.document_type),
                subtitle=request.subtitle,
                document_type=request.document_type,
                template_id=template.id,
                author=request.author,
                company=request.company,
                style=request.style or template.style,
                status="generating"
            )
            
            # Generate content using AI
            sections = await self.ai_generator.generate_document_content(
                query=request.query,
                template=template,
                document_type=request.document_type,
                tone=request.tone,
                length=request.length,
                language=request.language,
                additional_requirements=request.additional_requirements
            )
            
            # Process and structure sections
            document.sections = process_sections_data(sections, template)
            document.word_count = sum(calculate_word_count(section.content) for section in document.sections)
            document.page_count = self._estimate_page_count(document.word_count, document.style)
            document.status = "completed"
            document.date_modified = datetime.utcnow()
            
            # Store document
            self.storage.save(document)
            
            generation_time = time.time() - start_time
            log_performance(
                "document_generation",
                generation_time,
                word_count=document.word_count,
                page_count=document.page_count
            )
            
            return DocumentGenerationResponse(
                success=True,
                document=document,
                message="Document generated successfully",
                generation_time=generation_time,
                word_count=document.word_count,
                estimated_pages=document.page_count
            )
            
        except DocumentGenerationError:
            raise
        except Exception as e:
            generation_time = time.time() - start_time
            log_error_with_context(
                e,
                "generate_document",
                document_type=request.document_type.value,
                generation_time=generation_time
            )
            raise DocumentGenerationError(f"Failed to generate document: {str(e)}") from e
    
    def _estimate_page_count(self, word_count: int, style: DocumentStyle) -> int:
        """
        Estimate page count based on word count and styling.
        
        Args:
            word_count: Total number of words in the document
            style: Document style configuration
            
        Returns:
            Estimated number of pages (minimum 1)
        """
        if word_count <= 0:
            return 1
        
        # Adjust for line spacing if different from default
        spacing_factor = style.line_spacing / 1.5  # Default is 1.5
        adjusted_words = int(word_count / spacing_factor)
        
        return max(1, adjusted_words // WORDS_PER_PAGE)
    
    def get_document(self, document_id: str) -> Optional[ProfessionalDocument]:
        """Get a document by ID."""
        return self.storage.get(document_id)
    
    def list_documents(
        self, 
        limit: int = 50, 
        offset: int = 0,
        document_type: Optional[DocumentType] = None
    ) -> List[ProfessionalDocument]:
        """
        List documents with pagination and optional filtering.
        
        Args:
            limit: Maximum number of documents to return
            offset: Number of documents to skip
            document_type: Optional filter by document type
            
        Returns:
            List of documents matching the criteria
        """
        documents = self.storage.list_all()
        
        # Filter by document type if specified
        if document_type:
            documents = [doc for doc in documents if doc.document_type == document_type]
        
        # Sort by creation date (newest first)
        documents.sort(key=lambda x: x.date_created, reverse=True)
        
        # Apply pagination
        return documents[offset:offset + limit]
    
    def count_documents(self, document_type: Optional[DocumentType] = None) -> int:
        """
        Count documents with optional filtering.
        
        Args:
            document_type: Optional filter by document type
            
        Returns:
            Total number of documents matching the criteria
        """
        documents = self.storage.list_all()
        
        if document_type:
            return sum(1 for doc in documents if doc.document_type == document_type)
        
        return len(documents)
    
    def update_document(self, document_id: str, updates: Dict[str, Any]) -> ProfessionalDocument:
        """
        Update a document with provided updates.
        
        Args:
            document_id: The ID of the document to update
            updates: Dictionary of field names and values to update
            
        Returns:
            The updated document
            
        Raises:
            DocumentNotFoundError: If the document doesn't exist
        """
        document = self.storage.get(document_id)
        if not document:
            raise DocumentNotFoundError(f"Document with ID {document_id} not found")
        
        allowed_fields = {
            "title", "subtitle", "sections", "style", 
            "metadata", "author", "company", "status"
        }
        
        for key, value in updates.items():
            if key in allowed_fields and hasattr(document, key):
                setattr(document, key, value)
        
        document.date_modified = datetime.utcnow()
        self.storage.save(document)
        return document


class DocumentExportService:
    """Service for exporting documents in various formats."""
    
    def __init__(self, output_dir: str = DEFAULT_EXPORT_DIR):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    async def export_document(
        self, 
        document: ProfessionalDocument, 
        request: DocumentExportRequest
    ) -> DocumentExportResponse:
        """Export a document in the specified format."""
        start_time = time.time()
        
        try:
            exporter = self._get_exporter(request.format)
            filename = self._generate_filename(document, request)
            file_path = self.output_dir / filename
            
            await exporter.export(document, request, file_path)
            
            file_size = os.path.getsize(file_path) if file_path.exists() else 0
            export_time = time.time() - start_time
            
            log_performance(
                "document_export",
                export_time,
                format=request.format.value,
                file_size=format_file_size(file_size)
            )
            
            return DocumentExportResponse(
                success=True,
                file_path=str(file_path),
                file_size=file_size,
                download_url=f"/download/{file_path.name}",
                message="Document exported successfully",
                export_time=export_time
            )
            
        except DocumentExportError:
            raise
        except Exception as e:
            logger.error(f"Error exporting document: {str(e)}")
            export_time = time.time() - start_time
            raise DocumentExportError(f"Failed to export document: {str(e)}") from e
    
    def _generate_filename(self, document: ProfessionalDocument, request: DocumentExportRequest) -> str:
        """Generate filename for export."""
        if request.custom_filename:
            return request.custom_filename
        
        safe_title = sanitize_title(document.title)
        extension = EXPORT_FORMAT_EXTENSIONS.get(request.format.value, "txt")
        return f"{safe_title}.{extension}"
    
    def _get_exporter(self, format: ExportFormat) -> DocumentExporter:
        """Get exporter for the specified format."""
        return ExporterFactory.create_exporter(format)
    


class TemplateService:
    """Service for managing document templates."""
    
    def __init__(self):
        self.template_manager = template_manager
    
    def get_template(self, template_id: str) -> DocumentTemplate:
        """Get a template by ID."""
        return self.template_manager.get_template(template_id)
    
    def get_templates_by_type(self, document_type: DocumentType) -> List[DocumentTemplate]:
        """Get templates by document type."""
        return self.template_manager.get_templates_by_type(document_type)
    
    def get_all_templates(self) -> List[DocumentTemplate]:
        """Get all available templates."""
        return self.template_manager.get_all_templates()
    
    def get_default_template(self, document_type: DocumentType) -> DocumentTemplate:
        """Get default template for document type."""
        return self.template_manager.get_default_template(document_type)
    
    def add_custom_template(self, template: DocumentTemplate) -> None:
        """Add a custom template."""
        self.template_manager.add_custom_template(template)
    
    def remove_template(self, template_id: str) -> None:
        """Remove a template."""
        self.template_manager.remove_template(template_id)














