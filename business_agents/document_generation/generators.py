"""
Document Generators
==================

Main document generation engine and orchestrator.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from uuid import uuid4

from .types import (
    DocumentRequest, GeneratedDocument, DocumentGenerationResult,
    DocumentType, DocumentFormat, DocumentStatus
)
from .templates import TemplateManager, BuiltinTemplates
from .formatters import FormatterFactory, BaseFormatter

logger = logging.getLogger(__name__)

class DocumentGenerator:
    """Main document generation engine."""
    
    def __init__(self, templates_dir: str = "templates", output_dir: str = "generated_documents"):
        self.template_manager = TemplateManager(templates_dir)
        self.output_dir = output_dir
        self.generation_history: Dict[str, GeneratedDocument] = {}
    
    async def initialize(self):
        """Initialize the document generator."""
        try:
            await self.template_manager.load_templates()
            
            # Load built-in templates if none exist
            if not self.template_manager.list_templates():
                await self._load_builtin_templates()
            
            logger.info("Document generator initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize document generator: {str(e)}")
            raise
    
    async def _load_builtin_templates(self):
        """Load built-in templates."""
        try:
            builtin_templates = [
                BuiltinTemplates.get_business_plan_template(),
                BuiltinTemplates.get_marketing_report_template(),
                BuiltinTemplates.get_technical_spec_template()
            ]
            
            for template in builtin_templates:
                await self.template_manager.create_template(template)
            
            logger.info(f"Loaded {len(builtin_templates)} built-in templates")
            
        except Exception as e:
            logger.error(f"Failed to load built-in templates: {str(e)}")
    
    async def generate_document(
        self, 
        request: DocumentRequest,
        template_id: Optional[str] = None
    ) -> DocumentGenerationResult:
        """Generate a document from a request."""
        start_time = datetime.now()
        
        try:
            # Get template
            template = await self._get_template(request, template_id)
            if not template:
                return DocumentGenerationResult(
                    success=False,
                    error=f"No template found for document type: {request.document_type}"
                )
            
            # Get formatter
            formatter = FormatterFactory.get_formatter(request.format, self.output_dir)
            
            # Generate filename
            filename = self._generate_filename(request.title, request.format)
            
            # Format document
            document = await formatter.format_document(template, request.variables, filename)
            
            # Set request ID
            document.request_id = request.request_id
            
            # Store in history
            self.generation_history[document.document_id] = document
            
            # Calculate generation time
            generation_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Generated document: {document.document_id} in {generation_time:.2f}s")
            
            return DocumentGenerationResult(
                success=True,
                document=document,
                generation_time=generation_time
            )
            
        except Exception as e:
            logger.error(f"Failed to generate document: {str(e)}")
            return DocumentGenerationResult(
                success=False,
                error=str(e),
                generation_time=(datetime.now() - start_time).total_seconds()
            )
    
    async def _get_template(
        self, 
        request: DocumentRequest, 
        template_id: Optional[str] = None
    ) -> Optional[Any]:
        """Get template for the request."""
        if template_id:
            return self.template_manager.get_template(template_id)
        
        # Try to find default template
        return self.template_manager.get_default_template(
            request.document_type, 
            request.format
        )
    
    def _generate_filename(self, title: str, format: DocumentFormat) -> str:
        """Generate a filename from title and format."""
        # Clean title for filename
        import re
        clean_title = re.sub(r'[^\w\s-]', '', title)
        clean_title = re.sub(r'[-\s]+', '-', clean_title)
        clean_title = clean_title.strip('-').lower()
        
        # Add timestamp for uniqueness
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        return f"{clean_title}_{timestamp}"
    
    async def get_document(self, document_id: str) -> Optional[GeneratedDocument]:
        """Get a generated document by ID."""
        return self.generation_history.get(document_id)
    
    async def list_documents(
        self, 
        document_type: Optional[DocumentType] = None,
        format: Optional[DocumentFormat] = None,
        status: Optional[DocumentStatus] = None
    ) -> List[GeneratedDocument]:
        """List generated documents with optional filters."""
        documents = list(self.generation_history.values())
        
        if document_type:
            documents = [d for d in documents if d.metadata.get("document_type") == document_type.value]
        
        if format:
            documents = [d for d in documents if d.format == format]
        
        if status:
            documents = [d for d in documents if d.status == status]
        
        return sorted(documents, key=lambda x: x.created_at, reverse=True)
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete a generated document."""
        try:
            document = self.generation_history.get(document_id)
            if not document:
                return False
            
            # Delete file if it exists
            if document.file_path:
                import os
                try:
                    os.remove(document.file_path)
                except OSError:
                    pass  # File might already be deleted
            
            # Remove from history
            del self.generation_history[document_id]
            
            logger.info(f"Deleted document: {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {str(e)}")
            return False
    
    async def get_templates(self) -> List[Any]:
        """Get all available templates."""
        return self.template_manager.list_templates()
    
    async def get_templates_by_type(self, document_type: DocumentType) -> List[Any]:
        """Get templates for a specific document type."""
        return self.template_manager.get_templates_by_type(document_type)
    
    async def create_template(self, template: Any) -> bool:
        """Create a new template."""
        return await self.template_manager.create_template(template)
    
    async def update_template(self, template_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing template."""
        return await self.template_manager.update_template(template_id, updates)
    
    async def delete_template(self, template_id: str) -> bool:
        """Delete a template."""
        return await self.template_manager.delete_template(template_id)
    
    def get_supported_formats(self) -> List[DocumentFormat]:
        """Get list of supported document formats."""
        return FormatterFactory.get_supported_formats()
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get document generation statistics."""
        total_documents = len(self.generation_history)
        
        if total_documents == 0:
            return {
                "total_documents": 0,
                "formats": {},
                "document_types": {},
                "status_counts": {}
            }
        
        # Count by format
        format_counts = {}
        for doc in self.generation_history.values():
            format_name = doc.format.value
            format_counts[format_name] = format_counts.get(format_name, 0) + 1
        
        # Count by document type
        type_counts = {}
        for doc in self.generation_history.values():
            doc_type = doc.metadata.get("document_type", "unknown")
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
        
        # Count by status
        status_counts = {}
        for doc in self.generation_history.values():
            status_name = doc.status.value
            status_counts[status_name] = status_counts.get(status_name, 0) + 1
        
        return {
            "total_documents": total_documents,
            "formats": format_counts,
            "document_types": type_counts,
            "status_counts": status_counts
        }

class DocumentGenerationService:
    """Service layer for document generation."""
    
    def __init__(self, generator: DocumentGenerator):
        self.generator = generator
    
    async def generate_business_document(
        self,
        document_type: DocumentType,
        title: str,
        description: str,
        business_area: str,
        created_by: str,
        variables: Dict[str, Any],
        format: DocumentFormat = DocumentFormat.MARKDOWN
    ) -> DocumentGenerationResult:
        """Generate a business document."""
        
        request = DocumentRequest(
            request_id=str(uuid4()),
            document_type=document_type,
            title=title,
            description=description,
            business_area=business_area,
            variables=variables,
            format=format,
            created_by=created_by
        )
        
        return await self.generator.generate_document(request)
    
    async def list_generated_documents(
        self,
        business_area: Optional[str] = None,
        document_type: Optional[DocumentType] = None,
        created_by: Optional[str] = None
    ) -> List[GeneratedDocument]:
        """List generated documents with filters."""
        
        documents = await self.generator.list_documents(document_type=document_type)
        
        # Apply additional filters
        if business_area:
            documents = [
                d for d in documents 
                if d.metadata.get("business_area") == business_area
            ]
        
        if created_by:
            documents = [
                d for d in documents 
                if d.metadata.get("created_by") == created_by
            ]
        
        return documents
    
    async def get_document_by_id(self, document_id: str) -> Optional[GeneratedDocument]:
        """Get a document by ID."""
        return await self.generator.get_document(document_id)
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete a document."""
        return await self.generator.delete_document(document_id)
