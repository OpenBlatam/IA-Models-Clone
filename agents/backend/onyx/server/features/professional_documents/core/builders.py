"""
Builder utilities for creating complex objects.

Helper functions and builders for constructing document-related objects.
"""

from typing import Optional
from .models import (
    ProfessionalDocument,
    DocumentTemplate,
    DocumentStyle,
    DocumentType,
    DocumentSection
)
from .helpers import generate_document_title
from datetime import datetime


class DocumentBuilder:
    """Builder for creating ProfessionalDocument instances."""
    
    def __init__(self):
        self._title: Optional[str] = None
        self._subtitle: Optional[str] = None
        self._document_type: Optional[DocumentType] = None
        self._template_id: Optional[str] = None
        self._author: Optional[str] = None
        self._company: Optional[str] = None
        self._style: Optional[DocumentStyle] = None
        self._sections: list[DocumentSection] = []
        self._query: Optional[str] = None
    
    def with_title(self, title: str) -> "DocumentBuilder":
        """Set document title."""
        self._title = title
        return self
    
    def with_subtitle(self, subtitle: str) -> "DocumentBuilder":
        """Set document subtitle."""
        self._subtitle = subtitle
        return self
    
    def with_type(self, document_type: DocumentType) -> "DocumentBuilder":
        """Set document type."""
        self._document_type = document_type
        return self
    
    def with_template_id(self, template_id: str) -> "DocumentBuilder":
        """Set template ID."""
        self._template_id = template_id
        return self
    
    def with_author(self, author: str) -> "DocumentBuilder":
        """Set document author."""
        self._author = author
        return self
    
    def with_company(self, company: str) -> "DocumentBuilder":
        """Set company name."""
        self._company = company
        return self
    
    def with_style(self, style: DocumentStyle) -> "DocumentBuilder":
        """Set document style."""
        self._style = style
        return self
    
    def with_sections(self, sections: list[DocumentSection]) -> "DocumentBuilder":
        """Set document sections."""
        self._sections = sections
        return self
    
    def with_query(self, query: str) -> "DocumentBuilder":
        """Set query for title generation."""
        self._query = query
        return self
    
    def build(self, template: DocumentTemplate) -> ProfessionalDocument:
        """
        Build the ProfessionalDocument instance.
        
        Args:
            template: Document template to use
            
        Returns:
            Constructed ProfessionalDocument
        """
        if not self._document_type:
            self._document_type = template.document_type
        
        title = self._title
        if not title and self._query:
            title = generate_document_title(self._query, self._document_type)
        
        return ProfessionalDocument(
            title=title or "Untitled Document",
            subtitle=self._subtitle,
            document_type=self._document_type,
            template_id=self._template_id or template.id,
            author=self._author,
            company=self._company,
            style=self._style or template.style,
            sections=self._sections,
            status="draft"
        )


def create_document_builder() -> DocumentBuilder:
    """Create a new DocumentBuilder instance."""
    return DocumentBuilder()






