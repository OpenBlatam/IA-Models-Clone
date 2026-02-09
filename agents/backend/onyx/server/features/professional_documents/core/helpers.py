"""
Helper functions for document processing.

Common utility functions used across the professional documents module.
"""

import re
from typing import List, Dict, Any, Optional
from .models import DocumentSection, DocumentTemplate, DocumentType
from .constants import DOCUMENT_TYPE_SUFFIXES
from .types import SectionData


def sanitize_title(title: str) -> str:
    """
    Sanitize document title for use in filenames.
    
    Removes or replaces characters that are not safe for filenames.
    
    Args:
        title: The title to sanitize
        
    Returns:
        A sanitized string safe for use in filenames
    """
    # Remove or replace unsafe characters
    sanitized = re.sub(r'[<>:"|?*\x00-\x1f]', '', title)
    # Replace spaces and path separators with underscores
    sanitized = re.sub(r'[\s/\\]+', '_', sanitized)
    # Remove multiple consecutive underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    # Remove leading/trailing underscores and dots
    sanitized = sanitized.strip('_.')
    # Ensure it's not empty
    return sanitized if sanitized else "untitled"


def calculate_word_count(content: str) -> int:
    """
    Calculate word count from content string.
    
    Args:
        content: The text content to count words in
        
    Returns:
        The number of words in the content
    """
    if not content or not content.strip():
        return 0
    return len(content.split())


def process_sections_data(
    sections: List[SectionData],
    template: DocumentTemplate
) -> List[DocumentSection]:
    """
    Process raw section data into DocumentSection objects.
    
    Args:
        sections: List of section dictionaries from AI response
        template: Document template for reference
        
    Returns:
        List of DocumentSection objects
    """
    return [
        DocumentSection(
            title=section_data.get("title", f"Section {i+1}"),
            content=section_data.get("content", ""),
            level=section_data.get("level", 1),
            order=i,
            metadata=section_data.get("metadata", {})
        )
        for i, section_data in enumerate(sections)
    ]


def generate_document_title(query: str, document_type: DocumentType, max_words: int = 5) -> str:
    """Generate a document title from query and document type."""
    query_words = query.strip().split()[:max_words]
    if not query_words:
        base_title = "Untitled"
    else:
        base_title = " ".join(query_words).title()
    
    suffix = DOCUMENT_TYPE_SUFFIXES.get(document_type.value, "Document")
    return f"{base_title} - {suffix}"

