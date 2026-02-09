"""
Helper functions for document export formatting.

Common utilities for formatting document metadata and content across different export formats.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from .models import ProfessionalDocument, DocumentSection


def format_date(date: datetime, format_str: str = "%B %d, %Y") -> str:
    """Format a datetime object to a string."""
    return date.strftime(format_str)


def get_document_metadata(document: ProfessionalDocument) -> Dict[str, Optional[str]]:
    """Extract metadata from a document."""
    return {
        "title": document.title,
        "subtitle": document.subtitle,
        "author": document.author,
        "company": document.company,
        "date": format_date(document.date_created)
    }


def format_section_heading(section: DocumentSection, base_level: int = 1) -> int:
    """Calculate the heading level for a section."""
    return min(section.level + base_level, 6)


def has_metadata(document: ProfessionalDocument) -> bool:
    """Check if document has any metadata to display."""
    return bool(document.author or document.company)


def build_markdown_metadata_lines(document: ProfessionalDocument) -> List[str]:
    """Build markdown-formatted metadata lines."""
    lines = []
    if document.author:
        lines.append(f"**Author:** {document.author}")
    if document.company:
        lines.append(f"**Company:** {document.company}")
    lines.append(f"**Date:** {format_date(document.date_created)}")
    return lines


def build_text_metadata_lines(document: ProfessionalDocument) -> List[str]:
    """Build plain text metadata lines."""
    lines = []
    if document.author:
        lines.append(f"Author: {document.author}")
    if document.company:
        lines.append(f"Company: {document.company}")
    lines.append(f"Date: {format_date(document.date_created)}")
    return lines






