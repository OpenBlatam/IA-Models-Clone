"""
Factory for creating document exporters.

Centralized exporter creation logic using the Factory pattern.
"""

from typing import Dict, Type
from .models import ExportFormat
from .exporters import (
    DocumentExporter,
    PDFExporter,
    MarkdownExporter,
    WordExporter,
    HTMLExporter
)
from .exceptions import DocumentExportError


class ExporterFactory:
    """Factory for creating document exporters based on format."""
    
    _exporters: Dict[ExportFormat, Type[DocumentExporter]] = {
        ExportFormat.PDF: PDFExporter,
        ExportFormat.MARKDOWN: MarkdownExporter,
        ExportFormat.WORD: WordExporter,
        ExportFormat.HTML: HTMLExporter,
    }
    
    @classmethod
    def create_exporter(cls, format: ExportFormat) -> DocumentExporter:
        """
        Create an exporter instance for the specified format.
        
        Args:
            format: The export format
            
        Returns:
            An instance of the appropriate exporter
            
        Raises:
            DocumentExportError: If the format is not supported
        """
        exporter_class = cls._exporters.get(format)
        if not exporter_class:
            raise DocumentExportError(f"Unsupported export format: {format.value}")
        return exporter_class()
    
    @classmethod
    def is_format_supported(cls, format: ExportFormat) -> bool:
        """
        Check if a format is supported.
        
        Args:
            format: The export format to check
            
        Returns:
            True if the format is supported, False otherwise
        """
        return format in cls._exporters
    
    @classmethod
    def get_supported_formats(cls) -> list[ExportFormat]:
        """
        Get list of all supported export formats.
        
        Returns:
            List of supported export formats
        """
        return list(cls._exporters.keys())

