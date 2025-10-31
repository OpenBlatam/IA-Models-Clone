"""
Factory for creating export format handlers.
"""

from typing import Dict, Type
from .base import BaseExporter
from .pdf_exporter import PDFExporter
from .docx_exporter import DOCXExporter
from .html_exporter import HTMLExporter
from .markdown_exporter import MarkdownExporter
from .rtf_exporter import RTFExporter
from .txt_exporter import TXTExporter
from .json_exporter import JSONExporter
from .xml_exporter import XMLExporter
from ..core.models import ExportFormat


class ExporterFactory:
    """Factory for creating export format handlers."""
    
    _exporters: Dict[ExportFormat, Type[BaseExporter]] = {
        ExportFormat.PDF: PDFExporter,
        ExportFormat.DOCX: DOCXExporter,
        ExportFormat.HTML: HTMLExporter,
        ExportFormat.MARKDOWN: MarkdownExporter,
        ExportFormat.RTF: RTFExporter,
        ExportFormat.TXT: TXTExporter,
        ExportFormat.JSON: JSONExporter,
        ExportFormat.XML: XMLExporter
    }
    
    @classmethod
    def create_exporter(cls, format_type: ExportFormat) -> BaseExporter:
        """
        Create an exporter for the specified format.
        
        Args:
            format_type: Export format type
            
        Returns:
            Exporter instance
            
        Raises:
            ValueError: If format is not supported
        """
        if format_type not in cls._exporters:
            raise ValueError(f"Unsupported export format: {format_type}")
        
        exporter_class = cls._exporters[format_type]
        return exporter_class()
    
    @classmethod
    def get_supported_formats(cls) -> list:
        """Get list of supported export formats."""
        return list(cls._exporters.keys())
    
    @classmethod
    def register_exporter(cls, format_type: ExportFormat, exporter_class: Type[BaseExporter]):
        """
        Register a new exporter for a format.
        
        Args:
            format_type: Export format type
            exporter_class: Exporter class
        """
        cls._exporters[format_type] = exporter_class
    
    @classmethod
    def is_format_supported(cls, format_type: ExportFormat) -> bool:
        """Check if a format is supported."""
        return format_type in cls._exporters




