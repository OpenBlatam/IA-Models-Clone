"""
Manual Exporter
==============

Exportador principal que compone todos los exportadores especializados.
"""

from typing import Optional, Dict, Any
from ...core.base.service_base import BaseService
from .markdown_exporter import MarkdownExporter
from .text_exporter import TextExporter
from .pdf_exporter import PDFExporter


class ManualExporter(BaseService):
    """Exportador principal de manuales a diferentes formatos."""
    
    def __init__(self):
        """Inicializar exportador."""
        super().__init__(logger_name=__name__)
        self.markdown_exporter = MarkdownExporter()
        self.text_exporter = TextExporter()
        self.pdf_exporter = PDFExporter()
    
    def export_to_markdown(
        self,
        manual_content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Exportar manual a Markdown."""
        return self.markdown_exporter.export(manual_content, metadata)
    
    def export_to_text(
        self,
        manual_content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Exportar manual a texto plano."""
        return self.text_exporter.export(manual_content, metadata)
    
    def export_to_pdf(
        self,
        manual_content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bytes:
        """Exportar manual a PDF."""
        return self.pdf_exporter.export(manual_content, metadata)

