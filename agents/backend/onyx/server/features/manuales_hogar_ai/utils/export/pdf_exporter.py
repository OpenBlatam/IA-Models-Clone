"""
PDF Exporter
============

Exportador especializado para formato PDF.
"""

from typing import Optional, Dict, Any
from ...core.base.service_base import BaseService


class PDFExporter(BaseService):
    """Exportador de manuales a PDF."""
    
    def __init__(self):
        """Inicializar exportador."""
        super().__init__(logger_name=__name__)
    
    def export(
        self,
        manual_content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bytes:
        """
        Exportar manual a PDF.
        
        Args:
            manual_content: Contenido del manual
            metadata: Metadata adicional
        
        Returns:
            Contenido en formato PDF (bytes)
        """
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
            from reportlab.lib.units import inch
            from io import BytesIO
            
            buffer = BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter)
            styles = getSampleStyleSheet()
            story = []
            
            if metadata and metadata.get("title"):
                title = Paragraph(metadata['title'], styles['Title'])
                story.append(title)
                story.append(Spacer(1, 0.2 * inch))
            
            content_paragraphs = manual_content.split('\n')
            for para in content_paragraphs:
                if para.strip():
                    p = Paragraph(para.strip(), styles['Normal'])
                    story.append(p)
                    story.append(Spacer(1, 0.1 * inch))
            
            doc.build(story)
            buffer.seek(0)
            return buffer.getvalue()
        
        except ImportError:
            self.log_warning("reportlab no está instalado, usando texto plano")
            from .text_exporter import TextExporter
            text_exporter = TextExporter()
            return text_exporter.export(manual_content, metadata).encode('utf-8')
        except Exception as e:
            self.log_error(f"Error exportando a PDF: {str(e)}")
            return b""

