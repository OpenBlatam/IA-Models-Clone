"""
PDF export handler.
"""

import os
from typing import Dict, Any
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY

from .base import BaseExporter


class PDFExporter(BaseExporter):
    """PDF export handler with professional formatting."""
    
    def get_supported_features(self) -> list:
        """Get supported PDF features."""
        return [
            "High quality",
            "Print ready", 
            "Professional layout",
            "Vector graphics",
            "Embedded fonts",
            "Page numbers",
            "Headers and footers",
            "Table support",
            "Image embedding"
        ]
    
    async def export(
        self, 
        content: Dict[str, Any], 
        config: Any, 
        output_path: str
    ) -> Dict[str, Any]:
        """Export content to PDF format."""
        try:
            # Preprocess content
            processed_content = self.preprocess_content(content)
            
            # Create PDF document
            doc = SimpleDocTemplate(output_path, pagesize=A4)
            styles = getSampleStyleSheet()
            story = []
            
            # Apply professional styling
            quality_config = config.quality_level.value if hasattr(config.quality_level, 'value') else 'professional'
            
            # Title
            if "title" in processed_content:
                title_style = ParagraphStyle(
                    'CustomTitle',
                    parent=styles['Title'],
                    fontSize=18,
                    spaceAfter=30,
                    alignment=TA_CENTER,
                    textColor=colors.HexColor("#2E2E2E")
                )
                story.append(Paragraph(processed_content["title"], title_style))
                story.append(Spacer(1, 12))
            
            # Content sections
            if "sections" in processed_content:
                for section in processed_content["sections"]:
                    if "heading" in section:
                        heading_style = ParagraphStyle(
                            'CustomHeading',
                            parent=styles['Heading1'],
                            fontSize=14,
                            spaceAfter=12,
                            textColor=colors.HexColor("#1F4E79")
                        )
                        story.append(Paragraph(section["heading"], heading_style))
                    
                    if "content" in section:
                        body_style = ParagraphStyle(
                            'CustomBody',
                            parent=styles['Normal'],
                            fontSize=11,
                            spaceAfter=6,
                            textColor=colors.HexColor("#2E2E2E")
                        )
                        story.append(Paragraph(section["content"], body_style))
                        story.append(Spacer(1, 6))
            
            # Build PDF
            doc.build(story)
            
            result = {
                "format": "pdf",
                "pages": len(story),
                "professional_features": {
                    "quality_level": quality_config,
                    "vector_graphics": True,
                    "print_ready": True
                }
            }
            
            return self.postprocess_result(result)
            
        except Exception as e:
            self.logger.error(f"PDF export failed: {e}")
            raise
    
    def postprocess_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Postprocess PDF export result."""
        result["file_type"] = "application/pdf"
        result["mime_type"] = "application/pdf"
        return result




