"""PDF Converter - Convert Markdown to PDF format"""
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.pdfgen import canvas
from typing import Dict, Any, Optional
import io

from .base_converter import BaseConverter
from ...utils.chart_generator import ChartGenerator


class PDFConverter(BaseConverter):
    """Convert Markdown to PDF format"""
    
    def __init__(self):
        self.chart_generator = ChartGenerator()
    
    async def convert(
        self,
        parsed_content: Dict[str, Any],
        output_path: str,
        include_charts: bool = True,
        include_tables: bool = True,
        custom_styling: Optional[Dict[str, Any]] = None
    ) -> None:
        """Convert to PDF format"""
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )
        
        story = []
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1a1a1a'),
            spaceAfter=30,
            alignment=TA_CENTER
        )
        
        heading_styles = {
            1: ParagraphStyle('H1', parent=styles['Heading1'], fontSize=18, spaceAfter=12),
            2: ParagraphStyle('H2', parent=styles['Heading2'], fontSize=16, spaceAfter=10),
            3: ParagraphStyle('H3', parent=styles['Heading3'], fontSize=14, spaceAfter=8),
        }
        
        # Add title
        if parsed_content.get("title"):
            story.append(Paragraph(parsed_content["title"], title_style))
            story.append(Spacer(1, 0.2*inch))
        
        # Add headings and content
        for heading in parsed_content.get("headings", []):
            level = heading["level"]
            style = heading_styles.get(level, styles['Heading1'])
            story.append(Paragraph(heading["text"], style))
            story.append(Spacer(1, 0.1*inch))
        
        # Add tables
        if include_tables:
            for table in parsed_content.get("tables", []):
                # Create table data
                table_data = [table["headers"]]
                table_data.extend(table["rows"])
                
                # Create table
                pdf_table = Table(table_data)
                pdf_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#366092')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
                ]))
                
                story.append(pdf_table)
                story.append(Spacer(1, 0.2*inch))
        
        # Add charts (as images if available)
        if include_charts and parsed_content.get("tables"):
            for idx, table in enumerate(parsed_content.get("tables", [])):
                try:
                    chart_img_bytes = self.chart_generator.create_chart_from_table(table, "bar")
                    if chart_img_bytes:
                        chart_img = io.BytesIO(chart_img_bytes)
                        story.append(Image(chart_img, width=5*inch, height=3*inch))
                        story.append(Spacer(1, 0.2*inch))
                except Exception as e:
                    # Skip chart if there's an error
                    pass
        
        # Add paragraphs
        for para in parsed_content.get("paragraphs", []):
            story.append(Paragraph(para, styles['Normal']))
            story.append(Spacer(1, 0.1*inch))
        
        # Build PDF
        doc.build(story)

