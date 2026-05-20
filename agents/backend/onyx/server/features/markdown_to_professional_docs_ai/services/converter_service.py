"""Converter Service - Main service for converting parsed Markdown to various formats"""
import os
import uuid
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from config import settings
from services.converters.excel_converter import ExcelConverter
from services.converters.pdf_converter import PDFConverter
from services.converters.word_converter import WordConverter
from services.converters.html_converter import HTMLConverter
from services.converters.tableau_converter import TableauConverter
from services.converters.powerbi_converter import PowerBIConverter
from services.converters.ppt_converter import PPTConverter
from services.converters.latex_converter import LaTeXConverter
from services.converters.rtf_converter import RTFConverter
from services.converters.epub_converter import EPUBConverter
from utils.exceptions import InvalidFormatException, ConversionException


class ConverterService:
    """Main service for converting Markdown to professional formats"""
    
    def __init__(self):
        self.converters = {
            "excel": ExcelConverter(),
            "pdf": PDFConverter(),
            "word": WordConverter(),
            "html": HTMLConverter(),
            "tableau": TableauConverter(),
            "powerbi": PowerBIConverter(),
            "ppt": PPTConverter(),
            "pptx": PPTConverter(),
            "latex": LaTeXConverter(),
            "rtf": RTFConverter(),
            "epub": EPUBConverter(),
        }
    
    async def convert(
        self,
        parsed_content: Dict[str, Any],
        output_format: str,
        include_charts: bool = True,
        include_tables: bool = True,
        custom_styling: Optional[Dict[str, Any]] = None,
        filename_suffix: str = ""
    ) -> str:
        """
        Convert parsed Markdown content to specified format
        
        Args:
            parsed_content: Parsed Markdown content from MarkdownParser
            output_format: Target format (excel, pdf, word, etc.)
            include_charts: Whether to include charts and diagrams
            include_tables: Whether to include tables
            custom_styling: Custom styling options
            filename_suffix: Suffix to add to output filename
            
        Returns:
            Path to generated output file
        """
        # Normalize format name
        output_format = output_format.lower().strip()
        
        if output_format not in self.converters:
            raise InvalidFormatException(output_format, list(self.converters.keys()))
        
        # Generate output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = parsed_content.get("title", "document")
        if not base_name or len(base_name) > 50:
            base_name = "document"
        
        # Sanitize filename
        base_name = "".join(c for c in base_name if c.isalnum() or c in (' ', '-', '_')).strip()
        base_name = base_name.replace(' ', '_')
        
        filename = f"{base_name}{filename_suffix}_{timestamp}.{self._get_extension(output_format)}"
        output_path = os.path.join(settings.output_dir, filename)
        
        # Get converter
        converter = self.converters[output_format]
        
        # Convert
        try:
            await converter.convert(
                parsed_content=parsed_content,
                output_path=output_path,
                include_charts=include_charts,
                include_tables=include_tables,
                custom_styling=custom_styling or {}
            )
        except Exception as e:
            raise ConversionException(
                output_format,
                str(e),
                {"output_path": output_path}
            )
        
        return output_path
    
    def _get_extension(self, format_name: str) -> str:
        """Get file extension for format"""
        extensions = {
            "excel": "xlsx",
            "pdf": "pdf",
            "word": "docx",
            "html": "html",
            "tableau": "twb",
            "powerbi": "pbix",
            "ppt": "pptx",
            "pptx": "pptx",
            "latex": "tex",
            "tex": "tex",
            "rtf": "rtf",
            "epub": "epub",
            "odt": "odt"
        }
        return extensions.get(format_name.lower(), "txt")

