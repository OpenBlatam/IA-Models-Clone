"""
Format Converter
================

Advanced format conversion system for documents.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import json
import markdown
import html2text
from pathlib import Path
import aiofiles
import subprocess
import tempfile
import os

logger = logging.getLogger(__name__)

class FormatConverter:
    """
    Advanced format converter for documents.
    
    Features:
    - Markdown to HTML
    - HTML to Markdown
    - PDF generation
    - JSON formatting
    - Custom formats
    - Batch conversion
    """
    
    def __init__(self):
        self.supported_formats = {
            'markdown': ['html', 'pdf', 'json'],
            'html': ['markdown', 'pdf', 'json'],
            'json': ['markdown', 'html', 'pdf'],
            'txt': ['markdown', 'html', 'pdf', 'json'],
            'pdf': ['markdown', 'html', 'txt']
        }
        
    async def initialize(self):
        """Initialize format converter."""
        logger.info("Initializing Format Converter...")
        
        try:
            # Check for required tools
            await self._check_dependencies()
            
            logger.info("Format Converter initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Format Converter: {str(e)}")
            raise
    
    async def _check_dependencies(self):
        """Check for required dependencies."""
        try:
            # Check for pandoc (for advanced conversions)
            try:
                result = subprocess.run(['pandoc', '--version'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    logger.info("Pandoc found - advanced conversions available")
                else:
                    logger.warning("Pandoc not found - using basic conversions only")
            except (subprocess.TimeoutExpired, FileNotFoundError):
                logger.warning("Pandoc not found - using basic conversions only")
            
            # Check for wkhtmltopdf (for PDF generation)
            try:
                result = subprocess.run(['wkhtmltopdf', '--version'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    logger.info("wkhtmltopdf found - PDF generation available")
                else:
                    logger.warning("wkhtmltopdf not found - PDF generation limited")
            except (subprocess.TimeoutExpired, FileNotFoundError):
                logger.warning("wkhtmltopdf not found - PDF generation limited")
                
        except Exception as e:
            logger.error(f"Failed to check dependencies: {str(e)}")
    
    async def convert(
        self, 
        content: str, 
        target_format: str,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """
        Convert content to target format.
        
        Args:
            content: Source content
            target_format: Target format
            metadata: Document metadata
            **kwargs: Additional conversion options
            
        Returns:
            Converted content
        """
        try:
            # Detect source format
            source_format = self._detect_format(content)
            
            # Check if conversion is supported
            if target_format not in self.supported_formats.get(source_format, []):
                logger.warning(f"Conversion from {source_format} to {target_format} not supported")
                return content
            
            # Perform conversion
            if source_format == 'markdown' and target_format == 'html':
                return await self._markdown_to_html(content, metadata, **kwargs)
            elif source_format == 'html' and target_format == 'markdown':
                return await self._html_to_markdown(content, metadata, **kwargs)
            elif target_format == 'json':
                return await self._to_json(content, metadata, **kwargs)
            elif target_format == 'pdf':
                return await self._to_pdf(content, source_format, metadata, **kwargs)
            elif target_format == 'txt':
                return await self._to_txt(content, source_format, metadata, **kwargs)
            else:
                logger.warning(f"Direct conversion from {source_format} to {target_format} not implemented")
                return content
                
        except Exception as e:
            logger.error(f"Failed to convert content: {str(e)}")
            return content
    
    def _detect_format(self, content: str) -> str:
        """Detect content format."""
        try:
            content_lower = content.lower().strip()
            
            # Check for JSON
            if content.startswith('{') and content.endswith('}'):
                try:
                    json.loads(content)
                    return 'json'
                except:
                    pass
            
            # Check for HTML
            if '<html>' in content_lower or '<div>' in content_lower or '<p>' in content_lower:
                return 'html'
            
            # Check for Markdown
            if any(marker in content for marker in ['# ', '## ', '### ', '**', '*', '- ', '1. ']):
                return 'markdown'
            
            # Default to text
            return 'txt'
            
        except Exception as e:
            logger.error(f"Failed to detect format: {str(e)}")
            return 'txt'
    
    async def _markdown_to_html(
        self, 
        content: str, 
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """Convert Markdown to HTML."""
        try:
            # Configure markdown extensions
            extensions = [
                'markdown.extensions.extra',
                'markdown.extensions.codehilite',
                'markdown.extensions.toc',
                'markdown.extensions.tables',
                'markdown.extensions.fenced_code'
            ]
            
            # Convert to HTML
            html = markdown.markdown(content, extensions=extensions)
            
            # Wrap in HTML document if metadata provided
            if metadata:
                title = metadata.get('title', 'Generated Document')
                html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
        h1, h2, h3 {{ color: #333; }}
        code {{ background-color: #f4f4f4; padding: 2px 4px; border-radius: 3px; }}
        pre {{ background-color: #f4f4f4; padding: 10px; border-radius: 5px; overflow-x: auto; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    {html}
</body>
</html>'''
            
            return html
            
        except Exception as e:
            logger.error(f"Failed to convert Markdown to HTML: {str(e)}")
            return content
    
    async def _html_to_markdown(
        self, 
        content: str, 
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """Convert HTML to Markdown."""
        try:
            # Configure html2text
            h = html2text.HTML2Text()
            h.ignore_links = kwargs.get('ignore_links', False)
            h.ignore_images = kwargs.get('ignore_images', False)
            h.ignore_emphasis = kwargs.get('ignore_emphasis', False)
            h.body_width = kwargs.get('body_width', 0)
            
            # Convert to Markdown
            markdown_content = h.handle(content)
            
            return markdown_content.strip()
            
        except Exception as e:
            logger.error(f"Failed to convert HTML to Markdown: {str(e)}")
            return content
    
    async def _to_json(
        self, 
        content: str, 
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """Convert content to JSON format."""
        try:
            # Prepare JSON structure
            json_data = {
                'content': content,
                'metadata': metadata or {},
                'conversion_info': {
                    'converted_at': datetime.utcnow().isoformat(),
                    'format': 'json',
                    'word_count': len(content.split()),
                    'character_count': len(content)
                }
            }
            
            # Add additional fields if provided
            if kwargs:
                json_data['conversion_options'] = kwargs
            
            return json.dumps(json_data, indent=2, ensure_ascii=False)
            
        except Exception as e:
            logger.error(f"Failed to convert to JSON: {str(e)}")
            return content
    
    async def _to_pdf(
        self, 
        content: str, 
        source_format: str,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """Convert content to PDF."""
        try:
            # First convert to HTML if not already
            if source_format != 'html':
                html_content = await self._markdown_to_html(content, metadata, **kwargs)
            else:
                html_content = content
            
            # Try to use wkhtmltopdf if available
            try:
                return await self._html_to_pdf_wkhtmltopdf(html_content, metadata, **kwargs)
            except Exception as e:
                logger.warning(f"wkhtmltopdf conversion failed: {str(e)}")
                
                # Fallback to basic HTML with print styles
                return await self._html_to_pdf_basic(html_content, metadata, **kwargs)
                
        except Exception as e:
            logger.error(f"Failed to convert to PDF: {str(e)}")
            return content
    
    async def _html_to_pdf_wkhtmltopdf(
        self, 
        html_content: str, 
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """Convert HTML to PDF using wkhtmltopdf."""
        try:
            # Create temporary files
            with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as html_file:
                html_file.write(html_content)
                html_path = html_file.name
            
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as pdf_file:
                pdf_path = pdf_file.name
            
            try:
                # Run wkhtmltopdf
                cmd = [
                    'wkhtmltopdf',
                    '--page-size', kwargs.get('page_size', 'A4'),
                    '--margin-top', str(kwargs.get('margin_top', '20mm')),
                    '--margin-right', str(kwargs.get('margin_right', '20mm')),
                    '--margin-bottom', str(kwargs.get('margin_bottom', '20mm')),
                    '--margin-left', str(kwargs.get('margin_left', '20mm')),
                    html_path,
                    pdf_path
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    # Read PDF content
                    with open(pdf_path, 'rb') as f:
                        pdf_content = f.read()
                    
                    # Return base64 encoded PDF
                    import base64
                    return base64.b64encode(pdf_content).decode('utf-8')
                else:
                    raise Exception(f"wkhtmltopdf failed: {result.stderr}")
                    
            finally:
                # Clean up temporary files
                os.unlink(html_path)
                os.unlink(pdf_path)
                
        except Exception as e:
            logger.error(f"wkhtmltopdf conversion failed: {str(e)}")
            raise
    
    async def _html_to_pdf_basic(
        self, 
        html_content: str, 
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """Convert HTML to PDF using basic method."""
        try:
            # Add print styles to HTML
            print_styles = '''
            <style>
                @media print {
                    body { font-size: 12pt; line-height: 1.4; }
                    h1, h2, h3 { page-break-after: avoid; }
                    p { orphans: 3; widows: 3; }
                    pre, blockquote { page-break-inside: avoid; }
                }
            </style>
            '''
            
            # Insert styles into HTML
            if '<head>' in html_content:
                html_content = html_content.replace('<head>', f'<head>{print_styles}')
            else:
                html_content = f'<html><head>{print_styles}</head><body>{html_content}</body></html>'
            
            # Return HTML with print styles (can be printed to PDF by browser)
            return html_content
            
        except Exception as e:
            logger.error(f"Basic PDF conversion failed: {str(e)}")
            return html_content
    
    async def _to_txt(
        self, 
        content: str, 
        source_format: str,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """Convert content to plain text."""
        try:
            if source_format == 'html':
                # Convert HTML to text
                h = html2text.HTML2Text()
                h.ignore_links = True
                h.ignore_images = True
                h.ignore_emphasis = True
                return h.handle(content).strip()
            elif source_format == 'markdown':
                # Convert Markdown to text (remove markdown syntax)
                import re
                # Remove headers
                text = re.sub(r'^#+\s*', '', content, flags=re.MULTILINE)
                # Remove bold/italic
                text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
                text = re.sub(r'\*(.*?)\*', r'\1', text)
                # Remove links
                text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
                # Remove code blocks
                text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
                text = re.sub(r'`([^`]+)`', r'\1', text)
                return text.strip()
            else:
                return content
                
        except Exception as e:
            logger.error(f"Failed to convert to text: {str(e)}")
            return content
    
    async def batch_convert(
        self, 
        contents: List[str], 
        target_format: str,
        metadata_list: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> List[str]:
        """Convert multiple contents to target format."""
        try:
            results = []
            
            for i, content in enumerate(contents):
                metadata = metadata_list[i] if metadata_list and i < len(metadata_list) else None
                converted = await self.convert(content, target_format, metadata, **kwargs)
                results.append(converted)
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to batch convert: {str(e)}")
            return contents
    
    async def get_supported_formats(self) -> Dict[str, List[str]]:
        """Get supported format conversions."""
        return self.supported_formats.copy()
    
    async def cleanup(self):
        """Cleanup format converter."""
        try:
            logger.info("Format Converter cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Failed to cleanup Format Converter: {str(e)}")











