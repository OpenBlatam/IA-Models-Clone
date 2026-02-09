"""Security utilities for document processing"""
from typing import Optional, List
import re
import html
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class SecuritySanitizer:
    """Sanitize content for security"""
    
    def __init__(self):
        # Dangerous patterns
        self.dangerous_patterns = [
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'on\w+\s*=',
            r'<iframe[^>]*>',
            r'<object[^>]*>',
            r'<embed[^>]*>',
            r'data:text/html',
            r'vbscript:',
        ]
    
    def sanitize_html(self, html_content: str) -> str:
        """
        Sanitize HTML content
        
        Args:
            html_content: HTML content to sanitize
            
        Returns:
            Sanitized HTML
        """
        # Remove dangerous patterns
        for pattern in self.dangerous_patterns:
            html_content = re.sub(pattern, '', html_content, flags=re.IGNORECASE | re.DOTALL)
        
        # Escape remaining HTML
        html_content = html.escape(html_content)
        
        return html_content
    
    def sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename to prevent path traversal
        
        Args:
            filename: Filename to sanitize
            
        Returns:
            Sanitized filename
        """
        # Remove path components
        filename = Path(filename).name
        
        # Remove dangerous characters
        dangerous_chars = ['..', '/', '\\', ':', '*', '?', '"', '<', '>', '|']
        for char in dangerous_chars:
            filename = filename.replace(char, '_')
        
        # Limit length
        if len(filename) > 255:
            filename = filename[:255]
        
        return filename
    
    def sanitize_path(self, file_path: str, base_dir: str) -> Path:
        """
        Sanitize file path to prevent directory traversal
        
        Args:
            file_path: File path to sanitize
            base_dir: Base directory
            
        Returns:
            Safe Path object
        """
        # Resolve to absolute path
        base = Path(base_dir).resolve()
        path = Path(file_path).resolve()
        
        # Ensure path is within base directory
        try:
            path.relative_to(base)
        except ValueError:
            # Path is outside base directory, use base
            logger.warning(f"Path {file_path} is outside base directory, using base")
            path = base / Path(file_path).name
        
        return path
    
    def validate_url(self, url: str) -> bool:
        """
        Validate URL for safety
        
        Args:
            url: URL to validate
            
        Returns:
            True if safe, False otherwise
        """
        # Check for dangerous protocols
        dangerous_protocols = ['javascript:', 'data:', 'vbscript:', 'file:']
        url_lower = url.lower()
        
        for protocol in dangerous_protocols:
            if url_lower.startswith(protocol):
                return False
        
        # Check for local file access
        if url_lower.startswith('file://'):
            return False
        
        return True
    
    def sanitize_markdown(self, markdown_content: str) -> str:
        """
        Sanitize Markdown content
        
        Args:
            markdown_content: Markdown content
            
        Returns:
            Sanitized Markdown
        """
        # Remove HTML tags that could contain scripts
        markdown_content = re.sub(r'<script[^>]*>.*?</script>', '', markdown_content, 
                                 flags=re.IGNORECASE | re.DOTALL)
        markdown_content = re.sub(r'<iframe[^>]*>.*?</iframe>', '', markdown_content,
                                 flags=re.IGNORECASE | re.DOTALL)
        
        # Remove dangerous URLs in links
        def sanitize_link(match):
            text = match.group(1)
            url = match.group(2)
            if self.validate_url(url):
                return f"[{text}]({url})"
            return text
        
        markdown_content = re.sub(r'\[([^\]]+)\]\(([^\)]+)\)', sanitize_link, markdown_content)
        
        return markdown_content


# Global sanitizer
_sanitizer_instance: Optional[SecuritySanitizer] = None


def get_security_sanitizer() -> SecuritySanitizer:
    """Get global security sanitizer"""
    global _sanitizer_instance
    if _sanitizer_instance is None:
        _sanitizer_instance = SecuritySanitizer()
    return _sanitizer_instance

