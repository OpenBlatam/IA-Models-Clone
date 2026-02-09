"""
Input Sanitizer
Sanitize user inputs
"""

from typing import Any, Dict
import re
import html
import logging

logger = logging.getLogger(__name__)


class InputSanitizer:
    """Sanitize user inputs"""
    
    def __init__(self):
        self.max_script_length = 10000
    
    def sanitize_text(self, text: str) -> str:
        """
        Sanitize text input
        
        Args:
            text: Input text
            
        Returns:
            Sanitized text
        """
        # Remove null bytes
        text = text.replace('\x00', '')
        
        # Remove control characters except newlines and tabs
        text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F]', '', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        # Truncate if too long
        if len(text) > self.max_script_length:
            text = text[:self.max_script_length]
            logger.warning(f"Text truncated to {self.max_script_length} characters")
        
        return text.strip()
    
    def sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename
        
        Args:
            filename: Input filename
            
        Returns:
            Sanitized filename
        """
        # Remove path components
        filename = filename.split('/')[-1].split('\\')[-1]
        
        # Remove dangerous characters
        filename = re.sub(r'[<>:"|?*\x00-\x1F]', '', filename)
        
        # Limit length
        if len(filename) > 255:
            name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
            filename = name[:250] + ('.' + ext if ext else '')
        
        return filename
    
    def sanitize_url(self, url: str) -> str:
        """
        Sanitize URL
        
        Args:
            url: Input URL
            
        Returns:
            Sanitized URL
        """
        # Remove dangerous protocols
        dangerous_protocols = ['javascript:', 'data:', 'vbscript:']
        url_lower = url.lower()
        
        for protocol in dangerous_protocols:
            if url_lower.startswith(protocol):
                return ''
        
        # Basic URL validation
        if not (url.startswith('http://') or url.startswith('https://')):
            return ''
        
        return url
    
    def sanitize_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize dictionary recursively
        
        Args:
            data: Input dictionary
            
        Returns:
            Sanitized dictionary
        """
        sanitized = {}
        
        for key, value in data.items():
            # Sanitize key
            sanitized_key = self.sanitize_text(str(key))
            
            # Sanitize value
            if isinstance(value, str):
                sanitized_value = self.sanitize_text(value)
            elif isinstance(value, dict):
                sanitized_value = self.sanitize_dict(value)
            elif isinstance(value, list):
                sanitized_value = [self.sanitize_text(str(v)) if isinstance(v, str) else v for v in value]
            else:
                sanitized_value = value
            
            sanitized[sanitized_key] = sanitized_value
        
        return sanitized


_input_sanitizer: Optional[InputSanitizer] = None


def get_input_sanitizer() -> InputSanitizer:
    """Get input sanitizer instance (singleton)"""
    global _input_sanitizer
    if _input_sanitizer is None:
        _input_sanitizer = InputSanitizer()
    return _input_sanitizer

