from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import re
from typing import List, Optional, Dict, Any
from urllib.parse import urlparse
from core.config import get_config
from core.exceptions import ValidationError, handle_async_exception
import structlog
from typing import Any, List, Dict, Optional
import logging
import asyncio
#!/usr/bin/env python3
"""
Validation Service for OS Content UGC Video Generator
Handles input validation and sanitization
"""



logger = structlog.get_logger("os_content.validation_service")

class ValidationService:
    """Input validation service"""
    
    def __init__(self) -> Any:
        self.config = get_config()
    
    @handle_async_exception
    async async def validate_video_request(self, 
                                   user_id: str,
                                   title: str,
                                   text_prompt: str,
                                   image_files: Optional[List[str]] = None,
                                   video_files: Optional[List[str]] = None) -> bool:
        """Validate video processing request"""
        
        # Validate user_id
        if not user_id or not user_id.strip():
            raise ValidationError("User ID is required", field="user_id")
        
        if len(user_id) > 100:
            raise ValidationError("User ID too long (max 100 characters)", field="user_id")
        
        # Validate title
        if not title or not title.strip():
            raise ValidationError("Title is required", field="title")
        
        if len(title) > 200:
            raise ValidationError("Title too long (max 200 characters)", field="title")
        
        # Validate text_prompt
        if not text_prompt or not text_prompt.strip():
            raise ValidationError("Text prompt is required", field="text_prompt")
        
        if len(text_prompt) > 1000:
            raise ValidationError("Text prompt too long (max 1000 characters)", field="text_prompt")
        
        # Validate files
        if not image_files and not video_files:
            raise ValidationError("At least one image or video file is required", field="files")
        
        if image_files:
            await self.validate_file_list(image_files, "image")
        
        if video_files:
            await self.validate_file_list(video_files, "video")
        
        return True
    
    @handle_async_exception
    async def validate_file_list(self, file_paths: List[str], file_type: str) -> bool:
        """Validate list of file paths"""
        
        if not isinstance(file_paths, list):
            raise ValidationError("File paths must be a list", field="file_paths")
        
        if len(file_paths) > 50:
            raise ValidationError(f"Too many {file_type} files (max 50)", field="file_paths")
        
        for file_path in file_paths:
            if not file_path or not isinstance(file_path, str):
                raise ValidationError("Invalid file path", field="file_path")
            
            if len(file_path) > 500:
                raise ValidationError("File path too long", field="file_path")
        
        return True
    
    @handle_async_exception
    async def validate_language(self, language: str) -> bool:
        """Validate language code"""
        
        valid_languages = ["es", "en", "fr", "de"]
        
        if not language or language not in valid_languages:
            raise ValidationError(
                f"Invalid language: {language}. Supported: {', '.join(valid_languages)}",
                field="language",
                value=language
            )
        
        return True
    
    @handle_async_exception
    async def validate_duration(self, duration: float) -> bool:
        """Validate duration value"""
        
        if not isinstance(duration, (int, float)):
            raise ValidationError("Duration must be a number", field="duration")
        
        if duration <= 0:
            raise ValidationError("Duration must be positive", field="duration")
        
        if duration > 60:
            raise ValidationError("Duration too long (max 60 seconds)", field="duration")
        
        return True
    
    @handle_async_exception
    async def validate_resolution(self, resolution: tuple) -> bool:
        """Validate video resolution"""
        
        if not isinstance(resolution, tuple) or len(resolution) != 2:
            raise ValidationError("Resolution must be a tuple of (width, height)", field="resolution")
        
        width, height = resolution
        
        if not isinstance(width, int) or not isinstance(height, int):
            raise ValidationError("Resolution dimensions must be integers", field="resolution")
        
        if width <= 0 or height <= 0:
            raise ValidationError("Resolution dimensions must be positive", field="resolution")
        
        if width > 4096 or height > 4096:
            raise ValidationError("Resolution too high (max 4096x4096)", field="resolution")
        
        return True
    
    @handle_async_exception
    async def validate_url(self, url: str) -> bool:
        """Validate URL format"""
        
        if not url or not isinstance(url, str):
            raise ValidationError("URL is required and must be a string", field="url")
        
        try:
            result = urlparse(url)
            if not all([result.scheme, result.netloc]):
                raise ValidationError("Invalid URL format", field="url", value=url)
        except Exception:
            raise ValidationError("Invalid URL format", field="url", value=url)
        
        return True
    
    @handle_async_exception
    async def validate_email(self, email: str) -> bool:
        """Validate email format"""
        
        if not email or not isinstance(email, str):
            raise ValidationError("Email is required and must be a string", field="email")
        
        # Simple email regex
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        
        if not re.match(email_pattern, email):
            raise ValidationError("Invalid email format", field="email", value=email)
        
        if len(email) > 254:
            raise ValidationError("Email too long", field="email", value=email)
        
        return True
    
    @handle_async_exception
    async def validate_text_length(self, text: str, field_name: str, max_length: int = 1000) -> bool:
        """Validate text length"""
        
        if not text or not isinstance(text, str):
            raise ValidationError(f"{field_name} is required and must be a string", field=field_name)
        
        if len(text) > max_length:
            raise ValidationError(
                f"{field_name} too long (max {max_length} characters)",
                field=field_name,
                value=len(text)
            )
        
        return True
    
    @handle_async_exception
    async def sanitize_text(self, text: str) -> str:
        """Sanitize text input"""
        
        if not text:
            return ""
        
        # Remove null bytes
        text = text.replace('\x00', '')
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        # Remove control characters except newlines and tabs
        text = ''.join(char for char in text if char.isprintable() or char in '\n\t')
        
        return text.strip()
    
    @handle_async_exception
    async def validate_metadata(self, metadata: Dict[str, Any]) -> bool:
        """Validate metadata dictionary"""
        
        if not isinstance(metadata, dict):
            raise ValidationError("Metadata must be a dictionary", field="metadata")
        
        if len(metadata) > 50:
            raise ValidationError("Too many metadata keys (max 50)", field="metadata")
        
        for key, value in metadata.items():
            # Validate key
            if not isinstance(key, str):
                raise ValidationError("Metadata keys must be strings", field="metadata")
            
            if len(key) > 100:
                raise ValidationError("Metadata key too long (max 100 characters)", field="metadata")
            
            # Validate value
            if isinstance(value, str) and len(value) > 1000:
                raise ValidationError("Metadata value too long (max 1000 characters)", field="metadata")
            
            if isinstance(value, (list, dict)) and len(str(value)) > 1000:
                raise ValidationError("Metadata value too large", field="metadata")
        
        return True
    
    @handle_async_exception
    async def validate_batch_size(self, items: List[Any], max_size: int = 100) -> bool:
        """Validate batch size"""
        
        if not isinstance(items, list):
            raise ValidationError("Items must be a list", field="items")
        
        if len(items) > max_size:
            raise ValidationError(f"Batch too large (max {max_size} items)", field="items")
        
        if len(items) == 0:
            raise ValidationError("Batch cannot be empty", field="items")
        
        return True

# Global validation service instance
validation_service = ValidationService() 