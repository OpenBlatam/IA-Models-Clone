"""Validators for Markdown to Professional Documents AI"""
import re
from typing import List, Optional
from pathlib import Path

from utils.exceptions import ValidationException, InvalidFormatException


# Supported formats
SUPPORTED_FORMATS = [
    "excel", "xlsx", "pdf", "word", "docx", "html", 
    "tableau", "twb", "powerbi", "pbix", "ppt", "pptx",
    "odt", "rtf"
]


def validate_format(format_name: str) -> str:
    """
    Validate and normalize format name
    
    Args:
        format_name: Format name to validate
        
    Returns:
        Normalized format name
        
    Raises:
        InvalidFormatException: If format is not supported
    """
    format_name = format_name.lower().strip()
    
    # Normalize aliases
    format_map = {
        "xlsx": "excel",
        "docx": "word",
        "twb": "tableau",
        "pbix": "powerbi",
        "pptx": "ppt"
    }
    
    normalized = format_map.get(format_name, format_name)
    
    if normalized not in SUPPORTED_FORMATS:
        raise InvalidFormatException(format_name, SUPPORTED_FORMATS)
    
    return normalized


def validate_markdown_content(content: str, min_length: int = 1) -> str:
    """
    Validate Markdown content
    
    Args:
        content: Markdown content to validate
        min_length: Minimum content length
        
    Returns:
        Validated content
        
    Raises:
        ValidationException: If content is invalid
    """
    if not isinstance(content, str):
        raise ValidationException("Content must be a string")
    
    content = content.strip()
    
    if len(content) < min_length:
        raise ValidationException(f"Content must be at least {min_length} characters")
    
    return content


def validate_file_path(file_path: str, must_exist: bool = False) -> Path:
    """
    Validate file path
    
    Args:
        file_path: Path to validate
        must_exist: Whether file must exist
        
    Returns:
        Path object
        
    Raises:
        ValidationException: If path is invalid
    """
    try:
        path = Path(file_path)
        
        if must_exist and not path.exists():
            raise ValidationException(f"Path does not exist: {file_path}")
        
        if must_exist and not path.is_file():
            raise ValidationException(f"Path is not a file: {file_path}")
        
        return path
    except Exception as e:
        raise ValidationException(f"Invalid file path: {str(e)}")


def validate_file_size(file_size: int, max_size: int) -> None:
    """
    Validate file size
    
    Args:
        file_size: File size in bytes
        max_size: Maximum allowed size in bytes
        
    Raises:
        ValidationException: If file size exceeds limit
    """
    if file_size > max_size:
        from utils.exceptions import FileSizeException
        raise FileSizeException(file_size, max_size)


def validate_filename(filename: str) -> str:
    """
    Validate and sanitize filename
    
    Args:
        filename: Filename to validate
        
    Returns:
        Sanitized filename
    """
    if not filename:
        raise ValidationException("Filename cannot be empty")
    
    # Remove path components
    filename = Path(filename).name
    
    # Remove invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '', filename)
    
    if not filename:
        raise ValidationException("Filename contains only invalid characters")
    
    return filename


def validate_batch_size(size: int, min_size: int = 1, max_size: int = 10) -> None:
    """
    Validate batch size
    
    Args:
        size: Batch size
        min_size: Minimum batch size
        max_size: Maximum batch size
        
    Raises:
        ValidationException: If batch size is invalid
    """
    if size < min_size:
        raise ValidationException(f"Batch size must be at least {min_size}")
    
    if size > max_size:
        raise ValidationException(f"Batch size cannot exceed {max_size}")


def validate_output_format(format_name: str) -> str:
    """
    Validate output format with detailed error messages
    
    Args:
        format_name: Format to validate
        
    Returns:
        Normalized format name
    """
    try:
        return validate_format(format_name)
    except InvalidFormatException as e:
        # Provide helpful suggestions
        format_name_lower = format_name.lower()
        suggestions = [f for f in SUPPORTED_FORMATS if format_name_lower in f or f in format_name_lower]
        
        if suggestions:
            raise InvalidFormatException(
                f"{format_name}. Did you mean: {', '.join(suggestions[:3])}?",
                SUPPORTED_FORMATS
            )
        raise

