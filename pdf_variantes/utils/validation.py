"""
PDF Variantes API - Enhanced Validation
Real-world validation utilities with detailed error messages
"""

from typing import Any, Dict, List, Optional, Tuple
import re
from pathlib import Path


class ValidationError(Exception):
    """Custom validation error with field-specific errors"""
    
    def __init__(self, message: str, field_errors: Optional[Dict[str, List[str]]] = None):
        self.message = message
        self.field_errors = field_errors or {}
        super().__init__(self.message)


def validate_filename(filename: str) -> Tuple[bool, Optional[str]]:
    """
    Validate filename
    
    Returns:
        (is_valid, error_message)
    """
    if not filename:
        return False, "Filename is required"
    
    if len(filename) > 255:
        return False, "Filename too long. Maximum length is 255 characters"
    
    # Check for invalid characters
    invalid_chars = '<>:"|?*\\'
    if any(char in filename for char in invalid_chars):
        return False, f"Filename contains invalid characters: {invalid_chars}"
    
    # Check for path traversal attempts
    if '..' in filename or filename.startswith('/'):
        return False, "Filename contains invalid path components"
    
    return True, None


def validate_file_extension(filename: str, allowed_extensions: List[str]) -> Tuple[bool, Optional[str]]:
    """
    Validate file extension
    
    Returns:
        (is_valid, error_message)
    """
    if not filename:
        return False, "Filename is required"
    
    file_ext = Path(filename).suffix.lower()
    if not file_ext:
        return False, "File must have an extension"
    
    allowed = [ext.lower() if not ext.startswith('.') else ext.lower() for ext in allowed_extensions]
    
    if file_ext not in allowed:
        return False, f"File extension not allowed. Allowed extensions: {', '.join(allowed)}"
    
    return True, None


def validate_integer_range(
    value: int,
    min_value: Optional[int] = None,
    max_value: Optional[int] = None,
    field_name: str = "value"
) -> Tuple[bool, Optional[str]]:
    """Validate integer range"""
    if min_value is not None and value < min_value:
        return False, f"{field_name} must be at least {min_value}"
    
    if max_value is not None and value > max_value:
        return False, f"{field_name} must be at most {max_value}"
    
    return True, None


def validate_string_length(
    value: str,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    field_name: str = "value"
) -> Tuple[bool, Optional[str]]:
    """Validate string length"""
    if not isinstance(value, str):
        return False, f"{field_name} must be a string"
    
    if min_length is not None and len(value) < min_length:
        return False, f"{field_name} must be at least {min_length} characters"
    
    if max_length is not None and len(value) > max_length:
        return False, f"{field_name} must be at most {max_length} characters"
    
    return True, None


def validate_email(email: str) -> Tuple[bool, Optional[str]]:
    """Validate email address"""
    if not email:
        return False, "Email is required"
    
    if len(email) > 254:
        return False, "Email address too long. Maximum length is 254 characters"
    
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(pattern, email):
        return False, "Invalid email format"
    
    return True, None


def validate_uuid(uuid_string: str, field_name: str = "ID") -> Tuple[bool, Optional[str]]:
    """Validate UUID format"""
    if not uuid_string:
        return False, f"{field_name} is required"
    
    pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
    if not re.match(pattern, uuid_string.lower()):
        return False, f"Invalid {field_name} format. Must be a valid UUID"
    
    return True, None


def validate_enum_value(
    value: Any,
    allowed_values: List[Any],
    field_name: str = "value"
) -> Tuple[bool, Optional[str]]:
    """Validate enum value"""
    if value not in allowed_values:
        return False, f"{field_name} must be one of: {', '.join(map(str, allowed_values))}"
    
    return True, None


def validate_list_not_empty(
    value: List[Any],
    field_name: str = "list",
    min_items: int = 1
) -> Tuple[bool, Optional[str]]:
    """Validate list is not empty"""
    if not isinstance(value, list):
        return False, f"{field_name} must be a list"
    
    if len(value) < min_items:
        return False, f"{field_name} must contain at least {min_items} item(s)"
    
    return True, None


class Validator:
    """Validators for common validation patterns"""
    
    @staticmethod
    def validate_document_id(document_id: str) -> Tuple[bool, Optional[str]]:
        """Validate document ID"""
        return validate_uuid(document_id, "Document ID")
    
    @staticmethod
    def validate_user_id(user_id: str) -> Tuple[bool, Optional[str]]:
        """Validate user ID"""
        if not user_id:
            return False, "User ID is required"
        
        if len(user_id) > 100:
            return False, "User ID too long. Maximum length is 100 characters"
        
        return True, None
    
    @staticmethod
    def validate_page_range(
        page_start: int,
        page_end: int,
        max_pages: int
    ) -> Tuple[bool, Optional[str]]:
        """Validate page range"""
        if page_start < 1:
            return False, "Page start must be at least 1"
        
        if page_end > max_pages:
            return False, f"Page end must be at most {max_pages}"
        
        if page_start > page_end:
            return False, "Page start must be less than or equal to page end"
        
        return True, None
    
    @staticmethod
    def validate_date_range(
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> Tuple[bool, Optional[str]]:
        """Validate date range"""
        if not start_date or not end_date:
            return True, None  # Optional validation
        
        try:
            from datetime import datetime
            start = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            end = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            
            if start > end:
                return False, "Start date must be before end date"
            
            return True, None
        except Exception as e:
            return False, f"Invalid date format: {str(e)}"






