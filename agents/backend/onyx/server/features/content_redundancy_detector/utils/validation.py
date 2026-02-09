"""
Enhanced Validation Utilities
Real-world validation with detailed error messages
"""

from typing import Tuple, Optional, List
import re


class ValidationError(Exception):
    """Custom validation error with detailed information"""
    
    def __init__(self, message: str, field: Optional[str] = None, code: Optional[str] = None):
        self.message = message
        self.field = field
        self.code = code or "VALIDATION_ERROR"
        super().__init__(self.message)


def validate_content_length(
    content: str,
    min_length: int = 10,
    max_length: int = 50000
) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Validate content length
    
    Returns:
        (is_valid, error_message, error_code)
    """
    if not isinstance(content, str):
        return False, "Content must be a string", "INVALID_TYPE"
    
    if not content or not content.strip():
        return False, "Content cannot be empty or whitespace only", "CONTENT_EMPTY"
    
    content_length = len(content)
    
    if content_length < min_length:
        return False, f"Content too short. Minimum length is {min_length} characters", "CONTENT_TOO_SHORT"
    
    if content_length > max_length:
        return False, f"Content too long. Maximum length is {max_length} characters", "CONTENT_TOO_LONG"
    
    return True, None, None


def validate_text_input(text: str, field_name: str = "text") -> Tuple[bool, Optional[str], Optional[str]]:
    """Validate text input with comprehensive checks"""
    if not isinstance(text, str):
        return False, f"{field_name} must be a string", "INVALID_TYPE"
    
    if not text or not text.strip():
        return False, f"{field_name} cannot be empty", "EMPTY_VALUE"
    
    # Check for suspicious content (basic check)
    if len(text) > 100000:  # Very large content might be suspicious
        return False, f"{field_name} exceeds reasonable size limit", "SIZE_EXCEEDED"
    
    return True, None, None


def validate_similarity_threshold(threshold: float) -> Tuple[bool, Optional[str], Optional[str]]:
    """Validate similarity threshold value"""
    if not isinstance(threshold, (int, float)):
        return False, "Threshold must be a number", "INVALID_TYPE"
    
    if threshold < 0 or threshold > 1:
        return False, "Threshold must be between 0 and 1", "OUT_OF_RANGE"
    
    return True, None, None


def validate_batch_size(size: int, max_size: int = 100) -> Tuple[bool, Optional[str], Optional[str]]:
    """Validate batch size"""
    if not isinstance(size, int):
        return False, "Batch size must be an integer", "INVALID_TYPE"
    
    if size < 1:
        return False, "Batch size must be at least 1", "TOO_SMALL"
    
    if size > max_size:
        return False, f"Batch size cannot exceed {max_size}", "TOO_LARGE"
    
    return True, None, None


def validate_list_not_empty(items: List, field_name: str = "list") -> Tuple[bool, Optional[str], Optional[str]]:
    """Validate that a list is not empty"""
    if not isinstance(items, list):
        return False, f"{field_name} must be a list", "INVALID_TYPE"
    
    if len(items) == 0:
        return False, f"{field_name} cannot be empty", "EMPTY_LIST"
    
    return True, None, None


def validate_uuid(uuid_string: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """Validate UUID format"""
    if not isinstance(uuid_string, str):
        return False, "UUID must be a string", "INVALID_TYPE"
    
    uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
    if not re.match(uuid_pattern, uuid_string.lower()):
        return False, "Invalid UUID format", "INVALID_FORMAT"
    
    return True, None, None


def validate_positive_number(value: float, field_name: str = "value") -> Tuple[bool, Optional[str], Optional[str]]:
    """Validate that a number is positive"""
    if not isinstance(value, (int, float)):
        return False, f"{field_name} must be a number", "INVALID_TYPE"
    
    if value <= 0:
        return False, f"{field_name} must be positive", "NOT_POSITIVE"
    
    return True, None, None


class ContentValidator:
    """Validator for content-related inputs"""
    
    @staticmethod
    def validate_analysis_input(content: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """Validate content for analysis"""
        return validate_content_length(content, min_length=10, max_length=50000)
    
    @staticmethod
    def validate_similarity_inputs(text1: str, text2: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """Validate inputs for similarity detection"""
        is_valid, error, code = validate_text_input(text1, "text1")
        if not is_valid:
            return is_valid, error, code
        
        is_valid, error, code = validate_text_input(text2, "text2")
        if not is_valid:
            return is_valid, error, code
        
        return True, None, None
    
    @staticmethod
    def validate_batch_input(texts: List[str]) -> Tuple[bool, Optional[str], Optional[str]]:
        """Validate batch input"""
        is_valid, error, code = validate_list_not_empty(texts, "texts")
        if not is_valid:
            return is_valid, error, code
        
        # Validate each text
        for i, text in enumerate(texts):
            is_valid, error, code = validate_text_input(text, f"texts[{i}]")
            if not is_valid:
                return is_valid, error, code
        
        return True, None, None






