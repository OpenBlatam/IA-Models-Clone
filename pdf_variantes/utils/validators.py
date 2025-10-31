"""
PDF Variantes Validators
Input validation utilities
"""

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from fastapi import UploadFile

@dataclass
class ValidationResult:
    """Validation result"""
    is_valid: bool
    errors: List[str]
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []

def validate_file_upload(file: UploadFile, max_size_mb: int = 100, 
                        allowed_types: List[str] = None) -> ValidationResult:
    """Validate file upload"""
    errors = []
    warnings = []
    
    if allowed_types is None:
        allowed_types = ["pdf"]
    
    # Check file size
    if hasattr(file, 'size') and file.size:
        max_size_bytes = max_size_mb * 1024 * 1024
        if file.size > max_size_bytes:
            errors.append(f"File size ({file.size} bytes) exceeds maximum allowed size ({max_size_bytes} bytes)")
    
    # Check file type
    if file.content_type:
        if not any(file.content_type.startswith(f"application/{t}") for t in allowed_types):
            errors.append(f"File type '{file.content_type}' is not allowed. Allowed types: {allowed_types}")
    
    # Check filename
    if file.filename:
        if not re.match(r'^[a-zA-Z0-9._-]+$', file.filename):
            errors.append("Filename contains invalid characters")
        
        if len(file.filename) > 255:
            errors.append("Filename is too long (max 255 characters)")
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings
    )

def validate_content_type(content_type: str) -> ValidationResult:
    """Validate content type"""
    errors = []
    warnings = []
    
    allowed_types = [
        "application/pdf",
        "text/plain",
        "application/json",
        "text/html",
        "application/zip"
    ]
    
    if content_type not in allowed_types:
        errors.append(f"Content type '{content_type}' is not supported")
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings
    )

def validate_email(email: str) -> ValidationResult:
    """Validate email address"""
    errors = []
    warnings = []
    
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    if not re.match(email_pattern, email):
        errors.append("Invalid email format")
    
    if len(email) > 254:
        errors.append("Email address is too long")
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings
    )

def validate_password(password: str, min_length: int = 8) -> ValidationResult:
    """Validate password strength"""
    errors = []
    warnings = []
    
    # Length check
    if len(password) < min_length:
        errors.append(f"Password must be at least {min_length} characters long")
    
    # Character checks
    if not re.search(r'[A-Z]', password):
        warnings.append("Password should contain at least one uppercase letter")
    
    if not re.search(r'[a-z]', password):
        warnings.append("Password should contain at least one lowercase letter")
    
    if not re.search(r'\d', password):
        warnings.append("Password should contain at least one number")
    
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        warnings.append("Password should contain at least one special character")
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings
    )

def validate_document_id(document_id: str) -> ValidationResult:
    """Validate document ID format"""
    errors = []
    warnings = []
    
    # UUID format check
    uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
    
    if not re.match(uuid_pattern, document_id, re.IGNORECASE):
        errors.append("Invalid document ID format")
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings
    )

def validate_variant_config(config: Dict[str, Any]) -> ValidationResult:
    """Validate variant generation configuration"""
    errors = []
    warnings = []
    
    # Check similarity level
    similarity_level = config.get("similarity_level", 0.7)
    if not isinstance(similarity_level, (int, float)) or not 0 <= similarity_level <= 1:
        errors.append("Similarity level must be a number between 0 and 1")
    
    # Check creativity level
    creativity_level = config.get("creativity_level", 0.6)
    if not isinstance(creativity_level, (int, float)) or not 0 <= creativity_level <= 1:
        errors.append("Creativity level must be a number between 0 and 1")
    
    # Check preserve flags
    preserve_structure = config.get("preserve_structure", True)
    if not isinstance(preserve_structure, bool):
        errors.append("preserve_structure must be a boolean")
    
    preserve_meaning = config.get("preserve_meaning", True)
    if not isinstance(preserve_meaning, bool):
        errors.append("preserve_meaning must be a boolean")
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings
    )

def validate_search_query(query: str) -> ValidationResult:
    """Validate search query"""
    errors = []
    warnings = []
    
    # Length check
    if len(query.strip()) < 1:
        errors.append("Search query cannot be empty")
    
    if len(query) > 500:
        errors.append("Search query is too long (max 500 characters)")
    
    # Check for potentially dangerous patterns
    dangerous_patterns = [
        r'<script',
        r'javascript:',
        r'union\s+select',
        r'drop\s+table',
        r'delete\s+from'
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, query, re.IGNORECASE):
            errors.append("Search query contains potentially dangerous content")
            break
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings
    )

def validate_pagination_params(page: int, page_size: int) -> ValidationResult:
    """Validate pagination parameters"""
    errors = []
    warnings = []
    
    # Page validation
    if page < 1:
        errors.append("Page number must be at least 1")
    
    if page > 10000:
        warnings.append("Page number is very high")
    
    # Page size validation
    if page_size < 1:
        errors.append("Page size must be at least 1")
    
    if page_size > 1000:
        errors.append("Page size cannot exceed 1000")
    
    if page_size > 100:
        warnings.append("Large page size may impact performance")
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings
    )

def validate_date_range(start_date: str, end_date: str) -> ValidationResult:
    """Validate date range"""
    errors = []
    warnings = []
    
    try:
        from datetime import datetime
        
        start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
        end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
        
        if start_dt >= end_dt:
            errors.append("Start date must be before end date")
        
        # Check if date range is too large
        days_diff = (end_dt - start_dt).days
        if days_diff > 365:
            warnings.append("Date range is very large (over 1 year)")
        
    except ValueError as e:
        errors.append(f"Invalid date format: {str(e)}")
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings
    )

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage"""
    # Remove dangerous characters
    filename = re.sub(r'[<>:"|?*\\/]', '', filename)
    
    # Remove leading/trailing spaces and dots
    filename = filename.strip(' .')
    
    # Limit length
    if len(filename) > 255:
        filename = filename[:255]
    
    # Ensure filename is not empty
    if not filename:
        filename = "unnamed_file"
    
    return filename

def sanitize_text(text: str, max_length: int = 10000) -> str:
    """Sanitize text content"""
    # Remove null bytes
    text = text.replace('\x00', '')
    
    # Limit length
    if len(text) > max_length:
        text = text[:max_length]
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()
