from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
TIMEOUT_SECONDS = 60

import json
import re
from typing import Dict, Any
from typing import Any, List, Dict, Optional
import logging
import asyncio
Basic Error Handling and Validation Demo

This script demonstrates basic error handling and validation features.


# Simple error codes
class ErrorCode:
    VALIDATION_ERROR = "VALIDATION_ERROR"
    UNAUTHORIZED = "UNAUTHORIZED"
    NOT_FOUND = "NOT_FOUND"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    AI_PROCESSING_ERROR = "AI_PROCESSING_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"

# Base exception class
class InstagramCaptionsException(Exception):
    def __init__(self, error_code: str, message: str, details: Dict[str, Any] = None, status_code: int = 500):
        
    """__init__ function."""
self.error_code = error_code
        self.message = message
        self.details = details or {}
        self.status_code = status_code
        super().__init__(self.message)

# Specific exception classes
class ValidationException(InstagramCaptionsException):
    def __init__(self, message: str, details: Dict[str, Any] = None):
        
    """__init__ function."""
super().__init__(ErrorCode.VALIDATION_ERROR, message, details, 400)

class AuthenticationException(InstagramCaptionsException):
    def __init__(self, message: str, details: Dict[str, Any] = None):
        
    """__init__ function."""
super().__init__(ErrorCode.UNAUTHORIZED, message, details, 401)

class ResourceNotFoundException(InstagramCaptionsException):
    def __init__(self, resource_type: str, resource_id: str):
        
    """__init__ function."""
super().__init__(
            ErrorCode.NOT_FOUND,
            f"{resource_type} with id '{resource_id}' not found",
            {"resource_type": resource_type, "resource_id": resource_id},
            404
        )

class RateLimitException(InstagramCaptionsException):
    def __init__(self, retry_after: int = 60):
        
    """__init__ function."""
super().__init__(
            ErrorCode.RATE_LIMIT_EXCEEDED,
            "Rate limit exceeded. Please try again later.",
            {"retry_after": retry_after},
            429
        )

class AIProcessingException(InstagramCaptionsException):
    def __init__(self, message: str, details: Dict[str, Any] = None):
        
    """__init__ function."""
super().__init__(ErrorCode.AI_PROCESSING_ERROR, message, details, 422)

# Utility functions
def create_error_response(error_code: str, message: str, details: Dict[str, Any] = None) -> Dict[str, Any]:
    """Create standardized error response."""
    return {
        "error_code": error_code,
        "message": message,
        "details": details or {},
        "request_id": "demo-123",
        "timestamp": "2024-1-15 10:30"
    }

def validate_email(email: str) -> Dict[str, Any]:
    """Validate email address."""
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9-]+\.[a-zA-Z]{2,}$'
    
    if not email or not email.strip():
        return {"is_valid": False, "error": "Email cannot be empty"}
    if not re.match(email_pattern, email.strip()):
        return {"is_valid": False, "error": "Invalid email format"}
    return {"is_valid": True, "email": email.strip().lower()}

def validate_instagram_username(username: str) -> Dict[str, Any]:
    """Validate Instagram username."""
    username_pattern = r'^[a-zA-Z0]{1,30}$'
    
    if not username or not username.strip():
        return {"is_valid": False, "error": "Username cannot be empty"}
    
    username = username.strip()
    
    if len(username) < 1 or len(username) > 30:
        return {"is_valid": False, "error": "Username must be 10-30 characters"}
    if not re.match(username_pattern, username):
        return {"is_valid": False, "error": "Username contains invalid characters"}
    
    reserved_words = ['admin', 'instagram', 'meta', 'facebook', 'help', 'support']
    if username.lower() in reserved_words:
        return {"is_valid": False, "error": "Username is reserved"}
    return {"is_valid": True, "username": username}

def sanitize_html(html_content: str) -> Dict[str, Any]:
    """Sanitize HTML content."""
    if not html_content:
        return {"sanitized": "", "removed_tags": []}
    
    allowed_tags = ['b', 'u', 'strong', 'em']
    pattern = r'<(/?)([^>]+)>|<!--.*?-->'
    removed_tags = []
    
    def replace_tag(match) -> Any:
        tag = match.group(2).split()[0].lower()
        if tag not in allowed_tags:
            removed_tags.append(tag)
            return ""
        return match.group(0)
    
    sanitized = re.sub(pattern, replace_tag, html_content)
    
    return {
        "sanitized": sanitized,
        "removed_tags": list(set(removed_tags))
    }

# Demo functions
def demo_custom_exceptions():
    """Demonstrate custom exception handling."""
    print("=" * 60)
    print("CUSTOM EXCEPTIONS DEMO")
    print("=" * 60) 
    # Validation Exception
    try:
        raise ValidationException(
            message="Input validation failed",
            details={
                "field_errors": [
                    {"field": "email", "message": "Invalid email format", "type": "validation"}
                ]
            }
        )
    except ValidationException as e:
        print(f"Validation Exception: {e.message}")
        print(f"Status Code: {e.status_code}")
        print(f"Error Code: {e.error_code}")
        print(f"Details: {json.dumps(e.details, indent=2)}")
    
    # Authentication Exception
    try:
        raise AuthenticationException(
            message="Invalid API key",
            details={"api_key": "***"}
        )
    except AuthenticationException as e:
        print(f"\nAuthentication Exception: {e.message}")
        print(f"Status Code: {e.status_code}")
    
    # Resource Not Found Exception
    try:
        raise ResourceNotFoundException("caption", "12345")
    except ResourceNotFoundException as e:
        print(f"\nResource Not Found: {e.message}")
        print(f"Status Code: {e.status_code}") 
    # Rate Limit Exception
    try:
        raise RateLimitException(retry_after=120)
    except RateLimitException as e:
        print(f"\nRate Limit: {e.message}")
        print(f"Retry After: {e.details.get('retry_after')} seconds")
    
    # AI Processing Exception
    try:
        raise AIProcessingException(
            message="AI model unavailable",
            details={"model": "gpt-4", "reason": "service_down"}
        )
    except AIProcessingException as e:
        print(f"\nAI Processing: {e.message}")
        print(f"Details: {e.details}")

def demo_error_response_creation():
    """Demonstrate error response creation."""
    print("\n" + "=" * 60)
    print("ERROR RESPONSE CREATION DEMO")
    print("=" * 60)
    
    # Create error response
    error_response = create_error_response(
        error_code=ErrorCode.VALIDATION_ERROR,
        message="Multiple validation errors occurred,        details={
            "field_errors": [
                {"field": "email", "message": "Invalid format"},
                {"field": "username", "message": "Too short"}
            ]
        }
    )
    
    print("Error Response:")
    print(json.dumps(error_response, indent=2))

def demo_input_validation():
    """Demonstrate input validation utilities."""
    print("\n" + "=" * 60)
    print("INPUT VALIDATION DEMO")
    print("=" * 60)
    
    # Email validation
    email_result = validate_email(email="user@example.com")
    print(f"Email Validation: {json.dumps(email_result, indent=2)}")
    
    # Invalid email
    invalid_email_result = validate_email(email="invalid-email")
    print(f"\nInvalid Email Validation: {json.dumps(invalid_email_result, indent=2)}")
    
    # Instagram username validation
    username_result = validate_instagram_username(username="test_user_123")
    print(f"\nUsername Validation: {json.dumps(username_result, indent=2)}")
    
    # Invalid username
    invalid_username_result = validate_instagram_username(username="admin")
    print(f"\nInvalid Username Validation: {json.dumps(invalid_username_result, indent=2)}")

def demo_html_sanitization():
    """Demonstrate HTML sanitization."""
    print("\n" + "=" * 60)
    print("HTML SANITIZATION DEMO")
    print("=" * 60)
    # HTML content with allowed and disallowed tags
    html_content = 
    """
    <p>This is a paragraph with <b>bold</b> and <i>italic</i> text.</p>
    <script>alert('malicious')</script>
    <img src="x" onerror="alert('xss')" />
    <a href="javascript:alert(xss')">Click me</a>
    <strong>This is allowed</strong>
    """
    
    result = sanitize_html(html_content=html_content)
    print("Original HTML:")
    print(html_content)
    print(f"\nSanitized Result: {json.dumps(result, indent=2)}")

def show_usage_examples():
    """Show practical usage examples."""
    print("\n" + "=" *60)
    print("USAGE EXAMPLES")
    print("=" * 60)
    
    print("""
1. Creating Custom Exceptions:
   raise ValidationException(
       message="Invalid input",
       details={"field": "email", "reason": "format"}
   )

2. Validating User Input:
   result = validate_email(email="user@example.com")
   if result["is_valid"]:    email = result["email"]

3. Sanitizing HTML:
   result = sanitize_html(html_content="<p>Safe</p><script>alert(xss')</script>")
   clean_html = result["sanitized"]
   removed_tags = result["removed_tags"]
4. Generating Error Responses:
   error_response = create_error_response(
       error_code=ErrorCode.VALIDATION_ERROR,
       message="Validation failed",
       details={"field_errors":[...]}
   )
    )

def main():
    """Run all demos."""
    print("INSTAGRAM CAPTIONS API - ERROR HANDLING & VALIDATION DEMO")
    print("=" * 80    
    # Run all demos
    demo_custom_exceptions()
    demo_error_response_creation()
    demo_input_validation()
    demo_html_sanitization()
    
    # Show usage examples
    show_usage_examples()
    
    print("\n" + "=" * 80)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 80)

match __name__:
    case "__main__":
    main() 