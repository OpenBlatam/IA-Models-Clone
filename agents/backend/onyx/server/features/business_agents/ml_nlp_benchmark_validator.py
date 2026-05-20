"""
ML NLP Benchmark Validator System
Real, working validation and sanitization for ML NLP Benchmark system
"""

import re
import html
import json
import hashlib
import base64
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class MLNLPBenchmarkValidator:
    """Advanced validation and sanitization system"""
    
    def __init__(self):
        self.validation_rules = {
            "text": {
                "min_length": 1,
                "max_length": 1000000,
                "allowed_chars": r"[\w\s.,!?;:()\-'\"@#$%&*+=<>/\\|`~]",
                "forbidden_patterns": [
                    r"<script.*?>.*?</script>",
                    r"javascript:",
                    r"vbscript:",
                    r"on\w+\s*=",
                    r"<iframe.*?>",
                    r"<object.*?>",
                    r"<embed.*?>"
                ]
            },
            "email": {
                "pattern": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
                "max_length": 254
            },
            "url": {
                "pattern": r"^https?://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(/.*)?$",
                "max_length": 2048
            },
            "api_key": {
                "pattern": r"^[a-zA-Z0-9_-]{20,100}$",
                "min_length": 20,
                "max_length": 100
            },
            "filename": {
                "pattern": r"^[a-zA-Z0-9._-]+$",
                "max_length": 255,
                "forbidden_extensions": [".exe", ".bat", ".cmd", ".scr", ".pif", ".com"]
            }
        }
        
        self.sanitization_rules = {
            "html": True,
            "sql_injection": True,
            "xss": True,
            "path_traversal": True,
            "command_injection": True
        }
    
    def validate_text(self, text: str, rules: Optional[Dict[str, Any]] = None) -> Tuple[bool, List[str]]:
        """Validate text input"""
        if not isinstance(text, str):
            return False, ["Text must be a string"]
        
        rules = rules or self.validation_rules["text"]
        errors = []
        
        # Length validation
        if len(text) < rules.get("min_length", 1):
            errors.append(f"Text too short (minimum {rules['min_length']} characters)")
        
        if len(text) > rules.get("max_length", 1000000):
            errors.append(f"Text too long (maximum {rules['max_length']} characters)")
        
        # Character validation
        allowed_chars = rules.get("allowed_chars")
        if allowed_chars:
            invalid_chars = re.findall(f"[^{allowed_chars[1:-1]}]", text)
            if invalid_chars:
                errors.append(f"Invalid characters found: {set(invalid_chars)}")
        
        # Forbidden patterns
        forbidden_patterns = rules.get("forbidden_patterns", [])
        for pattern in forbidden_patterns:
            if re.search(pattern, text, re.IGNORECASE | re.DOTALL):
                errors.append(f"Forbidden pattern detected: {pattern}")
        
        return len(errors) == 0, errors
    
    def validate_email(self, email: str) -> Tuple[bool, List[str]]:
        """Validate email address"""
        if not isinstance(email, str):
            return False, ["Email must be a string"]
        
        rules = self.validation_rules["email"]
        errors = []
        
        # Length validation
        if len(email) > rules["max_length"]:
            errors.append(f"Email too long (maximum {rules['max_length']} characters)")
        
        # Pattern validation
        if not re.match(rules["pattern"], email):
            errors.append("Invalid email format")
        
        return len(errors) == 0, errors
    
    def validate_url(self, url: str) -> Tuple[bool, List[str]]:
        """Validate URL"""
        if not isinstance(url, str):
            return False, ["URL must be a string"]
        
        rules = self.validation_rules["url"]
        errors = []
        
        # Length validation
        if len(url) > rules["max_length"]:
            errors.append(f"URL too long (maximum {rules['max_length']} characters)")
        
        # Pattern validation
        if not re.match(rules["pattern"], url):
            errors.append("Invalid URL format")
        
        return len(errors) == 0, errors
    
    def validate_api_key(self, api_key: str) -> Tuple[bool, List[str]]:
        """Validate API key"""
        if not isinstance(api_key, str):
            return False, ["API key must be a string"]
        
        rules = self.validation_rules["api_key"]
        errors = []
        
        # Length validation
        if len(api_key) < rules["min_length"]:
            errors.append(f"API key too short (minimum {rules['min_length']} characters)")
        
        if len(api_key) > rules["max_length"]:
            errors.append(f"API key too long (maximum {rules['max_length']} characters)")
        
        # Pattern validation
        if not re.match(rules["pattern"], api_key):
            errors.append("Invalid API key format")
        
        return len(errors) == 0, errors
    
    def validate_filename(self, filename: str) -> Tuple[bool, List[str]]:
        """Validate filename"""
        if not isinstance(filename, str):
            return False, ["Filename must be a string"]
        
        rules = self.validation_rules["filename"]
        errors = []
        
        # Length validation
        if len(filename) > rules["max_length"]:
            errors.append(f"Filename too long (maximum {rules['max_length']} characters)")
        
        # Pattern validation
        if not re.match(rules["pattern"], filename):
            errors.append("Invalid filename format")
        
        # Extension validation
        forbidden_extensions = rules["forbidden_extensions"]
        for ext in forbidden_extensions:
            if filename.lower().endswith(ext):
                errors.append(f"Forbidden file extension: {ext}")
        
        return len(errors) == 0, errors
    
    def sanitize_text(self, text: str, rules: Optional[Dict[str, bool]] = None) -> str:
        """Sanitize text input"""
        if not isinstance(text, str):
            return str(text)
        
        rules = rules or self.sanitization_rules
        sanitized = text
        
        # HTML sanitization
        if rules.get("html", True):
            sanitized = html.escape(sanitized)
        
        # XSS prevention
        if rules.get("xss", True):
            # Remove script tags
            sanitized = re.sub(r"<script.*?>.*?</script>", "", sanitized, flags=re.IGNORECASE | re.DOTALL)
            # Remove javascript: URLs
            sanitized = re.sub(r"javascript:", "", sanitized, flags=re.IGNORECASE)
            # Remove vbscript: URLs
            sanitized = re.sub(r"vbscript:", "", sanitized, flags=re.IGNORECASE)
            # Remove event handlers
            sanitized = re.sub(r"on\w+\s*=", "", sanitized, flags=re.IGNORECASE)
        
        # SQL injection prevention
        if rules.get("sql_injection", True):
            # Remove common SQL injection patterns
            sql_patterns = [
                r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)",
                r"(--|#|/\*|\*/)",
                r"(\b(OR|AND)\s+\d+\s*=\s*\d+)",
                r"(\b(OR|AND)\s+'.*?'\s*=\s*'.*?')"
            ]
            for pattern in sql_patterns:
                sanitized = re.sub(pattern, "", sanitized, flags=re.IGNORECASE)
        
        # Path traversal prevention
        if rules.get("path_traversal", True):
            sanitized = re.sub(r"\.\./|\.\.\\|\.\.%2f|\.\.%5c", "", sanitized, flags=re.IGNORECASE)
        
        # Command injection prevention
        if rules.get("command_injection", True):
            # Remove command injection patterns
            cmd_patterns = [
                r"[;&|`$]",
                r"\b(cat|ls|dir|type|more|less|head|tail|grep|find|locate|which|whereis)\b",
                r"\b(rm|del|mv|cp|chmod|chown|sudo|su)\b"
            ]
            for pattern in cmd_patterns:
                sanitized = re.sub(pattern, "", sanitized, flags=re.IGNORECASE)
        
        return sanitized.strip()
    
    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename"""
        if not isinstance(filename, str):
            return "sanitized_file"
        
        # Remove path separators
        sanitized = re.sub(r"[\\/:*?\"<>|]", "_", filename)
        
        # Remove leading/trailing dots and spaces
        sanitized = sanitized.strip(". ")
        
        # Limit length
        if len(sanitized) > 255:
            name, ext = sanitized.rsplit(".", 1) if "." in sanitized else (sanitized, "")
            max_name_length = 255 - len(ext) - 1 if ext else 255
            sanitized = name[:max_name_length] + ("." + ext if ext else "")
        
        # Ensure it's not empty
        if not sanitized:
            sanitized = "sanitized_file"
        
        return sanitized
    
    def validate_json(self, json_string: str) -> Tuple[bool, List[str]]:
        """Validate JSON string"""
        if not isinstance(json_string, str):
            return False, ["JSON must be a string"]
        
        errors = []
        try:
            json.loads(json_string)
        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON: {str(e)}")
        
        return len(errors) == 0, errors
    
    def validate_request_data(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate request data"""
        errors = []
        
        # Check required fields
        required_fields = ["text"]
        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")
        
        # Validate text field
        if "text" in data:
            is_valid, text_errors = self.validate_text(data["text"])
            if not is_valid:
                errors.extend([f"Text validation error: {error}" for error in text_errors])
        
        # Validate optional fields
        if "analysis_type" in data:
            valid_types = ["nlp", "ml", "benchmark", "comprehensive"]
            if data["analysis_type"] not in valid_types:
                errors.append(f"Invalid analysis_type. Must be one of: {valid_types}")
        
        if "method" in data:
            valid_methods = ["default", "fast", "accurate", "balanced"]
            if data["method"] not in valid_methods:
                errors.append(f"Invalid method. Must be one of: {valid_methods}")
        
        return len(errors) == 0, errors
    
    def sanitize_request_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize request data"""
        sanitized = {}
        
        for key, value in data.items():
            if isinstance(value, str):
                sanitized[key] = self.sanitize_text(value)
            elif isinstance(value, dict):
                sanitized[key] = self.sanitize_request_data(value)
            elif isinstance(value, list):
                sanitized[key] = [
                    self.sanitize_text(item) if isinstance(item, str) else item
                    for item in value
                ]
            else:
                sanitized[key] = value
        
        return sanitized
    
    def generate_secure_token(self, length: int = 32) -> str:
        """Generate secure random token"""
        import secrets
        return secrets.token_urlsafe(length)
    
    def hash_password(self, password: str, salt: Optional[str] = None) -> Tuple[str, str]:
        """Hash password with salt"""
        import hashlib
        import secrets
        
        if salt is None:
            salt = secrets.token_hex(16)
        
        # Combine password and salt
        combined = password + salt
        
        # Hash using SHA-256
        hashed = hashlib.sha256(combined.encode()).hexdigest()
        
        return hashed, salt
    
    def verify_password(self, password: str, hashed: str, salt: str) -> bool:
        """Verify password against hash"""
        import hashlib
        
        # Hash the provided password with the salt
        combined = password + salt
        computed_hash = hashlib.sha256(combined.encode()).hexdigest()
        
        return computed_hash == hashed
    
    def validate_file_upload(self, filename: str, content_type: str, file_size: int) -> Tuple[bool, List[str]]:
        """Validate file upload"""
        errors = []
        
        # Validate filename
        is_valid, filename_errors = self.validate_filename(filename)
        if not is_valid:
            errors.extend([f"Filename validation error: {error}" for error in filename_errors])
        
        # Validate content type
        allowed_types = [
            "text/plain",
            "text/csv",
            "application/json",
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation"
        ]
        
        if content_type not in allowed_types:
            errors.append(f"Invalid content type: {content_type}")
        
        # Validate file size (10MB limit)
        max_size = 10 * 1024 * 1024  # 10MB
        if file_size > max_size:
            errors.append(f"File too large (maximum {max_size} bytes)")
        
        return len(errors) == 0, errors
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get validation rules summary"""
        return {
            "validation_rules": self.validation_rules,
            "sanitization_rules": self.sanitization_rules,
            "supported_validations": [
                "text", "email", "url", "api_key", "filename", "json", "request_data", "file_upload"
            ],
            "supported_sanitizations": [
                "html", "xss", "sql_injection", "path_traversal", "command_injection"
            ]
        }

# Global validator instance
ml_nlp_benchmark_validator = MLNLPBenchmarkValidator()

def validate_text(text: str, rules: Optional[Dict[str, Any]] = None) -> Tuple[bool, List[str]]:
    """Validate text input"""
    return ml_nlp_benchmark_validator.validate_text(text, rules)

def validate_email(email: str) -> Tuple[bool, List[str]]:
    """Validate email address"""
    return ml_nlp_benchmark_validator.validate_email(email)

def validate_url(url: str) -> Tuple[bool, List[str]]:
    """Validate URL"""
    return ml_nlp_benchmark_validator.validate_url(url)

def validate_api_key(api_key: str) -> Tuple[bool, List[str]]:
    """Validate API key"""
    return ml_nlp_benchmark_validator.validate_api_key(api_key)

def validate_filename(filename: str) -> Tuple[bool, List[str]]:
    """Validate filename"""
    return ml_nlp_benchmark_validator.validate_filename(filename)

def validate_json(json_string: str) -> Tuple[bool, List[str]]:
    """Validate JSON string"""
    return ml_nlp_benchmark_validator.validate_json(json_string)

def validate_request_data(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate request data"""
    return ml_nlp_benchmark_validator.validate_request_data(data)

def validate_file_upload(filename: str, content_type: str, file_size: int) -> Tuple[bool, List[str]]:
    """Validate file upload"""
    return ml_nlp_benchmark_validator.validate_file_upload(filename, content_type, file_size)

def sanitize_text(text: str, rules: Optional[Dict[str, bool]] = None) -> str:
    """Sanitize text input"""
    return ml_nlp_benchmark_validator.sanitize_text(text, rules)

def sanitize_filename(filename: str) -> str:
    """Sanitize filename"""
    return ml_nlp_benchmark_validator.sanitize_filename(filename)

def sanitize_request_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize request data"""
    return ml_nlp_benchmark_validator.sanitize_request_data(data)

def generate_secure_token(length: int = 32) -> str:
    """Generate secure random token"""
    return ml_nlp_benchmark_validator.generate_secure_token(length)

def hash_password(password: str, salt: Optional[str] = None) -> Tuple[str, str]:
    """Hash password with salt"""
    return ml_nlp_benchmark_validator.hash_password(password, salt)

def verify_password(password: str, hashed: str, salt: str) -> bool:
    """Verify password against hash"""
    return ml_nlp_benchmark_validator.verify_password(password, hashed, salt)

def get_validation_summary() -> Dict[str, Any]:
    """Get validation rules summary"""
    return ml_nlp_benchmark_validator.get_validation_summary()











