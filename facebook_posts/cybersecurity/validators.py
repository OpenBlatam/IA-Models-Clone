from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import re
import hashlib
import json
import base64
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass
import ipaddress
from urllib.parse import urlparse
import email_validator
from email_validator import validate_email, EmailNotValidError
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Input validation utilities with proper async/def distinction.
Def for CPU-bound validation logic.
"""


@dataclass
class ValidationConfig:
    """Configuration for validation operations."""
    max_length: int = 1000
    min_length: int = 1
    allowed_chars: str = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
    block_sql_injection: bool = True
    block_xss: bool = True
    block_path_traversal: bool = True
    strict_mode: bool = True

def validate_input(data: str, config: ValidationConfig) -> Dict[str, Any]:
    """Validate input data for security issues."""
    issues = []
    is_valid = True
    
    # Length validation
    if len(data) < config.min_length:
        issues.append(f"Input too short (minimum {config.min_length} characters)")
        is_valid = False
    
    if len(data) > config.max_length:
        issues.append(f"Input too long (maximum {config.max_length} characters)")
        is_valid = False
    
    # Character validation
    if config.strict_mode:
        invalid_chars = [char for char in data if char not in config.allowed_chars]
        if invalid_chars:
            issues.append(f"Invalid characters found: {set(invalid_chars)}")
            is_valid = False
    
    # SQL Injection detection
    if config.block_sql_injection:
        sql_patterns = [
            r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)",
            r"(\b(OR|AND)\b\s+\d+\s*=\s*\d+)",
            r"(\b(OR|AND)\b\s+['\"]\w+['\"]\s*=\s*['\"]\w+['\"])",
            r"(--|#|/\*|\*/)",
            r"(\b(WAITFOR|DELAY)\b)",
            r"(\b(BENCHMARK|SLEEP)\b)"
        ]
        
        for pattern in sql_patterns:
            if re.search(pattern, data, re.IGNORECASE):
                issues.append("Potential SQL injection detected")
                is_valid = False
                break
    
    # XSS detection
    if config.block_xss:
        xss_patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"on\w+\s*=",
            r"<iframe[^>]*>",
            r"<object[^>]*>",
            r"<embed[^>]*>",
            r"<form[^>]*>",
            r"<input[^>]*>",
            r"<textarea[^>]*>",
            r"<select[^>]*>"
        ]
        
        for pattern in xss_patterns:
            if re.search(pattern, data, re.IGNORECASE):
                issues.append("Potential XSS attack detected")
                is_valid = False
                break
    
    # Path traversal detection
    if config.block_path_traversal:
        path_patterns = [
            r"\.\./",
            r"\.\.\\",
            r"\.\.%2f",
            r"\.\.%5c",
            r"\.\.%2e%2e",
            r"\.\.%252e%252e"
        ]
        
        for pattern in path_patterns:
            if re.search(pattern, data, re.IGNORECASE):
                issues.append("Potential path traversal attack detected")
                is_valid = False
                break
    
    return {
        "is_valid": is_valid,
        "issues": issues,
        "length": len(data),
        "has_special_chars": bool(re.search(r'[^a-zA-Z0-9\s]', data))
    }

def sanitize_data(data: str, config: ValidationConfig) -> str:
    """Sanitize input data by removing dangerous content."""
    sanitized = data
    
    # Remove HTML tags
    sanitized = re.sub(r'<[^>]+>', '', sanitized)
    
    # Remove JavaScript
    sanitized = re.sub(r'javascript:', '', sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r'<script[^>]*>.*?</script>', '', sanitized, flags=re.IGNORECASE | re.DOTALL)
    
    # Remove event handlers
    sanitized = re.sub(r'on\w+\s*=\s*["\'][^"\']*["\']', '', sanitized, flags=re.IGNORECASE)
    
    # Remove SQL keywords
    if config.block_sql_injection:
        sql_keywords = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER', 'EXEC', 'UNION']
        for keyword in sql_keywords:
            sanitized = re.sub(rf'\b{keyword}\b', '', sanitized, flags=re.IGNORECASE)
    
    # Remove path traversal
    if config.block_path_traversal:
        sanitized = re.sub(r'\.\./', '', sanitized)
        sanitized = re.sub(r'\.\.\\', '', sanitized)
    
    # Truncate if too long
    if len(sanitized) > config.max_length:
        sanitized = sanitized[:config.max_length]
    
    return sanitized.strip()

def check_file_integrity(file_path: str, expected_hash: str, algorithm: str = "sha256") -> Dict[str, Any]:
    """Check file integrity using hash comparison."""
    try:
        hash_obj = hashlib.new(algorithm)
        
        with open(file_path, 'rb') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            for chunk in iter(lambda: f.read(4096), b""):
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                hash_obj.update(chunk)
        
        actual_hash = hash_obj.hexdigest()
        is_valid = actual_hash.lower() == expected_hash.lower()
        
        return {
            "file_path": file_path,
            "expected_hash": expected_hash,
            "actual_hash": actual_hash,
            "algorithm": algorithm,
            "is_valid": is_valid,
            "file_size": f.tell() if 'f' in locals() else 0
        }
        
    except Exception as e:
        return {
            "file_path": file_path,
            "error": str(e),
            "is_valid": False
        }

def validate_credentials(username: str, password: str, config: ValidationConfig) -> Dict[str, Any]:
    """Validate username and password for security requirements."""
    issues = []
    is_valid = True
    
    # Username validation
    if len(username) < 3:
        issues.append("Username too short (minimum 3 characters)")
        is_valid = False
    
    if len(username) > 50:
        issues.append("Username too long (maximum 50 characters)")
        is_valid = False
    
    if not re.match(r'^[a-zA-Z0-9_-]+$', username):
        issues.append("Username contains invalid characters")
        is_valid = False
    
    # Password validation
    if len(password) < 8:
        issues.append("Password too short (minimum 8 characters)")
        is_valid = False
    
    if len(password) > 128:
        issues.append("Password too long (maximum 128 characters)")
        is_valid = False
    
    # Password strength checks
    if not re.search(r'[A-Z]', password):
        issues.append("Password must contain uppercase letter")
        is_valid = False
    
    if not re.search(r'[a-z]', password):
        issues.append("Password must contain lowercase letter")
        is_valid = False
    
    if not re.search(r'\d', password):
        issues.append("Password must contain digit")
        is_valid = False
    
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        issues.append("Password must contain special character")
        is_valid = False
    
    # Check for common weak passwords
    weak_passwords = [
        'password', '123456', 'qwerty', 'admin', 'letmein',
        'welcome', 'monkey', 'dragon', 'master', 'football'
    ]
    
    if password.lower() in weak_passwords:
        issues.append("Password is too common")
        is_valid = False
    
    return {
        "is_valid": is_valid,
        "issues": issues,
        "username_length": len(username),
        "password_length": len(password),
        "password_strength": calculate_password_strength(password)
    }

def calculate_password_strength(password: str) -> Dict[str, Any]:
    """Calculate password strength score."""
    score = 0
    feedback = []
    
    # Length bonus
    if len(password) >= 8:
        score += 1
    if len(password) >= 12:
        score += 1
    if len(password) >= 16:
        score += 1
    
    # Character variety bonus
    if re.search(r'[a-z]', password):
        score += 1
    if re.search(r'[A-Z]', password):
        score += 1
    if re.search(r'\d', password):
        score += 1
    if re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        score += 1
    
    # Deductions
    if re.search(r'(.)\1{2,}', password):  # Repeated characters
        score -= 1
        feedback.append("Avoid repeated characters")
    
    if re.search(r'(123|abc|qwe)', password.lower()):  # Sequential patterns
        score -= 1
        feedback.append("Avoid sequential patterns")
    
    # Determine strength level
    if score <= 2:
        strength = "weak"
    elif score <= 4:
        strength = "medium"
    elif score <= 6:
        strength = "strong"
    else:
        strength = "very_strong"
    
    return {
        "score": max(0, score),
        "strength": strength,
        "feedback": feedback
    }

def validate_email_address(email: str) -> Dict[str, Any]:
    """Validate email address format and security."""
    try:
        # Use email-validator library
        valid = validate_email(email)
        email = valid.email
        
        # Additional security checks
        issues = []
        
        # Check for disposable email domains
        disposable_domains = [
            'tempmail.org', '10minutemail.com', 'guerrillamail.com',
            'mailinator.com', 'throwaway.email', 'temp-mail.org'
        ]
        
        domain = email.split('@')[1].lower()
        if domain in disposable_domains:
            issues.append("Disposable email domain detected")
        
        # Check for suspicious patterns
        if re.search(r'[<>"\']', email):
            issues.append("Email contains invalid characters")
        
        return {
            "email": email,
            "is_valid": True,
            "domain": domain,
            "issues": issues,
            "is_suspicious": len(issues) > 0
        }
        
    except EmailNotValidError as e:
        return {
            "email": email,
            "is_valid": False,
            "error": str(e),
            "issues": ["Invalid email format"]
        }

def validate_ip_address(ip: str) -> Dict[str, Any]:
    """Validate IP address and check for security issues."""
    try:
        ip_obj = ipaddress.ip_address(ip)
        
        issues = []
        
        # Check for private IP ranges
        if ip_obj.is_private:
            issues.append("Private IP address")
        
        # Check for loopback
        if ip_obj.is_loopback:
            issues.append("Loopback address")
        
        # Check for multicast
        if ip_obj.is_multicast:
            issues.append("Multicast address")
        
        # Check for reserved ranges
        if ip_obj.is_reserved:
            issues.append("Reserved IP address")
        
        return {
            "ip": str(ip_obj),
            "version": ip_obj.version,
            "is_valid": True,
            "is_private": ip_obj.is_private,
            "is_loopback": ip_obj.is_loopback,
            "is_multicast": ip_obj.is_multicast,
            "is_reserved": ip_obj.is_reserved,
            "issues": issues
        }
        
    except ValueError as e:
        return {
            "ip": ip,
            "is_valid": False,
            "error": str(e)
        }

def validate_url(url: str) -> Dict[str, Any]:
    """Validate URL format and security."""
    try:
        parsed = urlparse(url)
        
        issues = []
        
        # Check for required components
        if not parsed.scheme:
            issues.append("Missing scheme (http/https)")
        
        if not parsed.netloc:
            issues.append("Missing domain")
        
        # Check for secure protocols
        if parsed.scheme not in ['http', 'https', 'ftp', 'sftp']:
            issues.append("Unsupported protocol")
        
        # Check for suspicious patterns
        if re.search(r'javascript:', url, re.IGNORECASE):
            issues.append("JavaScript protocol detected")
        
        if re.search(r'data:', url, re.IGNORECASE):
            issues.append("Data URI detected")
        
        # Check for IP addresses in URL
        if re.search(r'\d+\.\d+\.\d+\.\d+', parsed.netloc):
            issues.append("IP address in URL")
        
        return {
            "url": url,
            "scheme": parsed.scheme,
            "netloc": parsed.netloc,
            "path": parsed.path,
            "query": parsed.query,
            "fragment": parsed.fragment,
            "is_valid": len(issues) == 0,
            "issues": issues
        }
        
    except Exception as e:
        return {
            "url": url,
            "is_valid": False,
            "error": str(e)
        }

# Named exports for main functionality
__all__ = [
    'validate_input',
    'sanitize_data',
    'check_file_integrity',
    'validate_credentials',
    'calculate_password_strength',
    'validate_email_address',
    'validate_ip_address',
    'validate_url',
    'ValidationConfig'
] 