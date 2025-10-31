"""
Security Integration for Test Generation System
==============================================

This module provides comprehensive security integration with the existing
security middleware system, ensuring secure test generation capabilities.
"""

import asyncio
import hashlib
import hmac
import json
import time
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import jwt
from pathlib import Path

from .base_architecture import TestCase, TestGenerationConfig
from .unified_api import TestGenerationAPI, create_api

logger = logging.getLogger(__name__)


@dataclass
class SecurityConfig:
    """Security configuration for test generation system"""
    # Authentication
    jwt_secret: str = "test_generation_secret_key"
    jwt_algorithm: str = "HS256"
    token_expiry_hours: int = 24
    
    # Rate Limiting
    rate_limit_requests: int = 100
    rate_limit_window_minutes: int = 15
    rate_limit_by_user: bool = True
    
    # Security Headers
    content_security_policy: str = "default-src 'self'"
    x_frame_options: str = "DENY"
    x_content_type_options: str = "nosniff"
    referrer_policy: str = "strict-origin-when-cross-origin"
    
    # Input Validation
    max_function_signature_length: int = 1000
    max_docstring_length: int = 5000
    max_test_cases: int = 1000
    allowed_file_extensions: List[str] = field(default_factory=lambda: [".py", ".ts", ".js"])
    
    # Output Security
    sanitize_output: bool = True
    validate_generated_code: bool = True
    block_dangerous_patterns: bool = True
    
    # Logging
    log_security_events: bool = True
    log_sensitive_data: bool = False
    security_log_level: str = "info"


@dataclass
class SecurityContext:
    """Security context for request processing"""
    user_id: Optional[str] = None
    user_role: Optional[str] = None
    user_permissions: List[str] = field(default_factory=list)
    ip_address: str = ""
    user_agent: str = ""
    request_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    session_id: Optional[str] = None


@dataclass
class SecurityMetrics:
    """Security metrics for monitoring"""
    total_requests: int = 0
    authenticated_requests: int = 0
    rate_limited_requests: int = 0
    blocked_requests: int = 0
    suspicious_requests: int = 0
    security_violations: int = 0
    last_reset: datetime = field(default_factory=datetime.now)


class SecurityValidator:
    """Security validator for test generation requests"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.dangerous_patterns = self._load_dangerous_patterns()
    
    def _load_dangerous_patterns(self) -> List[str]:
        """Load dangerous patterns that should be blocked"""
        return [
            r"import\s+os",
            r"import\s+subprocess",
            r"import\s+sys",
            r"exec\s*\(",
            r"eval\s*\(",
            r"__import__\s*\(",
            r"open\s*\(",
            r"file\s*\(",
            r"input\s*\(",
            r"raw_input\s*\(",
            r"compile\s*\(",
            r"reload\s*\(",
            r"getattr\s*\(",
            r"setattr\s*\(",
            r"delattr\s*\(",
            r"hasattr\s*\(",
            r"globals\s*\(",
            r"locals\s*\(",
            r"vars\s*\(",
            r"dir\s*\(",
            r"help\s*\(",
            r"exit\s*\(",
            r"quit\s*\(",
            r"breakpoint\s*\(",
            r"__.*__",
            r"\.__.*__",
            r"getattr\s*\(",
            r"setattr\s*\(",
            r"delattr\s*\(",
            r"hasattr\s*\(",
            r"globals\s*\(",
            r"locals\s*\(",
            r"vars\s*\(",
            r"dir\s*\(",
            r"help\s*\(",
            r"exit\s*\(",
            r"quit\s*\(",
            r"breakpoint\s*\(",
        ]
    
    def validate_function_signature(self, signature: str) -> Tuple[bool, List[str]]:
        """Validate function signature for security"""
        errors = []
        
        # Check length
        if len(signature) > self.config.max_function_signature_length:
            errors.append(f"Function signature too long: {len(signature)} > {self.config.max_function_signature_length}")
        
        # Check for dangerous patterns
        for pattern in self.dangerous_patterns:
            if re.search(pattern, signature, re.IGNORECASE):
                errors.append(f"Dangerous pattern detected in function signature: {pattern}")
        
        # Check for valid Python syntax
        try:
            compile(signature, '<string>', 'exec')
        except SyntaxError as e:
            errors.append(f"Invalid Python syntax in function signature: {e}")
        
        return len(errors) == 0, errors
    
    def validate_docstring(self, docstring: str) -> Tuple[bool, List[str]]:
        """Validate docstring for security"""
        errors = []
        
        # Check length
        if len(docstring) > self.config.max_docstring_length:
            errors.append(f"Docstring too long: {len(docstring)} > {self.config.max_docstring_length}")
        
        # Check for dangerous patterns
        for pattern in self.dangerous_patterns:
            if re.search(pattern, docstring, re.IGNORECASE):
                errors.append(f"Dangerous pattern detected in docstring: {pattern}")
        
        return len(errors) == 0, errors
    
    def validate_test_cases(self, test_cases: List[TestCase]) -> Tuple[bool, List[str]]:
        """Validate test cases for security"""
        errors = []
        
        # Check count
        if len(test_cases) > self.config.max_test_cases:
            errors.append(f"Too many test cases: {len(test_cases)} > {self.config.max_test_cases}")
        
        # Validate each test case
        for i, test_case in enumerate(test_cases):
            # Check test code
            if self.config.block_dangerous_patterns:
                for pattern in self.dangerous_patterns:
                    if re.search(pattern, test_case.test_code, re.IGNORECASE):
                        errors.append(f"Dangerous pattern in test case {i}: {pattern}")
            
            # Check setup code
            if test_case.setup_code:
                for pattern in self.dangerous_patterns:
                    if re.search(pattern, test_case.setup_code, re.IGNORECASE):
                        errors.append(f"Dangerous pattern in setup code for test case {i}: {pattern}")
            
            # Check teardown code
            if test_case.teardown_code:
                for pattern in self.dangerous_patterns:
                    if re.search(pattern, test_case.teardown_code, re.IGNORECASE):
                        errors.append(f"Dangerous pattern in teardown code for test case {i}: {pattern}")
        
        return len(errors) == 0, errors
    
    def sanitize_input(self, input_data: str) -> str:
        """Sanitize input data"""
        if not self.config.sanitize_output:
            return input_data
        
        # Remove dangerous patterns
        sanitized = input_data
        for pattern in self.dangerous_patterns:
            sanitized = re.sub(pattern, "", sanitized, flags=re.IGNORECASE)
        
        # Remove excessive whitespace
        sanitized = re.sub(r'\s+', ' ', sanitized).strip()
        
        return sanitized


class RateLimiter:
    """Rate limiter for test generation requests"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.requests: Dict[str, List[datetime]] = {}
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def is_rate_limited(self, context: SecurityContext) -> Tuple[bool, str]:
        """Check if request should be rate limited"""
        
        # Get identifier for rate limiting
        if self.config.rate_limit_by_user and context.user_id:
            identifier = f"user:{context.user_id}"
        else:
            identifier = f"ip:{context.ip_address}"
        
        now = datetime.now()
        window_start = now - timedelta(minutes=self.config.rate_limit_window_minutes)
        
        # Get existing requests for this identifier
        if identifier not in self.requests:
            self.requests[identifier] = []
        
        # Remove old requests outside the window
        self.requests[identifier] = [
            req_time for req_time in self.requests[identifier]
            if req_time > window_start
        ]
        
        # Check if rate limit exceeded
        if len(self.requests[identifier]) >= self.config.rate_limit_requests:
            return True, f"Rate limit exceeded: {len(self.requests[identifier])} requests in {self.config.rate_limit_window_minutes} minutes"
        
        # Add current request
        self.requests[identifier].append(now)
        
        return False, ""
    
    def get_rate_limit_info(self, context: SecurityContext) -> Dict[str, Any]:
        """Get rate limit information for context"""
        
        if self.config.rate_limit_by_user and context.user_id:
            identifier = f"user:{context.user_id}"
        else:
            identifier = f"ip:{context.ip_address}"
        
        if identifier not in self.requests:
            return {
                "requests_remaining": self.config.rate_limit_requests,
                "window_minutes": self.config.rate_limit_window_minutes,
                "reset_time": datetime.now() + timedelta(minutes=self.config.rate_limit_window_minutes)
            }
        
        now = datetime.now()
        window_start = now - timedelta(minutes=self.config.rate_limit_window_minutes)
        
        # Count requests in current window
        current_requests = len([
            req_time for req_time in self.requests[identifier]
            if req_time > window_start
        ])
        
        return {
            "requests_remaining": max(0, self.config.rate_limit_requests - current_requests),
            "window_minutes": self.config.rate_limit_window_minutes,
            "reset_time": window_start + timedelta(minutes=self.config.rate_limit_window_minutes)
        }


class SecurityAuthenticator:
    """Security authenticator for test generation requests"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def authenticate_request(self, token: str) -> Tuple[bool, Optional[SecurityContext]]:
        """Authenticate request using JWT token"""
        
        try:
            # Decode JWT token
            payload = jwt.decode(
                token,
                self.config.jwt_secret,
                algorithms=[self.config.jwt_algorithm]
            )
            
            # Create security context
            context = SecurityContext(
                user_id=payload.get("sub"),
                user_role=payload.get("role"),
                user_permissions=payload.get("permissions", []),
                session_id=payload.get("session_id")
            )
            
            return True, context
            
        except jwt.ExpiredSignatureError:
            self.logger.warning("JWT token expired")
            return False, None
        except jwt.InvalidTokenError:
            self.logger.warning("Invalid JWT token")
            return False, None
        except Exception as e:
            self.logger.error(f"Authentication error: {e}")
            return False, None
    
    def authorize_request(self, context: SecurityContext, required_permissions: List[str]) -> bool:
        """Authorize request based on user permissions"""
        
        if not context.user_permissions:
            return False
        
        # Check if user has all required permissions
        return all(perm in context.user_permissions for perm in required_permissions)
    
    def generate_token(self, user_id: str, role: str, permissions: List[str]) -> str:
        """Generate JWT token for user"""
        
        payload = {
            "sub": user_id,
            "role": role,
            "permissions": permissions,
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + timedelta(hours=self.config.token_expiry_hours)
        }
        
        return jwt.encode(
            payload,
            self.config.jwt_secret,
            algorithm=self.config.jwt_algorithm
        )


class SecurityLogger:
    """Security logger for test generation system"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.security_events: List[Dict[str, Any]] = []
    
    def log_security_event(
        self,
        event_type: str,
        context: SecurityContext,
        details: Dict[str, Any],
        severity: str = "info"
    ):
        """Log security event"""
        
        if not self.config.log_security_events:
            return
        
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "user_id": context.user_id,
            "ip_address": context.ip_address,
            "user_agent": context.user_agent,
            "request_id": context.request_id,
            "severity": severity,
            "details": details
        }
        
        # Add to in-memory log
        self.security_events.append(event)
        
        # Log to logger
        log_message = f"Security Event: {event_type} - {details}"
        if severity == "error":
            self.logger.error(log_message)
        elif severity == "warn":
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
    
    def get_security_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent security events"""
        return self.security_events[-limit:]


class SecureTestGenerator:
    """Secure test generator with comprehensive security integration"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.api = create_api()
        self.validator = SecurityValidator(config)
        self.rate_limiter = RateLimiter(config)
        self.authenticator = SecurityAuthenticator(config)
        self.logger = SecurityLogger(config)
        self.metrics = SecurityMetrics()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def generate_secure_tests(
        self,
        function_signature: str,
        docstring: str,
        context: SecurityContext,
        test_config: Optional[TestGenerationConfig] = None
    ) -> Dict[str, Any]:
        """Generate tests with comprehensive security validation"""
        
        try:
            # Update metrics
            self.metrics.total_requests += 1
            
            # Log request
            self.logger.log_security_event(
                "test_generation_request",
                context,
                {
                    "function_signature_length": len(function_signature),
                    "docstring_length": len(docstring)
                }
            )
            
            # Check rate limiting
            is_limited, rate_limit_message = self.rate_limiter.is_rate_limited(context)
            if is_limited:
                self.metrics.rate_limited_requests += 1
                self.logger.log_security_event(
                    "rate_limit_exceeded",
                    context,
                    {"message": rate_limit_message},
                    "warn"
                )
                return {
                    "test_cases": [],
                    "error": "Rate limit exceeded",
                    "rate_limit_info": self.rate_limiter.get_rate_limit_info(context),
                    "success": False
                }
            
            # Validate function signature
            is_valid, validation_errors = self.validator.validate_function_signature(function_signature)
            if not is_valid:
                self.metrics.blocked_requests += 1
                self.logger.log_security_event(
                    "invalid_function_signature",
                    context,
                    {"errors": validation_errors},
                    "warn"
                )
                return {
                    "test_cases": [],
                    "error": f"Invalid function signature: {', '.join(validation_errors)}",
                    "success": False
                }
            
            # Validate docstring
            is_valid, validation_errors = self.validator.validate_docstring(docstring)
            if not is_valid:
                self.metrics.blocked_requests += 1
                self.logger.log_security_event(
                    "invalid_docstring",
                    context,
                    {"errors": validation_errors},
                    "warn"
                )
                return {
                    "test_cases": [],
                    "error": f"Invalid docstring: {', '.join(validation_errors)}",
                    "success": False
                }
            
            # Sanitize inputs
            sanitized_signature = self.validator.sanitize_input(function_signature)
            sanitized_docstring = self.validator.sanitize_input(docstring)
            
            # Generate tests
            result = await self.api.generate_tests(
                sanitized_signature,
                sanitized_docstring,
                "enhanced",
                test_config
            )
            
            if not result["success"]:
                self.metrics.security_violations += 1
                self.logger.log_security_event(
                    "test_generation_failed",
                    context,
                    {"error": result.get("error", "Unknown error")},
                    "error"
                )
                return result
            
            # Validate generated test cases
            is_valid, validation_errors = self.validator.validate_test_cases(result["test_cases"])
            if not is_valid:
                self.metrics.blocked_requests += 1
                self.logger.log_security_event(
                    "invalid_test_cases",
                    context,
                    {"errors": validation_errors},
                    "warn"
                )
                return {
                    "test_cases": [],
                    "error": f"Invalid test cases generated: {', '.join(validation_errors)}",
                    "success": False
                }
            
            # Log successful generation
            self.metrics.authenticated_requests += 1
            self.logger.log_security_event(
                "test_generation_success",
                context,
                {
                    "test_count": len(result["test_cases"]),
                    "generation_time": result.get("generation_time", 0)
                }
            )
            
            return {
                **result,
                "security_validated": True,
                "rate_limit_info": self.rate_limiter.get_rate_limit_info(context)
            }
            
        except Exception as e:
            self.metrics.security_violations += 1
            self.logger.log_security_event(
                "security_error",
                context,
                {"error": str(e)},
                "error"
            )
            return {
                "test_cases": [],
                "error": f"Security error: {str(e)}",
                "success": False
            }
    
    def get_security_metrics(self) -> Dict[str, Any]:
        """Get security metrics"""
        return {
            "total_requests": self.metrics.total_requests,
            "authenticated_requests": self.metrics.authenticated_requests,
            "rate_limited_requests": self.metrics.rate_limited_requests,
            "blocked_requests": self.metrics.blocked_requests,
            "suspicious_requests": self.metrics.suspicious_requests,
            "security_violations": self.metrics.security_violations,
            "last_reset": self.metrics.last_reset.isoformat()
        }
    
    def get_security_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get security events"""
        return self.logger.get_security_events(limit)


# Convenience functions
def create_secure_generator(config: Optional[SecurityConfig] = None) -> SecureTestGenerator:
    """Create a secure test generator"""
    if config is None:
        config = SecurityConfig()
    return SecureTestGenerator(config)


async def generate_secure_tests(
    function_signature: str,
    docstring: str,
    context: SecurityContext,
    config: Optional[SecurityConfig] = None
) -> Dict[str, Any]:
    """Generate secure tests"""
    generator = create_secure_generator(config)
    return await generator.generate_secure_tests(function_signature, docstring, context)
