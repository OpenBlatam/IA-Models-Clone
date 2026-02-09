from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import re
import time
import traceback
from typing import List, Optional, Dict, Any, Union, Tuple
from enum import Enum
from dataclasses import dataclass
import logging
import hashlib
import secrets
from contextlib import contextmanager
from typing import Any, List, Dict, Optional
import asyncio
"""
Instagram Captions API v14.0 - Comprehensive Error Handling and Validation
Advanced error handling, validation, and security utilities
"""


logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorType(Enum):
    """Error types for categorization"""
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    RESOURCE = "resource"
    RATE_LIMIT = "rate_limit"
    SYSTEM = "system"
    NETWORK = "network"
    AI_MODEL = "ai_model"
    CACHE = "cache"
    BATCH_PROCESSING = "batch_processing"

@dataclass
class ValidationError:
    """Structured validation error"""
    field: str
    message: str
    value: Any
    expected_type: str
    severity: ErrorSeverity = ErrorSeverity.MEDIUM

@dataclass
class SecurityError:
    """Structured security error"""
    threat_type: str
    detected_pattern: str
    severity: ErrorSeverity
    timestamp: float
    request_id: str

@dataclass
class PerformanceError:
    """Structured performance error"""
    metric: str
    threshold: float
    actual_value: float
    severity: ErrorSeverity
    timestamp: float

class ErrorTracker:
    """Comprehensive error tracking and monitoring"""
    
    def __init__(self) -> Any:
        self.errors: List[Dict[str, Any]] = []
        self.error_counts: Dict[str, int] = {}
        self.security_incidents: List[SecurityError] = []
        self.performance_issues: List[PerformanceError] = []
        self.start_time = time.time()
    
    def record_error(self, error_type: ErrorType, message: str, severity: ErrorSeverity, 
                    details: Optional[Dict[str, Any]] = None, request_id: Optional[str] = None):
        """Record an error with full context"""
        error_record = {
            "timestamp": time.time(),
            "error_type": error_type.value,
            "message": message,
            "severity": severity.value,
            "details": details or {},
            "request_id": request_id,
            "traceback": traceback.format_exc() if severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL] else None
        }
        
        self.errors.append(error_record)
        self.error_counts[error_type.value] = self.error_counts.get(error_type.value, 0) + 1
        
        # Log based on severity
        if severity == ErrorSeverity.CRITICAL:
            logger.critical(f"CRITICAL ERROR [{error_type.value}]: {message}")
        elif severity == ErrorSeverity.HIGH:
            logger.error(f"HIGH SEVERITY ERROR [{error_type.value}]: {message}")
        elif severity == ErrorSeverity.MEDIUM:
            logger.warning(f"MEDIUM SEVERITY ERROR [{error_type.value}]: {message}")
        else:
            logger.info(f"LOW SEVERITY ERROR [{error_type.value}]: {message}")
    
    def record_security_incident(self, threat_type: str, pattern: str, severity: ErrorSeverity, request_id: str):
        """Record security incident"""
        incident = SecurityError(
            threat_type=threat_type,
            detected_pattern=pattern,
            severity=severity,
            timestamp=time.time(),
            request_id=request_id
        )
        self.security_incidents.append(incident)
        logger.warning(f"SECURITY INCIDENT: {threat_type} - {pattern} (Request: {request_id})")
    
    def record_performance_issue(self, metric: str, threshold: float, actual_value: float, severity: ErrorSeverity):
        """Record performance issue"""
        issue = PerformanceError(
            metric=metric,
            threshold=threshold,
            actual_value=actual_value,
            severity=severity,
            timestamp=time.time()
        )
        self.performance_issues.append(issue)
        logger.warning(f"PERFORMANCE ISSUE: {metric} = {actual_value} (threshold: {threshold})")
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get comprehensive error summary"""
        return {
            "total_errors": len(self.errors),
            "error_counts": self.error_counts,
            "security_incidents": len(self.security_incidents),
            "performance_issues": len(self.performance_issues),
            "uptime": time.time() - self.start_time,
            "error_rate": len(self.errors) / max(time.time() - self.start_time, 1) * 3600,  # errors per hour
            "critical_errors": len([e for e in self.errors if e["severity"] == ErrorSeverity.CRITICAL.value]),
            "high_severity_errors": len([e for e in self.errors if e["severity"] == ErrorSeverity.HIGH.value])
        }

# Global error tracker
error_tracker = ErrorTracker()

class ValidationEngine:
    """Advanced validation engine with comprehensive checks"""
    
    def __init__(self) -> Any:
        self.validation_rules: Dict[str, Dict[str, Any]] = {
            "content_description": {
                "min_length": 3,
                "max_length": 1000,
                "required": True,
                "pattern": r"^[a-zA-Z0-9\s\-_.,!?@#$%&*()+=:;\"'<>/\\|~`\[\]{}]+$"
            },
            "style": {
                "allowed_values": ["casual", "professional", "inspirational", "playful"],
                "required": True
            },
            "hashtag_count": {
                "min_value": 5,
                "max_value": 30,
                "required": True,
                "type": int
            },
            "optimization_level": {
                "allowed_values": ["ultra_fast", "balanced", "quality"],
                "required": False,
                "default": "balanced"
            }
        }
    
    async def validate_request(self, request_data: Dict[str, Any], request_id: str) -> Tuple[bool, List[ValidationError]]:
        """Comprehensive request validation"""
        errors = []
        
        for field, rules in self.validation_rules.items():
            value = request_data.get(field)
            
            # Check required fields
            if rules.get("required", False) and value is None:
                errors.append(ValidationError(
                    field=field,
                    message=f"Required field '{field}' is missing",
                    value=None,
                    expected_type="required",
                    severity=ErrorSeverity.HIGH
                ))
                continue
            
            # Skip validation if field is not present and not required
            if value is None:
                continue
            
            # Type validation
            if "type" in rules and not isinstance(value, rules["type"]):
                errors.append(ValidationError(
                    field=field,
                    message=f"Field '{field}' must be of type {rules['type'].__name__}",
                    value=value,
                    expected_type=rules["type"].__name__,
                    severity=ErrorSeverity.MEDIUM
                ))
            
            # String length validation
            if isinstance(value, str):
                if "min_length" in rules and len(value) < rules["min_length"]:
                    errors.append(ValidationError(
                        field=field,
                        message=f"Field '{field}' must be at least {rules['min_length']} characters",
                        value=value,
                        expected_type=f"string (min: {rules['min_length']})",
                        severity=ErrorSeverity.MEDIUM
                    ))
                
                if "max_length" in rules and len(value) > rules["max_length"]:
                    errors.append(ValidationError(
                        field=field,
                        message=f"Field '{field}' must be at most {rules['max_length']} characters",
                        value=value,
                        expected_type=f"string (max: {rules['max_length']})",
                        severity=ErrorSeverity.MEDIUM
                    ))
                
                # Pattern validation
                if "pattern" in rules and not re.match(rules["pattern"], value):
                    errors.append(ValidationError(
                        field=field,
                        message=f"Field '{field}' contains invalid characters",
                        value=value,
                        expected_type="pattern matched string",
                        severity=ErrorSeverity.MEDIUM
                    ))
            
            # Numeric range validation
            if isinstance(value, (int, float)):
                if "min_value" in rules and value < rules["min_value"]:
                    errors.append(ValidationError(
                        field=field,
                        message=f"Field '{field}' must be at least {rules['min_value']}",
                        value=value,
                        expected_type=f"number (min: {rules['min_value']})",
                        severity=ErrorSeverity.MEDIUM
                    ))
                
                if "max_value" in rules and value > rules["max_value"]:
                    errors.append(ValidationError(
                        field=field,
                        message=f"Field '{field}' must be at most {rules['max_value']}",
                        value=value,
                        expected_type=f"number (max: {rules['max_value']})",
                        severity=ErrorSeverity.MEDIUM
                    ))
            
            # Allowed values validation
            if "allowed_values" in rules and value not in rules["allowed_values"]:
                errors.append(ValidationError(
                    field=field,
                    message=f"Field '{field}' must be one of: {', '.join(rules['allowed_values'])}",
                    value=value,
                    expected_type=f"one of {rules['allowed_values']}",
                    severity=ErrorSeverity.MEDIUM
                ))
        
        # Record validation errors
        for error in errors:
            error_tracker.record_error(
                error_type=ErrorType.VALIDATION,
                message=f"Validation error in field '{error.field}': {error.message}",
                severity=error.severity,
                details={"field": error.field, "value": error.value, "expected": error.expected_type},
                request_id=request_id
            )
        
        return len(errors) == 0, errors

class SecurityEngine:
    """Advanced security engine with threat detection"""
    
    def __init__(self) -> Any:
        self.threat_patterns = {
            "sql_injection": [
                r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)",
                r"(\b(OR|AND)\s+\d+\s*=\s*\d+)",
                r"(\b(OR|AND)\s+['\"]\w+['\"]\s*=\s*['\"]\w+['\"])"
            ],
            "xss": [
                r"<script[^>]*>.*?</script>",
                r"javascript:",
                r"on\w+\s*=",
                r"<iframe[^>]*>",
                r"<object[^>]*>",
                r"<embed[^>]*>"
            ],
            "path_traversal": [
                r"\.\./",
                r"\.\.\\",
                r"~",
                r"/etc/",
                r"/var/",
                r"C:\\"
            ],
            "command_injection": [
                r"(\b(cmd|command|exec|system|eval|subprocess)\b)",
                r"[;&|`$()]",
                r"(\b(rm|del|format|shutdown|reboot)\b)"
            ],
            "malicious_content": [
                r"<script",
                r"javascript:",
                r"data:",
                r"vbscript:",
                r"onload=",
                r"onerror=",
                r"onclick="
            ]
        }
        
        self.rate_limits = {
            "requests_per_minute": 60,
            "requests_per_hour": 1000,
            "requests_per_day": 10000
        }
        
        self.request_history: Dict[str, List[float]] = {}
    
    def scan_content(self, content: str, request_id: str) -> Tuple[bool, List[SecurityError]]:
        """Scan content for security threats"""
        threats = []
        
        for threat_type, patterns in self.threat_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    severity = ErrorSeverity.CRITICAL if threat_type in ["sql_injection", "command_injection"] else ErrorSeverity.HIGH
                    
                    threat = SecurityError(
                        threat_type=threat_type,
                        detected_pattern=match.group(),
                        severity=severity,
                        timestamp=time.time(),
                        request_id=request_id
                    )
                    threats.append(threat)
                    
                    error_tracker.record_security_incident(
                        threat_type=threat_type,
                        pattern=match.group(),
                        severity=severity,
                        request_id=request_id
                    )
        
        return len(threats) == 0, threats
    
    def check_rate_limit(self, client_id: str) -> Tuple[bool, Optional[str]]:
        """Check rate limiting for client"""
        current_time = time.time()
        
        if client_id not in self.request_history:
            self.request_history[client_id] = []
        
        # Clean old requests
        self.request_history[client_id] = [
            req_time for req_time in self.request_history[client_id]
            if current_time - req_time < 86400  # Keep last 24 hours
        ]
        
        # Add current request
        self.request_history[client_id].append(current_time)
        
        # Check limits
        requests_last_minute = len([req for req in self.request_history[client_id] if current_time - req < 60])
        requests_last_hour = len([req for req in self.request_history[client_id] if current_time - req < 3600])
        requests_last_day = len(self.request_history[client_id])
        
        if requests_last_minute > self.rate_limits["requests_per_minute"]:
            return False, f"Rate limit exceeded: {requests_last_minute} requests in last minute"
        
        if requests_last_hour > self.rate_limits["requests_per_hour"]:
            return False, f"Rate limit exceeded: {requests_last_hour} requests in last hour"
        
        if requests_last_day > self.rate_limits["requests_per_day"]:
            return False, f"Rate limit exceeded: {requests_last_day} requests in last day"
        
        return True, None

class PerformanceMonitor:
    """Advanced performance monitoring with error detection"""
    
    def __init__(self) -> Any:
        self.thresholds = {
            "response_time": {
                "warning": 0.050,  # 50ms
                "error": 0.100,    # 100ms
                "critical": 0.500  # 500ms
            },
            "memory_usage": {
                "warning": 0.8,    # 80%
                "error": 0.9,      # 90%
                "critical": 0.95   # 95%
            },
            "error_rate": {
                "warning": 0.05,   # 5%
                "error": 0.10,     # 10%
                "critical": 0.20   # 20%
            }
        }
    
    def check_performance(self, metric: str, value: float) -> Optional[PerformanceError]:
        """Check performance against thresholds"""
        if metric not in self.thresholds:
            return None
        
        thresholds = self.thresholds[metric]
        
        if value >= thresholds["critical"]:
            severity = ErrorSeverity.CRITICAL
        elif value >= thresholds["error"]:
            severity = ErrorSeverity.HIGH
        elif value >= thresholds["warning"]:
            severity = ErrorSeverity.MEDIUM
        else:
            return None
        
        error = PerformanceError(
            metric=metric,
            threshold=thresholds["critical"] if severity == ErrorSeverity.CRITICAL else thresholds["error"],
            actual_value=value,
            severity=severity,
            timestamp=time.time()
        )
        
        error_tracker.record_performance_issue(
            metric=metric,
            threshold=error.threshold,
            actual_value=value,
            severity=severity
        )
        
        return error

# Global instances
validation_engine = ValidationEngine()
security_engine = SecurityEngine()
performance_monitor = PerformanceMonitor()

@contextmanager
def error_context(operation: str, request_id: str):
    """Context manager for error handling"""
    start_time = time.time()
    try:
        yield
    except Exception as e:
        error_tracker.record_error(
            error_type=ErrorType.SYSTEM,
            message=f"Error in {operation}: {str(e)}",
            severity=ErrorSeverity.HIGH,
            details={"operation": operation, "exception_type": type(e).__name__},
            request_id=request_id
        )
        raise
    finally:
        # Record performance
        duration = time.time() - start_time
        performance_monitor.check_performance("response_time", duration)

async def generate_request_id() -> str:
    """Generate unique request ID with timestamp"""
    timestamp = int(time.time() * 1000)
    random_part = secrets.token_urlsafe(6)
    return f"v14-{timestamp}-{random_part}"

async def validate_api_key(api_key: str) -> bool:
    """Validate API key with enhanced security"""
    if not api_key or not isinstance(api_key, str):
        return False
    
    # Check for common attack patterns
    if len(api_key) > 100 or len(api_key) < 10:
        return False
    
    # Check for suspicious characters
    if re.search(r"[<>\"'&]", api_key):
        return False
    
    valid_keys = ["optimized-v14-key", "ultra-fast-key", "performance-key"]
    return api_key in valid_keys

def sanitize_content(content: str) -> str:
    """Enhanced content sanitization"""
    if not content or not isinstance(content, str):
        raise ValueError("Content must be a non-empty string")
    
    # Remove null bytes and control characters
    content = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', content)
    
    # Normalize whitespace
    content = re.sub(r'\s+', ' ', content)
    
    # Remove potentially harmful patterns
    harmful_patterns = [
        r'<script[^>]*>.*?</script>',
        r'javascript:',
        r'data:',
        r'vbscript:',
        r'on\w+\s*=',
        r'<iframe[^>]*>',
        r'<object[^>]*>',
        r'<embed[^>]*>'
    ]
    
    for pattern in harmful_patterns:
        if re.search(pattern, content, re.IGNORECASE):
            raise ValueError(f"Potentially harmful content detected: {pattern}")
    
    return content.strip()

def validate_style(style: str) -> bool:
    """Validate caption style with enhanced checks"""
    if not style or not isinstance(style, str):
        return False
    
    valid_styles = ["casual", "professional", "inspirational", "playful"]
    return style.lower() in valid_styles

def validate_optimization_level(level: str) -> bool:
    """Validate optimization level"""
    if not level or not isinstance(level, str):
        return False
    
    valid_levels = ["ultra_fast", "balanced", "quality"]
    return level.lower() in valid_levels

def validate_hashtag_count(count: int) -> bool:
    """Validate hashtag count with enhanced checks"""
    if not isinstance(count, int):
        return False
    
    return 5 <= count <= 30

def sanitize_hashtags(hashtags: List[str]) -> List[str]:
    """Enhanced hashtag sanitization"""
    if not isinstance(hashtags, list):
        raise ValueError("Hashtags must be a list")
    
    sanitized = []
    for hashtag in hashtags:
        if not isinstance(hashtag, str):
            continue
        
        # Remove special characters and ensure proper format
        clean_hashtag = re.sub(r'[^a-zA-Z0-9_]', '', hashtag)
        
        if clean_hashtag and not clean_hashtag.startswith('#'):
            clean_hashtag = f"#{clean_hashtag}"
        
        if clean_hashtag and len(clean_hashtag) > 1:
            sanitized.append(clean_hashtag.lower())
    
    # Remove duplicates and limit length
    unique_hashtags = list(dict.fromkeys(sanitized))
    return unique_hashtags[:30]  # Limit to 30 hashtags

def validate_batch_size(size: int) -> bool:
    """Validate batch size with enhanced checks"""
    if not isinstance(size, int):
        return False
    
    return 1 <= size <= 100

def generate_cache_key(content: str, style: str, hashtag_count: int) -> str:
    """Generate secure cache key"""
    if not all([content, style, isinstance(hashtag_count, int)]):
        raise ValueError("Invalid parameters for cache key generation")
    
    key_data = f"{content}:{style}:{hashtag_count}"
    return hashlib.sha256(key_data.encode()).hexdigest()

def validate_performance_thresholds(
    avg_response_time: float,
    cache_hit_rate: float,
    success_rate: float
) -> dict:
    """Validate performance thresholds with enhanced logic"""
    if not all(isinstance(x, (int, float)) for x in [avg_response_time, cache_hit_rate, success_rate]):
        raise ValueError("All performance metrics must be numeric")
    
    thresholds = {
        "response_time_grade": "SLOW",
        "cache_grade": "POOR", 
        "success_grade": "POOR"
    }
    
    # Response time grading with enhanced thresholds
    if avg_response_time < 0.015:
        thresholds["response_time_grade"] = "ULTRA_FAST"
    elif avg_response_time < 0.025:
        thresholds["response_time_grade"] = "FAST"
    elif avg_response_time < 0.050:
        thresholds["response_time_grade"] = "NORMAL"
    elif avg_response_time < 0.100:
        thresholds["response_time_grade"] = "SLOW"
    else:
        thresholds["response_time_grade"] = "VERY_SLOW"
    
    # Cache hit rate grading
    if cache_hit_rate >= 95:
        thresholds["cache_grade"] = "EXCELLENT"
    elif cache_hit_rate >= 80:
        thresholds["cache_grade"] = "GOOD"
    elif cache_hit_rate >= 60:
        thresholds["cache_grade"] = "FAIR"
    elif cache_hit_rate >= 40:
        thresholds["cache_grade"] = "POOR"
    else:
        thresholds["cache_grade"] = "VERY_POOR"
    
    # Success rate grading
    if success_rate >= 99:
        thresholds["success_grade"] = "EXCELLENT"
    elif success_rate >= 95:
        thresholds["success_grade"] = "GOOD"
    elif success_rate >= 90:
        thresholds["success_grade"] = "FAIR"
    elif success_rate >= 80:
        thresholds["success_grade"] = "POOR"
    else:
        thresholds["success_grade"] = "VERY_POOR"
    
    return thresholds 