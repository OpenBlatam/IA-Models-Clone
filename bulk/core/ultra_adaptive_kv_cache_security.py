"""
Security Features for Ultra-Adaptive K/V Cache Engine
Includes: request sanitization, rate limiting, access control, and security monitoring
"""

import time
import hashlib
import hmac
import re
from typing import Dict, Any, Optional, List, Tuple
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security level classifications."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityEvent:
    """Security event record."""
    event_type: str
    severity: SecurityLevel
    timestamp: float
    details: Dict[str, Any]
    source: Optional[str] = None


class RequestSanitizer:
    """Advanced request sanitization."""
    
    # Potentially dangerous patterns
    DANGEROUS_PATTERNS = [
        r'<script[^>]*>.*?</script>',  # XSS scripts
        r'javascript:',  # JavaScript protocol
        r'on\w+\s*=',  # Event handlers
        r'eval\s*\(',  # eval() calls
        r'exec\s*\(',  # exec() calls
        r'__import__',  # Python import
        r'subprocess',  # Subprocess calls
        r'shell=True',  # Shell injection
        r'union\s+select',  # SQL injection
        r';\s*drop\s+table',  # SQL injection
        r'../',  # Path traversal
        r'\.\.\\',  # Path traversal (Windows)
    ]
    
    MAX_TEXT_LENGTH = 100000  # 100KB
    MAX_SESSION_ID_LENGTH = 255
    
    @classmethod
    def sanitize(cls, request: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """
        Sanitize request and return sanitized request with warnings.
        
        Returns:
            Tuple of (sanitized_request, warnings)
        """
        warnings = []
        sanitized = request.copy()
        
        # Sanitize text
        if 'text' in sanitized:
            text = sanitized['text']
            
            # Check length
            if len(text) > cls.MAX_TEXT_LENGTH:
                warnings.append(f"Text truncated from {len(text)} to {cls.MAX_TEXT_LENGTH} characters")
                sanitized['text'] = text[:cls.MAX_TEXT_LENGTH]
            
            # Check for dangerous patterns
            import re
            for pattern in cls.DANGEROUS_PATTERNS:
                if re.search(pattern, text, re.IGNORECASE):
                    warnings.append(f"Dangerous pattern detected: {pattern}")
                    # Remove or escape the pattern
                    sanitized['text'] = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Sanitize session_id
        if 'session_id' in sanitized:
            session_id = str(sanitized['session_id'])
            if len(session_id) > cls.MAX_SESSION_ID_LENGTH:
                warnings.append("Session ID truncated")
                sanitized['session_id'] = session_id[:cls.MAX_SESSION_ID_LENGTH]
            
            # Remove dangerous characters
            sanitized['session_id'] = re.sub(r'[^\w\-_]', '', sanitized['session_id'])
        
        # Validate numeric fields
        for field in ['max_length', 'temperature']:
            if field in sanitized:
                try:
                    value = float(sanitized[field])
                    # Clamp to safe ranges
                    if field == 'max_length':
                        sanitized[field] = int(max(1, min(10000, value)))
                    elif field == 'temperature':
                        sanitized[field] = max(0.0, min(2.0, value))
                except (ValueError, TypeError):
                    warnings.append(f"Invalid {field}, using default")
                    if field == 'max_length':
                        sanitized[field] = 100
                    elif field == 'temperature':
                        sanitized[field] = 1.0
        
        return sanitized, warnings


class RateLimiter:
    """Advanced rate limiting with multiple strategies."""
    
    def __init__(self, max_requests: int = 100, window_seconds: float = 60.0,
                 strategy: str = 'sliding_window'):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.strategy = strategy
        
        # Sliding window: track request timestamps
        self.request_timestamps = defaultdict(list)
        
        # Token bucket
        self.tokens = defaultdict(lambda: max_requests)
        self.last_refill = defaultdict(lambda: time.time())
        self.refill_rate = max_requests / window_seconds
    
    def is_allowed(self, identifier: str) -> Tuple[bool, Optional[str]]:
        """
        Check if request is allowed.
        
        Returns:
            Tuple of (is_allowed, reason_if_denied)
        """
        current_time = time.time()
        
        if self.strategy == 'sliding_window':
            return self._sliding_window_check(identifier, current_time)
        elif self.strategy == 'token_bucket':
            return self._token_bucket_check(identifier, current_time)
        else:
            return True, None
    
    def _sliding_window_check(self, identifier: str, current_time: float) -> Tuple[bool, Optional[str]]:
        """Check using sliding window strategy."""
        window_start = current_time - self.window_seconds
        
        # Remove old requests
        self.request_timestamps[identifier] = [
            ts for ts in self.request_timestamps[identifier]
            if ts > window_start
        ]
        
        # Check limit
        if len(self.request_timestamps[identifier]) >= self.max_requests:
            return False, f"Rate limit exceeded: {self.max_requests} requests per {self.window_seconds}s"
        
        # Add current request
        self.request_timestamps[identifier].append(current_time)
        return True, None
    
    def _token_bucket_check(self, identifier: str, current_time: float) -> Tuple[bool, Optional[str]]:
        """Check using token bucket strategy."""
        # Refill tokens
        time_passed = current_time - self.last_refill[identifier]
        tokens_to_add = time_passed * self.refill_rate
        self.tokens[identifier] = min(
            self.max_requests,
            self.tokens[identifier] + tokens_to_add
        )
        self.last_refill[identifier] = current_time
        
        # Check if token available
        if self.tokens[identifier] >= 1:
            self.tokens[identifier] -= 1
            return True, None
        else:
            return False, f"Rate limit exceeded: bucket empty"
    
    def reset(self, identifier: Optional[str] = None):
        """Reset rate limiter for identifier or all."""
        if identifier:
            if identifier in self.request_timestamps:
                del self.request_timestamps[identifier]
            if identifier in self.tokens:
                self.tokens[identifier] = self.max_requests
                self.last_refill[identifier] = time.time()
        else:
            self.request_timestamps.clear()
            self.tokens.clear()
            self.last_refill.clear()


class AccessController:
    """Access control and authentication."""
    
    def __init__(self):
        self.allowed_ips = set()
        self.blocked_ips = set()
        self.api_keys = {}
        self.roles = defaultdict(set)
    
    def allow_ip(self, ip: str):
        """Allow IP address."""
        self.allowed_ips.add(ip)
        if ip in self.blocked_ips:
            self.blocked_ips.remove(ip)
    
    def block_ip(self, ip: str):
        """Block IP address."""
        self.blocked_ips.add(ip)
        if ip in self.allowed_ips:
            self.allowed_ips.remove(ip)
    
    def register_api_key(self, api_key: str, roles: List[str], metadata: Optional[Dict[str, Any]] = None):
        """Register API key with roles."""
        key_hash = self._hash_key(api_key)
        self.api_keys[key_hash] = {
            'roles': set(roles),
            'metadata': metadata or {},
            'created_at': time.time()
        }
    
    def validate_api_key(self, api_key: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Validate API key and return roles if valid."""
        key_hash = self._hash_key(api_key)
        
        if key_hash in self.api_keys:
            return True, {
                'roles': list(self.api_keys[key_hash]['roles']),
                'metadata': self.api_keys[key_hash]['metadata']
            }
        
        return False, None
    
    def check_ip_access(self, ip: str) -> Tuple[bool, Optional[str]]:
        """Check if IP is allowed."""
        if self.blocked_ips and ip in self.blocked_ips:
            return False, "IP is blocked"
        
        if self.allowed_ips and ip not in self.allowed_ips:
            return False, "IP not in allowed list"
        
        return True, None
    
    def check_permission(self, roles: List[str], required_permission: str) -> bool:
        """Check if roles have required permission."""
        for role in roles:
            if role in self.roles and required_permission in self.roles[role]:
                return True
        return False
    
    def _hash_key(self, api_key: str) -> str:
        """Hash API key for storage."""
        return hashlib.sha256(api_key.encode()).hexdigest()


class SecurityMonitor:
    """Monitor security events and detect threats."""
    
    def __init__(self):
        self.events: List[SecurityEvent] = []
        self.alert_thresholds = {
            'rate_limit_exceeded': {'count': 10, 'window': 60},
            'invalid_api_key': {'count': 5, 'window': 300},
            'blocked_ip': {'count': 3, 'window': 60},
            'sanitization_warning': {'count': 20, 'window': 300}
        }
        self.suspicious_patterns = []
    
    def record_event(self, event_type: str, severity: SecurityLevel, 
                   details: Dict[str, Any], source: Optional[str] = None):
        """Record a security event."""
        event = SecurityEvent(
            event_type=event_type,
            severity=severity,
            timestamp=time.time(),
            details=details,
            source=source
        )
        
        self.events.append(event)
        
        # Keep only last 1000 events
        if len(self.events) > 1000:
            self.events = self.events[-1000:]
        
        # Check for alerts
        self._check_alerts(event_type)
    
    def _check_alerts(self, event_type: str):
        """Check if event type exceeds alert threshold."""
        if event_type not in self.alert_thresholds:
            return
        
        threshold = self.alert_thresholds[event_type]
        window_start = time.time() - threshold['window']
        
        recent_events = [
            e for e in self.events
            if e.event_type == event_type and e.timestamp >= window_start
        ]
        
        if len(recent_events) >= threshold['count']:
            logger.warning(
                f"Security alert: {event_type} occurred {len(recent_events)} times "
                f"in the last {threshold['window']} seconds"
            )
    
    def get_security_report(self, window_seconds: float = 3600) -> Dict[str, Any]:
        """Get security report for time window."""
        window_start = time.time() - window_seconds
        recent_events = [e for e in self.events if e.timestamp >= window_start]
        
        event_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        
        for event in recent_events:
            event_counts[event.event_type] += 1
            severity_counts[event.severity.value] += 1
        
        return {
            'window_seconds': window_seconds,
            'total_events': len(recent_events),
            'event_types': dict(event_counts),
            'severity_breakdown': dict(severity_counts),
            'recent_critical': [
                {
                    'type': e.event_type,
                    'timestamp': e.timestamp,
                    'details': e.details
                }
                for e in recent_events
                if e.severity == SecurityLevel.CRITICAL
            ][-10:]  # Last 10 critical events
        }


class HMACValidator:
    """HMAC signature validation for requests."""
    
    def __init__(self, secret_key: str):
        self.secret_key = secret_key.encode()
    
    def generate_signature(self, data: str) -> str:
        """Generate HMAC signature for data."""
        return hmac.new(
            self.secret_key,
            data.encode(),
            hashlib.sha256
        ).hexdigest()
    
    def validate_signature(self, data: str, signature: str) -> bool:
        """Validate HMAC signature."""
        expected = self.generate_signature(data)
        return hmac.compare_digest(expected, signature)
    
    def sign_request(self, request: Dict[str, Any]) -> str:
        """Sign a request dictionary."""
        import json
        data_str = json.dumps(request, sort_keys=True)
        return self.generate_signature(data_str)


# Security wrapper for engine
class SecureEngineWrapper:
    """Wrapper that adds security features to engine."""
    
    def __init__(self, engine, enable_sanitization: bool = True,
                 enable_rate_limiting: bool = True,
                 enable_access_control: bool = False,
                 enable_hmac: bool = False,
                 hmac_secret: Optional[str] = None):
        self.engine = engine
        self.enable_sanitization = enable_sanitization
        self.enable_rate_limiting = enable_rate_limiting
        self.enable_access_control = enable_access_control
        self.enable_hmac = enable_hmac
        
        self.sanitizer = RequestSanitizer() if enable_sanitization else None
        self.rate_limiter = RateLimiter() if enable_rate_limiting else None
        self.access_controller = AccessController() if enable_access_control else None
        self.hmac_validator = HMACValidator(hmac_secret) if enable_hmac and hmac_secret else None
        self.security_monitor = SecurityMonitor()
    
    async def process_request_secure(self, request: Dict[str, Any], 
                                   client_ip: Optional[str] = None,
                                   api_key: Optional[str] = None,
                                   signature: Optional[str] = None) -> Dict[str, Any]:
        """Process request with security checks."""
        # Check access control
        if self.enable_access_control and client_ip:
            allowed, reason = self.access_controller.check_ip_access(client_ip)
            if not allowed:
                self.security_monitor.record_event(
                    'blocked_ip',
                    SecurityLevel.HIGH,
                    {'ip': client_ip, 'reason': reason},
                    source=client_ip
                )
                return {
                    'success': False,
                    'error': f'Access denied: {reason}'
                }
        
        # Validate API key
        if self.enable_access_control and api_key:
            valid, key_info = self.access_controller.validate_api_key(api_key)
            if not valid:
                self.security_monitor.record_event(
                    'invalid_api_key',
                    SecurityLevel.MEDIUM,
                    {'api_key_prefix': api_key[:8] if len(api_key) > 8 else '***'},
                    source=client_ip
                )
                return {
                    'success': False,
                    'error': 'Invalid API key'
                }
        
        # Validate HMAC signature
        if self.enable_hmac and self.hmac_validator and signature:
            import json
            data_str = json.dumps(request, sort_keys=True)
            if not self.hmac_validator.validate_signature(data_str, signature):
                self.security_monitor.record_event(
                    'invalid_signature',
                    SecurityLevel.HIGH,
                    {'request_keys': list(request.keys())},
                    source=client_ip
                )
                return {
                    'success': False,
                    'error': 'Invalid request signature'
                }
        
        # Rate limiting
        identifier = client_ip or api_key or 'anonymous'
        if self.enable_rate_limiting and self.rate_limiter:
            allowed, reason = self.rate_limiter.is_allowed(identifier)
            if not allowed:
                self.security_monitor.record_event(
                    'rate_limit_exceeded',
                    SecurityLevel.MEDIUM,
                    {'identifier': identifier, 'reason': reason},
                    source=client_ip
                )
                return {
                    'success': False,
                    'error': reason
                }
        
        # Sanitization
        if self.enable_sanitization and self.sanitizer:
            sanitized, warnings = self.sanitizer.sanitize(request)
            if warnings:
                self.security_monitor.record_event(
                    'sanitization_warning',
                    SecurityLevel.LOW,
                    {'warnings': warnings},
                    source=client_ip
                )
            request = sanitized
        
        # Process request
        try:
            result = await self.engine.process_request(request)
            
            if not result.get('success'):
                self.security_monitor.record_event(
                    'processing_error',
                    SecurityLevel.LOW,
                    {'error': result.get('error')},
                    source=client_ip
                )
            
            return result
        
        except Exception as e:
            self.security_monitor.record_event(
                'exception',
                SecurityLevel.MEDIUM,
                {'error': str(e)},
                source=client_ip
            )
            raise
    
    def get_security_report(self) -> Dict[str, Any]:
        """Get security monitoring report."""
        return self.security_monitor.get_security_report()

