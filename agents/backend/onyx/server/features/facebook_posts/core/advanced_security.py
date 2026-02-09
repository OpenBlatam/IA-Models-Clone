"""
Advanced Security System for Facebook Posts
Following functional programming principles and enterprise security practices
"""

import asyncio
import hashlib
import hmac
import time
import json
from typing import Dict, Any, List, Optional, Tuple, Union, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import logging
from collections import defaultdict, deque
import ipaddress
import re

logger = logging.getLogger(__name__)


# Pure functions for security

class ThreatLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecurityEventType(str, Enum):
    AUTHENTICATION_FAILURE = "authentication_failure"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    MALICIOUS_REQUEST = "malicious_request"
    BRUTE_FORCE_ATTACK = "brute_force_attack"
    DDoS_ATTACK = "ddos_attack"
    DATA_BREACH_ATTEMPT = "data_breach_attempt"
    INJECTION_ATTEMPT = "injection_attempt"


@dataclass(frozen=True)
class SecurityEvent:
    """Immutable security event - pure data structure"""
    event_type: SecurityEventType
    threat_level: ThreatLevel
    source_ip: str
    user_agent: str
    request_path: str
    description: str
    metadata: Dict[str, Any]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary - pure function"""
        return {
            "event_type": self.event_type.value,
            "threat_level": self.threat_level.value,
            "source_ip": self.source_ip,
            "user_agent": self.user_agent,
            "request_path": self.request_path,
            "description": self.description,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass(frozen=True)
class SecurityRule:
    """Immutable security rule - pure data structure"""
    rule_id: str
    name: str
    pattern: str
    threat_level: ThreatLevel
    action: str
    enabled: bool
    created_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary - pure function"""
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "pattern": self.pattern,
            "threat_level": self.threat_level.value,
            "action": self.action,
            "enabled": self.enabled,
            "created_at": self.created_at.isoformat()
        }


def is_valid_ip_address(ip: str) -> bool:
    """Validate IP address - pure function"""
    try:
        ipaddress.ip_address(ip)
        return True
    except ValueError:
        return False


def is_private_ip(ip: str) -> bool:
    """Check if IP is private - pure function"""
    try:
        ip_obj = ipaddress.ip_address(ip)
        return ip_obj.is_private
    except ValueError:
        return False


def is_suspicious_user_agent(user_agent: str) -> bool:
    """Check if user agent is suspicious - pure function"""
    if not user_agent or len(user_agent) < 10:
        return True
    
    # Check for common bot patterns
    bot_patterns = [
        r'bot', r'crawler', r'spider', r'scraper', r'wget', r'curl',
        r'python-requests', r'go-http-client', r'java/', r'okhttp'
    ]
    
    user_agent_lower = user_agent.lower()
    for pattern in bot_patterns:
        if re.search(pattern, user_agent_lower):
            return True
    
    # Check for suspicious characteristics
    if len(user_agent) > 500:  # Unusually long
        return True
    
    if user_agent.count(' ') > 20:  # Too many spaces
        return True
    
    return False


def detect_sql_injection(content: str) -> bool:
    """Detect SQL injection attempts - pure function"""
    if not content:
        return False
    
    # Common SQL injection patterns
    sql_patterns = [
        r'union\s+select', r'drop\s+table', r'delete\s+from',
        r'insert\s+into', r'update\s+set', r'alter\s+table',
        r'exec\s*\(', r'execute\s*\(', r'sp_', r'xp_',
        r'--', r'/\*', r'\*/', r';\s*drop', r';\s*delete',
        r'1\s*=\s*1', r'1\s*=\s*1\s*--', r'\'\s*or\s*\'',
        r'\"\s*or\s*\"', r'or\s+1\s*=\s*1', r'and\s+1\s*=\s*1'
    ]
    
    content_lower = content.lower()
    for pattern in sql_patterns:
        if re.search(pattern, content_lower):
            return True
    
    return False


def detect_xss_attempt(content: str) -> bool:
    """Detect XSS attempts - pure function"""
    if not content:
        return False
    
    # Common XSS patterns
    xss_patterns = [
        r'<script', r'</script>', r'javascript:', r'vbscript:',
        r'onload\s*=', r'onerror\s*=', r'onclick\s*=', r'onmouseover\s*=',
        r'<iframe', r'<object', r'<embed', r'<form',
        r'alert\s*\(', r'confirm\s*\(', r'prompt\s*\(',
        r'document\.cookie', r'document\.write', r'window\.location'
    ]
    
    content_lower = content.lower()
    for pattern in xss_patterns:
        if re.search(pattern, content_lower):
            return True
    
    return False


def detect_path_traversal(path: str) -> bool:
    """Detect path traversal attempts - pure function"""
    if not path:
        return False
    
    # Common path traversal patterns
    traversal_patterns = [
        r'\.\./', r'\.\.\\', r'\.\.%2f', r'\.\.%5c',
        r'%2e%2e%2f', r'%2e%2e%5c', r'\.\.%252f', r'\.\.%255c'
    ]
    
    for pattern in traversal_patterns:
        if re.search(pattern, path, re.IGNORECASE):
            return True
    
    return False


def calculate_threat_score(
    event_type: SecurityEventType,
    source_ip: str,
    user_agent: str,
    request_path: str,
    content: str = ""
) -> ThreatLevel:
    """Calculate threat score - pure function"""
    score = 0
    
    # Base score by event type
    base_scores = {
        SecurityEventType.AUTHENTICATION_FAILURE: 2,
        SecurityEventType.RATE_LIMIT_EXCEEDED: 3,
        SecurityEventType.SUSPICIOUS_ACTIVITY: 4,
        SecurityEventType.MALICIOUS_REQUEST: 6,
        SecurityEventType.BRUTE_FORCE_ATTACK: 7,
        SecurityEventType.DDoS_ATTACK: 8,
        SecurityEventType.DATA_BREACH_ATTEMPT: 9,
        SecurityEventType.INJECTION_ATTEMPT: 8
    }
    
    score += base_scores.get(event_type, 1)
    
    # IP-based scoring
    if is_private_ip(source_ip):
        score += 1  # Private IPs are less suspicious
    
    # User agent scoring
    if is_suspicious_user_agent(user_agent):
        score += 2
    
    # Content-based scoring
    if content:
        if detect_sql_injection(content):
            score += 3
        if detect_xss_attempt(content):
            score += 3
        if detect_path_traversal(request_path):
            score += 2
    
    # Determine threat level
    if score >= 8:
        return ThreatLevel.CRITICAL
    elif score >= 6:
        return ThreatLevel.HIGH
    elif score >= 4:
        return ThreatLevel.MEDIUM
    else:
        return ThreatLevel.LOW


def create_security_event(
    event_type: SecurityEventType,
    source_ip: str,
    user_agent: str,
    request_path: str,
    description: str,
    metadata: Optional[Dict[str, Any]] = None,
    content: str = ""
) -> SecurityEvent:
    """Create security event - pure function"""
    threat_level = calculate_threat_score(
        event_type, source_ip, user_agent, request_path, content
    )
    
    return SecurityEvent(
        event_type=event_type,
        threat_level=threat_level,
        source_ip=source_ip,
        user_agent=user_agent,
        request_path=request_path,
        description=description,
        metadata=metadata or {},
        timestamp=datetime.utcnow()
    )


def generate_api_key(prefix: str = "fbp") -> str:
    """Generate secure API key - pure function"""
    timestamp = str(int(time.time()))
    random_data = hashlib.sha256(f"{timestamp}{time.time()}".encode()).hexdigest()[:16]
    return f"{prefix}_{timestamp}_{random_data}"


def verify_api_key(api_key: str, stored_hash: str) -> bool:
    """Verify API key - pure function"""
    try:
        # In a real implementation, you'd use proper password hashing
        expected_hash = hashlib.sha256(api_key.encode()).hexdigest()
        return hmac.compare_digest(expected_hash, stored_hash)
    except Exception:
        return False


def hash_api_key(api_key: str) -> str:
    """Hash API key for storage - pure function"""
    return hashlib.sha256(api_key.encode()).hexdigest()


# Advanced Security System Class

class AdvancedSecuritySystem:
    """Advanced Security System following functional principles"""
    
    def __init__(
        self,
        max_failed_attempts: int = 5,
        lockout_duration: int = 900,  # 15 minutes
        rate_limit_window: int = 3600,  # 1 hour
        max_requests_per_window: int = 1000
    ):
        self.max_failed_attempts = max_failed_attempts
        self.lockout_duration = lockout_duration
        self.rate_limit_window = rate_limit_window
        self.max_requests_per_window = max_requests_per_window
        
        # Security data
        self.failed_attempts: Dict[str, List[datetime]] = defaultdict(list)
        self.rate_limits: Dict[str, List[datetime]] = defaultdict(list)
        self.blocked_ips: Set[str] = set()
        self.security_events: deque = deque(maxlen=10000)
        self.security_rules: Dict[str, SecurityRule] = {}
        
        # API keys
        self.api_keys: Dict[str, Dict[str, Any]] = {}
        
        # Statistics
        self.stats = {
            "total_events": 0,
            "blocked_ips": 0,
            "failed_attempts": 0,
            "rate_limit_hits": 0,
            "threat_levels": defaultdict(int)
        }
        
        # Background tasks
        self.cleanup_task: Optional[asyncio.Task] = None
        self.is_running = False
        
        # Initialize default rules
        self._setup_default_rules()
    
    async def start(self) -> None:
        """Start security system"""
        if self.is_running:
            return
        
        self.is_running = True
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("Advanced security system started")
    
    async def stop(self) -> None:
        """Stop security system"""
        if not self.is_running:
            return
        
        self.is_running = False
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Advanced security system stopped")
    
    async def _cleanup_loop(self) -> None:
        """Background cleanup loop"""
        while self.is_running:
            try:
                await self._cleanup_expired_data()
                await asyncio.sleep(300)  # Run every 5 minutes
            except Exception as e:
                logger.error("Error in security cleanup loop", error=str(e))
                await asyncio.sleep(60)
    
    async def _cleanup_expired_data(self) -> None:
        """Clean up expired security data"""
        try:
            current_time = datetime.utcnow()
            cutoff_time = current_time - timedelta(hours=24)
            
            # Clean up old failed attempts
            for ip in list(self.failed_attempts.keys()):
                self.failed_attempts[ip] = [
                    attempt for attempt in self.failed_attempts[ip]
                    if attempt > cutoff_time
                ]
                if not self.failed_attempts[ip]:
                    del self.failed_attempts[ip]
            
            # Clean up old rate limit data
            for ip in list(self.rate_limits.keys()):
                self.rate_limits[ip] = [
                    request for request in self.rate_limits[ip]
                    if request > cutoff_time
                ]
                if not self.rate_limits[ip]:
                    del self.rate_limits[ip]
            
            # Remove expired IP blocks
            expired_blocks = set()
            for ip in self.blocked_ips:
                # Simple expiration logic - in practice, you'd store timestamps
                if ip in self.failed_attempts:
                    latest_attempt = max(self.failed_attempts[ip])
                    if (current_time - latest_attempt).total_seconds() > self.lockout_duration:
                        expired_blocks.add(ip)
            
            self.blocked_ips -= expired_blocks
            
            logger.debug(f"Cleaned up expired security data")
            
        except Exception as e:
            logger.error("Error cleaning up expired data", error=str(e))
    
    async def check_request_security(
        self,
        source_ip: str,
        user_agent: str,
        request_path: str,
        content: str = "",
        api_key: Optional[str] = None
    ) -> Tuple[bool, Optional[SecurityEvent]]:
        """Check request security"""
        try:
            # Check if IP is blocked
            if source_ip in self.blocked_ips:
                event = create_security_event(
                    SecurityEventType.SUSPICIOUS_ACTIVITY,
                    source_ip, user_agent, request_path,
                    "Request from blocked IP",
                    {"blocked": True}
                )
                await self._record_security_event(event)
                return False, event
            
            # Check rate limiting
            if not await self._check_rate_limit(source_ip):
                event = create_security_event(
                    SecurityEventType.RATE_LIMIT_EXCEEDED,
                    source_ip, user_agent, request_path,
                    "Rate limit exceeded",
                    {"rate_limit": True}
                )
                await self._record_security_event(event)
                return False, event
            
            # Check for malicious patterns
            malicious_event = await self._check_malicious_patterns(
                source_ip, user_agent, request_path, content
            )
            if malicious_event:
                await self._record_security_event(malicious_event)
                return False, malicious_event
            
            # Check API key if provided
            if api_key and not await self._verify_api_key(api_key):
                event = create_security_event(
                    SecurityEventType.AUTHENTICATION_FAILURE,
                    source_ip, user_agent, request_path,
                    "Invalid API key",
                    {"api_key": api_key}
                )
                await self._record_security_event(event)
                return False, event
            
            return True, None
            
        except Exception as e:
            logger.error("Error checking request security", error=str(e))
            return False, None
    
    async def _check_rate_limit(self, source_ip: str) -> bool:
        """Check rate limiting"""
        current_time = datetime.utcnow()
        window_start = current_time - timedelta(seconds=self.rate_limit_window)
        
        # Get recent requests
        recent_requests = [
            req_time for req_time in self.rate_limits[source_ip]
            if req_time > window_start
        ]
        
        # Check if limit exceeded
        if len(recent_requests) >= self.max_requests_per_window:
            self.stats["rate_limit_hits"] += 1
            return False
        
        # Add current request
        self.rate_limits[source_ip].append(current_time)
        return True
    
    async def _check_malicious_patterns(
        self,
        source_ip: str,
        user_agent: str,
        request_path: str,
        content: str
    ) -> Optional[SecurityEvent]:
        """Check for malicious patterns"""
        # Check SQL injection
        if detect_sql_injection(content):
            return create_security_event(
                SecurityEventType.INJECTION_ATTEMPT,
                source_ip, user_agent, request_path,
                "SQL injection attempt detected",
                {"pattern": "sql_injection"},
                content
            )
        
        # Check XSS
        if detect_xss_attempt(content):
            return create_security_event(
                SecurityEventType.INJECTION_ATTEMPT,
                source_ip, user_agent, request_path,
                "XSS attempt detected",
                {"pattern": "xss"},
                content
            )
        
        # Check path traversal
        if detect_path_traversal(request_path):
            return create_security_event(
                SecurityEventType.MALICIOUS_REQUEST,
                source_ip, user_agent, request_path,
                "Path traversal attempt detected",
                {"pattern": "path_traversal"}
            )
        
        # Check suspicious user agent
        if is_suspicious_user_agent(user_agent):
            return create_security_event(
                SecurityEventType.SUSPICIOUS_ACTIVITY,
                source_ip, user_agent, request_path,
                "Suspicious user agent detected",
                {"pattern": "suspicious_user_agent"}
            )
        
        return None
    
    async def _verify_api_key(self, api_key: str) -> bool:
        """Verify API key"""
        if not api_key:
            return False
        
        # Check if API key exists and is valid
        for key_data in self.api_keys.values():
            if verify_api_key(api_key, key_data["hash"]):
                return True
        
        return False
    
    async def _record_security_event(self, event: SecurityEvent) -> None:
        """Record security event"""
        self.security_events.append(event)
        self.stats["total_events"] += 1
        self.stats["threat_levels"][event.threat_level.value] += 1
        
        # Handle high-threat events
        if event.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
            await self._handle_high_threat_event(event)
    
    async def _handle_high_threat_event(self, event: SecurityEvent) -> None:
        """Handle high-threat security events"""
        try:
            # Block IP for high-threat events
            if event.threat_level == ThreatLevel.CRITICAL:
                self.blocked_ips.add(event.source_ip)
                self.stats["blocked_ips"] += 1
                logger.warning(f"Blocked IP {event.source_ip} due to critical threat")
            
            # Log the event
            logger.warning(f"High-threat security event: {event.event_type.value} from {event.source_ip}")
            
        except Exception as e:
            logger.error("Error handling high-threat event", error=str(e))
    
    def generate_api_key(self, user_id: str, permissions: List[str]) -> str:
        """Generate new API key"""
        api_key = generate_api_key()
        key_hash = hash_api_key(api_key)
        
        self.api_keys[api_key] = {
            "hash": key_hash,
            "user_id": user_id,
            "permissions": permissions,
            "created_at": datetime.utcnow(),
            "last_used": None,
            "is_active": True
        }
        
        logger.info(f"Generated API key for user {user_id}")
        return api_key
    
    def revoke_api_key(self, api_key: str) -> bool:
        """Revoke API key"""
        if api_key in self.api_keys:
            del self.api_keys[api_key]
            logger.info(f"Revoked API key: {api_key}")
            return True
        return False
    
    def add_security_rule(self, rule: SecurityRule) -> None:
        """Add security rule"""
        self.security_rules[rule.rule_id] = rule
        logger.info(f"Added security rule: {rule.name}")
    
    def remove_security_rule(self, rule_id: str) -> bool:
        """Remove security rule"""
        if rule_id in self.security_rules:
            del self.security_rules[rule_id]
            logger.info(f"Removed security rule: {rule_id}")
            return True
        return False
    
    def get_security_statistics(self) -> Dict[str, Any]:
        """Get security statistics"""
        return {
            "statistics": self.stats.copy(),
            "blocked_ips": list(self.blocked_ips),
            "active_api_keys": len(self.api_keys),
            "security_rules": len(self.security_rules),
            "recent_events": [event.to_dict() for event in list(self.security_events)[-20:]],
            "is_running": self.is_running
        }
    
    def get_threat_analysis(self) -> Dict[str, Any]:
        """Get threat analysis"""
        if not self.security_events:
            return {"threats": [], "analysis": "No data available"}
        
        # Analyze recent events
        recent_events = list(self.security_events)[-100:]  # Last 100 events
        
        # Group by threat level
        threat_counts = defaultdict(int)
        for event in recent_events:
            threat_counts[event.threat_level.value] += 1
        
        # Group by event type
        event_counts = defaultdict(int)
        for event in recent_events:
            event_counts[event.event_type.value] += 1
        
        # Top threat sources
        ip_counts = defaultdict(int)
        for event in recent_events:
            ip_counts[event.source_ip] += 1
        
        top_threat_sources = sorted(
            ip_counts.items(), key=lambda x: x[1], reverse=True
        )[:10]
        
        return {
            "threat_levels": dict(threat_counts),
            "event_types": dict(event_counts),
            "top_threat_sources": top_threat_sources,
            "total_events": len(recent_events),
            "analysis_period_hours": 24
        }
    
    def _setup_default_rules(self) -> None:
        """Setup default security rules"""
        default_rules = [
            SecurityRule(
                rule_id="sql_injection",
                name="SQL Injection Detection",
                pattern="union|select|drop|delete|insert|update|alter|exec|execute",
                threat_level=ThreatLevel.HIGH,
                action="block",
                enabled=True,
                created_at=datetime.utcnow()
            ),
            SecurityRule(
                rule_id="xss_attempt",
                name="XSS Attempt Detection",
                pattern="<script|javascript:|onload=|onerror=|onclick=",
                threat_level=ThreatLevel.HIGH,
                action="block",
                enabled=True,
                created_at=datetime.utcnow()
            ),
            SecurityRule(
                rule_id="path_traversal",
                name="Path Traversal Detection",
                pattern="\\.\\./|\\.\\.\\\\|%2e%2e%2f|%2e%2e%5c",
                threat_level=ThreatLevel.MEDIUM,
                action="block",
                enabled=True,
                created_at=datetime.utcnow()
            )
        ]
        
        for rule in default_rules:
            self.security_rules[rule.rule_id] = rule


# Factory functions

def create_advanced_security_system(
    max_failed_attempts: int = 5,
    lockout_duration: int = 900,
    rate_limit_window: int = 3600,
    max_requests_per_window: int = 1000
) -> AdvancedSecuritySystem:
    """Create advanced security system - pure function"""
    return AdvancedSecuritySystem(
        max_failed_attempts, lockout_duration,
        rate_limit_window, max_requests_per_window
    )


async def get_advanced_security_system() -> AdvancedSecuritySystem:
    """Get advanced security system instance"""
    system = create_advanced_security_system()
    await system.start()
    return system

