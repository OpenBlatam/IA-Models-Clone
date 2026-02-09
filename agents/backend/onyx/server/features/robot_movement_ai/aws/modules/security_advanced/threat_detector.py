"""
Threat Detector
===============

Advanced threat detection and prevention.
"""

import logging
import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ThreatLevel(Enum):
    """Threat severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Threat:
    """Threat detection result."""
    type: str
    level: ThreatLevel
    source: str
    description: str
    timestamp: datetime
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ThreatDetector:
    """Advanced threat detector."""
    
    def __init__(self):
        self._threats: List[Threat] = []
        self._patterns: Dict[str, List[re.Pattern]] = {}
        self._rate_limits: Dict[str, Dict[str, Any]] = {}
        self._blocked_ips: set = set()
    
    def register_threat_pattern(self, threat_type: str, patterns: List[str]):
        """Register threat detection patterns."""
        compiled = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
        self._patterns[threat_type] = compiled
        logger.info(f"Registered {len(patterns)} patterns for {threat_type}")
    
    def detect_threat(
        self,
        request_data: Dict[str, Any],
        source_ip: str
    ) -> Optional[Threat]:
        """Detect threats in request."""
        # Check if IP is blocked
        if source_ip in self._blocked_ips:
            return Threat(
                type="blocked_ip",
                level=ThreatLevel.HIGH,
                source=source_ip,
                description=f"Request from blocked IP: {source_ip}",
                timestamp=datetime.now()
            )
        
        # Check rate limiting
        if self._is_rate_limited(source_ip):
            return Threat(
                type="rate_limit_exceeded",
                level=ThreatLevel.MEDIUM,
                source=source_ip,
                description="Rate limit exceeded",
                timestamp=datetime.now()
            )
        
        # Check for SQL injection
        sql_threat = self._detect_sql_injection(request_data)
        if sql_threat:
            return sql_threat
        
        # Check for XSS
        xss_threat = self._detect_xss(request_data)
        if xss_threat:
            return xss_threat
        
        # Check for path traversal
        path_threat = self._detect_path_traversal(request_data)
        if path_threat:
            return path_threat
        
        return None
    
    def _detect_sql_injection(self, data: Dict[str, Any]) -> Optional[Threat]:
        """Detect SQL injection attempts."""
        sql_patterns = [
            r"('|(\\')|(;)|(--)|(/\*)|(\*/)|(xp_)|(exec)|(execute))",
            r"(\bunion\b.*\bselect\b)",
            r"(\bor\b.*=.*)",
            r"(\band\b.*=.*)"
        ]
        
        for key, value in data.items():
            if isinstance(value, str):
                for pattern in sql_patterns:
                    if re.search(pattern, value, re.IGNORECASE):
                        return Threat(
                            type="sql_injection",
                            level=ThreatLevel.HIGH,
                            source="request_data",
                            description=f"SQL injection attempt detected in {key}",
                            timestamp=datetime.now(),
                            metadata={"field": key, "value": value[:100]}
                        )
        
        return None
    
    def _detect_xss(self, data: Dict[str, Any]) -> Optional[Threat]:
        """Detect XSS attempts."""
        xss_patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"onerror\s*=",
            r"onload\s*=",
            r"<iframe[^>]*>"
        ]
        
        for key, value in data.items():
            if isinstance(value, str):
                for pattern in xss_patterns:
                    if re.search(pattern, value, re.IGNORECASE):
                        return Threat(
                            type="xss",
                            level=ThreatLevel.HIGH,
                            source="request_data",
                            description=f"XSS attempt detected in {key}",
                            timestamp=datetime.now(),
                            metadata={"field": key, "value": value[:100]}
                        )
        
        return None
    
    def _detect_path_traversal(self, data: Dict[str, Any]) -> Optional[Threat]:
        """Detect path traversal attempts."""
        traversal_patterns = [
            r"\.\./",
            r"\.\.\\",
            r"\.\.%2f",
            r"\.\.%5c"
        ]
        
        for key, value in data.items():
            if isinstance(value, str):
                for pattern in traversal_patterns:
                    if re.search(pattern, value, re.IGNORECASE):
                        return Threat(
                            type="path_traversal",
                            level=ThreatLevel.MEDIUM,
                            source="request_data",
                            description=f"Path traversal attempt detected in {key}",
                            timestamp=datetime.now(),
                            metadata={"field": key, "value": value[:100]}
                        )
        
        return None
    
    def _is_rate_limited(self, source_ip: str) -> bool:
        """Check if source is rate limited."""
        if source_ip not in self._rate_limits:
            self._rate_limits[source_ip] = {
                "count": 0,
                "window_start": datetime.now()
            }
            return False
        
        limit_info = self._rate_limits[source_ip]
        window_start = limit_info["window_start"]
        
        # Reset window if > 1 minute
        if (datetime.now() - window_start).total_seconds() > 60:
            limit_info["count"] = 0
            limit_info["window_start"] = datetime.now()
        
        limit_info["count"] += 1
        
        # Block if > 100 requests per minute
        if limit_info["count"] > 100:
            return True
        
        return False
    
    def block_ip(self, ip: str, reason: str = ""):
        """Block IP address."""
        self._blocked_ips.add(ip)
        logger.warning(f"Blocked IP: {ip} - {reason}")
    
    def unblock_ip(self, ip: str):
        """Unblock IP address."""
        self._blocked_ips.discard(ip)
        logger.info(f"Unblocked IP: {ip}")
    
    def get_threats(self, level: Optional[ThreatLevel] = None, limit: int = 100) -> List[Threat]:
        """Get detected threats."""
        threats = self._threats
        
        if level:
            threats = [t for t in threats if t.level == level]
        
        return threats[-limit:]
    
    def get_threat_stats(self) -> Dict[str, Any]:
        """Get threat statistics."""
        return {
            "total_threats": len(self._threats),
            "blocked_ips": len(self._blocked_ips),
            "by_level": {
                level.value: sum(1 for t in self._threats if t.level == level)
                for level in ThreatLevel
            },
            "by_type": {
                threat_type: sum(1 for t in self._threats if t.type == threat_type)
                for threat_type in set(t.type for t in self._threats)
            }
        }















