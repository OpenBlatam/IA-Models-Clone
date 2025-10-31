"""
Gamma App - Security Middleware
Advanced security middleware with threat detection
"""

import time
import hashlib
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import redis
from fastapi import Request, Response
from fastapi.responses import JSONResponse
import logging
import re
import ipaddress

logger = logging.getLogger(__name__)

class ThreatLevel(Enum):
    """Threat levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class SecurityEvent:
    """Security event"""
    id: str
    timestamp: float
    threat_level: ThreatLevel
    event_type: str
    source_ip: str
    user_agent: str
    request_path: str
    details: Dict[str, Any]
    blocked: bool = False

class SecurityMiddleware:
    """Advanced security middleware"""
    
    def __init__(self, redis_client: redis.Redis, config: Dict[str, Any]):
        self.redis = redis_client
        self.config = config
        self.blocked_ips: set = set()
        self.suspicious_patterns = self._load_suspicious_patterns()
        self.rate_limits = self._init_rate_limits()
    
    def _load_suspicious_patterns(self) -> Dict[str, List[str]]:
        """Load suspicious patterns for detection"""
        return {
            "sql_injection": [
                r"union\s+select",
                r"drop\s+table",
                r"delete\s+from",
                r"insert\s+into",
                r"update\s+set",
                r"exec\s*\(",
                r"script\s*>",
                r"<script",
                r"javascript:",
                r"onload\s*=",
                r"onerror\s*=",
                r"onclick\s*=",
            ],
            "xss": [
                r"<script[^>]*>.*?</script>",
                r"javascript:",
                r"on\w+\s*=",
                r"<iframe[^>]*>",
                r"<object[^>]*>",
                r"<embed[^>]*>",
                r"<link[^>]*>",
                r"<meta[^>]*>",
            ],
            "path_traversal": [
                r"\.\./",
                r"\.\.\\",
                r"%2e%2e%2f",
                r"%2e%2e%5c",
                r"\.\.%2f",
                r"\.\.%5c",
            ],
            "command_injection": [
                r";\s*rm\s+",
                r";\s*cat\s+",
                r";\s*ls\s+",
                r";\s*ps\s+",
                r";\s*whoami",
                r";\s*id",
                r"|\s*rm\s+",
                r"|\s*cat\s+",
                r"`.*`",
                r"\$\(.*\)",
            ]
        }
    
    def _init_rate_limits(self) -> Dict[str, Dict[str, Any]]:
        """Initialize rate limits for different threat types"""
        return {
            "failed_auth": {"limit": 5, "window": 300},  # 5 attempts per 5 minutes
            "suspicious_requests": {"limit": 10, "window": 600},  # 10 per 10 minutes
            "blocked_requests": {"limit": 3, "window": 3600},  # 3 per hour
        }
    
    async def __call__(self, request: Request, call_next) -> Response:
        """Security middleware entry point"""
        start_time = time.time()
        
        try:
            # Extract request information
            client_ip = self._get_client_ip(request)
            user_agent = request.headers.get("User-Agent", "")
            request_path = str(request.url.path)
            request_method = request.method
            
            # Check if IP is blocked
            if await self._is_ip_blocked(client_ip):
                return self._create_blocked_response("IP address is blocked")
            
            # Analyze request for threats
            threat_analysis = await self._analyze_request(request)
            
            # Check rate limits
            rate_limit_violation = await self._check_security_rate_limits(
                client_ip, threat_analysis
            )
            
            if rate_limit_violation:
                await self._block_ip(client_ip, "Rate limit violation")
                return self._create_blocked_response("Rate limit exceeded")
            
            # Process request
            response = await call_next(request)
            
            # Log security event if threat detected
            if threat_analysis["threat_level"] != ThreatLevel.LOW:
                await self._log_security_event(
                    client_ip, user_agent, request_path, threat_analysis
                )
            
            # Add security headers
            response = self._add_security_headers(response)
            
            # Update rate limits
            await self._update_rate_limits(client_ip, threat_analysis, response.status_code)
            
            return response
            
        except Exception as e:
            logger.error(f"Security middleware error: {e}")
            return await call_next(request)
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address"""
        # Check for forwarded IP
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        # Check for real IP
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fall back to direct connection
        return request.client.host
    
    async def _is_ip_blocked(self, client_ip: str) -> bool:
        """Check if IP is blocked"""
        # Check in-memory cache
        if client_ip in self.blocked_ips:
            return True
        
        # Check Redis
        blocked_key = f"blocked_ip:{client_ip}"
        is_blocked = await self.redis.get(blocked_key)
        
        if is_blocked:
            self.blocked_ips.add(client_ip)
            return True
        
        return False
    
    async def _analyze_request(self, request: Request) -> Dict[str, Any]:
        """Analyze request for security threats"""
        threat_level = ThreatLevel.LOW
        threats_detected = []
        
        # Get request data
        request_path = str(request.url.path)
        query_params = str(request.query_params)
        headers = dict(request.headers)
        
        # Check for suspicious patterns
        for threat_type, patterns in self.suspicious_patterns.items():
            for pattern in patterns:
                if re.search(pattern, request_path, re.IGNORECASE):
                    threats_detected.append(f"{threat_type}_in_path")
                    threat_level = ThreatLevel.HIGH
                
                if re.search(pattern, query_params, re.IGNORECASE):
                    threats_detected.append(f"{threat_type}_in_query")
                    threat_level = ThreatLevel.HIGH
                
                for header_value in headers.values():
                    if re.search(pattern, str(header_value), re.IGNORECASE):
                        threats_detected.append(f"{threat_type}_in_header")
                        threat_level = ThreatLevel.MEDIUM
        
        # Check for suspicious user agents
        user_agent = headers.get("User-Agent", "")
        if self._is_suspicious_user_agent(user_agent):
            threats_detected.append("suspicious_user_agent")
            threat_level = ThreatLevel.MEDIUM
        
        # Check for missing security headers
        if not headers.get("X-Requested-With") and request.method == "POST":
            threats_detected.append("missing_csrf_protection")
            threat_level = ThreatLevel.LOW
        
        # Check for unusual request patterns
        if self._is_unusual_request(request):
            threats_detected.append("unusual_request_pattern")
            threat_level = ThreatLevel.MEDIUM
        
        return {
            "threat_level": threat_level,
            "threats_detected": threats_detected,
            "risk_score": self._calculate_risk_score(threats_detected)
        }
    
    def _is_suspicious_user_agent(self, user_agent: str) -> bool:
        """Check if user agent is suspicious"""
        suspicious_agents = [
            "sqlmap",
            "nikto",
            "nmap",
            "masscan",
            "zap",
            "burp",
            "w3af",
            "havij",
            "pangolin",
            "acunetix",
            "nessus",
            "openvas",
            "wget",
            "curl",
            "python-requests",
            "bot",
            "crawler",
            "spider",
            "scraper"
        ]
        
        user_agent_lower = user_agent.lower()
        return any(agent in user_agent_lower for agent in suspicious_agents)
    
    def _is_unusual_request(self, request: Request) -> bool:
        """Check for unusual request patterns"""
        # Check for unusually long paths
        if len(request.url.path) > 1000:
            return True
        
        # Check for unusual number of query parameters
        if len(request.query_params) > 50:
            return True
        
        # Check for unusual headers
        if len(request.headers) > 50:
            return True
        
        # Check for suspicious content types
        content_type = request.headers.get("Content-Type", "")
        if "application/x-www-form-urlencoded" in content_type and len(request.url.path) > 100:
            return True
        
        return False
    
    def _calculate_risk_score(self, threats_detected: List[str]) -> int:
        """Calculate risk score based on detected threats"""
        score = 0
        
        for threat in threats_detected:
            if "sql_injection" in threat:
                score += 50
            elif "xss" in threat:
                score += 40
            elif "command_injection" in threat:
                score += 60
            elif "path_traversal" in threat:
                score += 30
            elif "suspicious_user_agent" in threat:
                score += 20
            elif "unusual_request_pattern" in threat:
                score += 15
            else:
                score += 10
        
        return min(score, 100)
    
    async def _check_security_rate_limits(
        self, client_ip: str, threat_analysis: Dict[str, Any]
    ) -> bool:
        """Check security-specific rate limits"""
        threat_level = threat_analysis["threat_level"]
        
        if threat_level == ThreatLevel.HIGH:
            # Check for high threat rate limit
            key = f"security_rate_limit:{client_ip}:high_threat"
            current = await self.redis.incr(key)
            if current == 1:
                await self.redis.expire(key, 300)  # 5 minutes
            
            if current > 5:  # 5 high threat requests per 5 minutes
                return True
        
        elif threat_level == ThreatLevel.MEDIUM:
            # Check for medium threat rate limit
            key = f"security_rate_limit:{client_ip}:medium_threat"
            current = await self.redis.incr(key)
            if current == 1:
                await self.redis.expire(key, 600)  # 10 minutes
            
            if current > 10:  # 10 medium threat requests per 10 minutes
                return True
        
        return False
    
    async def _log_security_event(
        self, client_ip: str, user_agent: str, request_path: str, threat_analysis: Dict[str, Any]
    ):
        """Log security event"""
        try:
            event = SecurityEvent(
                id=hashlib.md5(f"{client_ip}{time.time()}".encode()).hexdigest(),
                timestamp=time.time(),
                threat_level=threat_analysis["threat_level"],
                event_type="security_threat",
                source_ip=client_ip,
                user_agent=user_agent,
                request_path=request_path,
                details=threat_analysis
            )
            
            # Store in Redis
            event_key = f"security_event:{event.id}"
            await self.redis.setex(
                event_key, 86400 * 30, json.dumps(event.__dict__, default=str)
            )
            
            # Add to security events list
            await self.redis.lpush("security_events", event.id)
            await self.redis.ltrim("security_events", 0, 9999)  # Keep last 10000 events
            
            logger.warning(f"Security threat detected: {event.id} from {client_ip}")
            
        except Exception as e:
            logger.error(f"Error logging security event: {e}")
    
    async def _block_ip(self, client_ip: str, reason: str):
        """Block IP address"""
        try:
            # Add to blocked IPs set
            self.blocked_ips.add(client_ip)
            
            # Store in Redis
            blocked_key = f"blocked_ip:{client_ip}"
            await self.redis.setex(blocked_key, 86400, reason)  # Block for 24 hours
            
            # Log blocking event
            logger.warning(f"IP {client_ip} blocked: {reason}")
            
        except Exception as e:
            logger.error(f"Error blocking IP: {e}")
    
    async def _update_rate_limits(
        self, client_ip: str, threat_analysis: Dict[str, Any], status_code: int
    ):
        """Update rate limits based on request"""
        try:
            # Update general request rate limit
            key = f"request_rate_limit:{client_ip}"
            current = await self.redis.incr(key)
            if current == 1:
                await self.redis.expire(key, 60)  # 1 minute
            
            # Update threat-specific rate limits
            threat_level = threat_analysis["threat_level"]
            if threat_level != ThreatLevel.LOW:
                threat_key = f"threat_rate_limit:{client_ip}:{threat_level.value}"
                current = await self.redis.incr(threat_key)
                if current == 1:
                    await self.redis.expire(threat_key, 300)  # 5 minutes
            
        except Exception as e:
            logger.error(f"Error updating rate limits: {e}")
    
    def _add_security_headers(self, response: Response) -> Response:
        """Add security headers to response"""
        security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": "default-src 'self'",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()"
        }
        
        for header, value in security_headers.items():
            response.headers[header] = value
        
        return response
    
    def _create_blocked_response(self, message: str) -> JSONResponse:
        """Create blocked response"""
        return JSONResponse(
            status_code=403,
            content={
                "error": "Access Denied",
                "message": message,
                "timestamp": time.time()
            }
        )
    
    async def get_security_stats(self) -> Dict[str, Any]:
        """Get security statistics"""
        try:
            stats = {
                "blocked_ips": len(self.blocked_ips),
                "total_events": 0,
                "threats_by_level": {},
                "threats_by_type": {},
                "top_threat_ips": []
            }
            
            # Get total events
            total_events = await self.redis.llen("security_events")
            stats["total_events"] = total_events
            
            # Get recent events for analysis
            recent_events = await self.redis.lrange("security_events", 0, 99)
            
            for event_id in recent_events:
                event_key = f"security_event:{event_id.decode()}"
                event_data = await self.redis.get(event_key)
                
                if event_data:
                    event = json.loads(event_data)
                    
                    # Count by threat level
                    threat_level = event.get("threat_level", "low")
                    stats["threats_by_level"][threat_level] = stats["threats_by_level"].get(threat_level, 0) + 1
                    
                    # Count by threat type
                    threats_detected = event.get("details", {}).get("threats_detected", [])
                    for threat in threats_detected:
                        stats["threats_by_type"][threat] = stats["threats_by_type"].get(threat, 0) + 1
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting security stats: {e}")
            return {"error": str(e)}
    
    async def unblock_ip(self, client_ip: str) -> bool:
        """Unblock IP address"""
        try:
            # Remove from in-memory cache
            self.blocked_ips.discard(client_ip)
            
            # Remove from Redis
            blocked_key = f"blocked_ip:{client_ip}"
            await self.redis.delete(blocked_key)
            
            logger.info(f"IP {client_ip} unblocked")
            return True
            
        except Exception as e:
            logger.error(f"Error unblocking IP: {e}")
            return False

























