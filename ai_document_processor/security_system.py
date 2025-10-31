"""
Security System for AI Document Processor
Real, working security features for document processing
"""

import asyncio
import logging
import json
import hashlib
import secrets
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import re
import os

logger = logging.getLogger(__name__)

class SecuritySystem:
    """Real working security system for AI document processing"""
    
    def __init__(self):
        self.api_keys = {}
        self.rate_limits = {}
        self.blocked_ips = set()
        self.suspicious_activities = []
        self.security_logs = []
        
        # Security configuration
        self.max_requests_per_minute = 100
        self.max_requests_per_hour = 1000
        self.max_file_size = 50 * 1024 * 1024  # 50MB
        self.allowed_file_types = ['.pdf', '.docx', '.xlsx', '.pptx', '.txt', '.jpg', '.jpeg', '.png']
        self.blocked_patterns = [
            r'<script.*?>.*?</script>',
            r'javascript:',
            r'data:text/html',
            r'vbscript:',
            r'onload=',
            r'onerror='
        ]
        
        # Security stats
        self.stats = {
            "total_requests": 0,
            "blocked_requests": 0,
            "rate_limited_requests": 0,
            "suspicious_activities": 0,
            "security_violations": 0,
            "start_time": time.time()
        }
    
    async def validate_request(self, request_data: Dict[str, Any], 
                             client_ip: str, api_key: str = None) -> Dict[str, Any]:
        """Validate incoming request for security"""
        try:
            validation_result = {
                "valid": True,
                "reason": "",
                "security_level": "normal"
            }
            
            # Check API key if provided
            if api_key:
                if not await self._validate_api_key(api_key):
                    validation_result["valid"] = False
                    validation_result["reason"] = "Invalid API key"
                    self._log_security_event("invalid_api_key", client_ip, api_key)
                    return validation_result
            
            # Check rate limiting
            if not await self._check_rate_limit(client_ip):
                validation_result["valid"] = False
                validation_result["reason"] = "Rate limit exceeded"
                self._log_security_event("rate_limit_exceeded", client_ip)
                return validation_result
            
            # Check if IP is blocked
            if client_ip in self.blocked_ips:
                validation_result["valid"] = False
                validation_result["reason"] = "IP address blocked"
                self._log_security_event("blocked_ip_access", client_ip)
                return validation_result
            
            # Validate text content
            if "text" in request_data:
                text_validation = await self._validate_text_content(request_data["text"])
                if not text_validation["valid"]:
                    validation_result["valid"] = False
                    validation_result["reason"] = text_validation["reason"]
                    self._log_security_event("malicious_content", client_ip, request_data["text"][:100])
                    return validation_result
            
            # Check for suspicious patterns
            if await self._detect_suspicious_activity(request_data, client_ip):
                validation_result["security_level"] = "high"
                self._log_security_event("suspicious_activity", client_ip, str(request_data))
            
            # Update stats
            self.stats["total_requests"] += 1
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error validating request: {e}")
            return {
                "valid": False,
                "reason": "Security validation error",
                "security_level": "high"
            }
    
    async def validate_file_upload(self, file_content: bytes, filename: str, 
                                 client_ip: str) -> Dict[str, Any]:
        """Validate file upload for security"""
        try:
            validation_result = {
                "valid": True,
                "reason": "",
                "security_level": "normal"
            }
            
            # Check file size
            if len(file_content) > self.max_file_size:
                validation_result["valid"] = False
                validation_result["reason"] = f"File too large: {len(file_content)} bytes"
                self._log_security_event("file_too_large", client_ip, filename)
                return validation_result
            
            # Check file extension
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext not in self.allowed_file_types:
                validation_result["valid"] = False
                validation_result["reason"] = f"File type not allowed: {file_ext}"
                self._log_security_event("invalid_file_type", client_ip, filename)
                return validation_result
            
            # Check for malicious content in file
            if await self._scan_file_content(file_content, filename):
                validation_result["valid"] = False
                validation_result["reason"] = "Malicious content detected in file"
                self._log_security_event("malicious_file", client_ip, filename)
                return validation_result
            
            # Check file signature
            if not await self._validate_file_signature(file_content, file_ext):
                validation_result["valid"] = False
                validation_result["reason"] = "Invalid file signature"
                self._log_security_event("invalid_file_signature", client_ip, filename)
                return validation_result
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error validating file upload: {e}")
            return {
                "valid": False,
                "reason": "File validation error",
                "security_level": "high"
            }
    
    async def _validate_api_key(self, api_key: str) -> bool:
        """Validate API key"""
        try:
            # Check if API key exists and is not expired
            if api_key in self.api_keys:
                key_data = self.api_keys[api_key]
                if datetime.now() < key_data["expires"]:
                    return True
                else:
                    # Remove expired key
                    del self.api_keys[api_key]
            return False
        except Exception as e:
            logger.error(f"Error validating API key: {e}")
            return False
    
    async def _check_rate_limit(self, client_ip: str) -> bool:
        """Check rate limiting for client IP"""
        try:
            current_time = time.time()
            
            # Initialize rate limit data for IP
            if client_ip not in self.rate_limits:
                self.rate_limits[client_ip] = {
                    "requests": [],
                    "last_cleanup": current_time
                }
            
            ip_data = self.rate_limits[client_ip]
            
            # Clean old requests (older than 1 hour)
            if current_time - ip_data["last_cleanup"] > 3600:
                ip_data["requests"] = [
                    req_time for req_time in ip_data["requests"] 
                    if current_time - req_time < 3600
                ]
                ip_data["last_cleanup"] = current_time
            
            # Check minute rate limit
            minute_requests = [
                req_time for req_time in ip_data["requests"] 
                if current_time - req_time < 60
            ]
            if len(minute_requests) >= self.max_requests_per_minute:
                self.stats["rate_limited_requests"] += 1
                return False
            
            # Check hour rate limit
            hour_requests = [
                req_time for req_time in ip_data["requests"] 
                if current_time - req_time < 3600
            ]
            if len(hour_requests) >= self.max_requests_per_hour:
                self.stats["rate_limited_requests"] += 1
                return False
            
            # Add current request
            ip_data["requests"].append(current_time)
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking rate limit: {e}")
            return False
    
    async def _validate_text_content(self, text: str) -> Dict[str, Any]:
        """Validate text content for malicious patterns"""
        try:
            if not text or len(text.strip()) == 0:
                return {"valid": True, "reason": ""}
            
            # Check for blocked patterns
            for pattern in self.blocked_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return {
                        "valid": False,
                        "reason": f"Malicious pattern detected: {pattern}"
                    }
            
            # Check for excessive length
            if len(text) > 1000000:  # 1MB text limit
                return {
                    "valid": False,
                    "reason": "Text content too long"
                }
            
            # Check for suspicious encoding
            try:
                text.encode('utf-8')
            except UnicodeEncodeError:
                return {
                    "valid": False,
                    "reason": "Invalid text encoding"
                }
            
            return {"valid": True, "reason": ""}
            
        except Exception as e:
            logger.error(f"Error validating text content: {e}")
            return {"valid": False, "reason": "Text validation error"}
    
    async def _detect_suspicious_activity(self, request_data: Dict[str, Any], 
                                       client_ip: str) -> bool:
        """Detect suspicious activity patterns"""
        try:
            # Check for rapid repeated requests
            if client_ip in self.rate_limits:
                recent_requests = [
                    req_time for req_time in self.rate_limits[client_ip]["requests"]
                    if time.time() - req_time < 10  # Last 10 seconds
                ]
                if len(recent_requests) > 10:  # More than 10 requests in 10 seconds
                    return True
            
            # Check for unusual request patterns
            if "text" in request_data:
                text = request_data["text"]
                # Check for repeated identical requests
                if len(text) > 1000 and text.count(text[:100]) > 5:
                    return True
                
                # Check for random character patterns
                if len(set(text)) < 10 and len(text) > 100:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error detecting suspicious activity: {e}")
            return False
    
    async def _scan_file_content(self, file_content: bytes, filename: str) -> bool:
        """Scan file content for malicious patterns"""
        try:
            # Convert to string for pattern matching
            content_str = file_content.decode('utf-8', errors='ignore')
            
            # Check for malicious patterns
            for pattern in self.blocked_patterns:
                if re.search(pattern, content_str, re.IGNORECASE):
                    return True
            
            # Check for suspicious file headers
            suspicious_headers = [
                b'<script',
                b'javascript:',
                b'vbscript:',
                b'data:text/html'
            ]
            
            for header in suspicious_headers:
                if header in file_content:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error scanning file content: {e}")
            return True  # Err on the side of caution
    
    async def _validate_file_signature(self, file_content: bytes, file_ext: str) -> bool:
        """Validate file signature matches extension"""
        try:
            if not file_content:
                return False
            
            # Check file signatures
            signatures = {
                '.pdf': [b'%PDF'],
                '.docx': [b'PK\x03\x04', b'PK\x05\x06'],
                '.xlsx': [b'PK\x03\x04', b'PK\x05\x06'],
                '.pptx': [b'PK\x03\x04', b'PK\x05\x06'],
                '.txt': [],  # No specific signature for text files
                '.jpg': [b'\xff\xd8\xff'],
                '.jpeg': [b'\xff\xd8\xff'],
                '.png': [b'\x89PNG\r\n\x1a\n']
            }
            
            if file_ext in signatures:
                expected_signatures = signatures[file_ext]
                if expected_signatures:
                    return any(file_content.startswith(sig) for sig in expected_signatures)
                else:
                    return True  # No signature to check
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating file signature: {e}")
            return False
    
    def _log_security_event(self, event_type: str, client_ip: str, details: str = ""):
        """Log security event"""
        try:
            event = {
                "timestamp": datetime.now().isoformat(),
                "event_type": event_type,
                "client_ip": client_ip,
                "details": details
            }
            
            self.security_logs.append(event)
            self.stats["security_violations"] += 1
            
            # Keep only last 1000 events
            if len(self.security_logs) > 1000:
                self.security_logs = self.security_logs[-1000:]
            
            logger.warning(f"Security event: {event_type} from {client_ip}")
            
        except Exception as e:
            logger.error(f"Error logging security event: {e}")
    
    async def generate_api_key(self, expires_hours: int = 24) -> str:
        """Generate new API key"""
        try:
            api_key = secrets.token_urlsafe(32)
            expires = datetime.now() + timedelta(hours=expires_hours)
            
            self.api_keys[api_key] = {
                "created": datetime.now(),
                "expires": expires,
                "uses": 0
            }
            
            return api_key
            
        except Exception as e:
            logger.error(f"Error generating API key: {e}")
            return ""
    
    async def block_ip(self, client_ip: str, reason: str = "Security violation"):
        """Block IP address"""
        try:
            self.blocked_ips.add(client_ip)
            self._log_security_event("ip_blocked", client_ip, reason)
            logger.warning(f"IP {client_ip} blocked: {reason}")
        except Exception as e:
            logger.error(f"Error blocking IP: {e}")
    
    async def unblock_ip(self, client_ip: str):
        """Unblock IP address"""
        try:
            self.blocked_ips.discard(client_ip)
            self._log_security_event("ip_unblocked", client_ip)
            logger.info(f"IP {client_ip} unblocked")
        except Exception as e:
            logger.error(f"Error unblocking IP: {e}")
    
    def get_security_stats(self) -> Dict[str, Any]:
        """Get security statistics"""
        uptime = time.time() - self.stats["start_time"]
        return {
            "stats": self.stats.copy(),
            "uptime_seconds": round(uptime, 2),
            "uptime_hours": round(uptime / 3600, 2),
            "blocked_ips_count": len(self.blocked_ips),
            "active_api_keys": len(self.api_keys),
            "recent_security_events": self.security_logs[-10:]
        }
    
    def get_security_config(self) -> Dict[str, Any]:
        """Get security configuration"""
        return {
            "max_requests_per_minute": self.max_requests_per_minute,
            "max_requests_per_hour": self.max_requests_per_hour,
            "max_file_size": self.max_file_size,
            "allowed_file_types": self.allowed_file_types,
            "blocked_patterns": self.blocked_patterns,
            "blocked_ips": list(self.blocked_ips),
            "security_level": "high"
        }

# Global instance
security_system = SecuritySystem()













