"""
PDF Variantes Security Service
Security service wrapper for main API
"""

import logging
from typing import Any, Dict, Optional
from fastapi import Request
from ..utils.config import Settings
from ..utils.auth import SecurityService as BaseSecurityService

logger = logging.getLogger(__name__)

class SecurityResult:
    """Security check result"""
    def __init__(self, is_safe: bool = True, reason: Optional[str] = None):
        self.is_safe = is_safe
        self.reason = reason

class SecurityService:
    """Security service for PDF Variantes API"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.security_service = BaseSecurityService(settings)
    
    async def initialize(self):
        """Initialize security service"""
        try:
            await self.security_service.initialize()
            logger.info("Security Service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Security Service: {e}")
            # Don't raise - security will work with defaults
    
    async def cleanup(self):
        """Cleanup security service"""
        try:
            if hasattr(self.security_service, 'cleanup'):
                await self.security_service.cleanup()
            elif hasattr(self.security_service, 'close'):
                await self.security_service.close()
        except Exception as e:
            logger.error(f"Error cleaning up Security Service: {e}")
    
    async def check_request(self, request: Request) -> SecurityResult:
        """Check request for security issues"""
        try:
            # Basic security checks
            client_ip = request.client.host if request.client else "unknown"
            
            # Check for blocked IPs
            if hasattr(self.security_service, 'is_ip_blocked'):
                if await self.security_service.is_ip_blocked(client_ip):
                    return SecurityResult(is_safe=False, reason="IP address is blocked")
            
            # Check for suspicious patterns
            if hasattr(self.security_service, 'is_suspicious_request'):
                if await self.security_service.is_suspicious_request(request):
                    return SecurityResult(is_safe=False, reason="Suspicious request pattern detected")
            
            return SecurityResult(is_safe=True)
            
        except Exception as e:
            logger.error(f"Error checking request security: {e}")
            # Fail open - allow request but log error
            return SecurityResult(is_safe=True)
    
    async def log_request(self, request: Request, response: Any):
        """Log security event for request"""
        try:
            # Log security metrics
            if response.status_code >= 400:
                if hasattr(self.security_service, 'track_security_event'):
                    await self.security_service.track_security_event(
                        "request_error",
                        {"status_code": response.status_code}
                    )
        except Exception as e:
            logger.error(f"Error logging security event: {e}")





