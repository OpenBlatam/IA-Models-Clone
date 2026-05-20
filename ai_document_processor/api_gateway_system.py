"""
API Gateway System for AI Document Processor
Real, working API gateway features for document processing
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import hashlib
import secrets

logger = logging.getLogger(__name__)

class APIGatewaySystem:
    """Real working API gateway system for AI document processing"""
    
    def __init__(self):
        self.routes = {}
        self.middleware = []
        self.rate_limits = {}
        self.api_keys = {}
        self.request_logs = []
        self.circuit_breakers = {}
        
        # Gateway stats
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "rate_limited_requests": 0,
            "circuit_breaker_trips": 0,
            "start_time": time.time()
        }
        
        # Initialize default routes
        self._initialize_default_routes()
    
    def _initialize_default_routes(self):
        """Initialize default API routes"""
        self.routes = {
            "/api/v1/real": {
                "service": "basic_ai",
                "endpoint": "http://localhost:8001",
                "methods": ["GET", "POST"],
                "rate_limit": 100,
                "timeout": 30
            },
            "/api/v1/advanced-real": {
                "service": "advanced_ai",
                "endpoint": "http://localhost:8002",
                "methods": ["GET", "POST"],
                "rate_limit": 50,
                "timeout": 60
            },
            "/api/v1/upload": {
                "service": "document_upload",
                "endpoint": "http://localhost:8003",
                "methods": ["GET", "POST"],
                "rate_limit": 20,
                "timeout": 120
            },
            "/api/v1/monitoring": {
                "service": "monitoring",
                "endpoint": "http://localhost:8004",
                "methods": ["GET"],
                "rate_limit": 200,
                "timeout": 10
            },
            "/api/v1/security": {
                "service": "security",
                "endpoint": "http://localhost:8005",
                "methods": ["GET", "POST"],
                "rate_limit": 100,
                "timeout": 15
            },
            "/api/v1/notifications": {
                "service": "notifications",
                "endpoint": "http://localhost:8006",
                "methods": ["GET", "POST"],
                "rate_limit": 50,
                "timeout": 30
            },
            "/api/v1/analytics": {
                "service": "analytics",
                "endpoint": "http://localhost:8007",
                "methods": ["GET", "POST"],
                "rate_limit": 30,
                "timeout": 60
            },
            "/api/v1/backup": {
                "service": "backup",
                "endpoint": "http://localhost:8008",
                "methods": ["GET", "POST"],
                "rate_limit": 10,
                "timeout": 300
            },
            "/api/v1/workflow": {
                "service": "workflow",
                "endpoint": "http://localhost:8009",
                "methods": ["GET", "POST"],
                "rate_limit": 20,
                "timeout": 180
            },
            "/api/v1/config": {
                "service": "config",
                "endpoint": "http://localhost:8010",
                "methods": ["GET", "POST"],
                "rate_limit": 50,
                "timeout": 30
            }
        }
    
    async def register_route(self, path: str, service: str, endpoint: str, 
                           methods: List[str] = None, rate_limit: int = 100,
                           timeout: int = 30) -> Dict[str, Any]:
        """Register a new route"""
        try:
            self.routes[path] = {
                "service": service,
                "endpoint": endpoint,
                "methods": methods or ["GET", "POST"],
                "rate_limit": rate_limit,
                "timeout": timeout,
                "registered_at": datetime.now().isoformat()
            }
            
            return {
                "status": "registered",
                "path": path,
                "service": service,
                "endpoint": endpoint
            }
            
        except Exception as e:
            logger.error(f"Error registering route: {e}")
            return {"error": str(e)}
    
    async def unregister_route(self, path: str) -> Dict[str, Any]:
        """Unregister a route"""
        try:
            if path in self.routes:
                del self.routes[path]
                return {
                    "status": "unregistered",
                    "path": path
                }
            else:
                return {"error": f"Route '{path}' not found"}
                
        except Exception as e:
            logger.error(f"Error unregistering route: {e}")
            return {"error": str(e)}
    
    async def add_middleware(self, middleware_name: str, middleware_func: callable) -> Dict[str, Any]:
        """Add middleware to the gateway"""
        try:
            self.middleware.append({
                "name": middleware_name,
                "function": middleware_func,
                "added_at": datetime.now().isoformat()
            })
            
            return {
                "status": "added",
                "middleware": middleware_name
            }
            
        except Exception as e:
            logger.error(f"Error adding middleware: {e}")
            return {"error": str(e)}
    
    async def check_rate_limit(self, client_ip: str, route: str) -> Dict[str, Any]:
        """Check rate limit for client"""
        try:
            if route not in self.routes:
                return {"allowed": True, "reason": "route_not_found"}
            
            route_config = self.routes[route]
            rate_limit = route_config["rate_limit"]
            
            # Initialize rate limit tracking for this client/route
            key = f"{client_ip}:{route}"
            if key not in self.rate_limits:
                self.rate_limits[key] = {
                    "requests": 0,
                    "window_start": time.time(),
                    "blocked_until": 0
                }
            
            rate_data = self.rate_limits[key]
            current_time = time.time()
            
            # Reset window if needed (1 minute windows)
            if current_time - rate_data["window_start"] > 60:
                rate_data["requests"] = 0
                rate_data["window_start"] = current_time
                rate_data["blocked_until"] = 0
            
            # Check if client is still blocked
            if current_time < rate_data["blocked_until"]:
                return {
                    "allowed": False,
                    "reason": "rate_limited",
                    "retry_after": int(rate_data["blocked_until"] - current_time)
                }
            
            # Check rate limit
            if rate_data["requests"] >= rate_limit:
                # Block client for 1 minute
                rate_data["blocked_until"] = current_time + 60
                self.stats["rate_limited_requests"] += 1
                
                return {
                    "allowed": False,
                    "reason": "rate_limited",
                    "retry_after": 60
                }
            
            # Increment request count
            rate_data["requests"] += 1
            
            return {
                "allowed": True,
                "requests_remaining": rate_limit - rate_data["requests"],
                "window_reset": int(60 - (current_time - rate_data["window_start"]))
            }
            
        except Exception as e:
            logger.error(f"Error checking rate limit: {e}")
            return {"allowed": True, "error": str(e)}
    
    async def generate_api_key(self, service: str, expires_hours: int = 24) -> Dict[str, Any]:
        """Generate API key for service access"""
        try:
            api_key = secrets.token_urlsafe(32)
            expires_at = datetime.now() + timedelta(hours=expires_hours)
            
            self.api_keys[api_key] = {
                "service": service,
                "created_at": datetime.now().isoformat(),
                "expires_at": expires_at.isoformat(),
                "active": True
            }
            
            return {
                "api_key": api_key,
                "service": service,
                "expires_at": expires_at.isoformat(),
                "expires_in_hours": expires_hours
            }
            
        except Exception as e:
            logger.error(f"Error generating API key: {e}")
            return {"error": str(e)}
    
    async def validate_api_key(self, api_key: str) -> Dict[str, Any]:
        """Validate API key"""
        try:
            if api_key not in self.api_keys:
                return {"valid": False, "reason": "key_not_found"}
            
            key_data = self.api_keys[api_key]
            
            # Check if key is active
            if not key_data["active"]:
                return {"valid": False, "reason": "key_inactive"}
            
            # Check if key is expired
            expires_at = datetime.fromisoformat(key_data["expires_at"])
            if datetime.now() > expires_at:
                return {"valid": False, "reason": "key_expired"}
            
            return {
                "valid": True,
                "service": key_data["service"],
                "expires_at": key_data["expires_at"]
            }
            
        except Exception as e:
            logger.error(f"Error validating API key: {e}")
            return {"valid": False, "error": str(e)}
    
    async def revoke_api_key(self, api_key: str) -> Dict[str, Any]:
        """Revoke API key"""
        try:
            if api_key in self.api_keys:
                self.api_keys[api_key]["active"] = False
                return {
                    "status": "revoked",
                    "api_key": api_key
                }
            else:
                return {"error": "API key not found"}
                
        except Exception as e:
            logger.error(f"Error revoking API key: {e}")
            return {"error": str(e)}
    
    async def check_circuit_breaker(self, service: str) -> Dict[str, Any]:
        """Check circuit breaker status for service"""
        try:
            if service not in self.circuit_breakers:
                self.circuit_breakers[service] = {
                    "state": "closed",  # closed, open, half-open
                    "failure_count": 0,
                    "last_failure": None,
                    "next_attempt": None
                }
            
            breaker = self.circuit_breakers[service]
            current_time = time.time()
            
            # Check if circuit should be reset
            if breaker["state"] == "open":
                if breaker["next_attempt"] and current_time >= breaker["next_attempt"]:
                    breaker["state"] = "half-open"
                    breaker["next_attempt"] = None
            
            return {
                "service": service,
                "state": breaker["state"],
                "failure_count": breaker["failure_count"],
                "last_failure": breaker["last_failure"]
            }
            
        except Exception as e:
            logger.error(f"Error checking circuit breaker: {e}")
            return {"error": str(e)}
    
    async def record_circuit_breaker_failure(self, service: str) -> Dict[str, Any]:
        """Record circuit breaker failure"""
        try:
            if service not in self.circuit_breakers:
                self.circuit_breakers[service] = {
                    "state": "closed",
                    "failure_count": 0,
                    "last_failure": None,
                    "next_attempt": None
                }
            
            breaker = self.circuit_breakers[service]
            breaker["failure_count"] += 1
            breaker["last_failure"] = datetime.now().isoformat()
            
            # Open circuit if failure threshold reached (5 failures)
            if breaker["failure_count"] >= 5:
                breaker["state"] = "open"
                breaker["next_attempt"] = time.time() + 300  # 5 minutes
                self.stats["circuit_breaker_trips"] += 1
            
            return {
                "service": service,
                "state": breaker["state"],
                "failure_count": breaker["failure_count"]
            }
            
        except Exception as e:
            logger.error(f"Error recording circuit breaker failure: {e}")
            return {"error": str(e)}
    
    async def record_circuit_breaker_success(self, service: str) -> Dict[str, Any]:
        """Record circuit breaker success"""
        try:
            if service not in self.circuit_breakers:
                return {"error": "Circuit breaker not found for service"}
            
            breaker = self.circuit_breakers[service]
            
            # Reset circuit breaker on success
            if breaker["state"] == "half-open":
                breaker["state"] = "closed"
                breaker["failure_count"] = 0
                breaker["last_failure"] = None
                breaker["next_attempt"] = None
            
            return {
                "service": service,
                "state": breaker["state"],
                "failure_count": breaker["failure_count"]
            }
            
        except Exception as e:
            logger.error(f"Error recording circuit breaker success: {e}")
            return {"error": str(e)}
    
    async def log_request(self, client_ip: str, method: str, path: str, 
                         status_code: int, response_time: float, 
                         service: str = None) -> Dict[str, Any]:
        """Log API request"""
        try:
            request_log = {
                "timestamp": datetime.now().isoformat(),
                "client_ip": client_ip,
                "method": method,
                "path": path,
                "status_code": status_code,
                "response_time": response_time,
                "service": service
            }
            
            self.request_logs.append(request_log)
            
            # Keep only last 1000 logs
            if len(self.request_logs) > 1000:
                self.request_logs = self.request_logs[-1000:]
            
            # Update stats
            self.stats["total_requests"] += 1
            if 200 <= status_code < 300:
                self.stats["successful_requests"] += 1
            else:
                self.stats["failed_requests"] += 1
            
            return {"logged": True}
            
        except Exception as e:
            logger.error(f"Error logging request: {e}")
            return {"error": str(e)}
    
    def get_routes(self) -> Dict[str, Any]:
        """Get all registered routes"""
        return {
            "routes": self.routes,
            "route_count": len(self.routes)
        }
    
    def get_middleware(self) -> List[Dict[str, Any]]:
        """Get all middleware"""
        return self.middleware
    
    def get_api_keys(self, active_only: bool = True) -> Dict[str, Any]:
        """Get API keys"""
        if active_only:
            active_keys = {k: v for k, v in self.api_keys.items() if v["active"]}
            return {
                "api_keys": active_keys,
                "active_count": len(active_keys)
            }
        else:
            return {
                "api_keys": self.api_keys,
                "total_count": len(self.api_keys)
            }
    
    def get_request_logs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent request logs"""
        return self.request_logs[-limit:]
    
    def get_circuit_breakers(self) -> Dict[str, Any]:
        """Get circuit breaker status"""
        return {
            "circuit_breakers": self.circuit_breakers,
            "total_services": len(self.circuit_breakers)
        }
    
    def get_gateway_stats(self) -> Dict[str, Any]:
        """Get gateway statistics"""
        uptime = time.time() - self.stats["start_time"]
        return {
            "stats": self.stats.copy(),
            "uptime_seconds": round(uptime, 2),
            "uptime_hours": round(uptime / 3600, 2),
            "routes_count": len(self.routes),
            "middleware_count": len(self.middleware),
            "api_keys_count": len(self.api_keys),
            "circuit_breakers_count": len(self.circuit_breakers)
        }

# Global instance
api_gateway_system = APIGatewaySystem()













