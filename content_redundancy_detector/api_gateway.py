"""
API Gateway Integration and Configuration
Supports Kong, AWS API Gateway, Traefik patterns
"""

import logging
import os
from typing import Dict, Any, Optional, List
from fastapi import Request, Header
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class APIGatewayMiddleware:
    """
    Middleware for API Gateway integration
    Handles gateway-specific headers and routing
    """
    
    def __init__(self):
        self.gateway_type = self._detect_gateway()
    
    def _detect_gateway(self) -> str:
        """Detect API Gateway type"""
        if os.getenv("AWS_API_GATEWAY_REQUEST_ID"):
            return "aws"
        elif os.getenv("KONG_REQUEST_ID"):
            return "kong"
        elif os.getenv("TRAEFIK_BACKEND"):
            return "traefik"
        elif "x-forwarded-for" in os.environ:
            return "generic"
        return "none"
    
    async def process_request(self, request: Request) -> Dict[str, Any]:
        """Process gateway-specific request headers"""
        gateway_info = {
            "type": self.gateway_type,
            "request_id": None,
            "client_ip": None,
            "protocol": None,
        }
        
        # AWS API Gateway
        if self.gateway_type == "aws":
            gateway_info["request_id"] = request.headers.get("x-amzn-requestid")
            gateway_info["client_ip"] = request.headers.get("x-forwarded-for", "").split(",")[0]
            gateway_info["protocol"] = request.headers.get("x-forwarded-proto", "https")
        
        # Kong Gateway
        elif self.gateway_type == "kong":
            gateway_info["request_id"] = request.headers.get("kong-request-id")
            gateway_info["client_ip"] = request.headers.get("x-real-ip")
            gateway_info["protocol"] = request.headers.get("x-forwarded-proto", "https")
        
        # Traefik
        elif self.gateway_type == "traefik":
            gateway_info["request_id"] = request.headers.get("x-request-id")
            gateway_info["client_ip"] = request.headers.get("x-forwarded-for", "").split(",")[0]
            gateway_info["protocol"] = request.headers.get("x-forwarded-proto", "https")
        
        # Generic (NGINX, etc.)
        else:
            gateway_info["request_id"] = request.headers.get("x-request-id")
            gateway_info["client_ip"] = request.headers.get("x-forwarded-for", "").split(",")[0] or request.client.host
            gateway_info["protocol"] = request.headers.get("x-forwarded-proto") or "http"
        
        return gateway_info


# Global gateway middleware
api_gateway = APIGatewayMiddleware()


class RateLimitConfig:
    """Rate limiting configuration for API Gateway"""
    
    # Per-endpoint rate limits
    endpoint_limits: Dict[str, Dict[str, int]] = {
        "/api/v1/analyze": {"requests": 100, "window": 60},
        "/api/v1/similarity": {"requests": 100, "window": 60},
        "/api/v1/quality": {"requests": 100, "window": 60},
        "/api/v1/ai/comprehensive": {"requests": 50, "window": 60},
        "/api/v1/batch/process": {"requests": 20, "window": 60},
    }
    
    # Per-user rate limits (if authenticated)
    user_limits: Dict[str, int] = {
        "free": 100,  # per hour
        "premium": 1000,  # per hour
        "enterprise": 10000,  # per hour
    }
    
    # IP-based rate limits
    ip_limits: Dict[str, int] = {
        "default": 1000,  # per hour
        "strict": 100,  # per hour
    }


class APIGatewayHeaders:
    """API Gateway response headers"""
    
    @staticmethod
    def add_gateway_headers(response: JSONResponse, request: Request) -> JSONResponse:
        """Add API Gateway specific headers"""
        gateway_info = api_gateway.process_request(request)
        
        if gateway_info["request_id"]:
            response.headers["X-Request-ID"] = gateway_info["request_id"]
        
        response.headers["X-Gateway-Type"] = gateway_info["type"]
        
        # CORS headers for API Gateway
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-API-Key"
        
        return response


def get_api_key_from_gateway(request: Request) -> Optional[str]:
    """Extract API key from gateway headers"""
    # AWS API Gateway
    api_key = request.headers.get("x-api-key")
    
    # Kong Gateway
    if not api_key:
        api_key = request.headers.get("kong-api-key")
    
    # Generic
    if not api_key:
        api_key = request.headers.get("authorization")
        if api_key and api_key.startswith("Bearer "):
            api_key = api_key.replace("Bearer ", "")
    
    return api_key


class GatewayHealthCheck:
    """Health check endpoint for API Gateway"""
    
    @staticmethod
    async def health_check() -> Dict[str, Any]:
        """Health check response for gateway monitoring"""
        return {
            "status": "healthy",
            "service": "content-redundancy-detector",
            "gateway": api_gateway.gateway_type,
            "version": os.getenv("APP_VERSION", "1.0.0"),
        }


# Kong-specific configuration
KONG_PLUGINS = {
    "rate-limiting": {
        "minute": 100,
        "hour": 1000,
    },
    "cors": {
        "origins": ["*"],
        "methods": ["GET", "POST", "PUT", "DELETE"],
        "headers": ["Content-Type", "Authorization"],
    },
    "request-id": {
        "header_name": "X-Request-ID",
    },
    "security": {
        "hide_credentials": True,
    },
}


# AWS API Gateway configuration
AWS_API_GATEWAY_CONFIG = {
    "throttle_burst_limit": 100,
    "throttle_rate_limit": 100,
    "api_key_source": "HEADER",
    "cors": {
        "allow_origins": ["*"],
        "allow_methods": ["*"],
        "allow_headers": ["*"],
    },
}


# Traefik configuration example
TRAEFIK_CONFIG = """
[http.middlewares]
  [http.middlewares.content-detector-headers.headers]
    customRequestHeaders = "X-Service-Name=content-redundancy-detector"
  
  [http.middlewares.content-detector-ratelimit.ratelimit]
    average = 100
    burst = 50
"""


def configure_for_gateway(gateway_type: str) -> Dict[str, Any]:
    """
    Get configuration for specific API Gateway
    
    Returns:
        Configuration dictionary for the gateway
    """
    configs = {
        "kong": KONG_PLUGINS,
        "aws": AWS_API_GATEWAY_CONFIG,
        "traefik": {"middleware": TRAEFIK_CONFIG},
    }
    
    return configs.get(gateway_type, {})






