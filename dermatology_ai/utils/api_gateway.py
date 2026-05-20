"""
API Gateway Integration Patterns
Supports Kong, AWS API Gateway, and NGINX patterns
"""

import os
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class GatewayType(str, Enum):
    """Supported API Gateway types"""
    KONG = "kong"
    AWS = "aws"
    NGINX = "nginx"
    TRAEFIK = "traefik"
    NONE = "none"


@dataclass
class GatewayConfig:
    """API Gateway configuration"""
    gateway_type: GatewayType = GatewayType.NONE
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    rate_limit_enabled: bool = True
    rate_limit_per_minute: int = 100
    cors_enabled: bool = True
    authentication_enabled: bool = True


class APIGatewayClient:
    """
    Client for API Gateway operations.
    Supports Kong, AWS API Gateway patterns.
    """
    
    def __init__(self, config: GatewayConfig):
        self.config = config
        self.gateway_type = config.gateway_type
    
    async def register_service(
        self,
        service_name: str,
        service_url: str,
        routes: List[Dict[str, Any]]
    ) -> bool:
        """
        Register service with API Gateway
        
        Args:
            service_name: Name of the service
            service_url: URL of the service
            routes: List of routes to register
            
        Returns:
            True if successful
        """
        if self.gateway_type == GatewayType.KONG:
            return await self._register_kong(service_name, service_url, routes)
        elif self.gateway_type == GatewayType.AWS:
            return await self._register_aws(service_name, service_url, routes)
        else:
            logger.warning(f"Gateway type {self.gateway_type} not supported for registration")
            return False
    
    async def _register_kong(
        self,
        service_name: str,
        service_url: str,
        routes: List[Dict[str, Any]]
    ) -> bool:
        """Register service with Kong"""
        try:
            import aiohttp
            
            kong_admin_url = self.config.base_url or os.getenv("KONG_ADMIN_URL", "http://localhost:8001")
            
            async with aiohttp.ClientSession() as session:
                # Create service
                service_data = {
                    "name": service_name,
                    "url": service_url,
                }
                
                async with session.post(
                    f"{kong_admin_url}/services",
                    json=service_data
                ) as resp:
                    if resp.status not in (200, 201, 409):  # 409 = already exists
                        logger.error(f"Failed to create Kong service: {resp.status}")
                        return False
                
                # Create routes
                for route in routes:
                    route_data = {
                        "service": {"name": service_name},
                        "paths": route.get("paths", []),
                        "methods": route.get("methods", ["GET", "POST", "PUT", "DELETE"]),
                    }
                    
                    async with session.post(
                        f"{kong_admin_url}/services/{service_name}/routes",
                        json=route_data
                    ) as route_resp:
                        if route_resp.status not in (200, 201, 409):
                            logger.warning(f"Failed to create route: {route_resp.status}")
                
                logger.info(f"✅ Registered service {service_name} with Kong")
                return True
                
        except ImportError:
            logger.warning("aiohttp not installed for Kong integration")
            return False
        except Exception as e:
            logger.error(f"Failed to register with Kong: {e}")
            return False
    
    async def _register_aws(
        self,
        service_name: str,
        service_url: str,
        routes: List[Dict[str, Any]]
    ) -> bool:
        """Register service with AWS API Gateway (placeholder)"""
        # AWS API Gateway registration typically done via CloudFormation/Terraform
        # This is a placeholder for programmatic registration
        logger.info(f"AWS API Gateway registration for {service_name} - use CloudFormation/Terraform")
        return True
    
    def get_gateway_headers(self) -> Dict[str, str]:
        """Get headers to add for API Gateway integration"""
        headers = {}
        
        if self.config.api_key:
            headers["X-API-Key"] = self.config.api_key
        
        if self.gateway_type == GatewayType.AWS:
            # AWS API Gateway specific headers
            headers["X-Amzn-Trace-Id"] = os.getenv("_X_AMZN_TRACE_ID", "")
        
        return headers
    
    def should_handle_rate_limit(self) -> bool:
        """Check if rate limiting should be handled by gateway"""
        return self.config.rate_limit_enabled and self.gateway_type != GatewayType.NONE


def get_api_gateway_client() -> Optional[APIGatewayClient]:
    """Get API Gateway client from environment"""
    gateway_type = os.getenv("API_GATEWAY_TYPE", "none").lower()
    
    if gateway_type == "none":
        return None
    
    config = GatewayConfig(
        gateway_type=GatewayType(gateway_type),
        base_url=os.getenv("API_GATEWAY_URL"),
        api_key=os.getenv("API_GATEWAY_KEY"),
        rate_limit_enabled=os.getenv("GATEWAY_RATE_LIMIT_ENABLED", "true").lower() == "true",
        rate_limit_per_minute=int(os.getenv("GATEWAY_RATE_LIMIT_PER_MINUTE", "100")),
    )
    
    return APIGatewayClient(config)


# Middleware for API Gateway integration
class APIGatewayMiddleware:
    """Middleware for API Gateway request/response transformation"""
    
    def __init__(self, gateway_client: Optional[APIGatewayClient] = None):
        self.gateway_client = gateway_client or get_api_gateway_client()
    
    async def process_request(self, request) -> Dict[str, Any]:
        """Process incoming request from API Gateway"""
        if not self.gateway_client:
            return {}
        
        # Extract gateway-specific headers
        gateway_headers = {}
        
        # AWS API Gateway
        if "x-amzn-trace-id" in request.headers:
            gateway_headers["trace_id"] = request.headers["x-amzn-trace-id"]
        
        # Kong
        if "x-kong-request-id" in request.headers:
            gateway_headers["request_id"] = request.headers["x-kong-request-id"]
        
        return gateway_headers
    
    async def process_response(self, response, gateway_headers: Dict[str, Any]):
        """Process response for API Gateway"""
        if not self.gateway_client:
            return
        
        # Add gateway-specific headers
        gateway_headers_to_add = self.gateway_client.get_gateway_headers()
        for key, value in gateway_headers_to_add.items():
            response.headers[key] = value















