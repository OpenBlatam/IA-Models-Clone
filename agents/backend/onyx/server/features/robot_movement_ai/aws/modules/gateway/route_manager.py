"""
Route Manager
=============

Manage API Gateway routes.
"""

import logging
from typing import Dict, Any, List, Optional
from aws.modules.gateway.gateway_client import GatewayClient, GatewayType

logger = logging.getLogger(__name__)


class RouteManager:
    """Manage API Gateway routes."""
    
    def __init__(self, gateway_client: GatewayClient):
        self.gateway_client = gateway_client
        self._routes: Dict[str, Dict[str, Any]] = {}
    
    def register_service_route(
        self,
        service_name: str,
        path: str,
        target_url: str,
        methods: List[str] = None,
        rate_limit: Optional[int] = None,
        auth_required: bool = False
    ):
        """Register service route."""
        route_key = f"{service_name}:{path}"
        
        route_config = {
            "service_name": service_name,
            "path": path,
            "target_url": target_url,
            "methods": methods or ["GET", "POST", "PUT", "DELETE"],
            "rate_limit": rate_limit,
            "auth_required": auth_required
        }
        
        self._routes[route_key] = route_config
        
        # Register in gateway
        asyncio.create_task(
            self.gateway_client.register_route(
                path=path,
                target=target_url,
                methods=route_config["methods"],
                rate_limit=rate_limit,
                auth_required=auth_required
            )
        )
        
        logger.info(f"Registered route: {route_key}")
    
    def get_routes_for_service(self, service_name: str) -> List[Dict[str, Any]]:
        """Get routes for service."""
        return [
            route for route in self._routes.values()
            if route["service_name"] == service_name
        ]
    
    def update_rate_limit(self, service_name: str, path: str, limit: int):
        """Update rate limit for route."""
        route_key = f"{service_name}:{path}"
        if route_key in self._routes:
            self._routes[route_key]["rate_limit"] = limit
            asyncio.create_task(
                self.gateway_client.update_rate_limit(path, limit)
            )


# Import asyncio
import asyncio















