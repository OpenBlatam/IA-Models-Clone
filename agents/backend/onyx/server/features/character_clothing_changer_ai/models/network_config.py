"""
Network Configuration for Flux2 Clothing Changer
================================================

Advanced network configuration management.
"""

import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class NetworkProtocol(Enum):
    """Network protocols."""
    HTTP = "http"
    HTTPS = "https"
    TCP = "tcp"
    UDP = "udp"
    WEBSOCKET = "websocket"


@dataclass
class NetworkEndpoint:
    """Network endpoint configuration."""
    endpoint_id: str
    host: str
    port: int
    protocol: NetworkProtocol
    path: Optional[str] = None
    timeout: float = 30.0
    retry_count: int = 3
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def url(self) -> str:
        """Get full URL."""
        base = f"{self.protocol.value}://{self.host}:{self.port}"
        if self.path:
            return f"{base}{self.path}"
        return base


class NetworkConfig:
    """Advanced network configuration system."""
    
    def __init__(self):
        """Initialize network config."""
        self.endpoints: Dict[str, NetworkEndpoint] = {}
        self.connection_pools: Dict[str, Dict[str, Any]] = {}
    
    def register_endpoint(
        self,
        endpoint_id: str,
        host: str,
        port: int,
        protocol: NetworkProtocol = NetworkProtocol.HTTP,
        path: Optional[str] = None,
        timeout: float = 30.0,
        retry_count: int = 3,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> NetworkEndpoint:
        """
        Register network endpoint.
        
        Args:
            endpoint_id: Endpoint identifier
            host: Host address
            port: Port number
            protocol: Network protocol
            path: Optional path
            timeout: Connection timeout
            retry_count: Retry count
            metadata: Optional metadata
            
        Returns:
            Created endpoint
        """
        endpoint = NetworkEndpoint(
            endpoint_id=endpoint_id,
            host=host,
            port=port,
            protocol=protocol,
            path=path,
            timeout=timeout,
            retry_count=retry_count,
            metadata=metadata or {},
        )
        
        self.endpoints[endpoint_id] = endpoint
        logger.info(f"Registered endpoint: {endpoint_id} ({endpoint.url})")
        return endpoint
    
    def get_endpoint(self, endpoint_id: str) -> Optional[NetworkEndpoint]:
        """Get endpoint by ID."""
        return self.endpoints.get(endpoint_id)
    
    def configure_connection_pool(
        self,
        endpoint_id: str,
        max_connections: int = 10,
        max_keepalive: float = 30.0,
    ) -> None:
        """
        Configure connection pool for endpoint.
        
        Args:
            endpoint_id: Endpoint identifier
            max_connections: Maximum connections
            max_keepalive: Maximum keepalive time
        """
        self.connection_pools[endpoint_id] = {
            "max_connections": max_connections,
            "max_keepalive": max_keepalive,
        }
        logger.info(f"Configured connection pool for {endpoint_id}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get network config statistics."""
        return {
            "total_endpoints": len(self.endpoints),
            "endpoints_by_protocol": {
                protocol.value: len([
                    e for e in self.endpoints.values()
                    if e.protocol == protocol
                ])
                for protocol in NetworkProtocol
            },
            "connection_pools": len(self.connection_pools),
        }


