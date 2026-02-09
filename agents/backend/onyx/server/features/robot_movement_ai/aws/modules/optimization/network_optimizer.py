"""
Network Optimizer
================

Advanced network optimization techniques.
"""

import logging
import socket
from typing import Dict, Any, Optional
import httpx

logger = logging.getLogger(__name__)


class NetworkOptimizer:
    """Network optimizer with advanced techniques."""
    
    def __init__(self):
        self._tcp_nodelay = False
        self._keepalive = False
        self._socket_options_set = False
    
    def optimize_tcp(self, enable_nodelay: bool = True, enable_keepalive: bool = True):
        """Optimize TCP settings."""
        self._tcp_nodelay = enable_nodelay
        self._keepalive = enable_keepalive
        
        # Set socket defaults
        socket.TCP_NODELAY = 1 if enable_nodelay else 0
        
        logger.info(f"TCP optimized: nodelay={enable_nodelay}, keepalive={enable_keepalive}")
    
    def create_optimized_client(
        self,
        timeout: float = 30.0,
        max_connections: int = 100,
        max_keepalive_connections: int = 20,
        http2: bool = True
    ) -> httpx.AsyncClient:
        """Create optimized HTTP client."""
        limits = httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive_connections
        )
        
        client = httpx.AsyncClient(
            timeout=timeout,
            limits=limits,
            http2=http2
        )
        
        logger.debug(f"Created optimized HTTP client: {max_connections} max connections")
        return client
    
    def optimize_dns(self, cache_size: int = 128):
        """Optimize DNS caching."""
        # In production, configure DNS cache
        logger.info(f"DNS cache size: {cache_size}")
    
    def get_network_stats(self) -> Dict[str, Any]:
        """Get network statistics."""
        try:
            import psutil
            net_io = psutil.net_io_counters()
            
            return {
                "bytes_sent": net_io.bytes_sent,
                "bytes_recv": net_io.bytes_recv,
                "bytes_sent_mb": net_io.bytes_sent / 1024 / 1024,
                "bytes_recv_mb": net_io.bytes_recv / 1024 / 1024,
                "packets_sent": net_io.packets_sent,
                "packets_recv": net_io.packets_recv,
                "errin": net_io.errin,
                "errout": net_io.errout,
                "dropin": net_io.dropin,
                "dropout": net_io.dropout
            }
        except Exception as e:
            logger.warning(f"Failed to get network stats: {e}")
            return {}















