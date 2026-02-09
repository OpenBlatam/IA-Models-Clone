"""
Network Optimizer
Advanced network optimizations for maximum throughput
"""

import logging
import asyncio
from typing import Dict, Optional, List, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class CompressionLevel(Enum):
    """Compression levels"""
    NONE = "none"
    FAST = "fast"
    BALANCED = "balanced"
    MAXIMUM = "maximum"


@dataclass
class NetworkConfig:
    """Network configuration"""
    enable_compression: bool = True
    compression_level: CompressionLevel = CompressionLevel.BALANCED
    enable_http2: bool = True
    enable_keep_alive: bool = True
    keep_alive_timeout: int = 5
    max_connections: int = 100
    tcp_nodelay: bool = True
    tcp_keepalive: bool = True


class NetworkOptimizer:
    """
    Network optimizer
    
    Features:
    - Adaptive compression
    - Connection reuse
    - HTTP/2 optimization
    - TCP optimization
    - Bandwidth management
    - Latency optimization
    """
    
    def __init__(self, config: Optional[NetworkConfig] = None):
        self.config = config or NetworkConfig()
        self._connection_stats: Dict[str, Any] = {}
        self._compression_stats: Dict[str, int] = {
            "compressed": 0,
            "uncompressed": 0,
            "bytes_saved": 0
        }
        
        logger.info("✅ Network optimizer initialized")
    
    def get_compression_headers(
        self,
        content_type: str,
        content_length: int
    ) -> Dict[str, str]:
        """
        Get compression headers
        
        Args:
            content_type: Content type
            content_length: Content length
            
        Returns:
            Compression headers
        """
        headers = {}
        
        if not self.config.enable_compression:
            return headers
        
        # Determine if should compress
        compressible_types = [
            "application/json",
            "text/html",
            "text/css",
            "text/javascript",
            "application/javascript",
            "text/plain"
        ]
        
        should_compress = (
            any(ct in content_type for ct in compressible_types) and
            content_length > 1024  # Only compress if > 1KB
        )
        
        if should_compress:
            # Compression level hint
            if self.config.compression_level == CompressionLevel.FAST:
                headers["X-Compression-Level"] = "fast"
            elif self.config.compression_level == CompressionLevel.MAXIMUM:
                headers["X-Compression-Level"] = "maximum"
            else:
                headers["X-Compression-Level"] = "balanced"
        
        return headers
    
    def get_connection_headers(self) -> Dict[str, str]:
        """Get connection optimization headers"""
        headers = {}
        
        if self.config.enable_keep_alive:
            headers["Connection"] = "keep-alive"
            headers["Keep-Alive"] = f"timeout={self.config.keep_alive_timeout}, max=1000"
        
        if self.config.enable_http2:
            headers["Upgrade"] = "h2"
            headers["HTTP2-Settings"] = "AAMAAABkAARAAAAAAAIAAAAA"
        
        return headers
    
    def get_tcp_optimization_hints(self) -> Dict[str, Any]:
        """Get TCP optimization hints"""
        return {
            "tcp_nodelay": self.config.tcp_nodelay,
            "tcp_keepalive": self.config.tcp_keepalive,
            "max_connections": self.config.max_connections
        }
    
    def record_compression(
        self,
        original_size: int,
        compressed_size: int
    ):
        """Record compression statistics"""
        self._compression_stats["compressed"] += 1
        bytes_saved = original_size - compressed_size
        self._compression_stats["bytes_saved"] += bytes_saved
    
    def record_no_compression(self):
        """Record uncompressed response"""
        self._compression_stats["uncompressed"] += 1
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """Get compression statistics"""
        total = self._compression_stats["compressed"] + self._compression_stats["uncompressed"]
        compression_rate = (
            self._compression_stats["compressed"] / total * 100
            if total > 0 else 0
        )
        
        return {
            **self._compression_stats,
            "compression_rate": compression_rate,
            "total_responses": total
        }
    
    def optimize_for_latency(self) -> Dict[str, str]:
        """Get headers optimized for low latency"""
        return {
            "X-Latency-Optimized": "true",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Keep-Alive": "timeout=5, max=1000"
        }
    
    def optimize_for_throughput(self) -> Dict[str, str]:
        """Get headers optimized for high throughput"""
        return {
            "X-Throughput-Optimized": "true",
            "Connection": "keep-alive",
            "Keep-Alive": "timeout=30, max=10000",
            "Transfer-Encoding": "chunked"
        }
    
    def get_network_stats(self) -> Dict[str, Any]:
        """Get network statistics"""
        return {
            "compression": self.get_compression_stats(),
            "config": {
                "compression_enabled": self.config.enable_compression,
                "compression_level": self.config.compression_level.value,
                "http2_enabled": self.config.enable_http2,
                "keep_alive_enabled": self.config.enable_keep_alive
            }
        }


# Global optimizer instance
_network_optimizer: Optional[NetworkOptimizer] = None


def get_network_optimizer() -> NetworkOptimizer:
    """Get global network optimizer instance"""
    global _network_optimizer
    if _network_optimizer is None:
        _network_optimizer = NetworkOptimizer()
    return _network_optimizer















