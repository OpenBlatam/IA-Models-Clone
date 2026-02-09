"""
Network Optimizer for Flux2 Clothing Changer
=============================================

Network optimization and bandwidth management.
"""

import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from collections import deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class NetworkMetrics:
    """Network metrics snapshot."""
    timestamp: float
    bandwidth_mbps: float
    latency_ms: float
    packet_loss: float
    throughput_mbps: float


class NetworkOptimizer:
    """Network optimization system."""
    
    def __init__(
        self,
        target_bandwidth_mbps: Optional[float] = None,
        enable_compression: bool = True,
        enable_caching: bool = True,
    ):
        """
        Initialize network optimizer.
        
        Args:
            target_bandwidth_mbps: Target bandwidth in Mbps
            enable_compression: Enable compression
            enable_caching: Enable network caching
        """
        self.target_bandwidth_mbps = target_bandwidth_mbps
        self.enable_compression = enable_compression
        self.enable_caching = enable_caching
        
        self.metrics_history: deque = deque(maxlen=1000)
        self.optimization_count = 0
    
    def record_metrics(
        self,
        bandwidth_mbps: float,
        latency_ms: float,
        packet_loss: float = 0.0,
        throughput_mbps: float = 0.0,
    ) -> None:
        """
        Record network metrics.
        
        Args:
            bandwidth_mbps: Bandwidth in Mbps
            latency_ms: Latency in milliseconds
            packet_loss: Packet loss percentage
            throughput_mbps: Throughput in Mbps
        """
        metrics = NetworkMetrics(
            timestamp=time.time(),
            bandwidth_mbps=bandwidth_mbps,
            latency_ms=latency_ms,
            packet_loss=packet_loss,
            throughput_mbps=throughput_mbps,
        )
        
        self.metrics_history.append(metrics)
    
    def optimize_transfer(
        self,
        data_size_mb: float,
        priority: str = "normal",
    ) -> Dict[str, Any]:
        """
        Optimize data transfer.
        
        Args:
            data_size_mb: Data size in MB
            priority: Transfer priority
            
        Returns:
            Optimization recommendations
        """
        if not self.metrics_history:
            return {
                "compression": self.enable_compression,
                "chunking": False,
                "estimated_time": 0.0,
            }
        
        recent_metrics = list(self.metrics_history)[-10:]
        avg_bandwidth = sum(m.bandwidth_mbps for m in recent_metrics) / len(recent_metrics)
        avg_latency = sum(m.latency_ms for m in recent_metrics) / len(recent_metrics)
        
        # Calculate estimated transfer time
        if avg_bandwidth > 0:
            estimated_time = (data_size_mb * 8) / avg_bandwidth  # Convert MB to Mb
        else:
            estimated_time = 0.0
        
        # Optimization recommendations
        recommendations = {
            "compression": self.enable_compression,
            "chunking": data_size_mb > 100,  # Chunk if > 100MB
            "estimated_time": estimated_time,
            "priority": priority,
        }
        
        # Adjust based on network conditions
        if avg_latency > 100:  # High latency
            recommendations["use_cdn"] = True
            recommendations["prefetch"] = True
        else:
            recommendations["use_cdn"] = False
            recommendations["prefetch"] = False
        
        if avg_bandwidth < 10:  # Low bandwidth
            recommendations["aggressive_compression"] = True
            recommendations["reduce_quality"] = True
        else:
            recommendations["aggressive_compression"] = False
            recommendations["reduce_quality"] = False
        
        return recommendations
    
    def get_network_statistics(
        self,
        time_range: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Get network statistics.
        
        Args:
            time_range: Time range in seconds
            
        Returns:
            Network statistics
        """
        if not self.metrics_history:
            return {}
        
        cutoff_time = time.time() - time_range if time_range else 0
        
        relevant_metrics = [
            m for m in self.metrics_history
            if m.timestamp >= cutoff_time
        ]
        
        if not relevant_metrics:
            return {}
        
        return {
            "avg_bandwidth_mbps": sum(m.bandwidth_mbps for m in relevant_metrics) / len(relevant_metrics),
            "avg_latency_ms": sum(m.latency_ms for m in relevant_metrics) / len(relevant_metrics),
            "avg_packet_loss": sum(m.packet_loss for m in relevant_metrics) / len(relevant_metrics),
            "avg_throughput_mbps": sum(m.throughput_mbps for m in relevant_metrics) / len(relevant_metrics),
            "min_bandwidth_mbps": min(m.bandwidth_mbps for m in relevant_metrics),
            "max_bandwidth_mbps": max(m.bandwidth_mbps for m in relevant_metrics),
        }
    
    def should_optimize(self) -> bool:
        """Check if network optimization is needed."""
        if not self.metrics_history:
            return False
        
        recent = list(self.metrics_history)[-5:]
        avg_bandwidth = sum(m.bandwidth_mbps for m in recent) / len(recent)
        avg_latency = sum(m.latency_ms for m in recent) / len(recent)
        
        # Optimize if bandwidth is low or latency is high
        if self.target_bandwidth_mbps:
            return avg_bandwidth < self.target_bandwidth_mbps * 0.8
        
        return avg_latency > 100 or avg_bandwidth < 10
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get optimizer statistics."""
        return {
            "optimization_count": self.optimization_count,
            "metrics_recorded": len(self.metrics_history),
            "compression_enabled": self.enable_compression,
            "caching_enabled": self.enable_caching,
        }


