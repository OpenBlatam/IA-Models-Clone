"""
Monitoring models for Ultimate Enhanced Supreme Production system
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum

class HealthStatus(Enum):
    """Health status enumeration."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

@dataclass
class PerformanceMetrics:
    """Performance metrics model."""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    gpu_usage: float = 0.0
    gpu_memory_usage: float = 0.0
    disk_usage: float = 0.0
    network_usage: float = 0.0
    response_time: float = 0.0
    throughput: float = 0.0
    error_rate: float = 0.0
    availability: float = 0.0
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

@dataclass
class SystemMetrics:
    """System metrics model."""
    supreme_optimization_level: str
    ultra_fast_level: str
    refactored_ultimate_hybrid_level: str
    cuda_kernel_level: str
    gpu_utilization_level: str
    memory_optimization_level: str
    reward_function_level: str
    truthgpt_adapter_level: str
    microservices_level: str
    max_concurrent_generations: int
    max_documents_per_query: int
    max_continuous_documents: int
    ultimate_enhanced_supreme_ready: bool
    ultra_fast_ready: bool
    refactored_ultimate_hybrid_ready: bool
    cuda_kernel_ready: bool
    gpu_utils_ready: bool
    memory_utils_ready: bool
    reward_function_ready: bool
    truthgpt_adapter_ready: bool
    microservices_ready: bool
    ultimate_ready: bool
    ultra_advanced_ready: bool
    advanced_ready: bool
    performance_metrics: PerformanceMetrics
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'supreme_optimization_level': self.supreme_optimization_level,
            'ultra_fast_level': self.ultra_fast_level,
            'refactored_ultimate_hybrid_level': self.refactored_ultimate_hybrid_level,
            'cuda_kernel_level': self.cuda_kernel_level,
            'gpu_utilization_level': self.gpu_utilization_level,
            'memory_optimization_level': self.memory_optimization_level,
            'reward_function_level': self.reward_function_level,
            'truthgpt_adapter_level': self.truthgpt_adapter_level,
            'microservices_level': self.microservices_level,
            'max_concurrent_generations': self.max_concurrent_generations,
            'max_documents_per_query': self.max_documents_per_query,
            'max_continuous_documents': self.max_continuous_documents,
            'ultimate_enhanced_supreme_ready': self.ultimate_enhanced_supreme_ready,
            'ultra_fast_ready': self.ultra_fast_ready,
            'refactored_ultimate_hybrid_ready': self.refactored_ultimate_hybrid_ready,
            'cuda_kernel_ready': self.cuda_kernel_ready,
            'gpu_utils_ready': self.gpu_utils_ready,
            'memory_utils_ready': self.memory_utils_ready,
            'reward_function_ready': self.reward_function_ready,
            'truthgpt_adapter_ready': self.truthgpt_adapter_ready,
            'microservices_ready': self.microservices_ready,
            'ultimate_ready': self.ultimate_ready,
            'ultra_advanced_ready': self.ultra_advanced_ready,
            'advanced_ready': self.advanced_ready,
            'performance_metrics': {
                'cpu_usage': self.performance_metrics.cpu_usage,
                'memory_usage': self.performance_metrics.memory_usage,
                'gpu_usage': self.performance_metrics.gpu_usage,
                'gpu_memory_usage': self.performance_metrics.gpu_memory_usage,
                'disk_usage': self.performance_metrics.disk_usage,
                'network_usage': self.performance_metrics.network_usage,
                'response_time': self.performance_metrics.response_time,
                'throughput': self.performance_metrics.throughput,
                'error_rate': self.performance_metrics.error_rate,
                'availability': self.performance_metrics.availability,
                'timestamp': self.performance_metrics.timestamp.isoformat() if self.performance_metrics.timestamp else None
            },
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }









