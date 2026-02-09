"""
Mesh Config
===========

Service mesh configuration.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List


@dataclass
class MeshConfig:
    """Service mesh configuration."""
    service_name: str
    mesh_type: str = "istio"  # istio, linkerd, consul
    enable_tracing: bool = True
    enable_metrics: bool = True
    enable_circuit_breaker: bool = True
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 60.0
    load_balancer: str = "round_robin"  # round_robin, least_conn, random
    health_check_interval: float = 30.0
    service_endpoints: Dict[str, str] = None
    
    def __post_init__(self):
        if self.service_endpoints is None:
            self.service_endpoints = {}















