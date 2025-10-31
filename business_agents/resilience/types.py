"""
Resilience Types and Definitions
================================

Type definitions for resilience patterns and circuit breakers.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable, Union
from datetime import datetime, timedelta
import uuid

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit is open, failing fast
    HALF_OPEN = "half_open"  # Testing if service is back

class FailureType(Enum):
    """Failure types for circuit breaker."""
    TIMEOUT = "timeout"
    CONNECTION_ERROR = "connection_error"
    HTTP_ERROR = "http_error"
    BUSINESS_ERROR = "business_error"
    UNKNOWN = "unknown"

@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5
    success_threshold: int = 3
    timeout_duration: int = 60  # seconds
    half_open_max_calls: int = 3
    failure_rate_threshold: float = 0.5  # 50%
    slow_call_threshold: int = 60  # seconds
    slow_call_rate_threshold: float = 0.5  # 50%
    wait_duration_in_open_state: int = 60  # seconds
    permitted_calls_in_half_open_state: int = 3
    sliding_window_size: int = 100
    minimum_number_of_calls: int = 10
    automatic_transition_from_open_to_half_open_enabled: bool = True

@dataclass
class CircuitBreakerMetrics:
    """Circuit breaker metrics."""
    state: CircuitState = CircuitState.CLOSED
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    slow_calls: int = 0
    failure_rate: float = 0.0
    slow_call_rate: float = 0.0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    state_transitions: List[Dict[str, Any]] = field(default_factory=list)
    
    def record_call(self, success: bool, duration: float, slow_threshold: int):
        """Record a call."""
        self.total_calls += 1
        if success:
            self.successful_calls += 1
            self.last_success_time = datetime.now()
        else:
            self.failed_calls += 1
            self.last_failure_time = datetime.now()
        
        if duration > slow_threshold:
            self.slow_calls += 1
        
        # Calculate rates
        if self.total_calls > 0:
            self.failure_rate = self.failed_calls / self.total_calls
            self.slow_call_rate = self.slow_calls / self.total_calls
    
    def record_state_transition(self, from_state: CircuitState, to_state: CircuitState):
        """Record state transition."""
        transition = {
            "from": from_state.value,
            "to": to_state.value,
            "timestamp": datetime.now().isoformat()
        }
        self.state_transitions.append(transition)
        self.state = to_state

@dataclass
class RetryConfig:
    """Retry configuration."""
    max_attempts: int = 3
    base_delay: float = 1.0  # seconds
    max_delay: float = 60.0  # seconds
    multiplier: float = 2.0
    jitter: bool = True
    retry_on_exceptions: List[type] = field(default_factory=list)
    retry_on_conditions: List[Callable] = field(default_factory=list)

@dataclass
class TimeoutConfig:
    """Timeout configuration."""
    timeout_seconds: float = 30.0
    cancel_on_timeout: bool = True
    timeout_exception: type = TimeoutError

@dataclass
class BulkheadConfig:
    """Bulkhead configuration."""
    max_concurrent_calls: int = 25
    max_wait_duration: int = 0  # milliseconds
    max_thread_pool_size: int = 10
    core_thread_pool_size: int = 5
    keep_alive_duration: int = 20  # seconds
    queue_capacity: int = 100

@dataclass
class RateLimitConfig:
    """Rate limit configuration."""
    requests_per_second: float = 10.0
    burst_size: int = 20
    window_size: int = 60  # seconds
    algorithm: str = "token_bucket"  # token_bucket, sliding_window, fixed_window

@dataclass
class ResilienceConfig:
    """Overall resilience configuration."""
    circuit_breaker: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    retry: RetryConfig = field(default_factory=RetryConfig)
    timeout: TimeoutConfig = field(default_factory=TimeoutConfig)
    bulkhead: BulkheadConfig = field(default_factory=BulkheadConfig)
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    enabled: bool = True
    monitoring_enabled: bool = True

@dataclass
class HealthCheck:
    """Health check definition."""
    name: str
    check_function: Callable
    interval: int = 30  # seconds
    timeout: int = 5  # seconds
    retries: int = 3
    enabled: bool = True
    last_check: Optional[datetime] = None
    last_result: Optional[bool] = None
    consecutive_failures: int = 0

@dataclass
class ServiceEndpoint:
    """Service endpoint definition."""
    name: str
    url: str
    method: str = "GET"
    headers: Dict[str, str] = field(default_factory=dict)
    timeout: int = 30
    retries: int = 3
    circuit_breaker_enabled: bool = True
    health_check_enabled: bool = True
    weight: int = 1
    tags: List[str] = field(default_factory=list)

@dataclass
class LoadBalancerConfig:
    """Load balancer configuration."""
    strategy: str = "round_robin"  # round_robin, least_connections, weighted, random
    health_check_interval: int = 30
    unhealthy_threshold: int = 3
    healthy_threshold: int = 2
    timeout: int = 5
    retries: int = 3

@dataclass
class ServiceDiscovery:
    """Service discovery configuration."""
    enabled: bool = True
    provider: str = "consul"  # consul, etcd, eureka, kubernetes
    endpoint: str = "http://localhost:8500"
    service_name: str = "business-agents"
    tags: List[str] = field(default_factory=list)
    health_check: HealthCheck = field(default_factory=lambda: HealthCheck("default", lambda: True))
    refresh_interval: int = 30

@dataclass
class FailureEvent:
    """Failure event definition."""
    id: str
    service_name: str
    failure_type: FailureType
    error_message: str
    timestamp: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)
    severity: str = "medium"  # low, medium, high, critical
    resolved: bool = False

@dataclass
class ResilienceMetrics:
    """Resilience metrics."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    timeout_requests: int = 0
    circuit_breaker_opens: int = 0
    circuit_breaker_closes: int = 0
    retry_attempts: int = 0
    bulkhead_rejections: int = 0
    rate_limit_rejections: int = 0
    average_response_time: float = 0.0
    p95_response_time: float = 0.0
    p99_response_time: float = 0.0
    error_rate: float = 0.0
    availability: float = 100.0
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class AlertRule:
    """Alert rule definition."""
    name: str
    condition: str
    threshold: float
    duration: int = 300  # seconds
    severity: str = "warning"  # info, warning, error, critical
    enabled: bool = True
    actions: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

@dataclass
class Alert:
    """Alert definition."""
    id: str
    rule_name: str
    message: str
    severity: str
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    context: Dict[str, Any] = field(default_factory=dict)
    actions_taken: List[str] = field(default_factory=list)

@dataclass
class ChaosExperiment:
    """Chaos engineering experiment."""
    name: str
    description: str
    target_service: str
    experiment_type: str  # latency, failure, resource_exhaustion
    parameters: Dict[str, Any] = field(default_factory=dict)
    duration: int = 300  # seconds
    enabled: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    last_run: Optional[datetime] = None
    results: Dict[str, Any] = field(default_factory=dict)
