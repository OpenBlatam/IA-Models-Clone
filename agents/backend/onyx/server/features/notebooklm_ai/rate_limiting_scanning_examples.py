from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import asyncio
import logging
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, Set, Tuple
from enum import Enum
import hashlib
import json
import threading
from contextlib import contextmanager
from collections import deque, defaultdict
import statistics
    import redis
    import psutil
from typing import Any, List, Dict, Optional
"""
Rate-Limiting and Back-Off for Network Scanning Examples
=======================================================

This module provides comprehensive rate-limiting and back-off mechanisms for
network scanning operations to avoid detection and prevent abuse.

Features:
- Adaptive rate limiting with multiple strategies
- Intelligent back-off algorithms (exponential, jitter, adaptive)
- Detection avoidance mechanisms
- Resource usage monitoring and throttling
- Distributed rate limiting coordination
- Scan pattern randomization
- Traffic shaping and timing optimization
- Abuse prevention and compliance checking
- Performance monitoring and optimization
- Stealth scanning techniques

Author: AI Assistant
License: MIT
"""


try:
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RateLimitStrategy(Enum):
    """Rate limiting strategies."""
    FIXED = "fixed"
    ADAPTIVE = "adaptive"
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"
    SLIDING_WINDOW = "sliding_window"
    DISTRIBUTED = "distributed"


class BackOffStrategy(Enum):
    """Back-off strategies."""
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    FIBONACCI = "fibonacci"
    JITTER = "jitter"
    ADAPTIVE = "adaptive"


class ScanType(Enum):
    """Types of network scans."""
    PING = "ping"
    TCP_CONNECT = "tcp_connect"
    TCP_SYN = "tcp_syn"
    UDP = "udp"
    SERVICE_DETECTION = "service_detection"
    VULNERABILITY_SCAN = "vulnerability_scan"
    PORT_SCAN = "port_scan"
    OS_DETECTION = "os_detection"


class DetectionLevel(Enum):
    """Detection risk levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""
    strategy: RateLimitStrategy = RateLimitStrategy.ADAPTIVE
    max_requests_per_second: float = 10.0
    max_requests_per_minute: float = 300.0
    max_requests_per_hour: float = 10000.0
    burst_size: int = 5
    window_size: int = 60
    adaptive_threshold: float = 0.8
    jitter_factor: float = 0.1
    distributed_key_prefix: str = "rate_limit"


@dataclass
class BackOffConfig:
    """Back-off configuration."""
    strategy: BackOffStrategy = BackOffStrategy.EXPONENTIAL
    initial_delay: float = 1.0
    max_delay: float = 300.0
    multiplier: float = 2.0
    jitter_factor: float = 0.1
    max_attempts: int = 10
    reset_threshold: int = 5


@dataclass
class ScanRequest:
    """Network scan request."""
    target: str
    scan_type: ScanType
    priority: int = 1
    timestamp: float = field(default_factory=time.time)
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RateLimitResult:
    """Result of rate limiting check."""
    allowed: bool
    delay_required: float = 0.0
    retry_after: float = 0.0
    current_rate: float = 0.0
    limit_exceeded: bool = False
    back_off_required: bool = False
    detection_risk: DetectionLevel = DetectionLevel.LOW


@dataclass
class ScanMetrics:
    """Scan performance and detection metrics."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    blocked_requests: int = 0
    average_response_time: float = 0.0
    detection_events: int = 0
    rate_limit_hits: int = 0
    back_off_events: int = 0
    resource_usage: Dict[str, float] = field(default_factory=dict)


class RateLimitExceededError(Exception):
    """Custom exception for rate limit exceeded."""
    pass


class DetectionRiskError(Exception):
    """Custom exception for high detection risk."""
    pass


class ResourceExhaustedError(Exception):
    """Custom exception for resource exhaustion."""
    pass


class AdaptiveRateLimiter:
    """Adaptive rate limiter with intelligent throttling."""
    
    def __init__(self, config: RateLimitConfig):
        """Initialize adaptive rate limiter."""
        self.config = config
        self.request_history = deque(maxlen=1000)
        self.response_times = deque(maxlen=100)
        self.error_counts = defaultdict(int)
        self.success_counts = defaultdict(int)
        self.last_adjustment = time.time()
        self.current_rate = config.max_requests_per_second
        self._lock = threading.Lock()
        
        # Token bucket for burst handling
        self.tokens = config.burst_size
        self.last_token_refill = time.time()
        self.token_refill_rate = config.max_requests_per_second
        
        # Sliding window for rate tracking
        self.window_requests = deque()
        self.window_start = time.time()
    
    def _refill_tokens(self) -> Any:
        """Refill tokens based on time elapsed."""
        now = time.time()
        time_passed = now - self.last_token_refill
        tokens_to_add = time_passed * self.token_refill_rate
        
        self.tokens = min(self.config.burst_size, self.tokens + tokens_to_add)
        self.last_token_refill = now
    
    def _update_sliding_window(self) -> Any:
        """Update sliding window for rate calculation."""
        now = time.time()
        window_start = now - self.config.window_size
        
        # Remove old requests from window
        while self.window_requests and self.window_requests[0] < window_start:
            self.window_requests.popleft()
        
        self.window_start = window_start
    
    def _calculate_current_rate(self) -> float:
        """Calculate current request rate."""
        with self._lock:
            self._update_sliding_window()
            return len(self.window_requests) / self.config.window_size
    
    def _detect_anomalies(self) -> List[str]:
        """Detect anomalous patterns that might indicate detection."""
        anomalies = []
        
        # Check for unusual error patterns
        recent_errors = sum(self.error_counts.values())
        recent_successes = sum(self.success_counts.values())
        
        if recent_errors > 0 and recent_successes > 0:
            error_rate = recent_errors / (recent_errors + recent_successes)
            if error_rate > 0.5:  # More than 50% errors
                anomalies.append("high_error_rate")
        
        # Check for unusual response times
        if len(self.response_times) > 10:
            avg_response_time = statistics.mean(self.response_times)
            if avg_response_time > 5.0:  # More than 5 seconds average
                anomalies.append("slow_response_times")
        
        # Check for rate limit violations
        current_rate = self._calculate_current_rate()
        if current_rate > self.config.max_requests_per_second * 1.5:
            anomalies.append("rate_limit_violation")
        
        return anomalies
    
    def _adjust_rate(self, anomalies: List[str]):
        """Adjust rate based on detected anomalies."""
        if not anomalies:
            return
        
        adjustment_factor = 0.8  # Reduce rate by 20%
        
        for anomaly in anomalies:
            if anomaly == "high_error_rate":
                adjustment_factor *= 0.7
            elif anomaly == "slow_response_times":
                adjustment_factor *= 0.8
            elif anomaly == "rate_limit_violation":
                adjustment_factor *= 0.5
        
        with self._lock:
            self.current_rate = max(1.0, self.current_rate * adjustment_factor)
            self.last_adjustment = time.time()
            
            logger.warning(f"Rate adjusted to {self.current_rate:.2f} req/s due to anomalies: {anomalies}")
    
    def check_rate_limit(self, request: ScanRequest) -> RateLimitResult:
        """Check if request is allowed under current rate limits."""
        with self._lock:
            # Refill tokens
            self._refill_tokens()
            
            # Check token bucket
            if self.tokens < 1:
                return RateLimitResult(
                    allowed=False,
                    delay_required=1.0 / self.token_refill_rate,
                    retry_after=1.0 / self.token_refill_rate,
                    current_rate=self._calculate_current_rate(),
                    limit_exceeded=True
                )
            
            # Check sliding window rate
            current_rate = self._calculate_current_rate()
            if current_rate >= self.current_rate:
                return RateLimitResult(
                    allowed=False,
                    delay_required=1.0 / self.current_rate,
                    retry_after=1.0 / self.current_rate,
                    current_rate=current_rate,
                    limit_exceeded=True
                )
            
            # Check for anomalies and adjust rate
            anomalies = self._detect_anomalies()
            if anomalies:
                self._adjust_rate(anomalies)
                return RateLimitResult(
                    allowed=False,
                    delay_required=2.0,  # Force delay for anomaly
                    retry_after=2.0,
                    current_rate=current_rate,
                    back_off_required=True,
                    detection_risk=DetectionLevel.MEDIUM
                )
            
            # Allow request
            self.tokens -= 1
            self.window_requests.append(time.time())
            
            return RateLimitResult(
                allowed=True,
                current_rate=current_rate,
                detection_risk=DetectionLevel.LOW
            )
    
    def record_request_result(self, request: ScanRequest, success: bool, 
                            response_time: float = 0.0, error_type: str = None):
        """Record the result of a request for adaptive learning."""
        with self._lock:
            self.request_history.append({
                'timestamp': time.time(),
                'target': request.target,
                'scan_type': request.scan_type.value,
                'success': success,
                'response_time': response_time,
                'error_type': error_type
            })
            
            if success:
                self.success_counts[request.target] += 1
                if response_time > 0:
                    self.response_times.append(response_time)
            else:
                self.error_counts[request.target] += 1
                if error_type:
                    self.error_counts[f"{request.target}_{error_type}"] += 1


class IntelligentBackOff:
    """Intelligent back-off with multiple strategies."""
    
    def __init__(self, config: BackOffConfig):
        """Initialize intelligent back-off."""
        self.config = config
        self.attempt_counts = defaultdict(int)
        self.last_attempts = defaultdict(float)
        self.back_off_delays = defaultdict(float)
        self._lock = threading.Lock()
    
    def _calculate_exponential_delay(self, attempt: int) -> float:
        """Calculate exponential back-off delay."""
        delay = self.config.initial_delay * (self.config.multiplier ** (attempt - 1))
        return min(delay, self.config.max_delay)
    
    def _calculate_fibonacci_delay(self, attempt: int) -> float:
        """Calculate Fibonacci back-off delay."""
        if attempt <= 2:
            return self.config.initial_delay
        
        a, b = self.config.initial_delay, self.config.initial_delay
        for _ in range(3, attempt + 1):
            a, b = b, a + b
        
        return min(b, self.config.max_delay)
    
    def _add_jitter(self, delay: float) -> float:
        """Add jitter to delay to prevent thundering herd."""
        if self.config.jitter_factor <= 0:
            return delay
        
        jitter = delay * self.config.jitter_factor * random.uniform(-1, 1)
        return max(0.1, delay + jitter)
    
    def _calculate_adaptive_delay(self, target: str, error_type: str = None) -> float:
        """Calculate adaptive delay based on target and error history."""
        base_delay = self.back_off_delays[target]
        
        # Increase delay for repeated failures
        if error_type:
            error_key = f"{target}_{error_type}"
            error_count = self.attempt_counts[error_key]
            if error_count > 3:
                base_delay *= 1.5
        
        # Consider target-specific patterns
        target_attempts = self.attempt_counts[target]
        if target_attempts > 10:
            base_delay *= 2.0
        
        return min(base_delay, self.config.max_delay)
    
    def get_delay(self, target: str, error_type: str = None) -> float:
        """Get back-off delay for target."""
        with self._lock:
            attempt = self.attempt_counts[target] + 1
            
            if attempt > self.config.max_attempts:
                return self.config.max_delay
            
            # Calculate base delay based on strategy
            if self.config.strategy == BackOffStrategy.EXPONENTIAL:
                delay = self._calculate_exponential_delay(attempt)
            elif self.config.strategy == BackOffStrategy.LINEAR:
                delay = self.config.initial_delay * attempt
            elif self.config.strategy == BackOffStrategy.FIBONACCI:
                delay = self._calculate_fibonacci_delay(attempt)
            elif self.config.strategy == BackOffStrategy.ADAPTIVE:
                delay = self._calculate_adaptive_delay(target, error_type)
            else:
                delay = self._calculate_exponential_delay(attempt)
            
            # Add jitter
            delay = self._add_jitter(delay)
            
            # Update state
            self.attempt_counts[target] = attempt
            self.last_attempts[target] = time.time()
            self.back_off_delays[target] = delay
            
            return delay
    
    def record_success(self, target: str):
        """Record successful request to reset back-off."""
        with self._lock:
            # Reset back-off if we've had enough successes
            if self.attempt_counts[target] >= self.config.reset_threshold:
                self.attempt_counts[target] = 0
                self.back_off_delays[target] = self.config.initial_delay
    
    def should_skip_target(self, target: str) -> bool:
        """Check if target should be skipped due to excessive failures."""
        with self._lock:
            return self.attempt_counts[target] > self.config.max_attempts


class StealthScanner:
    """Stealth network scanner with rate limiting and back-off."""
    
    def __init__(self, rate_limit_config: RateLimitConfig, back_off_config: BackOffConfig):
        """Initialize stealth scanner."""
        self.rate_limiter = AdaptiveRateLimiter(rate_limit_config)
        self.back_off = IntelligentBackOff(back_off_config)
        self.metrics = ScanMetrics()
        self.scan_patterns = self._generate_scan_patterns()
        self.resource_monitor = ResourceMonitor()
        self._lock = threading.Lock()
    
    def _generate_scan_patterns(self) -> Dict[str, List[Dict[str, Any]]]:
        """Generate randomized scan patterns to avoid detection."""
        patterns = {
            'timing': [
                {'type': 'random', 'min_delay': 0.1, 'max_delay': 2.0},
                {'type': 'human', 'min_delay': 0.5, 'max_delay': 3.0},
                {'type': 'burst', 'min_delay': 0.05, 'max_delay': 0.5, 'burst_size': 3},
                {'type': 'slow', 'min_delay': 2.0, 'max_delay': 10.0}
            ],
            'ports': [
                {'type': 'sequential', 'start': 1, 'end': 1024},
                {'type': 'random', 'count': 100},
                {'type': 'common', 'ports': [21, 22, 23, 25, 53, 80, 110, 143, 443, 993, 995]},
                {'type': 'sparse', 'start': 1, 'end': 65535, 'step': 100}
            ],
            'targets': [
                {'type': 'sequential', 'start': 1, 'end': 254},
                {'type': 'random', 'count': 50},
                {'type': 'sparse', 'start': 1, 'end': 254, 'step': 5}
            ]
        }
        return patterns
    
    def _get_random_pattern(self, pattern_type: str) -> Dict[str, Any]:
        """Get random pattern for scan type."""
        patterns = self.scan_patterns.get(pattern_type, [])
        if not patterns:
            return {}
        
        return random.choice(patterns)
    
    def _calculate_stealth_delay(self, target: str, scan_type: ScanType) -> float:
        """Calculate stealth delay to avoid detection."""
        base_delay = 1.0
        
        # Adjust based on scan type
        if scan_type == ScanType.TCP_SYN:
            base_delay *= 0.5  # Faster for SYN scans
        elif scan_type == ScanType.VULNERABILITY_SCAN:
            base_delay *= 3.0  # Slower for vulnerability scans
        
        # Add randomization
        jitter = base_delay * random.uniform(0.5, 1.5)
        
        # Consider target-specific factors
        target_hash = int(hashlib.md5(target.encode()).hexdigest()[:8], 16)
        target_factor = (target_hash % 100) / 100.0  # 0.0 to 1.0
        
        return base_delay + jitter + target_factor
    
    def _check_resource_limits(self) -> bool:
        """Check if system resources are within limits."""
        return self.resource_monitor.check_limits()
    
    def _should_skip_target(self, target: str, scan_type: ScanType) -> bool:
        """Check if target should be skipped."""
        # Check back-off status
        if self.back_off.should_skip_target(target):
            return True
        
        # Check resource limits
        if not self._check_resource_limits():
            return True
        
        # Check for known blocked targets
        if self._is_target_blocked(target):
            return True
        
        return False
    
    def _is_target_blocked(self, target: str) -> bool:
        """Check if target is known to be blocked."""
        # This would typically check against a database of blocked targets
        # For demonstration, we'll use a simple hash-based check
        target_hash = int(hashlib.md5(target.encode()).hexdigest()[:8], 16)
        return target_hash % 100 < 5  # 5% chance of being "blocked"
    
    async def scan_target(self, target: str, scan_type: ScanType, 
                         ports: List[int] = None) -> Dict[str, Any]:
        """Scan a single target with rate limiting and back-off."""
        if self._should_skip_target(target, scan_type):
            return {
                'target': target,
                'status': 'skipped',
                'reason': 'target_skipped'
            }
        
        # Create scan request
        request = ScanRequest(
            target=target,
            scan_type=scan_type,
            metadata={'ports': ports or []}
        )
        
        # Check rate limit
        rate_result = self.rate_limiter.check_rate_limit(request)
        if not rate_result.allowed:
            if rate_result.back_off_required:
                delay = self.back_off.get_delay(target, 'rate_limit')
            else:
                delay = rate_result.delay_required
            
            await asyncio.sleep(delay)
            return {
                'target': target,
                'status': 'rate_limited',
                'delay': delay
            }
        
        # Calculate stealth delay
        stealth_delay = self._calculate_stealth_delay(target, scan_type)
        await asyncio.sleep(stealth_delay)
        
        # Perform scan
        start_time = time.time()
        try:
            result = await self._perform_scan(target, scan_type, ports)
            response_time = time.time() - start_time
            
            # Record success
            self.rate_limiter.record_request_result(request, True, response_time)
            self.back_off.record_success(target)
            
            with self._lock:
                self.metrics.successful_requests += 1
                self.metrics.total_requests += 1
            
            return {
                'target': target,
                'status': 'success',
                'result': result,
                'response_time': response_time
            }
        
        except Exception as e:
            response_time = time.time() - start_time
            error_type = type(e).__name__
            
            # Record failure
            self.rate_limiter.record_request_result(request, False, response_time, error_type)
            
            with self._lock:
                self.metrics.failed_requests += 1
                self.metrics.total_requests += 1
            
            return {
                'target': target,
                'status': 'failed',
                'error': str(e),
                'error_type': error_type,
                'response_time': response_time
            }
    
    async def _perform_scan(self, target: str, scan_type: ScanType, 
                           ports: List[int] = None) -> Dict[str, Any]:
        """Perform the actual network scan."""
        # This is a placeholder for actual scanning logic
        # In a real implementation, this would use scapy, nmap, or similar
        
        if scan_type == ScanType.PING:
            return await self._ping_scan(target)
        elif scan_type == ScanType.TCP_CONNECT:
            return await self._tcp_connect_scan(target, ports)
        elif scan_type == ScanType.PORT_SCAN:
            return await self._port_scan(target, ports)
        else:
            raise NotImplementedError(f"Scan type {scan_type} not implemented")
    
    async def _ping_scan(self, target: str) -> Dict[str, Any]:
        """Perform ping scan."""
        # Simulate ping scan
        await asyncio.sleep(random.uniform(0.1, 0.5))
        return {'alive': random.choice([True, False])}
    
    async def _tcp_connect_scan(self, target: str, ports: List[int]) -> Dict[str, Any]:
        """Perform TCP connect scan."""
        # Simulate TCP connect scan
        await asyncio.sleep(random.uniform(0.2, 1.0))
        open_ports = []
        
        for port in (ports or [80, 443, 22, 21]):
            if random.random() < 0.3:  # 30% chance of port being open
                open_ports.append(port)
        
        return {'open_ports': open_ports}
    
    async def _port_scan(self, target: str, ports: List[int]) -> Dict[str, Any]:
        """Perform comprehensive port scan."""
        # Simulate port scan
        await asyncio.sleep(random.uniform(1.0, 3.0))
        
        results = {}
        for port in (ports or range(1, 1025)):
            if random.random() < 0.1:  # 10% chance of port being open
                results[port] = {
                    'state': 'open',
                    'service': f'service_{port}',
                    'version': 'unknown'
                }
        
        return {'ports': results}
    
    async def scan_network(self, network: str, scan_type: ScanType, 
                          max_targets: int = 100) -> List[Dict[str, Any]]:
        """Scan a network with intelligent rate limiting and back-off."""
        results = []
        
        # Generate target list
        targets = self._generate_targets(network, max_targets)
        
        # Randomize target order
        random.shuffle(targets)
        
        for target in targets:
            result = await self.scan_target(target, scan_type)
            results.append(result)
            
            # Check if we should stop
            if len(results) >= max_targets:
                break
        
        return results
    
    def _generate_targets(self, network: str, max_targets: int) -> List[str]:
        """Generate list of targets from network."""
        # Simple network parsing (e.g., "192.168.1.0/24")
        if '/' in network:
            base, mask = network.split('/')
            base_parts = base.split('.')
            
            targets = []
            for i in range(1, min(255, max_targets + 1)):
                target = f"{base_parts[0]}.{base_parts[1]}.{base_parts[2]}.{i}"
                targets.append(target)
            
            return targets
        else:
            return [network]
    
    def get_metrics(self) -> ScanMetrics:
        """Get current scan metrics."""
        with self._lock:
            return ScanMetrics(
                total_requests=self.metrics.total_requests,
                successful_requests=self.metrics.successful_requests,
                failed_requests=self.metrics.failed_requests,
                blocked_requests=self.metrics.blocked_requests,
                average_response_time=self.metrics.average_response_time,
                detection_events=self.metrics.detection_events,
                rate_limit_hits=self.metrics.rate_limit_hits,
                back_off_events=self.metrics.back_off_events,
                resource_usage=self.metrics.resource_usage
            )


class ResourceMonitor:
    """Monitor system resources to prevent overload."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize resource monitor."""
        self.config = config or {}
        self.cpu_threshold = self.config.get('cpu_threshold', 80.0)
        self.memory_threshold = self.config.get('memory_threshold', 80.0)
        self.network_threshold = self.config.get('network_threshold', 70.0)
        self.disk_threshold = self.config.get('disk_threshold', 90.0)
        
        self.resource_history = deque(maxlen=100)
        self._lock = threading.Lock()
    
    def check_limits(self) -> bool:
        """Check if system resources are within limits."""
        if not PSUTIL_AVAILABLE:
            return True  # Assume OK if psutil not available
        
        try:
            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            if cpu_percent > self.cpu_threshold:
                return False
            
            # Check memory usage
            memory = psutil.virtual_memory()
            if memory.percent > self.memory_threshold:
                return False
            
            # Check disk usage
            disk = psutil.disk_usage('/')
            if disk.percent > self.disk_threshold:
                return False
            
            # Record resource usage
            with self._lock:
                self.resource_history.append({
                    'timestamp': time.time(),
                    'cpu': cpu_percent,
                    'memory': memory.percent,
                    'disk': disk.percent
                })
            
            return True
        
        except Exception as e:
            logger.warning(f"Resource monitoring error: {e}")
            return True  # Assume OK on error
    
    def get_resource_usage(self) -> Dict[str, float]:
        """Get current resource usage."""
        if not PSUTIL_AVAILABLE:
            return {}
        
        try:
            return {
                'cpu_percent': psutil.cpu_percent(interval=0.1),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': psutil.disk_usage('/').percent
            }
        except Exception:
            return {}


class DistributedRateLimiter:
    """Distributed rate limiter using Redis."""
    
    def __init__(self, redis_url: str, config: RateLimitConfig):
        """Initialize distributed rate limiter."""
        if not REDIS_AVAILABLE:
            raise ImportError("Redis is required for distributed rate limiting")
        
        self.redis_client = redis.from_url(redis_url)
        self.config = config
        self.key_prefix = config.distributed_key_prefix
    
    def _get_key(self, identifier: str, window: str) -> str:
        """Get Redis key for rate limiting."""
        return f"{self.key_prefix}:{identifier}:{window}"
    
    def check_rate_limit(self, identifier: str) -> RateLimitResult:
        """Check distributed rate limit."""
        now = time.time()
        
        # Check per-second limit
        second_key = self._get_key(identifier, "second")
        second_count = self.redis_client.get(second_key)
        
        if second_count and int(second_count) >= self.config.max_requests_per_second:
            return RateLimitResult(
                allowed=False,
                delay_required=1.0,
                retry_after=1.0,
                limit_exceeded=True
            )
        
        # Check per-minute limit
        minute_key = self._get_key(identifier, "minute")
        minute_count = self.redis_client.get(minute_key)
        
        if minute_count and int(minute_count) >= self.config.max_requests_per_minute:
            return RateLimitResult(
                allowed=False,
                delay_required=60.0,
                retry_after=60.0,
                limit_exceeded=True
            )
        
        # Increment counters
        pipe = self.redis_client.pipeline()
        pipe.incr(second_key)
        pipe.expire(second_key, 1)
        pipe.incr(minute_key)
        pipe.expire(minute_key, 60)
        pipe.execute()
        
        return RateLimitResult(allowed=True)


# Example usage functions
def demonstrate_rate_limiting():
    """Demonstrate rate limiting functionality."""
    rate_config = RateLimitConfig(
        strategy=RateLimitStrategy.ADAPTIVE,
        max_requests_per_second=5.0,
        max_requests_per_minute=200.0,
        burst_size=3,
        adaptive_threshold=0.8
    )
    
    rate_limiter = AdaptiveRateLimiter(rate_config)
    
    # Test rate limiting
    for i in range(10):
        request = ScanRequest(
            target=f"192.168.1.{i}",
            scan_type=ScanType.PING
        )
        
        result = rate_limiter.check_rate_limit(request)
        print(f"Request {i+1}: Allowed={result.allowed}, Delay={result.delay_required:.2f}s")
        
        if not result.allowed:
            time.sleep(result.delay_required)


def demonstrate_back_off():
    """Demonstrate back-off functionality."""
    back_off_config = BackOffConfig(
        strategy=BackOffStrategy.EXPONENTIAL,
        initial_delay=1.0,
        max_delay=60.0,
        multiplier=2.0,
        jitter_factor=0.1
    )
    
    back_off = IntelligentBackOff(back_off_config)
    
    # Test back-off for different targets
    targets = ["192.168.1.1", "192.168.1.2", "192.168.1.3"]
    
    for target in targets:
        for attempt in range(5):
            delay = back_off.get_delay(target, "connection_error")
            print(f"{target} attempt {attempt+1}: delay={delay:.2f}s")
        
        # Simulate success
        back_off.record_success(target)
        print(f"{target}: back-off reset after success")


async def demonstrate_stealth_scanning():
    """Demonstrate stealth scanning with rate limiting and back-off."""
    rate_config = RateLimitConfig(
        max_requests_per_second=2.0,
        max_requests_per_minute=100.0,
        burst_size=3
    )
    
    back_off_config = BackOffConfig(
        strategy=BackOffStrategy.EXPONENTIAL,
        initial_delay=1.0,
        max_delay=30.0
    )
    
    scanner = StealthScanner(rate_config, back_off_config)
    
    # Scan a small network
    results = await scanner.scan_network("192.168.1.0/24", ScanType.PING, max_targets=5)
    
    print("Scan Results:")
    for result in results:
        print(f"  {result['target']}: {result['status']}")
        if result['status'] == 'success':
            print(f"    Response time: {result.get('response_time', 0):.2f}s")
    
    # Get metrics
    metrics = scanner.get_metrics()
    print(f"\nScan Metrics:")
    print(f"  Total requests: {metrics.total_requests}")
    print(f"  Successful: {metrics.successful_requests}")
    print(f"  Failed: {metrics.failed_requests}")
    print(f"  Rate limit hits: {metrics.rate_limit_hits}")


def demonstrate_distributed_rate_limiting():
    """Demonstrate distributed rate limiting."""
    if not REDIS_AVAILABLE:
        print("Redis not available, skipping distributed rate limiting demo")
        return
    
    rate_config = RateLimitConfig(
        distributed_key_prefix="scan_rate_limit",
        max_requests_per_second=10.0,
        max_requests_per_minute=300.0
    )
    
    # This would require a Redis server
    # limiter = DistributedRateLimiter("redis://localhost:6379", rate_config)
    
    print("Distributed rate limiting would be used in a multi-instance environment")


def main():
    """Main function demonstrating rate limiting and back-off for network scanning."""
    logger.info("Starting rate limiting and back-off examples")
    
    # Demonstrate rate limiting
    try:
        demonstrate_rate_limiting()
    except Exception as e:
        logger.error(f"Rate limiting demonstration failed: {e}")
    
    # Demonstrate back-off
    try:
        demonstrate_back_off()
    except Exception as e:
        logger.error(f"Back-off demonstration failed: {e}")
    
    # Demonstrate stealth scanning
    try:
        asyncio.run(demonstrate_stealth_scanning())
    except Exception as e:
        logger.error(f"Stealth scanning demonstration failed: {e}")
    
    # Demonstrate distributed rate limiting
    try:
        demonstrate_distributed_rate_limiting()
    except Exception as e:
        logger.error(f"Distributed rate limiting demonstration failed: {e}")
    
    logger.info("Rate limiting and back-off examples completed")


match __name__:
    case "__main__":
    main() 