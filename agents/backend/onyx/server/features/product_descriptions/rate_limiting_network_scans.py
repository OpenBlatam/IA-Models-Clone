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
import time
import random
import logging
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
import yaml
from pathlib import Path
import hashlib
import socket
import ipaddress
from collections import defaultdict, deque
import statistics
from contextlib import asynccontextmanager
import aiohttp
import aiofiles
from fastapi import FastAPI, HTTPException, status, Depends
from pydantic import BaseModel, Field, field_validator
import redis.asyncio as redis
from roro_models import ScanRequestModel, ScanResponseModel, ErrorModel, ScanType, RateLimitType, BackoffStrategy
from middleware.centralized import CentralizedMiddleware
        import re
from fastapi import APIRouter
    import uvicorn
from typing import Any, List, Dict, Optional
"""
Rate Limiting and Back-off System for Network Scans
Implements intelligent rate limiting, back-off strategies, and detection avoidance
"""


class ScanType(Enum):
    """Types of network scans"""
    PORT_SCAN = "port_scan"
    SERVICE_DETECTION = "service_detection"
    VULNERABILITY_SCAN = "vulnerability_scan"
    OS_DETECTION = "os_detection"
    BANNER_GRABBING = "banner_grabbing"
    DNS_ENUMERATION = "dns_enumeration"
    SUBDOMAIN_SCAN = "subdomain_scan"
    WEB_CRAWLING = "web_crawling"

class BackoffStrategy(Enum):
    """Back-off strategies for rate limiting"""
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    FIBONACCI = "fibonacci"
    RANDOM = "random"
    ADAPTIVE = "adaptive"

class RateLimitType(Enum):
    """Types of rate limiting"""
    FIXED = "fixed"
    ADAPTIVE = "adaptive"
    BURST = "burst"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"

@dataclass
class RateLimitConfig:
    """Rate limiting configuration"""
    requests_per_second: float = 1.0
    requests_per_minute: int = 60
    requests_per_hour: int = 3600
    burst_size: int = 10
    window_size: int = 60  # seconds
    token_bucket_capacity: int = 100
    token_bucket_rate: float = 10.0  # tokens per second
    
    # Back-off configuration
    initial_delay: float = 1.0  # seconds
    max_delay: float = 300.0  # 5 minutes
    backoff_multiplier: float = 2.0
    jitter_factor: float = 0.1
    
    # Detection avoidance
    randomize_delays: bool = True
    vary_user_agents: bool = True
    rotate_ips: bool = False
    respect_robots_txt: bool = True
    
    # Adaptive rate limiting
    adaptive_enabled: bool = True
    success_threshold: float = 0.8
    failure_threshold: float = 0.2
    adjustment_factor: float = 0.1

@dataclass
class NetworkTarget:
    """Network target information"""
    host: str
    port: Optional[int] = None
    protocol: str = "tcp"
    scan_type: ScanType = ScanType.PORT_SCAN
    priority: int = 1
    retry_count: int = 0
    last_scan: Optional[datetime] = None
    success_count: int = 0
    failure_count: int = 0
    response_time: Optional[float] = None
    
    def __post_init__(self) -> Any:
        if isinstance(self.host, str):
            self.host = self.host.strip()

@dataclass
class ScanResult:
    """Scan result information"""
    target: NetworkTarget
    success: bool
    response_time: float
    data: Dict[str, Any]
    timestamp: datetime
    error_message: Optional[str] = None
    retry_count: int = 0

class RateLimiter:
    """Base rate limiter class"""
    
    def __init__(self, config: RateLimitConfig):
        
    """__init__ function."""
self.config = config
        self.logger = logging.getLogger(__name__)
        self.request_history = deque(maxlen=1000)
        self.last_request_time = 0.0
    
    async def acquire(self) -> bool:
        """Acquire permission to make a request"""
        raise NotImplementedError
    
    async def release(self) -> Any:
        """Release the rate limit after request completion"""
        pass
    
    def calculate_delay(self, retry_count: int = 0) -> float:
        """Calculate delay for back-off strategy"""
        base_delay = self.config.initial_delay * (self.config.backoff_multiplier ** retry_count)
        delay = min(base_delay, self.config.max_delay)
        
        if self.config.randomize_delays:
            jitter = delay * self.config.jitter_factor * random.uniform(-1, 1)
            delay += jitter
        
        return max(0, delay)

class FixedRateLimiter(RateLimiter):
    """Fixed rate limiter with constant rate"""
    
    def __init__(self, config: RateLimitConfig):
        
    """__init__ function."""
super().__init__(config)
        self.min_interval = 1.0 / config.requests_per_second
    
    async def acquire(self) -> bool:
        """Acquire permission with fixed rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_interval:
            delay = self.min_interval - time_since_last
            await asyncio.sleep(delay)
        
        self.last_request_time = time.time()
        self.request_history.append(current_time)
        return True

class AdaptiveRateLimiter(RateLimiter):
    """Adaptive rate limiter that adjusts based on success/failure rates"""
    
    def __init__(self, config: RateLimitConfig):
        
    """__init__ function."""
super().__init__(config)
        self.current_rate = config.requests_per_second
        self.success_history = deque(maxlen=100)
        self.failure_history = deque(maxlen=100)
        self.rate_history = deque(maxlen=50)
    
    async def acquire(self) -> bool:
        """Acquire permission with adaptive rate limiting"""
        current_time = time.time()
        
        # Calculate current success rate
        if len(self.success_history) > 0:
            success_rate = sum(self.success_history) / len(self.success_history)
            
            # Adjust rate based on success rate
            if success_rate > self.config.success_threshold:
                # Increase rate
                self.current_rate *= (1 + self.config.adjustment_factor)
            elif success_rate < self.config.failure_threshold:
                # Decrease rate
                self.current_rate *= (1 - self.config.adjustment_factor)
            
            # Ensure rate stays within bounds
            self.current_rate = max(0.1, min(self.current_rate, 10.0))
        
        min_interval = 1.0 / self.current_rate
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < min_interval:
            delay = min_interval - time_since_last
            await asyncio.sleep(delay)
        
        self.last_request_time = time.time()
        self.request_history.append(current_time)
        self.rate_history.append(self.current_rate)
        return True
    
    def record_success(self) -> Any:
        """Record successful request"""
        self.success_history.append(1)
        self.failure_history.append(0)
    
    def record_failure(self) -> Any:
        """Record failed request"""
        self.success_history.append(0)
        self.failure_history.append(1)

class SlidingWindowRateLimiter(RateLimiter):
    """Sliding window rate limiter"""
    
    def __init__(self, config: RateLimitConfig):
        
    """__init__ function."""
super().__init__(config)
        self.window_size = config.window_size
        self.max_requests = config.requests_per_minute
    
    async def acquire(self) -> bool:
        """Acquire permission with sliding window rate limiting"""
        current_time = time.time()
        
        # Remove old requests outside the window
        while self.request_history and current_time - self.request_history[0] > self.window_size:
            self.request_history.popleft()
        
        # Check if we can make a request
        if len(self.request_history) >= self.max_requests:
            # Wait until we can make another request
            wait_time = self.window_size - (current_time - self.request_history[0])
            if wait_time > 0:
                await asyncio.sleep(wait_time)
        
        self.request_history.append(current_time)
        return True

class TokenBucketRateLimiter(RateLimiter):
    """Token bucket rate limiter"""
    
    def __init__(self, config: RateLimitConfig):
        
    """__init__ function."""
super().__init__(config)
        self.tokens = config.token_bucket_capacity
        self.capacity = config.token_bucket_capacity
        self.rate = config.token_bucket_rate
        self.last_refill = time.time()
    
    async def acquire(self) -> bool:
        """Acquire permission with token bucket rate limiting"""
        current_time = time.time()
        
        # Refill tokens
        time_passed = current_time - self.last_refill
        tokens_to_add = time_passed * self.rate
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = current_time
        
        # Check if we have tokens
        if self.tokens < 1:
            # Wait for tokens to refill
            wait_time = (1 - self.tokens) / self.rate
            await asyncio.sleep(wait_time)
            self.tokens = 0
        else:
            self.tokens -= 1
        
        self.request_history.append(current_time)
        return True

class BackoffManager:
    """Manages back-off strategies for failed requests"""
    
    def __init__(self, config: RateLimitConfig):
        
    """__init__ function."""
self.config = config
        self.failure_counts = defaultdict(int)
        self.last_failure_times = defaultdict(float)
        self.fibonacci_sequence = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610]
    
    def calculate_backoff(self, target: NetworkTarget, strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL) -> float:
        """Calculate back-off delay for a target"""
        failure_count = self.failure_counts[target.host]
        
        if strategy == BackoffStrategy.LINEAR:
            delay = self.config.initial_delay * (1 + failure_count)
        elif strategy == BackoffStrategy.EXPONENTIAL:
            delay = self.config.initial_delay * (self.config.backoff_multiplier ** failure_count)
        elif strategy == BackoffStrategy.FIBONACCI:
            index = min(failure_count, len(self.fibonacci_sequence) - 1)
            delay = self.config.initial_delay * self.fibonacci_sequence[index]
        elif strategy == BackoffStrategy.RANDOM:
            base_delay = self.config.initial_delay * (self.config.backoff_multiplier ** failure_count)
            delay = random.uniform(0.5 * base_delay, 1.5 * base_delay)
        elif strategy == BackoffStrategy.ADAPTIVE:
            # Adaptive back-off based on target response patterns
            success_rate = target.success_count / max(1, target.success_count + target.failure_count)
            if success_rate > 0.8:
                delay = self.config.initial_delay
            elif success_rate > 0.5:
                delay = self.config.initial_delay * 2
            else:
                delay = self.config.initial_delay * (self.config.backoff_multiplier ** failure_count)
        else:
            delay = self.config.initial_delay
        
        # Apply jitter
        if self.config.randomize_delays:
            jitter = delay * self.config.jitter_factor * random.uniform(-1, 1)
            delay += jitter
        
        return min(delay, self.config.max_delay)
    
    def record_failure(self, target: NetworkTarget):
        """Record a failure for a target"""
        self.failure_counts[target.host] += 1
        self.last_failure_times[target.host] = time.time()
        target.failure_count += 1
    
    def record_success(self, target: NetworkTarget):
        """Record a success for a target"""
        self.failure_counts[target.host] = max(0, self.failure_counts[target.host] - 1)
        target.success_count += 1
    
    def should_retry(self, target: NetworkTarget, max_retries: int = 3) -> bool:
        """Check if target should be retried"""
        return target.retry_count < max_retries

class DetectionAvoidance:
    """Implements detection avoidance techniques"""
    
    def __init__(self, config: RateLimitConfig):
        
    """__init__ function."""
self.config = config
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:91.0) Gecko/20100101",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15"
        ]
        self.current_user_agent_index = 0
        self.robots_cache = {}
    
    def get_random_user_agent(self) -> str:
        """Get a random user agent"""
        if self.config.vary_user_agents:
            return random.choice(self.user_agents)
        else:
            return self.user_agents[self.current_user_agent_index]
    
    def rotate_user_agent(self) -> Any:
        """Rotate to next user agent"""
        if self.config.vary_user_agents:
            self.current_user_agent_index = (self.current_user_agent_index + 1) % len(self.user_agents)
    
    async def check_robots_txt(self, base_url: str) -> bool:
        """Check robots.txt for allowed paths"""
        if not self.config.respect_robots_txt:
            return True
        
        if base_url in self.robots_cache:
            return self.robots_cache[base_url]
        
        try:
            robots_url = f"{base_url}/robots.txt"
            async with aiohttp.ClientSession() as session:
                async with session.get(robots_url, timeout=10) as response:
                    if response.status == 200:
                        content = await response.text()
                        # Simple robots.txt parsing
                        allowed = "Disallow:" not in content or "User-agent: *" not in content
                        self.robots_cache[base_url] = allowed
                        return allowed
        except Exception:
            pass
        
        self.robots_cache[base_url] = True
        return True
    
    def add_random_delay(self, base_delay: float) -> float:
        """Add random delay to avoid detection"""
        if self.config.randomize_delays:
            jitter = base_delay * self.config.jitter_factor * random.uniform(-0.5, 0.5)
            return max(0, base_delay + jitter)
        return base_delay

class NetworkScanner:
    """Network scanner with rate limiting and back-off"""
    
    def __init__(self, config: RateLimitConfig, rate_limiter_type: RateLimitType = RateLimitType.ADAPTIVE):
        
    """__init__ function."""
self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize rate limiter
        if rate_limiter_type == RateLimitType.FIXED:
            self.rate_limiter = FixedRateLimiter(config)
        elif rate_limiter_type == RateLimitType.ADAPTIVE:
            self.rate_limiter = AdaptiveRateLimiter(config)
        elif rate_limiter_type == RateLimitType.SLIDING_WINDOW:
            self.rate_limiter = SlidingWindowRateLimiter(config)
        elif rate_limiter_type == RateLimitType.TOKEN_BUCKET:
            self.rate_limiter = TokenBucketRateLimiter(config)
        else:
            self.rate_limiter = FixedRateLimiter(config)
        
        # Initialize back-off manager
        self.backoff_manager = BackoffManager(config)
        
        # Initialize detection avoidance
        self.detection_avoidance = DetectionAvoidance(config)
        
        # Scan state
        self.scan_queue = asyncio.Queue()
        self.results = []
        self.active_scans = set()
        self.scan_stats = {
            "total_scans": 0,
            "successful_scans": 0,
            "failed_scans": 0,
            "average_response_time": 0.0
        }
    
    async def add_target(self, target: NetworkTarget):
        """Add target to scan queue"""
        await self.scan_queue.put(target)
    
    async def scan_target(self, target: NetworkTarget) -> ScanResult:
        """Scan a single target with rate limiting and back-off"""
        start_time = time.time()
        
        # Check if we should retry
        if not self.backoff_manager.should_retry(target):
            return ScanResult(
                target=target,
                success=False,
                response_time=0.0,
                data={},
                timestamp=datetime.utcnow(),
                error_message="Max retries exceeded",
                retry_count=target.retry_count
            )
        
        # Calculate back-off delay
        backoff_delay = self.backoff_manager.calculate_backoff(target)
        if backoff_delay > 0:
            await asyncio.sleep(backoff_delay)
        
        # Acquire rate limit permission
        await self.rate_limiter.acquire()
        
        try:
            # Perform the scan
            result = await self._perform_scan(target)
            
            # Record success
            self.backoff_manager.record_success(target)
            if isinstance(self.rate_limiter, AdaptiveRateLimiter):
                self.rate_limiter.record_success()
            
            # Update target stats
            target.last_scan = datetime.utcnow()
            target.response_time = result.response_time
            
            return result
            
        except Exception as e:
            # Record failure
            self.backoff_manager.record_failure(target)
            if isinstance(self.rate_limiter, AdaptiveRateLimiter):
                self.rate_limiter.record_failure()
            
            # Increment retry count
            target.retry_count += 1
            
            return ScanResult(
                target=target,
                success=False,
                response_time=time.time() - start_time,
                data={},
                timestamp=datetime.utcnow(),
                error_message=str(e),
                retry_count=target.retry_count
            )
    
    async def _perform_scan(self, target: NetworkTarget) -> ScanResult:
        """Perform the actual scan based on scan type"""
        start_time = time.time()
        
        if target.scan_type == ScanType.PORT_SCAN:
            return await self._port_scan(target)
        elif target.scan_type == ScanType.SERVICE_DETECTION:
            return await self._service_detection(target)
        elif target.scan_type == ScanType.BANNER_GRABBING:
            return await self._banner_grabbing(target)
        elif target.scan_type == ScanType.DNS_ENUMERATION:
            return await self._dns_enumeration(target)
        elif target.scan_type == ScanType.WEB_CRAWLING:
            return await self._web_crawling(target)
        else:
            raise ValueError(f"Unsupported scan type: {target.scan_type}")
    
    async def _port_scan(self, target: NetworkTarget) -> ScanResult:
        """Perform port scan"""
        start_time = time.time()
        
        try:
            # Create socket connection
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(10)
            
            result = sock.connect_ex((target.host, target.port or 80))
            response_time = time.time() - start_time
            
            data = {
                "port": target.port,
                "open": result == 0,
                "error_code": result
            }
            
            sock.close()
            
            return ScanResult(
                target=target,
                success=True,
                response_time=response_time,
                data=data,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            return ScanResult(
                target=target,
                success=False,
                response_time=time.time() - start_time,
                data={},
                timestamp=datetime.utcnow(),
                error_message=str(e)
            )
    
    async def _service_detection(self, target: NetworkTarget) -> ScanResult:
        """Perform service detection"""
        start_time = time.time()
        
        try:
            # Use aiohttp for HTTP-based service detection
            headers = {"User-Agent": self.detection_avoidance.get_random_user_agent()}
            
            async with aiohttp.ClientSession() as session:
                url = f"http://{target.host}:{target.port or 80}"
                async with session.get(url, headers=headers, timeout=10) as response:
                    response_time = time.time() - start_time
                    
                    data = {
                        "status_code": response.status,
                        "headers": dict(response.headers),
                        "content_type": response.headers.get("content-type", ""),
                        "server": response.headers.get("server", "")
                    }
                    
                    return ScanResult(
                        target=target,
                        success=True,
                        response_time=response_time,
                        data=data,
                        timestamp=datetime.utcnow()
                    )
                    
        except Exception as e:
            return ScanResult(
                target=target,
                success=False,
                response_time=time.time() - start_time,
                data={},
                timestamp=datetime.utcnow(),
                error_message=str(e)
            )
    
    async def _banner_grabbing(self, target: NetworkTarget) -> ScanResult:
        """Perform banner grabbing"""
        start_time = time.time()
        
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(10)
            sock.connect((target.host, target.port or 80))
            
            # Send a simple request
            request = b"GET / HTTP/1.1\r\nHost: " + target.host.encode() + b"\r\n\r\n"
            sock.send(request)
            
            # Receive response
            response = sock.recv(1024)
            response_time = time.time() - start_time
            
            data = {
                "banner": response.decode('utf-8', errors='ignore'),
                "port": target.port
            }
            
            sock.close()
            
            return ScanResult(
                target=target,
                success=True,
                response_time=response_time,
                data=data,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            return ScanResult(
                target=target,
                success=False,
                response_time=time.time() - start_time,
                data={},
                timestamp=datetime.utcnow(),
                error_message=str(e)
            )
    
    async def _dns_enumeration(self, target: NetworkTarget) -> ScanResult:
        """Perform DNS enumeration"""
        start_time = time.time()
        
        try:
            # Resolve hostname
            ip_address = socket.gethostbyname(target.host)
            response_time = time.time() - start_time
            
            data = {
                "ip_address": ip_address,
                "hostname": target.host
            }
            
            return ScanResult(
                target=target,
                success=True,
                response_time=response_time,
                data=data,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            return ScanResult(
                target=target,
                success=False,
                response_time=time.time() - start_time,
                data={},
                timestamp=datetime.utcnow(),
                error_message=str(e)
            )
    
    async def _web_crawling(self, target: NetworkTarget) -> ScanResult:
        """Perform web crawling"""
        start_time = time.time()
        
        try:
            # Check robots.txt first
            base_url = f"http://{target.host}"
            if not await self.detection_avoidance.check_robots_txt(base_url):
                return ScanResult(
                    target=target,
                    success=False,
                    response_time=time.time() - start_time,
                    data={},
                    timestamp=datetime.utcnow(),
                    error_message="Blocked by robots.txt"
                )
            
            headers = {"User-Agent": self.detection_avoidance.get_random_user_agent()}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(base_url, headers=headers, timeout=10) as response:
                    response_time = time.time() - start_time
                    
                    content = await response.text()
                    
                    data = {
                        "status_code": response.status,
                        "title": self._extract_title(content),
                        "content_length": len(content),
                        "headers": dict(response.headers)
                    }
                    
                    return ScanResult(
                        target=target,
                        success=True,
                        response_time=response_time,
                        data=data,
                        timestamp=datetime.utcnow()
                    )
                    
        except Exception as e:
            return ScanResult(
                target=target,
                success=False,
                response_time=time.time() - start_time,
                data={},
                timestamp=datetime.utcnow(),
                error_message=str(e)
            )
    
    def _extract_title(self, html_content: str) -> str:
        """Extract title from HTML content"""
        title_match = re.search(r'<title>(.*?)</title>', html_content, re.IGNORECASE)
        return title_match.group(1) if title_match else ""
    
    async def run_scan(self, max_concurrent: int = 5):
        """Run the scan with rate limiting and back-off"""
        workers = []
        
        # Start worker tasks
        for i in range(max_concurrent):
            worker = asyncio.create_task(self._scan_worker(f"worker-{i}"))
            workers.append(worker)
        
        # Wait for all workers to complete
        await asyncio.gather(*workers)
    
    async def _scan_worker(self, worker_name: str):
        """Worker task for scanning"""
        while True:
            try:
                # Get target from queue
                target = await asyncio.wait_for(self.scan_queue.get(), timeout=1.0)
                
                # Perform scan
                result = await self.scan_target(target)
                
                # Store result
                self.results.append(result)
                
                # Update stats
                self.scan_stats["total_scans"] += 1
                if result.success:
                    self.scan_stats["successful_scans"] += 1
                else:
                    self.scan_stats["failed_scans"] += 1
                
                # Mark task as done
                self.scan_queue.task_done()
                
            except asyncio.TimeoutError:
                # No more targets
                break
            except Exception as e:
                self.logger.error(f"Worker {worker_name} error: {e}")
    
    def get_scan_stats(self) -> Dict[str, Any]:
        """Get scan statistics"""
        if self.scan_stats["total_scans"] > 0:
            success_rate = self.scan_stats["successful_scans"] / self.scan_stats["total_scans"]
        else:
            success_rate = 0.0
        
        response_times = [r.response_time for r in self.results if r.success]
        avg_response_time = statistics.mean(response_times) if response_times else 0.0
        
        return {
            **self.scan_stats,
            "success_rate": success_rate,
            "average_response_time": avg_response_time,
            "total_results": len(self.results)
        }

# FastAPI router

router = APIRouter(prefix="/network-scanner", tags=["Network Scanner"])

@router.post("/scan", response_model=ScanResponseModel, responses={500: {"model": ErrorModel}})
async def start_network_scan(request: ScanRequestModel) -> ScanResponseModel:
    """Start a network scan with rate limiting and back-off"""
    
    # Create rate limit config
    config = RateLimitConfig()
    
    # Create scanner
    scanner = NetworkScanner(config, request.rate_limit_type)
    
    # Add targets
    for target_str in request.targets:
        target = NetworkTarget(
            host=target_str,
            scan_type=request.scan_type
        )
        await scanner.add_target(target)
    
    # Run scan
    await scanner.run_scan(request.max_concurrent)
    
    # Get results
    results = []
    for result in scanner.results:
        results.append({
            "target": result.target.host,
            "success": result.success,
            "response_time": result.response_time,
            "data": result.data,
            "error": result.error_message,
            "retry_count": result.retry_count
        })
    
    return ScanResponseModel(
        scan_id=f"scan_{int(time.time())}",
        status="completed",
        results=results,
        stats=scanner.get_scan_stats(),
        timestamp=datetime.utcnow()
    )

# Demo function
async def demo_network_scanner():
    """Demonstrate network scanner with rate limiting"""
    print("=== Network Scanner with Rate Limiting Demo ===\n")
    
    # Create configuration
    config = RateLimitConfig(
        requests_per_second=0.5,  # 1 request every 2 seconds
        requests_per_minute=30,
        initial_delay=1.0,
        max_delay=60.0,
        backoff_multiplier=2.0,
        randomize_delays=True,
        vary_user_agents=True,
        respect_robots_txt=True
    )
    
    # Create scanner with adaptive rate limiting
    scanner = NetworkScanner(config, RateLimitType.ADAPTIVE)
    
    # Add test targets
    test_targets = [
        NetworkTarget("google.com", 80, scan_type=ScanType.PORT_SCAN),
        NetworkTarget("github.com", 443, scan_type=ScanType.SERVICE_DETECTION),
        NetworkTarget("example.com", 80, scan_type=ScanType.BANNER_GRABBING),
        NetworkTarget("httpbin.org", 80, scan_type=ScanType.WEB_CRAWLING),
        NetworkTarget("invalid-domain-12345.com", 80, scan_type=ScanType.DNS_ENUMERATION)
    ]
    
    print("1. Adding targets to scan queue...")
    for target in test_targets:
        await scanner.add_target(target)
        print(f"   Added: {target.host}:{target.port} ({target.scan_type.value})")
    
    print(f"\n2. Starting scan with rate limiting...")
    print(f"   Rate limit: {config.requests_per_second} requests/second")
    print(f"   Back-off strategy: Exponential with jitter")
    print(f"   Detection avoidance: Enabled")
    
    start_time = time.time()
    await scanner.run_scan(max_concurrent=2)
    end_time = time.time()
    
    print(f"\n3. Scan completed in {end_time - start_time:.2f} seconds")
    
    # Display results
    print("\n4. Scan Results:")
    stats = scanner.get_scan_stats()
    print(f"   Total scans: {stats['total_scans']}")
    print(f"   Successful: {stats['successful_scans']}")
    print(f"   Failed: {stats['failed_scans']}")
    print(f"   Success rate: {stats['success_rate']:.2%}")
    print(f"   Average response time: {stats['average_response_time']:.3f}s")
    
    print("\n5. Detailed Results:")
    for result in scanner.results:
        status = "✓" if result.success else "✗"
        print(f"   {status} {result.target.host}:{result.target.port}")
        print(f"      Type: {result.target.scan_type.value}")
        print(f"      Response time: {result.response_time:.3f}s")
        print(f"      Retries: {result.retry_count}")
        if result.error_message:
            print(f"      Error: {result.error_message}")
        if result.success and result.data:
            print(f"      Data: {list(result.data.keys())}")
        print()
    
    print("=== Network Scanner Demo Completed! ===")

# FastAPI app
app = FastAPI(
    title="Network Scanner with Rate Limiting",
    description="Network scanner with intelligent rate limiting and back-off strategies",
    version="1.0.0"
)

# Add centralized middleware
app.add_middleware(CentralizedMiddleware)

# Include router
app.include_router(router)

if __name__ == "__main__":
    print("Network Scanner with Rate Limiting")
    print("Access API at: http://localhost:8000")
    print("API Documentation at: http://localhost:8000/docs")
    
    # Run demo
    asyncio.run(demo_network_scanner())
    
    # Start server
    uvicorn.run(app, host="0.0.0.0", port=8000) 