"""
Rate Limiting Tests for LinkedIn Posts

This module contains comprehensive tests for rate limiting functionality,
different rate limiting strategies, and rate limiting scenarios used in the LinkedIn posts feature.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import time
import uuid

# Rate limiting strategies and implementations
class RateLimitStrategy:
    """Base rate limiting strategy"""
    
    def __init__(self, name: str):
        self.name = name
        self.requests = []
        self.blocked_requests = 0
        self.allowed_requests = 0
    
    async def is_allowed(self, identifier: str, timestamp: Optional[datetime] = None) -> bool:
        """Check if request is allowed"""
        raise NotImplementedError
    
    async def record_request(self, identifier: str, timestamp: Optional[datetime] = None):
        """Record a request"""
        raise NotImplementedError
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiting statistics"""
        return {
            'name': self.name,
            'total_requests': len(self.requests),
            'blocked_requests': self.blocked_requests,
            'allowed_requests': self.allowed_requests,
            'block_rate': (self.blocked_requests / len(self.requests) * 100) if self.requests else 0
        }

class FixedWindowRateLimit(RateLimitStrategy):
    """Fixed window rate limiting strategy"""
    
    def __init__(self, name: str, max_requests: int, window_seconds: int):
        super().__init__(name)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.windows: Dict[str, Dict[str, Any]] = {}
    
    async def is_allowed(self, identifier: str, timestamp: Optional[datetime] = None) -> bool:
        """Check if request is allowed in fixed window"""
        if timestamp is None:
            timestamp = datetime.now()
        
        window_key = self._get_window_key(identifier, timestamp)
        
        if window_key not in self.windows:
            return True
        
        window = self.windows[window_key]
        return window['count'] < self.max_requests
    
    async def record_request(self, identifier: str, timestamp: Optional[datetime] = None):
        """Record a request in fixed window"""
        if timestamp is None:
            timestamp = datetime.now()
        
        window_key = self._get_window_key(identifier, timestamp)
        
        if window_key not in self.windows:
            self.windows[window_key] = {
                'count': 0,
                'start_time': timestamp
            }
        
        self.windows[window_key]['count'] += 1
        self.requests.append({
            'identifier': identifier,
            'timestamp': timestamp,
            'window_key': window_key
        })
        
        if await self.is_allowed(identifier, timestamp):
            self.allowed_requests += 1
        else:
            self.blocked_requests += 1
    
    def _get_window_key(self, identifier: str, timestamp: datetime) -> str:
        """Get window key for fixed window"""
        window_start = timestamp.replace(
            second=timestamp.second - (timestamp.second % self.window_seconds),
            microsecond=0
        )
        return f"{identifier}:{window_start.isoformat()}"

class SlidingWindowRateLimit(RateLimitStrategy):
    """Sliding window rate limiting strategy"""
    
    def __init__(self, name: str, max_requests: int, window_seconds: int):
        super().__init__(name)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests_by_identifier: Dict[str, List[datetime]] = {}
    
    async def is_allowed(self, identifier: str, timestamp: Optional[datetime] = None) -> bool:
        """Check if request is allowed in sliding window"""
        if timestamp is None:
            timestamp = datetime.now()
        
        # Clean old requests
        await self._clean_old_requests(identifier, timestamp)
        
        # Count requests in window
        if identifier not in self.requests_by_identifier:
            return True
        
        requests_in_window = len(self.requests_by_identifier[identifier])
        return requests_in_window < self.max_requests
    
    async def record_request(self, identifier: str, timestamp: Optional[datetime] = None):
        """Record a request in sliding window"""
        if timestamp is None:
            timestamp = datetime.now()
        
        if identifier not in self.requests_by_identifier:
            self.requests_by_identifier[identifier] = []
        
        self.requests_by_identifier[identifier].append(timestamp)
        self.requests.append({
            'identifier': identifier,
            'timestamp': timestamp
        })
        
        if await self.is_allowed(identifier, timestamp):
            self.allowed_requests += 1
        else:
            self.blocked_requests += 1
    
    async def _clean_old_requests(self, identifier: str, timestamp: datetime):
        """Clean old requests outside the window"""
        if identifier not in self.requests_by_identifier:
            return
        
        cutoff_time = timestamp - timedelta(seconds=self.window_seconds)
        self.requests_by_identifier[identifier] = [
            req_time for req_time in self.requests_by_identifier[identifier]
            if req_time > cutoff_time
        ]

class TokenBucketRateLimit(RateLimitStrategy):
    """Token bucket rate limiting strategy"""
    
    def __init__(self, name: str, capacity: int, refill_rate: float):
        super().__init__(name)
        self.capacity = capacity
        self.refill_rate = refill_rate  # tokens per second
        self.buckets: Dict[str, Dict[str, Any]] = {}
    
    async def is_allowed(self, identifier: str, timestamp: Optional[datetime] = None) -> bool:
        """Check if request is allowed with token bucket"""
        if timestamp is None:
            timestamp = datetime.now()
        
        bucket = await self._get_bucket(identifier, timestamp)
        return bucket['tokens'] >= 1
    
    async def record_request(self, identifier: str, timestamp: Optional[datetime] = None):
        """Record a request with token bucket"""
        if timestamp is None:
            timestamp = datetime.now()
        
        bucket = await self._get_bucket(identifier, timestamp)
        
        if bucket['tokens'] >= 1:
            bucket['tokens'] -= 1
            self.allowed_requests += 1
        else:
            self.blocked_requests += 1
        
        self.requests.append({
            'identifier': identifier,
            'timestamp': timestamp,
            'tokens_remaining': bucket['tokens']
        })
    
    async def _get_bucket(self, identifier: str, timestamp: datetime) -> Dict[str, Any]:
        """Get or create token bucket for identifier"""
        if identifier not in self.buckets:
            self.buckets[identifier] = {
                'tokens': self.capacity,
                'last_refill': timestamp
            }
            return self.buckets[identifier]
        
        bucket = self.buckets[identifier]
        time_diff = (timestamp - bucket['last_refill']).total_seconds()
        
        # Refill tokens
        tokens_to_add = time_diff * self.refill_rate
        bucket['tokens'] = min(self.capacity, bucket['tokens'] + tokens_to_add)
        bucket['last_refill'] = timestamp
        
        return bucket

class RateLimitManager:
    """Rate limit manager for multiple strategies"""
    
    def __init__(self):
        self.strategies: Dict[str, RateLimitStrategy] = {}
        self.default_strategy = None
    
    def register_strategy(self, name: str, strategy: RateLimitStrategy, is_default: bool = False):
        """Register a rate limiting strategy"""
        self.strategies[name] = strategy
        if is_default:
            self.default_strategy = strategy
    
    async def is_allowed(self, identifier: str, strategy_name: Optional[str] = None,
                        timestamp: Optional[datetime] = None) -> bool:
        """Check if request is allowed"""
        strategy = self.strategies.get(strategy_name) if strategy_name else self.default_strategy
        if strategy:
            return await strategy.is_allowed(identifier, timestamp)
        return True
    
    async def record_request(self, identifier: str, strategy_name: Optional[str] = None,
                           timestamp: Optional[datetime] = None):
        """Record a request"""
        strategy = self.strategies.get(strategy_name) if strategy_name else self.default_strategy
        if strategy:
            await strategy.record_request(identifier, timestamp)
    
    def get_strategy_stats(self, strategy_name: Optional[str] = None) -> Dict[str, Any]:
        """Get strategy statistics"""
        if strategy_name:
            strategy = self.strategies.get(strategy_name)
            return strategy.get_stats() if strategy else {}
        
        return {name: strategy.get_stats() for name, strategy in self.strategies.items()}

class RateLimitMiddleware:
    """Rate limiting middleware"""
    
    def __init__(self, rate_limit_manager: RateLimitManager):
        self.rate_limit_manager = rate_limit_manager
        self.blocked_ips = set()
        self.whitelist = set()
        self.blacklist = set()
    
    async def process_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process request through rate limiting"""
        identifier = self._get_identifier(request_data)
        
        # Check whitelist/blacklist
        if identifier in self.blacklist:
            return {
                'allowed': False,
                'reason': 'blacklisted',
                'identifier': identifier
            }
        
        if identifier in self.whitelist:
            return {
                'allowed': True,
                'reason': 'whitelisted',
                'identifier': identifier
            }
        
        # Check rate limit
        is_allowed = await self.rate_limit_manager.is_allowed(identifier)
        
        if is_allowed:
            await self.rate_limit_manager.record_request(identifier)
            return {
                'allowed': True,
                'reason': 'rate_limit_passed',
                'identifier': identifier
            }
        else:
            return {
                'allowed': False,
                'reason': 'rate_limit_exceeded',
                'identifier': identifier
            }
    
    def _get_identifier(self, request_data: Dict[str, Any]) -> str:
        """Get identifier from request data"""
        # Could be IP, user ID, API key, etc.
        return request_data.get('ip_address', request_data.get('user_id', 'unknown'))
    
    def add_to_whitelist(self, identifier: str):
        """Add identifier to whitelist"""
        self.whitelist.add(identifier)
    
    def add_to_blacklist(self, identifier: str):
        """Add identifier to blacklist"""
        self.blacklist.add(identifier)
    
    def remove_from_whitelist(self, identifier: str):
        """Remove identifier from whitelist"""
        self.whitelist.discard(identifier)
    
    def remove_from_blacklist(self, identifier: str):
        """Remove identifier from blacklist"""
        self.blacklist.discard(identifier)

class RateLimitMonitor:
    """Rate limiting monitor"""
    
    def __init__(self):
        self.alerts = []
        self.thresholds = {
            'high_block_rate': 0.1,  # 10%
            'excessive_requests': 1000,
            'suspicious_pattern': 0.8  # 80% blocked
        }
    
    def check_alerts(self, stats: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for rate limiting alerts"""
        alerts = []
        
        # Check block rate
        if stats.get('block_rate', 0) > self.thresholds['high_block_rate'] * 100:
            alerts.append({
                'type': 'high_block_rate',
                'message': f"High block rate detected: {stats.get('block_rate', 0):.2f}%",
                'severity': 'warning'
            })
        
        # Check total requests
        if stats.get('total_requests', 0) > self.thresholds['excessive_requests']:
            alerts.append({
                'type': 'excessive_requests',
                'message': f"Excessive requests detected: {stats.get('total_requests', 0)}",
                'severity': 'info'
            })
        
        # Check suspicious patterns
        if (stats.get('blocked_requests', 0) / max(stats.get('total_requests', 1), 1) > 
            self.thresholds['suspicious_pattern']):
            alerts.append({
                'type': 'suspicious_pattern',
                'message': "Suspicious request pattern detected",
                'severity': 'critical'
            })
        
        self.alerts.extend(alerts)
        return alerts

@pytest.fixture
def fixed_window_rate_limit():
    """Fixed window rate limit fixture"""
    return FixedWindowRateLimit("test_fixed_window", max_requests=10, window_seconds=60)

@pytest.fixture
def sliding_window_rate_limit():
    """Sliding window rate limit fixture"""
    return SlidingWindowRateLimit("test_sliding_window", max_requests=10, window_seconds=60)

@pytest.fixture
def token_bucket_rate_limit():
    """Token bucket rate limit fixture"""
    return TokenBucketRateLimit("test_token_bucket", capacity=10, refill_rate=1.0)

@pytest.fixture
def rate_limit_manager():
    """Rate limit manager fixture"""
    return RateLimitManager()

@pytest.fixture
def rate_limit_middleware(rate_limit_manager):
    """Rate limit middleware fixture"""
    return RateLimitMiddleware(rate_limit_manager)

@pytest.fixture
def rate_limit_monitor():
    """Rate limit monitor fixture"""
    return RateLimitMonitor()

@pytest.fixture
def sample_requests():
    """Sample requests for testing"""
    return [
        {'ip_address': '192.168.1.1', 'user_id': 'user1', 'endpoint': '/api/posts'},
        {'ip_address': '192.168.1.2', 'user_id': 'user2', 'endpoint': '/api/posts'},
        {'ip_address': '192.168.1.1', 'user_id': 'user1', 'endpoint': '/api/posts'},
        {'ip_address': '192.168.1.3', 'user_id': 'user3', 'endpoint': '/api/posts'},
        {'ip_address': '192.168.1.1', 'user_id': 'user1', 'endpoint': '/api/posts'}
    ]

class TestRateLimiting:
    """Test rate limiting strategies and implementations"""
    
    async def test_fixed_window_basic_operations(self, fixed_window_rate_limit):
        """Test basic fixed window rate limiting operations"""
        identifier = "test_user"
        
        # First 10 requests should be allowed
        for i in range(10):
            is_allowed = await fixed_window_rate_limit.is_allowed(identifier)
            await fixed_window_rate_limit.record_request(identifier)
            assert is_allowed is True
        
        # 11th request should be blocked
        is_allowed = await fixed_window_rate_limit.is_allowed(identifier)
        assert is_allowed is False
        
        stats = fixed_window_rate_limit.get_stats()
        assert stats['allowed_requests'] == 10
        assert stats['blocked_requests'] == 1
    
    async def test_fixed_window_window_reset(self, fixed_window_rate_limit):
        """Test fixed window reset after window expires"""
        identifier = "test_user"
        
        # Fill the window
        for i in range(10):
            await fixed_window_rate_limit.record_request(identifier)
        
        # Wait for window to expire
        await asyncio.sleep(1)  # Simulate time passing
        
        # Create new window with different timestamp
        future_time = datetime.now() + timedelta(seconds=60)
        is_allowed = await fixed_window_rate_limit.is_allowed(identifier, future_time)
        assert is_allowed is True
    
    async def test_sliding_window_basic_operations(self, sliding_window_rate_limit):
        """Test basic sliding window rate limiting operations"""
        identifier = "test_user"
        
        # First 10 requests should be allowed
        for i in range(10):
            is_allowed = await sliding_window_rate_limit.is_allowed(identifier)
            await sliding_window_rate_limit.record_request(identifier)
            assert is_allowed is True
        
        # 11th request should be blocked
        is_allowed = await sliding_window_rate_limit.is_allowed(identifier)
        assert is_allowed is False
    
    async def test_sliding_window_cleanup(self, sliding_window_rate_limit):
        """Test sliding window cleanup of old requests"""
        identifier = "test_user"
        
        # Add requests at different times
        base_time = datetime.now()
        
        # Add 5 requests at base time
        for i in range(5):
            await sliding_window_rate_limit.record_request(identifier, base_time)
        
        # Add 5 more requests 30 seconds later
        later_time = base_time + timedelta(seconds=30)
        for i in range(5):
            await sliding_window_rate_limit.record_request(identifier, later_time)
        
        # Check that all 10 requests are counted
        is_allowed = await sliding_window_rate_limit.is_allowed(identifier, later_time)
        assert is_allowed is False
        
        # Simulate time passing (requests should be cleaned up)
        future_time = base_time + timedelta(seconds=90)
        is_allowed = await sliding_window_rate_limit.is_allowed(identifier, future_time)
        assert is_allowed is True
    
    async def test_token_bucket_basic_operations(self, token_bucket_rate_limit):
        """Test basic token bucket rate limiting operations"""
        identifier = "test_user"
        
        # First 10 requests should be allowed (bucket capacity)
        for i in range(10):
            is_allowed = await token_bucket_rate_limit.is_allowed(identifier)
            await token_bucket_rate_limit.record_request(identifier)
            assert is_allowed is True
        
        # 11th request should be blocked
        is_allowed = await token_bucket_rate_limit.is_allowed(identifier)
        assert is_allowed is False
    
    async def test_token_bucket_refill(self, token_bucket_rate_limit):
        """Test token bucket refill mechanism"""
        identifier = "test_user"
        
        # Use all tokens
        for i in range(10):
            await token_bucket_rate_limit.record_request(identifier)
        
        # Wait for refill (1 token per second)
        await asyncio.sleep(1.1)
        
        # Should have 1 token refilled
        is_allowed = await token_bucket_rate_limit.is_allowed(identifier)
        assert is_allowed is True
        
        # Use the refilled token
        await token_bucket_rate_limit.record_request(identifier)
        
        # Should be blocked again
        is_allowed = await token_bucket_rate_limit.is_allowed(identifier)
        assert is_allowed is False
    
    async def test_rate_limit_manager_multiple_strategies(self, rate_limit_manager, 
                                                        fixed_window_rate_limit,
                                                        sliding_window_rate_limit):
        """Test rate limit manager with multiple strategies"""
        # Register strategies
        rate_limit_manager.register_strategy('fixed_window', fixed_window_rate_limit, is_default=True)
        rate_limit_manager.register_strategy('sliding_window', sliding_window_rate_limit)
        
        identifier = "test_user"
        
        # Test default strategy
        is_allowed = await rate_limit_manager.is_allowed(identifier)
        await rate_limit_manager.record_request(identifier)
        assert is_allowed is True
        
        # Test specific strategy
        is_allowed = await rate_limit_manager.is_allowed(identifier, 'sliding_window')
        await rate_limit_manager.record_request(identifier, 'sliding_window')
        assert is_allowed is True
    
    async def test_rate_limit_middleware_processing(self, rate_limit_middleware, 
                                                  fixed_window_rate_limit):
        """Test rate limit middleware processing"""
        # Register strategy
        rate_limit_middleware.rate_limit_manager.register_strategy('fixed_window', 
                                                                 fixed_window_rate_limit, 
                                                                 is_default=True)
        
        # Process requests
        request_data = {'ip_address': '192.168.1.1', 'user_id': 'user1'}
        
        # First request should be allowed
        result = await rate_limit_middleware.process_request(request_data)
        assert result['allowed'] is True
        assert result['reason'] == 'rate_limit_passed'
        
        # Add to blacklist
        rate_limit_middleware.add_to_blacklist('192.168.1.1')
        
        # Request should be blocked
        result = await rate_limit_middleware.process_request(request_data)
        assert result['allowed'] is False
        assert result['reason'] == 'blacklisted'
    
    async def test_rate_limit_middleware_whitelist(self, rate_limit_middleware, 
                                                  fixed_window_rate_limit):
        """Test rate limit middleware whitelist functionality"""
        # Register strategy
        rate_limit_middleware.rate_limit_manager.register_strategy('fixed_window', 
                                                                 fixed_window_rate_limit, 
                                                                 is_default=True)
        
        request_data = {'ip_address': '192.168.1.1', 'user_id': 'user1'}
        
        # Add to whitelist
        rate_limit_middleware.add_to_whitelist('192.168.1.1')
        
        # Request should be allowed regardless of rate limit
        result = await rate_limit_middleware.process_request(request_data)
        assert result['allowed'] is True
        assert result['reason'] == 'whitelisted'
    
    async def test_rate_limit_monitor_alerts(self, rate_limit_monitor):
        """Test rate limit monitor alert generation"""
        # Test high block rate alert
        stats = {'block_rate': 15.0, 'total_requests': 100, 'blocked_requests': 15}
        alerts = rate_limit_monitor.check_alerts(stats)
        
        assert len(alerts) == 1
        assert alerts[0]['type'] == 'high_block_rate'
        assert alerts[0]['severity'] == 'warning'
        
        # Test excessive requests alert
        stats = {'total_requests': 1500, 'blocked_requests': 100}
        alerts = rate_limit_monitor.check_alerts(stats)
        
        assert len(alerts) == 1
        assert alerts[0]['type'] == 'excessive_requests'
    
    async def test_rate_limit_concurrent_access(self, fixed_window_rate_limit):
        """Test concurrent access to rate limiting"""
        identifier = "test_user"
        
        # Concurrent requests
        async def make_request():
            is_allowed = await fixed_window_rate_limit.is_allowed(identifier)
            await fixed_window_rate_limit.record_request(identifier)
            return is_allowed
        
        tasks = [make_request() for _ in range(15)]
        results = await asyncio.gather(*tasks)
        
        # First 10 should be allowed, rest blocked
        allowed_count = sum(1 for result in results if result)
        blocked_count = sum(1 for result in results if not result)
        
        assert allowed_count == 10
        assert blocked_count == 5
    
    async def test_rate_limit_different_identifiers(self, fixed_window_rate_limit):
        """Test rate limiting with different identifiers"""
        # Test with different users
        for user_id in ['user1', 'user2', 'user3']:
            is_allowed = await fixed_window_rate_limit.is_allowed(user_id)
            await fixed_window_rate_limit.record_request(user_id)
            assert is_allowed is True
        
        # Each user should have their own limit
        for user_id in ['user1', 'user2', 'user3']:
            is_allowed = await fixed_window_rate_limit.is_allowed(user_id)
            assert is_allowed is True
    
    async def test_rate_limit_error_handling(self, fixed_window_rate_limit):
        """Test rate limiting error handling"""
        # Test with invalid identifier
        is_allowed = await fixed_window_rate_limit.is_allowed("")
        assert is_allowed is True
        
        # Test with None identifier
        is_allowed = await fixed_window_rate_limit.is_allowed(None)
        assert is_allowed is True
    
    async def test_rate_limit_performance(self, fixed_window_rate_limit):
        """Test rate limiting performance"""
        identifier = "test_user"
        
        # Measure performance for many requests
        start_time = time.time()
        
        for i in range(100):
            await fixed_window_rate_limit.record_request(identifier)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete quickly
        assert duration < 1.0
        
        stats = fixed_window_rate_limit.get_stats()
        assert stats['total_requests'] == 100
    
    async def test_rate_limit_strategy_comparison(self, fixed_window_rate_limit,
                                                sliding_window_rate_limit,
                                                token_bucket_rate_limit):
        """Test comparison between different rate limiting strategies"""
        identifier = "test_user"
        
        # Test all strategies with same parameters
        strategies = [
            ('fixed_window', fixed_window_rate_limit),
            ('sliding_window', sliding_window_rate_limit),
            ('token_bucket', token_bucket_rate_limit)
        ]
        
        for name, strategy in strategies:
            # Reset strategy
            strategy.requests.clear()
            strategy.blocked_requests = 0
            strategy.allowed_requests = 0
            
            # Test 15 requests (should allow 10, block 5)
            for i in range(15):
                is_allowed = await strategy.is_allowed(identifier)
                await strategy.record_request(identifier)
            
            stats = strategy.get_stats()
            assert stats['allowed_requests'] == 10
            assert stats['blocked_requests'] == 5
    
    async def test_rate_limit_custom_limits(self):
        """Test rate limiting with custom limits"""
        # Test with very low limit
        low_limit = FixedWindowRateLimit("low_limit", max_requests=1, window_seconds=60)
        identifier = "test_user"
        
        # First request should be allowed
        is_allowed = await low_limit.is_allowed(identifier)
        await low_limit.record_request(identifier)
        assert is_allowed is True
        
        # Second request should be blocked
        is_allowed = await low_limit.is_allowed(identifier)
        assert is_allowed is False
        
        # Test with high limit
        high_limit = FixedWindowRateLimit("high_limit", max_requests=1000, window_seconds=60)
        
        # Many requests should be allowed
        for i in range(500):
            is_allowed = await high_limit.is_allowed(identifier)
            await high_limit.record_request(identifier)
            assert is_allowed is True
    
    async def test_rate_limit_window_boundaries(self, fixed_window_rate_limit):
        """Test rate limiting at window boundaries"""
        identifier = "test_user"
        base_time = datetime.now()
        
        # Fill window at end of window
        window_end = base_time.replace(second=59, microsecond=999999)
        
        for i in range(10):
            await fixed_window_rate_limit.record_request(identifier, window_end)
        
        # Request at start of next window should be allowed
        next_window_start = window_end + timedelta(seconds=1)
        is_allowed = await fixed_window_rate_limit.is_allowed(identifier, next_window_start)
        assert is_allowed is True
    
    async def test_rate_limit_middleware_identifier_extraction(self, rate_limit_middleware):
        """Test rate limit middleware identifier extraction"""
        # Test with IP address
        request_data = {'ip_address': '192.168.1.1', 'user_id': 'user1'}
        identifier = rate_limit_middleware._get_identifier(request_data)
        assert identifier == '192.168.1.1'
        
        # Test with user ID only
        request_data = {'user_id': 'user1'}
        identifier = rate_limit_middleware._get_identifier(request_data)
        assert identifier == 'user1'
        
        # Test with no identifier
        request_data = {}
        identifier = rate_limit_middleware._get_identifier(request_data)
        assert identifier == 'unknown'

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
