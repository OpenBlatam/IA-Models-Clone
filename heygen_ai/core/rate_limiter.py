#!/usr/bin/env python3
"""
Intelligent Rate Limiter for Enhanced HeyGen AI
Adaptive rate limiting based on user behavior and system load.
"""

import asyncio
import time
import json
from typing import Dict, Optional, List, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import structlog
from collections import defaultdict, deque
import hashlib
import psutil

logger = structlog.get_logger()

class RateLimitStrategy(Enum):
    """Rate limiting strategies."""
    FIXED = "fixed"
    ADAPTIVE = "adaptive"
    BURST = "burst"
    SLIDING_WINDOW = "sliding_window"

class RateLimitTier(Enum):
    """Rate limit tiers based on user type."""
    FREE = "free"
    BASIC = "basic"
    PRO = "pro"
    ENTERPRISE = "enterprise"

@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    requests_per_minute: int
    burst_size: int
    window_size: int  # seconds
    strategy: RateLimitStrategy
    tier: RateLimitTier
    adaptive_factor: float = 1.0
    min_requests: int = 1
    max_requests: int = 1000

@dataclass
class RateLimitState:
    """Current state of rate limiting for a user."""
    user_id: str
    tier: RateLimitTier
    current_requests: int
    window_start: float
    burst_tokens: int
    last_request: float
    violation_count: int
    adaptive_factor: float
    is_blocked: bool
    block_until: Optional[float]

class IntelligentRateLimiter:
    """Intelligent rate limiter with adaptive limits and user tier management."""
    
    def __init__(
        self,
        default_config: Optional[RateLimitConfig] = None,
        enable_adaptive: bool = True,
        system_load_threshold: float = 0.8,
        violation_threshold: int = 5
    ):
        self.enable_adaptive = enable_adaptive
        self.system_load_threshold = system_load_threshold
        self.violation_threshold = violation_threshold
        
        # Default configurations for each tier
        self.tier_configs = {
            RateLimitTier.FREE: RateLimitConfig(
                requests_per_minute=10,
                burst_size=5,
                window_size=60,
                strategy=RateLimitStrategy.FIXED,
                tier=RateLimitTier.FREE
            ),
            RateLimitTier.BASIC: RateLimitConfig(
                requests_per_minute=60,
                burst_size=20,
                window_size=60,
                strategy=RateLimitStrategy.ADAPTIVE,
                tier=RateLimitTier.BASIC,
                adaptive_factor=1.2
            ),
            RateLimitTier.PRO: RateLimitConfig(
                requests_per_minute=300,
                burst_size=100,
                window_size=60,
                strategy=RateLimitStrategy.ADAPTIVE,
                tier=RateLimitTier.PRO,
                adaptive_factor=1.5
            ),
            RateLimitTier.ENTERPRISE: RateLimitConfig(
                requests_per_minute=1000,
                burst_size=500,
                window_size=60,
                strategy=RateLimitStrategy.BURST,
                tier=RateLimitTier.ENTERPRISE,
                adaptive_factor=2.0
            )
        }
        
        # User states
        self.user_states: Dict[str, RateLimitState] = {}
        
        # Global system state
        self.system_stats = {
            "total_requests": 0,
            "blocked_requests": 0,
            "adaptive_adjustments": 0,
            "system_load_history": deque(maxlen=100)
        }
        
        # Background tasks
        self.cleanup_task: Optional[asyncio.Task] = None
        self.monitoring_task: Optional[asyncio.Task] = None
        
        # Start background tasks
        self._start_background_tasks()
    
    def _start_background_tasks(self):
        """Start background cleanup and monitoring tasks."""
        self.cleanup_task = asyncio.create_task(self._cleanup_old_states())
        self.monitoring_task = asyncio.create_task(self._monitor_system_load())
    
    async def _cleanup_old_states(self):
        """Clean up old user states."""
        while True:
            try:
                current_time = time.time()
                cutoff_time = current_time - 3600  # 1 hour
                
                users_to_remove = []
                for user_id, state in self.user_states.items():
                    if (state.last_request < cutoff_time and 
                        state.violation_count == 0 and
                        not state.is_blocked):
                        users_to_remove.append(user_id)
                
                for user_id in users_to_remove:
                    del self.user_states[user_id]
                
                if users_to_remove:
                    logger.debug(f"Cleaned up {len(users_to_remove)} old user states")
                
                await asyncio.sleep(300)  # Run every 5 minutes
                
            except Exception as e:
                logger.error(f"Cleanup task error: {e}")
                await asyncio.sleep(60)
    
    async def _monitor_system_load(self):
        """Monitor system load and adjust rate limits accordingly."""
        while True:
            try:
                # Get system load
                cpu_percent = psutil.cpu_percent(interval=1)
                memory_percent = psutil.virtual_memory().percent
                system_load = max(cpu_percent, memory_percent) / 100.0
                
                self.system_stats["system_load_history"].append(system_load)
                
                # Calculate average system load
                avg_load = sum(self.system_stats["system_load_history"]) / len(self.system_stats["system_load_history"])
                
                # Adjust rate limits based on system load
                if self.enable_adaptive and avg_load > self.system_load_threshold:
                    await self._adjust_rate_limits_for_load(avg_load)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Monitoring task error: {e}")
                await asyncio.sleep(60)
    
    async def _adjust_rate_limits_for_load(self, system_load: float):
        """Adjust rate limits based on system load."""
        load_factor = 1.0 - (system_load - self.system_load_threshold)
        load_factor = max(0.5, min(1.0, load_factor))  # Clamp between 0.5 and 1.0
        
        for tier, config in self.tier_configs.items():
            if config.strategy == RateLimitStrategy.ADAPTIVE:
                # Reduce rate limits under high load
                adjusted_requests = int(config.requests_per_minute * load_factor)
                config.requests_per_minute = max(config.min_requests, adjusted_requests)
                
                logger.info(f"Adjusted rate limits for {tier.value} tier", 
                           load_factor=load_factor,
                           new_limit=config.requests_per_minute)
        
        self.system_stats["adaptive_adjustments"] += 1
    
    def get_user_tier(self, user_id: str) -> RateLimitTier:
        """Get the rate limit tier for a user."""
        # This would typically query a database or user service
        # For now, use a simple hash-based assignment
        hash_value = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        
        if hash_value % 100 < 60:  # 60% of users
            return RateLimitTier.FREE
        elif hash_value % 100 < 85:  # 25% of users
            return RateLimitTier.BASIC
        elif hash_value % 100 < 95:  # 10% of users
            return RateLimitTier.PRO
        else:  # 5% of users
            return RateLimitTier.ENTERPRISE
    
    def _get_or_create_user_state(self, user_id: str) -> RateLimitState:
        """Get or create rate limit state for a user."""
        if user_id not in self.user_states:
            tier = self.get_user_tier(user_id)
            config = self.tier_configs[tier]
            
            state = RateLimitState(
                user_id=user_id,
                tier=tier,
                current_requests=0,
                window_start=time.time(),
                burst_tokens=config.burst_size,
                last_request=time.time(),
                violation_count=0,
                adaptive_factor=config.adaptive_factor,
                is_blocked=False,
                block_until=None
            )
            
            self.user_states[user_id] = state
        
        return self.user_states[user_id]
    
    async def check_rate_limit(
        self,
        user_id: str,
        endpoint: str = "default",
        request_weight: int = 1
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check if a request is allowed under rate limiting."""
        current_time = time.time()
        state = self._get_or_create_user_state(user_id)
        config = self.tier_configs[state.tier]
        
        # Check if user is blocked
        if state.is_blocked and state.block_until and current_time < state.block_until:
            remaining_block_time = state.block_until - current_time
            return False, {
                "allowed": False,
                "reason": "user_blocked",
                "remaining_block_time": remaining_block_time,
                "retry_after": state.block_until
            }
        
        # Reset block if time has passed
        if state.is_blocked and state.block_until and current_time >= state.block_until:
            state.is_blocked = False
            state.block_until = None
        
        # Check if window has expired
        if current_time - state.window_start >= config.window_size:
            state.window_start = current_time
            state.current_requests = 0
            state.burst_tokens = config.burst_size
        
        # Apply rate limiting strategy
        allowed = False
        reason = "unknown"
        
        if config.strategy == RateLimitStrategy.FIXED:
            allowed = state.current_requests + request_weight <= config.requests_per_minute
            
        elif config.strategy == RateLimitStrategy.ADAPTIVE:
            # Adaptive limits based on user behavior and system load
            adaptive_limit = int(config.requests_per_minute * state.adaptive_factor)
            allowed = state.current_requests + request_weight <= adaptive_limit
            
        elif config.strategy == RateLimitStrategy.BURST:
            # Allow burst requests up to burst size
            if request_weight <= state.burst_tokens:
                allowed = True
                state.burst_tokens -= request_weight
            else:
                allowed = state.current_requests + request_weight <= config.requests_per_minute
                
        elif config.strategy == RateLimitStrategy.SLIDING_WINDOW:
            # Sliding window with exponential backoff for violations
            base_limit = config.requests_per_minute
            violation_penalty = 2 ** min(state.violation_count, 5)
            effective_limit = max(config.min_requests, base_limit // violation_penalty)
            allowed = state.current_requests + request_weight <= effective_limit
        
        # Update state
        if allowed:
            state.current_requests += request_weight
            state.last_request = current_time
            state.violation_count = max(0, state.violation_count - 1)  # Reduce violations over time
            
            # Improve adaptive factor for good behavior
            if config.strategy == RateLimitStrategy.ADAPTIVE:
                state.adaptive_factor = min(
                    config.adaptive_factor,
                    state.adaptive_factor + 0.01
                )
        else:
            # Handle violation
            state.violation_count += 1
            
            # Block user if too many violations
            if state.violation_count >= self.violation_threshold:
                block_duration = min(300, 60 * (2 ** (state.violation_count - self.violation_threshold)))
                state.is_blocked = True
                state.block_until = current_time + block_duration
                
                logger.warning(f"User blocked due to rate limit violations", 
                             user_id=user_id,
                             violations=state.violation_count,
                             block_duration=block_duration)
            
            # Reduce adaptive factor for violations
            if config.strategy == RateLimitStrategy.ADAPTIVE:
                state.adaptive_factor = max(0.5, state.adaptive_factor - 0.1)
        
        # Update global stats
        self.system_stats["total_requests"] += 1
        if not allowed:
            self.system_stats["blocked_requests"] += 1
        
        # Prepare response
        response_data = {
            "allowed": allowed,
            "reason": reason,
            "user_tier": state.tier.value,
            "current_requests": state.current_requests,
            "limit": config.requests_per_minute,
            "window_reset": state.window_start + config.window_size,
            "burst_tokens": state.burst_tokens if config.strategy == RateLimitStrategy.BURST else None,
            "adaptive_factor": state.adaptive_factor if config.strategy == RateLimitStrategy.ADAPTIVE else None,
            "violation_count": state.violation_count,
            "is_blocked": state.is_blocked,
            "block_until": state.block_until
        }
        
        if not allowed:
            response_data["retry_after"] = state.window_start + config.window_size
        
        return allowed, response_data
    
    async def get_user_stats(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get rate limiting statistics for a user."""
        if user_id not in self.user_states:
            return None
        
        state = self.user_states[user_id]
        config = self.tier_configs[state.tier]
        
        return {
            "user_id": user_id,
            "tier": state.tier.value,
            "current_requests": state.current_requests,
            "limit": config.requests_per_minute,
            "window_start": state.window_start,
            "window_reset": state.window_start + config.window_size,
            "burst_tokens": state.burst_tokens,
            "last_request": state.last_request,
            "violation_count": state.violation_count,
            "adaptive_factor": state.adaptive_factor,
            "is_blocked": state.is_blocked,
            "block_until": state.block_until,
            "strategy": config.strategy.value
        }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system-wide rate limiting statistics."""
        total_users = len(self.user_states)
        blocked_users = sum(1 for state in self.user_states.values() if state.is_blocked)
        tier_distribution = defaultdict(int)
        
        for state in self.user_states.values():
            tier_distribution[state.tier.value] += 1
        
        # Calculate average system load
        avg_load = 0.0
        if self.system_stats["system_load_history"]:
            avg_load = sum(self.system_stats["system_load_history"]) / len(self.system_stats["system_load_history"])
        
        return {
            **self.system_stats,
            "total_users": total_users,
            "blocked_users": blocked_users,
            "active_users": total_users - blocked_users,
            "tier_distribution": dict(tier_distribution),
            "average_system_load": avg_load,
            "rate_limit_configs": {
                tier.value: {
                    "requests_per_minute": config.requests_per_minute,
                    "strategy": config.strategy.value,
                    "adaptive_factor": config.adaptive_factor
                }
                for tier, config in self.tier_configs.items()
            }
        }
    
    def update_user_tier(self, user_id: str, new_tier: RateLimitTier) -> bool:
        """Update the rate limit tier for a user."""
        if user_id in self.user_states:
            old_tier = self.user_states[user_id].tier
            self.user_states[user_id].tier = new_tier
            
            # Reset state for new tier
            config = self.tier_configs[new_tier]
            self.user_states[user_id].burst_tokens = config.burst_size
            self.user_states[user_id].adaptive_factor = config.adaptive_factor
            
            logger.info(f"Updated user tier", 
                       user_id=user_id,
                       old_tier=old_tier.value,
                       new_tier=new_tier.value)
            
            return True
        
        return False
    
    def reset_user_violations(self, user_id: str) -> bool:
        """Reset violation count for a user."""
        if user_id in self.user_states:
            state = self.user_states[user_id]
            state.violation_count = 0
            state.is_blocked = False
            state.block_until = None
            
            # Reset adaptive factor
            config = self.tier_configs[state.tier]
            state.adaptive_factor = config.adaptive_factor
            
            logger.info(f"Reset user violations", user_id=user_id)
            return True
        
        return False
    
    async def shutdown(self):
        """Shutdown the rate limiter."""
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Rate limiter shutdown complete")

# Global rate limiter instance
rate_limiter: Optional[IntelligentRateLimiter] = None

def get_rate_limiter() -> IntelligentRateLimiter:
    """Get global rate limiter instance."""
    global rate_limiter
    if rate_limiter is None:
        rate_limiter = IntelligentRateLimiter()
    return rate_limiter

async def shutdown_rate_limiter():
    """Shutdown global rate limiter."""
    global rate_limiter
    if rate_limiter:
        await rate_limiter.shutdown()
        rate_limiter = None

