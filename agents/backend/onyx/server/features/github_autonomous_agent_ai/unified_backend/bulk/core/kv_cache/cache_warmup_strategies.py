"""
Advanced warmup strategies for KV cache.

This module provides intelligent cache warming strategies to preload
frequently accessed data before it's needed.
"""

import time
import threading
from typing import Dict, Any, List, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque


class WarmupStrategy(Enum):
    """Warmup strategies."""
    STATIC = "static"  # Predefined list of keys
    PATTERN_BASED = "pattern_based"  # Based on key patterns
    ACCESS_HISTORY = "access_history"  # Based on historical access
    PREDICTIVE = "predictive"  # ML-based prediction
    PROGRESSIVE = "progressive"  # Gradual warmup
    ON_DEMAND = "on_demand"  # Warmup on first access


@dataclass
class WarmupConfig:
    """Warmup configuration."""
    strategy: WarmupStrategy
    keys: Optional[List[str]] = None
    key_pattern: Optional[str] = None
    data_loader: Optional[Callable[[str], Any]] = None
    max_keys: int = 1000
    batch_size: int = 100
    priority_keys: Optional[List[str]] = None


@dataclass
class WarmupResult:
    """Result of warmup operation."""
    keys_warmed: int
    keys_failed: int
    duration: float
    strategy_used: WarmupStrategy
    timestamp: float = field(default_factory=time.time)


class CacheWarmer:
    """Cache warmer with multiple strategies."""
    
    def __init__(self, cache: Any, config: WarmupConfig):
        self.cache = cache
        self.config = config
        self._access_history: deque = deque(maxlen=10000)
        self._lock = threading.Lock()
        
    def warmup(self) -> WarmupResult:
        """Perform cache warmup."""
        start_time = time.time()
        keys_warmed = 0
        keys_failed = 0
        
        if self.config.strategy == WarmupStrategy.STATIC:
            keys_warmed, keys_failed = self._warmup_static()
        elif self.config.strategy == WarmupStrategy.PATTERN_BASED:
            keys_warmed, keys_failed = self._warmup_pattern()
        elif self.config.strategy == WarmupStrategy.ACCESS_HISTORY:
            keys_warmed, keys_failed = self._warmup_history()
        elif self.config.strategy == WarmupStrategy.PROGRESSIVE:
            keys_warmed, keys_failed = self._warmup_progressive()
        else:
            keys_warmed, keys_failed = self._warmup_static()
            
        duration = time.time() - start_time
        
        return WarmupResult(
            keys_warmed=keys_warmed,
            keys_failed=keys_failed,
            duration=duration,
            strategy_used=self.config.strategy
        )
        
    def _warmup_static(self) -> tuple:
        """Warmup using static key list."""
        keys_warmed = 0
        keys_failed = 0
        
        if not self.config.keys:
            return keys_warmed, keys_failed
            
        # Warmup priority keys first
        priority_keys = self.config.priority_keys or []
        for key in priority_keys:
            if self._warmup_key(key):
                keys_warmed += 1
            else:
                keys_failed += 1
                
        # Warmup remaining keys
        remaining_keys = [k for k in self.config.keys if k not in priority_keys]
        for key in remaining_keys[:self.config.max_keys]:
            if self._warmup_key(key):
                keys_warmed += 1
            else:
                keys_failed += 1
                
        return keys_warmed, keys_failed
        
    def _warmup_pattern(self) -> tuple:
        """Warmup using key patterns."""
        keys_warmed = 0
        keys_failed = 0
        
        if not self.config.key_pattern or not self.config.data_loader:
            return keys_warmed, keys_failed
            
        # Generate keys from pattern
        import re
        pattern = re.compile(self.config.key_pattern)
        
        # Try to find keys matching pattern
        # This is simplified - would need actual key discovery
        # For now, generate keys based on pattern
        keys = []
        for i in range(self.config.max_keys):
            key = self.config.key_pattern.replace('*', str(i))
            if pattern.match(key):
                keys.append(key)
                
        for key in keys:
            if self._warmup_key(key):
                keys_warmed += 1
            else:
                keys_failed += 1
                
        return keys_warmed, keys_failed
        
    def _warmup_history(self) -> tuple:
        """Warmup based on access history."""
        keys_warmed = 0
        keys_failed = 0
        
        with self._lock:
            # Get most frequently accessed keys
            access_counts = defaultdict(int)
            for key in self._access_history:
                access_counts[key] += 1
                
            # Sort by frequency
            sorted_keys = sorted(
                access_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:self.config.max_keys]
            
        for key, _ in sorted_keys:
            if self._warmup_key(key):
                keys_warmed += 1
            else:
                keys_failed += 1
                
        return keys_warmed, keys_failed
        
    def _warmup_progressive(self) -> tuple:
        """Progressive warmup - warmup in batches."""
        keys_warmed = 0
        keys_failed = 0
        
        if not self.config.keys:
            return keys_warmed, keys_failed
            
        # Warmup in batches
        for i in range(0, min(len(self.config.keys), self.config.max_keys), self.config.batch_size):
            batch = self.config.keys[i:i + self.config.batch_size]
            
            for key in batch:
                if self._warmup_key(key):
                    keys_warmed += 1
                else:
                    keys_failed += 1
                    
            # Small delay between batches
            time.sleep(0.1)
            
        return keys_warmed, keys_failed
        
    def _warmup_key(self, key: str) -> bool:
        """Warmup a single key."""
        if not self.config.data_loader:
            return False
            
        try:
            value = self.config.data_loader(key)
            if value is not None:
                return self.cache.put(key, value)
            return False
        except Exception:
            return False
            
    def record_access(self, key: str) -> None:
        """Record key access for history-based warmup."""
        with self._lock:
            self._access_history.append(key)


class IntelligentWarmer:
    """Intelligent cache warmer with adaptive strategies."""
    
    def __init__(self, cache: Any):
        self.cache = cache
        self._warmer = None
        
    def configure(self, config: WarmupConfig) -> None:
        """Configure warmer."""
        self._warmer = CacheWarmer(self.cache, config)
        
    def warmup(self) -> WarmupResult:
        """Perform warmup."""
        if not self._warmer:
            return WarmupResult(
                keys_warmed=0,
                keys_failed=0,
                duration=0.0,
                strategy_used=WarmupStrategy.STATIC
            )
        return self._warmer.warmup()
        
    def auto_warmup(self, threshold_hit_rate: float = 0.5) -> None:
        """Automatically warmup if hit rate is low."""
        if hasattr(self.cache, 'stats'):
            hit_rate = getattr(self.cache.stats, 'hit_rate', 1.0)
            if hit_rate < threshold_hit_rate:
                # Trigger warmup based on access history
                config = WarmupConfig(
                    strategy=WarmupStrategy.ACCESS_HISTORY,
                    max_keys=1000
                )
                self.configure(config)
                self.warmup()



