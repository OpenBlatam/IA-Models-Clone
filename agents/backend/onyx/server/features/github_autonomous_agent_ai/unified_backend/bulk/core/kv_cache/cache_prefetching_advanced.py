"""
Advanced prefetching system for KV cache.

This module provides intelligent prefetching strategies based on
access patterns, predictions, and machine learning.
"""

import time
import threading
from typing import Dict, Any, List, Optional, Set, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import statistics


class PrefetchStrategy(Enum):
    """Prefetching strategies."""
    SEQUENTIAL = "sequential"  # Prefetch next items in sequence
    RANDOM = "random"  # Random prefetching
    ML_BASED = "ml_based"  # Machine learning based
    PATTERN_BASED = "pattern_based"  # Pattern-based prefetching
    ACCESS_PATTERN = "access_pattern"  # Based on access patterns
    CORRELATION = "correlation"  # Based on key correlations


@dataclass
class PrefetchPrediction:
    """Prediction for prefetching."""
    key: str
    confidence: float
    reason: str
    predicted_access_time: float


@dataclass
class AccessPattern:
    """Access pattern information."""
    key: str
    access_times: List[float]
    access_count: int
    last_access: float
    average_interval: Optional[float] = None
    
    def update(self) -> None:
        """Update pattern statistics."""
        if len(self.access_times) >= 2:
            intervals = [
                self.access_times[i] - self.access_times[i-1]
                for i in range(1, len(self.access_times))
            ]
            self.average_interval = statistics.mean(intervals) if intervals else None


class AdvancedCachePrefetcher:
    """Advanced prefetching system."""
    
    def __init__(
        self,
        cache: Any,
        strategy: PrefetchStrategy = PrefetchStrategy.ACCESS_PATTERN,
        prefetch_count: int = 5,
        enable_async_prefetch: bool = True
    ):
        self.cache = cache
        self.strategy = strategy
        self.prefetch_count = prefetch_count
        self.enable_async_prefetch = enable_async_prefetch
        
        self._access_patterns: Dict[str, AccessPattern] = {}
        self._key_correlations: Dict[str, Set[str]] = defaultdict(set)
        self._access_sequence: deque = deque(maxlen=1000)
        self._prefetch_queue: List[str] = []
        self._lock = threading.Lock()
        
    def record_access(self, key: str) -> None:
        """Record key access for pattern analysis."""
        current_time = time.time()
        
        with self._lock:
            if key not in self._access_patterns:
                self._access_patterns[key] = AccessPattern(
                    key=key,
                    access_times=[],
                    access_count=0,
                    last_access=current_time
                )
                
            pattern = self._access_patterns[key]
            pattern.access_times.append(current_time)
            pattern.access_count += 1
            pattern.last_access = current_time
            pattern.update()
            
            # Record in sequence
            self._access_sequence.append(key)
            
            # Update correlations
            if len(self._access_sequence) >= 2:
                prev_key = self._access_sequence[-2]
                self._key_correlations[prev_key].add(key)
                
    def predict_next_keys(self, current_key: str) -> List[PrefetchPrediction]:
        """Predict next keys that will be accessed."""
        predictions = []
        
        if self.strategy == PrefetchStrategy.SEQUENTIAL:
            predictions = self._predict_sequential(current_key)
        elif self.strategy == PrefetchStrategy.ACCESS_PATTERN:
            predictions = self._predict_access_pattern(current_key)
        elif self.strategy == PrefetchStrategy.CORRELATION:
            predictions = self._predict_correlation(current_key)
        elif self.strategy == PrefetchStrategy.PATTERN_BASED:
            predictions = self._predict_pattern_based(current_key)
            
        # Sort by confidence and return top N
        predictions.sort(key=lambda p: p.confidence, reverse=True)
        return predictions[:self.prefetch_count]
        
    def _predict_sequential(self, current_key: str) -> List[PrefetchPrediction]:
        """Predict sequential access."""
        predictions = []
        
        # Try to find next key in sequence
        if current_key.isdigit():
            next_key = str(int(current_key) + 1)
            predictions.append(PrefetchPrediction(
                key=next_key,
                confidence=0.7,
                reason="sequential",
                predicted_access_time=time.time() + 1.0
            ))
        elif '_' in current_key:
            parts = current_key.split('_')
            if len(parts) == 2 and parts[1].isdigit():
                next_num = int(parts[1]) + 1
                next_key = f"{parts[0]}_{next_num}"
                predictions.append(PrefetchPrediction(
                    key=next_key,
                    confidence=0.7,
                    reason="sequential",
                    predicted_access_time=time.time() + 1.0
                ))
                
        return predictions
        
    def _predict_access_pattern(self, current_key: str) -> List[PrefetchPrediction]:
        """Predict based on access patterns."""
        predictions = []
        
        pattern = self._access_patterns.get(current_key)
        if pattern and pattern.average_interval:
            # Predict keys that are accessed at similar intervals
            current_time = time.time()
            
            for key, key_pattern in self._access_patterns.items():
                if key == current_key:
                    continue
                    
                if key_pattern.average_interval:
                    # Check if this key is likely to be accessed soon
                    time_since_last = current_time - key_pattern.last_access
                    
                    # If we're close to the average interval, predict access
                    if abs(time_since_last - key_pattern.average_interval) < key_pattern.average_interval * 0.2:
                        confidence = 0.8 - (abs(time_since_last - key_pattern.average_interval) / key_pattern.average_interval)
                        predictions.append(PrefetchPrediction(
                            key=key,
                            confidence=max(0.0, confidence),
                            reason="access_pattern",
                            predicted_access_time=key_pattern.last_access + key_pattern.average_interval
                        ))
                        
        return predictions
        
    def _predict_correlation(self, current_key: str) -> List[PrefetchPrediction]:
        """Predict based on key correlations."""
        predictions = []
        
        correlated_keys = self._key_correlations.get(current_key, set())
        
        for key in correlated_keys:
            # Calculate correlation strength
            # Count how many times this key follows current_key
            correlation_count = 0
            sequence_list = list(self._access_sequence)
            
            for i in range(len(sequence_list) - 1):
                if sequence_list[i] == current_key and sequence_list[i+1] == key:
                    correlation_count += 1
                    
            if correlation_count > 0:
                confidence = min(1.0, correlation_count / 10.0)  # Normalize
                predictions.append(PrefetchPrediction(
                    key=key,
                    confidence=confidence,
                    reason="correlation",
                    predicted_access_time=time.time() + 0.5
                ))
                
        return predictions
        
    def _predict_pattern_based(self, current_key: str) -> List[PrefetchPrediction]:
        """Predict based on identified patterns."""
        # Combine multiple prediction methods
        predictions = []
        
        # Add sequential predictions
        predictions.extend(self._predict_sequential(current_key))
        
        # Add correlation predictions
        predictions.extend(self._predict_correlation(current_key))
        
        # Deduplicate by key, keeping highest confidence
        seen_keys = {}
        for pred in predictions:
            if pred.key not in seen_keys or pred.confidence > seen_keys[pred.key].confidence:
                seen_keys[pred.key] = pred
                
        return list(seen_keys.values())
        
    def prefetch(self, key: str, prefetch_function: Callable[[str], Any]) -> List[str]:
        """Prefetch keys based on predictions."""
        predictions = self.predict_next_keys(key)
        prefetched = []
        
        for prediction in predictions:
            if prediction.confidence > 0.5:  # Threshold
                try:
                    # Prefetch using provided function
                    prefetch_function(prediction.key)
                    prefetched.append(prediction.key)
                except Exception as e:
                    print(f"Prefetch failed for {prediction.key}: {e}")
                    
        return prefetched
        
    def get_access_pattern(self, key: str) -> Optional[AccessPattern]:
        """Get access pattern for a key."""
        return self._access_patterns.get(key)
        
    def get_correlations(self, key: str) -> Set[str]:
        """Get correlated keys."""
        return self._key_correlations.get(key, set())
        
    def get_prefetch_stats(self) -> Dict[str, Any]:
        """Get prefetching statistics."""
        total_patterns = len(self._access_patterns)
        total_correlations = sum(len(v) for v in self._key_correlations.values())
        
        return {
            'total_patterns': total_patterns,
            'total_correlations': total_correlations,
            'strategy': self.strategy.value,
            'prefetch_count': self.prefetch_count
        }














