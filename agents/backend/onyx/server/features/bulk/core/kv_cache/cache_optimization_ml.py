"""
ML-based optimization for KV cache.

This module provides machine learning capabilities for optimizing
cache behavior and performance.
"""

import time
import threading
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import random


class OptimizationTarget(Enum):
    """Optimization targets."""
    HIT_RATE = "hit_rate"
    LATENCY = "latency"
    MEMORY_USAGE = "memory_usage"
    THROUGHPUT = "throughput"
    COST = "cost"


@dataclass
class FeatureVector:
    """Feature vector for ML model."""
    hit_rate: float
    miss_rate: float
    cache_size: int
    memory_usage: float
    avg_latency: float
    throughput: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class Prediction:
    """ML prediction result."""
    target: OptimizationTarget
    predicted_value: float
    confidence: float
    recommended_action: str
    expected_improvement: float


class MLOptimizer:
    """ML-based cache optimizer."""
    
    def __init__(self, cache: Any):
        self.cache = cache
        self._training_data: List[Tuple[FeatureVector, float]] = []
        self._model_weights: Dict[str, float] = {}
        self._lock = threading.Lock()
        self._initialize_model()
        
    def _initialize_model(self) -> None:
        """Initialize ML model with default weights."""
        # Simplified linear model weights
        self._model_weights = {
            'hit_rate': 0.3,
            'miss_rate': -0.2,
            'cache_size': 0.1,
            'memory_usage': -0.15,
            'avg_latency': -0.1,
            'throughput': 0.25,
            'bias': 0.5
        }
        
    def extract_features(self) -> FeatureVector:
        """Extract features from current cache state."""
        hit_rate = 0.0
        miss_rate = 0.0
        cache_size = 0
        memory_usage = 0.0
        avg_latency = 0.0
        throughput = 0.0
        
        if hasattr(self.cache, 'stats'):
            stats = self.cache.stats
            hit_rate = getattr(stats, 'hit_rate', 0.0)
            miss_rate = getattr(stats, 'miss_rate', 0.0)
            
        if hasattr(self.cache, '_cache'):
            cache_size = len(self.cache._cache)
            if hasattr(self.cache, 'max_size'):
                max_size = self.cache.max_size
                memory_usage = cache_size / max_size if max_size > 0 else 0.0
                
        return FeatureVector(
            hit_rate=hit_rate,
            miss_rate=miss_rate,
            cache_size=cache_size,
            memory_usage=memory_usage,
            avg_latency=avg_latency,
            throughput=throughput
        )
        
    def predict(self, target: OptimizationTarget) -> Prediction:
        """Make prediction for optimization target."""
        features = self.extract_features()
        
        # Simple linear model prediction
        score = self._model_weights['bias']
        score += features.hit_rate * self._model_weights['hit_rate']
        score += features.miss_rate * self._model_weights['miss_rate']
        score += (features.cache_size / 1000.0) * self._model_weights['cache_size']
        score += features.memory_usage * self._model_weights['memory_usage']
        score += features.avg_latency * self._model_weights['avg_latency']
        score += features.throughput * self._model_weights['throughput']
        
        # Normalize to 0-1 range
        predicted_value = max(0.0, min(1.0, score))
        
        # Determine recommended action
        if target == OptimizationTarget.HIT_RATE:
            if features.hit_rate < 0.7:
                action = "increase_cache_size"
                improvement = 0.15
            else:
                action = "maintain"
                improvement = 0.0
        elif target == OptimizationTarget.MEMORY_USAGE:
            if features.memory_usage > 0.9:
                action = "enable_compression"
                improvement = 0.3
            else:
                action = "maintain"
                improvement = 0.0
        else:
            action = "monitor"
            improvement = 0.0
            
        return Prediction(
            target=target,
            predicted_value=predicted_value,
            confidence=0.7,  # Simplified
            recommended_action=action,
            expected_improvement=improvement
        )
        
    def train(self, features: FeatureVector, target_value: float) -> None:
        """Train model with new data point."""
        with self._lock:
            self._training_data.append((features, target_value))
            
            # Keep only last 1000 training examples
            if len(self._training_data) > 1000:
                self._training_data = self._training_data[-1000:]
                
            # Simplified training - would use proper ML algorithm
            self._update_weights()
            
    def _update_weights(self) -> None:
        """Update model weights (simplified gradient descent)."""
        if len(self._training_data) < 10:
            return
            
        # Simplified weight update
        # In real implementation, would use proper ML algorithm
        learning_rate = 0.01
        
        for features, target in self._training_data[-10:]:
            prediction = self._predict_internal(features)
            error = target - prediction
            
            # Update weights
            self._model_weights['hit_rate'] += learning_rate * error * features.hit_rate
            self._model_weights['miss_rate'] += learning_rate * error * features.miss_rate
            self._model_weights['cache_size'] += learning_rate * error * (features.cache_size / 1000.0)
            self._model_weights['memory_usage'] += learning_rate * error * features.memory_usage
            self._model_weights['avg_latency'] += learning_rate * error * features.avg_latency
            self._model_weights['throughput'] += learning_rate * error * features.throughput
            self._model_weights['bias'] += learning_rate * error
            
    def _predict_internal(self, features: FeatureVector) -> float:
        """Internal prediction method."""
        score = self._model_weights['bias']
        score += features.hit_rate * self._model_weights['hit_rate']
        score += features.miss_rate * self._model_weights['miss_rate']
        score += (features.cache_size / 1000.0) * self._model_weights['cache_size']
        score += features.memory_usage * self._model_weights['memory_usage']
        score += features.avg_latency * self._model_weights['avg_latency']
        score += features.throughput * self._model_weights['throughput']
        return max(0.0, min(1.0, score))
        
    def optimize(self, target: OptimizationTarget) -> Dict[str, Any]:
        """Optimize cache for target."""
        prediction = self.predict(target)
        
        # Apply optimization based on recommendation
        if prediction.recommended_action == "increase_cache_size":
            if hasattr(self.cache, 'max_size'):
                current = self.cache.max_size
                new_size = int(current * 1.5)
                self.cache.max_size = new_size
                
        return {
            'target': target.value,
            'predicted_value': prediction.predicted_value,
            'confidence': prediction.confidence,
            'action_taken': prediction.recommended_action,
            'expected_improvement': prediction.expected_improvement
        }
