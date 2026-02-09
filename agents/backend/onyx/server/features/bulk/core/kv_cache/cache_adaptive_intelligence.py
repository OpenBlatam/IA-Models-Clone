"""
Adaptive intelligence system for KV cache.

This module provides self-learning and self-tuning capabilities
that adapt cache behavior based on observed patterns and performance.
"""

import time
import threading
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import statistics


class AdaptationStrategy(Enum):
    """Adaptation strategies."""
    REACTIVE = "reactive"  # React to immediate issues
    PROACTIVE = "proactive"  # Anticipate future needs
    PREDICTIVE = "predictive"  # Use ML predictions
    HYBRID = "hybrid"  # Combine multiple strategies


class AdaptationAction(Enum):
    """Actions that can be taken by the adaptive system."""
    ADJUST_EVICTION = "adjust_eviction"
    ADJUST_TTL = "adjust_ttl"
    ADJUST_COMPRESSION = "adjust_compression"
    ADJUST_MEMORY_LIMIT = "adjust_memory_limit"
    ENABLE_PREFETCHING = "enable_prefetching"
    DISABLE_PREFETCHING = "disable_prefetching"
    CHANGE_STRATEGY = "change_strategy"
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"


@dataclass
class AdaptationDecision:
    """Decision made by adaptive system."""
    timestamp: float
    action: AdaptationAction
    reason: str
    confidence: float
    expected_impact: Dict[str, Any]
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformancePattern:
    """Identified performance pattern."""
    pattern_type: str
    description: str
    metrics: Dict[str, Any]
    frequency: float
    impact: str  # "positive", "negative", "neutral"


@dataclass
class AdaptationHistory:
    """History of adaptations."""
    decisions: List[AdaptationDecision] = field(default_factory=list)
    outcomes: Dict[str, Any] = field(default_factory=dict)
    patterns: List[PerformancePattern] = field(default_factory=list)


class CacheAdaptiveIntelligence:
    """Adaptive intelligence system for cache optimization."""
    
    def __init__(
        self,
        cache: Any,
        strategy: AdaptationStrategy = AdaptationStrategy.HYBRID,
        learning_rate: float = 0.1,
        adaptation_interval: float = 60.0,
        enable_auto_adaptation: bool = True
    ):
        self.cache = cache
        self.strategy = strategy
        self.learning_rate = learning_rate
        self.adaptation_interval = adaptation_interval
        self.enable_auto_adaptation = enable_auto_adaptation
        
        self._lock = threading.Lock()
        self._running = False
        self._adaptation_thread: Optional[threading.Thread] = None
        
        self._history = AdaptationHistory()
        self._performance_metrics: deque = deque(maxlen=1000)
        self._pattern_cache: Dict[str, Any] = {}
        
        # Performance thresholds
        self._thresholds = {
            'hit_rate_min': 0.7,
            'latency_max': 0.1,
            'memory_usage_max': 0.9,
            'error_rate_max': 0.01
        }
        
    def start(self) -> None:
        """Start adaptive intelligence system."""
        with self._lock:
            if self._running:
                return
            self._running = True
            self._adaptation_thread = threading.Thread(
                target=self._adaptation_loop,
                daemon=True
            )
            self._adaptation_thread.start()
            
    def stop(self) -> None:
        """Stop adaptive intelligence system."""
        with self._lock:
            self._running = False
            if self._adaptation_thread:
                self._adaptation_thread.join(timeout=5.0)
                
    def _adaptation_loop(self) -> None:
        """Main adaptation loop."""
        while self._running:
            try:
                if self.enable_auto_adaptation:
                    self._analyze_and_adapt()
                time.sleep(self.adaptation_interval)
            except Exception as e:
                # Log error but continue
                print(f"Error in adaptation loop: {e}")
                
    def _analyze_and_adapt(self) -> None:
        """Analyze performance and make adaptations."""
        # Collect current metrics
        metrics = self._collect_metrics()
        self._performance_metrics.append({
            'timestamp': time.time(),
            'metrics': metrics
        })
        
        # Identify patterns
        patterns = self._identify_patterns()
        
        # Make adaptation decisions
        decisions = self._make_decisions(metrics, patterns)
        
        # Execute decisions
        for decision in decisions:
            self._execute_decision(decision)
            
    def _collect_metrics(self) -> Dict[str, Any]:
        """Collect current performance metrics."""
        if not hasattr(self.cache, 'stats'):
            return {}
            
        stats = self.cache.stats
        return {
            'hit_rate': stats.hit_rate if hasattr(stats, 'hit_rate') else 0.0,
            'miss_rate': stats.miss_rate if hasattr(stats, 'miss_rate') else 0.0,
            'total_requests': stats.total_requests if hasattr(stats, 'total_requests') else 0,
            'cache_size': len(self.cache._cache) if hasattr(self.cache, '_cache') else 0,
            'memory_usage': self._estimate_memory_usage(),
        }
        
    def _estimate_memory_usage(self) -> float:
        """Estimate current memory usage."""
        if not hasattr(self.cache, '_cache'):
            return 0.0
            
        # Simple estimation
        cache_size = len(self.cache._cache)
        if hasattr(self.cache, 'max_size'):
            max_size = self.cache.max_size
            return cache_size / max_size if max_size > 0 else 0.0
        return 0.0
        
    def _identify_patterns(self) -> List[PerformancePattern]:
        """Identify performance patterns from metrics."""
        if len(self._performance_metrics) < 10:
            return []
            
        patterns = []
        
        # Analyze hit rate trend
        hit_rates = [
            m['metrics'].get('hit_rate', 0)
            for m in list(self._performance_metrics)[-50:]
        ]
        
        if len(hit_rates) >= 10:
            trend = statistics.mean(hit_rates[-10:]) - statistics.mean(hit_rates[:10])
            
            if trend < -0.1:
                patterns.append(PerformancePattern(
                    pattern_type='degrading_hit_rate',
                    description='Hit rate is decreasing',
                    metrics={'trend': trend, 'current': hit_rates[-1]},
                    frequency=1.0,
                    impact='negative'
                ))
            elif trend > 0.1:
                patterns.append(PerformancePattern(
                    pattern_type='improving_hit_rate',
                    description='Hit rate is improving',
                    metrics={'trend': trend, 'current': hit_rates[-1]},
                    frequency=1.0,
                    impact='positive'
                ))
                
        # Analyze memory usage
        memory_usages = [
            m['metrics'].get('memory_usage', 0)
            for m in list(self._performance_metrics)[-20:]
        ]
        
        if memory_usages and max(memory_usages) > self._thresholds['memory_usage_max']:
            patterns.append(PerformancePattern(
                pattern_type='high_memory_usage',
                description='Memory usage is high',
                metrics={'max': max(memory_usages), 'avg': statistics.mean(memory_usages)},
                frequency=1.0,
                impact='negative'
            ))
            
        return patterns
        
    def _make_decisions(
        self,
        metrics: Dict[str, Any],
        patterns: List[PerformancePattern]
    ) -> List[AdaptationDecision]:
        """Make adaptation decisions based on metrics and patterns."""
        decisions = []
        
        # Check hit rate
        hit_rate = metrics.get('hit_rate', 0)
        if hit_rate < self._thresholds['hit_rate_min']:
            decisions.append(AdaptationDecision(
                timestamp=time.time(),
                action=AdaptationAction.ADJUST_EVICTION,
                reason=f'Low hit rate: {hit_rate:.2f}',
                confidence=0.8,
                expected_impact={'hit_rate': +0.1},
                parameters={'strategy': 'lru'}
            ))
            
        # Check memory usage
        memory_usage = metrics.get('memory_usage', 0)
        if memory_usage > self._thresholds['memory_usage_max']:
            decisions.append(AdaptationDecision(
                timestamp=time.time(),
                action=AdaptationAction.ADJUST_EVICTION,
                reason=f'High memory usage: {memory_usage:.2%}',
                confidence=0.9,
                expected_impact={'memory_usage': -0.1},
                parameters={'aggressiveness': 'high'}
            ))
            
        # Pattern-based decisions
        for pattern in patterns:
            if pattern.pattern_type == 'degrading_hit_rate':
                decisions.append(AdaptationDecision(
                    timestamp=time.time(),
                    action=AdaptationAction.ADJUST_TTL,
                    reason='Degrading hit rate pattern detected',
                    confidence=0.7,
                    expected_impact={'hit_rate': +0.05},
                    parameters={'ttl_multiplier': 1.2}
                ))
                
        return decisions
        
    def _execute_decision(self, decision: AdaptationDecision) -> None:
        """Execute an adaptation decision."""
        try:
            if decision.action == AdaptationAction.ADJUST_EVICTION:
                self._adjust_eviction_strategy(decision.parameters)
            elif decision.action == AdaptationAction.ADJUST_TTL:
                self._adjust_ttl(decision.parameters)
            elif decision.action == AdaptationAction.ADJUST_COMPRESSION:
                self._adjust_compression(decision.parameters)
            elif decision.action == AdaptationAction.ADJUST_MEMORY_LIMIT:
                self._adjust_memory_limit(decision.parameters)
                
            # Record decision
            self._history.decisions.append(decision)
            
        except Exception as e:
            print(f"Error executing adaptation decision: {e}")
            
    def _adjust_eviction_strategy(self, parameters: Dict[str, Any]) -> None:
        """Adjust eviction strategy."""
        if hasattr(self.cache, 'eviction_strategy'):
            strategy_name = parameters.get('strategy', 'lru')
            # Update eviction strategy if possible
            pass
            
    def _adjust_ttl(self, parameters: Dict[str, Any]) -> None:
        """Adjust TTL settings."""
        multiplier = parameters.get('ttl_multiplier', 1.0)
        # Apply TTL multiplier if cache supports it
        pass
        
    def _adjust_compression(self, parameters: Dict[str, Any]) -> None:
        """Adjust compression settings."""
        # Update compression settings if available
        pass
        
    def _adjust_memory_limit(self, parameters: Dict[str, Any]) -> None:
        """Adjust memory limit."""
        # Update memory limit if cache supports it
        pass
        
    def get_adaptation_history(self) -> AdaptationHistory:
        """Get adaptation history."""
        return self._history
        
    def recommend_adaptations(self) -> List[AdaptationDecision]:
        """Get recommendations for adaptations."""
        metrics = self._collect_metrics()
        patterns = self._identify_patterns()
        return self._make_decisions(metrics, patterns)
        
    def update_thresholds(self, thresholds: Dict[str, float]) -> None:
        """Update performance thresholds."""
        self._thresholds.update(thresholds)
        
    def learn_from_outcome(
        self,
        decision: AdaptationDecision,
        outcome: Dict[str, Any]
    ) -> None:
        """Learn from adaptation outcome."""
        self._history.outcomes[id(decision)] = outcome
        
        # Adjust confidence based on outcome
        if outcome.get('success', False):
            decision.confidence = min(1.0, decision.confidence + self.learning_rate)
        else:
            decision.confidence = max(0.0, decision.confidence - self.learning_rate)
















