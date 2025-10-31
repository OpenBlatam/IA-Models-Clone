"""
Advanced Analytics and Insights for Ultra-Adaptive K/V Cache Engine
"""

import time
import statistics
from typing import Dict, Any, Optional, List
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import logging

try:
    from ultra_adaptive_kv_cache_engine import UltraAdaptiveKVCacheEngine
except ImportError:
    UltraAdaptiveKVCacheEngine = None

logger = logging.getLogger(__name__)


@dataclass
class TimeSeriesPoint:
    """Time series data point."""
    timestamp: float
    value: float
    metadata: Optional[Dict[str, Any]] = None


class PerformanceAnalytics:
    """Advanced performance analytics."""
    
    def __init__(self, engine: UltraAdaptiveKVCacheEngine, window_size: int = 1000):
        self.engine = engine
        self.window_size = window_size
        self.metrics_history = defaultdict(lambda: deque(maxlen=window_size))
        self.anomalies = deque(maxlen=100)
    
    def record_metric(self, name: str, value: float, metadata: Optional[Dict[str, Any]] = None):
        """Record a metric value."""
        point = TimeSeriesPoint(
            timestamp=time.time(),
            value=value,
            metadata=metadata
        )
        self.metrics_history[name].append(point)
        
        # Check for anomalies
        self._detect_anomaly(name, value)
    
    def _detect_anomaly(self, metric_name: str, value: float):
        """Detect anomalies in metrics."""
        history = list(self.metrics_history[metric_name])
        
        if len(history) < 10:
            return
        
        values = [p.value for p in history[-50:]]  # Last 50 points
        
        if not values:
            return
        
        mean = statistics.mean(values)
        stdev = statistics.stdev(values) if len(values) > 1 else 0
        
        if stdev > 0:
            z_score = abs((value - mean) / stdev)
            
            if z_score > 3:  # 3 standard deviations
                self.anomalies.append({
                    'metric': metric_name,
                    'value': value,
                    'mean': mean,
                    'z_score': z_score,
                    'timestamp': time.time()
                })
                logger.warning(f"Anomaly detected in {metric_name}: {value} (z-score: {z_score:.2f})")
    
    def get_trend(self, metric_name: str, window_minutes: int = 60) -> Dict[str, Any]:
        """Get trend analysis for a metric."""
        cutoff_time = time.time() - (window_minutes * 60)
        recent_points = [
            p for p in self.metrics_history[metric_name]
            if p.timestamp >= cutoff_time
        ]
        
        if len(recent_points) < 2:
            return {'error': 'insufficient_data'}
        
        values = [p.value for p in recent_points]
        
        # Calculate trend
        if len(values) >= 2:
            first_half = values[:len(values)//2]
            second_half = values[len(values)//2:]
            
            first_avg = statistics.mean(first_half)
            second_avg = statistics.mean(second_half)
            
            change = second_avg - first_avg
            change_percent = (change / first_avg * 100) if first_avg != 0 else 0
            
            return {
                'metric': metric_name,
                'window_minutes': window_minutes,
                'data_points': len(recent_points),
                'current_value': values[-1],
                'average': statistics.mean(values),
                'min': min(values),
                'max': max(values),
                'std_dev': statistics.stdev(values) if len(values) > 1 else 0,
                'trend': 'increasing' if change > 0 else 'decreasing',
                'change': change,
                'change_percent': change_percent,
                'slope': change / len(values)
            }
        
        return {'error': 'insufficient_data'}
    
    def get_correlation(self, metric1: str, metric2: str) -> Dict[str, Any]:
        """Calculate correlation between two metrics."""
        points1 = list(self.metrics_history[metric1])
        points2 = list(self.metrics_history[metric2])
        
        if len(points1) < 10 or len(points2) < 10:
            return {'error': 'insufficient_data'}
        
        # Align timestamps (simple approach)
        aligned_values1 = [p.value for p in points1[-100:]]
        aligned_values2 = [p.value for p in points2[-100:]]
        
        min_len = min(len(aligned_values1), len(aligned_values2))
        aligned_values1 = aligned_values1[:min_len]
        aligned_values2 = aligned_values2[:min_len]
        
        # Calculate correlation
        try:
            correlation = statistics.correlation(aligned_values1, aligned_values2)
        except:
            correlation = 0.0
        
        return {
            'metric1': metric1,
            'metric2': metric2,
            'correlation': correlation,
            'data_points': min_len,
            'interpretation': self._interpret_correlation(correlation)
        }
    
    def _interpret_correlation(self, corr: float) -> str:
        """Interpret correlation coefficient."""
        abs_corr = abs(corr)
        
        if abs_corr < 0.1:
            return "no correlation"
        elif abs_corr < 0.3:
            return "weak correlation"
        elif abs_corr < 0.7:
            return "moderate correlation"
        else:
            return "strong correlation"
    
    def get_insights(self) -> Dict[str, Any]:
        """Get actionable insights from analytics."""
        insights = {
            'timestamp': time.time(),
            'recommendations': [],
            'anomalies': list(self.anomalies)[-10:],
            'metrics_summary': {}
        }
        
        # Get current engine stats
        stats = self.engine.get_performance_stats()
        engine_stats = stats.get('engine_stats', {})
        
        # Check error rate
        error_rate = engine_stats.get('error_rate', 0)
        if error_rate > 0.05:
            insights['recommendations'].append({
                'type': 'error_rate',
                'severity': 'high',
                'message': f'Error rate is {error_rate*100:.1f}%. Consider reducing load or checking system health.',
                'action': 'reduce_throughput'
            })
        
        # Check memory usage
        memory_usage = stats.get('memory_usage', 0)
        if memory_usage > 0.9:
            insights['recommendations'].append({
                'type': 'memory',
                'severity': 'critical',
                'message': f'Memory usage is {memory_usage*100:.1f}%. Consider increasing compression or reducing cache size.',
                'action': 'increase_compression'
            })
        
        # Check cache hit rate
        cache_hit_rate = engine_stats.get('cache_hit_rate', 1.0)
        if cache_hit_rate < 0.5:
            insights['recommendations'].append({
                'type': 'cache',
                'severity': 'medium',
                'message': f'Cache hit rate is {cache_hit_rate*100:.1f}%. Consider increasing cache size or improving session reuse.',
                'action': 'increase_cache_size'
            })
        
        # Check response time trend
        response_time_trend = self.get_trend('response_time', 30)
        if response_time_trend and 'change_percent' in response_time_trend:
            if response_time_trend['change_percent'] > 50:
                insights['recommendations'].append({
                    'type': 'performance',
                    'severity': 'high',
                    'message': f'Response time has increased by {response_time_trend["change_percent"]:.1f}% in the last 30 minutes.',
                    'action': 'investigate_performance_degradation'
                })
        
        return insights


class UsageAnalytics:
    """Usage analytics and patterns."""
    
    def __init__(self, engine: UltraAdaptiveKVCacheEngine):
        self.engine = engine
        self.session_patterns = defaultdict(list)
        self.request_patterns = defaultdict(int)
        self.time_patterns = defaultdict(int)
    
    def record_request(self, request: Dict[str, Any]):
        """Record request for pattern analysis."""
        session_id = request.get('session_id', 'unknown')
        text_length = len(request.get('text', ''))
        hour = datetime.now().hour
        
        self.session_patterns[session_id].append({
            'timestamp': time.time(),
            'text_length': text_length,
            'max_length': request.get('max_length', 100)
        })
        
        # Track request patterns
        length_category = 'short' if text_length < 100 else 'medium' if text_length < 1000 else 'long'
        self.request_patterns[length_category] += 1
        
        # Track time patterns
        time_category = 'morning' if 6 <= hour < 12 else 'afternoon' if 12 <= hour < 18 else 'evening' if 18 <= hour < 24 else 'night'
        self.time_patterns[time_category] += 1
    
    def get_usage_patterns(self) -> Dict[str, Any]:
        """Get usage patterns analysis."""
        return {
            'sessions': {
                'total': len(self.session_patterns),
                'active_sessions': len([s for s in self.session_patterns.values() if len(s) > 0]),
                'avg_requests_per_session': statistics.mean([len(s) for s in self.session_patterns.values()]) if self.session_patterns else 0
            },
            'request_sizes': dict(self.request_patterns),
            'time_distribution': dict(self.time_patterns)
        }
    
    def get_session_analysis(self, session_id: str) -> Dict[str, Any]:
        """Analyze a specific session."""
        if session_id not in self.session_patterns:
            return {'error': 'session_not_found'}
        
        session_requests = self.session_patterns[session_id]
        
        if not session_requests:
            return {'error': 'no_requests'}
        
        text_lengths = [r['text_length'] for r in session_requests]
        max_lengths = [r['max_length'] for r in session_requests]
        
        return {
            'session_id': session_id,
            'total_requests': len(session_requests),
            'avg_text_length': statistics.mean(text_lengths),
            'avg_max_length': statistics.mean(max_lengths),
            'first_request': min(r['timestamp'] for r in session_requests),
            'last_request': max(r['timestamp'] for r in session_requests),
            'duration': max(r['timestamp'] for r in session_requests) - min(r['timestamp'] for r in session_requests)
        }


class CostAnalytics:
    """Cost analysis and optimization."""
    
    def __init__(self, engine: UltraAdaptiveKVCacheEngine):
        self.engine = engine
        self.cost_history = deque(maxlen=1000)
    
    def calculate_cost(self, tokens_processed: int, cost_per_1k_tokens: float = 0.01) -> Dict[str, Any]:
        """Calculate cost for processing."""
        cost = (tokens_processed / 1000.0) * cost_per_1k_tokens
        
        cost_record = {
            'timestamp': time.time(),
            'tokens': tokens_processed,
            'cost': cost,
            'cost_per_1k_tokens': cost_per_1k_tokens
        }
        
        self.cost_history.append(cost_record)
        
        return cost_record
    
    def get_cost_summary(self, window_hours: int = 24) -> Dict[str, Any]:
        """Get cost summary for time window."""
        cutoff_time = time.time() - (window_hours * 3600)
        recent_costs = [c for c in self.cost_history if c['timestamp'] >= cutoff_time]
        
        if not recent_costs:
            return {'error': 'no_data'}
        
        total_tokens = sum(c['tokens'] for c in recent_costs)
        total_cost = sum(c['cost'] for c in recent_costs)
        
        return {
            'window_hours': window_hours,
            'total_requests': len(recent_costs),
            'total_tokens': total_tokens,
            'total_cost': total_cost,
            'avg_cost_per_request': total_cost / len(recent_costs) if recent_costs else 0,
            'tokens_per_hour': total_tokens / window_hours,
            'cost_per_hour': total_cost / window_hours
        }
    
    def estimate_savings_from_cache(self, cache_hit_rate: float, avg_cost_per_request: float) -> Dict[str, Any]:
        """Estimate cost savings from caching."""
        savings_rate = cache_hit_rate
        estimated_savings = avg_cost_per_request * savings_rate
        
        return {
            'cache_hit_rate': cache_hit_rate,
            'estimated_savings_per_request': estimated_savings,
            'savings_percentage': savings_rate * 100
        }

