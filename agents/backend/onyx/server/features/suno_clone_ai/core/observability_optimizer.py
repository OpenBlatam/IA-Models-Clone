"""
Advanced Observability Optimizations

Optimizations for:
- Distributed tracing
- Metrics aggregation
- Log aggregation
- APM integration
- Alerting
"""

import logging
import time
from typing import Optional, Dict, Any, List
from functools import wraps
from contextlib import contextmanager
import uuid

logger = logging.getLogger(__name__)


class DistributedTracer:
    """Distributed tracing optimization."""
    
    def __init__(self):
        """Initialize distributed tracer."""
        self.traces: Dict[str, Dict[str, Any]] = {}
        self.current_trace_id: Optional[str] = None
        self.current_span_id: Optional[str] = None
    
    def start_trace(self, operation: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Start new trace.
        
        Args:
            operation: Operation name
            metadata: Optional metadata
            
        Returns:
            Trace ID
        """
        trace_id = str(uuid.uuid4())
        span_id = str(uuid.uuid4())
        
        self.traces[trace_id] = {
            'operation': operation,
            'start_time': time.time(),
            'spans': [],
            'metadata': metadata or {}
        }
        
        self.current_trace_id = trace_id
        self.current_span_id = span_id
        
        return trace_id
    
    def start_span(self, name: str, parent_span_id: Optional[str] = None) -> str:
        """
        Start new span.
        
        Args:
            name: Span name
            parent_span_id: Parent span ID
            
        Returns:
            Span ID
        """
        span_id = str(uuid.uuid4())
        
        if self.current_trace_id:
            span = {
                'id': span_id,
                'name': name,
                'parent_id': parent_span_id or self.current_span_id,
                'start_time': time.time()
            }
            
            self.traces[self.current_trace_id]['spans'].append(span)
            self.current_span_id = span_id
        
        return span_id
    
    def end_span(self, span_id: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        End span.
        
        Args:
            span_id: Span ID
            metadata: Optional metadata
        """
        if not self.current_trace_id:
            return
        
        trace = self.traces[self.current_trace_id]
        for span in trace['spans']:
            if span['id'] == span_id:
                span['end_time'] = time.time()
                span['duration'] = span['end_time'] - span['start_time']
                if metadata:
                    span['metadata'] = metadata
                break
    
    def end_trace(self, trace_id: str) -> Dict[str, Any]:
        """
        End trace and get results.
        
        Args:
            trace_id: Trace ID
            
        Returns:
            Trace results
        """
        if trace_id not in self.traces:
            return {}
        
        trace = self.traces[trace_id]
        trace['end_time'] = time.time()
        trace['duration'] = trace['end_time'] - trace['start_time']
        
        self.current_trace_id = None
        self.current_span_id = None
        
        return trace


class MetricsAggregator:
    """Metrics aggregation optimization."""
    
    def __init__(self, aggregation_window: int = 60):
        """
        Initialize metrics aggregator.
        
        Args:
            aggregation_window: Aggregation window in seconds
        """
        self.window = aggregation_window
        self.metrics: Dict[str, List[Dict[str, Any]]] = {}
    
    def record_metric(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Record metric.
        
        Args:
            name: Metric name
            value: Metric value
            tags: Optional tags
        """
        if name not in self.metrics:
            self.metrics[name] = []
        
        self.metrics[name].append({
            'value': value,
            'tags': tags or {},
            'timestamp': time.time()
        })
        
        # Clean old metrics
        cutoff = time.time() - self.window
        self.metrics[name] = [
            m for m in self.metrics[name]
            if m['timestamp'] > cutoff
        ]
    
    def get_aggregated(self, name: str) -> Optional[Dict[str, float]]:
        """
        Get aggregated metrics.
        
        Args:
            name: Metric name
            
        Returns:
            Aggregated statistics
        """
        if name not in self.metrics or not self.metrics[name]:
            return None
        
        values = [m['value'] for m in self.metrics[name]]
        
        return {
            'count': len(values),
            'sum': sum(values),
            'avg': sum(values) / len(values),
            'min': min(values),
            'max': max(values)
        }


class AlertManager:
    """Alert management optimization."""
    
    def __init__(self):
        """Initialize alert manager."""
        self.alerts: List[Dict[str, Any]] = []
        self.thresholds: Dict[str, float] = {}
        self.handlers: Dict[str, callable] = {}
    
    def set_threshold(self, metric: str, threshold: float, operator: str = ">") -> None:
        """
        Set alert threshold.
        
        Args:
            metric: Metric name
            threshold: Threshold value
            operator: Comparison operator (>, <, >=, <=)
        """
        self.thresholds[metric] = {'value': threshold, 'operator': operator}
    
    def register_handler(self, alert_type: str, handler: callable) -> None:
        """
        Register alert handler.
        
        Args:
            alert_type: Alert type
            handler: Handler function
        """
        self.handlers[alert_type] = handler
    
    def check_metric(self, metric: str, value: float) -> bool:
        """
        Check metric against threshold.
        
        Args:
            metric: Metric name
            value: Metric value
            
        Returns:
            True if alert should be triggered
        """
        if metric not in self.thresholds:
            return False
        
        threshold = self.thresholds[metric]
        operator = threshold['operator']
        threshold_value = threshold['value']
        
        if operator == ">":
            triggered = value > threshold_value
        elif operator == "<":
            triggered = value < threshold_value
        elif operator == ">=":
            triggered = value >= threshold_value
        elif operator == "<=":
            triggered = value <= threshold_value
        else:
            return False
        
        if triggered:
            self._trigger_alert(metric, value, threshold_value)
        
        return triggered
    
    def _trigger_alert(self, metric: str, value: float, threshold: float) -> None:
        """Trigger alert."""
        alert = {
            'metric': metric,
            'value': value,
            'threshold': threshold,
            'timestamp': time.time()
        }
        
        self.alerts.append(alert)
        
        # Call handler if registered
        if metric in self.handlers:
            try:
                self.handlers[metric](alert)
            except Exception as e:
                logger.error(f"Alert handler error: {e}")


class APMIntegration:
    """APM (Application Performance Monitoring) integration."""
    
    def __init__(self, apm_type: str = "custom"):
        """
        Initialize APM integration.
        
        Args:
            apm_type: APM type (datadog, newrelic, custom)
        """
        self.apm_type = apm_type
        self.transactions: List[Dict[str, Any]] = []
    
    def start_transaction(self, name: str) -> str:
        """
        Start APM transaction.
        
        Args:
            name: Transaction name
            
        Returns:
            Transaction ID
        """
        transaction_id = str(uuid.uuid4())
        
        self.transactions.append({
            'id': transaction_id,
            'name': name,
            'start_time': time.time(),
            'spans': []
        })
        
        return transaction_id
    
    def add_span(
        self,
        transaction_id: str,
        name: str,
        duration: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add span to transaction.
        
        Args:
            transaction_id: Transaction ID
            name: Span name
            duration: Duration in seconds
            metadata: Optional metadata
        """
        for transaction in self.transactions:
            if transaction['id'] == transaction_id:
                transaction['spans'].append({
                    'name': name,
                    'duration': duration,
                    'metadata': metadata or {}
                })
                break
    
    def end_transaction(self, transaction_id: str) -> Dict[str, Any]:
        """
        End transaction.
        
        Args:
            transaction_id: Transaction ID
            
        Returns:
            Transaction data
        """
        for i, transaction in enumerate(self.transactions):
            if transaction['id'] == transaction_id:
                transaction['end_time'] = time.time()
                transaction['duration'] = transaction['end_time'] - transaction['start_time']
                
                result = transaction.copy()
                del self.transactions[i]
                
                return result
        
        return {}








