"""
Metrics Testing Helpers
Specialized helpers for metrics and monitoring testing
"""

from typing import Any, Dict, List, Optional
from unittest.mock import Mock
from datetime import datetime, timedelta
from collections import defaultdict


class MetricsTestHelpers:
    """Helpers for metrics testing"""
    
    @staticmethod
    def create_mock_metrics_collector(
        initial_metrics: Optional[Dict[str, Any]] = None
    ) -> Mock:
        """Create mock metrics collector"""
        metrics = initial_metrics or {}
        collector = Mock()
        
        def increment_side_effect(name: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None):
            if name not in metrics:
                metrics[name] = 0.0
            metrics[name] += value
        
        def gauge_side_effect(name: str, value: float, tags: Optional[Dict[str, str]] = None):
            metrics[name] = value
        
        def histogram_side_effect(name: str, value: float, tags: Optional[Dict[str, str]] = None):
            if name not in metrics:
                metrics[name] = []
            metrics[name].append(value)
        
        collector.increment = Mock(side_effect=increment_side_effect)
        collector.gauge = Mock(side_effect=gauge_side_effect)
        collector.histogram = Mock(side_effect=histogram_side_effect)
        collector.get_metrics = Mock(return_value=metrics)
        collector.metrics = metrics
        return collector
    
    @staticmethod
    def assert_metric_recorded(
        collector: Mock,
        metric_name: str,
        expected_value: Optional[float] = None
    ):
        """Assert metric was recorded"""
        if hasattr(collector, "metrics"):
            assert metric_name in collector.metrics, f"Metric {metric_name} was not recorded"
            if expected_value is not None:
                assert collector.metrics[metric_name] == expected_value, \
                    f"Metric {metric_name} is {collector.metrics[metric_name]}, expected {expected_value}"
        else:
            # Check if increment, gauge, or histogram was called
            assert collector.increment.called or collector.gauge.called or collector.histogram.called, \
                "No metrics were recorded"
    
    @staticmethod
    def assert_metric_incremented(collector: Mock, metric_name: str):
        """Assert metric was incremented"""
        assert collector.increment.called, f"Metric {metric_name} was not incremented"


class PrometheusHelpers:
    """Helpers for Prometheus metrics testing"""
    
    @staticmethod
    def create_mock_prometheus_client() -> Mock:
        """Create mock Prometheus client"""
        client = Mock()
        client.Counter = Mock(return_value=Mock())
        client.Gauge = Mock(return_value=Mock())
        client.Histogram = Mock(return_value=Mock())
        client.Summary = Mock(return_value=Mock())
        return client
    
    @staticmethod
    def assert_prometheus_metric_exists(
        client: Mock,
        metric_name: str,
        metric_type: str = "Counter"
    ):
        """Assert Prometheus metric exists"""
        metric_class = getattr(client, metric_type)
        assert metric_class.called, f"Prometheus {metric_type} {metric_name} was not created"


class TracingHelpers:
    """Helpers for distributed tracing testing"""
    
    @staticmethod
    def create_mock_tracer(
        spans: Optional[List[Dict[str, Any]]] = None
    ) -> Mock:
        """Create mock tracer"""
        span_list = spans or []
        tracer = Mock()
        
        def start_span_side_effect(name: str, **kwargs):
            span = Mock()
            span.name = name
            span.start_time = datetime.utcnow()
            span.end_time = None
            span.tags = kwargs.get("tags", {})
            span.finish = Mock(side_effect=lambda: setattr(span, "end_time", datetime.utcnow()))
            span_list.append(span)
            return span
        
        tracer.start_span = Mock(side_effect=start_span_side_effect)
        tracer.spans = span_list
        return tracer
    
    @staticmethod
    def assert_span_created(
        tracer: Mock,
        span_name: str
    ):
        """Assert span was created"""
        assert tracer.start_span.called, "Span was not created"
        
        if hasattr(tracer, "spans"):
            matching = [s for s in tracer.spans if s.name == span_name]
            assert len(matching) > 0, f"Span {span_name} was not created"
    
    @staticmethod
    def assert_span_finished(span: Mock):
        """Assert span was finished"""
        assert span.finish.called, "Span was not finished"
        assert span.end_time is not None, "Span end_time is not set"


class LoggingHelpers:
    """Helpers for logging testing"""
    
    @staticmethod
    def create_mock_logger(
        log_entries: Optional[List[Dict[str, Any]]] = None
    ) -> Mock:
        """Create mock logger with log tracking"""
        entries = log_entries or []
        logger = Mock()
        
        def log_side_effect(level: str, message: str, **kwargs):
            entries.append({
                "level": level,
                "message": message,
                "timestamp": datetime.utcnow().isoformat(),
                "extra": kwargs
            })
        
        logger.debug = Mock(side_effect=lambda msg, **kw: log_side_effect("DEBUG", msg, **kw))
        logger.info = Mock(side_effect=lambda msg, **kw: log_side_effect("INFO", msg, **kw))
        logger.warning = Mock(side_effect=lambda msg, **kw: log_side_effect("WARNING", msg, **kw))
        logger.error = Mock(side_effect=lambda msg, **kw: log_side_effect("ERROR", msg, **kw))
        logger.critical = Mock(side_effect=lambda msg, **kw: log_side_effect("CRITICAL", msg, **kw))
        logger.entries = entries
        return logger
    
    @staticmethod
    def assert_logged(
        logger: Mock,
        level: str,
        message_pattern: Optional[str] = None
    ):
        """Assert log entry was created"""
        method = getattr(logger, level.lower())
        assert method.called, f"Log level {level} was not called"
        
        if hasattr(logger, "entries") and message_pattern:
            found = any(
                message_pattern in entry["message"]
                for entry in logger.entries
                if entry["level"] == level
            )
            assert found, f"Log message matching '{message_pattern}' not found"


# Convenience exports
create_mock_metrics_collector = MetricsTestHelpers.create_mock_metrics_collector
assert_metric_recorded = MetricsTestHelpers.assert_metric_recorded
assert_metric_incremented = MetricsTestHelpers.assert_metric_incremented

create_mock_prometheus_client = PrometheusHelpers.create_mock_prometheus_client
assert_prometheus_metric_exists = PrometheusHelpers.assert_prometheus_metric_exists

create_mock_tracer = TracingHelpers.create_mock_tracer
assert_span_created = TracingHelpers.assert_span_created
assert_span_finished = TracingHelpers.assert_span_finished

create_mock_logger = LoggingHelpers.create_mock_logger
assert_logged = LoggingHelpers.assert_logged



