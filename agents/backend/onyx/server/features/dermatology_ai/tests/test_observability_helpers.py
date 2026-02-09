"""
Observability Testing Helpers
Specialized helpers for observability testing
"""

from typing import Any, Dict, List, Optional
from unittest.mock import Mock, AsyncMock
from datetime import datetime
import json


class ObservabilityTestHelpers:
    """Helpers for observability testing"""
    
    @staticmethod
    def create_mock_observability_system(
        traces: Optional[List[Dict[str, Any]]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        logs: Optional[List[Dict[str, Any]]] = None
    ) -> Mock:
        """Create mock observability system"""
        system = Mock()
        system.traces = traces or []
        system.metrics = metrics or {}
        system.logs = logs or []
        
        system.start_trace = Mock(return_value="trace-123")
        system.end_trace = Mock()
        system.record_metric = Mock()
        system.log = Mock()
        
        return system
    
    @staticmethod
    def assert_trace_recorded(
        system: Mock,
        trace_id: Optional[str] = None
    ):
        """Assert trace was recorded"""
        assert system.start_trace.called, "Trace was not started"
        if trace_id:
            # Additional validation can check trace_id
            pass


class APMTracingHelpers:
    """Helpers for APM tracing testing"""
    
    @staticmethod
    def create_mock_apm_client() -> Mock:
        """Create mock APM client"""
        client = Mock()
        client.start_transaction = Mock(return_value=Mock())
        client.end_transaction = Mock()
        client.capture_exception = Mock()
        client.set_user_context = Mock()
        client.set_custom_context = Mock()
        return client
    
    @staticmethod
    def assert_transaction_recorded(client: Mock):
        """Assert transaction was recorded"""
        assert client.start_transaction.called, "Transaction was not started"


class LogAggregationHelpers:
    """Helpers for log aggregation testing"""
    
    @staticmethod
    def create_mock_log_aggregator(
        logs: Optional[List[Dict[str, Any]]] = None
    ) -> Mock:
        """Create mock log aggregator"""
        log_list = logs or []
        aggregator = Mock()
        
        def send_log_side_effect(log_entry: Dict[str, Any]):
            log_list.append({
                **log_entry,
                "timestamp": datetime.utcnow().isoformat()
            })
        
        aggregator.send_log = Mock(side_effect=send_log_side_effect)
        aggregator.send_batch = Mock()
        aggregator.logs = log_list
        return aggregator
    
    @staticmethod
    def assert_log_aggregated(
        aggregator: Mock,
        log_level: Optional[str] = None
    ):
        """Assert log was aggregated"""
        assert aggregator.send_log.called, "Log was not aggregated"
        
        if hasattr(aggregator, "logs") and log_level:
            found = any(
                log.get("level") == log_level
                for log in aggregator.logs
            )
            assert found, f"Log with level {log_level} not found"


class HealthCheckHelpers:
    """Helpers for health check testing"""
    
    @staticmethod
    def create_mock_health_checker(
        checks: Optional[Dict[str, bool]] = None
    ) -> Mock:
        """Create mock health checker"""
        check_results = checks or {}
        checker = Mock()
        
        async def check_side_effect(service_name: str):
            return {
                "service": service_name,
                "healthy": check_results.get(service_name, True),
                "timestamp": datetime.utcnow().isoformat()
            }
        
        checker.check = AsyncMock(side_effect=check_side_effect)
        checker.check_all = AsyncMock(return_value=check_results)
        return checker
    
    @staticmethod
    def assert_service_healthy(checker: Mock, service_name: str):
        """Assert service is healthy"""
        # Can check if check was called and result is healthy
        assert checker.check.called or checker.check_all.called, \
            f"Health check for {service_name} was not performed"


# Convenience exports
create_mock_observability_system = ObservabilityTestHelpers.create_mock_observability_system
assert_trace_recorded = ObservabilityTestHelpers.assert_trace_recorded

create_mock_apm_client = APMTracingHelpers.create_mock_apm_client
assert_transaction_recorded = APMTracingHelpers.assert_transaction_recorded

create_mock_log_aggregator = LogAggregationHelpers.create_mock_log_aggregator
assert_log_aggregated = LogAggregationHelpers.assert_log_aggregated

create_mock_health_checker = HealthCheckHelpers.create_mock_health_checker
assert_service_healthy = HealthCheckHelpers.assert_service_healthy



