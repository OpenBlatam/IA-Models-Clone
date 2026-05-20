"""
Monitoring and observability tests for copywriting service.
"""
import pytest
import time
import asyncio
from typing import Dict, Any, List
from unittest.mock import patch, Mock
import json

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.test_utils import TestDataFactory, MockAIService
from service import CopywritingService
from models import CopywritingInput


class TestMetricsCollection:
    """Test metrics collection and monitoring."""
    
    @pytest.fixture
    def service(self):
        """Create service instance for testing."""
        return CopywritingService()
    
    def test_request_metrics_collection(self, service):
        """Test collection of request metrics."""
        request = TestDataFactory.create_copywriting_input()
        
        # Mock metrics collection
        with patch('agents.backend.onyx.server.features.copywriting.service.metrics') as mock_metrics:
            mock_metrics.increment_request_count.return_value = None
            mock_metrics.record_request_duration.return_value = None
            mock_metrics.record_request_size.return_value = None
            
            # Simulate request processing
            with patch.object(service, '_call_ai_model') as mock_call:
                mock_call.return_value = {
                    "variants": [{"headline": "Test", "primary_text": "Content"}],
                    "model_used": "gpt-3.5-turbo",
                    "generation_time": 1.0,
                    "extra_metadata": {}
                }
                
                asyncio.run(service.generate_copywriting(request, "gpt-3.5-turbo"))
                
                # Verify metrics were collected
                mock_metrics.increment_request_count.assert_called_once()
                mock_metrics.record_request_duration.assert_called_once()
                mock_metrics.record_request_size.assert_called_once()
    
    def test_error_metrics_collection(self, service):
        """Test collection of error metrics."""
        request = TestDataFactory.create_sample_request()
        
        # Mock metrics collection
        with patch('agents.backend.onyx.server.features.copywriting.service.metrics') as mock_metrics:
            mock_metrics.increment_error_count.return_value = None
            mock_metrics.record_error_type.return_value = None
            
            # Simulate error
            with patch.object(service, '_call_ai_model') as mock_call:
                mock_call.side_effect = Exception("Test error")
                
                with pytest.raises(Exception):
                    asyncio.run(service.generate_copywriting(request, "gpt-3.5-turbo"))
                
                # Verify error metrics were collected
                mock_metrics.increment_error_count.assert_called_once()
                mock_metrics.record_error_type.assert_called_once()
    
    def test_performance_metrics_collection(self, service):
        """Test collection of performance metrics."""
        request = TestDataFactory.create_sample_request()
        
        # Mock metrics collection
        with patch('agents.backend.onyx.server.features.copywriting.service.metrics') as mock_metrics:
            mock_metrics.record_response_time.return_value = None
            mock_metrics.record_tokens_used.return_value = None
            mock_metrics.record_model_usage.return_value = None
            
            with patch.object(service, '_call_ai_model') as mock_call:
                mock_call.return_value = {
                    "variants": [{"headline": "Test", "primary_text": "Content"}],
                    "model_used": "gpt-3.5-turbo",
                    "generation_time": 2.5,
                    "extra_metadata": {"tokens_used": 150}
                }
                
                asyncio.run(service.generate_copywriting(request, "gpt-3.5-turbo"))
                
                # Verify performance metrics were collected
                mock_metrics.record_response_time.assert_called_once()
                mock_metrics.record_tokens_used.assert_called_once()
                mock_metrics.record_model_usage.assert_called_once()


class TestHealthChecks:
    """Test health check functionality."""
    
    @pytest.fixture
    def service(self):
        """Create service instance for testing."""
        return CopywritingService()
    
    def test_health_check_success(self, service):
        """Test successful health check."""
        # Mock health check dependencies
        with patch('agents.backend.onyx.server.features.copywriting.service.database') as mock_db:
            mock_db.is_healthy.return_value = True
            
            with patch('agents.backend.onyx.server.features.copywriting.service.cache') as mock_cache:
                mock_cache.is_healthy.return_value = True
                
                with patch('agents.backend.onyx.server.features.copywriting.service.ai_service') as mock_ai:
                    mock_ai.is_healthy.return_value = True
                    
                    health_status = service.health_check()
                    
                    assert health_status["status"] == "healthy"
                    assert health_status["database"] == "healthy"
                    assert health_status["cache"] == "healthy"
                    assert health_status["ai_service"] == "healthy"
    
    def test_health_check_failure(self, service):
        """Test health check with failures."""
        # Mock health check dependencies with failures
        with patch('agents.backend.onyx.server.features.copywriting.service.database') as mock_db:
            mock_db.is_healthy.return_value = False
            
            with patch('agents.backend.onyx.server.features.copywriting.service.cache') as mock_cache:
                mock_cache.is_healthy.return_value = True
                
                with patch('agents.backend.onyx.server.features.copywriting.service.ai_service') as mock_ai:
                    mock_ai.is_healthy.return_value = True
                    
                    health_status = service.health_check()
                    
                    assert health_status["status"] == "unhealthy"
                    assert health_status["database"] == "unhealthy"
                    assert health_status["cache"] == "healthy"
                    assert health_status["ai_service"] == "healthy"
    
    def test_health_check_partial_failure(self, service):
        """Test health check with partial failures."""
        # Mock health check dependencies with partial failures
        with patch('agents.backend.onyx.server.features.copywriting.service.database') as mock_db:
            mock_db.is_healthy.return_value = True
            
            with patch('agents.backend.onyx.server.features.copywriting.service.cache') as mock_cache:
                mock_cache.is_healthy.return_value = False
                
                with patch('agents.backend.onyx.server.features.copywriting.service.ai_service') as mock_ai:
                    mock_ai.is_healthy.return_value = True
                    
                    health_status = service.health_check()
                    
                    assert health_status["status"] == "degraded"
                    assert health_status["database"] == "healthy"
                    assert health_status["cache"] == "unhealthy"
                    assert health_status["ai_service"] == "healthy"


class TestLogging:
    """Test logging functionality."""
    
    @pytest.fixture
    def service(self):
        """Create service instance for testing."""
        return CopywritingService()
    
    def test_request_logging(self, service):
        """Test request logging."""
        request = TestDataFactory.create_sample_request()
        
        # Mock logger
        with patch('agents.backend.onyx.server.features.copywriting.service.logger') as mock_logger:
            with patch.object(service, '_call_ai_model') as mock_call:
                mock_call.return_value = {
                    "variants": [{"headline": "Test", "primary_text": "Content"}],
                    "model_used": "gpt-3.5-turbo",
                    "generation_time": 1.0,
                    "extra_metadata": {}
                }
                
                asyncio.run(service.generate_copywriting(request, "gpt-3.5-turbo"))
                
                # Verify logging calls
                mock_logger.info.assert_called()
                mock_logger.debug.assert_called()
    
    def test_error_logging(self, service):
        """Test error logging."""
        request = TestDataFactory.create_sample_request()
        
        # Mock logger
        with patch('agents.backend.onyx.server.features.copywriting.service.logger') as mock_logger:
            with patch.object(service, '_call_ai_model') as mock_call:
                mock_call.side_effect = Exception("Test error")
                
                with pytest.raises(Exception):
                    asyncio.run(service.generate_copywriting(request, "gpt-3.5-turbo"))
                
                # Verify error logging
                mock_logger.error.assert_called()
                mock_logger.exception.assert_called()
    
    def test_performance_logging(self, service):
        """Test performance logging."""
        request = TestDataFactory.create_sample_request()
        
        # Mock logger
        with patch('agents.backend.onyx.server.features.copywriting.service.logger') as mock_logger:
            with patch.object(service, '_call_ai_model') as mock_call:
                mock_call.return_value = {
                    "variants": [{"headline": "Test", "primary_text": "Content"}],
                    "model_used": "gpt-3.5-turbo",
                    "generation_time": 2.5,
                    "extra_metadata": {"tokens_used": 150}
                }
                
                asyncio.run(service.generate_copywriting(request, "gpt-3.5-turbo"))
                
                # Verify performance logging
                mock_logger.info.assert_called()
                # Check that performance metrics are logged
                logged_messages = [call[0][0] for call in mock_logger.info.call_args_list]
                assert any("generation_time" in msg for msg in logged_messages)


class TestTracing:
    """Test distributed tracing functionality."""
    
    @pytest.fixture
    def service(self):
        """Create service instance for testing."""
        return CopywritingService()
    
    def test_request_tracing(self, service):
        """Test request tracing."""
        request = TestDataFactory.create_sample_request()
        
        # Mock tracer
        with patch('agents.backend.onyx.server.features.copywriting.service.tracer') as mock_tracer:
            mock_span = Mock()
            mock_tracer.start_span.return_value.__enter__.return_value = mock_span
            
            with patch.object(service, '_call_ai_model') as mock_call:
                mock_call.return_value = {
                    "variants": [{"headline": "Test", "primary_text": "Content"}],
                    "model_used": "gpt-3.5-turbo",
                    "generation_time": 1.0,
                    "extra_metadata": {}
                }
                
                asyncio.run(service.generate_copywriting(request, "gpt-3.5-turbo"))
                
                # Verify tracing
                mock_tracer.start_span.assert_called()
                mock_span.set_tag.assert_called()
                mock_span.log.assert_called()
    
    def test_error_tracing(self, service):
        """Test error tracing."""
        request = TestDataFactory.create_sample_request()
        
        # Mock tracer
        with patch('agents.backend.onyx.server.features.copywriting.service.tracer') as mock_tracer:
            mock_span = Mock()
            mock_tracer.start_span.return_value.__enter__.return_value = mock_span
            
            with patch.object(service, '_call_ai_model') as mock_call:
                mock_call.side_effect = Exception("Test error")
                
                with pytest.raises(Exception):
                    asyncio.run(service.generate_copywriting(request, "gpt-3.5-turbo"))
                
                # Verify error tracing
                mock_span.set_tag.assert_called()
                mock_span.log.assert_called()
                mock_span.set_error.assert_called()


class TestAlerting:
    """Test alerting functionality."""
    
    @pytest.fixture
    def service(self):
        """Create service instance for testing."""
        return CopywritingService()
    
    def test_error_rate_alerting(self, service):
        """Test error rate alerting."""
        # Mock alerting system
        with patch('agents.backend.onyx.server.features.copywriting.service.alerter') as mock_alerter:
            # Simulate high error rate
            for i in range(10):
                request = TestDataFactory.create_sample_request()
                
                with patch.object(service, '_call_ai_model') as mock_call:
                    if i < 8:  # 80% error rate
                        mock_call.side_effect = Exception("Test error")
                    else:
                        mock_call.return_value = {
                            "variants": [{"headline": "Test", "primary_text": "Content"}],
                            "model_used": "gpt-3.5-turbo",
                            "generation_time": 1.0,
                            "extra_metadata": {}
                        }
                    
                    try:
                        asyncio.run(service.generate_copywriting(request, "gpt-3.5-turbo"))
                    except Exception:
                        pass
            
            # Verify alert was triggered
            mock_alerter.send_alert.assert_called()
    
    def test_performance_alerting(self, service):
        """Test performance alerting."""
        # Mock alerting system
        with patch('agents.backend.onyx.server.features.copywriting.service.alerter') as mock_alerter:
            # Simulate slow performance
            request = TestDataFactory.create_sample_request()
            
            with patch.object(service, '_call_ai_model') as mock_call:
                mock_call.return_value = {
                    "variants": [{"headline": "Test", "primary_text": "Content"}],
                    "model_used": "gpt-3.5-turbo",
                    "generation_time": 10.0,  # Very slow
                    "extra_metadata": {}
                }
                
                asyncio.run(service.generate_copywriting(request, "gpt-3.5-turbo"))
                
                # Verify performance alert was triggered
                mock_alerter.send_alert.assert_called()
    
    def test_resource_alerting(self, service):
        """Test resource usage alerting."""
        import psutil
        import os
        
        # Mock alerting system
        with patch('agents.backend.onyx.server.features.copywriting.service.alerter') as mock_alerter:
            # Simulate high memory usage
            with patch('psutil.Process') as mock_process:
                mock_process.return_value.memory_info.return_value.rss = 2 * 1024 * 1024 * 1024  # 2GB
                
                # Trigger resource check
                service.check_resource_usage()
                
                # Verify resource alert was triggered
                mock_alerter.send_alert.assert_called()


class TestMonitoringIntegration:
    """Test monitoring integration scenarios."""
    
    @pytest.fixture
    def service(self):
        """Create service instance for testing."""
        return CopywritingService()
    
    @pytest.mark.asyncio
    async def test_comprehensive_monitoring(self, service):
        """Test comprehensive monitoring integration."""
        request = TestDataFactory.create_sample_request()
        
        # Mock all monitoring components
        with patch('agents.backend.onyx.server.features.copywriting.service.metrics') as mock_metrics:
            with patch('agents.backend.onyx.server.features.copywriting.service.logger') as mock_logger:
                with patch('agents.backend.onyx.server.features.copywriting.service.tracer') as mock_tracer:
                    with patch('agents.backend.onyx.server.features.copywriting.service.alerter') as mock_alerter:
                        # Mock tracer
                        mock_span = Mock()
                        mock_tracer.start_span.return_value.__enter__.return_value = mock_span
                        
                        with patch.object(service, '_call_ai_model') as mock_call:
                            mock_call.return_value = {
                                "variants": [{"headline": "Test", "primary_text": "Content"}],
                                "model_used": "gpt-3.5-turbo",
                                "generation_time": 1.0,
                                "extra_metadata": {"tokens_used": 100}
                            }
                            
                            response = await service.generate_copywriting(request, "gpt-3.5-turbo")
                            
                            # Verify all monitoring components were called
                            mock_metrics.increment_request_count.assert_called_once()
                            mock_metrics.record_request_duration.assert_called_once()
                            mock_metrics.record_response_time.assert_called_once()
                            mock_metrics.record_tokens_used.assert_called_once()
                            
                            mock_logger.info.assert_called()
                            mock_logger.debug.assert_called()
                            
                            mock_tracer.start_span.assert_called_once()
                            mock_span.set_tag.assert_called()
                            mock_span.log.assert_called()
                            
                            # No alerts should be triggered for normal operation
                            mock_alerter.send_alert.assert_not_called()
    
    def test_monitoring_failure_handling(self, service):
        """Test handling of monitoring failures."""
        request = TestDataFactory.create_sample_request()
        
        # Mock monitoring components to fail
        with patch('agents.backend.onyx.server.features.copywriting.service.metrics') as mock_metrics:
            mock_metrics.increment_request_count.side_effect = Exception("Metrics error")
            
            with patch('agents.backend.onyx.server.features.copywriting.service.logger') as mock_logger:
                mock_logger.info.side_effect = Exception("Logging error")
                
                with patch.object(service, '_call_ai_model') as mock_call:
                    mock_call.return_value = {
                        "variants": [{"headline": "Test", "primary_text": "Content"}],
                        "model_used": "gpt-3.5-turbo",
                        "generation_time": 1.0,
                        "extra_metadata": {}
                    }
                    
                    # Service should still work even if monitoring fails
                    response = asyncio.run(service.generate_copywriting(request, "gpt-3.5-turbo"))
                    
                    assert response is not None
                    assert response.variants[0]["headline"] == "Test"
    
    def test_monitoring_data_validation(self, service):
        """Test validation of monitoring data."""
        request = TestDataFactory.create_sample_request()
        
        # Mock metrics collection
        with patch('agents.backend.onyx.server.features.copywriting.service.metrics') as mock_metrics:
            with patch.object(service, '_call_ai_model') as mock_call:
                mock_call.return_value = {
                    "variants": [{"headline": "Test", "primary_text": "Content"}],
                    "model_used": "gpt-3.5-turbo",
                    "generation_time": 1.0,
                    "extra_metadata": {"tokens_used": 100}
                }
                
                asyncio.run(service.generate_copywriting(request, "gpt-3.5-turbo"))
                
                # Verify metrics data is valid
                call_args = mock_metrics.record_request_duration.call_args[0]
                assert isinstance(call_args[0], (int, float))
                assert call_args[0] > 0
                
                call_args = mock_metrics.record_tokens_used.call_args[0]
                assert isinstance(call_args[0], int)
                assert call_args[0] > 0
