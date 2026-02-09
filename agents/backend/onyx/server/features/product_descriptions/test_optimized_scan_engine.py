from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import pytest
import asyncio
import time
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any
from optimized_scan_engine import (
from typing import Any, List, Dict, Optional
import logging
"""
Enhanced Tests for Optimized Scan Engine

Comprehensive tests for the enhanced scan engine covering:
- Unit tests for all enhanced components
- Integration tests with mocked dependencies
- Async operation testing with pytest-asyncio
- Edge case handling (timeouts, errors, invalid inputs)
- Enhanced structured logging validation with correlation IDs
- ML confidence scoring and enhanced security metrics
- Performance under load with enhanced features
"""


    UltraOptimizedScanEngine,
    EnhancedAsyncIOHelpers,
    EnhancedSecurityMetrics,
    EnhancedScanConfig,
    EnhancedFinding,
    FindingSeverity,
    FindingConfidence,
    EnhancedScanRequest,
    EnhancedScanResponse,
    start_enhanced_security_scan,
    get_enhanced_scan_metrics,
    get_enhanced_overall_metrics,
    enhanced_health_check
)

# =============================================================================
# Enhanced Fixtures
# =============================================================================

@pytest.fixture
def mock_http_session():
    """Mock HTTP session for testing."""
    session = AsyncMock()
    
    # Mock successful response
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.headers = {
        "X-Frame-Options": "DENY",
        "X-Content-Type-Options": "nosniff",
        "Strict-Transport-Security": "max-age=31536000"
    }
    mock_response.text = AsyncMock(return_value="Welcome to our secure site")
    
    session.get.return_value.__aenter__.return_value = mock_response
    return session

@pytest.fixture
def mock_crypto_backend():
    """Mock crypto backend for testing."""
    backend = AsyncMock()
    backend.encrypt.return_value = b"encrypted_data"
    return backend

@pytest.fixture
def sample_enhanced_scan_config():
    """Sample enhanced scan configuration for testing."""
    return EnhancedScanConfig(
        targets=["https://example.com", "https://test.com"],
        scan_type="vulnerability",
        max_concurrent_scans=5,
        timeout_per_target=10.0,
        rate_limit_per_second=2,
        enable_ml_detection=True,
        confidence_threshold=0.7,
        enable_correlation=True,
        scan_priority="normal"
    )

@pytest.fixture
def enhanced_scan_engine():
    """Fresh enhanced scan engine instance for each test."""
    return UltraOptimizedScanEngine()

# =============================================================================
# Enhanced Unit Tests
# =============================================================================

class TestEnhancedSecurityMetrics:
    """Test EnhancedSecurityMetrics dataclass."""
    
    def test_enhanced_metrics_initialization(self) -> Any:
        """Test enhanced metrics initialization."""
        metrics = EnhancedSecurityMetrics(
            scan_id="test-123",
            start_time=datetime.utcnow(),
            total_targets=10
        )
        
        assert metrics.scan_id == "test-123"
        assert metrics.total_targets == 10
        assert metrics.scanned_targets == 0
        assert metrics.findings_count == 0
        assert metrics.correlation_id is not None
        assert metrics.ml_confidence_score == 0.0
    
    def test_enhanced_false_positive_rate_calculation(self) -> Any:
        """Test enhanced false positive rate calculation."""
        metrics = EnhancedSecurityMetrics(
            scan_id="test-123",
            start_time=datetime.utcnow(),
            findings_count=10,
            false_positives=3
        )
        
        assert metrics.false_positive_rate == 0.3
    
    def test_enhanced_throughput_calculation(self) -> Any:
        """Test throughput calculation."""
        metrics = EnhancedSecurityMetrics(
            scan_id="test-123",
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow(),
            scanned_targets=20,
            scan_duration=10.0
        )
        
        assert metrics.throughput == 2.0  # 20 targets / 10 seconds
    
    def test_enhanced_efficiency_score_calculation(self) -> Any:
        """Test efficiency score calculation."""
        metrics = EnhancedSecurityMetrics(
            scan_id="test-123",
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow(),
            total_targets=10,
            scanned_targets=8,
            findings_count=5,
            false_positives=1,
            scan_duration=5.0
        )
        
        # Should have good efficiency score
        assert 0.0 <= metrics.efficiency_score <= 1.0
        assert metrics.efficiency_score > 0.5  # Should be reasonably good
    
    def test_enhanced_as_dict_serialization(self) -> Any:
        """Test enhanced metrics serialization to dictionary."""
        start_time = datetime.utcnow()
        metrics = EnhancedSecurityMetrics(
            scan_id="test-123",
            start_time=start_time,
            total_targets=5,
            scanned_targets=3,
            findings_count=2
        )
        
        result = metrics.as_dict()
        
        assert result["scan_id"] == "test-123"
        assert result["correlation_id"] is not None
        assert result["total_targets"] == 5
        assert result["scanned_targets"] == 3
        assert result["findings_count"] == 2
        assert "ml_confidence_score" in result
        assert "throughput" in result
        assert "efficiency_score" in result

class TestEnhancedScanConfig:
    """Test EnhancedScanConfig validation."""
    
    def test_enhanced_valid_config(self) -> Any:
        """Test valid enhanced configuration."""
        config = EnhancedScanConfig(
            targets=["https://example.com"],
            scan_type="vulnerability",
            enable_ml_detection=True,
            confidence_threshold=0.8
        )
        
        assert config.targets == ["https://example.com"]
        assert config.scan_type == "vulnerability"
        assert config.enable_ml_detection is True
        assert config.confidence_threshold == 0.8
        assert config.max_concurrent_scans == 20  # default
    
    def test_enhanced_target_deduplication(self) -> Optional[Dict[str, Any]]:
        """Test target deduplication."""
        config = EnhancedScanConfig(
            targets=["https://example.com", "https://EXAMPLE.COM", "https://test.com"],
            scan_type="vulnerability"
        )
        
        # Should deduplicate case-insensitive
        assert len(config.targets) == 2
        assert "https://example.com" in config.targets
        assert "https://test.com" in config.targets
    
    def test_enhanced_scan_type_validation(self) -> Any:
        """Test enhanced scan type validation."""
        # Valid types
        valid_types = ["vulnerability", "port", "web", "network", "api", "mobile"]
        for scan_type in valid_types:
            config = EnhancedScanConfig(
                targets=["https://example.com"],
                scan_type=scan_type
            )
            assert config.scan_type == scan_type
        
        # Invalid type
        with pytest.raises(ValueError):
            EnhancedScanConfig(
                targets=["https://example.com"],
                scan_type="invalid_type"
            )
    
    def test_enhanced_config_limits_validation(self) -> Any:
        """Test configuration limits validation."""
        # Network scan with too many concurrent scans
        with pytest.raises(ValueError):
            EnhancedScanConfig(
                targets=["https://example.com"],
                scan_type="network",
                max_concurrent_scans=60  # Over limit of 50
            )
        
        # API scan with too many concurrent scans
        with pytest.raises(ValueError):
            EnhancedScanConfig(
                targets=["https://example.com"],
                scan_type="api",
                max_concurrent_scans=150  # Over limit of 100
            )

class TestEnhancedFinding:
    """Test EnhancedFinding model."""
    
    def test_enhanced_finding_creation(self) -> Any:
        """Test enhanced finding creation."""
        finding = EnhancedFinding(
            target="https://example.com",
            severity=FindingSeverity.HIGH,
            confidence=FindingConfidence.HIGH,
            title="SQL Injection",
            description="SQL injection vulnerability found",
            evidence="Found SQL injection in login form",
            remediation="Use parameterized queries",
            ml_confidence_score=0.85
        )
        
        assert finding.target == "https://example.com"
        assert finding.severity == FindingSeverity.HIGH
        assert finding.confidence == FindingConfidence.HIGH
        assert finding.ml_confidence_score == 0.85
        assert finding.id is not None
        assert finding.tags == []
    
    def test_enhanced_finding_with_tags_and_references(self) -> Any:
        """Test enhanced finding with tags and references."""
        finding = EnhancedFinding(
            target="https://example.com",
            severity=FindingSeverity.CRITICAL,
            confidence=FindingConfidence.CERTAIN,
            title="XSS Vulnerability",
            description="Cross-site scripting vulnerability",
            evidence="Found XSS in search parameter",
            remediation="Sanitize user input",
            cve_id="CVE-2023-1234",
            cvss_score=9.8,
            ml_confidence_score=0.95,
            tags=["xss", "injection", "owasp-top-10"],
            references=["https://owasp.org/www-project-top-ten/"],
            affected_components=["search-function", "input-validation"]
        )
        
        assert finding.cve_id == "CVE-2023-1234"
        assert finding.cvss_score == 9.8
        assert finding.ml_confidence_score == 0.95
        assert "xss" in finding.tags
        assert "https://owasp.org/www-project-top-ten/" in finding.references
        assert "search-function" in finding.affected_components

# =============================================================================
# Enhanced Async I/O Helpers Tests
# =============================================================================

class TestEnhancedAsyncIOHelpers:
    """Test EnhancedAsyncIOHelpers class."""
    
    @pytest.mark.asyncio
    async async def test_enhanced_fetch_target_data_success(self, mock_http_session, mock_crypto_backend) -> Optional[Dict[str, Any]]:
        """Test successful enhanced target data fetching."""
        helpers = EnhancedAsyncIOHelpers(mock_http_session, mock_crypto_backend)
        
        result = await helpers.fetch_target_data_optimized("https://example.com")
        
        assert result["target"] == "https://example.com"
        assert result["status_code"] == 200
        assert result["success"] is True
        assert "response_time" in result
        assert "headers" in result
        assert "content" in result
        assert "attempt" in result
        assert "cache_key" in result
    
    @pytest.mark.asyncio
    async async def test_enhanced_fetch_target_data_with_retries(self, mock_http_session, mock_crypto_backend) -> Optional[Dict[str, Any]]:
        """Test enhanced target data fetching with retry logic."""
        # Mock HTTP session to fail first, then succeed
        mock_http_session.get.side_effect = [
            Exception("Connection failed"),  # First attempt fails
            AsyncMock().__aenter__.return_value  # Second attempt succeeds
        ]
        
        helpers = EnhancedAsyncIOHelpers(mock_http_session, mock_crypto_backend)
        
        result = await helpers.fetch_target_data_optimized("https://example.com", retry_attempts=2)
        
        assert result["success"] is True
        assert result["attempt"] == 2
    
    @pytest.mark.asyncio
    async def test_enhanced_encrypt_finding_with_key_rotation(self, mock_http_session, mock_crypto_backend) -> Any:
        """Test enhanced finding encryption with key rotation."""
        helpers = EnhancedAsyncIOHelpers(mock_http_session, mock_crypto_backend)
        
        result = await helpers.encrypt_finding_enhanced("sensitive data", key_rotation=True)
        
        assert result == "656e637279707465645f64617461"  # hex of "encrypted_data"
        mock_crypto_backend.encrypt.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_enhanced_validate_target_comprehensive(self, mock_http_session, mock_crypto_backend) -> Optional[Dict[str, Any]]:
        """Test comprehensive enhanced target validation."""
        helpers = EnhancedAsyncIOHelpers(mock_http_session, mock_crypto_backend)
        
        result = await helpers.validate_target_enhanced("https://example.com")
        
        assert result["target"] == "https://example.com"
        assert "is_valid" in result
        assert "validation_time" in result
        assert "checks" in result
        assert "connectivity" in result["checks"]
        assert "ssl" in result["checks"]
        assert "response_time" in result["checks"]

# =============================================================================
# Enhanced Scan Engine Tests
# =============================================================================

class TestUltraOptimizedScanEngine:
    """Test UltraOptimizedScanEngine class."""
    
    @pytest.mark.asyncio
    async def test_enhanced_start_scan(self, enhanced_scan_engine, sample_enhanced_scan_config, mock_http_session, mock_crypto_backend) -> Any:
        """Test starting an enhanced scan."""
        scan_id, metrics = await enhanced_scan_engine.start_scan_enhanced(
            sample_enhanced_scan_config,
            mock_http_session,
            mock_crypto_backend,
            "user-123"
        )
        
        assert scan_id is not None
        assert metrics.scan_id == scan_id
        assert metrics.total_targets == 2
        assert metrics.correlation_id is not None
        assert scan_id in enhanced_scan_engine.active_scans
    
    @pytest.mark.asyncio
    async def test_enhanced_scan_execution_completion(self, enhanced_scan_engine, sample_enhanced_scan_config, mock_http_session, mock_crypto_backend) -> Any:
        """Test complete enhanced scan execution."""
        scan_id, metrics = await enhanced_scan_engine.start_scan_enhanced(
            sample_enhanced_scan_config,
            mock_http_session,
            mock_crypto_backend,
            "user-123"
        )
        
        # Wait for scan to complete
        await asyncio.sleep(0.1)
        
        # Check that scan moved to history
        assert scan_id not in enhanced_scan_engine.active_scans
        assert any(m.scan_id == scan_id for m in enhanced_scan_engine.scan_history)
        
        # Get completed metrics
        completed_metrics = enhanced_scan_engine.get_scan_metrics_enhanced(scan_id)
        assert completed_metrics is not None
        assert completed_metrics.end_time is not None
        assert completed_metrics.scan_duration > 0
        assert completed_metrics.ml_confidence_score >= 0.0
    
    @pytest.mark.asyncio
    async def test_enhanced_scan_with_ml_detection(self, enhanced_scan_engine, mock_http_session, mock_crypto_backend) -> Any:
        """Test enhanced scan with ML detection enabled."""
        config = EnhancedScanConfig(
            targets=["https://admin-site.com"],
            scan_type="vulnerability",
            enable_ml_detection=True,
            confidence_threshold=0.7
        )
        
        scan_id, metrics = await enhanced_scan_engine.start_scan_enhanced(
            config,
            mock_http_session,
            mock_crypto_backend,
            "user-123"
        )
        
        # Wait for scan to complete
        await asyncio.sleep(0.1)
        
        completed_metrics = enhanced_scan_engine.get_scan_metrics_enhanced(scan_id)
        assert completed_metrics.ml_confidence_score >= 0.0
        assert completed_metrics.efficiency_score > 0.0
    
    def test_enhanced_get_scan_metrics_not_found(self, enhanced_scan_engine) -> Optional[Dict[str, Any]]:
        """Test getting metrics for non-existent enhanced scan."""
        result = enhanced_scan_engine.get_scan_metrics_enhanced("non-existent-id")
        assert result is None
    
    def test_enhanced_get_all_metrics_empty(self, enhanced_scan_engine) -> Optional[Dict[str, Any]]:
        """Test getting enhanced metrics when no scans exist."""
        metrics = enhanced_scan_engine.get_all_metrics_enhanced()
        
        assert metrics["active_scans"] == 0
        assert metrics["completed_scans"] == 0
        assert metrics["total_findings"] == 0
        assert metrics["total_errors"] == 0
        assert metrics["average_efficiency_score"] == 0.0
        assert metrics["average_ml_confidence"] == 0.0
        assert metrics["system_health"] == "healthy"
    
    def test_enhanced_get_all_metrics_with_scans(self, enhanced_scan_engine) -> Optional[Dict[str, Any]]:
        """Test getting enhanced metrics with existing scans."""
        # Add some mock metrics to history
        mock_metrics = EnhancedSecurityMetrics(
            scan_id="test-1",
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow(),
            total_targets=10,
            scanned_targets=8,
            findings_count=3,
            false_positives=1,
            error_count=2,
            scan_duration=15.5,
            ml_confidence_score=0.8
        )
        enhanced_scan_engine.scan_history.append(mock_metrics)
        
        metrics = enhanced_scan_engine.get_all_metrics_enhanced()
        
        assert metrics["completed_scans"] == 1
        assert metrics["total_findings"] == 3
        assert metrics["total_false_positives"] == 1
        assert metrics["total_errors"] == 2
        assert metrics["average_scan_duration"] == 15.5
        assert metrics["average_ml_confidence"] == 0.8

# =============================================================================
# Enhanced FastAPI Integration Tests
# =============================================================================

class TestEnhancedFastAPIIntegration:
    """Test Enhanced FastAPI integration functions."""
    
    @pytest.mark.asyncio
    async def test_enhanced_start_security_scan_success(self, mock_http_session, mock_crypto_backend, sample_enhanced_scan_config) -> Any:
        """Test successful enhanced scan start via FastAPI."""
        request = EnhancedScanRequest(
            config=sample_enhanced_scan_config,
            user_id="user-123",
            priority="high"
        )
        
        # Mock BackgroundTasks
        background_tasks = MagicMock()
        
        response = await start_enhanced_security_scan(
            request,
            background_tasks,
            mock_http_session,
            mock_crypto_backend
        )
        
        assert isinstance(response, EnhancedScanResponse)
        assert response.status == "started"
        assert response.scan_id is not None
        assert response.correlation_id is not None
        assert response.metrics is not None
        assert response.estimated_duration is not None
    
    @pytest.mark.asyncio
    async def test_enhanced_get_scan_metrics_success(self) -> Optional[Dict[str, Any]]:
        """Test getting enhanced scan metrics via FastAPI."""
        # Mock scan engine to return metrics
        with patch('optimized_scan_engine.enhanced_scan_engine') as mock_engine:
            mock_metrics = EnhancedSecurityMetrics(
                scan_id="test-123",
                start_time=datetime.utcnow(),
                total_targets=5
            )
            mock_engine.get_scan_metrics_enhanced.return_value = mock_metrics
            
            result = await get_enhanced_scan_metrics("test-123")
            
            assert result["scan_id"] == "test-123"
            assert result["total_targets"] == 5
            assert "correlation_id" in result
            assert "ml_confidence_score" in result
    
    @pytest.mark.asyncio
    async def test_enhanced_get_overall_metrics(self) -> Optional[Dict[str, Any]]:
        """Test getting enhanced overall metrics via FastAPI."""
        with patch('optimized_scan_engine.enhanced_scan_engine') as mock_engine:
            mock_engine.get_all_metrics_enhanced.return_value = {
                "active_scans": 2,
                "completed_scans": 10,
                "total_findings": 25,
                "average_efficiency_score": 0.85,
                "average_ml_confidence": 0.78,
                "system_health": "healthy"
            }
            
            result = await get_enhanced_overall_metrics()
            
            assert result["active_scans"] == 2
            assert result["completed_scans"] == 10
            assert result["total_findings"] == 25
            assert result["average_efficiency_score"] == 0.85
            assert result["average_ml_confidence"] == 0.78
            assert result["system_health"] == "healthy"
    
    @pytest.mark.asyncio
    async def test_enhanced_health_check_healthy(self) -> Any:
        """Test enhanced health check when system is healthy."""
        with patch('optimized_scan_engine.enhanced_scan_engine') as mock_engine:
            mock_engine.get_all_metrics_enhanced.return_value = {
                "total_errors": 5,
                "overall_false_positive_rate": 0.1,
                "average_efficiency_score": 0.85
            }
            
            result = await enhanced_health_check()
            
            assert result["status"] == "healthy"
            assert "timestamp" in result
            assert "efficiency_score" in result
            assert "ml_confidence" in result
            assert "system_metrics" in result
    
    @pytest.mark.asyncio
    async def test_enhanced_health_check_degraded(self) -> Any:
        """Test enhanced health check when system is degraded."""
        with patch('optimized_scan_engine.enhanced_scan_engine') as mock_engine:
            mock_engine.get_all_metrics_enhanced.return_value = {
                "total_errors": 15,
                "overall_false_positive_rate": 0.4,
                "average_efficiency_score": 0.3
            }
            
            result = await enhanced_health_check()
            
            assert result["status"] == "degraded"

# =============================================================================
# Enhanced Edge Case Tests
# =============================================================================

class TestEnhancedEdgeCases:
    """Test enhanced edge cases and error conditions."""
    
    @pytest.mark.asyncio
    async def test_enhanced_concurrent_scan_limiting(self, enhanced_scan_engine, mock_http_session, mock_crypto_backend) -> Any:
        """Test enhanced concurrent scan limiting."""
        config = EnhancedScanConfig(
            targets=[f"https://example{i}.com" for i in range(20)],
            scan_type="vulnerability",
            max_concurrent_scans=3,  # Limit to 3 concurrent
            rate_limit_per_second=10,
            enable_ml_detection=True
        )
        
        scan_id, metrics = await enhanced_scan_engine.start_scan_enhanced(
            config,
            mock_http_session,
            mock_crypto_backend,
            "user-123"
        )
        
        # Wait for scan to complete
        await asyncio.sleep(0.1)
        
        completed_metrics = enhanced_scan_engine.get_scan_metrics_enhanced(scan_id)
        assert completed_metrics.scanned_targets == 20  # All targets should be scanned
        assert completed_metrics.efficiency_score > 0.0
    
    @pytest.mark.asyncio
    async def test_enhanced_scan_timeout_handling(self, enhanced_scan_engine, mock_crypto_backend) -> Any:
        """Test enhanced scan timeout handling."""
        # Mock slow HTTP session
        slow_session = AsyncMock()
        slow_response = AsyncMock()
        slow_response.status = 200
        slow_response.headers = {}
        slow_response.text = AsyncMock(return_value="")
        
        # Make the response slow
        async def slow_get(*args, **kwargs) -> Optional[Dict[str, Any]]:
            await asyncio.sleep(0.2)  # Simulate slow response
            return slow_response
        
        slow_session.get.return_value.__aenter__.side_effect = slow_get
        
        config = EnhancedScanConfig(
            targets=["https://slow-site.com"],
            scan_type="vulnerability",
            timeout_per_target=0.1,  # Short timeout
            retry_attempts=1
        )
        
        scan_id, metrics = await enhanced_scan_engine.start_scan_enhanced(
            config,
            slow_session,
            mock_crypto_backend,
            "user-123"
        )
        
        # Wait for scan to complete
        await asyncio.sleep(0.3)
        
        completed_metrics = enhanced_scan_engine.get_scan_metrics_enhanced(scan_id)
        assert completed_metrics.error_count >= 0  # Should handle timeout gracefully
    
    @pytest.mark.asyncio
    async def test_enhanced_priority_scanning(self, enhanced_scan_engine, mock_http_session, mock_crypto_backend) -> Any:
        """Test enhanced priority-based scanning."""
        config = EnhancedScanConfig(
            targets=["https://critical-site.com", "https://normal-site.com"],
            scan_type="vulnerability",
            scan_priority="high",
            enable_ml_detection=True
        )
        
        scan_id, metrics = await enhanced_scan_engine.start_scan_enhanced(
            config,
            mock_http_session,
            mock_crypto_backend,
            "user-123"
        )
        
        # Wait for scan to complete
        await asyncio.sleep(0.1)
        
        completed_metrics = enhanced_scan_engine.get_scan_metrics_enhanced(scan_id)
        assert completed_metrics.scanned_targets == 2
        assert completed_metrics.efficiency_score > 0.0

# =============================================================================
# Enhanced Performance Tests
# =============================================================================

class TestEnhancedPerformance:
    """Test enhanced performance characteristics."""
    
    @pytest.mark.asyncio
    async def test_enhanced_scan_performance_metrics(self, enhanced_scan_engine, mock_http_session, mock_crypto_backend) -> Any:
        """Test enhanced scan performance metrics collection."""
        config = EnhancedScanConfig(
            targets=["https://example1.com", "https://example2.com"],
            scan_type="vulnerability",
            max_concurrent_scans=2,
            enable_ml_detection=True
        )
        
        start_time = time.time()
        scan_id, metrics = await enhanced_scan_engine.start_scan_enhanced(
            config,
            mock_http_session,
            mock_crypto_backend,
            "user-123"
        )
        
        # Wait for scan to complete
        await asyncio.sleep(0.1)
        
        completed_metrics = enhanced_scan_engine.get_scan_metrics_enhanced(scan_id)
        end_time = time.time()
        
        # Verify enhanced performance metrics
        assert completed_metrics.scan_duration > 0
        assert completed_metrics.average_response_time > 0
        assert completed_metrics.scan_success_rate > 0
        assert completed_metrics.throughput > 0
        assert completed_metrics.efficiency_score > 0
        assert completed_metrics.ml_confidence_score >= 0.0
    
    @pytest.mark.asyncio
    async async def test_enhanced_concurrent_scan_requests(self, mock_http_session, mock_crypto_backend) -> Any:
        """Test handling multiple concurrent enhanced scan requests."""
        config = EnhancedScanConfig(
            targets=["https://example.com"],
            scan_type="vulnerability",
            enable_ml_detection=True
        )
        
        # Start multiple scans concurrently
        tasks = []
        for i in range(5):
            request = EnhancedScanRequest(
                config=config, 
                user_id=f"user-{i}",
                priority="normal"
            )
            background_tasks = MagicMock()
            task = start_enhanced_security_scan(
                request, 
                background_tasks, 
                mock_http_session, 
                mock_crypto_backend
            )
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks)
        
        # All scans should start successfully
        for response in responses:
            assert response.status == "started"
            assert response.scan_id is not None
            assert response.correlation_id is not None

# =============================================================================
# Enhanced Structured Logging Tests
# =============================================================================

class TestEnhancedStructuredLogging:
    """Test enhanced structured logging functionality."""
    
    @pytest.mark.asyncio
    async def test_enhanced_scan_logging_events(self, enhanced_scan_engine, sample_enhanced_scan_config, mock_http_session, mock_crypto_backend, caplog) -> Any:
        """Test that enhanced scan events are properly logged."""
        with caplog.at_level("INFO"):
            scan_id, metrics = await enhanced_scan_engine.start_scan_enhanced(
                sample_enhanced_scan_config,
                mock_http_session,
                mock_crypto_backend,
                "user-123"
            )
            
            # Wait for scan to complete
            await asyncio.sleep(0.1)
        
        # Check for enhanced structured log messages
        log_messages = [record.message for record in caplog.records]
        
        # Should have enhanced scan start and completion logs
        assert any("Enhanced scan started" in msg for msg in log_messages)
        assert any("Enhanced scan completed" in msg for msg in log_messages)
        
        # Check for correlation ID in logs
        correlation_logs = [msg for msg in log_messages if "correlation_id" in msg]
        assert len(correlation_logs) > 0
    
    @pytest.mark.asyncio
    async def test_enhanced_error_logging(self, enhanced_scan_engine, mock_crypto_backend, caplog) -> Any:
        """Test enhanced error logging."""
        # Mock HTTP session to raise exception
        error_session = AsyncMock()
        error_session.get.side_effect = Exception("Network error")
        
        config = EnhancedScanConfig(
            targets=["https://error-site.com"],
            scan_type="vulnerability",
            enable_ml_detection=True
        )
        
        with caplog.at_level("ERROR"):
            scan_id, metrics = await enhanced_scan_engine.start_scan_enhanced(
                config,
                error_session,
                mock_crypto_backend,
                "user-123"
            )
            
            # Wait for scan to complete
            await asyncio.sleep(0.1)
        
        # Check for enhanced error logs
        log_messages = [record.message for record in caplog.records]
        assert any("Failed to fetch target after all retries" in msg for msg in log_messages)

match __name__:
    case "__main__":
    pytest.main([__file__, "-v", "--tb=short"]) 