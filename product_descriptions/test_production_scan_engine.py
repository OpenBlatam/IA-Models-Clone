from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import asyncio
import json
import time
import uuid
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, List, Optional
import aiohttp
import numpy as np
import pytest
import structlog
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY
from pydantic import ValidationError
from production_scan_engine import (
        from fastapi import FastAPI
from typing import Any, List, Dict, Optional
import logging
"""
Comprehensive Test Suite for Production-Grade Cybersecurity Scan Engine
Tests all functionality, edge cases, and integration scenarios
"""



    ProductionScanEngine,
    ProductionScanConfiguration,
    ScanTarget,
    SecurityFinding,
    ScanResult,
    SecurityMetrics,
    ScanStatus,
    ScanType,
    Severity,
    DatabaseManager,
    RedisCache,
    SystemMonitor,
    ProductionScanRequest,
    ProductionScanResponse,
    get_production_scan_engine,
    start_production_scan,
    get_production_scan_status,
    cancel_production_scan,
    production_health_check
)


class TestProductionScanConfiguration:
    """Test production scan configuration validation"""
    
    def test_valid_configuration(self) -> Any:
        """Test valid configuration creation"""
        config = ProductionScanConfiguration(
            scan_type=ScanType.VULNERABILITY,
            max_concurrent_scans=50,
            timeout_per_target=120,
            enable_ml_detection=True,
            ml_confidence_threshold=0.9
        )
        
        assert config.scan_type == ScanType.VULNERABILITY
        assert config.max_concurrent_scans == 50
        assert config.timeout_per_target == 120
        assert config.enable_ml_detection is True
        assert config.ml_confidence_threshold == 0.9
    
    def test_invalid_ml_threshold(self) -> Any:
        """Test invalid ML confidence threshold"""
        with pytest.raises(ValidationError):
            ProductionScanConfiguration(ml_confidence_threshold=1.5)
        
        with pytest.raises(ValidationError):
            ProductionScanConfiguration(ml_confidence_threshold=0.0)
    
    def test_invalid_concurrent_scans(self) -> Any:
        """Test invalid concurrent scans limit"""
        with pytest.raises(ValidationError):
            ProductionScanConfiguration(max_concurrent_scans=0)
        
        with pytest.raises(ValidationError):
            ProductionScanConfiguration(max_concurrent_scans=300)
    
    def test_default_values(self) -> Any:
        """Test default configuration values"""
        config = ProductionScanConfiguration()
        
        assert config.scan_type == ScanType.VULNERABILITY
        assert config.max_concurrent_scans == 20
        assert config.timeout_per_target == 60
        assert config.enable_ml_detection is True
        assert config.ml_confidence_threshold == 0.85


class TestScanTarget:
    """Test scan target validation and functionality"""
    
    def test_valid_target(self) -> Optional[Dict[str, Any]]:
        """Test valid target creation"""
        target = ScanTarget(
            url="example.com",
            port=443,
            protocol="https",
            timeout=30,
            retries=3
        )
        
        assert target.url == "example.com"
        assert target.port == 443
        assert target.protocol == "https"
        assert target.timeout == 30
        assert target.retries == 3
    
    def test_invalid_url(self) -> Any:
        """Test invalid URL validation"""
        with pytest.raises(ValueError, match="URL cannot be empty"):
            ScanTarget(url="")
    
    def test_invalid_timeout(self) -> Any:
        """Test invalid timeout validation"""
        with pytest.raises(ValueError, match="Timeout must be positive"):
            ScanTarget(url="example.com", timeout=0)
        
        with pytest.raises(ValueError, match="Timeout must be positive"):
            ScanTarget(url="example.com", timeout=-1)
    
    def test_invalid_retries(self) -> Any:
        """Test invalid retries validation"""
        with pytest.raises(ValueError, match="Retries cannot be negative"):
            ScanTarget(url="example.com", retries=-1)
    
    def test_invalid_port(self) -> Any:
        """Test invalid port validation"""
        with pytest.raises(ValueError, match="Port must be between 1 and 65535"):
            ScanTarget(url="example.com", port=0)
        
        with pytest.raises(ValueError, match="Port must be between 1 and 65535"):
            ScanTarget(url="example.com", port=70000)


class TestSecurityFinding:
    """Test security finding functionality"""
    
    def test_finding_creation(self) -> Any:
        """Test security finding creation"""
        finding = SecurityFinding(
            title="Test Vulnerability",
            description="Test description",
            severity=Severity.HIGH,
            category="test_category",
            cvss_score=7.5,
            cve_id="CVE-2023-1234"
        )
        
        assert finding.title == "Test Vulnerability"
        assert finding.severity == Severity.HIGH
        assert finding.cvss_score == 7.5
        assert finding.cve_id == "CVE-2023-1234"
        assert finding.confidence == 1.0
        assert finding.false_positive is False
    
    def test_finding_with_optional_fields(self) -> Any:
        """Test finding with optional fields"""
        finding = SecurityFinding(
            title="Test Finding",
            description="Test description",
            severity=Severity.MEDIUM,
            category="test",
            confidence=0.8,
            false_positive=True,
            tags=["test", "vulnerability"]
        )
        
        assert finding.confidence == 0.8
        assert finding.false_positive is True
        assert finding.tags == ["test", "vulnerability"]


class TestSecurityMetrics:
    """Test security metrics functionality"""
    
    def test_metrics_creation(self) -> Any:
        """Test metrics creation"""
        metrics = SecurityMetrics(
            scan_id="test-scan-123",
            start_time=time.time(),
            total_targets=10,
            completed_targets=8,
            failed_targets=2
        )
        
        assert metrics.scan_id == "test-scan-123"
        assert metrics.total_targets == 10
        assert metrics.completed_targets == 8
        assert metrics.failed_targets == 2
    
    def test_metrics_calculation(self) -> Any:
        """Test metrics calculation"""
        start_time = time.time()
        metrics = SecurityMetrics(
            scan_id="test-scan-123",
            start_time=start_time,
            total_targets=10,
            completed_targets=8,
            failed_targets=1,
            timeout_targets=1,
            total_findings=15,
            false_positives=3,
            true_positives=12
        )
        
        # Set end time
        metrics.end_time = start_time + 60.0  # 60 seconds later
        metrics.calculate_metrics()
        
        assert metrics.scan_duration == 60.0
        assert metrics.throughput == 8.0 / 60.0
        assert metrics.efficiency_score == 12.0 / 15.0
        assert metrics.ml_confidence == 1.0 - (3.0 / 15.0)


class TestDatabaseManager:
    """Test database manager functionality"""
    
    @pytest.fixture
    def db_manager(self) -> Any:
        """Create database manager instance"""
        return DatabaseManager("postgresql+asyncpg://test:test@localhost/test")
    
    @pytest.mark.asyncio
    async def test_initialization(self, db_manager) -> Any:
        """Test database initialization"""
        with patch('production_scan_engine.create_async_engine') as mock_engine:
            mock_engine.return_value = MagicMock()
            
            await db_manager.initialize()
            
            mock_engine.assert_called_once()
            assert db_manager.engine is not None
            assert db_manager.session_factory is not None
    
    @pytest.mark.asyncio
    async def test_store_scan_result(self, db_manager) -> Any:
        """Test storing scan result"""
        db_manager.session_factory = MagicMock()
        mock_session = AsyncMock()
        db_manager.session_factory.return_value.__aenter__.return_value = mock_session
        
        result = ScanResult(
            target="example.com",
            status=ScanStatus.COMPLETED,
            scan_id="test-123"
        )
        
        await db_manager.store_scan_result(result)
        
        mock_session.assert_called()
    
    @pytest.mark.asyncio
    async def test_close(self, db_manager) -> Any:
        """Test database close"""
        db_manager.engine = AsyncMock()
        
        await db_manager.close()
        
        db_manager.engine.dispose.assert_called_once()


class TestRedisCache:
    """Test Redis cache functionality"""
    
    @pytest.fixture
    def redis_cache(self) -> Any:
        """Create Redis cache instance"""
        return RedisCache("redis://localhost:6379")
    
    @pytest.mark.asyncio
    async def test_initialization(self, redis_cache) -> Any:
        """Test Redis initialization"""
        with patch('production_scan_engine.aioredis.from_url') as mock_redis:
            mock_redis.return_value = AsyncMock()
            mock_redis.return_value.ping = AsyncMock()
            
            await redis_cache.initialize()
            
            mock_redis.assert_called_once()
            assert redis_cache.redis is not None
    
    @pytest.mark.asyncio
    async def test_cache_scan_result(self, redis_cache) -> Any:
        """Test caching scan result"""
        redis_cache.redis = AsyncMock()
        
        result_data = {"scan_id": "test-123", "status": "completed"}
        
        await redis_cache.cache_scan_result("test-123", result_data, ttl=3600)
        
        redis_cache.redis.setex.assert_called_once_with(
            "scan_result:test-123",
            3600,
            json.dumps(result_data)
        )
    
    @pytest.mark.asyncio
    async def test_get_cached_result(self, redis_cache) -> Optional[Dict[str, Any]]:
        """Test getting cached result"""
        redis_cache.redis = AsyncMock()
        cached_data = {"scan_id": "test-123", "status": "completed"}
        redis_cache.redis.get.return_value = json.dumps(cached_data)
        
        result = await redis_cache.get_cached_result("test-123")
        
        assert result == cached_data
        redis_cache.redis.get.assert_called_once_with("scan_result:test-123")
    
    @pytest.mark.asyncio
    async def test_get_cached_result_not_found(self, redis_cache) -> Optional[Dict[str, Any]]:
        """Test getting non-existent cached result"""
        redis_cache.redis = AsyncMock()
        redis_cache.redis.get.return_value = None
        
        result = await redis_cache.get_cached_result("test-123")
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_close(self, redis_cache) -> Any:
        """Test Redis close"""
        redis_cache.redis = AsyncMock()
        
        await redis_cache.close()
        
        redis_cache.redis.close.assert_called_once()


class TestSystemMonitor:
    """Test system monitor functionality"""
    
    @pytest.fixture
    def system_monitor(self) -> Any:
        """Create system monitor instance"""
        return SystemMonitor()
    
    def test_get_system_metrics(self, system_monitor) -> Optional[Dict[str, Any]]:
        """Test getting system metrics"""
        with patch('production_scan_engine.psutil.cpu_percent') as mock_cpu, \
             patch('production_scan_engine.psutil.virtual_memory') as mock_memory, \
             patch('production_scan_engine.psutil.disk_usage') as mock_disk:
            
            mock_cpu.return_value = 50.0
            mock_memory.return_value = MagicMock(
                percent=60.0,
                available=8 * 1024**3  # 8GB
            )
            mock_disk.return_value = MagicMock(
                percent=40.0,
                free=100 * 1024**3  # 100GB
            )
            
            metrics = system_monitor.get_system_metrics()
            
            assert 'cpu_percent' in metrics
            assert 'memory_percent' in metrics
            assert 'disk_percent' in metrics
            assert metrics['cpu_percent'] == 50.0
            assert metrics['memory_percent'] == 60.0
            assert metrics['disk_percent'] == 40.0


class TestProductionScanEngine:
    """Test production scan engine functionality"""
    
    @pytest.fixture
    def config(self) -> Any:
        """Create test configuration"""
        return ProductionScanConfiguration(
            max_concurrent_scans=5,
            timeout_per_target=30,
            enable_prometheus_metrics=False,
            enable_redis_caching=False,
            enable_database_storage=False,
            enable_health_checks=False
        )
    
    @pytest.fixture
    def engine(self, config) -> Any:
        """Create test engine instance"""
        return ProductionScanEngine(config)
    
    @pytest.mark.asyncio
    async def test_initialization(self, engine) -> Any:
        """Test engine initialization"""
        await engine.initialize()
        
        assert engine._scan_semaphore._value == 5
        assert engine._rate_limiter._value == 100
    
    @pytest.mark.asyncio
    async def test_deduplicate_targets(self, engine) -> Optional[Dict[str, Any]]:
        """Test target deduplication"""
        targets = [
            ScanTarget(url="example.com", port=443),
            ScanTarget(url="example.com", port=443),  # Duplicate
            ScanTarget(url="test.com", port=80),
            ScanTarget(url="test.com", port=443)  # Different port
        ]
        
        unique_targets = engine._deduplicate_targets(targets)
        
        assert len(unique_targets) == 3
        assert unique_targets[0].url == "example.com"
        assert unique_targets[1].url == "test.com"
        assert unique_targets[2].url == "test.com"
    
    @pytest.mark.asyncio
    async def test_scan_single_target_success(self, engine) -> Optional[Dict[str, Any]]:
        """Test successful single target scan"""
        target = ScanTarget(url="example.com", port=443)
        scan_id = "test-scan-123"
        
        with patch.object(engine, '_perform_comprehensive_scan') as mock_scan:
            mock_scan.return_value = [
                SecurityFinding(
                    title="Test Finding",
                    description="Test description",
                    severity=Severity.MEDIUM,
                    category="test"
                )
            ]
            
            result = await engine._scan_single_target(target, scan_id)
            
            assert result.status == ScanStatus.COMPLETED
            assert len(result.findings) == 1
            assert result.findings[0].title == "Test Finding"
            assert result.metrics['findings_count'] == 1
    
    @pytest.mark.asyncio
    async def test_scan_single_target_timeout(self, engine) -> Optional[Dict[str, Any]]:
        """Test single target scan timeout"""
        target = ScanTarget(url="example.com", port=443, timeout=1)
        scan_id = "test-scan-123"
        
        with patch.object(engine, '_perform_comprehensive_scan') as mock_scan:
            mock_scan.side_effect = asyncio.TimeoutError()
            
            result = await engine._scan_single_target(target, scan_id)
            
            assert result.status == ScanStatus.TIMEOUT
            assert "timeout" in result.error_message.lower()
    
    @pytest.mark.asyncio
    async def test_scan_single_target_exception(self, engine) -> Optional[Dict[str, Any]]:
        """Test single target scan exception"""
        target = ScanTarget(url="example.com", port=443)
        scan_id = "test-scan-123"
        
        with patch.object(engine, '_perform_comprehensive_scan') as mock_scan:
            mock_scan.side_effect = Exception("Test error")
            
            result = await engine._scan_single_target(target, scan_id)
            
            assert result.status == ScanStatus.FAILED
            assert "Test error" in result.error_message
    
    @pytest.mark.asyncio
    async def test_perform_comprehensive_scan(self, engine) -> Any:
        """Test comprehensive scan execution"""
        target = ScanTarget(url="example.com", port=443)
        
        with patch.object(engine, '_check_ssl_tls_security') as mock_ssl, \
             patch.object(engine, '_check_security_headers') as mock_headers, \
             patch.object(engine, '_check_open_ports') as mock_ports:
            
            mock_ssl.return_value = [
                SecurityFinding(title="SSL Issue", severity=Severity.HIGH)
            ]
            mock_headers.return_value = [
                SecurityFinding(title="Header Issue", severity=Severity.MEDIUM)
            ]
            mock_ports.return_value = []
            
            findings = await engine._perform_comprehensive_scan(target)
            
            assert len(findings) == 2
            assert findings[0].title == "SSL Issue"
            assert findings[1].title == "Header Issue"
    
    @pytest.mark.asyncio
    async def test_ssl_tls_security_check(self, engine) -> Any:
        """Test SSL/TLS security check"""
        target = ScanTarget(url="example.com", port=443)
        
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = AsyncMock()
            mock_response.conn = MagicMock()
            mock_response.conn.transport = MagicMock()
            mock_response.conn.transport.get_extra_info.return_value = MagicMock()
            mock_response.conn.transport.get_extra_info.return_value.version.return_value = "TLSv1.0"
            mock_response.conn.transport.get_extra_info.return_value.cipher.return_value = ("RC4-SHA", None, 128)
            
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
            
            findings = await engine._check_ssl_tls_security(target)
            
            assert len(findings) >= 1
            assert any("Weak SSL/TLS Version" in f.title for f in findings)
    
    @pytest.mark.asyncio
    async def test_security_headers_check(self, engine) -> Any:
        """Test security headers check"""
        target = ScanTarget(url="example.com", port=443)
        
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = AsyncMock()
            mock_response.headers = {}  # No security headers
            
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
            
            findings = await engine._check_security_headers(target)
            
            assert len(findings) >= 1
            assert any("Missing Security Header" in f.title for f in findings)
    
    @pytest.mark.asyncio
    async def test_open_ports_check(self, engine) -> Any:
        """Test open ports check"""
        target = ScanTarget(url="example.com", port=22)  # SSH port
        
        with patch('asyncio.open_connection') as mock_conn:
            mock_reader = AsyncMock()
            mock_writer = AsyncMock()
            mock_conn.return_value = (mock_reader, mock_writer)
            
            findings = await engine._check_open_ports(target)
            
            assert len(findings) >= 1
            assert any("Risky Port Open" in f.title for f in findings)
            mock_writer.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_web_vulnerabilities_check(self, engine) -> Any:
        """Test web vulnerabilities check"""
        target = ScanTarget(url="example.com", port=443)
        
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.text = AsyncMock(return_value="root:x:0:0:root:/root:/bin/bash")
            
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
            
            findings = await engine._check_web_vulnerabilities(target)
            
            assert len(findings) >= 1
            assert any("Directory Traversal" in f.title for f in findings)
    
    @pytest.mark.asyncio
    async def test_scan_targets_success(self, engine) -> Optional[Dict[str, Any]]:
        """Test successful multi-target scan"""
        targets = [
            ScanTarget(url="example.com", port=443),
            ScanTarget(url="test.com", port=80)
        ]
        scan_id = "test-scan-123"
        
        with patch.object(engine, '_scan_single_target') as mock_scan:
            mock_scan.return_value = ScanResult(
                target="example.com",
                status=ScanStatus.COMPLETED,
                findings=[SecurityFinding(title="Test", severity=Severity.MEDIUM)]
            )
            
            results = await engine.scan_targets(targets, scan_id)
            
            assert len(results) == 2
            assert all(r.status == ScanStatus.COMPLETED for r in results)
            assert engine.scan_metrics[scan_id].completed_targets == 2
    
    @pytest.mark.asyncio
    async def test_scan_targets_with_failures(self, engine) -> Optional[Dict[str, Any]]:
        """Test multi-target scan with failures"""
        targets = [
            ScanTarget(url="example.com", port=443),
            ScanTarget(url="test.com", port=80)
        ]
        scan_id = "test-scan-123"
        
        with patch.object(engine, '_scan_single_target') as mock_scan:
            mock_scan.side_effect = [
                ScanResult(target="example.com", status=ScanStatus.COMPLETED),
                Exception("Test error")
            ]
            
            results = await engine.scan_targets(targets, scan_id)
            
            assert len(results) == 1
            assert results[0].status == ScanStatus.COMPLETED
            assert engine.scan_metrics[scan_id].completed_targets == 1
            assert engine.scan_metrics[scan_id].failed_targets == 1
    
    @pytest.mark.asyncio
    async def test_cancel_scan(self, engine) -> Any:
        """Test scan cancellation"""
        scan_id = "test-scan-123"
        mock_task = AsyncMock()
        engine.active_scans[scan_id] = mock_task
        
        success = await engine.cancel_scan(scan_id)
        
        assert success is True
        assert scan_id not in engine.active_scans
        mock_task.cancel.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cancel_nonexistent_scan(self, engine) -> Any:
        """Test cancelling non-existent scan"""
        success = await engine.cancel_scan("nonexistent")
        
        assert success is False
    
    @pytest.mark.asyncio
    async def test_get_scan_metrics(self, engine) -> Optional[Dict[str, Any]]:
        """Test getting scan metrics"""
        scan_id = "test-scan-123"
        metrics = SecurityMetrics(scan_id=scan_id, start_time=time.time())
        engine.scan_metrics[scan_id] = metrics
        
        result = engine.get_scan_metrics(scan_id)
        
        assert result == metrics
    
    @pytest.mark.asyncio
    async def test_get_scan_metrics_not_found(self, engine) -> Optional[Dict[str, Any]]:
        """Test getting non-existent scan metrics"""
        result = engine.get_scan_metrics("nonexistent")
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_shutdown(self, engine) -> Any:
        """Test engine shutdown"""
        # Add some active scans
        engine.active_scans["scan-1"] = AsyncMock()
        engine.active_scans["scan-2"] = AsyncMock()
        
        await engine.shutdown()
        
        # Check that all scans were cancelled
        for task in engine.active_scans.values():
            task.cancel.assert_called_once()


class TestFastAPIIntegration:
    """Test FastAPI integration"""
    
    @pytest.fixture
    def client(self) -> Any:
        """Create test client"""
        app = FastAPI()
        
        # Add routes
        app.post("/api/v1/scans/production")(start_production_scan)
        app.get("/api/v1/scans/production/{scan_id}")(get_production_scan_status)
        app.delete("/api/v1/scans/production/{scan_id}")(cancel_production_scan)
        app.get("/api/v1/health/production")(production_health_check)
        
        return TestClient(app)
    
    @pytest.mark.asyncio
    async def test_start_production_scan(self, client) -> Any:
        """Test starting production scan"""
        with patch('production_scan_engine.get_production_scan_engine') as mock_get_engine:
            mock_engine = AsyncMock()
            mock_get_engine.return_value = mock_engine
            
            request_data = {
                "targets": [
                    {
                        "url": "example.com",
                        "port": 443,
                        "protocol": "https"
                    }
                ],
                "configuration": {
                    "scan_type": "vulnerability",
                    "max_concurrent_scans": 10
                }
            }
            
            response = client.post("/api/v1/scans/production", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert "scan_id" in data
            assert data["status"] == "pending"
    
    @pytest.mark.asyncio
    async def test_get_production_scan_status(self, client) -> Optional[Dict[str, Any]]:
        """Test getting scan status"""
        with patch('production_scan_engine.get_production_scan_engine') as mock_get_engine:
            mock_engine = AsyncMock()
            mock_engine.get_cached_result.return_value = None
            mock_engine.get_scan_metrics.return_value = SecurityMetrics(
                scan_id="test-123",
                start_time=time.time()
            )
            mock_get_engine.return_value = mock_engine
            
            response = client.get("/api/v1/scans/production/test-123")
            
            assert response.status_code == 200
            data = response.json()
            assert data["scan_id"] == "test-123"
    
    @pytest.mark.asyncio
    async def test_get_production_scan_status_not_found(self, client) -> Optional[Dict[str, Any]]:
        """Test getting non-existent scan status"""
        with patch('production_scan_engine.get_production_scan_engine') as mock_get_engine:
            mock_engine = AsyncMock()
            mock_engine.get_cached_result.return_value = None
            mock_engine.get_scan_metrics.return_value = None
            mock_get_engine.return_value = mock_engine
            
            response = client.get("/api/v1/scans/production/nonexistent")
            
            assert response.status_code == 404
    
    @pytest.mark.asyncio
    async def test_cancel_production_scan(self, client) -> Any:
        """Test cancelling production scan"""
        with patch('production_scan_engine.get_production_scan_engine') as mock_get_engine:
            mock_engine = AsyncMock()
            mock_engine.cancel_scan.return_value = True
            mock_get_engine.return_value = mock_engine
            
            response = client.delete("/api/v1/scans/production/test-123")
            
            assert response.status_code == 200
            data = response.json()
            assert "cancelled successfully" in data["message"]
    
    @pytest.mark.asyncio
    async def test_cancel_production_scan_not_found(self, client) -> Any:
        """Test cancelling non-existent scan"""
        with patch('production_scan_engine.get_production_scan_engine') as mock_get_engine:
            mock_engine = AsyncMock()
            mock_engine.cancel_scan.return_value = False
            mock_get_engine.return_value = mock_engine
            
            response = client.delete("/api/v1/scans/production/nonexistent")
            
            assert response.status_code == 404
    
    @pytest.mark.asyncio
    async def test_production_health_check(self, client) -> Any:
        """Test production health check"""
        with patch('production_scan_engine.get_production_scan_engine') as mock_get_engine:
            mock_engine = AsyncMock()
            mock_engine.active_scans = {"scan-1": None, "scan-2": None}
            mock_engine.scan_metrics = {"scan-1": None, "scan-2": None, "scan-3": None}
            mock_engine.system_monitor.get_system_metrics.return_value = {
                "cpu_percent": 50.0,
                "memory_percent": 60.0
            }
            mock_get_engine.return_value = mock_engine
            
            response = client.get("/api/v1/health/production")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["active_scans"] == 2
            assert data["total_metrics"] == 3


class TestProductionScanRequest:
    """Test production scan request validation"""
    
    async def test_valid_request(self) -> Any:
        """Test valid request creation"""
        request = ProductionScanRequest(
            targets=[
                ScanTarget(url="example.com", port=443),
                ScanTarget(url="test.com", port=80)
            ],
            configuration=ProductionScanConfiguration(
                scan_type=ScanType.VULNERABILITY,
                max_concurrent_scans=10
            )
        )
        
        assert len(request.targets) == 2
        assert request.configuration.scan_type == ScanType.VULNERABILITY
    
    def test_empty_targets(self) -> Optional[Dict[str, Any]]:
        """Test request with empty targets"""
        with pytest.raises(ValidationError, match="At least one target is required"):
            ProductionScanRequest(targets=[])
    
    async def test_request_without_configuration(self) -> Any:
        """Test request without configuration"""
        request = ProductionScanRequest(
            targets=[ScanTarget(url="example.com")]
        )
        
        assert request.configuration is None


class TestProductionScanResponse:
    """Test production scan response"""
    
    def test_response_creation(self) -> Any:
        """Test response creation"""
        response = ProductionScanResponse(
            scan_id="test-123",
            status=ScanStatus.COMPLETED,
            results=[],
            metrics=SecurityMetrics(scan_id="test-123", start_time=time.time()),
            message="Test message",
            correlation_id="corr-123"
        )
        
        assert response.scan_id == "test-123"
        assert response.status == ScanStatus.COMPLETED
        assert response.message == "Test message"
        assert response.correlation_id == "corr-123"


class TestIntegrationScenarios:
    """Test integration scenarios"""
    
    @pytest.mark.asyncio
    async def test_full_scan_workflow(self) -> Any:
        """Test complete scan workflow"""
        config = ProductionScanConfiguration(
            max_concurrent_scans=2,
            timeout_per_target=10,
            enable_prometheus_metrics=False,
            enable_redis_caching=False,
            enable_database_storage=False,
            enable_health_checks=False
        )
        
        engine = ProductionScanEngine(config)
        await engine.initialize()
        
        try:
            targets = [
                ScanTarget(url="example.com", port=443),
                ScanTarget(url="test.com", port=80)
            ]
            scan_id = "integration-test-123"
            
            # Start scan
            results = await engine.scan_targets(targets, scan_id)
            
            # Verify results
            assert len(results) == 2
            assert all(isinstance(r, ScanResult) for r in results)
            
            # Check metrics
            metrics = engine.get_scan_metrics(scan_id)
            assert metrics is not None
            assert metrics.total_targets == 2
            
            # Cancel scan (should not affect completed scan)
            success = await engine.cancel_scan(scan_id)
            assert success is False  # Scan already completed
            
        finally:
            await engine.shutdown()
    
    @pytest.mark.asyncio
    async def test_error_handling_scenarios(self) -> Any:
        """Test various error handling scenarios"""
        config = ProductionScanConfiguration(
            max_concurrent_scans=1,
            timeout_per_target=5,
            enable_prometheus_metrics=False,
            enable_redis_caching=False,
            enable_database_storage=False,
            enable_health_checks=False
        )
        
        engine = ProductionScanEngine(config)
        await engine.initialize()
        
        try:
            # Test with invalid target
            targets = [ScanTarget(url="invalid-url", port=99999)]
            scan_id = "error-test-123"
            
            results = await engine.scan_targets(targets, scan_id)
            
            # Should handle errors gracefully
            assert len(results) == 1
            assert results[0].status in [ScanStatus.FAILED, ScanStatus.TIMEOUT]
            
        finally:
            await engine.shutdown()


class TestPerformanceMetrics:
    """Test performance metrics and monitoring"""
    
    @pytest.mark.asyncio
    async def test_scan_performance_tracking(self) -> Any:
        """Test scan performance tracking"""
        config = ProductionScanConfiguration(
            max_concurrent_scans=5,
            timeout_per_target=30,
            enable_prometheus_metrics=True,
            enable_redis_caching=False,
            enable_database_storage=False,
            enable_health_checks=False
        )
        
        engine = ProductionScanEngine(config)
        await engine.initialize()
        
        try:
            targets = [ScanTarget(url="example.com") for _ in range(3)]
            scan_id = "perf-test-123"
            
            start_time = time.time()
            results = await engine.scan_targets(targets, scan_id)
            end_time = time.time()
            
            # Check performance metrics
            metrics = engine.get_scan_metrics(scan_id)
            assert metrics is not None
            assert metrics.scan_duration > 0
            assert metrics.throughput > 0
            assert metrics.completed_targets == 3
            
            # Verify actual duration matches metrics
            actual_duration = end_time - start_time
            assert abs(metrics.scan_duration - actual_duration) < 1.0  # Within 1 second
            
        finally:
            await engine.shutdown()
    
    @pytest.mark.asyncio
    async def test_concurrent_scan_limits(self) -> Any:
        """Test concurrent scan limits"""
        config = ProductionScanConfiguration(
            max_concurrent_scans=2,
            timeout_per_target=5,
            enable_prometheus_metrics=False,
            enable_redis_caching=False,
            enable_database_storage=False,
            enable_health_checks=False
        )
        
        engine = ProductionScanEngine(config)
        await engine.initialize()
        
        try:
            # Create targets that will take time to process
            targets = [ScanTarget(url=f"test{i}.com") for i in range(5)]
            scan_id = "concurrent-test-123"
            
            start_time = time.time()
            results = await engine.scan_targets(targets, scan_id)
            end_time = time.time()
            
            # Should respect concurrent limits
            metrics = engine.get_scan_metrics(scan_id)
            assert metrics.completed_targets == 5
            
            # Duration should reflect concurrent processing
            # With 2 concurrent scans, 5 targets should take at least 2.5 time units
            # (assuming each scan takes 1 time unit)
            
        finally:
            await engine.shutdown()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"]) 