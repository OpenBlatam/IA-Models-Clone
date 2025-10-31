from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
TIMEOUT_SECONDS = 60

import asyncio
import time
import json
from datetime import datetime
from typing import Dict, List, Any
import structlog
from optimized_scan_engine import (
from typing import Any, List, Dict, Optional
import logging
"""
Optimized Scan Engine Demo

Comprehensive demonstration of the production-ready cybersecurity scanning engine:
- Real-world scan configurations
- Performance monitoring and metrics
- Error handling and recovery
- Integration with FastAPI
- Structured logging examples
- Security best practices implementation
"""


    OptimizedScanEngine,
    ScanConfig,
    ScanRequest,
    FindingSeverity,
    start_security_scan,
    get_scan_metrics,
    get_overall_metrics,
    health_check
)

# Configure structured logging for demo
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

# =============================================================================
# Demo Configuration
# =============================================================================

class DemoConfig:
    """Demo configuration and test data."""
    
    # Sample targets for testing
    SAMPLE_TARGETS = [
        "https://httpbin.org/get",
        "https://httpbin.org/status/200",
        "https://httpbin.org/status/404",
        "https://httpbin.org/delay/1",
        "https://httpbin.org/delay/2"
    ]
    
    # Security test targets
    SECURITY_TEST_TARGETS = [
        "https://example.com",
        "https://httpbin.org/headers",
        "https://httpbin.org/user-agent",
        "https://httpbin.org/ip"
    ]
    
    # Performance test targets
    PERFORMANCE_TEST_TARGETS = [
        f"https://httpbin.org/delay/{i}" for i in range(1, 11)
    ]
    
    # Scan configurations
    BASIC_SCAN_CONFIG = ScanConfig(
        targets=SAMPLE_TARGETS[:3],
        scan_type="vulnerability",
        max_concurrent_scans=3,
        timeout_per_target=10.0,
        rate_limit_per_second=2
    )
    
    SECURITY_SCAN_CONFIG = ScanConfig(
        targets=SECURITY_TEST_TARGETS,
        scan_type="web",
        max_concurrent_scans=5,
        timeout_per_target=15.0,
        rate_limit_per_second=3,
        enable_ssl_verification=True,
        custom_headers={
            "User-Agent": "SecurityScanner/1.0",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
        }
    )
    
    PERFORMANCE_SCAN_CONFIG = ScanConfig(
        targets=PERFORMANCE_TEST_TARGETS,
        scan_type="network",
        max_concurrent_scans=10,
        timeout_per_target=30.0,
        rate_limit_per_second=5
    )

# =============================================================================
# Demo Functions
# =============================================================================

async def demo_basic_scan_workflow():
    """Demonstrate basic scan workflow."""
    logger.info("Starting basic scan workflow demo")
    
    # Initialize scan engine
    scan_engine = OptimizedScanEngine()
    
    # Create mock dependencies (in real app, these would be injected)
    mock_http_session = create_mock_http_session()
    mock_crypto_backend = create_mock_crypto_backend()
    
    # Start scan
    scan_id, metrics = await scan_engine.start_scan(
        DemoConfig.BASIC_SCAN_CONFIG,
        mock_http_session,
        mock_crypto_backend,
        "demo-user-123"
    )
    
    logger.info("Scan started", 
               scan_id=scan_id,
               total_targets=metrics.total_targets,
               max_concurrent=DemoConfig.BASIC_SCAN_CONFIG.max_concurrent_scans)
    
    # Wait for scan to complete
    await asyncio.sleep(2)
    
    # Get results
    completed_metrics = scan_engine.get_scan_metrics(scan_id)
    
    logger.info("Scan completed",
               scan_id=scan_id,
               duration=completed_metrics.scan_duration,
               findings=completed_metrics.findings_count,
               success_rate=completed_metrics.scan_success_rate)
    
    return scan_id, completed_metrics

async def demo_security_scan_with_findings():
    """Demonstrate security scan with vulnerability detection."""
    logger.info("Starting security scan demo")
    
    scan_engine = OptimizedScanEngine()
    mock_http_session = create_mock_http_session_with_vulnerabilities()
    mock_crypto_backend = create_mock_crypto_backend()
    
    # Start security scan
    scan_id, metrics = await scan_engine.start_scan(
        DemoConfig.SECURITY_SCAN_CONFIG,
        mock_http_session,
        mock_crypto_backend,
        "security-analyst-456"
    )
    
    logger.info("Security scan started",
               scan_id=scan_id,
               scan_type=DemoConfig.SECURITY_SCAN_CONFIG.scan_type,
               targets_count=len(DemoConfig.SECURITY_SCAN_CONFIG.targets))
    
    # Wait for scan to complete
    await asyncio.sleep(3)
    
    # Get results
    completed_metrics = scan_engine.get_scan_metrics(scan_id)
    
    logger.info("Security scan completed",
               scan_id=scan_id,
               findings=completed_metrics.findings_count,
               false_positive_rate=completed_metrics.false_positive_rate,
               average_response_time=completed_metrics.average_response_time)
    
    return scan_id, completed_metrics

async def demo_performance_scan():
    """Demonstrate performance scanning with large target set."""
    logger.info("Starting performance scan demo")
    
    scan_engine = OptimizedScanEngine()
    mock_http_session = create_mock_http_session()
    mock_crypto_backend = create_mock_crypto_backend()
    
    start_time = time.time()
    
    # Start performance scan
    scan_id, metrics = await scan_engine.start_scan(
        DemoConfig.PERFORMANCE_SCAN_CONFIG,
        mock_http_session,
        mock_crypto_backend,
        "performance-tester-789"
    )
    
    logger.info("Performance scan started",
               scan_id=scan_id,
               targets_count=len(DemoConfig.PERFORMANCE_SCAN_CONFIG.targets),
               max_concurrent=DemoConfig.PERFORMANCE_SCAN_CONFIG.max_concurrent_scans)
    
    # Wait for scan to complete
    await asyncio.sleep(5)
    
    # Get results
    completed_metrics = scan_engine.get_scan_metrics(scan_id)
    total_time = time.time() - start_time
    
    logger.info("Performance scan completed",
               scan_id=scan_id,
               total_time=total_time,
               scan_duration=completed_metrics.scan_duration,
               throughput=len(DemoConfig.PERFORMANCE_SCAN_CONFIG.targets) / total_time,
               success_rate=completed_metrics.scan_success_rate)
    
    return scan_id, completed_metrics

async def demo_concurrent_scans():
    """Demonstrate handling multiple concurrent scans."""
    logger.info("Starting concurrent scans demo")
    
    scan_engine = OptimizedScanEngine()
    mock_http_session = create_mock_http_session()
    mock_crypto_backend = create_mock_crypto_backend()
    
    # Create multiple scan configurations
    scan_configs = [
        ScanConfig(
            targets=[f"https://site{i}.com" for i in range(1, 6)],
            scan_type="vulnerability",
            max_concurrent_scans=3
        ) for _ in range(3)
    ]
    
    # Start multiple scans concurrently
    tasks = []
    for i, config in enumerate(scan_configs):
        task = scan_engine.start_scan(
            config,
            mock_http_session,
            mock_crypto_backend,
            f"concurrent-user-{i}"
        )
        tasks.append(task)
    
    # Wait for all scans to start
    scan_results = await asyncio.gather(*tasks)
    
    logger.info("Concurrent scans started",
               scan_count=len(scan_results),
               active_scans=len(scan_engine.active_scans))
    
    # Wait for all scans to complete
    await asyncio.sleep(3)
    
    # Get overall metrics
    overall_metrics = scan_engine.get_all_metrics()
    
    logger.info("Concurrent scans completed",
               active_scans=overall_metrics["active_scans"],
               completed_scans=overall_metrics["completed_scans"],
               total_findings=overall_metrics["total_findings"])
    
    return [scan_id for scan_id, _ in scan_results], overall_metrics

async def demo_error_handling():
    """Demonstrate error handling and recovery."""
    logger.info("Starting error handling demo")
    
    scan_engine = OptimizedScanEngine()
    mock_http_session = create_mock_http_session_with_errors()
    mock_crypto_backend = create_mock_crypto_backend()
    
    # Create config with problematic targets
    error_config = ScanConfig(
        targets=[
            "https://invalid-site-12345.com",
            "https://timeout-site.com",
            "https://error-site.com",
            "https://httpbin.org/status/500"
        ],
        scan_type="vulnerability",
        timeout_per_target=5.0
    )
    
    # Start scan
    scan_id, metrics = await scan_engine.start_scan(
        error_config,
        mock_http_session,
        mock_crypto_backend,
        "error-tester-999"
    )
    
    logger.info("Error handling scan started",
               scan_id=scan_id,
               targets_count=len(error_config.targets))
    
    # Wait for scan to complete
    await asyncio.sleep(2)
    
    # Get results
    completed_metrics = scan_engine.get_scan_metrics(scan_id)
    
    logger.info("Error handling scan completed",
               scan_id=scan_id,
               errors=completed_metrics.error_count,
               success_rate=completed_metrics.scan_success_rate,
               scanned_targets=completed_metrics.scanned_targets)
    
    return scan_id, completed_metrics

async def demo_metrics_and_monitoring():
    """Demonstrate metrics collection and monitoring."""
    logger.info("Starting metrics and monitoring demo")
    
    scan_engine = OptimizedScanEngine()
    mock_http_session = create_mock_http_session()
    mock_crypto_backend = create_mock_crypto_backend()
    
    # Run multiple scans to generate metrics
    scan_configs = [
        ScanConfig(
            targets=[f"https://metrics-test-{i}.com" for i in range(1, 4)],
            scan_type="vulnerability"
        ) for _ in range(5)
    ]
    
    # Start scans
    for i, config in enumerate(scan_configs):
        await scan_engine.start_scan(
            config,
            mock_http_session,
            mock_crypto_backend,
            f"metrics-user-{i}"
        )
    
    # Wait for scans to complete
    await asyncio.sleep(3)
    
    # Get comprehensive metrics
    overall_metrics = scan_engine.get_all_metrics()
    
    logger.info("Metrics collection completed",
               active_scans=overall_metrics["active_scans"],
               completed_scans=overall_metrics["completed_scans"],
               total_findings=overall_metrics["total_findings"],
               average_scan_duration=overall_metrics["average_scan_duration"],
               false_positive_rate=overall_metrics["overall_false_positive_rate"])
    
    # Simulate health check
    health_status = await health_check()
    
    logger.info("Health check result",
               status=health_status["status"],
               active_scans=health_status["active_scans"],
               total_errors=health_status["total_errors"])
    
    return overall_metrics, health_status

async def demo_fastapi_integration():
    """Demonstrate FastAPI integration patterns."""
    logger.info("Starting FastAPI integration demo")
    
    mock_http_session = create_mock_http_session()
    mock_crypto_backend = create_mock_crypto_backend()
    
    # Create scan request
    request = ScanRequest(
        config=DemoConfig.BASIC_SCAN_CONFIG,
        user_id="fastapi-user-123"
    )
    
    # Start scan via FastAPI function
    response = await start_security_scan(
        request,
        mock_http_session,
        mock_crypto_backend
    )
    
    logger.info("FastAPI scan started",
               scan_id=response.scan_id,
               status=response.status,
               message=response.message)
    
    # Wait for scan to complete
    await asyncio.sleep(2)
    
    # Get metrics via FastAPI function
    metrics = await get_scan_metrics(response.scan_id)
    
    logger.info("FastAPI scan completed",
               scan_id=response.scan_id,
               duration=metrics["scan_duration"],
               findings=metrics["findings_count"])
    
    return response.scan_id, metrics

# =============================================================================
# Mock Dependencies
# =============================================================================

def create_mock_http_session():
    """Create mock HTTP session for demo."""
    class MockHTTPSession:
        async def get(self, url, timeout=None) -> Optional[Dict[str, Any]]:
            class MockResponse:
                def __init__(self, url) -> Any:
                    self.url = url
                    self.status = 200
                    self.headers = {
                        "X-Frame-Options": "DENY",
                        "X-Content-Type-Options": "nosniff",
                        "Content-Type": "text/html"
                    }
                
                async def text(self) -> Any:
                    return f"<html><body>Welcome to {self.url}</body></html>"
                
                async def __aenter__(self) -> Any:
                    return self
                
                async def __aexit__(self, exc_type, exc_val, exc_tb) -> Any:
                    pass
            
            return MockResponse(url)
    
    return MockHTTPSession()

def create_mock_http_session_with_vulnerabilities():
    """Create mock HTTP session that simulates vulnerabilities."""
    class MockHTTPSession:
        async def get(self, url, timeout=None) -> Optional[Dict[str, Any]]:
            class MockResponse:
                def __init__(self, url) -> Any:
                    self.url = url
                    self.status = 200
                    self.headers = {
                        "Content-Type": "text/html"
                        # Missing security headers to trigger findings
                    }
                
                async def text(self) -> Any:
                    # Include admin interface to trigger finding
                    return f"<html><body>Admin Panel: {self.url}/admin</body></html>"
                
                async def __aenter__(self) -> Any:
                    return self
                
                async def __aexit__(self, exc_type, exc_val, exc_tb) -> Any:
                    pass
            
            return MockResponse(url)
    
    return MockHTTPSession()

def create_mock_http_session_with_errors():
    """Create mock HTTP session that simulates various errors."""
    class MockHTTPSession:
        def __init__(self) -> Any:
            self.request_count = 0
        
        async def get(self, url, timeout=None) -> Optional[Dict[str, Any]]:
            self.request_count += 1
            
            if "invalid-site" in url:
                raise Exception("DNS resolution failed")
            elif "timeout-site" in url:
                await asyncio.sleep(timeout + 1)  # Exceed timeout
                raise Exception("Request timeout")
            elif "error-site" in url:
                raise Exception("Connection refused")
            else:
                class MockResponse:
                    def __init__(self, url) -> Any:
                        self.url = url
                        self.status = 500 if "500" in url else 200
                        self.headers = {}
                    
                    async def text(self) -> Any:
                        return "Error response" if self.status == 500 else "OK"
                    
                    async def __aenter__(self) -> Any:
                        return self
                    
                    async def __aexit__(self, exc_type, exc_val, exc_tb) -> Any:
                        pass
                
                return MockResponse(url)
    
    return MockHTTPSession()

def create_mock_crypto_backend():
    """Create mock crypto backend for demo."""
    class MockCryptoBackend:
        async def encrypt(self, data) -> Any:
            return b"encrypted_" + data
        
        async def decrypt(self, data) -> Any:
            return data.replace(b"encrypted_", b"")
    
    return MockCryptoBackend()

# =============================================================================
# Demo Runner
# =============================================================================

async def run_comprehensive_demo():
    """Run comprehensive demo of all features."""
    logger.info("Starting comprehensive optimized scan engine demo")
    
    demo_results = {}
    
    try:
        # Basic scan workflow
        logger.info("=" * 60)
        logger.info("DEMO 1: Basic Scan Workflow")
        logger.info("=" * 60)
        scan_id, metrics = await demo_basic_scan_workflow()
        demo_results["basic_scan"] = {"scan_id": scan_id, "metrics": metrics.as_dict()}
        
        # Security scan with findings
        logger.info("=" * 60)
        logger.info("DEMO 2: Security Scan with Findings")
        logger.info("=" * 60)
        scan_id, metrics = await demo_security_scan_with_findings()
        demo_results["security_scan"] = {"scan_id": scan_id, "metrics": metrics.as_dict()}
        
        # Performance scan
        logger.info("=" * 60)
        logger.info("DEMO 3: Performance Scan")
        logger.info("=" * 60)
        scan_id, metrics = await demo_performance_scan()
        demo_results["performance_scan"] = {"scan_id": scan_id, "metrics": metrics.as_dict()}
        
        # Concurrent scans
        logger.info("=" * 60)
        logger.info("DEMO 4: Concurrent Scans")
        logger.info("=" * 60)
        scan_ids, overall_metrics = await demo_concurrent_scans()
        demo_results["concurrent_scans"] = {"scan_ids": scan_ids, "overall_metrics": overall_metrics}
        
        # Error handling
        logger.info("=" * 60)
        logger.info("DEMO 5: Error Handling")
        logger.info("=" * 60)
        scan_id, metrics = await demo_error_handling()
        demo_results["error_handling"] = {"scan_id": scan_id, "metrics": metrics.as_dict()}
        
        # Metrics and monitoring
        logger.info("=" * 60)
        logger.info("DEMO 6: Metrics and Monitoring")
        logger.info("=" * 60)
        overall_metrics, health_status = await demo_metrics_and_monitoring()
        demo_results["metrics"] = {"overall_metrics": overall_metrics, "health_status": health_status}
        
        # FastAPI integration
        logger.info("=" * 60)
        logger.info("DEMO 7: FastAPI Integration")
        logger.info("=" * 60)
        scan_id, metrics = await demo_fastapi_integration()
        demo_results["fastapi_integration"] = {"scan_id": scan_id, "metrics": metrics}
        
        # Demo summary
        logger.info("=" * 60)
        logger.info("DEMO SUMMARY")
        logger.info("=" * 60)
        
        total_scans = len(demo_results)
        total_findings = sum(
            result["metrics"]["findings_count"] 
            for result in demo_results.values() 
            if "metrics" in result and "findings_count" in result["metrics"]
        )
        
        logger.info("Demo completed successfully",
                   total_demos=total_scans,
                   total_findings=total_findings,
                   demo_results=json.dumps(demo_results, indent=2, default=str))
        
        return demo_results
        
    except Exception as e:
        logger.error("Demo failed", error=str(e))
        raise

def print_demo_instructions():
    """Print demo instructions."""
    print("""
Optimized Scan Engine Demo
==========================

This demo showcases a production-ready cybersecurity scanning engine with:

1. Basic Scan Workflow
   - Simple vulnerability scanning
   - Metrics collection
   - Structured logging

2. Security Scan with Findings
   - Vulnerability detection
   - False positive tracking
   - Security header analysis

3. Performance Scan
   - Large target set handling
   - Concurrent execution
   - Performance metrics

4. Concurrent Scans
   - Multiple simultaneous scans
   - Resource management
   - Overall metrics

5. Error Handling
   - Network failures
   - Timeouts
   - Invalid targets

6. Metrics and Monitoring
   - Comprehensive metrics
   - Health checks
   - Performance monitoring

7. FastAPI Integration
   - REST API patterns
   - Dependency injection
   - Error handling

Features Demonstrated:
- Dependency injection for shared resources
- Measurable security metrics (scan time, false-positive rate)
- Non-blocking async operations with dedicated I/O helpers
- Structured JSON logging for SIEM integration
- Comprehensive edge case testing
- OWASP/NIST security best practices

Run the demo with: python optimized_scan_engine_demo.py
""")

# =============================================================================
# Main Execution
# =============================================================================

async def main():
    """Main demo execution."""
    print_demo_instructions()
    
    print("\nStarting demo in 3 seconds...")
    await asyncio.sleep(3)
    
    try:
        results = await run_comprehensive_demo()
        print("\n✅ Demo completed successfully!")
        print(f"Results saved with {len(results)} demo scenarios")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        logger.error("Demo execution failed", error=str(e))

match __name__:
    case "__main__":
    asyncio.run(main()) 