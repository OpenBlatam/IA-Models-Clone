from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import pytest
import asyncio
import tempfile
import os
import json
from typing import Dict, Any, List, Optional, Callable
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass
from . import TEST_CONFIG, get_test_config, create_mock_network_response
import sys
import os
from cybersecurity.scanners.async_helpers import AsyncHelperManager, AsyncHelperConfig
from cybersecurity.scanners.port_scanner import PortScanConfig, PortScanner
from cybersecurity.core.non_blocking_scanner import NonBlockingScanner, NonBlockingScanConfig
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
Pytest Configuration and Fixtures
Configuration for automated testing with pytest and pytest-asyncio.
"""


# Import test configuration

# Import cybersecurity modules for testing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class MockNetworkConfig:
    """Configuration for mock network responses."""
    success_rate: float = 0.8
    delay_range: tuple = (0.05, 0.2)
    error_types: List[str] = None
    timeout_rate: float = 0.05
    
    def __post_init__(self) -> Any:
        if self.error_types is None:
            self.error_types = ["connection_refused", "timeout", "dns_error", "ssl_error"]

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def test_config():
    """Provide test configuration."""
    return get_test_config()

@pytest.fixture
def mock_network_config():
    """Provide mock network configuration."""
    return MockNetworkConfig()

@pytest.fixture
def temp_file():
    """Create a temporary file for testing."""
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f:
        f.write('{"test": "data", "number": 42}')
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        f.flush()
        yield f.name
    os.unlink(f.name)

@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir

@pytest.fixture
def async_helper_config():
    """Provide async helper configuration for testing."""
    return AsyncHelperConfig(
        timeout=TEST_CONFIG["timeout"],
        max_workers=TEST_CONFIG["max_workers"],
        retry_attempts=TEST_CONFIG["retry_attempts"],
        chunk_size=TEST_CONFIG["chunk_size"]
    )

@pytest.fixture
def port_scan_config():
    """Provide port scan configuration for testing."""
    return PortScanConfig(
        timeout=1.0,
        max_workers=5,
        retry_count=1,
        banner_grab=True,
        ssl_check=True
    )

@pytest.fixture
def non_blocking_scan_config():
    """Provide non-blocking scan configuration for testing."""
    return NonBlockingScanConfig(
        max_concurrent_scans=10,
        scan_timeout=5.0,
        chunk_size=50,
        enable_dns_cache=True,
        enable_result_cache=True,
        cache_ttl=300
    )

@pytest.fixture
async def async_helper_manager(async_helper_config) -> Any:
    """Provide async helper manager for testing."""
    manager = AsyncHelperManager(async_helper_config)
    yield manager
    await manager.close()

@pytest.fixture
async def port_scanner(port_scan_config) -> Any:
    """Provide port scanner for testing."""
    scanner = PortScanner(port_scan_config)
    yield scanner
    await scanner.close()

@pytest.fixture
async def non_blocking_scanner(non_blocking_scan_config) -> Any:
    """Provide non-blocking scanner for testing."""
    scanner = NonBlockingScanner(non_blocking_scan_config)
    yield scanner
    await scanner.close()

@pytest.fixture
def mock_socket():
    """Mock socket for network testing."""
    mock_sock = MagicMock()
    mock_sock.connect_ex.return_value = 0  # Success
    mock_sock.recv.return_value = b"HTTP/1.1 200 OK\r\n\r\n"
    mock_sock.send.return_value = 20
    mock_sock.close.return_value = None
    return mock_sock

@pytest.fixture
def mock_ssl_context():
    """Mock SSL context for testing."""
    mock_context = MagicMock()
    mock_context.wrap_socket.return_value.__enter__.return_value.getpeercert.return_value = {
        "subject": [("commonName", "test.example.com")],
        "issuer": [("commonName", "Test CA")],
        "version": 3,
        "serialNumber": "123456789",
        "notBefore": "Jan 1 00:00:00 2023 GMT",
        "notAfter": "Jan 1 00:00:00 2024 GMT"
    }
    return mock_context

@pytest.fixture
def mock_aiohttp_session():
    """Mock aiohttp session for testing."""
    mock_session = AsyncMock()
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.headers = {"content-type": "text/html"}
    mock_response.text.return_value = "<html>Test</html>"
    mock_response.read.return_value = b"<html>Test</html>"
    mock_response.__aenter__.return_value = mock_response
    mock_response.__aexit__.return_value = None
    mock_session.request.return_value = mock_response
    return mock_session

@pytest.fixture
def mock_httpx_client():
    """Mock httpx client for testing."""
    mock_client = AsyncMock()
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "text/html"}
    mock_response.text = "<html>Test</html>"
    mock_client.request.return_value = mock_response
    return mock_client

@pytest.fixture
def mock_dns_response():
    """Mock DNS response for testing."""
    return "192.168.1.1"

@pytest.fixture
def mock_network_responses():
    """Provide various mock network responses for testing edge cases."""
    return {
        "success": create_mock_network_response(True, 0.1, {"status": "ok"}),
        "timeout": create_mock_network_response(False, 5.0, None, "timeout"),
        "connection_refused": create_mock_network_response(False, 0.1, None, "connection refused"),
        "dns_error": create_mock_network_response(False, 0.1, None, "dns resolution failed"),
        "ssl_error": create_mock_network_response(False, 0.1, None, "ssl handshake failed"),
        "partial_response": create_mock_network_response(True, 0.1, {"partial": "data"}),
        "large_response": create_mock_network_response(True, 0.1, {"data": "x" * 10000}),
        "empty_response": create_mock_network_response(True, 0.1, {}),
        "malformed_response": create_mock_network_response(True, 0.1, "invalid json"),
        "slow_response": create_mock_network_response(True, 2.0, {"status": "slow"}),
        "fast_response": create_mock_network_response(True, 0.01, {"status": "fast"})
    }

@pytest.fixture
def edge_case_targets():
    """Provide edge case targets for testing."""
    return [
        "localhost",  # Local target
        "127.0.0.1",  # Loopback
        "0.0.0.0",  # Invalid target
        "255.255.255.255",  # Broadcast
        "192.168.1.1",  # Private network
        "8.8.8.8",  # Public DNS
        "invalid-hostname",  # Invalid hostname
        "very-long-hostname-that-exceeds-normal-limits.example.com",  # Long hostname
        "192.168.1.256",  # Invalid IP
        "fe80::1",  # IPv6
        "",  # Empty string
        "   ",  # Whitespace only
        "192.168.1.1:8080",  # IP with port
        "example.com:443",  # Hostname with port
        "http://example.com",  # URL format
        "https://example.com",  # HTTPS URL
        "ftp://example.com",  # FTP URL
        "ssh://example.com",  # SSH URL
        "telnet://example.com",  # Telnet URL
        "smtp://example.com"  # SMTP URL
    ]

@pytest.fixture
def edge_case_ports():
    """Provide edge case ports for testing."""
    return [
        0,  # Reserved port
        1,  # System port
        22,  # SSH
        80,  # HTTP
        443,  # HTTPS
        1024,  # User port
        65535,  # Maximum port
        65536,  # Invalid port
        -1,  # Negative port
        99999,  # Very large port
        8080,  # Common alternative HTTP
        8443,  # Common alternative HTTPS
        3306,  # MySQL
        5432,  # PostgreSQL
        27017,  # MongoDB
        6379,  # Redis
        21,  # FTP
        23,  # Telnet
        25,  # SMTP
        53,  # DNS
        110,  # POP3
        143,  # IMAP
        993,  # IMAPS
        995  # POP3S
    ]

@pytest.fixture
def performance_test_data():
    """Provide performance test data."""
    return {
        "small_dataset": [f"target_{i}.com" for i in range(10)],
        "medium_dataset": [f"target_{i}.com" for i in range(100)],
        "large_dataset": [f"target_{i}.com" for i in range(1000)],
        "mixed_targets": [
            "localhost", "127.0.0.1", "google.com", "github.com",
            "stackoverflow.com", "invalid-host", "192.168.1.1"
        ] * 50,  # 350 targets
        "common_ports": [22, 23, 25, 53, 80, 110, 143, 443, 993, 995, 3306, 5432, 8080],
        "extended_ports": list(range(1, 1025)),  # First 1024 ports
        "ssl_ports": [443, 993, 995, 8443, 9443]
    }

@pytest.fixture
def mock_network_patcher():
    """Provide a context manager for mocking network operations."""
    class NetworkPatcher:
        def __init__(self) -> Any:
            self.patches = []
        
        def patch_socket(self, mock_socket) -> Any:
            """Patch socket operations."""
            patch_socket = patch('socket.socket', return_value=mock_socket)
            self.patches.append(patch_socket)
            return patch_socket
        
        def patch_ssl(self, mock_ssl_context) -> Any:
            """Patch SSL operations."""
            patch_ssl = patch('ssl.create_default_context', return_value=mock_ssl_context)
            self.patches.append(patch_ssl)
            return patch_ssl
        
        async def patch_aiohttp(self, mock_session) -> Any:
            """Patch aiohttp operations."""
            patch_aiohttp = patch('aiohttp.ClientSession', return_value=mock_session)
            self.patches.append(patch_aiohttp)
            return patch_aiohttp
        
        async def patch_httpx(self, mock_client) -> Any:
            """Patch httpx operations."""
            patch_httpx = patch('httpx.AsyncClient', return_value=mock_client)
            self.patches.append(patch_httpx)
            return patch_httpx
        
        def patch_dns(self, mock_dns_response) -> Any:
            """Patch DNS operations."""
            patch_dns = patch('socket.gethostbyname', return_value=mock_dns_response)
            self.patches.append(patch_dns)
            return patch_dns
        
        def __enter__(self) -> Any:
            """Start all patches."""
            for patch_obj in self.patches:
                patch_obj.start()
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb) -> Any:
            """Stop all patches."""
            for patch_obj in self.patches:
                patch_obj.stop()
    
    return NetworkPatcher()

@pytest.fixture
def async_test_runner():
    """Provide an async test runner utility."""
    class AsyncTestRunner:
        def __init__(self) -> Any:
            self.results = []
            self.errors = []
        
        async def run_concurrent_tests(self, test_functions: List[Callable], 
                                     max_concurrent: int = 5):
            """Run multiple test functions concurrently."""
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def run_with_semaphore(func) -> Any:
                async with semaphore:
                    try:
                        result = await func()
                        self.results.append(result)
                        return result
                    except Exception as e:
                        self.errors.append(e)
                        raise
            
            tasks = [run_with_semaphore(func) for func in test_functions]
            return await asyncio.gather(*tasks, return_exceptions=True)
        
        async def run_with_timeout(self, test_func: Callable, timeout: float = 10.0):
            """Run a test function with timeout."""
            try:
                return await asyncio.wait_for(test_func(), timeout=timeout)
            except asyncio.TimeoutError:
                raise TimeoutError(f"Test timed out after {timeout} seconds")
        
        def get_results_summary(self) -> Optional[Dict[str, Any]]:
            """Get summary of test results."""
            return {
                "total_results": len(self.results),
                "total_errors": len(self.errors),
                "success_rate": len(self.results) / (len(self.results) + len(self.errors)) if (len(self.results) + len(self.errors)) > 0 else 0,
                "results": self.results,
                "errors": [str(e) for e in self.errors]
            }
    
    return AsyncTestRunner()

# Pytest markers for test organization
pytest_plugins = ["pytest_asyncio"]

def pytest_configure(config) -> Any:
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "network: marks tests that require network access"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "edge_case: marks tests that test edge cases"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )
    config.addinivalue_line(
        "markers", "security: marks tests as security tests"
    )

def pytest_collection_modifyitems(config, items) -> Any:
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Add markers based on test name patterns
        if "test_async" in item.name:
            item.add_marker(pytest.mark.asyncio)
        if "test_network" in item.name:
            item.add_marker(pytest.mark.network)
        if "test_edge" in item.name:
            item.add_marker(pytest.mark.edge_case)
        if "test_performance" in item.name:
            item.add_marker(pytest.mark.performance)
        if "test_security" in item.name:
            item.add_marker(pytest.mark.security)
        if "test_integration" in item.name:
            item.add_marker(pytest.mark.integration)
        if "test_unit" in item.name:
            item.add_marker(pytest.mark.unit) 