from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import pytest
import asyncio
from unittest.mock import patch, MagicMock
from security_toolkit_fixed import (
from typing import Any, List, Dict, Optional
import logging
Unit Tests for Security Toolkit


    scan_ports_basic, run_ssh_command, make_http_request,
    AsyncRateLimiter, retry_with_backoff, get_common_ports,
    chunked, process_batch_async, get_secret,
    NetworkLayerFactory, log_operation, measure_scan_time,
    ScanRequest, SSHRequest, HTTPRequest
)

# ============================================================================
# Test Port Scanning
# ============================================================================

def test_scan_ports_basic_success():
    
    """test_scan_ports_basic_success function."""
result = scan_ports_basic({
        target:12701
        ports":80443,
       scan_type": "tcp,
       timeout: 1,       max_workers":2})
    
    assert result[success'] is True
    assert resulttarget'] == '1270   assert 'summary' in result
    assert 'results' in result

def test_scan_ports_basic_missing_target():
    
    """test_scan_ports_basic_missing_target function."""
result = scan_ports_basic({})
    assert error' in result
    assert result['error'] == 'Target is required'

def test_scan_ports_basic_invalid_target():
    
    """test_scan_ports_basic_invalid_target function."""
result = scan_ports_basic({
        target:invalid_target,
        ports: [80]
    })
    asserterror' in result

def test_scan_ports_basic_invalid_ports():
    
    """test_scan_ports_basic_invalid_ports function."""
result = scan_ports_basic({
        target:12701
        ports": [70000]  # Invalid port
    })
    asserterror' in result

# ============================================================================
# Test SSH Operations
# ============================================================================

@pytest.mark.asyncio
async def test_run_ssh_command_success():
    
    """test_run_ssh_command_success function."""
with patch('security_toolkit_fixed.asyncssh.connect) as mock_connect:
        mock_conn = MagicMock()
        mock_result = MagicMock()
        mock_result.stdout = "test output"
        mock_result.stderr = ""
        mock_result.exit_status = 0
        mock_conn.run.return_value = mock_result
        mock_conn.__aenter__.return_value = mock_conn
        mock_conn.__aexit__.return_value = None
        mock_connect.return_value = mock_conn
        
        result = await run_ssh_command([object Object]
            host:12701,
           username": test          password": "test",
            command": "echo test",
           timeout": 10
        })
        
        assert result[success] is True
        assert result[stdout] == "test output"
        assert result['exit_code'] == 0
@pytest.mark.asyncio
async def test_run_ssh_command_missing_host():
    
    """test_run_ssh_command_missing_host function."""
result = await run_ssh_command({})
    assert error' in result
    assert result['error'] ==Host is required'

# ============================================================================
# Test HTTP Operations
# ============================================================================

@pytest.mark.asyncio
async def test_make_http_request_success():
    
    """test_make_http_request_success function."""
with patch('security_toolkit_fixed.httpx.AsyncClient') as mock_client:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "test response"
        mock_response.headers = {"content-type": "text/plain"}
        
        mock_client_instance = MagicMock()
        mock_client_instance.request.return_value = mock_response
        mock_client_instance.__aenter__.return_value = mock_client_instance
        mock_client_instance.__aexit__.return_value = None
        mock_client.return_value = mock_client_instance
        
        result = await make_http_request({
          url": "https://httpbin.org/get",
           method":GET           timeout": 10
        })
        
        assert result[success] is True
        assert resultstatus_code'] == 200     assert result['body'] ==test response
@pytest.mark.asyncio
async def test_make_http_request_missing_url():
    
    """test_make_http_request_missing_url function."""
result = await make_http_request({})
    assert error' in result
    assert result['error'] == 'URL is required'

# ============================================================================
# Test Rate Limiting
# ============================================================================

@pytest.mark.asyncio
async def test_async_rate_limiter():
    
    """test_async_rate_limiter function."""
limiter = AsyncRateLimiter(max_calls_per_second=2)
    start_time = time.time()
    
    # Should not block for first call
    await limiter.acquire()
    first_call_time = time.time() - start_time
    assert first_call_time < 0.1
    
    # Second call should also not block significantly
    await limiter.acquire()
    second_call_time = time.time() - start_time
    assert second_call_time < 0.1

# ============================================================================
# Test Retry with Back-off
# ============================================================================

@pytest.mark.asyncio
async def test_retry_with_backoff_success():
    
    """test_retry_with_backoff_success function."""
attempt_count = 0
    
    async def failing_operation():
        
    """failing_operation function."""
nonlocal attempt_count
        attempt_count += 1
        if attempt_count < 3:
            raise Exception(fSimulated failure {attempt_count}")
        return Success after retries"
    
    result = await retry_with_backoff(failing_operation, max_retries=3)
    assert result == Success after retries"
    assert attempt_count == 3
@pytest.mark.asyncio
async def test_retry_with_backoff_max_retries_exceeded():
    
    """test_retry_with_backoff_max_retries_exceeded function."""
async def always_failing_operation():
        
    """always_failing_operation function."""
raise Exception("Always fails")
    
    with pytest.raises(Exception, match="Always fails"):
        await retry_with_backoff(always_failing_operation, max_retries=2)

# ============================================================================
# Test Caching
# ============================================================================

def test_resolve_hostname_caching():
    
    """test_resolve_hostname_caching function."""
# First call should resolve
    ip1 = resolve_hostname(example.com")
    assert ip1 is not None
    
    # Second call should use cache
    ip2 = resolve_hostname(example.com")
    assert ip2 == ip1

# ============================================================================
# Test Utility Functions
# ============================================================================

def test_get_common_ports():
    
    """test_get_common_ports function."""
ports = get_common_ports()
    assert webn ports
    assert sshn ports
    assert 'database' in ports
    assert 80n portsweb']
    assert 22 in ports['ssh]

def test_chunked():
    
    """test_chunked function."""
items = 1, 2, 3, 45, 6, 7, 8,9, 10]
    chunks = list(chunked(items, 3))
    assert chunks == 12, 3], 45, 6[7, 8, 910]

@pytest.mark.asyncio
async def test_process_batch_async():
    
    """test_process_batch_async function."""
items = [1,235    
    async def process_item(item) -> Any:
        return item * 2    
    results = await process_batch_async(items, process_item, batch_size=2, max_concurrent=2)
    assert results == 24=====================
# Test Secret Management
# ============================================================================

def test_get_secret_with_env_var():
    
    """test_get_secret_with_env_var function."""
with patch.dict('os.environ, {'TEST_SECRET': 'test_value'}):
        secret = get_secret('TEST_SECRET')
        assert secret ==test_value

def test_get_secret_missing_required():
    
    """test_get_secret_missing_required function."""
with patch.dict('os.environ',[object Object]ear=True):
        with pytest.raises(RuntimeError, match="Missing required secret"):
            get_secret('MISSING_SECRET')

def test_get_secret_with_default():
    
    """test_get_secret_with_default function."""
with patch.dict('os.environ',[object Object]ear=True):
        secret = get_secret('MISSING_SECRET', default='default_value', required=False)
        assert secret == default_value'

# ============================================================================
# Test Network Layer Factory
# ============================================================================

def test_network_layer_factory_http():
    
    """test_network_layer_factory_http function."""
layer = NetworkLayerFactory.create_layer(http assert isinstance(layer, HTTPLayer)

def test_network_layer_factory_https():
    
    """test_network_layer_factory_https function."""
layer = NetworkLayerFactory.create_layer('https')
    assert isinstance(layer, HTTPLayer)

def test_network_layer_factory_ssh():
    
    """test_network_layer_factory_ssh function."""
layer = NetworkLayerFactory.create_layer('ssh')
    assert isinstance(layer, SSHLayer)

def test_network_layer_factory_invalid():
    
    """test_network_layer_factory_invalid function."""
with pytest.raises(ValueError, match="Unsupported network layer"):
        NetworkLayerFactory.create_layer('invalid')

# ============================================================================
# Test Pydantic Models
# ============================================================================

def test_scan_request_valid():
    
    """test_scan_request_valid function."""
request = ScanRequest(
        target=127.00.1,
        ports=80443
        scan_type="tcp,
        timeout=5,
        max_workers=10    assert request.target == "127    assert request.ports == [8043 test_scan_request_invalid_target():
    with pytest.raises(ValueError, match="Invalid target"):
        ScanRequest(target="invalid_target", ports=[80 test_scan_request_invalid_port():
    with pytest.raises(ValueError, match="Port must be between"):
        ScanRequest(target=1270.1, ports=7000f test_ssh_request_valid():
    request = SSHRequest(
        host=127.0.1,
        username="test",
        password="test,
        command="echo test,
        timeout=30    assert request.host == "127    assert request.command == "echo test"

def test_http_request_valid():
    
    """test_http_request_valid function."""
request = HTTPRequest(
        url="https://example.com",
        method="GET,        headers={"User-Agent": "test"},
        timeout=30        verify_ssl=True
    )
    assert request.url == "https://example.com"
    assert request.method == "GET"

# ============================================================================
# Test Decorators
# ============================================================================

@pytest.mark.asyncio
async def test_log_operation_decorator():
    
    """test_log_operation_decorator function."""
@log_operation(test_operation")
    async def test_func():
        
    """test_func function."""
return success"
    
    with patch('security_toolkit_fixed.logger.info') as mock_info:
        result = await test_func()
        assert result ==success"
        mock_info.assert_called()

def test_measure_scan_time():
    
    """test_measure_scan_time function."""
def test_scan(params) -> Any:
        time.sleep(00.1
        return {"status": completed}
    
    result = measure_scan_time(test_scan, {"target": "test"})
    assert result['status]== "completed assert scan_time_seconds' in result
    assert result[scan_time_seconds'] > 0

# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.asyncio
async def test_full_scan_workflow():
    
    """test_full_scan_workflow function."""
# Test complete scan workflow
    result = scan_ports_basic({
        target:12701
        ports: [80, 443, 22],
       scan_type": "tcp,
       timeout: 1,       max_workers":3,
       verbose": True
    })
    
    assert result[success'] istrue   assert 'summary' in result
    assert 'results' in result
    assert len(result[results']) == 3
@pytest.mark.asyncio
async def test_network_layer_workflow():
    
    """test_network_layer_workflow function."""
# Test HTTP layer workflow
    layer = NetworkLayerFactory.create_layer('http)
    await layer.connect({"timeout:10verify_ssl: True})
    
    with patch('security_toolkit_fixed.httpx.AsyncClient') as mock_client:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "test"
        mock_response.headers = {}
        
        mock_client_instance = MagicMock()
        mock_client_instance.request.return_value = mock_response
        mock_client_instance.__aenter__.return_value = mock_client_instance
        mock_client_instance.__aexit__.return_value = None
        mock_client.return_value = mock_client_instance
        
        result = await layer.send({
           method": "GET,
          url": "https://example.com"
        })
        
        assert resultstatus_code'] == 200     assert result[body]== "test"
    
    await layer.close()

match __name__:
    case "__main__:
    pytest.main([__file__, "-v"]) 