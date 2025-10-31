from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import pytest
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch, call
from typing import List, Dict, Any
from .conftest import (
from cybersecurity.core.non_blocking_scanner import (
        import psutil
        import os
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
Tests for Non-Blocking Scanner
Comprehensive testing with pytest and pytest-asyncio, including network layer mocking.
"""


# Import test fixtures
    non_blocking_scanner, non_blocking_scan_config, mock_socket, mock_ssl_context,
    mock_aiohttp_session, mock_httpx_client, mock_dns_response,
    mock_network_responses, edge_case_targets, edge_case_ports,
    performance_test_data, mock_network_patcher, async_test_runner
)

# Import modules to test
    NonBlockingScanner, NonBlockingScanConfig, NonBlockingScanResult
)

class TestNonBlockingScanConfig:
    """Test NonBlockingScanConfig class."""
    
    def test_non_blocking_scan_config_defaults(self) -> Any:
        """Test NonBlockingScanConfig default values."""
        config = NonBlockingScanConfig()
        
        assert config.max_concurrent_scans == 50
        assert config.scan_timeout == 30.0
        assert config.chunk_size == 100
        assert config.enable_dns_cache is True
        assert config.enable_result_cache is True
        assert config.cache_ttl == 3600
    
    def test_non_blocking_scan_config_custom(self) -> Any:
        """Test NonBlockingScanConfig with custom values."""
        config = NonBlockingScanConfig(
            max_concurrent_scans=100,
            scan_timeout=60.0,
            chunk_size=200,
            enable_dns_cache=False,
            enable_result_cache=False,
            cache_ttl=1800
        )
        
        assert config.max_concurrent_scans == 100
        assert config.scan_timeout == 60.0
        assert config.chunk_size == 200
        assert config.enable_dns_cache is False
        assert config.enable_result_cache is False
        assert config.cache_ttl == 1800

class TestNonBlockingScanResult:
    """Test NonBlockingScanResult class."""
    
    def test_non_blocking_scan_result_creation(self) -> Any:
        """Test NonBlockingScanResult creation."""
        result = NonBlockingScanResult(
            target="192.168.1.1",
            scan_type="dns",
            success=True,
            data={"ip": "192.168.1.1"},
            duration=0.1
        )
        
        assert result.target == "192.168.1.1"
        assert result.scan_type == "dns"
        assert result.success is True
        assert result.data == {"ip": "192.168.1.1"}
        assert result.error is None
        assert result.duration == 0.1
        assert result.metadata is None
    
    def test_non_blocking_scan_result_defaults(self) -> Any:
        """Test NonBlockingScanResult with default values."""
        result = NonBlockingScanResult(target="192.168.1.1", scan_type="port")
        
        assert result.target == "192.168.1.1"
        assert result.scan_type == "port"
        assert result.success is False
        assert result.data is None
        assert result.error is None
        assert result.duration == 0.0
        assert result.metadata is None

class TestNonBlockingScanner:
    """Test NonBlockingScanner class."""
    
    @pytest.mark.asyncio
    async def test_non_blocking_scanner_creation(self, non_blocking_scan_config) -> Any:
        """Test NonBlockingScanner creation."""
        scanner = NonBlockingScanner(non_blocking_scan_config)
        
        assert scanner.config == non_blocking_scan_config
        assert scanner.helper is not None
        assert scanner.semaphore is not None
        assert len(scanner._scan_tasks) == 0
        
        await scanner.close()
    
    @pytest.mark.asyncio
    async def test_scan_targets_non_blocking_success(self, non_blocking_scanner) -> Optional[Dict[str, Any]]:
        """Test successful non-blocking target scanning."""
        targets = ["127.0.0.1", "localhost"]
        scan_types = ["dns", "port"]
        
        with mock_network_patcher() as patcher:
            # Mock successful operations
            mock_sock = MagicMock()
            mock_sock.connect_ex.return_value = 0
            patcher.patch_socket(mock_sock)
            
            results = await non_blocking_scanner.scan_targets_non_blocking(targets, scan_types)
            
            assert len(results) == len(targets)
            for target, target_results in results.items():
                assert len(target_results) == len(scan_types)
                for result in target_results:
                    assert isinstance(result, tuple)
                    assert len(result) == 2
                    assert isinstance(result[1], NonBlockingScanResult)
    
    @pytest.mark.asyncio
    async def test_scan_targets_non_blocking_failure(self, non_blocking_scanner) -> Optional[Dict[str, Any]]:
        """Test non-blocking target scanning with failures."""
        targets = ["invalid-host", "192.168.1.999"]
        scan_types = ["dns", "port"]
        
        results = await non_blocking_scanner.scan_targets_non_blocking(targets, scan_types)
        
        assert len(results) == len(targets)
        for target, target_results in results.items():
            assert len(target_results) == len(scan_types)
            for result in target_results:
                assert isinstance(result, tuple)
                assert len(result) == 2
                assert isinstance(result[1], NonBlockingScanResult)
    
    @pytest.mark.asyncio
    async def test_batch_scan_with_progress(self, non_blocking_scanner) -> Any:
        """Test batch scanning with progress tracking."""
        targets = ["127.0.0.1", "localhost", "192.168.1.1"]
        scan_types = ["dns", "port"]
        
        progress_calls = []
        
        async def progress_callback(progress, completed, total) -> Any:
            progress_calls.append((progress, completed, total))
        
        with mock_network_patcher() as patcher:
            mock_sock = MagicMock()
            mock_sock.connect_ex.return_value = 0
            patcher.patch_socket(mock_sock)
            
            results = await non_blocking_scanner.batch_scan_with_progress(
                targets, scan_types, progress_callback
            )
            
            assert len(results) == len(targets)
            assert len(progress_calls) > 0
            
            # Verify progress calls
            for progress, completed, total in progress_calls:
                assert 0 <= progress <= 100
                assert 0 <= completed <= total
                assert total == len(targets)
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, non_blocking_scanner) -> Any:
        """Test concurrent operations."""
        operations = [
            ("dns_lookup", lambda: non_blocking_scanner.helper.dns_lookup("google.com")),
            ("http_request", lambda: non_blocking_scanner.helper.http_request("http://httpbin.org/get")),
            ("port_scan", lambda: non_blocking_scanner.helper.port_scan("127.0.0.1", 80))
        ]
        
        results = await non_blocking_scanner.concurrent_operations(operations)
        
        assert len(results) == len(operations)
        for name, result in results.items():
            assert isinstance(result, NonBlockingScanResult)
            assert result.target is None  # These operations don't have targets
            assert result.scan_type == name
    
    @pytest.mark.asyncio
    async def test_batch_process_targets(self, non_blocking_scanner) -> Optional[Dict[str, Any]]:
        """Test batch processing of targets."""
        targets = ["target1.com", "target2.com", "target3.com"]
        
        async def process_target(target) -> Optional[Dict[str, Any]]:
            await asyncio.sleep(0.01)  # Simulate processing
            return {"processed": True, "target": target}
        
        results = await non_blocking_scanner.batch_process_targets(targets, process_target)
        
        assert len(results) == len(targets)
        for result in results:
            assert isinstance(result, NonBlockingScanResult)
            assert result.success is True
            assert result.data["processed"] is True
    
    @pytest.mark.asyncio
    async def test_get_scan_stats(self, non_blocking_scanner) -> Optional[Dict[str, Any]]:
        """Test getting scan statistics."""
        # Perform some scans to generate stats
        targets = ["127.0.0.1", "localhost"]
        scan_types = ["dns", "port"]
        
        with mock_network_patcher() as patcher:
            mock_sock = MagicMock()
            mock_sock.connect_ex.return_value = 0
            patcher.patch_socket(mock_sock)
            
            await non_blocking_scanner.scan_targets_non_blocking(targets, scan_types)
            
            stats = non_blocking_scanner.get_scan_stats()
            
            assert isinstance(stats, dict)
            assert "operations" in stats
            assert "errors" in stats
            assert "performance" in stats
    
    @pytest.mark.asyncio
    async def test_clear_cache(self, non_blocking_scanner) -> Any:
        """Test cache clearing."""
        # Perform some scans to populate cache
        targets = ["127.0.0.1"]
        scan_types = ["dns"]
        
        with mock_network_patcher() as patcher:
            mock_sock = MagicMock()
            mock_sock.connect_ex.return_value = 0
            patcher.patch_socket(mock_sock)
            
            await non_blocking_scanner.scan_targets_non_blocking(targets, scan_types)
            
            # Clear cache
            non_blocking_scanner.clear_cache()
            
            # Verify cache is cleared
            assert len(non_blocking_scanner._scan_cache) == 0
            assert len(non_blocking_scanner._dns_cache) == 0

class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.mark.asyncio
    @pytest.mark.edge_case
    async def test_edge_case_targets(self, non_blocking_scanner, edge_case_targets) -> Optional[Dict[str, Any]]:
        """Test scanning with edge case targets."""
        scan_types = ["dns", "port"]
        
        results = await non_blocking_scanner.scan_targets_non_blocking(
            edge_case_targets[:5], scan_types
        )
        
        assert len(results) == 5
        for target, target_results in results.items():
            assert len(target_results) == len(scan_types)
            for result in target_results:
                assert isinstance(result, tuple)
                assert len(result) == 2
                assert isinstance(result[1], NonBlockingScanResult)
    
    @pytest.mark.asyncio
    @pytest.mark.edge_case
    async def test_edge_case_scan_types(self, non_blocking_scanner) -> Any:
        """Test scanning with edge case scan types."""
        targets = ["127.0.0.1"]
        scan_types = ["dns", "port", "http", "ssl", "invalid_scan_type"]
        
        results = await non_blocking_scanner.scan_targets_non_blocking(targets, scan_types)
        
        assert len(results) == 1
        target_results = results["127.0.0.1"]
        assert len(target_results) == len(scan_types)
        
        # Should handle invalid scan types gracefully
        for result in target_results:
            assert isinstance(result, tuple)
            assert len(result) == 2
            assert isinstance(result[1], NonBlockingScanResult)
    
    @pytest.mark.asyncio
    @pytest.mark.edge_case
    async def test_empty_targets(self, non_blocking_scanner) -> Optional[Dict[str, Any]]:
        """Test scanning with empty targets list."""
        targets = []
        scan_types = ["dns", "port"]
        
        results = await non_blocking_scanner.scan_targets_non_blocking(targets, scan_types)
        
        assert len(results) == 0
    
    @pytest.mark.asyncio
    @pytest.mark.edge_case
    async def test_empty_scan_types(self, non_blocking_scanner) -> Any:
        """Test scanning with empty scan types list."""
        targets = ["127.0.0.1", "localhost"]
        scan_types = []
        
        results = await non_blocking_scanner.scan_targets_non_blocking(targets, scan_types)
        
        assert len(results) == len(targets)
        for target, target_results in results.items():
            assert len(target_results) == 0
    
    @pytest.mark.asyncio
    @pytest.mark.edge_case
    async def test_concurrent_edge_cases(self, non_blocking_scanner, edge_case_targets) -> Any:
        """Test concurrent scanning with edge cases."""
        targets = edge_case_targets[:3]
        scan_types = ["dns", "port"]
        
        # Test concurrent operations with edge cases
        operations = []
        for target in targets:
            for scan_type in scan_types:
                operations.append((
                    f"{scan_type}_{target}",
                    lambda t=target, s=scan_type: non_blocking_scanner.helper.dns_lookup(t) if s == "dns" else non_blocking_scanner.helper.port_scan(t, 80)
                ))
        
        results = await non_blocking_scanner.concurrent_operations(operations)
        
        assert len(results) == len(operations)
        # Should handle all edge cases gracefully
    
    @pytest.mark.asyncio
    @pytest.mark.edge_case
    async def test_invalid_config_handling(self, non_blocking_scan_config) -> Any:
        """Test handling of invalid configurations."""
        # Test with invalid max_concurrent_scans
        config = NonBlockingScanConfig(max_concurrent_scans=0)
        scanner = NonBlockingScanner(config)
        
        # Should still work with invalid config
        targets = ["127.0.0.1"]
        scan_types = ["dns"]
        
        results = await scanner.scan_targets_non_blocking(targets, scan_types)
        
        assert len(results) == 1
        await scanner.close()
        
        # Test with invalid scan_timeout
        config = NonBlockingScanConfig(scan_timeout=-1)
        scanner = NonBlockingScanner(config)
        
        results = await scanner.scan_targets_non_blocking(targets, scan_types)
        
        assert len(results) == 1
        await scanner.close()

class TestPerformance:
    """Test performance characteristics."""
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_performance_small_scan(self, non_blocking_scanner, performance_test_data) -> Any:
        """Test performance with small scan."""
        targets = performance_test_data["small_dataset"][:3]
        scan_types = ["dns", "port"]
        
        start_time = time.time()
        
        results = await non_blocking_scanner.scan_targets_non_blocking(targets, scan_types)
        
        duration = time.time() - start_time
        
        assert len(results) == len(targets)
        assert duration > 0
        # Performance should be reasonable for small scan
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_performance_medium_scan(self, non_blocking_scanner, performance_test_data) -> Any:
        """Test performance with medium scan."""
        targets = performance_test_data["medium_dataset"][:10]  # Limit for test
        scan_types = ["dns", "port"]
        
        start_time = time.time()
        
        results = await non_blocking_scanner.scan_targets_non_blocking(targets, scan_types)
        
        duration = time.time() - start_time
        
        assert len(results) == len(targets)
        assert duration > 0
        # Performance should be reasonable for medium scan
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_performance_concurrent_operations(self, non_blocking_scanner) -> Any:
        """Test performance with concurrent operations."""
        operations = []
        for i in range(20):
            operations.append((
                f"operation_{i}",
                lambda: asyncio.sleep(0.01)  # Simulate operation
            ))
        
        start_time = time.time()
        
        results = await non_blocking_scanner.concurrent_operations(operations)
        
        duration = time.time() - start_time
        
        assert len(results) == len(operations)
        assert duration > 0
        # Concurrent operations should be efficient
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_memory_efficiency(self, non_blocking_scanner) -> Any:
        """Test memory efficiency with large scans."""
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Perform large scan
        targets = [f"target_{i}.com" for i in range(100)]
        scan_types = ["dns", "port"]
        
        results = await non_blocking_scanner.scan_targets_non_blocking(targets, scan_types)
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        assert len(results) == len(targets)
        assert memory_increase < 100 * 1024 * 1024  # Should not increase by more than 100MB

class TestErrorHandling:
    """Test error handling and recovery."""
    
    @pytest.mark.asyncio
    async def test_network_timeout_handling(self, non_blocking_scanner) -> Any:
        """Test handling of network timeouts."""
        targets = ["192.168.1.999"]  # Invalid target
        scan_types = ["dns", "port"]
        
        results = await non_blocking_scanner.scan_targets_non_blocking(targets, scan_types)
        
        assert len(results) == 1
        target_results = results["192.168.1.999"]
        assert len(target_results) == len(scan_types)
        
        for result in target_results:
            assert isinstance(result, tuple)
            assert len(result) == 2
            scan_result = result[1]
            assert isinstance(scan_result, NonBlockingScanResult)
            assert scan_result.success is False
            assert scan_result.error is not None
    
    @pytest.mark.asyncio
    async def test_connection_refused_handling(self, non_blocking_scanner) -> Any:
        """Test handling of connection refused errors."""
        targets = ["127.0.0.1"]
        scan_types = ["port"]
        
        results = await non_blocking_scanner.scan_targets_non_blocking(targets, scan_types)
        
        assert len(results) == 1
        target_results = results["127.0.0.1"]
        assert len(target_results) == len(scan_types)
        
        for result in target_results:
            assert isinstance(result, tuple)
            assert len(result) == 2
            scan_result = result[1]
            assert isinstance(scan_result, NonBlockingScanResult)
    
    @pytest.mark.asyncio
    async def test_invalid_input_handling(self, non_blocking_scanner) -> Any:
        """Test handling of invalid inputs."""
        # Test with None targets
        results = await non_blocking_scanner.scan_targets_non_blocking(None, ["dns"])
        assert len(results) == 0
        
        # Test with None scan types
        results = await non_blocking_scanner.scan_targets_non_blocking(["127.0.0.1"], None)
        assert len(results) == 1
        assert len(results["127.0.0.1"]) == 0
    
    @pytest.mark.asyncio
    async def test_concurrent_error_handling(self, non_blocking_scanner) -> Any:
        """Test handling of concurrent errors."""
        # Mix valid and invalid operations
        operations = [
            ("valid_dns", lambda: non_blocking_scanner.helper.dns_lookup("127.0.0.1")),
            ("invalid_dns", lambda: non_blocking_scanner.helper.dns_lookup("invalid-host")),
            ("valid_port", lambda: non_blocking_scanner.helper.port_scan("127.0.0.1", 80)),
            ("invalid_port", lambda: non_blocking_scanner.helper.port_scan("invalid-host", 80))
        ]
        
        results = await non_blocking_scanner.concurrent_operations(operations)
        
        assert len(results) == len(operations)
        # Should handle all errors gracefully without crashing

class TestIntegration:
    """Integration tests combining multiple components."""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_full_scan_workflow(self, non_blocking_scanner) -> Any:
        """Test complete scan workflow."""
        targets = ["127.0.0.1", "localhost"]
        scan_types = ["dns", "port", "http"]
        
        # Perform comprehensive scan
        results = await non_blocking_scanner.scan_targets_non_blocking(targets, scan_types)
        
        # Verify results
        assert len(results) == len(targets)
        for target, target_results in results.items():
            assert len(target_results) == len(scan_types)
            for result in target_results:
                assert isinstance(result, tuple)
                assert len(result) == 2
                assert isinstance(result[1], NonBlockingScanResult)
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_concurrent_workflows(self, non_blocking_scanner) -> Any:
        """Test multiple concurrent workflows."""
        workflows = []
        
        for i in range(3):
            async def workflow():
                
    """workflow function."""
targets = [f"target_{i+1}.com", f"target_{i+2}.com"]
                scan_types = ["dns", "port"]
                
                # Scan
                scan_results = await non_blocking_scanner.scan_targets_non_blocking(targets, scan_types)
                
                # Process results
                processed_results = await non_blocking_scanner.batch_process_targets(
                    targets, lambda t: {"processed": True, "target": t}
                )
                
                return {"scan_results": scan_results, "processed_results": processed_results}
            
            workflows.append(workflow())
        
        results = await asyncio.gather(*workflows, return_exceptions=True)
        
        assert len(results) == 3
        for result in results:
            if isinstance(result, dict):
                assert "scan_results" in result
                assert "processed_results" in result
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_progress_tracking_integration(self, non_blocking_scanner) -> Any:
        """Test progress tracking integration."""
        targets = ["127.0.0.1", "localhost", "192.168.1.1", "8.8.8.8"]
        scan_types = ["dns", "port"]
        
        progress_updates = []
        
        async def progress_callback(progress, completed, total) -> Any:
            progress_updates.append((progress, completed, total))
        
        results = await non_blocking_scanner.batch_scan_with_progress(
            targets, scan_types, progress_callback
        )
        
        assert len(results) == len(targets)
        assert len(progress_updates) > 0
        
        # Verify progress updates are reasonable
        for progress, completed, total in progress_updates:
            assert 0 <= progress <= 100
            assert 0 <= completed <= total
            assert total == len(targets)
        
        # Verify progress increases
        progress_values = [p for p, _, _ in progress_updates]
        assert progress_values == sorted(progress_values)  # Should be increasing 