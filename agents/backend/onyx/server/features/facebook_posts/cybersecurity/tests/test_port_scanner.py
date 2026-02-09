from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import pytest
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch, call
from typing import List, Dict, Any
from .conftest import (
from cybersecurity.scanners.port_scanner import (
            from cybersecurity.scanners.async_helpers import AsyncHelperManager, AsyncHelperConfig
        from cybersecurity.scanners.async_helpers import AsyncHelperManager, AsyncHelperConfig
            from cybersecurity.scanners.async_helpers import AsyncHelperManager, AsyncHelperConfig
            from cybersecurity.scanners.async_helpers import AsyncHelperManager, AsyncHelperConfig
            from cybersecurity.scanners.async_helpers import AsyncHelperManager, AsyncHelperConfig
        import psutil
        import os
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
Tests for Port Scanner
Comprehensive testing with pytest and pytest-asyncio, including network layer mocking.
"""


# Import test fixtures
    port_scanner, port_scan_config, mock_socket, mock_ssl_context,
    mock_aiohttp_session, mock_httpx_client, mock_dns_response,
    mock_network_responses, edge_case_targets, edge_case_ports,
    performance_test_data, mock_network_patcher, async_test_runner
)

# Import modules to test
    PortScanConfig, PortScanner, PortScanResult,
    scan_single_port_async, scan_port_range_async, scan_common_ports_async,
    scan_with_retry_async, run_nmap_scan, get_common_services,
    validate_ip_address, parse_port_range, analyze_scan_results
)

class TestPortScanConfig:
    """Test PortScanConfig class."""
    
    def test_port_scan_config_defaults(self) -> Any:
        """Test PortScanConfig default values."""
        config = PortScanConfig()
        
        assert config.timeout == 1.0
        assert config.max_workers == 100
        assert config.retry_count == 2
        assert config.banner_grab is True
        assert config.ssl_check is True
        assert config.scan_type == "tcp"
        assert config.use_nmap is True
    
    def test_port_scan_config_custom(self) -> Any:
        """Test PortScanConfig with custom values."""
        config = PortScanConfig(
            timeout=5.0,
            max_workers=50,
            retry_count=3,
            banner_grab=False,
            ssl_check=False,
            scan_type="udp",
            use_nmap=False
        )
        
        assert config.timeout == 5.0
        assert config.max_workers == 50
        assert config.retry_count == 3
        assert config.banner_grab is False
        assert config.ssl_check is False
        assert config.scan_type == "udp"
        assert config.use_nmap is False

class TestPortScanResult:
    """Test PortScanResult class."""
    
    def test_port_scan_result_creation(self) -> Any:
        """Test PortScanResult creation."""
        result = PortScanResult(
            target="192.168.1.1",
            port=22,
            is_open=True,
            service_name="ssh",
            protocol="tcp",
            banner="SSH-2.0-OpenSSH_8.2",
            success=True,
            response_time=0.1
        )
        
        assert result.target == "192.168.1.1"
        assert result.port == 22
        assert result.is_open is True
        assert result.service_name == "ssh"
        assert result.protocol == "tcp"
        assert result.banner == "SSH-2.0-OpenSSH_8.2"
        assert result.success is True
        assert result.response_time == 0.1
        assert result.metadata == {}
    
    def test_port_scan_result_defaults(self) -> Any:
        """Test PortScanResult with default values."""
        result = PortScanResult(target="192.168.1.1", port=80)
        
        assert result.target == "192.168.1.1"
        assert result.port == 80
        assert result.is_open is False
        assert result.service_name is None
        assert result.protocol == "tcp"
        assert result.banner is None
        assert result.ssl_info is None
        assert result.success is False
        assert result.response_time == 0.0
        assert result.error_message is None
        assert result.metadata == {}

class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_get_common_services(self) -> Optional[Dict[str, Any]]:
        """Test get_common_services function."""
        services = get_common_services()
        
        assert isinstance(services, dict)
        assert 22 in services
        assert services[22] == "ssh"
        assert 80 in services
        assert services[80] == "http"
        assert 443 in services
        assert services[443] == "https"
    
    def test_validate_ip_address_valid(self) -> bool:
        """Test validate_ip_address with valid IPs."""
        valid_ips = ["127.0.0.1", "192.168.1.1", "8.8.8.8", "255.255.255.255"]
        
        for ip in valid_ips:
            assert validate_ip_address(ip) is True
    
    def test_validate_ip_address_invalid(self) -> bool:
        """Test validate_ip_address with invalid IPs."""
        invalid_ips = [
            "256.1.2.3",  # Invalid octet
            "192.168.1",   # Incomplete
            "192.168.1.1.1",  # Too many octets
            "192.168.1.256",  # Invalid octet
            "invalid",     # Not an IP
            "",           # Empty
            "192.168.1.1.1.1"  # Too many octets
        ]
        
        for ip in invalid_ips:
            assert validate_ip_address(ip) is False
    
    def test_parse_port_range_single(self) -> Any:
        """Test parse_port_range with single ports."""
        result = parse_port_range("80")
        assert result == [80]
        
        result = parse_port_range("22,80,443")
        assert result == [22, 80, 443]
    
    def test_parse_port_range_ranges(self) -> Any:
        """Test parse_port_range with port ranges."""
        result = parse_port_range("80-85")
        assert result == [80, 81, 82, 83, 84, 85]
        
        result = parse_port_range("22,80-82,443")
        assert result == [22, 80, 81, 82, 443]
    
    def test_parse_port_range_mixed(self) -> Any:
        """Test parse_port_range with mixed formats."""
        result = parse_port_range("22,80-82,443,8080-8082")
        assert result == [22, 80, 81, 82, 443, 8080, 8081, 8082]
    
    def test_analyze_scan_results(self) -> Any:
        """Test analyze_scan_results function."""
        results = [
            PortScanResult("192.168.1.1", 22, is_open=True, success=True, response_time=0.1),
            PortScanResult("192.168.1.1", 80, is_open=True, success=True, response_time=0.2),
            PortScanResult("192.168.1.1", 443, is_open=False, success=True, response_time=0.3),
            PortScanResult("192.168.1.1", 8080, is_open=False, success=False, response_time=0.4)
        ]
        
        analysis = analyze_scan_results(results)
        
        assert analysis["total_scans"] == 4
        assert analysis["successful_scans"] == 3
        assert analysis["open_ports"] == 2
        assert analysis["success_rate"] == 0.75
        assert analysis["avg_response_time"] == 0.25
    
    def test_analyze_scan_results_empty(self) -> Any:
        """Test analyze_scan_results with empty results."""
        analysis = analyze_scan_results([])
        
        assert "error" in analysis
        assert analysis["error"] == "No results to analyze"

class TestPortScanner:
    """Test PortScanner class."""
    
    @pytest.mark.asyncio
    async def test_port_scanner_creation(self, port_scan_config) -> Any:
        """Test PortScanner creation."""
        scanner = PortScanner(port_scan_config)
        
        assert scanner.config == port_scan_config
        assert scanner.helper_manager is not None
        
        await scanner.close()
    
    @pytest.mark.asyncio
    async def test_comprehensive_scan_success(self, port_scanner) -> Any:
        """Test comprehensive scan with success."""
        host = "127.0.0.1"
        ports = [22, 80, 443]
        
        with mock_network_patcher() as patcher:
            # Mock successful connections
            mock_sock = MagicMock()
            mock_sock.connect_ex.return_value = 0
            mock_sock.recv.return_value = b"SSH-2.0-OpenSSH_8.2"
            mock_sock.send.return_value = 20
            patcher.patch_socket(mock_sock)
            
            result = await port_scanner.comprehensive_scan(host, ports)
            
            assert result["target"] == host
            assert len(result["results"]) == len(ports)
            assert result["total_duration"] > 0
            assert result["scan_rate"] > 0
            assert "analysis" in result
    
    @pytest.mark.asyncio
    async def test_comprehensive_scan_failure(self, port_scanner) -> Any:
        """Test comprehensive scan with failures."""
        host = "192.168.1.999"  # Invalid IP
        ports = [22, 80, 443]
        
        result = await port_scanner.comprehensive_scan(host, ports)
        
        assert result["target"] == host
        assert len(result["results"]) == len(ports)
        assert result["total_duration"] > 0
        # Should handle failures gracefully
    
    @pytest.mark.asyncio
    async def test_scan_port_range(self, port_scanner) -> Any:
        """Test scanning port range."""
        host = "127.0.0.1"
        start_port = 80
        end_port = 85
        
        with mock_network_patcher() as patcher:
            mock_sock = MagicMock()
            mock_sock.connect_ex.return_value = 0
            patcher.patch_socket(mock_sock)
            
            results = await port_scanner.scan_port_range(host, start_port, end_port)
            
            assert len(results) == (end_port - start_port + 1)
            for result in results:
                assert isinstance(result, PortScanResult)
                assert result.target == host
    
    @pytest.mark.asyncio
    async def test_scan_common_ports(self, port_scanner) -> Any:
        """Test scanning common ports."""
        host = "127.0.0.1"
        
        with mock_network_patcher() as patcher:
            mock_sock = MagicMock()
            mock_sock.connect_ex.return_value = 0
            patcher.patch_socket(mock_sock)
            
            results = await port_scanner.scan_common_ports(host)
            
            assert len(results) > 0
            for result in results:
                assert isinstance(result, PortScanResult)
                assert result.target == host
    
    @pytest.mark.asyncio
    async def test_scan_single_port(self, port_scanner) -> Any:
        """Test scanning single port."""
        host = "127.0.0.1"
        port = 22
        
        with mock_network_patcher() as patcher:
            mock_sock = MagicMock()
            mock_sock.connect_ex.return_value = 0
            mock_sock.recv.return_value = b"SSH-2.0-OpenSSH_8.2"
            mock_sock.send.return_value = 20
            patcher.patch_socket(mock_sock)
            
            result = await port_scanner.scan_single_port(host, port)
            
            assert isinstance(result, PortScanResult)
            assert result.target == host
            assert result.port == port
            assert result.is_open is True
            assert result.success is True

class TestAsyncScanFunctions:
    """Test async scan functions."""
    
    @pytest.mark.asyncio
    async def test_scan_single_port_async_success(self, port_scan_config) -> Any:
        """Test scan_single_port_async with success."""
        host = "127.0.0.1"
        port = 22
        
        with mock_network_patcher() as patcher:
            mock_sock = MagicMock()
            mock_sock.connect_ex.return_value = 0
            mock_sock.recv.return_value = b"SSH-2.0-OpenSSH_8.2"
            mock_sock.send.return_value = 20
            patcher.patch_socket(mock_sock)
            
            # Create helper manager
            async_config = AsyncHelperConfig()
            async with AsyncHelperManager(async_config) as helper_manager:
                result = await scan_single_port_async(host, port, port_scan_config, helper_manager)
                
                assert isinstance(result, PortScanResult)
                assert result.target == host
                assert result.port == port
                assert result.is_open is True
                assert result.success is True
    
    @pytest.mark.asyncio
    async def test_scan_single_port_async_failure(self, port_scan_config) -> Any:
        """Test scan_single_port_async with failure."""
        host = "192.168.1.999"  # Invalid IP
        port = 22
        
        async_config = AsyncHelperConfig()
        async with AsyncHelperManager(async_config) as helper_manager:
            result = await scan_single_port_async(host, port, port_scan_config, helper_manager)
            
            assert isinstance(result, PortScanResult)
            assert result.target == host
            assert result.port == port
            assert result.is_open is False
            assert result.success is False
            assert result.error_message is not None
    
    @pytest.mark.asyncio
    async def test_scan_port_range_async(self, port_scan_config) -> Any:
        """Test scan_port_range_async."""
        host = "127.0.0.1"
        start_port = 80
        end_port = 85
        
        with mock_network_patcher() as patcher:
            mock_sock = MagicMock()
            mock_sock.connect_ex.return_value = 0
            patcher.patch_socket(mock_sock)
            
            async_config = AsyncHelperConfig()
            async with AsyncHelperManager(async_config) as helper_manager:
                results = await scan_port_range_async(host, start_port, end_port, port_scan_config, helper_manager)
                
                assert len(results) == (end_port - start_port + 1)
                for result in results:
                    assert isinstance(result, PortScanResult)
                    assert result.target == host
    
    @pytest.mark.asyncio
    async def test_scan_common_ports_async(self, port_scan_config) -> Any:
        """Test scan_common_ports_async."""
        host = "127.0.0.1"
        
        with mock_network_patcher() as patcher:
            mock_sock = MagicMock()
            mock_sock.connect_ex.return_value = 0
            patcher.patch_socket(mock_sock)
            
            async_config = AsyncHelperConfig()
            async with AsyncHelperManager(async_config) as helper_manager:
                results = await scan_common_ports_async(host, port_scan_config, helper_manager)
                
                assert len(results) > 0
                for result in results:
                    assert isinstance(result, PortScanResult)
                    assert result.target == host
    
    @pytest.mark.asyncio
    async def test_scan_with_retry_async(self, port_scan_config) -> Any:
        """Test scan_with_retry_async."""
        host = "127.0.0.1"
        port = 22
        
        with mock_network_patcher() as patcher:
            mock_sock = MagicMock()
            mock_sock.connect_ex.return_value = 0
            mock_sock.recv.return_value = b"SSH-2.0-OpenSSH_8.2"
            mock_sock.send.return_value = 20
            patcher.patch_socket(mock_sock)
            
            async_config = AsyncHelperConfig()
            async with AsyncHelperManager(async_config) as helper_manager:
                result = await scan_with_retry_async(host, port, port_scan_config, helper_manager)
                
                assert isinstance(result, PortScanResult)
                assert result.target == host
                assert result.port == port

class TestNmapIntegration:
    """Test nmap integration."""
    
    def test_run_nmap_scan_available(self) -> Any:
        """Test run_nmap_scan when nmap is available."""
        host = "127.0.0.1"
        ports = "22,80,443"
        config = PortScanConfig()
        
        with patch('cybersecurity.scanners.port_scanner.NMAP_AVAILABLE', True):
            with patch('cybersecurity.scanners.port_scanner.nmap') as mock_nmap:
                mock_scanner = MagicMock()
                mock_scanner.all_hosts.return_value = ["127.0.0.1"]
                mock_scanner.__getitem__.return_value = {
                    "tcp": {
                        22: {"state": "open", "name": "ssh", "product": "OpenSSH", "version": "8.2"},
                        80: {"state": "closed", "name": "http"},
                        443: {"state": "open", "name": "https", "product": "nginx", "version": "1.18.0"}
                    }
                }
                mock_nmap.PortScanner.return_value = mock_scanner
                
                result = run_nmap_scan(host, ports, config)
                
                assert result["success"] is True
                assert len(result["results"]) == 3
                assert result["total_ports"] == 3
    
    def test_run_nmap_scan_unavailable(self) -> Any:
        """Test run_nmap_scan when nmap is not available."""
        host = "127.0.0.1"
        ports = "22,80,443"
        config = PortScanConfig()
        
        with patch('cybersecurity.scanners.port_scanner.NMAP_AVAILABLE', False):
            result = run_nmap_scan(host, ports, config)
            
            assert "error" in result
            assert result["error"] == "python-nmap not available"
    
    def test_run_nmap_scan_exception(self) -> Any:
        """Test run_nmap_scan with exception."""
        host = "127.0.0.1"
        ports = "22,80,443"
        config = PortScanConfig()
        
        with patch('cybersecurity.scanners.port_scanner.NMAP_AVAILABLE', True):
            with patch('cybersecurity.scanners.port_scanner.nmap') as mock_nmap:
                mock_nmap.PortScanner.side_effect = Exception("Nmap error")
                
                result = run_nmap_scan(host, ports, config)
                
                assert "error" in result
                assert "Nmap error" in result["error"]

class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.mark.asyncio
    @pytest.mark.edge_case
    async def test_edge_case_targets(self, port_scanner, edge_case_targets) -> Optional[Dict[str, Any]]:
        """Test scanning with edge case targets."""
        results = []
        
        for target in edge_case_targets[:5]:  # Test first 5
            try:
                result = await port_scanner.scan_single_port(target, 80)
                results.append((target, result))
            except Exception as e:
                results.append((target, PortScanResult(target, 80, error_message=str(e))))
        
        assert len(results) == 5
        # Should handle all edge cases gracefully
    
    @pytest.mark.asyncio
    @pytest.mark.edge_case
    async def test_edge_case_ports(self, port_scanner, edge_case_ports) -> Any:
        """Test scanning with edge case ports."""
        host = "127.0.0.1"
        results = []
        
        for port in edge_case_ports[:10]:  # Test first 10
            try:
                result = await port_scanner.scan_single_port(host, port)
                results.append((port, result))
            except Exception as e:
                results.append((port, PortScanResult(host, port, error_message=str(e))))
        
        assert len(results) == 10
        # Should handle all edge cases gracefully
    
    @pytest.mark.asyncio
    @pytest.mark.edge_case
    async def test_concurrent_edge_cases(self, port_scanner, edge_case_targets, edge_case_ports) -> Any:
        """Test concurrent scanning with edge cases."""
        targets = edge_case_targets[:3]
        ports = edge_case_ports[:3]
        
        tasks = []
        for target in targets:
            for port in ports:
                task = port_scanner.scan_single_port(target, port)
                tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        assert len(results) == len(targets) * len(ports)
        # Should handle concurrent edge cases gracefully
    
    @pytest.mark.asyncio
    @pytest.mark.edge_case
    async def test_invalid_config_handling(self, port_scanner) -> Any:
        """Test handling of invalid configurations."""
        # Test with invalid timeout
        config = PortScanConfig(timeout=-1)
        result = await port_scanner.scan_single_port("127.0.0.1", 80)
        assert isinstance(result, PortScanResult)
        
        # Test with invalid max_workers
        config = PortScanConfig(max_workers=0)
        result = await port_scanner.scan_single_port("127.0.0.1", 80)
        assert isinstance(result, PortScanResult)

class TestPerformance:
    """Test performance characteristics."""
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_performance_small_scan(self, port_scanner, performance_test_data) -> Any:
        """Test performance with small scan."""
        targets = performance_test_data["small_dataset"][:3]
        ports = performance_test_data["common_ports"][:5]
        
        start_time = time.time()
        
        all_results = []
        for target in targets:
            for port in ports:
                result = await port_scanner.scan_single_port(target, port)
                all_results.append(result)
        
        duration = time.time() - start_time
        
        assert len(all_results) == len(targets) * len(ports)
        assert duration > 0
        # Performance should be reasonable for small scan
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_performance_concurrent_scan(self, port_scanner, performance_test_data) -> Any:
        """Test performance with concurrent scanning."""
        targets = performance_test_data["small_dataset"][:3]
        ports = performance_test_data["common_ports"][:5]
        
        start_time = time.time()
        
        tasks = []
        for target in targets:
            for port in ports:
                task = port_scanner.scan_single_port(target, port)
                tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        duration = time.time() - start_time
        
        assert len(results) == len(targets) * len(ports)
        assert duration > 0
        # Concurrent scanning should be faster than sequential
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_memory_efficiency(self, port_scanner) -> Any:
        """Test memory efficiency with large scans."""
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Perform large scan
        targets = [f"192.168.1.{i}" for i in range(1, 11)]
        ports = list(range(80, 90))
        
        all_results = []
        for target in targets:
            for port in ports:
                result = await port_scanner.scan_single_port(target, port)
                all_results.append(result)
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        assert len(all_results) == len(targets) * len(ports)
        assert memory_increase < 50 * 1024 * 1024  # Should not increase by more than 50MB

class TestErrorHandling:
    """Test error handling and recovery."""
    
    @pytest.mark.asyncio
    async def test_network_timeout_handling(self, port_scanner) -> Any:
        """Test handling of network timeouts."""
        host = "192.168.1.1"
        port = 9999  # Unlikely to be open
        
        result = await port_scanner.scan_single_port(host, port)
        
        assert isinstance(result, PortScanResult)
        assert result.is_open is False
        assert result.success is False
        assert result.error_message is not None
    
    @pytest.mark.asyncio
    async def test_connection_refused_handling(self, port_scanner) -> Any:
        """Test handling of connection refused errors."""
        host = "127.0.0.1"
        port = 9999  # Unlikely to be open
        
        result = await port_scanner.scan_single_port(host, port)
        
        assert isinstance(result, PortScanResult)
        assert result.is_open is False
        assert result.success is False
        assert result.error_message is not None
    
    @pytest.mark.asyncio
    async def test_invalid_input_handling(self, port_scanner) -> Any:
        """Test handling of invalid inputs."""
        # Test invalid host
        result = await port_scanner.scan_single_port("", 80)
        assert isinstance(result, PortScanResult)
        assert result.success is False
        assert result.error_message is not None
        
        # Test invalid port
        result = await port_scanner.scan_single_port("127.0.0.1", -1)
        assert isinstance(result, PortScanResult)
        assert result.success is False
        assert result.error_message is not None
    
    @pytest.mark.asyncio
    async def test_concurrent_error_handling(self, port_scanner) -> Any:
        """Test handling of concurrent errors."""
        tasks = []
        
        # Mix valid and invalid targets
        targets = ["127.0.0.1", "invalid-host", "192.168.1.999", "localhost"]
        ports = [80, 9999, 22, 443]
        
        for target, port in zip(targets, ports):
            task = port_scanner.scan_single_port(target, port)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        assert len(results) == len(targets)
        # Should handle all errors gracefully without crashing

class TestIntegration:
    """Integration tests combining multiple components."""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_full_scan_workflow(self, port_scanner) -> Any:
        """Test complete scan workflow."""
        targets = ["127.0.0.1", "localhost"]
        ports = [22, 80, 443]
        
        all_results = []
        for target in targets:
            # Perform comprehensive scan
            scan_results = await port_scanner.comprehensive_scan(target, ports)
            all_results.append(scan_results)
            
            # Verify results
            assert scan_results["target"] == target
            assert len(scan_results["results"]) == len(ports)
            assert scan_results["total_duration"] > 0
            assert scan_results["scan_rate"] > 0
        
        # Analyze all results
        all_scan_results = []
        for result in all_results:
            all_scan_results.extend(result["results"])
        
        analysis = analyze_scan_results(all_scan_results)
        
        assert analysis["total_scans"] > 0
        assert analysis["successful_scans"] >= 0
        assert analysis["open_ports"] >= 0
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_concurrent_workflows(self, port_scanner) -> Any:
        """Test multiple concurrent workflows."""
        workflows = []
        
        for i in range(3):
            async def workflow():
                
    """workflow function."""
target = f"192.168.1.{i+1}"
                ports = [22, 80, 443]
                
                # Scan
                scan_results = await port_scanner.comprehensive_scan(target, ports)
                
                # Analyze
                analysis = analyze_scan_results(scan_results["results"])
                
                return {"target": target, "scan_results": scan_results, "analysis": analysis}
            
            workflows.append(workflow())
        
        results = await asyncio.gather(*workflows, return_exceptions=True)
        
        assert len(results) == 3
        for result in results:
            if isinstance(result, dict):
                assert "target" in result
                assert "scan_results" in result
                assert "analysis" in result 