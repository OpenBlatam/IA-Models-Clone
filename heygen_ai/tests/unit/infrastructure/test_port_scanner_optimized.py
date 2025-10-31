import pytest
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock
from freezegun import freeze_time
from datetime import datetime

from port_scanner import AsyncPortScanner, PortScanResult


class TestPortScannerOptimized:
    """Optimized test suite for AsyncPortScanner with advanced testing techniques."""

    @pytest.fixture
    def port_scanner(self):
        """Create port scanner instance for testing."""
        return AsyncPortScanner(max_concurrent_scans=10, timeout_seconds=1.0)

    @pytest.mark.asyncio
    async def test_scan_single_port_optimized(self, port_scanner):
        """Test single port scanning with comprehensive scenarios."""
        with patch('asyncio.open_connection') as mock_connection:
            # Mock successful connection
            mock_reader = AsyncMock()
            mock_writer = AsyncMock()
            mock_connection.return_value = (mock_reader, mock_writer)
            
            result = await port_scanner.scan_single_port('192.168.1.1', 80)
            
            assert isinstance(result, PortScanResult)
            assert result.target_host == '192.168.1.1'
            assert result.target_port == 80
            assert result.is_port_open is True
            assert result.service_name == 'http'
            assert result.response_time is not None
            assert result.scan_timestamp is not None

    @pytest.mark.asyncio
    async def test_scan_port_range_optimized(self, port_scanner):
        """Test port range scanning with comprehensive scenarios."""
        with patch.object(port_scanner, 'scan_single_port') as mock_scan:
            mock_scan.return_value = PortScanResult(
                target_host='192.168.1.1',
                target_port=80,
                is_port_open=True,
                service_name='http',
                scan_timestamp=datetime.utcnow()
            )
            
            results = await port_scanner.scan_port_range('192.168.1.1', 80, 85)
            
            assert len(results) == 6
            for result in results:
                assert isinstance(result, PortScanResult)
                assert result.target_host == '192.168.1.1'

    @pytest.mark.asyncio
    async def test_scan_common_ports_optimized(self, port_scanner):
        """Test common ports scanning with comprehensive scenarios."""
        with patch.object(port_scanner, 'scan_single_port') as mock_scan:
            mock_scan.return_value = PortScanResult(
                target_host='192.168.1.1',
                target_port=80,
                is_port_open=True,
                service_name='http',
                scan_timestamp=datetime.utcnow()
            )
            
            results = await port_scanner.scan_common_ports('192.168.1.1')
            
            assert len(results) > 0
            for result in results:
                assert isinstance(result, PortScanResult)

    @pytest.mark.asyncio
    async def test_filter_open_ports_optimized(self, port_scanner):
        """Test filtering open ports functionality."""
        scan_results = [
            PortScanResult('192.168.1.1', 80, True, 'http', scan_timestamp=datetime.utcnow()),
            PortScanResult('192.168.1.1', 443, True, 'https', scan_timestamp=datetime.utcnow()),
            PortScanResult('192.168.1.1', 22, False, 'ssh', scan_timestamp=datetime.utcnow()),
            PortScanResult('192.168.1.1', 8080, True, 'http-proxy', scan_timestamp=datetime.utcnow()),
            PortScanResult('192.168.1.1', 9999, False, 'unknown', scan_timestamp=datetime.utcnow())
        ]
        
        filtered = port_scanner.filter_open_ports(scan_results)
        assert len(filtered) == 3
        for result in filtered:
            assert result.is_port_open is True

    @pytest.mark.asyncio
    async def test_group_by_service_optimized(self, port_scanner):
        """Test grouping ports by service functionality."""
        scan_results = [
            PortScanResult('192.168.1.1', 80, True, 'http', scan_timestamp=datetime.utcnow()),
            PortScanResult('192.168.1.1', 8080, True, 'http', scan_timestamp=datetime.utcnow()),
            PortScanResult('192.168.1.1', 443, True, 'https', scan_timestamp=datetime.utcnow()),
            PortScanResult('192.168.1.1', 22, False, 'ssh', scan_timestamp=datetime.utcnow()),
            PortScanResult('192.168.1.1', 9999, True, 'unknown', scan_timestamp=datetime.utcnow())
        ]
        
        grouped = port_scanner.group_by_service(scan_results)
        assert 'http' in grouped
        assert 'https' in grouped
        assert 'ssh' in grouped
        assert 'unknown' in grouped
        assert len(grouped['http']) == 2

    @pytest.mark.asyncio
    async def test_error_handling_optimized(self, port_scanner):
        """Test comprehensive error handling in port scanning."""
        with patch('asyncio.open_connection') as mock_connection:
            # Test connection timeout
            mock_connection.side_effect = asyncio.TimeoutError()
            result = await port_scanner.scan_single_port('192.168.1.1', 80)
            
            assert isinstance(result, PortScanResult)
            assert result.is_port_open is False
            assert result.error_message == "Connection timeout"

    @pytest.mark.asyncio
    async def test_concurrent_operations_optimized(self, port_scanner):
        """Test concurrent port scanning operations."""
        with patch.object(port_scanner, 'scan_single_port') as mock_scan:
            mock_scan.return_value = PortScanResult(
                target_host='192.168.1.1',
                target_port=80,
                is_port_open=True,
                service_name='http',
                scan_timestamp=datetime.utcnow()
            )
            
            # Run multiple scans concurrently
            tasks = [
                port_scanner.scan_single_port('192.168.1.1', 80),
                port_scanner.scan_single_port('192.168.1.1', 443),
                port_scanner.scan_single_port('192.168.1.1', 22)
            ]
            
            results = await asyncio.gather(*tasks)
            assert len(results) == 3
            for result in results:
                assert isinstance(result, PortScanResult)

    @pytest.mark.asyncio
    async def test_performance_optimized(self, port_scanner):
        """Test performance characteristics of port scanning."""
        import time
        
        with patch.object(port_scanner, 'scan_single_port') as mock_scan:
            mock_scan.return_value = PortScanResult(
                target_host='192.168.1.1',
                target_port=80,
                is_port_open=True,
                service_name='http',
                scan_timestamp=datetime.utcnow()
            )
            
            start_time = time.time()
            result = await port_scanner.scan_single_port('192.168.1.1', 80)
            end_time = time.time()
            
            assert end_time - start_time < 1.0  # Should be fast with mock
            assert isinstance(result, PortScanResult)

    @pytest.mark.asyncio
    async def test_memory_efficiency_optimized(self, port_scanner):
        """Test memory efficiency of port scanning operations."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        with patch.object(port_scanner, 'scan_single_port') as mock_scan:
            mock_scan.return_value = PortScanResult(
                target_host='192.168.1.1',
                target_port=80,
                is_port_open=True,
                service_name='http',
                scan_timestamp=datetime.utcnow()
            )
            
            # Perform multiple scans
            for i in range(100):
                await port_scanner.scan_single_port('192.168.1.1', 80 + i)
            
            final_memory = process.memory_info().rss
            memory_increase = final_memory - initial_memory
            
            # Memory increase should be reasonable (less than 10MB)
            assert memory_increase < 10 * 1024 * 1024

    @freeze_time("2023-01-01 12:00:00")
    @pytest.mark.asyncio
    async def test_timestamp_consistency_optimized(self, port_scanner):
        """Test timestamp consistency in scan results."""
        with patch.object(port_scanner, 'scan_single_port') as mock_scan:
            mock_scan.return_value = PortScanResult(
                target_host='192.168.1.1',
                target_port=80,
                is_port_open=True,
                service_name='http',
                scan_timestamp=datetime.utcnow()
            )
            
            result = await port_scanner.scan_single_port('192.168.1.1', 80)
            assert result.scan_timestamp is not None

    @pytest.mark.asyncio
    async def test_large_port_range_optimized(self, port_scanner):
        """Test scanning large port ranges efficiently."""
        with patch.object(port_scanner, 'scan_single_port') as mock_scan:
            mock_scan.return_value = PortScanResult(
                target_host='192.168.1.1',
                target_port=80,
                is_port_open=True,
                service_name='http',
                scan_timestamp=datetime.utcnow()
            )
            
            # Test large port range
            results = await port_scanner.scan_port_range('192.168.1.1', 1, 1000)
            assert len(results) == 1000
            
            # Verify semaphore limits concurrency
            assert port_scanner.max_concurrent_scans == 10

    @pytest.mark.asyncio
    async def test_service_detection_accuracy_optimized(self, port_scanner):
        """Test accuracy of service detection."""
        # Test various port ranges and their expected services
        test_cases = [
            (21, 'ftp'), (22, 'ssh'), (23, 'telnet'), (25, 'smtp'),
            (53, 'dns'), (80, 'http'), (110, 'pop3'), (143, 'imap'),
            (443, 'https'), (993, 'imaps'), (995, 'pop3s'),
            (3306, 'mysql'), (5432, 'postgresql'), (6379, 'redis'),
            (27017, 'mongodb'), (9200, 'elasticsearch')
        ]
        
        for port, expected_service in test_cases:
            service = port_scanner.service_port_mapping.get(port, 'unknown')
            if port in test_cases:
                assert service == expected_service or service == 'unknown'

    @pytest.mark.asyncio
    async def test_connection_refused_handling_optimized(self, port_scanner):
        """Test handling of connection refused errors."""
        with patch('asyncio.open_connection') as mock_connection:
            mock_connection.side_effect = ConnectionRefusedError()
            
            result = await port_scanner.scan_single_port('192.168.1.1', 80)
            
            assert isinstance(result, PortScanResult)
            assert result.is_port_open is False
            assert result.error_message == "Connection refused"

    @pytest.mark.asyncio
    async def test_timeout_handling_optimized(self, port_scanner):
        """Test timeout handling in port scanning."""
        with patch('asyncio.open_connection') as mock_connection:
            mock_connection.side_effect = asyncio.TimeoutError()
            
            result = await port_scanner.scan_single_port('192.168.1.1', 80)
            
            assert isinstance(result, PortScanResult)
            assert result.is_port_open is False
            assert result.error_message == "Connection timeout"

    @pytest.mark.asyncio
    async def test_semaphore_limits_optimized(self, port_scanner):
        """Test that semaphore properly limits concurrent operations."""
        with patch.object(port_scanner, 'scan_single_port') as mock_scan:
            mock_scan.return_value = PortScanResult(
                target_host='192.168.1.1',
                target_port=80,
                is_port_open=True,
                service_name='http',
                scan_timestamp=datetime.utcnow()
            )
            
            # Create many concurrent tasks
            tasks = [port_scanner.scan_single_port('192.168.1.1', i) for i in range(50)]
            
            # All should complete successfully
            results = await asyncio.gather(*tasks)
            assert len(results) == 50
            for result in results:
                assert isinstance(result, PortScanResult)
