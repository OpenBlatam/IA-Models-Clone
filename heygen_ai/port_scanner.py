"""
Port Scanner for HeyGen AI
==========================

Provides asynchronous port scanning capabilities for network security
and infrastructure monitoring.
"""

import asyncio
import socket
import time
import logging
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import ipaddress

logger = logging.getLogger(__name__)


class PortStatus(Enum):
    """Port status enumeration."""
    OPEN = "open"
    CLOSED = "closed"
    FILTERED = "filtered"
    UNKNOWN = "unknown"


@dataclass
class PortScanResult:
    """Result of a port scan."""
    host: str = None
    port: int = None
    status: PortStatus = None
    response_time: Optional[float] = None
    service: Optional[str] = None
    banner: Optional[str] = None
    error: Optional[str] = None
    # Alternative field names for compatibility
    target_host: str = None
    target_port: int = None
    is_port_open: bool = None
    service_name: str = None
    error_message: str = None
    
    def __post_init__(self):
        """Set alternative field names for compatibility."""
        if self.target_host is not None and self.host is None:
            self.host = self.target_host
        if self.target_port is not None and self.port is None:
            self.port = self.target_port
        if self.is_port_open is not None and self.status is None:
            self.status = PortStatus.OPEN if self.is_port_open else PortStatus.CLOSED
        if self.service_name is not None and self.service is None:
            self.service = self.service_name
        if self.error_message is not None and self.error is None:
            self.error = self.error_message


@dataclass
class ScanConfig:
    """Configuration for port scanning."""
    timeout: float = 3.0
    concurrent_limit: int = 100
    retry_count: int = 1
    scan_delay: float = 0.0
    banner_grab: bool = False
    service_detection: bool = True


class AsyncPortScanner:
    """Asynchronous port scanner."""
    
    def __init__(self, config: Optional[ScanConfig] = None, timeout_seconds: Optional[float] = None):
        """Initialize the port scanner."""
        self.config = config or ScanConfig()
        if timeout_seconds is not None:
            self.config.timeout = timeout_seconds
        self.logger = logging.getLogger(__name__)
        self._service_ports = self._load_service_ports()
    
    def _load_service_ports(self) -> Dict[int, str]:
        """Load common service port mappings."""
        return {
            21: "ftp",
            22: "ssh",
            23: "telnet",
            25: "smtp",
            53: "dns",
            80: "http",
            110: "pop3",
            143: "imap",
            443: "https",
            993: "imaps",
            995: "pop3s",
            1433: "mssql",
            3306: "mysql",
            3389: "rdp",
            5432: "postgresql",
            5900: "vnc",
            6379: "redis",
            8080: "http-alt",
            8443: "https-alt",
            9200: "elasticsearch",
            27017: "mongodb"
        }
    
    async def scan_single_port(self, host: str, port: int) -> PortScanResult:
        """Scan a single port (alias for scan_port)."""
        return await self.scan_port(host, port)
    
    async def scan_port(self, host: str, port: int) -> PortScanResult:
        """Scan a single port on a host."""
        start_time = time.time()
        
        try:
            # Create connection with timeout
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=self.config.timeout
            )
            
            response_time = time.time() - start_time
            status = PortStatus.OPEN
            
            # Try to grab banner if enabled
            banner = None
            if self.config.banner_grab:
                try:
                    # Send a simple probe
                    writer.write(b"\r\n")
                    await writer.drain()
                    
                    # Try to read response
                    banner_data = await asyncio.wait_for(
                        reader.read(1024),
                        timeout=1.0
                    )
                    banner = banner_data.decode('utf-8', errors='ignore').strip()
                except:
                    pass
            
            writer.close()
            await writer.wait_closed()
            
            # Determine service
            service = None
            if self.config.service_detection:
                service = self._service_ports.get(port)
            
            return PortScanResult(
                host=host,
                port=port,
                status=status,
                response_time=response_time,
                service=service,
                banner=banner,
                target_host=host,
                target_port=port,
                is_port_open=True,
                service_name=service
            )
            
        except asyncio.TimeoutError:
            return PortScanResult(
                host=host,
                port=port,
                status=PortStatus.FILTERED,
                error="Connection timeout",
                target_host=host,
                target_port=port,
                is_port_open=False,
                error_message="Connection timeout"
            )
        except ConnectionRefusedError:
            return PortScanResult(
                host=host,
                port=port,
                status=PortStatus.CLOSED,
                error="Connection refused",
                target_host=host,
                target_port=port,
                is_port_open=False,
                error_message="Connection refused"
            )
        except Exception as e:
            return PortScanResult(
                host=host,
                port=port,
                status=PortStatus.UNKNOWN,
                error=str(e),
                target_host=host,
                target_port=port,
                is_port_open=False,
                error_message=str(e)
            )
    
    async def scan_ports(self, host: str, ports: List[int]) -> List[PortScanResult]:
        """Scan multiple ports on a host."""
        semaphore = asyncio.Semaphore(self.config.concurrent_limit)
        
        async def scan_with_semaphore(port: int) -> PortScanResult:
            async with semaphore:
                if self.config.scan_delay > 0:
                    await asyncio.sleep(self.config.scan_delay)
                return await self.scan_port(host, port)
        
        tasks = [scan_with_semaphore(port) for port in ports]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and return valid results
        valid_results = []
        for result in results:
            if isinstance(result, PortScanResult):
                valid_results.append(result)
            elif isinstance(result, Exception):
                self.logger.error(f"Port scan error: {result}")
        
        return valid_results
    
    async def scan_range(self, host: str, start_port: int, end_port: int) -> List[PortScanResult]:
        """Scan a range of ports on a host."""
        ports = list(range(start_port, end_port + 1))
        return await self.scan_ports(host, ports)
    
    async def scan_common_ports(self, host: str) -> List[PortScanResult]:
        """Scan common ports on a host."""
        common_ports = [21, 22, 23, 25, 53, 80, 110, 143, 443, 993, 995, 
                       1433, 3306, 3389, 5432, 5900, 6379, 8080, 8443, 
                       9200, 27017]
        return await self.scan_ports(host, common_ports)
    
    async def scan_hosts(self, hosts: List[str], ports: List[int]) -> Dict[str, List[PortScanResult]]:
        """Scan multiple hosts with the same ports."""
        results = {}
        
        for host in hosts:
            try:
                host_results = await self.scan_ports(host, ports)
                results[host] = host_results
            except Exception as e:
                self.logger.error(f"Failed to scan host {host}: {e}")
                results[host] = []
        
        return results
    
    def parse_port_range(self, port_spec: str) -> List[int]:
        """Parse port specification string into list of ports.
        
        Examples:
        - "80" -> [80]
        - "80,443" -> [80, 443]
        - "80-90" -> [80, 81, ..., 90]
        - "80,443,8080-8090" -> [80, 443, 8080, 8081, ..., 8090]
        """
        ports = set()
        
        for part in port_spec.split(','):
            part = part.strip()
            
            if '-' in part:
                # Range specification
                start, end = part.split('-', 1)
                try:
                    start_port = int(start.strip())
                    end_port = int(end.strip())
                    ports.update(range(start_port, end_port + 1))
                except ValueError:
                    self.logger.warning(f"Invalid port range: {part}")
            else:
                # Single port
                try:
                    port = int(part)
                    ports.add(port)
                except ValueError:
                    self.logger.warning(f"Invalid port: {part}")
        
        return sorted(list(ports))
    
    def get_open_ports(self, results: List[PortScanResult]) -> List[int]:
        """Get list of open ports from scan results."""
        return [result.port for result in results if result.status == PortStatus.OPEN]
    
    def filter_open_ports(self, results: List[PortScanResult]) -> List[PortScanResult]:
        """Filter results to only open ports."""
        return [result for result in results if result.is_port_open or result.status == PortStatus.OPEN]
    
    def get_services(self, results: List[PortScanResult]) -> Dict[int, str]:
        """Get service mapping from scan results."""
        return {result.port: result.service for result in results 
                if result.status == PortStatus.OPEN and result.service}
    
    def group_by_status(self, results: List[PortScanResult]) -> Dict[PortStatus, List[PortScanResult]]:
        """Group scan results by status."""
        grouped = {status: [] for status in PortStatus}
        
        for result in results:
            grouped[result.status].append(result)
        
        return grouped
    
    def generate_report(self, results: List[PortScanResult]) -> str:
        """Generate a text report from scan results."""
        if not results:
            return "No scan results available."
        
        # Group by host
        hosts = {}
        for result in results:
            if result.host not in hosts:
                hosts[result.host] = []
            hosts[result.host].append(result)
        
        report_lines = []
        
        for host, host_results in hosts.items():
            report_lines.append(f"\nHost: {host}")
            report_lines.append("-" * 50)
            
            # Group by status
            grouped = self.group_by_status(host_results)
            
            for status in [PortStatus.OPEN, PortStatus.CLOSED, PortStatus.FILTERED, PortStatus.UNKNOWN]:
                if grouped[status]:
                    report_lines.append(f"\n{status.value.upper()} PORTS:")
                    for result in grouped[status]:
                        service_info = f" ({result.service})" if result.service else ""
                        time_info = f" [{result.response_time:.3f}s]" if result.response_time else ""
                        report_lines.append(f"  {result.port}{service_info}{time_info}")
        
        return "\n".join(report_lines)
    
    def validate_host(self, host: str) -> bool:
        """Validate if host is a valid IP address or hostname."""
        try:
            # Try to parse as IP address
            ipaddress.ip_address(host)
            return True
        except ValueError:
            # Try to parse as hostname
            if len(host) > 0 and len(host) <= 253:
                return True
            return False
    
    def validate_port(self, port: int) -> bool:
        """Validate if port is in valid range."""
        return 1 <= port <= 65535
