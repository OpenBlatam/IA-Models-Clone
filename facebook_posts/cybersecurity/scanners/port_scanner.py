from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import socket
import asyncio
import time
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import ssl
import struct
    import nmap
from ..core import BaseConfig, ScanResult, BaseScanner
from .async_helpers import AsyncHelperManager, AsyncHelperConfig
from typing import Any, List, Dict, Optional
import logging
"""
Port scanning utilities with proper async/def distinction.
Async for network operations, def for CPU-bound analysis.
"""


# Optional imports for enhanced scanning
try:
    NMAP_AVAILABLE = True
except ImportError:
    NMAP_AVAILABLE = False


@dataclass
class PortScanConfig(BaseConfig):
    """Configuration for port scanning operations."""
    timeout: float = 1.0
    max_workers: int = 100
    retry_count: int = 2
    banner_grab: bool = True
    ssl_check: bool = True
    scan_type: str = "tcp"  # tcp, udp, syn
    use_nmap: bool = True
    nmap_arguments: str = "-sS -sV -O --version-intensity 5"
    # Async helper configuration
    async_config: AsyncHelperConfig = None

@dataclass
class PortScanResult:
    """Result of a port scan operation."""
    target: str
    port: int
    is_open: bool = False
    service_name: Optional[str] = None
    protocol: str = "tcp"
    banner: Optional[str] = None
    ssl_info: Optional[Dict] = None
    success: bool = False
    response_time: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self) -> Any:
        if self.metadata is None:
            self.metadata = {}

def get_common_services() -> Dict[int, str]:
    """Get mapping of common port numbers to service names."""
    return {
        21: "ftp", 22: "ssh", 23: "telnet", 25: "smtp",
        53: "dns", 80: "http", 110: "pop3", 143: "imap",
        443: "https", 993: "imaps", 995: "pop3s",
        3306: "mysql", 5432: "postgresql", 27017: "mongodb",
        6379: "redis", 8080: "http-proxy", 8443: "https-alt"
    }

def calculate_checksum(data: bytes) -> int:
    """Calculate IP checksum for packet validation."""
    if len(data) % 2 == 1:
        data += b'\x00'
    
    checksum = 0
    for i in range(0, len(data), 2):
        checksum += struct.unpack('!H', data[i:i+2])[0]
    
    while checksum >> 16:
        checksum = (checksum & 0xFFFF) + (checksum >> 16)
    
    return ~checksum & 0xFFFF

def validate_ip_address(ip: str) -> bool:
    """Validate IP address format."""
    try:
        socket.inet_aton(ip)
        return True
    except socket.error:
        return False

def parse_port_range(port_spec: str) -> List[int]:
    """Parse port range specification (e.g., '80,443,8000-8010')."""
    ports = []
    for part in port_spec.split(','):
        part = part.strip()
        if '-' in part:
            start, end = map(int, part.split('-'))
            ports.extend(range(start, end + 1))
        else:
            ports.append(int(part))
    return sorted(set(ports))

def analyze_scan_results(results: List[PortScanResult]) -> Dict[str, any]:
    """Analyze scan results for patterns and statistics."""
    if not results:
        return {"error": "No results to analyze"}
    
    total_scans = len(results)
    successful_scans = len([r for r in results if r.success])
    open_ports = len([r for r in results if r.is_open])
    
    # Calculate response time statistics
    response_times = [r.response_time for r in results if r.response_time > 0]
    avg_response_time = sum(response_times) / len(response_times) if response_times else 0
    
    # Group by service
    services = {}
    for result in results:
        service = result.service_name or "unknown"
        if service not in services:
            services[service] = []
        services[service].append(result)
    
    return {
        "total_scans": total_scans,
        "successful_scans": successful_scans,
        "success_rate": successful_scans / total_scans if total_scans > 0 else 0,
        "open_ports": open_ports,
        "avg_response_time": avg_response_time,
        "services": {service: len(results) for service, results in services.items()},
        "top_services": sorted(services.items(), key=lambda x: len(x[1]), reverse=True)[:5]
    }

async def scan_single_port_async(host: str, port: int, config: PortScanConfig, 
                                helper_manager: AsyncHelperManager) -> PortScanResult:
    """Scan a single port using async helpers (non-blocking)."""
    start_time = time.time()
    
    # Guard clause for invalid inputs
    if not validate_ip_address(host):
        return PortScanResult(
            target=host, port=port, is_open=False,
            error_message="Invalid IP address format"
        )
    
    if not (1 <= port <= 65535):
        return PortScanResult(
            target=host, port=port, is_open=False,
            error_message="Port must be between 1 and 65535"
        )
    
    try:
        # Use async helper for TCP connection (non-blocking)
        tcp_success, tcp_duration, tcp_error = await helper_manager.network_io.tcp_connect(
            host, port, config.timeout
        )
        
        response_time = time.time() - start_time
        
        if not tcp_success:
            return PortScanResult(
                target=host, port=port, is_open=False,
                error_message=tcp_error or "Connection failed",
                success=False,
                response_time=response_time
            )
        
        # Get service name
        service_name = get_common_services().get(port)
        
        # Banner grab using async helper (non-blocking)
        banner = None
        if config.banner_grab:
            banner_success, banner_content, _ = await helper_manager.network_io.banner_grab(
                host, port, config.timeout
            )
            if banner_success:
                banner = banner_content
        
        # SSL check using async helper (non-blocking)
        ssl_info = None
        if config.ssl_check and port in [443, 993, 995, 8443]:
            ssl_success, _, ssl_data = await helper_manager.network_io.ssl_connect(
                host, port, config.timeout
            )
            if ssl_success:
                ssl_info = ssl_data
        
        return PortScanResult(
            target=host,
            port=port,
            is_open=True,
            service_name=service_name,
            banner=banner,
            ssl_info=ssl_info,
            success=True,
            response_time=response_time
        )
        
    except Exception as e:
        response_time = time.time() - start_time
        return PortScanResult(
            target=host, port=port, is_open=False,
            error_message=str(e), success=False,
            response_time=response_time
        )

async def scan_port_range_async(host: str, start_port: int, end_port: int, 
                               config: PortScanConfig, helper_manager: AsyncHelperManager) -> List[PortScanResult]:
    """Scan a range of ports concurrently using async helpers."""
    if not validate_ip_address(host):
        return [PortScanResult(
            target=host, port=0, is_open=False,
            error_message="Invalid IP address format"
        )]
    
    ports = list(range(start_port, end_port + 1))
    semaphore = asyncio.Semaphore(config.max_workers)
    
    async def scan_with_semaphore(port: int) -> PortScanResult:
        async with semaphore:
            return await scan_single_port_async(host, port, config, helper_manager)
    
    # Create tasks for concurrent scanning
    tasks = [scan_with_semaphore(port) for port in ports]
    
    # Gather results with exception handling
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter out exceptions and return valid results
    valid_results = []
    for result in results:
        if isinstance(result, PortScanResult):
            valid_results.append(result)
        else:
            # Log exception and create error result
            # logger.error(f"Scan error: {result}") # Assuming logger is available
            valid_results.append(PortScanResult(
                target=host, port=0, is_open=False,
                error_message=str(result), success=False
            ))
    
    return valid_results

async def scan_common_ports_async(host: str, config: PortScanConfig, 
                                 helper_manager: AsyncHelperManager) -> List[PortScanResult]:
    """Scan common service ports using async helpers."""
    common_ports = list(get_common_services().keys())
    return await scan_port_range_async(host, min(common_ports), max(common_ports), config, helper_manager)

async def scan_with_retry_async(host: str, port: int, config: PortScanConfig,
                               helper_manager: AsyncHelperManager) -> PortScanResult:
    """Scan port with retry logic using async helpers."""
    for attempt in range(config.retry_count):
        result = await scan_single_port_async(host, port, config, helper_manager)
        if result.success:
            return result
        await asyncio.sleep(0.1 * (attempt + 1))
    return result

def run_nmap_scan(host: str, ports: str, config: PortScanConfig) -> Dict[str, Any]:
    """Run nmap scan using python-nmap library (CPU-bound operation)."""
    if not NMAP_AVAILABLE:
        return {"error": "python-nmap not available"}
    
    try:
        nm = nmap.PortScanner()
        nm.scan(host, ports, arguments=config.nmap_arguments)
        
        results = []
        for host_result in nm.all_hosts():
            for proto in nm[host_result].all_protocols():
                ports_info = nm[host_result][proto]
                for port in ports_info:
                    results.append({
                        "target": host_result,
                        "port": port,
                        "state": ports_info[port]["state"],
                        "service": ports_info[port].get("name", "unknown"),
                        "product": ports_info[port].get("product", ""),
                        "version": ports_info[port].get("version", ""),
                        "extrainfo": ports_info[port].get("extrainfo", "")
                    })
        
        return {
            "success": True,
            "results": results,
            "total_ports": len(results)
        }
        
    except Exception as e:
        return {"error": str(e)}

class PortScanner(BaseScanner):
    """Enhanced port scanner using async helpers."""
    
    def __init__(self, config: PortScanConfig):
        
    """__init__ function."""
super().__init__(config)
        self.helper_manager = AsyncHelperManager(config.async_config or AsyncHelperConfig())
    
    async def comprehensive_scan(self, host: str, ports: List[int]) -> Dict[str, Any]:
        """Comprehensive scan using async helpers."""
        start_time = time.time()
        
        # Use async helper manager for comprehensive scanning
        scan_result = await self.helper_manager.comprehensive_scan_async(host, ports)
        
        # Convert to PortScanResult format
        results = []
        for result in scan_result.get("results", []):
            port_result = PortScanResult(
                target=result["target"],
                port=result["port"],
                is_open=result.get("is_open", False),
                service_name=result.get("service_name"),
                banner=result.get("banner_content"),
                ssl_info=result.get("ssl_info"),
                success=result.get("success", False),
                response_time=result.get("total_duration", 0),
                error_message=result.get("tcp_error")
            )
            results.append(port_result)
        
        # Analyze results
        analysis = analyze_scan_results(results)
        
        total_duration = time.time() - start_time
        
        return {
            "target": host,
            "ports_scanned": len(ports),
            "results": results,
            "analysis": analysis,
            "total_duration": total_duration,
            "scan_rate": len(ports) / total_duration if total_duration > 0 else 0,
            "helper_stats": scan_result.get("analysis", {})
        }
    
    async def scan_port_range(self, host: str, start_port: int, end_port: int) -> List[PortScanResult]:
        """Scan port range using async helpers."""
        return await scan_port_range_async(host, start_port, end_port, self.config, self.helper_manager)
    
    async def scan_common_ports(self, host: str) -> List[PortScanResult]:
        """Scan common ports using async helpers."""
        return await scan_common_ports_async(host, self.config, self.helper_manager)
    
    async def scan_single_port(self, host: str, port: int) -> PortScanResult:
        """Scan single port using async helpers."""
        return await scan_single_port_async(host, port, self.config, self.helper_manager)
    
    async def close(self) -> Any:
        """Close helper manager."""
        await self.helper_manager.close()

# Legacy functions for backward compatibility (now use async helpers)
async def scan_single_port(host: str, port: int, config: PortScanConfig) -> PortScanResult:
    """Legacy function - now uses async helpers."""
    helper_manager = AsyncHelperManager(config.async_config or AsyncHelperConfig())
    try:
        return await scan_single_port_async(host, port, config, helper_manager)
    finally:
        await helper_manager.close()

async def scan_port_range(host: str, start_port: int, end_port: int, config: PortScanConfig) -> List[PortScanResult]:
    """Legacy function - now uses async helpers."""
    helper_manager = AsyncHelperManager(config.async_config or AsyncHelperConfig())
    try:
        return await scan_port_range_async(host, start_port, end_port, config, helper_manager)
    finally:
        await helper_manager.close()

async def scan_common_ports(host: str, config: PortScanConfig) -> List[PortScanResult]:
    """Legacy function - now uses async helpers."""
    helper_manager = AsyncHelperManager(config.async_config or AsyncHelperConfig())
    try:
        return await scan_common_ports_async(host, config, helper_manager)
    finally:
        await helper_manager.close()

async def scan_with_retry(host: str, port: int, config: PortScanConfig) -> PortScanResult:
    """Legacy function - now uses async helpers."""
    helper_manager = AsyncHelperManager(config.async_config or AsyncHelperConfig())
    try:
        return await scan_with_retry_async(host, port, config, helper_manager)
    finally:
        await helper_manager.close() 