from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import socket
import asyncio
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import ssl
import struct
from typing import Any, List, Dict, Optional
import logging
"""
Port scanning utilities with async/def distinction.
Async for network operations, def for CPU-bound tasks.
"""


@dataclass
class ScanResult:
    """Result of a port scan operation."""
    host: str
    port: int
    is_open: bool
    service_name: Optional[str] = None
    response_time: float = 0.0
    protocol: str = "tcp"
    banner: Optional[str] = None

@dataclass
class ScanConfig:
    """Configuration for port scanning operations."""
    timeout: float = 1.0
    max_workers: int = 100
    retry_count: int = 2
    banner_grab: bool = True
    ssl_check: bool = True

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

async def scan_single_port(host: str, port: int, config: ScanConfig) -> ScanResult:
    """Scan a single port asynchronously."""
    start_time = time.time()
    
    try:
        # Create socket with timeout
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(config.timeout)
        
        # Attempt connection
        result = sock.connect_ex((host, port))
        is_port_open = result == 0
        response_time = time.time() - start_time
        
        banner = None
        if is_port_open and config.banner_grab:
            banner = await grab_banner(sock, config)
        
        return ScanResult(
            host=host,
            port=port,
            is_open=is_port_open,
            response_time=response_time,
            banner=banner
        )
        
    except Exception as e:
        return ScanResult(
            host=host,
            port=port,
            is_open=False,
            response_time=time.time() - start_time
        )
    finally:
        try:
            sock.close()
        except:
            pass

async def grab_banner(sock: socket.socket, config: ScanConfig) -> Optional[str]:
    """Grab service banner from open port."""
    try:
        # Send common probes
        probes = [b'\r\n', b'GET / HTTP/1.0\r\n\r\n', b'HELP\r\n']
        
        for probe in probes:
            try:
                sock.send(probe)
                banner = sock.recv(1024)
                if banner:
                    return banner.decode('utf-8', errors='ignore').strip()
            except:
                continue
                
    except Exception:
        pass
    
    return None

async def scan_port_range(host: str, start_port: int, end_port: int, 
                         config: ScanConfig) -> List[ScanResult]:
    """Scan a range of ports asynchronously."""
    ports_to_scan = range(start_port, end_port + 1)
    
    # Use ThreadPoolExecutor for I/O-bound operations
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        tasks = [
            loop.run_in_executor(executor, lambda p=port: 
                asyncio.run(scan_single_port(host, p, config)))
            for port in ports_to_scan
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter out exceptions and open ports
    valid_results = [
        result for result in results 
        if isinstance(result, ScanResult) and result.is_open
    ]
    
    return valid_results

def enrich_scan_results(results: List[ScanResult]) -> List[ScanResult]:
    """Add service names to scan results."""
    services = get_common_services()
    
    for result in results:
        if result.port in services:
            result.service_name = services[result.port]
    
    return results

def format_scan_report(results: List[ScanResult]) -> str:
    """Format scan results as a readable report."""
    if not results:
        return "No open ports found."
    
    report_lines = [f"Scan Results for {results[0].host}:"]
    
    for result in sorted(results, key=lambda r: r.port):
        service_info = f" ({result.service_name})" if result.service_name else ""
        banner_info = f" - {result.banner}" if result.banner else ""
        report_lines.append(
            f"  Port {result.port}{service_info}: "
            f"{'OPEN' if result.is_open else 'CLOSED'} "
            f"({result.response_time:.3f}s){banner_info}"
        )
    
    return "\n".join(report_lines)

async def scan_with_retry(host: str, port: int, config: ScanConfig) -> ScanResult:
    """Scan port with retry logic."""
    for attempt in range(config.retry_count):
        result = await scan_single_port(host, port, config)
        if result.is_open:
            return result
        await asyncio.sleep(0.1)  # Brief delay between retries
    
    return result

def analyze_scan_results(results: List[ScanResult]) -> Dict[str, any]:
    """Analyze scan results for patterns and statistics."""
    if not results:
        return {"total_ports": 0, "open_ports": 0}
    
    open_ports = [r for r in results if r.is_open]
    services = {}
    
    for result in open_ports:
        service = result.service_name or "unknown"
        services[service] = services.get(service, 0) + 1
    
    return {
        "total_ports": len(results),
        "open_ports": len(open_ports),
        "open_percentage": (len(open_ports) / len(results)) * 100,
        "services": services,
        "avg_response_time": sum(r.response_time for r in open_ports) / len(open_ports) if open_ports else 0
    }

# Named exports for main functionality
__all__ = [
    'scan_single_port',
    'scan_port_range', 
    'enrich_scan_results',
    'format_scan_report',
    'scan_with_retry',
    'analyze_scan_results',
    'ScanResult',
    'ScanConfig'
] 