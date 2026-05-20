from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import socket
import struct
import ipaddress
import asyncio
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union, Tuple
import aiohttp
import aiofiles
        from urllib.parse import urlparse
        from urllib.parse import urlparse
from typing import Any, List, Dict, Optional
import logging
"""
Network helpers for cybersecurity operations.

Provides tools for:
- Network protocol analysis
- Packet manipulation
- Network monitoring
- Protocol validation
- Network utilities
"""


@dataclass
class NetworkConfig:
    """Configuration for network operations."""
    timeout: float = 10.0
    max_retries: int = 3
    buffer_size: int = 4096
    enable_ipv6: bool = True
    default_port: int = 80
    user_agent: str = "Cybersecurity-Tool/1.0"

@dataclass
class NetworkResult:
    """Result of a network operation."""
    success: bool = False
    data: Optional[bytes] = None
    response_time: float = 0.0
    status_code: Optional[int] = None
    headers: Optional[Dict[str, str]] = None
    error_message: Optional[str] = None

# CPU-bound operations (use 'def')
def validate_ip_address(ip: str) -> bool:
    """Validate IP address format - CPU intensive."""
    try:
        ipaddress.ip_address(ip)
        return True
    except ValueError:
        return False

def validate_port(port: int) -> bool:
    """Validate port number - CPU intensive."""
    return 1 <= port <= 65535

def parse_ip_range(ip_range: str) -> List[str]:
    """Parse IP range (e.g., '192.168.1.0/24') - CPU intensive."""
    try:
        network = ipaddress.ip_network(ip_range, strict=False)
        return [str(ip) for ip in network.hosts()]
    except ValueError:
        return []

def calculate_checksum(data: bytes) -> int:
    """Calculate IP checksum - CPU intensive."""
    if len(data) % 2 == 1:
        data += b'\x00'
    
    checksum = 0
    for i in range(0, len(data), 2):
        checksum += struct.unpack('!H', data[i:i+2])[0]
    
    while checksum >> 16:
        checksum = (checksum & 0xFFFF) + (checksum >> 16)
    
    return ~checksum & 0xFFFF

async def parse_http_headers(headers_raw: bytes) -> Dict[str, str]:
    """Parse HTTP headers - CPU intensive."""
    headers = {}
    try:
        lines = headers_raw.decode('utf-8').split('\r\n')
        for line in lines[1:]:  # Skip status line
            if ':' in line:
                key, value = line.split(':', 1)
                headers[key.strip().lower()] = value.strip()
    except:
        pass
    return headers

def validate_url(url: str) -> bool:
    """Validate URL format - CPU intensive."""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

def extract_domain_from_url(url: str) -> Optional[str]:
    """Extract domain from URL - CPU intensive."""
    try:
        result = urlparse(url)
        return result.netloc
    except:
        return None

def parse_mac_address(mac: str) -> Optional[bytes]:
    """Parse MAC address - CPU intensive."""
    try:
        # Remove separators
        mac_clean = mac.replace(':', '').replace('-', '').replace('.', '')
        if len(mac_clean) != 12:
            return None
        
        # Convert to bytes
        return bytes.fromhex(mac_clean)
    except:
        return None

def format_mac_address(mac_bytes: bytes) -> str:
    """Format MAC address - CPU intensive."""
    return ':'.join(f'{b:02x}' for b in mac_bytes)

def calculate_network_broadcast(ip: str, netmask: str) -> str:
    """Calculate network broadcast address - CPU intensive."""
    try:
        network = ipaddress.IPv4Network(f"{ip}/{netmask}", strict=False)
        return str(network.broadcast_address)
    except:
        return ""

def is_private_ip(ip: str) -> bool:
    """Check if IP is private - CPU intensive."""
    try:
        ip_obj = ipaddress.ip_address(ip)
        return ip_obj.is_private
    except:
        return False

def is_reserved_ip(ip: str) -> bool:
    """Check if IP is reserved - CPU intensive."""
    try:
        ip_obj = ipaddress.ip_address(ip)
        return ip_obj.is_reserved
    except:
        return False

def parse_tcp_flags(flags: int) -> Dict[str, bool]:
    """Parse TCP flags - CPU intensive."""
    return {
        'fin': bool(flags & 0x01),
        'syn': bool(flags & 0x02),
        'rst': bool(flags & 0x04),
        'psh': bool(flags & 0x08),
        'ack': bool(flags & 0x10),
        'urg': bool(flags & 0x20)
    }

def format_tcp_flags(flags: Dict[str, bool]) -> str:
    """Format TCP flags as string - CPU intensive."""
    flag_names = ['fin', 'syn', 'rst', 'psh', 'ack', 'urg']
    return ','.join(flag for flag in flag_names if flags.get(flag, False))

# Async operations (use 'async def')
async def resolve_dns_async(hostname: str) -> List[str]:
    """Resolve hostname to IP addresses - I/O bound."""
    try:
        # Use asyncio to resolve DNS
        loop = asyncio.get_event_loop()
        info = await loop.run_in_executor(None, socket.getaddrinfo, hostname, None)
        
        ips = []
        for item in info:
            ip = item[4][0]
            if ip not in ips:
                ips.append(ip)
        
        return ips
    except Exception:
        return []

async def reverse_dns_lookup_async(ip: str) -> Optional[str]:
    """Perform reverse DNS lookup - I/O bound."""
    try:
        loop = asyncio.get_event_loop()
        hostname = await loop.run_in_executor(None, socket.gethostbyaddr, ip)
        return hostname[0]
    except Exception:
        return None

async def check_port_open_async(host: str, port: int, timeout: float = 5.0) -> bool:
    """Check if port is open - I/O bound."""
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port),
            timeout=timeout
        )
        writer.close()
        await writer.wait_closed()
        return True
    except:
        return False

async async def fetch_http_headers_async(url: str, config: NetworkConfig) -> NetworkResult:
    """Fetch HTTP headers - I/O bound."""
    start_time = time.time()
    
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=config.timeout)) as session:
            headers = {'User-Agent': config.user_agent}
            async with session.head(url, headers=headers) as response:
                response_time = time.time() - start_time
                
                return NetworkResult(
                    success=True,
                    status_code=response.status,
                    headers=dict(response.headers),
                    response_time=response_time
                )
    except Exception as e:
        return NetworkResult(
            success=False,
            response_time=time.time() - start_time,
            error_message=str(e)
        )

async async def fetch_url_content_async(url: str, config: NetworkConfig) -> NetworkResult:
    """Fetch URL content - I/O bound."""
    start_time = time.time()
    
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=config.timeout)) as session:
            headers = {'User-Agent': config.user_agent}
            async with session.get(url, headers=headers) as response:
                data = await response.read()
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                response_time = time.time() - start_time
                
                return NetworkResult(
                    success=True,
                    data=data,
                    status_code=response.status,
                    headers=dict(response.headers),
                    response_time=response_time
                )
    except Exception as e:
        return NetworkResult(
            success=False,
            response_time=time.time() - start_time,
            error_message=str(e)
        )

async def scan_port_range_async(host: str, start_port: int, end_port: int, 
                               config: NetworkConfig) -> List[Tuple[int, bool]]:
    """Scan port range asynchronously - I/O bound."""
    tasks = []
    for port in range(start_port, end_port + 1):
        task = check_port_open_async(host, port, config.timeout)
        tasks.append((port, task))
    
    results = []
    for port, task in tasks:
        try:
            is_open = await task
            results.append((port, is_open))
        except:
            results.append((port, False))
    
    return results

async def ping_host_async(host: str, count: int = 4, timeout: float = 5.0) -> Dict[str, Any]:
    """Ping host asynchronously - I/O bound."""
    try:
        # Use asyncio.subprocess for ping
        process = await asyncio.create_subprocess_exec(
            'ping', '-n', str(count), '-w', str(int(timeout * 1000)), host,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode == 0:
            # Parse ping output
            output = stdout.decode('utf-8', errors='ignore')
            lines = output.split('\n')
            
            # Extract statistics
            stats = {
                'host': host,
                'reachable': True,
                'packets_sent': count,
                'packets_received': 0,
                'packet_loss': 100.0,
                'min_time': 0.0,
                'max_time': 0.0,
                'avg_time': 0.0
            }
            
            for line in lines:
                if 'Packets: Sent =' in line:
                    # Parse packet statistics
                    parts = line.split(',')
                    for part in parts:
                        if 'Sent =' in part:
                            stats['packets_sent'] = int(part.split('=')[1].strip())
                        elif 'Received =' in part:
                            stats['packets_received'] = int(part.split('=')[1].strip())
                        elif 'Lost =' in part:
                            lost = int(part.split('=')[1].strip())
                            if stats['packets_sent'] > 0:
                                stats['packet_loss'] = (lost / stats['packets_sent']) * 100
                
                elif 'Minimum =' in line:
                    # Parse timing statistics
                    parts = line.split(',')
                    for part in parts:
                        if 'Minimum =' in part:
                            stats['min_time'] = float(part.split('=')[1].replace('ms', '').strip())
                        elif 'Maximum =' in part:
                            stats['max_time'] = float(part.split('=')[1].replace('ms', '').strip())
                        elif 'Average =' in part:
                            stats['avg_time'] = float(part.split('=')[1].replace('ms', '').strip())
            
            return stats
        else:
            return {
                'host': host,
                'reachable': False,
                'error': stderr.decode('utf-8', errors='ignore')
            }
    
    except Exception as e:
        return {
            'host': host,
            'reachable': False,
            'error': str(e)
        }

class NetworkHelper:
    """Main network helper class."""
    
    def __init__(self, config: NetworkConfig):
        
    """__init__ function."""
self.config = config
    
    async def get_host_info(self, hostname: str) -> Dict[str, Any]:
        """Get comprehensive host information."""
        info = {
            'hostname': hostname,
            'ips': [],
            'reverse_dns': None,
            'is_resolvable': False
        }
        
        # Resolve DNS
        ips = await resolve_dns_async(hostname)
        info['ips'] = ips
        info['is_resolvable'] = len(ips) > 0
        
        # Reverse DNS for first IP
        if ips:
            info['reverse_dns'] = await reverse_dns_lookup_async(ips[0])
        
        return info
    
    async def scan_common_ports(self, host: str) -> Dict[str, Any]:
        """Scan common ports on host."""
        common_ports = [21, 22, 23, 25, 53, 80, 110, 143, 443, 993, 995, 3306, 5432, 8080]
        
        results = await scan_port_range_async(host, min(common_ports), max(common_ports), self.config)
        
        open_ports = [port for port, is_open in results if is_open]
        
        return {
            'host': host,
            'scanned_ports': len(common_ports),
            'open_ports': open_ports,
            'open_count': len(open_ports)
        }
    
    async def check_web_server(self, url: str) -> Dict[str, Any]:
        """Check web server information."""
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url
        
        headers_result = await fetch_http_headers_async(url, self.config)
        
        info = {
            'url': url,
            'reachable': headers_result.success,
            'status_code': headers_result.status_code,
            'response_time': headers_result.response_time,
            'headers': headers_result.headers or {},
            'server': headers_result.headers.get('server', 'Unknown') if headers_result.headers else 'Unknown'
        }
        
        return info
    
    def validate_network_config(self, ip: str, netmask: str) -> Dict[str, Any]:
        """Validate network configuration."""
        try:
            network = ipaddress.IPv4Network(f"{ip}/{netmask}", strict=False)
            
            return {
                'valid': True,
                'network': str(network.network_address),
                'broadcast': str(network.broadcast_address),
                'netmask': str(network.netmask),
                'num_hosts': network.num_addresses - 2,  # Exclude network and broadcast
                'is_private': network.is_private
            }
        except Exception as e:
            return {
                'valid': False,
                'error': str(e)
            }

class ProtocolHelper:
    """Network protocol helper class."""
    
    def __init__(self, config: NetworkConfig):
        
    """__init__ function."""
self.config = config
    
    async def parse_http_request(self, request_data: bytes) -> Dict[str, Any]:
        """Parse HTTP request."""
        try:
            lines = request_data.decode('utf-8').split('\r\n')
            if not lines:
                return {'valid': False, 'error': 'Empty request'}
            
            # Parse request line
            request_line = lines[0].split(' ')
            if len(request_line) != 3:
                return {'valid': False, 'error': 'Invalid request line'}
            
            method, path, version = request_line
            
            # Parse headers
            headers = {}
            body_start = 0
            for i, line in enumerate(lines[1:], 1):
                if not line:  # Empty line indicates end of headers
                    body_start = i + 1
                    break
                if ':' in line:
                    key, value = line.split(':', 1)
                    headers[key.strip().lower()] = value.strip()
            
            # Get body
            body = b'\r\n'.join(lines[body_start:]).encode('utf-8') if body_start < len(lines) else b''
            
            return {
                'valid': True,
                'method': method,
                'path': path,
                'version': version,
                'headers': headers,
                'body': body,
                'content_length': len(body)
            }
        except Exception as e:
            return {'valid': False, 'error': str(e)}
    
    async def parse_http_response(self, response_data: bytes) -> Dict[str, Any]:
        """Parse HTTP response."""
        try:
            lines = response_data.decode('utf-8').split('\r\n')
            if not lines:
                return {'valid': False, 'error': 'Empty response'}
            
            # Parse status line
            status_line = lines[0].split(' ', 2)
            if len(status_line) < 2:
                return {'valid': False, 'error': 'Invalid status line'}
            
            version = status_line[0]
            status_code = int(status_line[1])
            status_text = status_line[2] if len(status_line) > 2 else ''
            
            # Parse headers
            headers = {}
            body_start = 0
            for i, line in enumerate(lines[1:], 1):
                if not line:  # Empty line indicates end of headers
                    body_start = i + 1
                    break
                if ':' in line:
                    key, value = line.split(':', 1)
                    headers[key.strip().lower()] = value.strip()
            
            # Get body
            body = b'\r\n'.join(lines[body_start:]).encode('utf-8') if body_start < len(lines) else b''
            
            return {
                'valid': True,
                'version': version,
                'status_code': status_code,
                'status_text': status_text,
                'headers': headers,
                'body': body,
                'content_length': len(body)
            }
        except Exception as e:
            return {'valid': False, 'error': str(e)}
    
    def validate_tcp_packet(self, packet_data: bytes) -> Dict[str, Any]:
        """Validate TCP packet structure."""
        if len(packet_data) < 20:  # Minimum TCP header size
            return {'valid': False, 'error': 'Packet too short'}
        
        try:
            # Parse TCP header
            header = struct.unpack('!HHLLBBHHH', packet_data[:20])
            
            source_port, dest_port, seq_num, ack_num, data_offset, flags, window, checksum, urgent_ptr = header
            
            # Extract flags
            tcp_flags = parse_tcp_flags(flags)
            
            return {
                'valid': True,
                'source_port': source_port,
                'dest_port': dest_port,
                'seq_num': seq_num,
                'ack_num': ack_num,
                'data_offset': data_offset * 4,  # Convert to bytes
                'flags': tcp_flags,
                'window': window,
                'checksum': checksum,
                'urgent_ptr': urgent_ptr,
                'payload_length': len(packet_data) - (data_offset * 4)
            }
        except Exception as e:
            return {'valid': False, 'error': str(e)}
    
    def create_tcp_packet(self, source_port: int, dest_port: int, seq_num: int, 
                         ack_num: int, flags: Dict[str, bool], payload: bytes = b'') -> bytes:
        """Create TCP packet."""
        # Calculate flags
        flag_value = 0
        if flags.get('fin', False): flag_value |= 0x01
        if flags.get('syn', False): flag_value |= 0x02
        if flags.get('rst', False): flag_value |= 0x04
        if flags.get('psh', False): flag_value |= 0x08
        if flags.get('ack', False): flag_value |= 0x10
        if flags.get('urg', False): flag_value |= 0x20
        
        # Create header
        data_offset = 5  # 5 32-bit words
        header = struct.pack('!HHLLBBHHH',
                           source_port, dest_port, seq_num, ack_num,
                           data_offset, flag_value, 65535, 0, 0)
        
        return header + payload 