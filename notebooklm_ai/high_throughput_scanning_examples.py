from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import asyncio
import aiohttp
import asyncssh
import socket
import ssl
import time
import logging
import json
import random
from typing import Dict, List, Set, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager
from collections import defaultdict, deque
import ipaddress
from concurrent.futures import ThreadPoolExecutor
import weakref
import gc
from pathlib import Path
import pickle
import hashlib
import threading
from datetime import datetime, timedelta
import signal
import sys
from typing import Any, List, Dict, Optional
"""
High-Throughput Scanning and Enumeration with Asyncio and Connection Pooling

This module provides comprehensive examples for high-throughput scanning and enumeration
using asyncio and connection pooling for optimal performance and resource management.

Key Features:
- Asynchronous network scanning with connection pooling
- Service enumeration with rate limiting
- Resource management and optimization
- Distributed scanning coordination
- Performance monitoring and metrics
- Error handling and retry logic
- Stealth and detection avoidance
"""


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ScanType(Enum):
    """Enumeration of scan types"""
    TCP_CONNECT = "tcp_connect"
    TCP_SYN = "tcp_syn"
    UDP = "udp"
    HTTP = "http"
    HTTPS = "https"
    SSH = "ssh"
    DNS = "dns"
    SMTP = "smtp"
    FTP = "ftp"
    CUSTOM = "custom"


class ConnectionState(Enum):
    """Connection state enumeration"""
    IDLE = "idle"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    BUSY = "busy"
    ERROR = "error"
    CLOSED = "closed"


@dataclass
class ScanTarget:
    """Target for scanning operations"""
    host: str
    port: int
    protocol: str = "tcp"
    timeout: float = 5.0
    retries: int = 3
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> Any:
        if not self.host or not self.port:
            raise ValueError("Host and port are required")
        
        if not (1 <= self.port <= 65535):
            raise ValueError("Port must be between 1 and 65535")
        
        if self.timeout <= 0:
            raise ValueError("Timeout must be positive")
        
        if self.retries < 0:
            raise ValueError("Retries cannot be negative")


@dataclass
class ScanResult:
    """Result of a scanning operation"""
    target: ScanTarget
    success: bool
    response_time: float
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary"""
        return {
            "host": self.target.host,
            "port": self.target.port,
            "protocol": self.target.protocol,
            "success": self.success,
            "response_time": self.response_time,
            "data": self.data,
            "error": self.error,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class ConnectionPoolConfig:
    """Configuration for connection pooling"""
    max_connections: int = 100
    max_connections_per_host: int = 10
    connection_timeout: float = 10.0
    keepalive_timeout: float = 30.0
    retry_attempts: int = 3
    retry_delay: float = 1.0
    enable_ssl: bool = True
    verify_ssl: bool = False
    user_agent: str = "HighThroughputScanner/1.0"
    enable_compression: bool = True
    max_redirects: int = 5
    enable_cookies: bool = False


class ConnectionPool:
    """Asynchronous connection pool for high-throughput scanning"""
    
    def __init__(self, config: ConnectionPoolConfig):
        
    """__init__ function."""
self.config = config
        self._connections: Dict[str, deque] = defaultdict(deque)
        self._connection_states: Dict[str, ConnectionState] = {}
        self._semaphores: Dict[str, asyncio.Semaphore] = {}
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._stats = {
            "total_connections": 0,
            "active_connections": 0,
            "failed_connections": 0,
            "reused_connections": 0
        }
        
    async def __aenter__(self) -> Any:
        """Async context manager entry"""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> Any:
        """Async context manager exit"""
        await self.close()
    
    async def start(self) -> Any:
        """Start the connection pool"""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("Connection pool started")
    
    async def close(self) -> Any:
        """Close the connection pool"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Close all connections
        async with self._lock:
            for host, connections in self._connections.items():
                while connections:
                    conn = connections.popleft()
                    try:
                        await conn.close()
                    except Exception as e:
                        logger.warning(f"Error closing connection to {host}: {e}")
        
        logger.info("Connection pool closed")
    
    async def get_connection(self, host: str, port: int, protocol: str = "http") -> Optional[Dict[str, Any]]:
        """Get a connection from the pool or create a new one"""
        key = f"{protocol}://{host}:{port}"
        
        async with self._lock:
            if key not in self._semaphores:
                self._semaphores[key] = asyncio.Semaphore(self.config.max_connections_per_host)
        
        async with self._semaphores[key]:
            # Try to reuse existing connection
            if self._connections[key]:
                conn = self._connections[key].popleft()
                if await self._is_connection_valid(conn):
                    self._stats["reused_connections"] += 1
                    return conn
                else:
                    await self._close_connection(conn)
            
            # Create new connection
            conn = await self._create_connection(host, port, protocol)
            self._stats["total_connections"] += 1
            self._stats["active_connections"] += 1
            return conn
    
    async def return_connection(self, host: str, port: int, protocol: str, connection: Any):
        """Return a connection to the pool"""
        key = f"{protocol}://{host}:{port}"
        
        if not await self._is_connection_valid(connection):
            await self._close_connection(connection)
            return
        
        async with self._lock:
            if len(self._connections[key]) < self.config.max_connections_per_host:
                self._connections[key].append(connection)
            else:
                await self._close_connection(connection)
    
    async def _create_connection(self, host: str, port: int, protocol: str) -> Any:
        """Create a new connection"""
        try:
            if protocol == "http":
                connector = aiohttp.TCPConnector(
                    limit=self.config.max_connections_per_host,
                    limit_per_host=self.config.max_connections_per_host,
                    ttl_dns_cache=300,
                    use_dns_cache=True,
                    enable_cleanup_closed=True
                )
                
                timeout = aiohttp.ClientTimeout(total=self.config.connection_timeout)
                
                session = aiohttp.ClientSession(
                    connector=connector,
                    timeout=timeout,
                    headers={"User-Agent": self.config.user_agent},
                    cookie_jar=None if not self.config.enable_cookies else None
                )
                
                return session
            
            elif protocol == "ssh":
                return await asyncssh.connect(
                    host,
                    port=port,
                    timeout=self.config.connection_timeout,
                    known_hosts=None
                )
            
            elif protocol == "tcp":
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(host, port),
                    timeout=self.config.connection_timeout
                )
                return (reader, writer)
            
            else:
                raise ValueError(f"Unsupported protocol: {protocol}")
                
        except Exception as e:
            self._stats["failed_connections"] += 1
            logger.error(f"Failed to create connection to {host}:{port}: {e}")
            raise
    
    async def _is_connection_valid(self, connection: Any) -> bool:
        """Check if a connection is still valid"""
        try:
            if hasattr(connection, 'closed'):
                return not connection.closed
            elif hasattr(connection, 'is_closed'):
                return not connection.is_closed()
            else:
                # For raw socket connections, try a simple operation
                return True
        except Exception:
            return False
    
    async def _close_connection(self, connection: Any):
        """Close a connection"""
        try:
            if hasattr(connection, 'close'):
                await connection.close()
            elif hasattr(connection, '__aexit__'):
                await connection.__aexit__(None, None, None)
            else:
                # For raw socket connections
                if isinstance(connection, tuple) and len(connection) == 2:
                    reader, writer = connection
                    writer.close()
                    await writer.wait_closed()
        except Exception as e:
            logger.warning(f"Error closing connection: {e}")
        finally:
            self._stats["active_connections"] -= 1
    
    async def _cleanup_loop(self) -> Any:
        """Background cleanup loop"""
        while True:
            try:
                await asyncio.sleep(60)  # Cleanup every minute
                await self._cleanup_expired_connections()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
    
    async def _cleanup_expired_connections(self) -> Any:
        """Clean up expired connections"""
        async with self._lock:
            for key, connections in list(self._connections.items()):
                valid_connections = deque()
                while connections:
                    conn = connections.popleft()
                    if await self._is_connection_valid(conn):
                        valid_connections.append(conn)
                    else:
                        await self._close_connection(conn)
                
                self._connections[key] = valid_connections
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics"""
        return {
            **self._stats,
            "pool_size": sum(len(conns) for conns in self._connections.values()),
            "host_count": len(self._connections)
        }


class HighThroughputScanner:
    """High-throughput scanner with asyncio and connection pooling"""
    
    def __init__(self, config: ConnectionPoolConfig):
        
    """__init__ function."""
self.config = config
        self.pool = ConnectionPool(config)
        self._scan_queue: asyncio.Queue = asyncio.Queue()
        self._results_queue: asyncio.Queue = asyncio.Queue()
        self._workers: List[asyncio.Task] = []
        self._running = False
        self._stats = {
            "scans_completed": 0,
            "scans_failed": 0,
            "total_response_time": 0.0,
            "start_time": None,
            "end_time": None
        }
        self._rate_limiter = asyncio.Semaphore(config.max_connections)
    
    async def __aenter__(self) -> Any:
        """Async context manager entry"""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> Any:
        """Async context manager exit"""
        await self.stop()
    
    async def start(self) -> Any:
        """Start the scanner"""
        if self._running:
            return
        
        self._running = True
        self._stats["start_time"] = datetime.now()
        
        # Start worker tasks
        worker_count = min(self.config.max_connections, 50)
        self._workers = [
            asyncio.create_task(self._worker(f"worker-{i}"))
            for i in range(worker_count)
        ]
        
        logger.info(f"High-throughput scanner started with {worker_count} workers")
    
    async def stop(self) -> Any:
        """Stop the scanner"""
        if not self._running:
            return
        
        self._running = False
        self._stats["end_time"] = datetime.now()
        
        # Cancel worker tasks
        for worker in self._workers:
            worker.cancel()
        
        # Wait for workers to finish
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
        
        # Close connection pool
        await self.pool.close()
        
        logger.info("High-throughput scanner stopped")
    
    async def scan_targets(self, targets: List[ScanTarget]) -> List[ScanResult]:
        """Scan multiple targets concurrently"""
        if not self._running:
            await self.start()
        
        # Add targets to queue
        for target in targets:
            await self._scan_queue.put(target)
        
        # Collect results
        results = []
        expected_results = len(targets)
        
        while len(results) < expected_results:
            try:
                result = await asyncio.wait_for(
                    self._results_queue.get(),
                    timeout=30.0
                )
                results.append(result)
            except asyncio.TimeoutError:
                logger.warning("Timeout waiting for scan results")
                break
        
        return results
    
    async def _worker(self, worker_id: str):
        """Worker task for processing scan targets"""
        logger.debug(f"Worker {worker_id} started")
        
        while self._running:
            try:
                # Get target from queue
                try:
                    target = await asyncio.wait_for(
                        self._scan_queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Process target
                async with self._rate_limiter:
                    result = await self._scan_target(target)
                    await self._results_queue.put(result)
                
                # Mark task as done
                self._scan_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                continue
        
        logger.debug(f"Worker {worker_id} stopped")
    
    async def _scan_target(self, target: ScanTarget) -> ScanResult:
        """Scan a single target"""
        start_time = time.time()
        
        try:
            if target.protocol in ["http", "https"]:
                data = await self._scan_http(target)
            elif target.protocol == "ssh":
                data = await self._scan_ssh(target)
            elif target.protocol == "tcp":
                data = await self._scan_tcp(target)
            else:
                data = await self._scan_custom(target)
            
            response_time = time.time() - start_time
            self._stats["scans_completed"] += 1
            self._stats["total_response_time"] += response_time
            
            return ScanResult(
                target=target,
                success=True,
                response_time=response_time,
                data=data
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            self._stats["scans_failed"] += 1
            
            return ScanResult(
                target=target,
                success=False,
                response_time=response_time,
                error=str(e)
            )
    
    async async def _scan_http(self, target: ScanTarget) -> Dict[str, Any]:
        """Scan HTTP/HTTPS target"""
        session = await self.pool.get_connection(
            target.host, target.port, "http"
        )
        
        try:
            protocol = "https" if target.protocol == "https" else "http"
            url = f"{protocol}://{target.host}:{target.port}/"
            
            async with session.get(url, timeout=target.timeout) as response:
                return {
                    "status_code": response.status,
                    "headers": dict(response.headers),
                    "content_length": response.content_length,
                    "content_type": response.content_type,
                    "server": response.headers.get("Server"),
                    "title": await self._extract_title(response)
                }
        finally:
            await self.pool.return_connection(
                target.host, target.port, "http", session
            )
    
    async def _scan_ssh(self, target: ScanTarget) -> Dict[str, Any]:
        """Scan SSH target"""
        conn = await self.pool.get_connection(
            target.host, target.port, "ssh"
        )
        
        try:
            # Get SSH server information
            transport = conn.get_transport()
            return {
                "ssh_version": transport.get_version(),
                "encryption": transport.get_encryption(),
                "compression": transport.get_compression(),
                "mac": transport.get_mac(),
                "key_exchange": transport.get_key_exchange()
            }
        finally:
            await self.pool.return_connection(
                target.host, target.port, "ssh", conn
            )
    
    async def _scan_tcp(self, target: ScanTarget) -> Dict[str, Any]:
        """Scan TCP target"""
        reader, writer = await self.pool.get_connection(
            target.host, target.port, "tcp"
        )
        
        try:
            # Send a simple probe
            writer.write(b"GET / HTTP/1.1\r\nHost: " + target.host.encode() + b"\r\n\r\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            await writer.drain()
            
            # Read response
            data = await asyncio.wait_for(
                reader.read(1024),
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                timeout=target.timeout
            )
            
            return {
                "response_size": len(data),
                "response_preview": data[:200].decode('utf-8', errors='ignore'),
                "banner": self._extract_banner(data)
            }
        finally:
            await self.pool.return_connection(
                target.host, target.port, "tcp", (reader, writer)
            )
    
    async def _scan_custom(self, target: ScanTarget) -> Dict[str, Any]:
        """Scan custom protocol target"""
        # Implement custom protocol scanning logic
        return {"protocol": target.protocol, "custom_scan": True}
    
    async def _extract_title(self, response) -> str:
        """Extract page title from HTTP response"""
        try:
            content = await response.text()
            if "<title>" in content:
                start = content.find("<title>") + 7
                end = content.find("</title>", start)
                if end > start:
                    return content[start:end].strip()
        except Exception:
            pass
        return ""
    
    def _extract_banner(self, data: bytes) -> str:
        """Extract service banner from response data"""
        try:
            text = data.decode('utf-8', errors='ignore')
            lines = text.split('\n')
            for line in lines[:5]:  # Check first 5 lines
                line = line.strip()
                if line and not line.startswith('HTTP/'):
                    return line
        except Exception:
            pass
        return ""
    
    def get_stats(self) -> Dict[str, Any]:
        """Get scanner statistics"""
        stats = {
            **self._stats,
            **self.pool.get_stats()
        }
        
        if stats["start_time"] and stats["end_time"]:
            duration = (stats["end_time"] - stats["start_time"]).total_seconds()
            stats["duration"] = duration
            stats["scans_per_second"] = stats["scans_completed"] / duration if duration > 0 else 0
            stats["avg_response_time"] = (
                stats["total_response_time"] / stats["scans_completed"]
                if stats["scans_completed"] > 0 else 0
            )
        
        return stats


class ServiceEnumerator:
    """Service enumeration with high-throughput capabilities"""
    
    def __init__(self, scanner: HighThroughputScanner):
        
    """__init__ function."""
self.scanner = scanner
        self._service_signatures = self._load_service_signatures()
        self._enumeration_results: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    
    def _load_service_signatures(self) -> Dict[str, Dict[str, Any]]:
        """Load service signatures for identification"""
        return {
            "http": {
                "ports": [80, 443, 8080, 8443, 3000, 8000, 9000],
                "protocols": ["http", "https"]
            },
            "ssh": {
                "ports": [22, 2222, 222],
                "protocols": ["ssh"]
            },
            "ftp": {
                "ports": [21, 2121],
                "protocols": ["ftp"]
            },
            "smtp": {
                "ports": [25, 587, 465],
                "protocols": ["smtp"]
            },
            "dns": {
                "ports": [53],
                "protocols": ["dns"]
            },
            "database": {
                "ports": [3306, 5432, 6379, 27017, 1433, 1521],
                "protocols": ["tcp"]
            }
        }
    
    async def enumerate_services(self, hosts: List[str], 
                               service_types: Optional[List[str]] = None) -> Dict[str, List[Dict[str, Any]]]:
        """Enumerate services on multiple hosts"""
        if service_types is None:
            service_types = list(self._service_signatures.keys())
        
        # Generate scan targets
        targets = []
        for host in hosts:
            for service_type in service_types:
                if service_type in self._service_signatures:
                    signature = self._service_signatures[service_type]
                    for port in signature["ports"]:
                        for protocol in signature["protocols"]:
                            targets.append(ScanTarget(
                                host=host,
                                port=port,
                                protocol=protocol,
                                timeout=5.0,
                                retries=2
                            ))
        
        # Perform scans
        results = await self.scanner.scan_targets(targets)
        
        # Process results
        for result in results:
            if result.success:
                service_info = self._identify_service(result)
                if service_info:
                    self._enumeration_results[result.target.host].append(service_info)
        
        return dict(self._enumeration_results)
    
    def _identify_service(self, result: ScanResult) -> Optional[Dict[str, Any]]:
        """Identify service based on scan result"""
        service_info = {
            "port": result.target.port,
            "protocol": result.target.protocol,
            "response_time": result.response_time,
            "data": result.data
        }
        
        # HTTP/HTTPS identification
        if result.target.protocol in ["http", "https"]:
            if "status_code" in result.data:
                service_info["service"] = "web"
                service_info["server"] = result.data.get("server", "Unknown")
                service_info["title"] = result.data.get("title", "")
        
        # SSH identification
        elif result.target.protocol == "ssh":
            if "ssh_version" in result.data:
                service_info["service"] = "ssh"
                service_info["version"] = result.data.get("ssh_version", "")
        
        # TCP banner identification
        elif result.target.protocol == "tcp":
            banner = result.data.get("banner", "")
            service_info["service"] = self._identify_from_banner(banner)
            service_info["banner"] = banner
        
        return service_info
    
    def _identify_from_banner(self, banner: str) -> str:
        """Identify service from banner"""
        banner_lower = banner.lower()
        
        if "ssh" in banner_lower:
            return "ssh"
        elif "ftp" in banner_lower:
            return "ftp"
        elif "smtp" in banner_lower:
            return "smtp"
        elif "mysql" in banner_lower:
            return "mysql"
        elif "postgresql" in banner_lower:
            return "postgresql"
        elif "redis" in banner_lower:
            return "redis"
        elif "mongodb" in banner_lower:
            return "mongodb"
        else:
            return "unknown"


class NetworkScanner:
    """Network scanner with high-throughput capabilities"""
    
    def __init__(self, scanner: HighThroughputScanner):
        
    """__init__ function."""
self.scanner = scanner
        self._discovered_hosts: Set[str] = set()
        self._network_ranges: List[str] = []
    
    async def scan_network(self, network: str, 
                          port_ranges: Optional[List[Tuple[int, int]]] = None) -> Dict[str, List[int]]:
        """Scan a network range for open ports"""
        if port_ranges is None:
            port_ranges = [(1, 1024)]  # Default to common ports
        
        # Generate IP addresses
        try:
            network_obj = ipaddress.ip_network(network, strict=False)
            hosts = [str(ip) for ip in network_obj.hosts()]
        except ValueError as e:
            logger.error(f"Invalid network range: {network}")
            return {}
        
        # Generate scan targets
        targets = []
        for host in hosts:
            for start_port, end_port in port_ranges:
                for port in range(start_port, end_port + 1):
                    targets.append(ScanTarget(
                        host=host,
                        port=port,
                        protocol="tcp",
                        timeout=3.0,
                        retries=1
                    ))
        
        # Perform scans
        results = await self.scanner.scan_targets(targets)
        
        # Process results
        open_ports: Dict[str, List[int]] = defaultdict(list)
        for result in results:
            if result.success:
                open_ports[result.target.host].append(result.target.port)
                self._discovered_hosts.add(result.target.host)
        
        return dict(open_ports)
    
    async def ping_sweep(self, network: str) -> List[str]:
        """Perform ping sweep to discover live hosts"""
        try:
            network_obj = ipaddress.ip_network(network, strict=False)
            hosts = [str(ip) for ip in network_obj.hosts()]
        except ValueError as e:
            logger.error(f"Invalid network range: {network}")
            return []
        
        # Create ping targets
        targets = [
            ScanTarget(host=host, port=80, protocol="tcp", timeout=2.0, retries=1)
            for host in hosts
        ]
        
        # Perform scans
        results = await self.scanner.scan_targets(targets)
        
        # Extract live hosts
        live_hosts = [
            result.target.host for result in results
            if result.success
        ]
        
        self._discovered_hosts.update(live_hosts)
        return live_hosts
    
    def get_discovered_hosts(self) -> List[str]:
        """Get list of discovered hosts"""
        return list(self._discovered_hosts)


class PerformanceMonitor:
    """Performance monitoring for high-throughput scanning"""
    
    def __init__(self) -> Any:
        self._metrics: Dict[str, List[float]] = defaultdict(list)
        self._start_time = time.time()
        self._monitoring_task: Optional[asyncio.Task] = None
    
    async def start_monitoring(self) -> Any:
        """Start performance monitoring"""
        self._monitoring_task = asyncio.create_task(self._monitor_loop())
    
    async def stop_monitoring(self) -> Any:
        """Stop performance monitoring"""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
    
    async def record_metric(self, name: str, value: float):
        """Record a performance metric"""
        self._metrics[name].append(value)
    
    async def _monitor_loop(self) -> Any:
        """Background monitoring loop"""
        while True:
            try:
                await asyncio.sleep(10)  # Monitor every 10 seconds
                await self._log_metrics()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
    
    async def _log_metrics(self) -> Any:
        """Log current metrics"""
        current_time = time.time()
        duration = current_time - self._start_time
        
        metrics_summary = {}
        for name, values in self._metrics.items():
            if values:
                metrics_summary[name] = {
                    "count": len(values),
                    "avg": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "latest": values[-1] if values else 0
                }
        
        logger.info(f"Performance metrics (duration: {duration:.2f}s): {json.dumps(metrics_summary, indent=2)}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all recorded metrics"""
        return dict(self._metrics)


# Example usage and demonstration functions

async def demonstrate_high_throughput_scanning():
    """Demonstrate high-throughput scanning capabilities"""
    logger.info("Starting high-throughput scanning demonstration")
    
    # Configuration
    config = ConnectionPoolConfig(
        max_connections=200,
        max_connections_per_host=20,
        connection_timeout=10.0,
        retry_attempts=3
    )
    
    # Create scanner
    async with HighThroughputScanner(config) as scanner:
        # Create service enumerator
        enumerator = ServiceEnumerator(scanner)
        
        # Create network scanner
        network_scanner = NetworkScanner(scanner)
        
        # Create performance monitor
        monitor = PerformanceMonitor()
        await monitor.start_monitoring()
        
        try:
            # Example 1: Service enumeration
            logger.info("Example 1: Service enumeration")
            hosts = ["example.com", "google.com", "github.com"]
            services = await enumerator.enumerate_services(hosts, ["http", "ssh"])
            
            for host, host_services in services.items():
                logger.info(f"Host: {host}")
                for service in host_services:
                    logger.info(f"  - {service['service']} on port {service['port']}")
            
            # Example 2: Network scanning
            logger.info("Example 2: Network scanning")
            network = "192.168.1.0/24"
            open_ports = await network_scanner.scan_network(network, [(80, 443), (22, 22)])
            
            for host, ports in open_ports.items():
                logger.info(f"Host {host}: open ports {ports}")
            
            # Example 3: Custom scan targets
            logger.info("Example 3: Custom scan targets")
            custom_targets = [
                ScanTarget("example.com", 80, "http"),
                ScanTarget("example.com", 443, "https"),
                ScanTarget("example.com", 22, "ssh"),
                ScanTarget("example.com", 3306, "tcp"),
            ]
            
            results = await scanner.scan_targets(custom_targets)
            
            for result in results:
                if result.success:
                    logger.info(f"Success: {result.target.host}:{result.target.port} "
                              f"({result.response_time:.3f}s)")
                else:
                    logger.error(f"Failed: {result.target.host}:{result.target.port} "
                               f"- {result.error}")
            
            # Print statistics
            stats = scanner.get_stats()
            logger.info(f"Scanner statistics: {json.dumps(stats, indent=2)}")
            
        finally:
            await monitor.stop_monitoring()


async def demonstrate_connection_pooling():
    """Demonstrate connection pooling capabilities"""
    logger.info("Starting connection pooling demonstration")
    
    config = ConnectionPoolConfig(
        max_connections=50,
        max_connections_per_host=5,
        connection_timeout=5.0
    )
    
    async with ConnectionPool(config) as pool:
        # Test HTTP connections
        hosts = ["httpbin.org", "jsonplaceholder.typicode.com", "api.github.com"]
        
        for host in hosts:
            logger.info(f"Testing connection to {host}")
            
            # Get connection
            conn = await pool.get_connection(host, 80, "http")
            
            try:
                # Use connection
                async with conn.get(f"http://{host}/") as response:
                    logger.info(f"Response status: {response.status}")
            finally:
                # Return connection
                await pool.return_connection(host, 80, "http", conn)
        
        # Print pool statistics
        stats = pool.get_stats()
        logger.info(f"Pool statistics: {json.dumps(stats, indent=2)}")


async def demonstrate_performance_optimization():
    """Demonstrate performance optimization techniques"""
    logger.info("Starting performance optimization demonstration")
    
    # Configuration for high performance
    config = ConnectionPoolConfig(
        max_connections=500,
        max_connections_per_host=50,
        connection_timeout=5.0,
        retry_attempts=2,
        enable_compression=True
    )
    
    async with HighThroughputScanner(config) as scanner:
        # Create large batch of targets
        targets = []
        base_hosts = ["example.com", "google.com", "github.com", "stackoverflow.com"]
        
        for host in base_hosts:
            for port in range(80, 90):  # Ports 80-89
                targets.append(ScanTarget(
                    host=host,
                    port=port,
                    protocol="http",
                    timeout=3.0,
                    retries=1
                ))
        
        logger.info(f"Scanning {len(targets)} targets")
        
        # Measure performance
        start_time = time.time()
        results = await scanner.scan_targets(targets)
        end_time = time.time()
        
        # Calculate statistics
        successful_scans = sum(1 for r in results if r.success)
        failed_scans = len(results) - successful_scans
        total_time = end_time - start_time
        scans_per_second = len(results) / total_time
        
        logger.info(f"Performance results:")
        logger.info(f"  Total targets: {len(targets)}")
        logger.info(f"  Successful scans: {successful_scans}")
        logger.info(f"  Failed scans: {failed_scans}")
        logger.info(f"  Total time: {total_time:.2f}s")
        logger.info(f"  Scans per second: {scans_per_second:.2f}")
        
        # Print scanner statistics
        stats = scanner.get_stats()
        logger.info(f"Scanner statistics: {json.dumps(stats, indent=2)}")


if __name__ == "__main__":
    # Run demonstrations
    async def main():
        
    """main function."""
try:
            await demonstrate_connection_pooling()
            await demonstrate_high_throughput_scanning()
            await demonstrate_performance_optimization()
        except KeyboardInterrupt:
            logger.info("Demonstration interrupted by user")
        except Exception as e:
            logger.error(f"Demonstration error: {e}")
    
    asyncio.run(main()) 