from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import asyncio
import socket
import ssl
import time
import weakref
from typing import Dict, Any, Optional, List, Tuple, Set, Callable
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
import logging
from collections import defaultdict
import aiohttp
import httpx
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
Connection Pooling for High-Throughput Scanning
Implements asyncio-based connection pooling for efficient network operations.
"""


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ConnectionConfig:
    """Configuration for connection pooling."""
    max_connections: int = 100
    max_connections_per_host: int = 10
    connection_timeout: float = 10.0
    keepalive_timeout: float = 30.0
    retry_attempts: int = 3
    retry_delay: float = 1.0
    enable_ssl: bool = True
    verify_ssl: bool = True
    enable_compression: bool = True
    user_agent: str = "Cybersecurity Scanner/1.0"

@dataclass
class PoolStats:
    """Connection pool statistics."""
    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    failed_connections: int = 0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    last_used: Optional[float] = None

class ConnectionPool:
    """Asyncio-based connection pool for high-throughput scanning."""
    
    def __init__(self, config: ConnectionConfig):
        
    """__init__ function."""
self.config = config
        self._connections: Dict[str, List[asyncio.StreamWriter]] = defaultdict(list)
        self._connection_locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._stats: Dict[str, PoolStats] = defaultdict(PoolStats)
        self._cleanup_task: Optional[asyncio.Task] = None
        self._executor = ThreadPoolExecutor(max_workers=20)
        self._closed = False
        
        # HTTP clients for different protocols
        self._http_client: Optional[aiohttp.ClientSession] = None
        self._httpx_client: Optional[httpx.AsyncClient] = None
        
        # Start cleanup task
        self._start_cleanup_task()
    
    def _start_cleanup_task(self) -> Any:
        """Start the cleanup task for idle connections."""
        async def cleanup_loop():
            
    """cleanup_loop function."""
while not self._closed:
                try:
                    await self._cleanup_idle_connections()
                    await asyncio.sleep(30)  # Cleanup every 30 seconds
                except Exception as e:
                    logger.error(f"Cleanup task error: {e}")
        
        self._cleanup_task = asyncio.create_task(cleanup_loop())
    
    async def _cleanup_idle_connections(self) -> Any:
        """Clean up idle connections."""
        current_time = time.time()
        
        for host, connections in list(self._connections.items()):
            active_connections = []
            
            for conn in connections:
                if not conn.is_closing():
                    # Check if connection is idle
                    if hasattr(conn, '_last_used'):
                        if current_time - conn._last_used > self.config.keepalive_timeout:
                            await self._close_connection(conn)
                            self._stats[host].idle_connections -= 1
                        else:
                            active_connections.append(conn)
                    else:
                        active_connections.append(conn)
                else:
                    self._stats[host].active_connections -= 1
            
            self._connections[host] = active_connections
    
    async def _close_connection(self, conn: asyncio.StreamWriter):
        """Close a connection safely."""
        try:
            if not conn.is_closing():
                conn.close()
                await conn.wait_closed()
        except Exception as e:
            logger.debug(f"Error closing connection: {e}")
    
    async def get_connection(self, host: str, port: int, protocol: str = "tcp") -> asyncio.StreamWriter:
        """Get a connection from the pool or create a new one."""
        key = f"{host}:{port}:{protocol}"
        
        async with self._connection_locks[key]:
            # Check for available connections
            if self._connections[key]:
                conn = self._connections[key].pop()
                if not conn.is_closing():
                    conn._last_used = time.time()
                    self._stats[key].active_connections += 1
                    self._stats[key].idle_connections -= 1
                    return conn
                else:
                    self._stats[key].active_connections -= 1
            
            # Create new connection if pool not full
            if len(self._connections[key]) < self.config.max_connections_per_host:
                try:
                    conn = await self._create_connection(host, port, protocol)
                    conn._last_used = time.time()
                    self._stats[key].total_connections += 1
                    self._stats[key].active_connections += 1
                    return conn
                except Exception as e:
                    self._stats[key].failed_connections += 1
                    raise ConnectionError(f"Failed to create connection to {host}:{port}: {e}")
            else:
                # Wait for a connection to become available
                while not self._connections[key]:
                    await asyncio.sleep(0.1)
                
                conn = self._connections[key].pop()
                conn._last_used = time.time()
                self._stats[key].active_connections += 1
                self._stats[key].idle_connections -= 1
                return conn
    
    async def _create_connection(self, host: str, port: int, protocol: str) -> asyncio.StreamWriter:
        """Create a new connection."""
        if protocol == "tcp":
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=self.config.connection_timeout
            )
            return writer
        elif protocol == "ssl":
            ssl_context = ssl.create_default_context()
            if not self.config.verify_ssl:
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
            
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port, ssl=ssl_context),
                timeout=self.config.connection_timeout
            )
            return writer
        else:
            raise ValueError(f"Unsupported protocol: {protocol}")
    
    async def return_connection(self, host: str, port: int, protocol: str, conn: asyncio.StreamWriter):
        """Return a connection to the pool."""
        key = f"{host}:{port}:{protocol}"
        
        async with self._connection_locks[key]:
            if not conn.is_closing() and len(self._connections[key]) < self.config.max_connections_per_host:
                self._connections[key].append(conn)
                self._stats[key].active_connections -= 1
                self._stats[key].idle_connections += 1
            else:
                await self._close_connection(conn)
                self._stats[key].active_connections -= 1
    
    async def close_connection(self, host: str, port: int, protocol: str, conn: asyncio.StreamWriter):
        """Close a connection and remove it from the pool."""
        key = f"{host}:{port}:{protocol}"
        
        async with self._connection_locks[key]:
            await self._close_connection(conn)
            self._stats[key].active_connections -= 1
    
    @asynccontextmanager
    async def get_connection_context(self, host: str, port: int, protocol: str = "tcp"):
        """Context manager for connection usage."""
        conn = await self.get_connection(host, port, protocol)
        try:
            yield conn
        except Exception:
            await self.close_connection(host, port, protocol, conn)
            raise
        else:
            await self.return_connection(host, port, protocol, conn)
    
    async def execute_with_retry(self, operation: Callable, *args, **kwargs):
        """Execute an operation with retry logic."""
        last_exception = None
        
        for attempt in range(self.config.retry_attempts):
            try:
                return await operation(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < self.config.retry_attempts - 1:
                    await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
        
        raise last_exception
    
    async async def get_http_client(self) -> aiohttp.ClientSession:
        """Get or create HTTP client session."""
        if self._http_client is None or self._http_client.closed:
            connector = aiohttp.TCPConnector(
                limit=self.config.max_connections,
                limit_per_host=self.config.max_connections_per_host,
                keepalive_timeout=self.config.keepalive_timeout,
                enable_cleanup_closed=True,
                ssl=self.config.verify_ssl
            )
            
            timeout = aiohttp.ClientTimeout(total=self.config.connection_timeout)
            
            self._http_client = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={"User-Agent": self.config.user_agent}
            )
        
        return self._http_client
    
    async async def get_httpx_client(self) -> httpx.AsyncClient:
        """Get or create HTTPX client."""
        if self._httpx_client is None:
            limits = httpx.Limits(
                max_connections=self.config.max_connections,
                max_keepalive_connections=self.config.max_connections_per_host
            )
            
            self._httpx_client = httpx.AsyncClient(
                limits=limits,
                timeout=self.config.connection_timeout,
                verify=self.config.verify_ssl,
                headers={"User-Agent": self.config.user_agent}
            )
        
        return self._httpx_client
    
    def get_stats(self, host: Optional[str] = None) -> Dict[str, Any]:
        """Get connection pool statistics."""
        if host:
            return asdict(self._stats.get(host, PoolStats()))
        
        total_stats = PoolStats()
        for stats in self._stats.values():
            total_stats.total_connections += stats.total_connections
            total_stats.active_connections += stats.active_connections
            total_stats.idle_connections += stats.idle_connections
            total_stats.failed_connections += stats.failed_connections
            total_stats.total_requests += stats.total_requests
            total_stats.successful_requests += stats.successful_requests
            total_stats.failed_requests += stats.failed_requests
        
        return asdict(total_stats)
    
    async def close(self) -> Any:
        """Close the connection pool and cleanup resources."""
        self._closed = True
        
        # Cancel cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Close all connections
        for host, connections in self._connections.items():
            for conn in connections:
                await self._close_connection(conn)
            self._connections[host].clear()
        
        # Close HTTP clients
        if self._http_client and not self._http_client.closed:
            await self._http_client.close()
        
        if self._httpx_client:
            await self._httpx_client.aclose()
        
        # Shutdown executor
        self._executor.shutdown(wait=True)

class HighThroughputScanner:
    """High-throughput scanner using connection pooling."""
    
    def __init__(self, pool_config: ConnectionConfig):
        
    """__init__ function."""
self.pool = ConnectionPool(pool_config)
        self.semaphore = asyncio.Semaphore(pool_config.max_connections)
        self._scan_tasks: Set[asyncio.Task] = set()
    
    async def scan_port_range(self, target: str, ports: List[int], 
                             scan_type: str = "tcp") -> List[Dict[str, Any]]:
        """Scan a range of ports with high throughput."""
        tasks = []
        
        for port in ports:
            task = asyncio.create_task(
                self._scan_single_port(target, port, scan_type)
            )
            self._scan_tasks.add(task)
            task.add_done_callback(self._scan_tasks.discard)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and return valid results
        valid_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Scan error: {result}")
            else:
                valid_results.append(result)
        
        return valid_results
    
    async def _scan_single_port(self, target: str, port: int, scan_type: str) -> Dict[str, Any]:
        """Scan a single port."""
        async with self.semaphore:
            start_time = time.time()
            
            try:
                async with self.pool.get_connection_context(target, port, scan_type) as conn:
                    # Perform port scan
                    result = await self._perform_port_scan(conn, target, port, scan_type)
                    
                    duration = time.time() - start_time
                    result.update({
                        "duration": duration,
                        "success": True,
                        "target": target,
                        "port": port,
                        "scan_type": scan_type
                    })
                    
                    # Update stats
                    key = f"{target}:{port}:{scan_type}"
                    self.pool._stats[key].total_requests += 1
                    self.pool._stats[key].successful_requests += 1
                    self.pool._stats[key].avg_response_time = (
                        (self.pool._stats[key].avg_response_time * 
                         (self.pool._stats[key].successful_requests - 1) + duration) /
                        self.pool._stats[key].successful_requests
                    )
                    
                    return result
                    
            except Exception as e:
                duration = time.time() - start_time
                
                # Update stats
                key = f"{target}:{port}:{scan_type}"
                self.pool._stats[key].total_requests += 1
                self.pool._stats[key].failed_requests += 1
                
                return {
                    "target": target,
                    "port": port,
                    "scan_type": scan_type,
                    "success": False,
                    "error": str(e),
                    "duration": duration
                }
    
    async def _perform_port_scan(self, conn: asyncio.StreamWriter, target: str, port: int, scan_type: str) -> Dict[str, Any]:
        """Perform the actual port scan."""
        # Send a simple probe
        try:
            conn.write(b"GET / HTTP/1.1\r\nHost: " + target.encode() + b"\r\n\r\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            await conn.drain()
            
            # For demo purposes, assume port is open if we can write
            return {
                "is_open": True,
                "service": "unknown",
                "banner": None
            }
            
        except Exception as e:
            return {
                "is_open": False,
                "service": "unknown",
                "banner": None,
                "error": str(e)
            }
    
    async def scan_multiple_targets(self, targets: List[str], ports: List[int], 
                                   scan_type: str = "tcp") -> Dict[str, List[Dict[str, Any]]]:
        """Scan multiple targets with high throughput."""
        all_results = {}
        
        for target in targets:
            results = await self.scan_port_range(target, ports, scan_type)
            all_results[target] = results
        
        return all_results
    
    async def enumerate_services(self, target: str, common_ports: List[int] = None) -> Dict[str, Any]:
        """Enumerate services on a target."""
        if common_ports is None:
            common_ports = [21, 22, 23, 25, 53, 80, 110, 143, 443, 993, 995, 3306, 5432, 8080]
        
        results = await self.scan_port_range(target, common_ports, "tcp")
        
        open_ports = [r for r in results if r.get("is_open", False)]
        closed_ports = [r for r in results if not r.get("is_open", False)]
        
        return {
            "target": target,
            "total_ports_scanned": len(results),
            "open_ports": len(open_ports),
            "closed_ports": len(closed_ports),
            "open_ports_details": open_ports,
            "scan_duration": sum(r.get("duration", 0) for r in results),
            "success_rate": len([r for r in results if r.get("success", False)]) / len(results)
        }
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        return self.pool.get_stats()
    
    async def close(self) -> Any:
        """Close the scanner and connection pool."""
        # Wait for all scan tasks to complete
        if self._scan_tasks:
            await asyncio.gather(*self._scan_tasks, return_exceptions=True)
        
        await self.pool.close()

# Global connection pool instance
_global_pool: Optional[ConnectionPool] = None
_global_scanner: Optional[HighThroughputScanner] = None

def get_connection_pool(config: ConnectionConfig = None) -> ConnectionPool:
    """Get or create global connection pool."""
    global _global_pool
    
    if _global_pool is None:
        if config is None:
            config = ConnectionConfig()
        _global_pool = ConnectionPool(config)
    
    return _global_pool

def get_high_throughput_scanner(config: ConnectionConfig = None) -> HighThroughputScanner:
    """Get or create global high-throughput scanner."""
    global _global_scanner
    
    if _global_scanner is None:
        if config is None:
            config = ConnectionConfig()
        _global_scanner = HighThroughputScanner(config)
    
    return _global_scanner

async def cleanup_global_resources():
    """Cleanup global resources."""
    global _global_scanner, _global_pool
    
    if _global_scanner:
        await _global_scanner.close()
        _global_scanner = None
    
    if _global_pool:
        await _global_pool.close()
        _global_pool = None 