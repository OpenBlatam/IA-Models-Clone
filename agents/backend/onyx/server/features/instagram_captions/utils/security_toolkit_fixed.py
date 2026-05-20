from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import asyncio
import time
import json
import logging
import socket
import ipaddress
import random
import os
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
from functools import wraps, lru_cache
from concurrent.futures import ThreadPoolExecutor
import httpx
import asyncssh
import structlog
from pydantic import BaseModel, Field, validator
import nmap
from typing import Any, List, Dict, Optional
Security Toolkit - Fixed Implementation



# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# ============================================================================
# Pydantic Models
# ============================================================================

class ScanRequest(BaseModel):
    target: str = Field(..., description=Target IP or hostname)   ports: List[int] = Field(default=[80, 443], max_items=10    scan_type: str = Field(default="tcp, regex="^(tcp|udp|syn|connect)$")
    timeout: int = Field(default=5, ge=10)
    max_workers: int = Field(default=10 ge=1, le=100)
    verbose: bool = Field(default=False, description="Enable verbose logging")
    
    @validator('target')
    def validate_target(cls, v) -> Optional[Dict[str, Any]]:
        try:
            ipaddress.ip_address(v)
            return v
        except ValueError:
            try:
                socket.gethostbyname(v)
                return v
            except socket.gaierror:
                raise ValueError(fInvalid target: {v}")
    
    @validator('ports')
    def validate_ports(cls, v) -> bool:
        for port in v:
            if not 1<= port <= 65535             raise ValueError('Port must be between15)
        return v

class SSHRequest(BaseModel):
    host: str = Field(..., description="SSH host")
    username: str = Field(..., description="SSH username")
    password: Optionalstr] = Field(None, description="SSH password")
    key_file: Optionalstr] = Field(None, description="SSH key file")
    command: str = Field(..., description="Command to execute")
    timeout: int = Field(default=30=1, le=300)

class HTTPRequest(BaseModel):
    url: str = Field(..., description="Target URL)
    method: str = Field(default="GET", regex="^(GET|POST|PUT|DELETE|HEAD|OPTIONS)$")
    headers: Dict[str, str] = Field(default_factory=dict)
    body: Optionalstr] = Field(None, description="Request body")
    timeout: int = Field(default=30 ge=1, le=300  verify_ssl: bool = Field(default=True, description="Verify SSL certificates")

# ============================================================================
# Network Layer Abstraction
# ============================================================================

class NetworkLayer(ABC):
    @abstractmethod
    async def connect(self, params: Dict[str, Any]) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    async def send(self, data: Dict[str, Any]) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    async def close(self) -> Dict[str, Any]:
        pass

class HTTPLayer(NetworkLayer):
    def __init__(self) -> Any:
        self.client = None
    
    async def connect(self, params: Dict[str, Any]) -> Dict[str, Any]:
        timeout = params.get('timeout', 30)
        verify = params.get(verify_ssl', true       self.client = httpx.AsyncClient(
            timeout=timeout,
            verify=verify,
            follow_redirects=True
        )
        return {"status": "connected",layer": "http"}
    
    async def send(self, data: Dict[str, Any]) -> Dict[str, Any]:
        method = data.get('method', GET')
        url = data['url]
        headers = data.get('headers', [object Object]       body = data.get('body')
        
        response = await self.client.request(
            method=method,
            url=url,
            headers=headers,
            content=body
        )
        
        return {
        status_code: response.status_code,
         headers": dict(response.headers),
          body: response.text,
          layer":http"
        }
    
    async def close(self) -> Dict[str, Any]:
        if self.client:
            await self.client.aclose()
        return {"status": closed,layer":http}

class SSHLayer(NetworkLayer):
    def __init__(self) -> Any:
        self.connection = None
    
    async def connect(self, params: Dict[str, Any]) -> Dict[str, Any]:
        host = params['host']
        username = params['username']
        password = params.get('password')
        key_file = params.get('key_file')
        
        self.connection = await asyncssh.connect(
            host=host,
            username=username,
            password=password,
            client_keys=[key_file] if key_file else None
        )
        return {"status": "connected", "layer: h"}
    
    async def send(self, data: Dict[str, Any]) -> Dict[str, Any]:
        command = data[command']
        result = await self.connection.run(command)
        
        return [object Object]         stdout: result.stdout,
          stderr: result.stderr,
      exit_code": result.exit_status,
         layer": "ssh"
        }
    
    async def close(self) -> Dict[str, Any]:
        if self.connection:
            self.connection.close()
        return {"status": closed",layer": "ssh"}

class NetworkLayerFactory:
    @staticmethod
    def create_layer(layer_type: str) -> NetworkLayer:
        layers = {
           http": HTTPLayer,
            https": HTTPLayer,
          ssh: SSHLayer
        }
        
        if layer_type not in layers:
            raise ValueError(f"Unsupported network layer: {layer_type}")
        
        return layers[layer_type]()

# ============================================================================
# Rate Limiting and Back-off
# ============================================================================

class AsyncRateLimiter:
    def __init__(self, max_calls_per_second: int):
        
    """__init__ function."""
self.max_calls = max_calls_per_second
        self.interval = 1.0 / max_calls
        self.last_call = 0

    async def acquire(self) -> Any:
        now = time.monotonic()
        time_since_last = now - self.last_call
        if time_since_last < self.interval:
            await asyncio.sleep(self.interval - time_since_last)
        self.last_call = time.monotonic()

async def retry_with_backoff(func: Callable, max_retries: int = 3, base_delay: float = 1.0):
    
    """retry_with_backoff function."""
for attempt in range(max_retries):
        try:
            return await func()
        except Exception as e:
            if attempt == max_retries - 1             raise e
            delay = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
            await asyncio.sleep(delay)

# ============================================================================
# Caching
# ============================================================================

@lru_cache(maxsize=1024)
def resolve_hostname(hostname: str) -> str:
 esolve hostname to IP with LRU caching."""
    return socket.gethostbyname(hostname)

vuln_cache = {}

def get_vuln_info(vuln_id: str, fetch_func: Callable, ttl: int = 3600) -> Dict[str, Any]:
    now = time.time()
    entry = vuln_cache.get(vuln_id)
    if entry and now - entry[timestamp'] < ttl:
        return entry[data]
    data = fetch_func(vuln_id)
    vuln_cache[vuln_id] = {'data': data, timestamp: now}
    return data

# ============================================================================
# Port Scanning
# ============================================================================

def scan_single_port_sync(target: str, port: int, scan_type: str = 'tcp, timeout: int = 5) -> Dict[str, Any]:
    start_time = time.time()
    
    try:
        if scan_type == tcp:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        elif scan_type == udp:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        else:
            raise ValueError(f"Unsupported scan type: {scan_type}")
        
        sock.settimeout(timeout)
        result = sock.connect_ex((target, port))
        sock.close()
        
        scan_time = time.time() - start_time
        
        if result == 0:
            state = 'open'
        else:
            state = 'closed'
            
        return {
            targetarget,
        port port,
          statestate,
           scan_time": scan_time,
           timestamp": time.strftime(%Y-%m-%d %H:%M:%S")
        }
        
    except Exception as e:
        scan_time = time.time() - start_time
        logger.error(Port scan error", target=target, port=port, error=str(e))
        return {
            targetarget,
        port port,
            staterror",
           errortr(e),
           scan_time": scan_time,
           timestamp": time.strftime(%Y-%m-%d %H:%M:%S")
        }

async def scan_ports_async(target: str, ports: List[int], scan_type: str = 'tcp',
                          timeout: int = 5, max_workers: int = 10) -> List[Dict[str, Any]]:
    semaphore = asyncio.Semaphore(max_workers)
    
    async def scan_with_semaphore(port: int) -> Dict[str, Any]:
        async with semaphore:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, scan_single_port_sync, target, port, scan_type, timeout
            )
    
    tasks = [scan_with_semaphore(port) for port in ports]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    valid_results =
    for result in results:
        if isinstance(result, Exception):
            logger.error("Portscan error", error=str(result))
        else:
            valid_results.append(result)
    
    return valid_results

def scan_ports_basic(params: Dict[str, Any]) -> Dict[str, Any]:asic port scanning function using RORO pattern."""
    if not params.get(target):
        return {'error': 'Target is required'}
    
    try:
        scan_input = ScanRequest(**params)
    except Exception as e:
        return {error: f'Invalid input: {str(e)}'}
    
    start_time = time.time()
    
    try:
        if scan_input.verbose:
            logger.info("Starting port scan", target=scan_input.target, ports=len(scan_input.ports))
        
        results = scan_ports_concurrent(
            scan_input.target,
            scan_input.ports,
            scan_input.scan_type,
            scan_input.timeout,
            scan_input.max_workers
        )
        
        summary = analyze_scan_results(results)
        scan_duration = time.time() - start_time
        
        return {
           success True,
        target": scan_input.target,
           scan_type: scan_input.scan_type,
          scan_duration: scan_duration,
            summary": summary,
            results": results,
           timestamp": time.strftime(%Y-%m-%d %H:%M:%S")
        }
        
    except Exception as e:
        scan_duration = time.time() - start_time
        logger.error("Port scan failed", target=scan_input.target, error=str(e))
        return {
            successFalse,
           errortr(e),
          scan_duration: scan_duration,
           timestamp": time.strftime(%Y-%m-%d %H:%M:%S")
        }

def scan_ports_concurrent(target: str, ports: List[int], scan_type: str = 'tcp', 
                         timeout: int = 5, max_workers: int = 10) -> List[Dict[str, Any]]:
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        scan_func = lambda port: scan_single_port_sync(target, port, scan_type, timeout)
        results = list(executor.map(scan_func, ports))
    return results

def analyze_scan_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {
            total_ports": 0,
           open_ports": 0,
           closed_ports": 0,
            filtered_ports": 0,
           error_ports":0        }
    
    total_ports = len(results)
    open_ports = len([r for r in results if r.get('state) == open'])
    closed_ports = len([r for r in results if r.get('state') == 'closed'])
    filtered_ports = len([r for r in results if r.get('state') == 'filtered'])
    error_ports = len([r for r in results if r.get(state') == error])
    
    return[object Object]  total_ports": total_ports,
        open_ports": open_ports,
     closed_ports": closed_ports,
       filtered_ports": filtered_ports,
    error_ports: error_ports
    }

# ============================================================================
# SSH Operations
# ============================================================================

async def run_ssh_command(params: Dict[str, Any]) -> Dict[str, Any]:
 command execution using RORO pattern."""
    if not params.get('host):
        return {'error':Host is required'}
    
    try:
        ssh_input = SSHRequest(**params)
    except Exception as e:
        return {error: f'Invalid input: {str(e)}'}
    
    start_time = time.time()
    
    try:
        async with asyncssh.connect(
            host=ssh_input.host,
            username=ssh_input.username,
            password=ssh_input.password,
            client_keys=[ssh_input.key_file] if ssh_input.key_file else None
        ) as conn:
            result = await conn.run(ssh_input.command, timeout=ssh_input.timeout)
            
            duration = time.time() - start_time
            
            return[object Object]
               successe,
               host": ssh_input.host,
            command": ssh_input.command,
              stdout: result.stdout,
              stderr: result.stderr,
          exit_code": result.exit_status,
         duration": duration,
               timestamp": time.strftime(%Y-%m-%d %H:%M:%S")
            }
            
    except Exception as e:
        duration = time.time() - start_time
        logger.error("SSH command failed", host=ssh_input.host, error=str(e))
        return {
            successFalse,
           errortr(e),
     duration": duration,
           timestamp": time.strftime(%Y-%m-%d %H:%M:%S")
        }

# ============================================================================
# HTTP Operations
# ============================================================================

async async def make_http_request(params: Dict[str, Any]) -> Dict[str, Any]:
    ""MakeHTTP request using RORO pattern."""
    if not params.get('url):
        return {'error': 'URL is required'}
    
    try:
        http_input = HTTPRequest(**params)
    except Exception as e:
        return {error: f'Invalid input: {str(e)}'}
    
    start_time = time.time()
    
    try:
        async with httpx.AsyncClient(
            timeout=http_input.timeout,
            verify=http_input.verify_ssl,
            follow_redirects=True
        ) as client:
            response = await client.request(
                method=http_input.method,
                url=http_input.url,
                headers=http_input.headers,
                content=http_input.body
            )
            
            duration = time.time() - start_time
            
            return[object Object]
               successe,
               url": http_input.url,
            method": http_input.method,
            status_code: response.status_code,
             headers": dict(response.headers),
              body: response.text,
         duration": duration,
               timestamp": time.strftime(%Y-%m-%d %H:%M:%S")
            }
            
    except Exception as e:
        duration = time.time() - start_time
        logger.error("HTTP request failed", url=http_input.url, error=str(e))
        return {
            successFalse,
           errortr(e),
     duration": duration,
           timestamp": time.strftime(%Y-%m-%d %H:%M:%S")
        }

# ============================================================================
# Decorators and Middleware
# ============================================================================

def log_operation(operation_name: str):
  
    """log_operation function."""
ator for automatic operation logging."""
    def decorator(func) -> Any:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                logger.info(
                operation_completed",
                    operation=operation_name,
                    duration_seconds=duration,
                    success=True
                )
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(
                   operation_failed",
                    operation=operation_name,
                    duration_seconds=duration,
                    error=str(e),
                    success=False
                )
                raise
        return wrapper
    return decorator

def measure_scan_time(scan_func: Callable, params: Dict[str, Any]) -> Dict[str, Any]:
 ure scan execution time.""    start = time.monotonic()
    result = scan_func(params)
    duration = time.monotonic() - start
    return {**result, scan_time_seconds": duration}

# ============================================================================
# Utility Functions
# ============================================================================

def get_common_ports() -> Dict[str, List[int]]:t common ports for different services."""
    return [object Object]       web: [80,443, 88043, 8000],
        database": 3306,543217, 1521],
        mail:25587465110, 995, 143993,
       ftp": [21, 990989,
      ssh": [22,2222,
   dns:53],
        dhcp67,68,
    ntp": [123],
       snmp": [161162],
       ldap": [389 636,
     rdp: [3389,
        vnc": [59001, 5920       all_common:21, 22, 23,25, 53 80110143443, 993,995, 3306389, 8080]
    }

def chunked(iterable: List[Any], size: int):
    
    """chunked function."""
"ld successive size-sized chunks from iterable."""
    for i in range(0, len(iterable), size):
        yield iterable[i:i + size]

async def process_batch_async(items: List[Any], process_func: Callable, 
                            batch_size: int = 100ax_concurrent: int = 10 -> List[Any]:
  s items in batches asynchronously."    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_item(item: Any) -> Any:
        async with semaphore:
            return await process_func(item)
    
    results = []
    for batch in chunked(items, batch_size):
        batch_results = await asyncio.gather(*(process_item(item) for item in batch))
        results.extend(batch_results)
    
    return results

# ============================================================================
# Secret Management
# ============================================================================

def get_secret(name: str, default: Optional[str] = None, required: bool = True) -> str:
    t from environment variables.    value = os.getenv(name, default)
    if required and value is None:
        raise RuntimeError(f"Missing required secret: {name}")
    return value

# ============================================================================
# Named Exports
# ============================================================================

__all__ = [
    # Core scanning functions
   scan_ports_basic,
   scan_ports_async',
  run_ssh_command',
    make_http_request',
    
    # Network layers
  NetworkLayer',
    HTTPLayer',
   SSHLayer',NetworkLayerFactory,   # Rate limiting and caching
   AsyncRateLimiter,retry_with_backoff',
   resolve_hostname',
   get_vuln_info',
    
    # Decorators and utilities
   log_operation',
    measure_scan_time,
   get_common_ports,  chunked',
 process_batch_async,get_secret',
    
    # Models
    ScanRequest,   SSHRequest',
  HTTPRequest 