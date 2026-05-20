from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
BUFFER_SIZE = 1024

import asyncio
import time
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import socket
import logging
    import asyncssh
from ..core import BaseConfig, ScanResult, BaseScanner
from typing import Any, List, Dict, Optional
"""
SSH scanning and interaction utilities using asyncssh.
Async operations for SSH connections and authentication testing.
"""


# Optional import for SSH operations
try:
    ASYNCSSH_AVAILABLE = True
except ImportError:
    ASYNCSSH_AVAILABLE = False


@dataclass
class SSHScanConfig(BaseConfig):
    """Configuration for SSH scanning operations."""
    timeout: float = 10.0
    max_workers: int = 10
    retry_count: int = 2
    banner_grab: bool = True
    version_detection: bool = True
    auth_testing: bool = False
    common_users: List[str] = None
    common_passwords: List[str] = None
    
    def __post_init__(self) -> Any:
        if self.common_users is None:
            self.common_users = ["root", "admin", "user", "test", "guest"]
        if self.common_passwords is None:
            self.common_passwords = ["password", "123456", "admin", "root", "test"]

@dataclass
class SSHScanResult:
    """Result of an SSH scan operation."""
    target: str
    port: int = 22
    is_open: bool = False
    ssh_version: Optional[str] = None
    banner: Optional[str] = None
    key_exchange_algorithms: List[str] = None
    encryption_algorithms: List[str] = None
    mac_algorithms: List[str] = None
    compression_algorithms: List[str] = None
    host_key_algorithms: List[str] = None
    success: bool = False
    response_time: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self) -> Any:
        if self.key_exchange_algorithms is None:
            self.key_exchange_algorithms = []
        if self.encryption_algorithms is None:
            self.encryption_algorithms = []
        if self.mac_algorithms is None:
            self.mac_algorithms = []
        if self.compression_algorithms is None:
            self.compression_algorithms = []
        if self.host_key_algorithms is None:
            self.host_key_algorithms = []
        if self.metadata is None:
            self.metadata = {}

def validate_ssh_target(target: str) -> bool:
    """Validate SSH target format."""
    try:
        if ':' in target:
            host, port = target.rsplit(':', 1)
            port = int(port)
            if not (1 <= port <= 65535):
                return False
        else:
            host = target
        
        # Basic host validation
        socket.gethostbyname(host)
        return True
    except (socket.gaierror, ValueError):
        return False

def parse_ssh_target(target: str) -> Tuple[str, int]:
    """Parse SSH target into host and port."""
    if ':' in target:
        host, port = target.rsplit(':', 1)
        return host, int(port)
    return target, 22

async def check_ssh_port(host: str, port: int, config: SSHScanConfig) -> SSHScanResult:
    """Check if SSH port is open and grab banner."""
    start_time = time.time()
    
    # Guard clause for invalid inputs
    if not validate_ssh_target(host):
        return SSHScanResult(
            target=host, port=port, is_open=False,
            error_message="Invalid SSH target format"
        )
    
    try:
        # Basic port check
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(config.timeout)
        
        result = await asyncio.get_event_loop().run_in_executor(
            None, sock.connect_ex, (host, port)
        )
        
        is_open = result == 0
        response_time = time.time() - start_time
        
        if not is_open:
            sock.close()
            return SSHScanResult(
                target=host, port=port, is_open=False,
                response_time=response_time, success=True
            )
        
        # Grab SSH banner
        banner = None
        if config.banner_grab:
            try:
                sock.send(b"\r\n")
                banner_data = await asyncio.get_event_loop().run_in_executor(
                    None, sock.recv, 1024
                )
                banner = banner_data.decode('utf-8', errors='ignore').strip()
            except Exception:
                pass
        
        sock.close()
        
        return SSHScanResult(
            target=host,
            port=port,
            is_open=is_open,
            banner=banner,
            response_time=response_time,
            success=True
        )
        
    except Exception as e:
        return SSHScanResult(
            target=host, port=port, is_open=False,
            error_message=str(e), success=False
        )

async def get_ssh_info(host: str, port: int, config: SSHScanConfig) -> SSHScanResult:
    """Get detailed SSH information using asyncssh."""
    if not ASYNCSSH_AVAILABLE:
        return SSHScanResult(
            target=host, port=port, is_open=False,
            error_message="asyncssh not available"
        )
    
    start_time = time.time()
    
    try:
        # Connect to SSH server
        conn, client = await asyncio.wait_for(
            asyncssh.connect(host, port=port, known_hosts=None),
            timeout=config.timeout
        )
        
        response_time = time.time() - start_time
        
        # Get connection info
        info = {
            'key_exchange_algorithms': client.get_key_exchange_algorithms(),
            'encryption_algorithms': client.get_encryption_algorithms(),
            'mac_algorithms': client.get_mac_algorithms(),
            'compression_algorithms': client.get_compression_algorithms(),
            'host_key_algorithms': client.get_host_key_algorithms()
        }
        
        await conn.close()
        
        return SSHScanResult(
            target=host,
            port=port,
            is_open=True,
            ssh_version=client.get_version(),
            banner=client.get_banner(),
            key_exchange_algorithms=info['key_exchange_algorithms'],
            encryption_algorithms=info['encryption_algorithms'],
            mac_algorithms=info['mac_algorithms'],
            compression_algorithms=info['compression_algorithms'],
            host_key_algorithms=info['host_key_algorithms'],
            response_time=response_time,
            success=True
        )
        
    except asyncio.TimeoutError:
        return SSHScanResult(
            target=host, port=port, is_open=False,
            error_message="Connection timeout", success=False
        )
    except Exception as e:
        return SSHScanResult(
            target=host, port=port, is_open=False,
            error_message=str(e), success=False
        )

async def test_ssh_auth(host: str, port: int, username: str, password: str, 
                        config: SSHScanConfig) -> Dict[str, Any]:
    """Test SSH authentication with username/password."""
    if not ASYNCSSH_AVAILABLE:
        return {"success": False, "error": "asyncssh not available"}
    
    try:
        conn, client = await asyncio.wait_for(
            asyncssh.connect(host, port=port, username=username, 
                           password=password, known_hosts=None),
            timeout=config.timeout
        )
        
        await conn.close()
        return {"success": True, "authenticated": True}
        
    except asyncssh.AuthenticationError:
        return {"success": True, "authenticated": False}
    except Exception as e:
        return {"success": False, "error": str(e)}

async def scan_ssh_targets(targets: List[str], config: SSHScanConfig) -> List[SSHScanResult]:
    """Scan multiple SSH targets concurrently."""
    semaphore = asyncio.Semaphore(config.max_workers)
    
    async def scan_single_target(target: str) -> SSHScanResult:
        async with semaphore:
            host, port = parse_ssh_target(target)
            return await check_ssh_port(host, port, config)
    
    tasks = [scan_single_target(target) for target in targets]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter out exceptions
    valid_results = []
    for result in results:
        if isinstance(result, SSHScanResult):
            valid_results.append(result)
    
    return valid_results

async def brute_force_ssh(host: str, port: int, config: SSHScanConfig) -> Dict[str, Any]:
    """Attempt brute force authentication on SSH."""
    if not config.auth_testing:
        return {"error": "Authentication testing disabled"}
    
    results = {
        "host": host,
        "port": port,
        "attempts": 0,
        "successful": [],
        "failed": []
    }
    
    for username in config.common_users:
        for password in config.common_passwords:
            results["attempts"] += 1
            
            auth_result = await test_ssh_auth(host, port, username, password, config)
            
            if auth_result.get("authenticated", False):
                results["successful"].append({
                    "username": username,
                    "password": password
                })
            else:
                results["failed"].append({
                    "username": username,
                    "password": password
                })
    
    return results

class SSHScanner(BaseScanner):
    """SSH scanner with comprehensive capabilities."""
    
    def __init__(self, config: SSHScanConfig):
        
    """__init__ function."""
self.config = config
    
    async def comprehensive_scan(self, target: str) -> Dict[str, Any]:
        """Perform comprehensive SSH scan."""
        host, port = parse_ssh_target(target)
        
        results = {
            "target": target,
            "timestamp": time.time(),
            "port_check": None,
            "ssh_info": None,
            "auth_test": None
        }
        
        # Basic port check
        port_result = await check_ssh_port(host, port, self.config)
        results["port_check"] = port_result.__dict__
        
        if port_result.is_open:
            # Get detailed SSH information
            ssh_info = await get_ssh_info(host, port, self.config)
            results["ssh_info"] = ssh_info.__dict__
            
            # Test authentication if enabled
            if self.config.auth_testing:
                auth_results = await brute_force_ssh(host, port, self.config)
                results["auth_test"] = auth_results
        
        return results
    
    async def scan_multiple_targets(self, targets: List[str]) -> Dict[str, Any]:
        """Scan multiple SSH targets."""
        results = await scan_ssh_targets(targets, self.config)
        
        return {
            "targets": targets,
            "timestamp": time.time(),
            "results": [r.__dict__ for r in results],
            "summary": {
                "total": len(results),
                "open": len([r for r in results if r.is_open]),
                "closed": len([r for r in results if not r.is_open])
            }
        } 