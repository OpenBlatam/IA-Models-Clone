from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import asyncio
import socket
import ssl
import time
import aiohttp
import httpx
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import logging
import struct
import json
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
Async Helpers for Cybersecurity Scanning
Extracts heavy I/O operations from core scanning loops to dedicated async helpers.
"""


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AsyncHelperConfig:
    """Configuration for async helpers."""
    timeout: float = 10.0
    max_workers: int = 50
    retry_attempts: int = 3
    retry_delay: float = 1.0
    chunk_size: int = 1024
    max_connections: int = 100
    enable_ssl: bool = True
    verify_ssl: bool = True

class NetworkIOHelper:
    """Dedicated async helper for network I/O operations."""
    
    def __init__(self, config: AsyncHelperConfig):
        
    """__init__ function."""
self.config = config
        self._executor = ThreadPoolExecutor(max_workers=config.max_workers)
        self._session_pool: Dict[str, aiohttp.ClientSession] = {}
        self._httpx_pool: Dict[str, httpx.AsyncClient] = {}
    
    async def __aenter__(self) -> Any:
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> Any:
        await self.close()
    
    async def close(self) -> Any:
        """Close all sessions and cleanup resources."""
        for session in self._session_pool.values():
            if not session.closed:
                await session.close()
        self._session_pool.clear()
        
        for client in self._httpx_pool.values():
            await client.aclose()
        self._httpx_pool.clear()
        
        self._executor.shutdown(wait=True)
    
    async def tcp_connect(self, host: str, port: int, timeout: float = None) -> Tuple[bool, float, Optional[str]]:
        """Async TCP connection with timeout and error handling."""
        start_time = time.time()
        timeout = timeout or self.config.timeout
        
        try:
            # Use executor for blocking socket operations
            result = await asyncio.get_event_loop().run_in_executor(
                self._executor,
                self._blocking_tcp_connect,
                host, port, timeout
            )
            
            duration = time.time() - start_time
            return result[0], duration, result[1]
            
        except Exception as e:
            duration = time.time() - start_time
            return False, duration, str(e)
    
    def _blocking_tcp_connect(self, host: str, port: int, timeout: float) -> Tuple[bool, Optional[str]]:
        """Blocking TCP connection (runs in executor)."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        
        try:
            sock.connect((host, port))
            return True, None
        except Exception as e:
            return False, str(e)
        finally:
            sock.close()
    
    async def ssl_connect(self, host: str, port: int, timeout: float = None) -> Tuple[bool, float, Optional[Dict]]:
        """Async SSL connection with certificate info."""
        start_time = time.time()
        timeout = timeout or self.config.timeout
        
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                self._executor,
                self._blocking_ssl_connect,
                host, port, timeout
            )
            
            duration = time.time() - start_time
            return result[0], duration, result[1]
            
        except Exception as e:
            duration = time.time() - start_time
            return False, duration, {"error": str(e)}
    
    def _blocking_ssl_connect(self, host: str, port: int, timeout: float) -> Tuple[bool, Optional[Dict]]:
        """Blocking SSL connection (runs in executor)."""
        try:
            context = ssl.create_default_context()
            if not self.config.verify_ssl:
                context.check_hostname = False
                context.verify_mode = ssl.CERT_NONE
            
            with socket.create_connection((host, port), timeout=timeout) as sock:
                with context.wrap_socket(sock, server_hostname=host) as ssock:
                    cert = ssock.getpeercert()
                    return True, {
                        "subject": dict(x[0] for x in cert['subject']),
                        "issuer": dict(x[0] for x in cert['issuer']),
                        "version": cert['version'],
                        "serial_number": cert['serialNumber'],
                        "not_before": cert['notBefore'],
                        "not_after": cert['notAfter']
                    }
        except Exception as e:
            return False, {"error": str(e)}
    
    async def banner_grab(self, host: str, port: int, timeout: float = None) -> Tuple[bool, str, float]:
        """Async banner grabbing with timeout."""
        start_time = time.time()
        timeout = timeout or self.config.timeout
        
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                self._executor,
                self._blocking_banner_grab,
                host, port, timeout
            )
            
            duration = time.time() - start_time
            return result[0], result[1], duration
            
        except Exception as e:
            duration = time.time() - start_time
            return False, str(e), duration
    
    def _blocking_banner_grab(self, host: str, port: int, timeout: float) -> Tuple[bool, str]:
        """Blocking banner grab (runs in executor)."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        
        try:
            sock.connect((host, port))
            sock.send(b"HEAD / HTTP/1.0\r\n\r\n")
            banner = sock.recv(1024)
            return True, banner.decode('utf-8', errors='ignore').strip()
        except Exception as e:
            return False, str(e)
        finally:
            sock.close()
    
    async async def http_request(self, url: str, method: str = "GET", headers: Dict = None, 
                          timeout: float = None) -> Tuple[int, Dict, str, float]:
        """Async HTTP request with session pooling."""
        start_time = time.time()
        timeout = timeout or self.config.timeout
        
        # Get or create session for the domain
        domain = url.split('/')[2] if '://' in url else url.split('/')[0]
        
        if domain not in self._session_pool or self._session_pool[domain].closed:
            connector = aiohttp.TCPConnector(
                limit=self.config.max_connections,
                limit_per_host=10,
                ssl=self.config.verify_ssl
            )
            timeout_obj = aiohttp.ClientTimeout(total=timeout)
            self._session_pool[domain] = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout_obj
            )
        
        session = self._session_pool[domain]
        
        try:
            async with session.request(method, url, headers=headers) as response:
                content = await response.text()
                duration = time.time() - start_time
                return response.status, dict(response.headers), content, duration
                
        except Exception as e:
            duration = time.time() - start_time
            return 0, {}, str(e), duration
    
    async async def httpx_request(self, url: str, method: str = "GET", headers: Dict = None,
                           timeout: float = None) -> Tuple[int, Dict, str, float]:
        """Async HTTP request using httpx with session pooling."""
        start_time = time.time()
        timeout = timeout or self.config.timeout
        
        # Get or create httpx client for the domain
        domain = url.split('/')[2] if '://' in url else url.split('/')[0]
        
        if domain not in self._httpx_pool:
            limits = httpx.Limits(max_connections=self.config.max_connections)
            self._httpx_pool[domain] = httpx.AsyncClient(
                limits=limits,
                timeout=timeout,
                verify=self.config.verify_ssl
            )
        
        client = self._httpx_pool[domain]
        
        try:
            response = await client.request(method, url, headers=headers)
            content = response.text
            duration = time.time() - start_time
            return response.status_code, dict(response.headers), content, duration
            
        except Exception as e:
            duration = time.time() - start_time
            return 0, {}, str(e), duration

class DataProcessingHelper:
    """Dedicated async helper for data processing operations."""
    
    def __init__(self, config: AsyncHelperConfig):
        
    """__init__ function."""
self.config = config
        self._executor = ThreadPoolExecutor(max_workers=config.max_workers)
    
    async def __aenter__(self) -> Any:
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> Any:
        await self.close()
    
    async def close(self) -> Any:
        """Cleanup resources."""
        self._executor.shutdown(wait=True)
    
    async def analyze_scan_data(self, scan_results: List[Dict]) -> Dict[str, Any]:
        """Async analysis of scan results."""
        return await asyncio.get_event_loop().run_in_executor(
            self._executor,
            self._blocking_analyze_scan_data,
            scan_results
        )
    
    def _blocking_analyze_scan_data(self, scan_results: List[Dict]) -> Dict[str, Any]:
        """Blocking scan data analysis (runs in executor)."""
        if not scan_results:
            return {"error": "No scan results to analyze"}
        
        total_scans = len(scan_results)
        successful_scans = len([r for r in scan_results if r.get("success", False)])
        open_ports = len([r for r in scan_results if r.get("is_open", False)])
        
        # Calculate statistics
        response_times = [r.get("response_time", 0) for r in scan_results if r.get("response_time")]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        # Group by service
        services = {}
        for result in scan_results:
            service = result.get("service_name", "unknown")
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
    
    async def process_large_dataset(self, data: List[Any], chunk_size: int = None) -> List[Dict]:
        """Process large datasets in chunks to avoid blocking."""
        chunk_size = chunk_size or self.config.chunk_size
        results = []
        
        # Process data in chunks
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i + chunk_size]
            
            # Process chunk in executor
            chunk_result = await asyncio.get_event_loop().run_in_executor(
                self._executor,
                self._blocking_process_chunk,
                chunk
            )
            
            results.extend(chunk_result)
            
            # Yield control to event loop
            await asyncio.sleep(0)
        
        return results
    
    def _blocking_process_chunk(self, chunk: List[Any]) -> List[Dict]:
        """Blocking chunk processing (runs in executor)."""
        results = []
        for item in chunk:
            # Process each item (CPU-bound operation)
            processed = self._process_single_item(item)
            results.append(processed)
        return results
    
    def _process_single_item(self, item: Any) -> Dict:
        """Process a single item (CPU-bound)."""
        # Example processing logic
        if isinstance(item, dict):
            return {
                "processed": True,
                "data": item,
                "checksum": self._calculate_checksum(str(item))
            }
        else:
            return {
                "processed": True,
                "data": str(item),
                "checksum": self._calculate_checksum(str(item))
            }
    
    def _calculate_checksum(self, data: str) -> int:
        """Calculate simple checksum for data validation."""
        checksum = 0
        for char in data:
            checksum = (checksum + ord(char)) & 0xFFFF
        return checksum
    
    async def validate_data_integrity(self, data: List[Dict]) -> Dict[str, Any]:
        """Async data integrity validation."""
        return await asyncio.get_event_loop().run_in_executor(
            self._executor,
            self._blocking_validate_data_integrity,
            data
        )
    
    def _blocking_validate_data_integrity(self, data: List[Dict]) -> Dict[str, Any]:
        """Blocking data integrity validation (runs in executor)."""
        total_items = len(data)
        valid_items = 0
        invalid_items = 0
        errors = []
        
        for item in data:
            try:
                # Validate item structure
                if not isinstance(item, dict):
                    invalid_items += 1
                    errors.append(f"Item is not a dictionary: {type(item)}")
                    continue
                
                # Check required fields
                required_fields = ["target", "port", "is_open"]
                missing_fields = [field for field in required_fields if field not in item]
                
                if missing_fields:
                    invalid_items += 1
                    errors.append(f"Missing required fields: {missing_fields}")
                    continue
                
                valid_items += 1
                
            except Exception as e:
                invalid_items += 1
                errors.append(f"Validation error: {str(e)}")
        
        return {
            "total_items": total_items,
            "valid_items": valid_items,
            "invalid_items": invalid_items,
            "integrity_score": valid_items / total_items if total_items > 0 else 0,
            "errors": errors[:10]  # Limit error list
        }

class FileIOHelper:
    """Dedicated async helper for file I/O operations."""
    
    def __init__(self, config: AsyncHelperConfig):
        
    """__init__ function."""
self.config = config
        self._executor = ThreadPoolExecutor(max_workers=config.max_workers)
    
    async def __aenter__(self) -> Any:
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> Any:
        await self.close()
    
    async def close(self) -> Any:
        """Cleanup resources."""
        self._executor.shutdown(wait=True)
    
    async def read_file_async(self, filepath: str) -> Tuple[bool, str, float]:
        """Async file reading with timeout."""
        start_time = time.time()
        
        try:
            content = await asyncio.get_event_loop().run_in_executor(
                self._executor,
                self._blocking_read_file,
                filepath
            )
            
            duration = time.time() - start_time
            return True, content, duration
            
        except Exception as e:
            duration = time.time() - start_time
            return False, str(e), duration
    
    def _blocking_read_file(self, filepath: str) -> str:
        """Blocking file read (runs in executor)."""
        with open(filepath, 'r', encoding='utf-8') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            return f.read()
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
    
    async def write_file_async(self, filepath: str, content: str) -> Tuple[bool, str, float]:
        """Async file writing with timeout."""
        start_time = time.time()
        
        try:
            await asyncio.get_event_loop().run_in_executor(
                self._executor,
                self._blocking_write_file,
                filepath, content
            )
            
            duration = time.time() - start_time
            return True, "File written successfully", duration
            
        except Exception as e:
            duration = time.time() - start_time
            return False, str(e), duration
    
    def _blocking_write_file(self, filepath: str, content: str):
        """Blocking file write (runs in executor)."""
        with open(filepath, 'w', encoding='utf-8') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write(content)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
    
    async def read_json_async(self, filepath: str) -> Tuple[bool, Any, float]:
        """Async JSON file reading."""
        success, content, duration = await self.read_file_async(filepath)
        
        if not success:
            return False, content, duration
        
        try:
            data = await asyncio.get_event_loop().run_in_executor(
                self._executor,
                json.loads,
                content
            )
            return True, data, duration
        except Exception as e:
            return False, str(e), duration
    
    async def write_json_async(self, filepath: str, data: Any) -> Tuple[bool, str, float]:
        """Async JSON file writing."""
        try:
            content = await asyncio.get_event_loop().run_in_executor(
                self._executor,
                json.dumps,
                data,
                indent=2
            )
            
            return await self.write_file_async(filepath, content)
            
        except Exception as e:
            return False, str(e), 0.0

class AsyncHelperManager:
    """Manager for all async helpers."""
    
    def __init__(self, config: AsyncHelperConfig = None):
        
    """__init__ function."""
self.config = config or AsyncHelperConfig()
        self.network_io = NetworkIOHelper(self.config)
        self.data_processing = DataProcessingHelper(self.config)
        self.file_io = FileIOHelper(self.config)
    
    async def __aenter__(self) -> Any:
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> Any:
        await self.close()
    
    async def close(self) -> Any:
        """Close all helpers."""
        await self.network_io.close()
        await self.data_processing.close()
        await self.file_io.close()
    
    async def comprehensive_scan_async(self, host: str, ports: List[int]) -> Dict[str, Any]:
        """Comprehensive async scan using all helpers."""
        start_time = time.time()
        results = []
        
        # Scan ports concurrently
        scan_tasks = []
        for port in ports:
            task = self._scan_port_with_helpers(host, port)
            scan_tasks.append(task)
        
        # Gather results
        scan_results = await asyncio.gather(*scan_tasks, return_exceptions=True)
        
        # Process results
        for result in scan_results:
            if isinstance(result, Exception):
                logger.error(f"Scan error: {result}")
            else:
                results.append(result)
        
        # Analyze results
        analysis = await self.data_processing.analyze_scan_data(results)
        
        total_duration = time.time() - start_time
        
        return {
            "target": host,
            "ports_scanned": len(ports),
            "results": results,
            "analysis": analysis,
            "total_duration": total_duration,
            "scan_rate": len(ports) / total_duration if total_duration > 0 else 0
        }
    
    async def _scan_port_with_helpers(self, host: str, port: int) -> Dict[str, Any]:
        """Scan a single port using async helpers."""
        start_time = time.time()
        
        # TCP connection
        tcp_success, tcp_duration, tcp_error = await self.network_io.tcp_connect(host, port)
        
        result = {
            "target": host,
            "port": port,
            "is_open": tcp_success,
            "tcp_duration": tcp_duration,
            "tcp_error": tcp_error,
            "success": tcp_success
        }
        
        if tcp_success:
            # SSL check for common SSL ports
            if port in [443, 993, 995, 8443]:
                ssl_success, ssl_duration, ssl_info = await self.network_io.ssl_connect(host, port)
                result.update({
                    "ssl_success": ssl_success,
                    "ssl_duration": ssl_duration,
                    "ssl_info": ssl_info
                })
            
            # Banner grab
            banner_success, banner_content, banner_duration = await self.network_io.banner_grab(host, port)
            result.update({
                "banner_success": banner_success,
                "banner_content": banner_content,
                "banner_duration": banner_duration
            })
        
        result["total_duration"] = time.time() - start_time
        return result

# Global helper manager
_global_helper_manager: Optional[AsyncHelperManager] = None

def get_async_helper_manager(config: AsyncHelperConfig = None) -> AsyncHelperManager:
    """Get or create global async helper manager."""
    global _global_helper_manager
    
    if _global_helper_manager is None:
        _global_helper_manager = AsyncHelperManager(config)
    
    return _global_helper_manager

async def cleanup_async_helpers():
    """Cleanup global async helpers."""
    global _global_helper_manager
    
    if _global_helper_manager:
        await _global_helper_manager.close()
        _global_helper_manager = None 