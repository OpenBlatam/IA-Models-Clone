from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import time
from typing import Dict, Any, Optional, List, Set, Callable
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
import logging
        from ..async_helpers import get_async_helper
        from ..async_helpers import get_async_helper
        from ..async_helpers import get_async_helper
        from ..async_helpers import get_async_helper
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
Non-Blocking Scanner Core
Implements non-blocking scanning operations using dedicated async helpers.
"""


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class NonBlockingScanConfig:
    """Configuration for non-blocking scanning."""
    max_concurrent_scans: int = 50
    scan_timeout: float = 30.0
    chunk_size: int = 100
    enable_dns_cache: bool = True
    enable_result_cache: bool = True
    cache_ttl: int = 3600

@dataclass
class NonBlockingScanResult:
    """Result of non-blocking scan operation."""
    target: str
    scan_type: str
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    duration: float = 0.0
    metadata: Optional[Dict[str, Any]] = None

class NonBlockingScanner:
    """Non-blocking scanner that extracts heavy I/O to async helpers."""
    
    def __init__(self, config: NonBlockingScanConfig):
        
    """__init__ function."""
self.config = config
        self._scan_semaphore = asyncio.Semaphore(config.max_concurrent_scans)
        self._active_scans: Set[asyncio.Task] = set()
        self._scan_cache: Dict[str, Any] = {}
        self._dns_cache: Dict[str, Any] = {}
        self._scan_stats: Dict[str, List[float]] = {}
    
    async def scan_targets_non_blocking(self, targets: List[str], 
                                      scan_types: List[str] = None) -> Dict[str, List[NonBlockingScanResult]]:
        """Scan multiple targets without blocking operations."""
        if scan_types is None:
            scan_types = ["dns", "port", "http", "ssl"]
        
        # Process targets in chunks to avoid memory issues
        results = {}
        
        for i in range(0, len(targets), self.config.chunk_size):
            chunk = targets[i:i + self.config.chunk_size]
            chunk_results = await self._scan_target_chunk(chunk, scan_types)
            results.update(chunk_results)
        
        return results
    
    async def _scan_target_chunk(self, targets: List[str], 
                                scan_types: List[str]) -> Dict[str, List[NonBlockingScanResult]]:
        """Scan a chunk of targets concurrently."""
        results = {}
        
        for target in targets:
            target_tasks = []
            
            for scan_type in scan_types:
                task = asyncio.create_task(
                    self._perform_single_scan(target, scan_type)
                )
                target_tasks.append((scan_type, task))
            
            # Wait for all scan types to complete for this target
            target_results = []
            for scan_type, task in target_tasks:
                try:
                    result = await task
                    target_results.append(result)
                except Exception as e:
                    target_results.append(NonBlockingScanResult(
                        target=target,
                        scan_type=scan_type,
                        success=False,
                        error=str(e)
                    ))
            
            results[target] = target_results
        
        return results
    
    async def _perform_single_scan(self, target: str, scan_type: str) -> NonBlockingScanResult:
        """Perform a single scan operation using async helpers."""
        start_time = time.time()
        
        async with self._scan_semaphore:
            try:
                # Check cache first
                cache_key = f"{target}:{scan_type}"
                if self.config.enable_result_cache and cache_key in self._scan_cache:
                    cached_result = self._scan_cache[cache_key]
                    if time.time() - cached_result.get("timestamp", 0) < self.config.cache_ttl:
                        return NonBlockingScanResult(
                            target=target,
                            scan_type=scan_type,
                            success=True,
                            data=cached_result["data"],
                            duration=time.time() - start_time,
                            metadata={"cached": True}
                        )
                
                # Perform actual scan using async helper
                if scan_type == "dns":
                    result = await self._dns_scan_async(target)
                elif scan_type == "port":
                    result = await self._port_scan_async(target)
                elif scan_type == "http":
                    result = await self._http_scan_async(target)
                elif scan_type == "ssl":
                    result = await self._ssl_scan_async(target)
                else:
                    raise ValueError(f"Unsupported scan type: {scan_type}")
                
                duration = time.time() - start_time
                
                # Cache result
                if self.config.enable_result_cache and result.success:
                    self._scan_cache[cache_key] = {
                        "data": result.data,
                        "timestamp": time.time()
                    }
                
                # Update stats
                if scan_type not in self._scan_stats:
                    self._scan_stats[scan_type] = []
                self._scan_stats[scan_type].append(duration)
                
                return NonBlockingScanResult(
                    target=target,
                    scan_type=scan_type,
                    success=result.success,
                    data=result.data,
                    error=result.error,
                    duration=duration
                )
                
            except Exception as e:
                duration = time.time() - start_time
                return NonBlockingScanResult(
                    target=target,
                    scan_type=scan_type,
                    success=False,
                    error=str(e),
                    duration=duration
                )
    
    async def _dns_scan_async(self, target: str):
        """DNS scan using async helper."""
        # Import async helper here to avoid circular imports
        
        helper = get_async_helper()
        return await helper.dns_lookup(target)
    
    async def _port_scan_async(self, target: str):
        """Port scan using async helper."""
        
        helper = get_async_helper()
        
        # Scan common ports concurrently
        common_ports = [22, 23, 25, 53, 80, 110, 143, 443, 993, 995]
        port_tasks = []
        
        for port in common_ports:
            task = helper.port_scan(target, port, "tcp")
            port_tasks.append((port, task))
        
        # Wait for all port scans
        results = {}
        for port, task in port_tasks:
            result = await task
            if result.success and result.data:
                results[port] = "open"
            else:
                results[port] = "closed"
        
        return type('obj', (object,), {
            'success': True,
            'data': results,
            'error': None
        })()
    
    async def _http_scan_async(self, target: str):
        """HTTP scan using async helper."""
        
        helper = get_async_helper()
        
        # Try both HTTP and HTTPS
        http_urls = [f"http://{target}", f"https://{target}"]
        http_tasks = []
        
        for url in http_urls:
            task = helper.http_request(url)
            http_tasks.append((url, task))
        
        # Wait for HTTP responses
        results = {}
        for url, task in http_tasks:
            result = await task
            if result.success:
                results[url] = {
                    "status_code": result.data.get("status_code"),
                    "headers": result.data.get("headers", {})
                }
            else:
                results[url] = {"error": result.error}
        
        return type('obj', (object,), {
            'success': True,
            'data': results,
            'error': None
        })()
    
    async def _ssl_scan_async(self, target: str):
        """SSL scan using async helper."""
        
        helper = get_async_helper()
        
        # Test SSL on common ports
        ssl_ports = [443, 993, 995, 8443]
        ssl_tasks = []
        
        for port in ssl_ports:
            task = helper.port_scan(target, port, "ssl")
            ssl_tasks.append((port, task))
        
        # Wait for SSL scans
        results = {}
        for port, task in ssl_tasks:
            result = await task
            if result.success and result.data:
                results[port] = "ssl_enabled"
            else:
                results[port] = "ssl_disabled"
        
        return type('obj', (object,), {
            'success': True,
            'data': results,
            'error': None
        })()
    
    async def batch_scan_with_progress(self, targets: List[str], 
                                     scan_types: List[str] = None,
                                     progress_callback: Optional[Callable] = None) -> Dict[str, List[NonBlockingScanResult]]:
        """Scan targets with progress reporting."""
        if scan_types is None:
            scan_types = ["dns", "port", "http"]
        
        total_targets = len(targets)
        completed_targets = 0
        results = {}
        
        for i in range(0, total_targets, self.config.chunk_size):
            chunk = targets[i:i + self.config.chunk_size]
            chunk_results = await self._scan_target_chunk(chunk, scan_types)
            results.update(chunk_results)
            
            completed_targets += len(chunk)
            if progress_callback:
                progress = (completed_targets / total_targets) * 100
                await progress_callback(progress, completed_targets, total_targets)
        
        return results
    
    def get_scan_stats(self) -> Dict[str, Any]:
        """Get scanning statistics."""
        stats = {}
        for scan_type, durations in self._scan_stats.items():
            if durations:
                stats[scan_type] = {
                    "total_scans": len(durations),
                    "avg_duration": sum(durations) / len(durations),
                    "min_duration": min(durations),
                    "max_duration": max(durations),
                    "total_duration": sum(durations)
                }
        
        stats["cache"] = {
            "result_cache_size": len(self._scan_cache),
            "dns_cache_size": len(self._dns_cache)
        }
        
        return stats
    
    def clear_cache(self) -> Any:
        """Clear all caches."""
        self._scan_cache.clear()
        self._dns_cache.clear()
    
    async def close(self) -> Any:
        """Close the scanner and cleanup resources."""
        # Wait for all active scans to complete
        if self._active_scans:
            await asyncio.gather(*self._active_scans, return_exceptions=True)
            self._active_scans.clear()

# Global non-blocking scanner instance
_global_non_blocking_scanner: Optional[NonBlockingScanner] = None

def get_non_blocking_scanner(config: NonBlockingScanConfig = None) -> NonBlockingScanner:
    """Get or create global non-blocking scanner."""
    global _global_non_blocking_scanner
    
    if _global_non_blocking_scanner is None:
        if config is None:
            config = NonBlockingScanConfig()
        _global_non_blocking_scanner = NonBlockingScanner(config)
    
    return _global_non_blocking_scanner

async def cleanup_non_blocking_resources():
    """Cleanup global non-blocking resources."""
    global _global_non_blocking_scanner
    
    if _global_non_blocking_scanner:
        await _global_non_blocking_scanner.close()
        _global_non_blocking_scanner = None 