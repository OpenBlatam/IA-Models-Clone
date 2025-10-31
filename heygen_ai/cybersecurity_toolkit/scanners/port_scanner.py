from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import asyncio
import socket
import time
from typing import Dict, List, Optional, Any, Union
from concurrent.futures import ThreadPoolExecutor
import logging
from ..utils.structured_logger import get_logger, log_async_function_call, log_function_call
from ..exceptions.custom_exceptions import (
from ..exceptions.error_mapper import format_cli_error, format_api_error, log_error
from typing import Any, List, Dict, Optional
"""
Port Scanner Module - Happy Path Pattern
========================================

Port scanning functionality using happy path pattern:
- Early returns for error conditions
- Avoiding nested conditionals
- Keeping main success logic at the end
- Guard clauses for validation
- Clean, readable code structure
"""


# Import structured logger

# Import custom exceptions
    ValidationError,
    MissingRequiredFieldError,
    InvalidFieldTypeError,
    FieldValueOutOfRangeError,
    NetworkError,
    ConnectionTimeoutError,
    ConnectionRefusedError,
    InvalidTargetError,
    ScanningError,
    PortScanError,
    ScanConfigurationError
)

# Import error mapper

# Get logger instance
logger = get_logger("port_scanner")

class PortScanner:
    """
    Port scanner using happy path pattern with early returns and clean structure.
    """
    
    def __init__(self, max_concurrent_scans: int = 100, timeout: float = 5.0):
        """
        Initialize port scanner with validation using early returns.
        
        Args:
            max_concurrent_scans: Maximum concurrent port scans
            timeout: Connection timeout in seconds
        """
        # Guard clause 1: Validate max_concurrent_scans
        if not isinstance(max_concurrent_scans, int) or max_concurrent_scans <= 0:
            raise FieldValueOutOfRangeError(
                "max_concurrent_scans",
                max_concurrent_scans,
                min_value=1,
                context={"operation": "scanner_initialization"}
            )
        
        # Guard clause 2: Validate timeout
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            raise FieldValueOutOfRangeError(
                "timeout",
                timeout,
                min_value=0.1,
                context={"operation": "scanner_initialization"}
            )
        
        # Happy path: Set valid parameters
        self.max_concurrent_scans = max_concurrent_scans
        self.timeout = timeout
        self.semaphore = asyncio.Semaphore(max_concurrent_scans)
        
        # Log initialization
        logger.log_function_entry(
            "__init__",
            {
                "max_concurrent_scans": max_concurrent_scans,
                "timeout": timeout
            },
            {"event_type": "scanner_initialization"}
        )
    
    @log_function_call
    def validate_scan_parameters(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate scan parameters using happy path pattern with guard clauses.
        
        Args:
            request: Scan request dictionary
            
        Returns:
            Validation result dictionary
            
        Raises:
            ValidationError: When validation fails
        """
        # Guard clause 1: Check if request is provided
        if not request:
            raise MissingRequiredFieldError(
                "request",
                context={"operation": "port_scan_validation"}
            )
        
        # Guard clause 2: Check if request is a dictionary
        if not isinstance(request, dict):
            raise InvalidFieldTypeError(
                "request",
                request,
                "dict",
                context={"operation": "port_scan_validation"}
            )
        
        # Guard clause 3: Check if target_host is provided
        target_host = request.get("target_host")
        if not target_host:
            raise MissingRequiredFieldError(
                "target_host",
                context={"operation": "port_scan_validation"}
            )
        
        # Guard clause 4: Check if target_host is a string
        if not isinstance(target_host, str):
            raise InvalidFieldTypeError(
                "target_host",
                target_host,
                "str",
                context={"operation": "port_scan_validation"}
            )
        
        # Guard clause 5: Check target_host length
        if len(target_host) > 253:
            raise FieldValueOutOfRangeError(
                "target_host",
                len(target_host),
                max_value=253,
                context={"operation": "port_scan_validation"}
            )
        
        # Guard clause 6: Validate target_ports
        target_ports = request.get("target_ports", [80, 443])
        if not isinstance(target_ports, list):
            raise InvalidFieldTypeError(
                "target_ports",
                target_ports,
                "list",
                context={"operation": "port_scan_validation"}
            )
        
        # Guard clause 7: Validate individual ports
        for i, port in enumerate(target_ports):
            if not isinstance(port, int):
                raise InvalidFieldTypeError(
                    f"target_ports[{i}]",
                    port,
                    "int",
                    context={"operation": "port_scan_validation"}
                )
            
            if port < 1 or port > 65535:
                raise FieldValueOutOfRangeError(
                    f"target_ports[{i}]",
                    port,
                    min_value=1,
                    max_value=65535,
                    context={"operation": "port_scan_validation"}
                )
        
        # Guard clause 8: Check port list size
        if len(target_ports) > 1000:
            raise FieldValueOutOfRangeError(
                "target_ports",
                len(target_ports),
                max_value=1000,
                context={"operation": "port_scan_validation"}
            )
        
        # Guard clause 9: Validate scan_timeout
        scan_timeout = request.get("scan_timeout", self.timeout)
        if not isinstance(scan_timeout, (int, float)):
            raise InvalidFieldTypeError(
                "scan_timeout",
                scan_timeout,
                "number",
                context={"operation": "port_scan_validation"}
            )
        
        if scan_timeout <= 0:
            raise FieldValueOutOfRangeError(
                "scan_timeout",
                scan_timeout,
                min_value=0.1,
                context={"operation": "port_scan_validation"}
            )
        
        # Happy path: All validation passed
        logger.log_function_exit(
            "validate_scan_parameters",
            {"is_valid": True},
            context={"validation_result": "success"}
        )
        
        return {
            "is_valid": True,
            "validated_request": {
                "target_host": target_host,
                "target_ports": target_ports,
                "scan_timeout": scan_timeout
            }
        }
    
    @log_async_function_call
    async def scan_single_port_async(self, host: str, port: int, timeout: float) -> Dict[str, Any]:
        """
        Scan a single port asynchronously using happy path pattern.
        
        Args:
            host: Target host
            port: Target port
            timeout: Connection timeout
            
        Returns:
            Port scan result dictionary
            
        Raises:
            ConnectionTimeoutError: When connection times out
            ConnectionRefusedError: When connection is refused
        """
        async with self.semaphore:
            try:
                # Happy path: Create connection with timeout
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(host, port),
                    timeout=timeout
                )
                
                # Happy path: Port is open, close connection
                writer.close()
                await writer.wait_closed()
                
                logger.log_function_exit(
                    "scan_single_port_async",
                    {"is_open": True, "port": port},
                    context={"scan_result": "port_open"}
                )
                
                return {
                    "port": port,
                    "is_open": True,
                    "service": self.get_service_name(port),
                    "scan_time": time.time()
                }
                
            except asyncio.TimeoutError:
                raise ConnectionTimeoutError(
                    target=host,
                    port=port,
                    timeout=timeout,
                    context={"operation": "port_scan"}
                )
                
            except ConnectionRefusedError:
                raise ConnectionRefusedError(
                    target=host,
                    port=port,
                    context={"operation": "port_scan"}
                )
                
            except Exception as e:
                logger.log_error(
                    e,
                    "scan_single_port_async",
                    {"host": host, "port": port, "timeout": timeout},
                    {"scan_result": "port_error"}
                )
                
                return {
                    "port": port,
                    "is_open": False,
                    "error": str(e),
                    "scan_time": time.time()
                }
    
    @log_function_call
    def get_service_name(self, port: int) -> str:
        """
        Get service name for port (CPU-bound operation).
        
        Args:
            port: Port number
            
        Returns:
            Service name
        """
        common_services = {
            21: "FTP", 22: "SSH", 23: "Telnet", 25: "SMTP", 53: "DNS",
            80: "HTTP", 110: "POP3", 143: "IMAP", 443: "HTTPS", 993: "IMAPS",
            995: "POP3S", 3306: "MySQL", 5432: "PostgreSQL", 6379: "Redis",
            27017: "MongoDB", 8080: "HTTP-Proxy", 8443: "HTTPS-Alt"
        }
        
        return common_services.get(port, "Unknown")
    
    @log_async_function_call
    async def scan_ports_async(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Scan multiple ports asynchronously using happy path pattern.
        
        Args:
            request: Scan request dictionary
            
        Returns:
            Scan results dictionary
            
        Raises:
            ValidationError: When request validation fails
            PortScanError: When port scanning fails
            InvalidTargetError: When target is invalid
        """
        # Guard clause 1: Validate request
        try:
            validation_result = self.validate_scan_parameters(request)
        except ValidationError as e:
            # Re-raise validation errors
            raise
        
        # Guard clause 2: Extract validated parameters
        validated_request = validation_result["validated_request"]
        target_host = validated_request["target_host"]
        target_ports = validated_request["target_ports"]
        scan_timeout = validated_request["scan_timeout"]
        
        # Guard clause 3: Check if target is reachable (simulated)
        if not self._is_target_reachable(target_host):
            raise InvalidTargetError(
                target=target_host,
                reason="Target is not reachable",
                context={"operation": "port_scan"}
            )
        
        # Happy path: Start scanning
        scan_start_time = time.time()
        
        try:
            # Create scan tasks
            scan_tasks = [
                self.scan_single_port_async(target_host, port, scan_timeout)
                for port in target_ports
            ]
            
            # Execute all scans concurrently
            scan_results = await asyncio.gather(*scan_tasks, return_exceptions=True)
            
            # Process results
            open_ports = []
            closed_ports = []
            error_ports = []
            
            for result in scan_results:
                if isinstance(result, Exception):
                    # Log the exception
                    log_error(
                        result,
                        logger,
                        {
                            "target_host": target_host,
                            "target_ports": target_ports,
                            "operation": "port_scan_gather"
                        }
                    )
                    
                    error_ports.append({
                        "port": "unknown",
                        "error": str(result)
                    })
                elif result["is_open"]:
                    open_ports.append(result)
                else:
                    closed_ports.append(result)
            
            scan_duration = time.time() - scan_start_time
            
            # Log performance metrics
            logger.log_performance(
                "port_scan",
                scan_duration,
                {
                    "target_host": target_host,
                    "ports_scanned": len(target_ports),
                    "open_ports": len(open_ports),
                    "closed_ports": len(closed_ports),
                    "error_ports": len(error_ports)
                }
            )
            
            # Log security event
            logger.log_security_event(
                "port_scan_completed",
                {
                    "target_host": target_host,
                    "ports_scanned": len(target_ports),
                    "open_ports": [p["port"] for p in open_ports],
                    "scan_duration": scan_duration
                },
                "INFO"
            )
            
            # Happy path: Return successful scan results
            return {
                "success": True,
                "target_host": target_host,
                "scan_results": {
                    "open_ports": open_ports,
                    "closed_ports": closed_ports,
                    "error_ports": error_ports
                },
                "metadata": {
                    "ports_scanned": len(target_ports),
                    "open_ports_count": len(open_ports),
                    "closed_ports_count": len(closed_ports),
                    "error_ports_count": len(error_ports),
                    "scan_duration": scan_duration,
                    "scan_start_time": scan_start_time,
                    "scan_end_time": time.time()
                }
            }
            
        except (ValidationError, NetworkError, ScanningError) as e:
            # Re-raise custom exceptions
            raise
        except Exception as e:
            scan_duration = time.time() - scan_start_time if 'scan_start_time' in locals() else 0
            
            # Wrap in PortScanError
            raise PortScanError(
                target=target_host,
                ports=target_ports,
                reason=str(e),
                context={"scan_duration": scan_duration},
                original_exception=e
            )
    
    def _is_target_reachable(self, target: str) -> bool:
        """
        Check if target is reachable (simulated).
        
        Args:
            target: Target to check
            
        Returns:
            True if target is reachable
        """
        # Simulate reachability check
        # In real implementation, this would do a ping or DNS lookup
        return True

# Global port scanner instance
_port_scanner = PortScanner()

@log_async_function_call
async def scan_ports_async(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Scan ports asynchronously using happy path pattern (RORO pattern).
    
    Args:
        request: Scan request dictionary
        
    Returns:
        Scan results dictionary
        
    Raises:
        ValidationError: When request validation fails
        PortScanError: When port scanning fails
        InvalidTargetError: When target is invalid
    """
    try:
        return await _port_scanner.scan_ports_async(request)
    except Exception as e:
        # Log the error with structured logging
        log_error(
            e,
            logger,
            {"operation": "scan_ports_async", "request": request}
        )
        
        # Re-raise the exception
        raise

@log_function_call
def scan_ports_sync(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Scan ports synchronously using happy path pattern (RORO pattern).
    
    Args:
        request: Scan request dictionary
        
    Returns:
        Scan results dictionary
        
    Raises:
        ValidationError: When request validation fails
        PortScanError: When port scanning fails
        InvalidTargetError: When target is invalid
    """
    try:
        # Run async function in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(scan_ports_async(request))
        finally:
            loop.close()
    except Exception as e:
        # Log the error with structured logging
        log_error(
            e,
            logger,
            {"operation": "scan_ports_sync", "request": request}
        )
        
        # Re-raise the exception
        raise

# --- Named Exports ---

__all__ = [
    'PortScanner',
    'scan_ports_async',
    'scan_ports_sync'
] 