from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, field_validator
import structlog
import socket
import asyncio
import concurrent.futures
from dataclasses import dataclass
from enum import Enum
        import time
        import time
        import socket
        import time
from typing import Any, List, Dict, Optional
import logging
"""
Port scanning utilities for cybersecurity tools.
"""

logger = structlog.get_logger(__name__)

class PortStatus(str, Enum):
    """Port status enumeration."""
    OPEN = "open"
    CLOSED = "closed"
    FILTERED = "filtered"
    UNKNOWN = "unknown"

@dataclass
class PortInfo:
    """Information about a port."""
    port: int
    status: PortStatus
    service: Optional[str] = None
    banner: Optional[str] = None
    response_time: Optional[float] = None

class PortCheckInput(BaseModel):
    """Input model for port status checking."""
    host: str
    port: int
    timeout: float = 5.0
    check_service: bool = True
    get_banner: bool = False
    
    @field_validator('host')
    def validate_host(cls, v) -> bool:
        if not v:
            raise ValueError("Host cannot be empty")
        return v
    
    @field_validator('port')
    def validate_port(cls, v) -> bool:
        if v < 1 or v > 65535:
            raise ValueError("Port must be between 1 and 65535")
        return v
    
    @field_validator('timeout')
    def validate_timeout(cls, v) -> bool:
        if v <= 0:
            raise ValueError("Timeout must be positive")
        return v

class PortScanInput(BaseModel):
    """Input model for port scanning."""
    host: str
    ports: List[int]
    timeout: float = 5.0
    max_workers: int = 10
    check_services: bool = True
    get_banners: bool = False
    
    @field_validator('host')
    def validate_host(cls, v) -> bool:
        if not v:
            raise ValueError("Host cannot be empty")
        return v
    
    @field_validator('ports')
    def validate_ports(cls, v) -> bool:
        if not v:
            raise ValueError("Ports list cannot be empty")
        for port in v:
            if port < 1 or port > 65535:
                raise ValueError(f"Port {port} must be between 1 and 65535")
        return v
    
    @field_validator('max_workers')
    def validate_max_workers(cls, v) -> bool:
        if v < 1 or v > 100:
            raise ValueError("Max workers must be between 1 and 100")
        return v

class PortCheckResult(BaseModel):
    """Result model for port status checking."""
    host: str
    port: int
    status: PortStatus
    service: Optional[str] = None
    banner: Optional[str] = None
    response_time: Optional[float] = None
    is_successful: bool
    error_message: Optional[str] = None

class PortScanResult(BaseModel):
    """Result model for port scanning."""
    host: str
    open_ports: List[PortInfo]
    closed_ports: List[PortInfo]
    filtered_ports: List[PortInfo]
    scan_time: float
    total_ports: int
    is_successful: bool
    error_message: Optional[str] = None

def check_port_status(input_data: PortCheckInput) -> PortCheckResult:
    """
    RORO: Receive PortCheckInput, return PortCheckResult
    
    Check the status of a specific port.
    """
    try:
        start_time = time.time()
        
        # Create socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(input_data.timeout)
        
        # Try to connect
        try:
            result = sock.connect_ex((input_data.host, input_data.port))
            response_time = time.time() - start_time
            
            if result == 0:
                status = PortStatus.OPEN
                service = get_service_name(input_data.port) if input_data.check_service else None
                banner = get_port_banner(sock) if input_data.get_banner else None
            else:
                status = PortStatus.CLOSED
                service = None
                banner = None
                
        except socket.timeout:
            status = PortStatus.FILTERED
            service = None
            banner = None
            response_time = time.time() - start_time
        except Exception as e:
            status = PortStatus.UNKNOWN
            service = None
            banner = None
            response_time = time.time() - start_time
            logger.warning(f"Port check failed for {input_data.host}:{input_data.port}", error=str(e))
        
        finally:
            sock.close()
        
        return PortCheckResult(
            host=input_data.host,
            port=input_data.port,
            status=status,
            service=service,
            banner=banner,
            response_time=response_time,
            is_successful=True
        )
        
    except Exception as e:
        logger.error("Port status check failed", error=str(e))
        return PortCheckResult(
            host=input_data.host,
            port=input_data.port,
            status=PortStatus.UNKNOWN,
            is_successful=False,
            error_message=str(e)
        )

def scan_ports(input_data: PortScanInput) -> PortScanResult:
    """
    RORO: Receive PortScanInput, return PortScanResult
    
    Scan multiple ports on a host.
    """
    try:
        start_time = time.time()
        
        # Use ThreadPoolExecutor for concurrent scanning
        with concurrent.futures.ThreadPoolExecutor(max_workers=input_data.max_workers) as executor:
            # Create futures for each port
            futures = []
            for port in input_data.ports:
                port_input = PortCheckInput(
                    host=input_data.host,
                    port=port,
                    timeout=input_data.timeout,
                    check_service=input_data.check_services,
                    get_banner=input_data.get_banners
                )
                future = executor.submit(check_port_status, port_input)
                futures.append(future)
            
            # Collect results
            open_ports = []
            closed_ports = []
            filtered_ports = []
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    port_info = PortInfo(
                        port=result.port,
                        status=result.status,
                        service=result.service,
                        banner=result.banner,
                        response_time=result.response_time
                    )
                    
                    if result.status == PortStatus.OPEN:
                        open_ports.append(port_info)
                    elif result.status == PortStatus.CLOSED:
                        closed_ports.append(port_info)
                    elif result.status == PortStatus.FILTERED:
                        filtered_ports.append(port_info)
                        
                except Exception as e:
                    logger.warning("Port scan result processing failed", error=str(e))
        
        scan_time = time.time() - start_time
        
        return PortScanResult(
            host=input_data.host,
            open_ports=open_ports,
            closed_ports=closed_ports,
            filtered_ports=filtered_ports,
            scan_time=scan_time,
            total_ports=len(input_data.ports),
            is_successful=True
        )
        
    except Exception as e:
        logger.error("Port scanning failed", error=str(e))
        return PortScanResult(
            host=input_data.host,
            open_ports=[],
            closed_ports=[],
            filtered_ports=[],
            scan_time=0.0,
            total_ports=len(input_data.ports),
            is_successful=False,
            error_message=str(e)
        )

def get_service_name(port: int) -> Optional[str]:
    """Get service name for a port."""
    try:
        
        # Common port to service mapping
        common_services = {
            21: "FTP",
            22: "SSH",
            23: "Telnet",
            25: "SMTP",
            53: "DNS",
            80: "HTTP",
            110: "POP3",
            143: "IMAP",
            443: "HTTPS",
            993: "IMAPS",
            995: "POP3S",
            3306: "MySQL",
            5432: "PostgreSQL",
            6379: "Redis",
            8080: "HTTP-Proxy",
            8443: "HTTPS-Alt"
        }
        
        return common_services.get(port)
        
    except Exception as e:
        logger.warning(f"Service name lookup failed for port {port}", error=str(e))
        return None

def get_port_banner(sock: socket.socket) -> Optional[str]:
    """Get banner from an open port."""
    try:
        # Send a simple probe
        probe = b"\r\n"
        sock.send(probe)
        
        # Try to receive response
        sock.settimeout(2.0)
        response = sock.recv(1024)
        
        if response:
            return response.decode('utf-8', errors='ignore').strip()
        
        return None
        
    except Exception as e:
        logger.debug(f"Banner grab failed", error=str(e))
        return None

async def check_port_status_async(input_data: PortCheckInput) -> PortCheckResult:
    """Async version of port status checking."""
    try:
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            result = await loop.run_in_executor(executor, check_port_status, input_data)
        return result
        
    except Exception as e:
        logger.error("Async port status check failed", error=str(e))
        return PortCheckResult(
            host=input_data.host,
            port=input_data.port,
            status=PortStatus.UNKNOWN,
            is_successful=False,
            error_message=str(e)
        )

async def scan_ports_async(input_data: PortScanInput) -> PortScanResult:
    """Async version of port scanning."""
    try:
        start_time = time.time()
        
        # Create tasks for each port
        tasks = []
        for port in input_data.ports:
            port_input = PortCheckInput(
                host=input_data.host,
                port=port,
                timeout=input_data.timeout,
                check_service=input_data.check_services,
                get_banner=input_data.get_banners
            )
            task = check_port_status_async(port_input)
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        open_ports = []
        closed_ports = []
        filtered_ports = []
        
        for result in results:
            if isinstance(result, PortCheckResult):
                port_info = PortInfo(
                    port=result.port,
                    status=result.status,
                    service=result.service,
                    banner=result.banner,
                    response_time=result.response_time
                )
                
                if result.status == PortStatus.OPEN:
                    open_ports.append(port_info)
                elif result.status == PortStatus.CLOSED:
                    closed_ports.append(port_info)
                elif result.status == PortStatus.FILTERED:
                    filtered_ports.append(port_info)
        
        scan_time = time.time() - start_time
        
        return PortScanResult(
            host=input_data.host,
            open_ports=open_ports,
            closed_ports=closed_ports,
            filtered_ports=filtered_ports,
            scan_time=scan_time,
            total_ports=len(input_data.ports),
            is_successful=True
        )
        
    except Exception as e:
        logger.error("Async port scanning failed", error=str(e))
        return PortScanResult(
            host=input_data.host,
            open_ports=[],
            closed_ports=[],
            filtered_ports=[],
            scan_time=0.0,
            total_ports=len(input_data.ports),
            is_successful=False,
            error_message=str(e)
        ) 