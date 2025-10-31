#!/usr/bin/env python3
"""
Port Scanner Module for Video-OpusClip
Scans network ports for security assessment
"""

import asyncio
import socket
import time
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import aiohttp
import nmap

class ScanType(str, Enum):
    """Types of port scans"""
    TCP_CONNECT = "tcp_connect"
    TCP_SYN = "tcp_syn"
    UDP = "udp"
    SERVICE_DETECTION = "service_detection"

class PortStatus(str, Enum):
    """Port status enumeration"""
    OPEN = "open"
    CLOSED = "closed"
    FILTERED = "filtered"
    UNFILTERED = "unfiltered"

@dataclass
class PortResult:
    """Result of a port scan"""
    port: int
    status: PortStatus
    service: Optional[str] = None
    version: Optional[str] = None
    banner: Optional[str] = None
    response_time: Optional[float] = None
    scan_time: float = 0.0

@dataclass
class ScanConfig:
    """Configuration for port scanning"""
    target_host: str
    start_port: int = 1
    end_port: int = 1024
    scan_type: ScanType = ScanType.TCP_CONNECT
    timeout: float = 3.0
    max_concurrent: int = 100
    service_detection: bool = True
    banner_grab: bool = True

class PortScanner:
    """Port scanner for network security assessment"""
    
    def __init__(self, config: ScanConfig):
        self.config = config
        self.results: List[PortResult] = []
        self.scan_start_time: float = 0.0
        self.scan_end_time: float = 0.0
    
    async def scan_ports(self) -> Dict[str, Any]:
        """Scan ports on target host"""
        self.scan_start_time = time.time()
        
        try:
            if self.config.scan_type == ScanType.TCP_CONNECT:
                await self._tcp_connect_scan()
            elif self.config.scan_type == ScanType.TCP_SYN:
                await self._tcp_syn_scan()
            elif self.config.scan_type == ScanType.UDP:
                await self._udp_scan()
            elif self.config.scan_type == ScanType.SERVICE_DETECTION:
                await self._service_detection_scan()
            
            self.scan_end_time = time.time()
            
            return {
                "success": True,
                "target": self.config.target_host,
                "scan_type": self.config.scan_type.value,
                "total_ports": len(self.results),
                "open_ports": len([r for r in self.results if r.status == PortStatus.OPEN]),
                "scan_duration": self.scan_end_time - self.scan_start_time,
                "results": [self._port_result_to_dict(r) for r in self.results]
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "target": self.config.target_host
            }
    
    async def _tcp_connect_scan(self) -> None:
        """Perform TCP connect scan"""
        ports = range(self.config.start_port, self.config.end_port + 1)
        
        # Use ThreadPoolExecutor for concurrent scanning
        with ThreadPoolExecutor(max_workers=self.config.max_concurrent) as executor:
            loop = asyncio.get_event_loop()
            tasks = [
                loop.run_in_executor(executor, self._check_tcp_port, port)
                for port in ports
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.results.append(PortResult(
                        port=ports[i],
                        status=PortStatus.FILTERED,
                        scan_time=0.0
                    ))
                else:
                    self.results.append(result)
    
    def _check_tcp_port(self, port: int) -> PortResult:
        """Check if a TCP port is open"""
        start_time = time.time()
        
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.config.timeout)
            
            result = sock.connect_ex((self.config.target_host, port))
            response_time = time.time() - start_time
            
            if result == 0:
                status = PortStatus.OPEN
                service = self._get_service_name(port)
                banner = self._grab_banner(sock) if self.config.banner_grab else None
            else:
                status = PortStatus.CLOSED
                service = None
                banner = None
            
            sock.close()
            
            return PortResult(
                port=port,
                status=status,
                service=service,
                banner=banner,
                response_time=response_time,
                scan_time=time.time() - start_time
            )
            
        except Exception:
            return PortResult(
                port=port,
                status=PortStatus.FILTERED,
                scan_time=time.time() - start_time
            )
    
    async def _tcp_syn_scan(self) -> None:
        """Perform TCP SYN scan (requires root privileges)"""
        try:
            nm = nmap.PortScanner()
            
            # Perform SYN scan
            nm.scan(
                self.config.target_host,
                f"{self.config.start_port}-{self.config.end_port}",
                arguments="-sS -T4"
            )
            
            # Process results
            for host in nm.all_hosts():
                for proto in nm[host].all_protocols():
                    ports = nm[host][proto].keys()
                    for port in ports:
                        port_info = nm[host][proto][port]
                        
                        status = PortStatus.OPEN if port_info['state'] == 'open' else PortStatus.CLOSED
                        service = port_info.get('name', None)
                        version = port_info.get('version', None)
                        
                        self.results.append(PortResult(
                            port=port,
                            status=status,
                            service=service,
                            version=version,
                            scan_time=0.0
                        ))
                        
        except Exception as e:
            raise Exception(f"SYN scan failed: {str(e)}")
    
    async def _udp_scan(self) -> None:
        """Perform UDP scan"""
        ports = range(self.config.start_port, self.config.end_port + 1)
        
        with ThreadPoolExecutor(max_workers=self.config.max_concurrent) as executor:
            loop = asyncio.get_event_loop()
            tasks = [
                loop.run_in_executor(executor, self._check_udp_port, port)
                for port in ports
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.results.append(PortResult(
                        port=ports[i],
                        status=PortStatus.FILTERED,
                        scan_time=0.0
                    ))
                else:
                    self.results.append(result)
    
    def _check_udp_port(self, port: int) -> PortResult:
        """Check if a UDP port is open"""
        start_time = time.time()
        
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(self.config.timeout)
            
            # Send empty packet
            sock.sendto(b"", (self.config.target_host, port))
            
            try:
                data, addr = sock.recvfrom(1024)
                status = PortStatus.OPEN
                service = self._get_service_name(port)
            except socket.timeout:
                status = PortStatus.OPEN  # UDP ports that don't respond might be open
                service = None
            
            sock.close()
            
            return PortResult(
                port=port,
                status=status,
                service=service,
                scan_time=time.time() - start_time
            )
            
        except Exception:
            return PortResult(
                port=port,
                status=PortStatus.FILTERED,
                scan_time=time.time() - start_time
            )
    
    async def _service_detection_scan(self) -> None:
        """Perform service detection scan"""
        # First do a quick TCP connect scan
        await self._tcp_connect_scan()
        
        # Then enhance with service detection for open ports
        open_ports = [r for r in self.results if r.status == PortStatus.OPEN]
        
        for port_result in open_ports:
            service_info = await self._detect_service(port_result.port)
            if service_info:
                port_result.service = service_info.get("service")
                port_result.version = service_info.get("version")
                port_result.banner = service_info.get("banner")
    
    async def _detect_service(self, port: int) -> Optional[Dict[str, str]]:
        """Detect service running on port"""
        try:
            # Common service detection patterns
            service_patterns = {
                21: {"service": "ftp", "banner": "FTP server"},
                22: {"service": "ssh", "banner": "SSH server"},
                23: {"service": "telnet", "banner": "Telnet server"},
                25: {"service": "smtp", "banner": "SMTP server"},
                53: {"service": "dns", "banner": "DNS server"},
                80: {"service": "http", "banner": "HTTP server"},
                110: {"service": "pop3", "banner": "POP3 server"},
                143: {"service": "imap", "banner": "IMAP server"},
                443: {"service": "https", "banner": "HTTPS server"},
                3306: {"service": "mysql", "banner": "MySQL server"},
                5432: {"service": "postgresql", "banner": "PostgreSQL server"},
                6379: {"service": "redis", "banner": "Redis server"},
                8080: {"service": "http-proxy", "banner": "HTTP proxy"}
            }
            
            if port in service_patterns:
                return service_patterns[port]
            
            # Try to grab banner
            banner = await self._grab_banner_async(port)
            if banner:
                return {"service": "unknown", "banner": banner}
            
            return None
            
        except Exception:
            return None
    
    def _grab_banner(self, sock: socket.socket) -> Optional[str]:
        """Grab banner from socket"""
        try:
            sock.send(b"\r\n")
            banner = sock.recv(1024).decode('utf-8', errors='ignore').strip()
            return banner if banner else None
        except Exception:
            return None
    
    async def _grab_banner_async(self, port: int) -> Optional[str]:
        """Grab banner asynchronously"""
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(self.config.target_host, port),
                timeout=self.config.timeout
            )
            
            writer.write(b"\r\n")
            await writer.drain()
            
            banner = await asyncio.wait_for(
                reader.read(1024),
                timeout=self.config.timeout
            )
            
            writer.close()
            await writer.wait_closed()
            
            return banner.decode('utf-8', errors='ignore').strip()
            
        except Exception:
            return None
    
    def _get_service_name(self, port: int) -> Optional[str]:
        """Get service name for common ports"""
        common_services = {
            21: "ftp", 22: "ssh", 23: "telnet", 25: "smtp", 53: "dns",
            80: "http", 110: "pop3", 143: "imap", 443: "https",
            3306: "mysql", 5432: "postgresql", 6379: "redis", 8080: "http-proxy"
        }
        return common_services.get(port)
    
    def _port_result_to_dict(self, result: PortResult) -> Dict[str, Any]:
        """Convert PortResult to dictionary"""
        return {
            "port": result.port,
            "status": result.status.value,
            "service": result.service,
            "version": result.version,
            "banner": result.banner,
            "response_time": result.response_time,
            "scan_time": result.scan_time
        }
    
    def get_open_ports(self) -> List[int]:
        """Get list of open ports"""
        return [r.port for r in self.results if r.status == PortStatus.OPEN]
    
    def get_services(self) -> Dict[int, str]:
        """Get dictionary of port -> service mappings"""
        return {r.port: r.service for r in self.results if r.service}
    
    def generate_report(self) -> str:
        """Generate text report of scan results"""
        report = f"Port Scan Report for {self.config.target_host}\n"
        report += "=" * 50 + "\n"
        report += f"Scan Type: {self.config.scan_type.value}\n"
        report += f"Scan Duration: {self.scan_end_time - self.scan_start_time:.2f} seconds\n"
        report += f"Total Ports Scanned: {len(self.results)}\n"
        report += f"Open Ports: {len(self.get_open_ports())}\n\n"
        
        report += "Open Ports:\n"
        report += "-" * 20 + "\n"
        
        for result in self.results:
            if result.status == PortStatus.OPEN:
                report += f"Port {result.port}: {result.service or 'unknown'}"
                if result.version:
                    report += f" ({result.version})"
                if result.banner:
                    report += f" - {result.banner[:50]}..."
                report += "\n"
        
        return report

# Example usage
async def main():
    """Example usage of port scanner"""
    print("🔍 Port Scanner Example")
    
    # Create scan configuration
    config = ScanConfig(
        target_host="localhost",
        start_port=1,
        end_port=1000,
        scan_type=ScanType.TCP_CONNECT,
        timeout=2.0,
        max_concurrent=50,
        service_detection=True,
        banner_grab=True
    )
    
    # Create scanner
    scanner = PortScanner(config)
    
    # Perform scan
    print(f"Scanning {config.target_host} ports {config.start_port}-{config.end_port}...")
    result = await scanner.scan_ports()
    
    if result["success"]:
        print(f"✅ Scan completed in {result['scan_duration']:.2f} seconds")
        print(f"📊 Found {result['open_ports']} open ports out of {result['total_ports']} scanned")
        
        # Print open ports
        print("\n🔓 Open Ports:")
        for port_result in result["results"]:
            if port_result["status"] == "open":
                print(f"  Port {port_result['port']}: {port_result['service'] or 'unknown'}")
                if port_result['banner']:
                    print(f"    Banner: {port_result['banner']}")
        
        # Generate report
        print("\n📋 Scan Report:")
        print(scanner.generate_report())
        
    else:
        print(f"❌ Scan failed: {result['error']}")

if __name__ == "__main__":
    asyncio.run(main()) 