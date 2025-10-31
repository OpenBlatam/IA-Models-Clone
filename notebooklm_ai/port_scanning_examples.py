from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import threading
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import structlog
    import nmap
    from libnmap.parser import NmapParser
    from libnmap.process import NmapProcess
    from libnmap.objects import NmapHost, NmapService
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
Port Scanning Examples using python-nmap and libnmap
===================================================

This module demonstrates advanced port scanning techniques using:
- python-nmap: Python interface to nmap
- libnmap: Advanced nmap library for Python
- Custom scanning strategies and analysis

Features:
- TCP/UDP port scanning
- Service detection and version identification
- OS fingerprinting
- Vulnerability scanning integration
- Scan result analysis and reporting
"""


# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

# Try to import python-nmap
try:
    NMAP_AVAILABLE = True
except ImportError:
    NMAP_AVAILABLE = False
    print("python-nmap not available. Install with: pip install python-nmap")

# Try to import libnmap
try:
    LIBNMAP_AVAILABLE = True
except ImportError:
    LIBNMAP_AVAILABLE = False
    print("libnmap not available. Install with: pip install python-libnmap")

# Custom exception classes
class PortScanError(Exception):
    """Base exception for port scanning operations."""
    pass

class NmapNotAvailableError(PortScanError):
    """Raised when nmap is not available."""
    pass

class ScanConfigurationError(PortScanError):
    """Raised when scan configuration is invalid."""
    pass

class ScanExecutionError(PortScanError):
    """Raised when scan execution fails."""
    pass

@dataclass
class PortInfo:
    """Information about a scanned port."""
    port: int
    protocol: str
    state: str
    service: Optional[str] = None
    version: Optional[str] = None
    product: Optional[str] = None
    extra_info: Optional[str] = None
    banner: Optional[str] = None
    script_output: Optional[Dict[str, str]] = None

@dataclass
class HostInfo:
    """Information about a scanned host."""
    ip_address: str
    hostname: Optional[str] = None
    os_info: Optional[Dict[str, str]] = None
    ports: List[PortInfo] = None
    status: str = "unknown"
    scan_time: Optional[datetime] = None
    mac_address: Optional[str] = None
    vendor: Optional[str] = None

@dataclass
class ScanResult:
    """Complete scan result."""
    target: str
    scan_type: str
    start_time: datetime
    end_time: Optional[datetime] = None
    hosts: List[HostInfo] = None
    scan_options: Dict[str, Any] = None
    is_successful: bool = False
    error_message: Optional[str] = None

class NmapScanner:
    """Advanced port scanner using python-nmap."""
    
    def __init__(self, nmap_path: Optional[str] = None):
        
    """__init__ function."""
if not NMAP_AVAILABLE:
            raise NmapNotAvailableError("python-nmap is required for scanning")
        
        self.nmap_scanner = nmap.PortScanner()
        self.nmap_path = nmap_path
        self.scan_history = []
        
        # Test nmap availability
        try:
            self.nmap_scanner.scan('127.0.0.1', arguments='-sn')
        except Exception as e:
            raise NmapNotAvailableError(f"Nmap not available or not working: {e}")
    
    def quick_scan(self, target: str, ports: str = "1-1000") -> ScanResult:
        """Perform a quick port scan."""
        # Guard clauses - all error conditions first
        if not target or not isinstance(target, str):
            raise ScanConfigurationError("Target must be a non-empty string")
        
        if not ports or not isinstance(ports, str):
            raise ScanConfigurationError("Ports must be a non-empty string")
        
        # Happy path - main scanning logic
        try:
            start_time = datetime.now()
            
            logger.info("Starting quick scan",
                       module="nmap",
                       function="quick_scan",
                       target=target,
                       ports=ports)
            
            # Perform the scan
            scan_result = self.nmap_scanner.scan(
                hosts=target,
                ports=ports,
                arguments='-sS -sV --version-intensity 2'
            )
            
            end_time = datetime.now()
            
            # Parse results
            hosts = self._parse_nmap_results(scan_result)
            
            result = ScanResult(
                target=target,
                scan_type="quick_scan",
                start_time=start_time,
                end_time=end_time,
                hosts=hosts,
                scan_options={"ports": ports, "arguments": "-sS -sV --version-intensity 2"},
                is_successful=True
            )
            
            self.scan_history.append(result)
            
            logger.info("Quick scan completed successfully",
                       module="nmap",
                       function="quick_scan",
                       target=target,
                       hosts_found=len(hosts),
                       duration_seconds=(end_time - start_time).total_seconds())
            
            return result
            
        except Exception as e:
            logger.error("Quick scan failed",
                        module="nmap",
                        function="quick_scan",
                        target=target,
                        error=str(e))
            raise ScanExecutionError(f"Quick scan failed: {str(e)}")
    
    def comprehensive_scan(self, target: str, ports: str = "1-65535") -> ScanResult:
        """Perform a comprehensive port scan with service detection."""
        # Guard clauses - all error conditions first
        if not target or not isinstance(target, str):
            raise ScanConfigurationError("Target must be a non-empty string")
        
        if not ports or not isinstance(ports, str):
            raise ScanConfigurationError("Ports must be a non-empty string")
        
        # Happy path - main scanning logic
        try:
            start_time = datetime.now()
            
            logger.info("Starting comprehensive scan",
                       module="nmap",
                       function="comprehensive_scan",
                       target=target,
                       ports=ports)
            
            # Perform the scan with comprehensive options
            scan_result = self.nmap_scanner.scan(
                hosts=target,
                ports=ports,
                arguments='-sS -sV -sC -A -O --version-all'
            )
            
            end_time = datetime.now()
            
            # Parse results
            hosts = self._parse_nmap_results(scan_result)
            
            result = ScanResult(
                target=target,
                scan_type="comprehensive_scan",
                start_time=start_time,
                end_time=end_time,
                hosts=hosts,
                scan_options={"ports": ports, "arguments": "-sS -sV -sC -A -O --version-all"},
                is_successful=True
            )
            
            self.scan_history.append(result)
            
            logger.info("Comprehensive scan completed successfully",
                       module="nmap",
                       function="comprehensive_scan",
                       target=target,
                       hosts_found=len(hosts),
                       duration_seconds=(end_time - start_time).total_seconds())
            
            return result
            
        except Exception as e:
            logger.error("Comprehensive scan failed",
                        module="nmap",
                        function="comprehensive_scan",
                        target=target,
                        error=str(e))
            raise ScanExecutionError(f"Comprehensive scan failed: {str(e)}")
    
    def udp_scan(self, target: str, ports: str = "53,67,68,69,123,161,162,389,636") -> ScanResult:
        """Perform UDP port scanning."""
        # Guard clauses - all error conditions first
        if not target or not isinstance(target, str):
            raise ScanConfigurationError("Target must be a non-empty string")
        
        if not ports or not isinstance(ports, str):
            raise ScanConfigurationError("Ports must be a non-empty string")
        
        # Happy path - main scanning logic
        try:
            start_time = datetime.now()
            
            logger.info("Starting UDP scan",
                       module="nmap",
                       function="udp_scan",
                       target=target,
                       ports=ports)
            
            # Perform UDP scan
            scan_result = self.nmap_scanner.scan(
                hosts=target,
                ports=ports,
                arguments='-sU -sV --version-intensity 1'
            )
            
            end_time = datetime.now()
            
            # Parse results
            hosts = self._parse_nmap_results(scan_result)
            
            result = ScanResult(
                target=target,
                scan_type="udp_scan",
                start_time=start_time,
                end_time=end_time,
                hosts=hosts,
                scan_options={"ports": ports, "arguments": "-sU -sV --version-intensity 1"},
                is_successful=True
            )
            
            self.scan_history.append(result)
            
            logger.info("UDP scan completed successfully",
                       module="nmap",
                       function="udp_scan",
                       target=target,
                       hosts_found=len(hosts),
                       duration_seconds=(end_time - start_time).total_seconds())
            
            return result
            
        except Exception as e:
            logger.error("UDP scan failed",
                        module="nmap",
                        function="udp_scan",
                        target=target,
                        error=str(e))
            raise ScanExecutionError(f"UDP scan failed: {str(e)}")
    
    def os_detection_scan(self, target: str) -> ScanResult:
        """Perform OS detection scan."""
        # Guard clauses - all error conditions first
        if not target or not isinstance(target, str):
            raise ScanConfigurationError("Target must be a non-empty string")
        
        # Happy path - main scanning logic
        try:
            start_time = datetime.now()
            
            logger.info("Starting OS detection scan",
                       module="nmap",
                       function="os_detection_scan",
                       target=target)
            
            # Perform OS detection scan
            scan_result = self.nmap_scanner.scan(
                hosts=target,
                arguments='-O --osscan-guess'
            )
            
            end_time = datetime.now()
            
            # Parse results
            hosts = self._parse_nmap_results(scan_result)
            
            result = ScanResult(
                target=target,
                scan_type="os_detection_scan",
                start_time=start_time,
                end_time=end_time,
                hosts=hosts,
                scan_options={"arguments": "-O --osscan-guess"},
                is_successful=True
            )
            
            self.scan_history.append(result)
            
            logger.info("OS detection scan completed successfully",
                       module="nmap",
                       function="os_detection_scan",
                       target=target,
                       hosts_found=len(hosts),
                       duration_seconds=(end_time - start_time).total_seconds())
            
            return result
            
        except Exception as e:
            logger.error("OS detection scan failed",
                        module="nmap",
                        function="os_detection_scan",
                        target=target,
                        error=str(e))
            raise ScanExecutionError(f"OS detection scan failed: {str(e)}")
    
    def _parse_nmap_results(self, scan_result: Dict) -> List[HostInfo]:
        """Parse nmap scan results into structured data."""
        hosts = []
        
        try:
            for host in scan_result['scan']:
                host_data = scan_result['scan'][host]
                
                # Extract basic host information
                host_info = HostInfo(
                    ip_address=host,
                    hostname=host_data.get('hostnames', [{}])[0].get('name'),
                    status=host_data.get('status', {}).get('state', 'unknown'),
                    scan_time=datetime.now(),
                    ports=[]
                )
                
                # Extract OS information
                if 'osmatch' in host_data and host_data['osmatch']:
                    os_match = host_data['osmatch'][0]
                    host_info.os_info = {
                        'name': os_match.get('name', 'Unknown'),
                        'accuracy': os_match.get('accuracy', '0'),
                        'line': os_match.get('line', 'Unknown')
                    }
                
                # Extract MAC address and vendor
                if 'addresses' in host_data:
                    addresses = host_data['addresses']
                    if 'mac' in addresses:
                        host_info.mac_address = addresses['mac']
                        host_info.vendor = addresses.get('vendor', 'Unknown')
                
                # Extract port information
                if 'tcp' in host_data:
                    for port, port_data in host_data['tcp'].items():
                        port_info = PortInfo(
                            port=int(port),
                            protocol='tcp',
                            state=port_data.get('state', 'unknown'),
                            service=port_data.get('name', 'unknown'),
                            version=port_data.get('version', ''),
                            product=port_data.get('product', ''),
                            extra_info=port_data.get('extrainfo', ''),
                            banner=port_data.get('banner', ''),
                            script_output=port_data.get('script', {})
                        )
                        host_info.ports.append(port_info)
                
                if 'udp' in host_data:
                    for port, port_data in host_data['udp'].items():
                        port_info = PortInfo(
                            port=int(port),
                            protocol='udp',
                            state=port_data.get('state', 'unknown'),
                            service=port_data.get('name', 'unknown'),
                            version=port_data.get('version', ''),
                            product=port_data.get('product', ''),
                            extra_info=port_data.get('extrainfo', ''),
                            banner=port_data.get('banner', ''),
                            script_output=port_data.get('script', {})
                        )
                        host_info.ports.append(port_info)
                
                hosts.append(host_info)
                
        except Exception as e:
            logger.error("Failed to parse nmap results",
                        module="nmap",
                        function="_parse_nmap_results",
                        error=str(e))
        
        return hosts

class LibNmapScanner:
    """Advanced port scanner using libnmap."""
    
    def __init__(self) -> Any:
        if not LIBNMAP_AVAILABLE:
            raise NmapNotAvailableError("libnmap is required for scanning")
        
        self.scan_history = []
    
    def async_scan(self, targets: str, options: str = "-sS -sV") -> ScanResult:
        """Perform asynchronous nmap scan using libnmap."""
        # Guard clauses - all error conditions first
        if not targets or not isinstance(targets, str):
            raise ScanConfigurationError("Targets must be a non-empty string")
        
        if not options or not isinstance(options, str):
            raise ScanConfigurationError("Options must be a non-empty string")
        
        # Happy path - main scanning logic
        try:
            start_time = datetime.now()
            
            logger.info("Starting async scan",
                       module="libnmap",
                       function="async_scan",
                       targets=targets,
                       options=options)
            
            # Create and run nmap process
            nmap_proc = NmapProcess(targets, options)
            nmap_proc.run()
            
            end_time = datetime.now()
            
            # Parse results
            if nmap_proc.rc == 0:
                parsed_results = NmapParser.parse(nmap_proc.stdout)
                hosts = self._parse_libnmap_results(parsed_results)
                
                result = ScanResult(
                    target=targets,
                    scan_type="async_scan",
                    start_time=start_time,
                    end_time=end_time,
                    hosts=hosts,
                    scan_options={"options": options},
                    is_successful=True
                )
                
                self.scan_history.append(result)
                
                logger.info("Async scan completed successfully",
                           module="libnmap",
                           function="async_scan",
                           targets=targets,
                           hosts_found=len(hosts),
                           duration_seconds=(end_time - start_time).total_seconds())
                
                return result
            else:
                error_msg = f"Nmap process failed with return code {nmap_proc.rc}"
                logger.error("Async scan failed",
                            module="libnmap",
                            function="async_scan",
                            targets=targets,
                            error=error_msg)
                raise ScanExecutionError(error_msg)
                
        except Exception as e:
            logger.error("Async scan failed",
                        module="libnmap",
                        function="async_scan",
                        targets=targets,
                        error=str(e))
            raise ScanExecutionError(f"Async scan failed: {str(e)}")
    
    def _parse_libnmap_results(self, parsed_results) -> List[HostInfo]:
        """Parse libnmap scan results into structured data."""
        hosts = []
        
        try:
            for host in parsed_results.hosts:
                host_info = HostInfo(
                    ip_address=host.address,
                    hostname=host.hostnames[0] if host.hostnames else None,
                    status=host.status,
                    scan_time=datetime.now(),
                    ports=[]
                )
                
                # Extract OS information
                if host.os_osmatch:
                    os_match = host.os_osmatch[0]
                    host_info.os_info = {
                        'name': os_match.name,
                        'accuracy': str(os_match.accuracy),
                        'line': os_match.line
                    }
                
                # Extract MAC address and vendor
                if host.mac:
                    host_info.mac_address = host.mac
                    host_info.vendor = host.vendor
                
                # Extract port information
                for service in host.services:
                    port_info = PortInfo(
                        port=service.port,
                        protocol=service.protocol,
                        state=service.state,
                        service=service.service,
                        version=service.servicefp,
                        product=service.service_dict.get('product', ''),
                        extra_info=service.service_dict.get('extrainfo', ''),
                        banner=service.banner,
                        script_output=service.scripts if hasattr(service, 'scripts') else {}
                    )
                    host_info.ports.append(port_info)
                
                hosts.append(host_info)
                
        except Exception as e:
            logger.error("Failed to parse libnmap results",
                        module="libnmap",
                        function="_parse_libnmap_results",
                        error=str(e))
        
        return hosts

class ScanAnalyzer:
    """Analyze and report on scan results."""
    
    @staticmethod
    def analyze_scan_result(scan_result: ScanResult) -> Dict[str, Any]:
        """Analyze a scan result and generate statistics."""
        # Guard clauses - all error conditions first
        if not scan_result or not scan_result.is_successful:
            return {"error": "Invalid or failed scan result"}
        
        # Happy path - main analysis logic
        try:
            analysis = {
                "target": scan_result.target,
                "scan_type": scan_result.scan_type,
                "scan_duration": None,
                "total_hosts": len(scan_result.hosts),
                "hosts_up": 0,
                "hosts_down": 0,
                "total_ports": 0,
                "open_ports": 0,
                "closed_ports": 0,
                "filtered_ports": 0,
                "services_found": {},
                "os_distribution": {},
                "vulnerability_indicators": [],
                "recommendations": []
            }
            
            # Calculate scan duration
            if scan_result.end_time:
                analysis["scan_duration"] = (scan_result.end_time - scan_result.start_time).total_seconds()
            
            # Analyze hosts
            for host in scan_result.hosts:
                if host.status == "up":
                    analysis["hosts_up"] += 1
                else:
                    analysis["hosts_down"] += 1
                
                # Analyze ports
                for port in host.ports:
                    analysis["total_ports"] += 1
                    
                    if port.state == "open":
                        analysis["open_ports"] += 1
                        
                        # Track services
                        service_name = port.service or "unknown"
                        if service_name not in analysis["services_found"]:
                            analysis["services_found"][service_name] = 0
                        analysis["services_found"][service_name] += 1
                        
                        # Check for common vulnerable services
                        if service_name in ["telnet", "ftp", "rsh", "rlogin"]:
                            analysis["vulnerability_indicators"].append(
                                f"Insecure service {service_name} on port {port.port}"
                            )
                        
                        # Check for default credentials services
                        if service_name in ["ssh", "ftp", "telnet", "http", "https"]:
                            analysis["recommendations"].append(
                                f"Check {service_name} on port {port.port} for default credentials"
                            )
                    
                    elif port.state == "closed":
                        analysis["closed_ports"] += 1
                    elif port.state == "filtered":
                        analysis["filtered_ports"] += 1
                
                # Track OS distribution
                if host.os_info:
                    os_name = host.os_info.get('name', 'Unknown')
                    if os_name not in analysis["os_distribution"]:
                        analysis["os_distribution"][os_name] = 0
                    analysis["os_distribution"][os_name] += 1
            
            # Generate recommendations
            if analysis["open_ports"] > 10:
                analysis["recommendations"].append("Consider closing unnecessary open ports")
            
            if not analysis["vulnerability_indicators"]:
                analysis["recommendations"].append("No obvious security issues detected")
            
            return analysis
            
        except Exception as e:
            logger.error("Failed to analyze scan result",
                        module="analyzer",
                        function="analyze_scan_result",
                        error=str(e))
            return {"error": f"Analysis failed: {str(e)}"}
    
    @staticmethod
    def generate_report(scan_result: ScanResult, output_file: Optional[str] = None) -> str:
        """Generate a detailed scan report."""
        # Guard clauses - all error conditions first
        if not scan_result or not scan_result.is_successful:
            return "Error: Invalid or failed scan result"
        
        # Happy path - main report generation logic
        try:
            analysis = ScanAnalyzer.analyze_scan_result(scan_result)
            
            report_lines = [
                "=" * 60,
                "NETWORK SCAN REPORT",
                "=" * 60,
                f"Target: {scan_result.target}",
                f"Scan Type: {scan_result.scan_type}",
                f"Scan Date: {scan_result.start_time.strftime('%Y-%m-%d %H:%M:%S')}",
                f"Duration: {analysis.get('scan_duration', 'Unknown'):.2f} seconds",
                "",
                "SUMMARY",
                "-" * 20,
                f"Total Hosts: {analysis['total_hosts']}",
                f"Hosts Up: {analysis['hosts_up']}",
                f"Hosts Down: {analysis['hosts_down']}",
                f"Total Ports: {analysis['total_ports']}",
                f"Open Ports: {analysis['open_ports']}",
                f"Closed Ports: {analysis['closed_ports']}",
                f"Filtered Ports: {analysis['filtered_ports']}",
                ""
            ]
            
            # Add services found
            if analysis['services_found']:
                report_lines.extend([
                    "SERVICES FOUND",
                    "-" * 20
                ])
                for service, count in sorted(analysis['services_found'].items(), 
                                           key=lambda x: x[1], reverse=True):
                    report_lines.append(f"{service}: {count}")
                report_lines.append("")
            
            # Add OS distribution
            if analysis['os_distribution']:
                report_lines.extend([
                    "OPERATING SYSTEMS",
                    "-" * 20
                ])
                for os_name, count in analysis['os_distribution'].items():
                    report_lines.append(f"{os_name}: {count}")
                report_lines.append("")
            
            # Add vulnerability indicators
            if analysis['vulnerability_indicators']:
                report_lines.extend([
                    "SECURITY ISSUES",
                    "-" * 20
                ])
                for issue in analysis['vulnerability_indicators']:
                    report_lines.append(f"• {issue}")
                report_lines.append("")
            
            # Add recommendations
            if analysis['recommendations']:
                report_lines.extend([
                    "RECOMMENDATIONS",
                    "-" * 20
                ])
                for rec in analysis['recommendations']:
                    report_lines.append(f"• {rec}")
                report_lines.append("")
            
            # Add detailed host information
            report_lines.extend([
                "DETAILED RESULTS",
                "-" * 20
            ])
            
            for host in scan_result.hosts:
                report_lines.append(f"Host: {host.ip_address}")
                if host.hostname:
                    report_lines.append(f"  Hostname: {host.hostname}")
                if host.os_info:
                    report_lines.append(f"  OS: {host.os_info.get('name', 'Unknown')}")
                if host.mac_address:
                    report_lines.append(f"  MAC: {host.mac_address}")
                
                if host.ports:
                    report_lines.append("  Open Ports:")
                    for port in host.ports:
                        if port.state == "open":
                            service_info = f"{port.service}"
                            if port.product:
                                service_info += f" ({port.product})"
                            if port.version:
                                service_info += f" {port.version}"
                            report_lines.append(f"    {port.port}/{port.protocol}: {service_info}")
                
                report_lines.append("")
            
            report = "\n".join(report_lines)
            
            # Save to file if specified
            if output_file:
                with open(output_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    f.write(report)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                logger.info("Scan report saved",
                           module="analyzer",
                           function="generate_report",
                           output_file=output_file)
            
            return report
            
        except Exception as e:
            logger.error("Failed to generate scan report",
                        module="analyzer",
                        function="generate_report",
                        error=str(e))
            return f"Error generating report: {str(e)}"

# Example usage and demonstration functions
def demonstrate_nmap_scanning():
    """Demonstrate nmap scanning capabilities."""
    if not NMAP_AVAILABLE:
        print("python-nmap not available. Install with: pip install python-nmap")
        return
    
    try:
        # Initialize scanner
        scanner = NmapScanner()
        
        print("=== Nmap Scanning Demonstration ===")
        
        # Quick scan
        print("Performing quick scan on localhost...")
        result = scanner.quick_scan("127.0.0.1", "22,80,443,8080")
        
        # Analyze results
        analyzer = ScanAnalyzer()
        analysis = analyzer.analyze_scan_result(result)
        
        print(f"Scan completed: {analysis['hosts_up']} hosts up, {analysis['open_ports']} open ports")
        
        # Generate report
        report = analyzer.generate_report(result)
        print("\n" + report)
        
    except Exception as e:
        print(f"Nmap scanning demonstration failed: {e}")

def demonstrate_libnmap_scanning():
    """Demonstrate libnmap scanning capabilities."""
    if not LIBNMAP_AVAILABLE:
        print("libnmap not available. Install with: pip install python-libnmap")
        return
    
    try:
        # Initialize scanner
        scanner = LibNmapScanner()
        
        print("=== LibNmap Scanning Demonstration ===")
        
        # Async scan
        print("Performing async scan on localhost...")
        result = scanner.async_scan("127.0.0.1", "-sS -sV -p 22,80,443")
        
        # Analyze results
        analyzer = ScanAnalyzer()
        analysis = analyzer.analyze_scan_result(result)
        
        print(f"Scan completed: {analysis['hosts_up']} hosts up, {analysis['open_ports']} open ports")
        
        # Generate report
        report = analyzer.generate_report(result)
        print("\n" + report)
        
    except Exception as e:
        print(f"Libnmap scanning demonstration failed: {e}")

if __name__ == "__main__":
    # Run demonstrations
    demonstrate_nmap_scanning()
    print()
    demonstrate_libnmap_scanning() 