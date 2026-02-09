#!/usr/bin/env python3
"""
Network Scanner for Video-OpusClip
Network scanning, port scanning, and service detection using Scapy
"""

import socket
import threading
import time
from typing import Dict, List, Any, Optional, Union, Tuple
from enum import Enum
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

from scapy.all import (
    sr, sr1, IP, TCP, UDP, ICMP, ARP, Ether, srp, conf,
    RandShort, RandInt, RandIP, get_if_addr, get_if_hwaddr
)

logger = logging.getLogger(__name__)


class ScanType(Enum):
    """Scan types"""
    PING = "ping"
    PORT = "port"
    SERVICE = "service"
    VULNERABILITY = "vulnerability"
    NETWORK = "network"


class PortState(Enum):
    """Port states"""
    OPEN = "open"
    CLOSED = "closed"
    FILTERED = "filtered"
    UNFILTERED = "unfiltered"
    OPEN_FILTERED = "open_filtered"
    CLOSED_FILTERED = "closed_filtered"


class ServiceType(Enum):
    """Service types"""
    HTTP = "http"
    HTTPS = "https"
    SSH = "ssh"
    FTP = "ftp"
    SMTP = "smtp"
    DNS = "dns"
    TELNET = "telnet"
    POP3 = "pop3"
    IMAP = "imap"
    SNMP = "snmp"
    UNKNOWN = "unknown"


@dataclass
class ScanResult:
    """Scan result"""
    target: str
    scan_type: ScanType
    ports: Dict[int, PortState] = field(default_factory=dict)
    services: Dict[int, ServiceType] = field(default_factory=dict)
    vulnerabilities: List[str] = field(default_factory=list)
    hosts: List[str] = field(default_factory=list)
    duration: float = 0.0
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ServiceInfo:
    """Service information"""
    port: int
    service: ServiceType
    version: Optional[str] = None
    banner: Optional[str] = None
    state: PortState = PortState.OPEN
    protocol: str = "tcp"


@dataclass
class VulnerabilityInfo:
    """Vulnerability information"""
    port: int
    service: ServiceType
    vulnerability: str
    severity: str = "medium"
    description: Optional[str] = None
    cve: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class NetworkScanner:
    """Network scanner base class"""
    
    def __init__(self, timeout: int = 3, verbose: bool = False):
        self.timeout = timeout
        self.verbose = verbose
        self.results: List[ScanResult] = []
    
    def scan_network(self, network: str) -> ScanResult:
        """Scan network for live hosts"""
        try:
            start_time = time.time()
            
            # Parse network
            if "/" not in network:
                network = f"{network}/24"  # Default to /24
            
            # Send ARP requests
            arp_packets = Ether(dst="ff:ff:ff:ff:ff:ff") / ARP(pdst=network)
            answered, _ = srp(arp_packets, timeout=self.timeout, verbose=self.verbose)
            
            # Extract live hosts
            hosts = [ans[1].psrc for ans in answered]
            
            duration = time.time() - start_time
            
            result = ScanResult(
                target=network,
                scan_type=ScanType.NETWORK,
                hosts=hosts,
                duration=duration,
                metadata={"answered_count": len(answered)}
            )
            
            self.results.append(result)
            return result
            
        except Exception as e:
            logger.error(f"Network scan failed: {e}")
            return ScanResult(
                target=network,
                scan_type=ScanType.NETWORK,
                success=False,
                error=str(e)
            )
    
    def ping_host(self, host: str) -> bool:
        """Ping a single host"""
        try:
            # Send ICMP echo request
            ping_packet = IP(dst=host) / ICMP()
            response = sr1(ping_packet, timeout=self.timeout, verbose=self.verbose)
            
            return response is not None
            
        except Exception as e:
            logger.error(f"Ping failed for {host}: {e}")
            return False
    
    def ping_network(self, network: str) -> List[str]:
        """Ping all hosts in network"""
        try:
            # Get all IPs in network
            ips = self._get_network_ips(network)
            
            # Ping each IP
            live_hosts = []
            with ThreadPoolExecutor(max_workers=50) as executor:
                future_to_ip = {executor.submit(self.ping_host, ip): ip for ip in ips}
                
                for future in as_completed(future_to_ip):
                    ip = future_to_ip[future]
                    try:
                        if future.result():
                            live_hosts.append(ip)
                    except Exception as e:
                        logger.error(f"Ping failed for {ip}: {e}")
            
            return live_hosts
            
        except Exception as e:
            logger.error(f"Network ping failed: {e}")
            return []
    
    def _get_network_ips(self, network: str) -> List[str]:
        """Get all IPs in network"""
        try:
            import ipaddress
            net = ipaddress.IPv4Network(network, strict=False)
            return [str(ip) for ip in net.hosts()]
        except Exception as e:
            logger.error(f"Failed to get network IPs: {e}")
            return []


class PortScanner(NetworkScanner):
    """Port scanner"""
    
    def __init__(self, timeout: int = 3, verbose: bool = False):
        super().__init__(timeout, verbose)
        self.common_ports = [21, 22, 23, 25, 53, 80, 110, 143, 443, 993, 995, 3306, 3389, 5432, 8080]
    
    def scan_ports(
        self,
        target: str,
        ports: Optional[List[int]] = None,
        scan_type: str = "tcp"
    ) -> ScanResult:
        """Scan ports on target"""
        try:
            start_time = time.time()
            
            if ports is None:
                ports = self.common_ports
            
            if scan_type.lower() == "tcp":
                return self._tcp_scan(target, ports)
            elif scan_type.lower() == "udp":
                return self._udp_scan(target, ports)
            else:
                raise ValueError(f"Unsupported scan type: {scan_type}")
                
        except Exception as e:
            logger.error(f"Port scan failed: {e}")
            return ScanResult(
                target=target,
                scan_type=ScanType.PORT,
                success=False,
                error=str(e)
            )
    
    def _tcp_scan(self, target: str, ports: List[int]) -> ScanResult:
        """TCP port scan"""
        try:
            start_time = time.time()
            
            # Create TCP packets
            tcp_packets = IP(dst=target) / TCP(dport=ports, flags="S")
            
            # Send packets
            answered, unanswered = sr(tcp_packets, timeout=self.timeout, verbose=self.verbose)
            
            # Analyze responses
            port_states = {}
            for packet in answered:
                port = packet[1].dport
                if packet[1].haslayer(TCP):
                    if packet[1][TCP].flags == 0x12:  # SYN-ACK
                        port_states[port] = PortState.OPEN
                    elif packet[1][TCP].flags == 0x14:  # RST-ACK
                        port_states[port] = PortState.CLOSED
            
            # Mark unanswered ports as filtered
            for packet in unanswered:
                port = packet[1].dport
                if port not in port_states:
                    port_states[port] = PortState.FILTERED
            
            duration = time.time() - start_time
            
            result = ScanResult(
                target=target,
                scan_type=ScanType.PORT,
                ports=port_states,
                duration=duration,
                metadata={"scan_type": "tcp", "total_ports": len(ports)}
            )
            
            self.results.append(result)
            return result
            
        except Exception as e:
            logger.error(f"TCP scan failed: {e}")
            raise
    
    def _udp_scan(self, target: str, ports: List[int]) -> ScanResult:
        """UDP port scan"""
        try:
            start_time = time.time()
            
            # Create UDP packets
            udp_packets = IP(dst=target) / UDP(dport=ports)
            
            # Send packets
            answered, unanswered = sr(udp_packets, timeout=self.timeout, verbose=self.verbose)
            
            # Analyze responses
            port_states = {}
            for packet in answered:
                port = packet[1].dport
                if packet[1].haslayer(UDP):
                    port_states[port] = PortState.OPEN
                elif packet[1].haslayer(ICMP):
                    icmp_type = packet[1][ICMP].type
                    if icmp_type == 3:  # Destination Unreachable
                        port_states[port] = PortState.CLOSED
            
            # Mark unanswered ports as open/filtered
            for packet in unanswered:
                port = packet[1].dport
                if port not in port_states:
                    port_states[port] = PortState.OPEN_FILTERED
            
            duration = time.time() - start_time
            
            result = ScanResult(
                target=target,
                scan_type=ScanType.PORT,
                ports=port_states,
                duration=duration,
                metadata={"scan_type": "udp", "total_ports": len(ports)}
            )
            
            self.results.append(result)
            return result
            
        except Exception as e:
            logger.error(f"UDP scan failed: {e}")
            raise
    
    def scan_port_range(
        self,
        target: str,
        start_port: int = 1,
        end_port: int = 1024,
        scan_type: str = "tcp"
    ) -> ScanResult:
        """Scan port range"""
        ports = list(range(start_port, end_port + 1))
        return self.scan_ports(target, ports, scan_type)
    
    def quick_scan(self, target: str) -> ScanResult:
        """Quick scan of common ports"""
        return self.scan_ports(target, self.common_ports, "tcp")
    
    def full_scan(self, target: str) -> ScanResult:
        """Full scan of all ports"""
        return self.scan_port_range(target, 1, 65535, "tcp")


class ServiceScanner(PortScanner):
    """Service scanner"""
    
    def __init__(self, timeout: int = 3, verbose: bool = False):
        super().__init__(timeout, verbose)
        self.service_signatures = {
            21: ServiceType.FTP,
            22: ServiceType.SSH,
            23: ServiceType.TELNET,
            25: ServiceType.SMTP,
            53: ServiceType.DNS,
            80: ServiceType.HTTP,
            110: ServiceType.POP3,
            143: ServiceType.IMAP,
            443: ServiceType.HTTPS,
            993: ServiceType.IMAP,
            995: ServiceType.POP3,
            3306: ServiceType.UNKNOWN,  # MySQL
            3389: ServiceType.UNKNOWN,  # RDP
            5432: ServiceType.UNKNOWN,  # PostgreSQL
            8080: ServiceType.HTTP
        }
    
    def scan_services(self, target: str, ports: Optional[List[int]] = None) -> ScanResult:
        """Scan services on target"""
        try:
            # First scan ports
            port_result = self.scan_ports(target, ports)
            
            if not port_result.success:
                return port_result
            
            # Then identify services
            services = {}
            for port, state in port_result.ports.items():
                if state == PortState.OPEN:
                    service = self._identify_service(target, port)
                    services[port] = service
            
            # Update result
            port_result.services = services
            port_result.scan_type = ScanType.SERVICE
            
            return port_result
            
        except Exception as e:
            logger.error(f"Service scan failed: {e}")
            return ScanResult(
                target=target,
                scan_type=ScanType.SERVICE,
                success=False,
                error=str(e)
            )
    
    def _identify_service(self, target: str, port: int) -> ServiceType:
        """Identify service on port"""
        try:
            # Check known port mappings
            if port in self.service_signatures:
                return self.service_signatures[port]
            
            # Try to get service banner
            banner = self._get_service_banner(target, port)
            if banner:
                return self._identify_service_from_banner(banner)
            
            return ServiceType.UNKNOWN
            
        except Exception as e:
            logger.error(f"Service identification failed for {target}:{port}: {e}")
            return ServiceType.UNKNOWN
    
    def _get_service_banner(self, target: str, port: int) -> Optional[str]:
        """Get service banner"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.timeout)
            sock.connect((target, port))
            
            # Send probe
            sock.send(b"HEAD / HTTP/1.0\r\n\r\n")
            
            # Receive response
            response = sock.recv(1024).decode('utf-8', errors='ignore')
            sock.close()
            
            return response
            
        except Exception:
            return None
    
    def _identify_service_from_banner(self, banner: str) -> ServiceType:
        """Identify service from banner"""
        banner_lower = banner.lower()
        
        if "http" in banner_lower:
            if "https" in banner_lower:
                return ServiceType.HTTPS
            return ServiceType.HTTP
        elif "ssh" in banner_lower:
            return ServiceType.SSH
        elif "ftp" in banner_lower:
            return ServiceType.FTP
        elif "smtp" in banner_lower:
            return ServiceType.SMTP
        elif "dns" in banner_lower:
            return ServiceType.DNS
        elif "pop3" in banner_lower:
            return ServiceType.POP3
        elif "imap" in banner_lower:
            return ServiceType.IMAP
        elif "telnet" in banner_lower:
            return ServiceType.TELNET
        elif "snmp" in banner_lower:
            return ServiceType.SNMP
        
        return ServiceType.UNKNOWN
    
    def get_service_info(self, target: str, port: int) -> Optional[ServiceInfo]:
        """Get detailed service information"""
        try:
            # Check if port is open
            port_result = self.scan_ports(target, [port])
            if not port_result.success or port not in port_result.ports:
                return None
            
            state = port_result.ports[port]
            if state != PortState.OPEN:
                return None
            
            # Identify service
            service = self._identify_service(target, port)
            
            # Get banner
            banner = self._get_service_banner(target, port)
            
            return ServiceInfo(
                port=port,
                service=service,
                banner=banner,
                state=state,
                protocol="tcp"
            )
            
        except Exception as e:
            logger.error(f"Failed to get service info for {target}:{port}: {e}")
            return None


class VulnerabilityScanner(ServiceScanner):
    """Vulnerability scanner"""
    
    def __init__(self, timeout: int = 3, verbose: bool = False):
        super().__init__(timeout, verbose)
        self.vulnerability_signatures = {
            ServiceType.HTTP: [
                "apache", "nginx", "iis", "tomcat", "jboss", "weblogic"
            ],
            ServiceType.SSH: [
                "openssh", "ssh", "telnet"
            ],
            ServiceType.FTP: [
                "vsftpd", "proftpd", "pure-ftpd"
            ],
            ServiceType.SMTP: [
                "postfix", "sendmail", "exim", "exchange"
            ]
        }
    
    def scan_vulnerabilities(self, target: str, ports: Optional[List[int]] = None) -> ScanResult:
        """Scan for vulnerabilities"""
        try:
            # First scan services
            service_result = self.scan_services(target, ports)
            
            if not service_result.success:
                return service_result
            
            # Then check for vulnerabilities
            vulnerabilities = []
            for port, service in service_result.services.items():
                port_vulns = self._check_vulnerabilities(target, port, service)
                vulnerabilities.extend(port_vulns)
            
            # Update result
            service_result.vulnerabilities = vulnerabilities
            service_result.scan_type = ScanType.VULNERABILITY
            
            return service_result
            
        except Exception as e:
            logger.error(f"Vulnerability scan failed: {e}")
            return ScanResult(
                target=target,
                scan_type=ScanType.VULNERABILITY,
                success=False,
                error=str(e)
            )
    
    def _check_vulnerabilities(
        self,
        target: str,
        port: int,
        service: ServiceType
    ) -> List[str]:
        """Check for vulnerabilities on specific port/service"""
        vulnerabilities = []
        
        try:
            # Get service banner
            banner = self._get_service_banner(target, port)
            if not banner:
                return vulnerabilities
            
            # Check for known vulnerabilities based on service
            if service in self.vulnerability_signatures:
                for signature in self.vulnerability_signatures[service]:
                    if signature.lower() in banner.lower():
                        vulnerabilities.append(f"Potential {signature} vulnerability on port {port}")
            
            # Check for default credentials
            if self._check_default_credentials(target, port, service):
                vulnerabilities.append(f"Default credentials possible on port {port}")
            
            # Check for weak configurations
            if self._check_weak_configuration(target, port, service):
                vulnerabilities.append(f"Weak configuration detected on port {port}")
            
            return vulnerabilities
            
        except Exception as e:
            logger.error(f"Vulnerability check failed for {target}:{port}: {e}")
            return vulnerabilities
    
    def _check_default_credentials(self, target: str, port: int, service: ServiceType) -> bool:
        """Check for default credentials"""
        # This is a simplified check - in practice, you'd test actual credentials
        default_creds = {
            ServiceType.SSH: ["root:root", "admin:admin"],
            ServiceType.FTP: ["anonymous:anonymous", "ftp:ftp"],
            ServiceType.TELNET: ["root:root", "admin:admin"]
        }
        
        return service in default_creds
    
    def _check_weak_configuration(self, target: str, port: int, service: ServiceType) -> bool:
        """Check for weak configurations"""
        # This is a simplified check - in practice, you'd check actual configurations
        weak_configs = {
            ServiceType.HTTP: ["directory listing enabled", "debug mode enabled"],
            ServiceType.SSH: ["root login enabled", "password authentication enabled"],
            ServiceType.FTP: ["anonymous access enabled", "write access enabled"]
        }
        
        return service in weak_configs
    
    def get_vulnerability_info(
        self,
        target: str,
        port: int,
        service: ServiceType
    ) -> List[VulnerabilityInfo]:
        """Get detailed vulnerability information"""
        vulnerabilities = []
        
        try:
            # Check for vulnerabilities
            vuln_list = self._check_vulnerabilities(target, port, service)
            
            for vuln in vuln_list:
                vuln_info = VulnerabilityInfo(
                    port=port,
                    service=service,
                    vulnerability=vuln,
                    severity="medium",
                    description=f"Potential vulnerability detected on {target}:{port}"
                )
                vulnerabilities.append(vuln_info)
            
            return vulnerabilities
            
        except Exception as e:
            logger.error(f"Failed to get vulnerability info for {target}:{port}: {e}")
            return vulnerabilities


# Convenience functions
def scan_network(network: str) -> ScanResult:
    """Convenience function for network scanning"""
    scanner = NetworkScanner()
    return scanner.scan_network(network)


def scan_ports(
    target: str,
    ports: Optional[List[int]] = None,
    scan_type: str = "tcp"
) -> ScanResult:
    """Convenience function for port scanning"""
    scanner = PortScanner()
    return scanner.scan_ports(target, ports, scan_type)


def scan_services(target: str, ports: Optional[List[int]] = None) -> ScanResult:
    """Convenience function for service scanning"""
    scanner = ServiceScanner()
    return scanner.scan_services(target, ports)


def scan_vulnerabilities(target: str, ports: Optional[List[int]] = None) -> ScanResult:
    """Convenience function for vulnerability scanning"""
    scanner = VulnerabilityScanner()
    return scanner.scan_vulnerabilities(target, ports)


# Example usage
if __name__ == "__main__":
    # Example network scanning
    print("🔍 Network Scanner Example")
    
    # Network scanning
    print("\n" + "="*60)
    print("NETWORK SCANNING")
    print("="*60)
    
    network_scanner = NetworkScanner()
    network_result = network_scanner.scan_network("192.168.1.0/24")
    
    if network_result.success:
        print(f"✅ Network scan completed in {network_result.duration:.2f} seconds")
        print(f"📡 Found {len(network_result.hosts)} live hosts:")
        for host in network_result.hosts[:5]:  # Show first 5 hosts
            print(f"   {host}")
    else:
        print(f"❌ Network scan failed: {network_result.error}")
    
    # Port scanning
    print("\n" + "="*60)
    print("PORT SCANNING")
    print("="*60)
    
    port_scanner = PortScanner()
    port_result = port_scanner.quick_scan("127.0.0.1")
    
    if port_result.success:
        print(f"✅ Port scan completed in {port_result.duration:.2f} seconds")
        print(f"🔌 Open ports:")
        for port, state in port_result.ports.items():
            if state == PortState.OPEN:
                print(f"   {port}/tcp - {state.value}")
    else:
        print(f"❌ Port scan failed: {port_result.error}")
    
    # Service scanning
    print("\n" + "="*60)
    print("SERVICE SCANNING")
    print("="*60)
    
    service_scanner = ServiceScanner()
    service_result = service_scanner.scan_services("127.0.0.1", [80, 443, 22, 21])
    
    if service_result.success:
        print(f"✅ Service scan completed in {service_result.duration:.2f} seconds")
        print(f"🔧 Services detected:")
        for port, service in service_result.services.items():
            print(f"   {port}/tcp - {service.value}")
    else:
        print(f"❌ Service scan failed: {service_result.error}")
    
    # Vulnerability scanning
    print("\n" + "="*60)
    print("VULNERABILITY SCANNING")
    print("="*60)
    
    vuln_scanner = VulnerabilityScanner()
    vuln_result = vuln_scanner.scan_vulnerabilities("127.0.0.1", [80, 443, 22])
    
    if vuln_result.success:
        print(f"✅ Vulnerability scan completed in {vuln_result.duration:.2f} seconds")
        print(f"🔒 Vulnerabilities found:")
        for vuln in vuln_result.vulnerabilities:
            print(f"   ⚠️  {vuln}")
        
        if not vuln_result.vulnerabilities:
            print("   ✅ No vulnerabilities detected")
    else:
        print(f"❌ Vulnerability scan failed: {vuln_result.error}")
    
    # Get detailed service info
    print("\n" + "="*60)
    print("DETAILED SERVICE INFO")
    print("="*60)
    
    service_info = service_scanner.get_service_info("127.0.0.1", 80)
    if service_info:
        print(f"✅ Service info for port {service_info.port}:")
        print(f"   Service: {service_info.service.value}")
        print(f"   State: {service_info.state.value}")
        print(f"   Protocol: {service_info.protocol}")
        if service_info.banner:
            print(f"   Banner: {service_info.banner[:100]}...")
    else:
        print("❌ Could not get service info")
    
    # Get vulnerability info
    print("\n" + "="*60)
    print("DETAILED VULNERABILITY INFO")
    print("="*60)
    
    vuln_info_list = vuln_scanner.get_vulnerability_info("127.0.0.1", 80, ServiceType.HTTP)
    if vuln_info_list:
        print(f"✅ Vulnerability info for port 80:")
        for vuln_info in vuln_info_list:
            print(f"   ⚠️  {vuln_info.vulnerability}")
            print(f"      Severity: {vuln_info.severity}")
            print(f"      Description: {vuln_info.description}")
    else:
        print("✅ No vulnerabilities found")
    
    print("\n✅ Network scanner example completed!") 