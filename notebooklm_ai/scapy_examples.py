from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import asyncio
import threading
import time
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from datetime import datetime
import json
import structlog
from pathlib import Path
    from scapy.all import *
    from scapy.layers.inet import IP, TCP, UDP, ICMP
    from scapy.layers.dns import DNS, DNSQR, DNSRR
    from scapy.layers.http import HTTP, HTTPRequest, HTTPResponse
    from scapy.layers.l2 import Ether, ARP
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
Scapy Packet Crafting and Sniffing Examples
===========================================

This module demonstrates advanced packet crafting and sniffing techniques using Scapy
for network security analysis, penetration testing, and network monitoring.

Features:
- Packet crafting for various protocols (TCP, UDP, ICMP, DNS, HTTP)
- Network sniffing and analysis
- Port scanning techniques
- Network reconnaissance
- Security testing and validation
"""


# Scapy imports
try:
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    print("Scapy not available. Install with: pip install scapy")

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

# Custom exception classes
class ScapyError(Exception):
    """Base exception for Scapy operations."""
    pass

class PacketCraftingError(ScapyError):
    """Raised when packet crafting fails."""
    pass

class SniffingError(ScapyError):
    """Raised when packet sniffing fails."""
    pass

class NetworkTimeoutError(ScapyError):
    """Raised when network operations timeout."""
    pass

@dataclass
class PacketInfo:
    """Information about a captured packet."""
    timestamp: datetime
    source_ip: str
    destination_ip: str
    source_port: Optional[int]
    destination_port: Optional[int]
    protocol: str
    packet_size: int
    payload_length: int
    flags: Optional[str] = None
    ttl: Optional[int] = None
    window_size: Optional[int] = None
    sequence_number: Optional[int] = None
    acknowledgment_number: Optional[int] = None

class PacketCraftingEngine:
    """Advanced packet crafting engine using Scapy."""
    
    def __init__(self, interface: Optional[str] = None):
        
    """__init__ function."""
self.interface = interface or conf.iface
        self.packet_count = 0
        self.crafted_packets = []
        
        if not SCAPY_AVAILABLE:
            raise ImportError("Scapy is required for packet crafting")
    
    def craft_tcp_syn_packet(self, target_ip: str, target_port: int, 
                           source_ip: Optional[str] = None, 
                           source_port: Optional[int] = None) -> Packet:
        """Craft a TCP SYN packet for port scanning."""
        # Guard clauses - all error conditions first
        if not target_ip or not isinstance(target_ip, str):
            raise PacketCraftingError("Target IP must be a non-empty string")
        
        if not isinstance(target_port, int) or target_port < 1 or target_port > 65535:
            raise PacketCraftingError("Target port must be between 1 and 65535")
        
        # Happy path - main packet crafting logic
        try:
            # Set default source IP if not provided
            if not source_ip:
                source_ip = get_if_addr(self.interface)
            
            # Set random source port if not provided
            if not source_port:
                source_port = RandShort()
            
            # Craft the packet
            packet = IP(dst=target_ip, src=source_ip) / \
                    TCP(sport=source_port, dport=target_port, flags="S")
            
            self.packet_count += 1
            self.crafted_packets.append({
                "type": "TCP_SYN",
                "target": f"{target_ip}:{target_port}",
                "timestamp": datetime.now().isoformat()
            })
            
            logger.info("TCP SYN packet crafted successfully",
                       module="scapy",
                       function="craft_tcp_syn_packet",
                       target_ip=target_ip,
                       target_port=target_port,
                       source_ip=source_ip,
                       source_port=source_port)
            
            return packet
            
        except Exception as e:
            logger.error("Failed to craft TCP SYN packet",
                        module="scapy",
                        function="craft_tcp_syn_packet",
                        target_ip=target_ip,
                        target_port=target_port,
                        error=str(e))
            raise PacketCraftingError(f"TCP SYN packet crafting failed: {str(e)}")
    
    def craft_udp_packet(self, target_ip: str, target_port: int, 
                        payload: str = "", source_ip: Optional[str] = None) -> Packet:
        """Craft a UDP packet with custom payload."""
        # Guard clauses - all error conditions first
        if not target_ip or not isinstance(target_ip, str):
            raise PacketCraftingError("Target IP must be a non-empty string")
        
        if not isinstance(target_port, int) or target_port < 1 or target_port > 65535:
            raise PacketCraftingError("Target port must be between 1 and 65535")
        
        # Happy path - main packet crafting logic
        try:
            # Set default source IP if not provided
            if not source_ip:
                source_ip = get_if_addr(self.interface)
            
            # Craft the packet
            packet = IP(dst=target_ip, src=source_ip) / \
                    UDP(dport=target_port) / \
                    Raw(load=payload)
            
            self.packet_count += 1
            self.crafted_packets.append({
                "type": "UDP",
                "target": f"{target_ip}:{target_port}",
                "payload_length": len(payload),
                "timestamp": datetime.now().isoformat()
            })
            
            logger.info("UDP packet crafted successfully",
                       module="scapy",
                       function="craft_udp_packet",
                       target_ip=target_ip,
                       target_port=target_port,
                       payload_length=len(payload))
            
            return packet
            
        except Exception as e:
            logger.error("Failed to craft UDP packet",
                        module="scapy",
                        function="craft_udp_packet",
                        target_ip=target_ip,
                        target_port=target_port,
                        error=str(e))
            raise PacketCraftingError(f"UDP packet crafting failed: {str(e)}")
    
    def craft_icmp_ping(self, target_ip: str, payload: str = "ping") -> Packet:
        """Craft an ICMP ping packet."""
        # Guard clauses - all error conditions first
        if not target_ip or not isinstance(target_ip, str):
            raise PacketCraftingError("Target IP must be a non-empty string")
        
        # Happy path - main packet crafting logic
        try:
            packet = IP(dst=target_ip) / \
                    ICMP(type=8, code=0) / \
                    Raw(load=payload)
            
            self.packet_count += 1
            self.crafted_packets.append({
                "type": "ICMP_PING",
                "target": target_ip,
                "payload": payload,
                "timestamp": datetime.now().isoformat()
            })
            
            logger.info("ICMP ping packet crafted successfully",
                       module="scapy",
                       function="craft_icmp_ping",
                       target_ip=target_ip,
                       payload=payload)
            
            return packet
            
        except Exception as e:
            logger.error("Failed to craft ICMP ping packet",
                        module="scapy",
                        function="craft_icmp_ping",
                        target_ip=target_ip,
                        error=str(e))
            raise PacketCraftingError(f"ICMP ping packet crafting failed: {str(e)}")
    
    def craft_dns_query(self, target_ip: str, domain: str, 
                       query_type: str = "A") -> Packet:
        """Craft a DNS query packet."""
        # Guard clauses - all error conditions first
        if not target_ip or not isinstance(target_ip, str):
            raise PacketCraftingError("Target IP must be a non-empty string")
        
        if not domain or not isinstance(domain, str):
            raise PacketCraftingError("Domain must be a non-empty string")
        
        # Happy path - main packet crafting logic
        try:
            packet = IP(dst=target_ip) / \
                    UDP(dport=53) / \
                    DNS(rd=1, qd=DNSQR(qname=domain, qtype=query_type))
            
            self.packet_count += 1
            self.crafted_packets.append({
                "type": "DNS_QUERY",
                "target": target_ip,
                "domain": domain,
                "query_type": query_type,
                "timestamp": datetime.now().isoformat()
            })
            
            logger.info("DNS query packet crafted successfully",
                       module="scapy",
                       function="craft_dns_query",
                       target_ip=target_ip,
                       domain=domain,
                       query_type=query_type)
            
            return packet
            
        except Exception as e:
            logger.error("Failed to craft DNS query packet",
                        module="scapy",
                        function="craft_dns_query",
                        target_ip=target_ip,
                        domain=domain,
                        error=str(e))
            raise PacketCraftingError(f"DNS query packet crafting failed: {str(e)}")
    
    async def craft_arp_request(self, target_ip: str) -> Packet:
        """Craft an ARP request packet."""
        # Guard clauses - all error conditions first
        if not target_ip or not isinstance(target_ip, str):
            raise PacketCraftingError("Target IP must be a non-empty string")
        
        # Happy path - main packet crafting logic
        try:
            packet = Ether(dst="ff:ff:ff:ff:ff:ff") / \
                    ARP(pdst=target_ip)
            
            self.packet_count += 1
            self.crafted_packets.append({
                "type": "ARP_REQUEST",
                "target": target_ip,
                "timestamp": datetime.now().isoformat()
            })
            
            logger.info("ARP request packet crafted successfully",
                       module="scapy",
                       function="craft_arp_request",
                       target_ip=target_ip)
            
            return packet
            
        except Exception as e:
            logger.error("Failed to craft ARP request packet",
                        module="scapy",
                        function="craft_arp_request",
                        target_ip=target_ip,
                        error=str(e))
            raise PacketCraftingError(f"ARP request packet crafting failed: {str(e)}")
    
    def send_packet(self, packet: Packet, timeout: float = 5.0) -> List[Packet]:
        """Send a packet and capture responses."""
        # Guard clauses - all error conditions first
        if not packet:
            raise PacketCraftingError("Packet cannot be None")
        
        if timeout <= 0:
            raise PacketCraftingError("Timeout must be positive")
        
        # Happy path - main packet sending logic
        try:
            responses = srp(packet, timeout=timeout, verbose=False)
            
            logger.info("Packet sent successfully",
                       module="scapy",
                       function="send_packet",
                       packet_type=type(packet).__name__,
                       responses_count=len(responses[0]) if responses[0] else 0)
            
            return responses[0] if responses[0] else []
            
        except Exception as e:
            logger.error("Failed to send packet",
                        module="scapy",
                        function="send_packet",
                        packet_type=type(packet).__name__,
                        error=str(e))
            raise PacketCraftingError(f"Packet sending failed: {str(e)}")

class PacketSniffer:
    """Advanced packet sniffing and analysis engine."""
    
    def __init__(self, interface: Optional[str] = None):
        
    """__init__ function."""
self.interface = interface or conf.iface
        self.is_sniffing = False
        self.captured_packets = []
        self.sniffing_thread = None
        self.packet_filters = {}
        
        if not SCAPY_AVAILABLE:
            raise ImportError("Scapy is required for packet sniffing")
    
    def start_sniffing(self, filter_expr: str = "", packet_count: int = 0, 
                      timeout: int = 0, callback: Optional[Callable] = None) -> None:
        """Start packet sniffing in a separate thread."""
        # Guard clauses - all error conditions first
        if self.is_sniffing:
            raise SniffingError("Sniffing is already active")
        
        if packet_count < 0:
            raise SniffingError("Packet count must be non-negative")
        
        if timeout < 0:
            raise SniffingError("Timeout must be non-negative")
        
        # Happy path - main sniffing logic
        try:
            self.is_sniffing = True
            self.sniffing_thread = threading.Thread(
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                target=self._sniff_packets,
                args=(filter_expr, packet_count, timeout, callback)
            )
            self.sniffing_thread.daemon = True
            self.sniffing_thread.start()
            
            logger.info("Packet sniffing started successfully",
                       module="scapy",
                       function="start_sniffing",
                       interface=self.interface,
                       filter_expr=filter_expr,
                       packet_count=packet_count,
                       timeout=timeout)
            
        except Exception as e:
            self.is_sniffing = False
            logger.error("Failed to start packet sniffing",
                        module="scapy",
                        function="start_sniffing",
                        interface=self.interface,
                        error=str(e))
            raise SniffingError(f"Failed to start sniffing: {str(e)}")
    
    def _sniff_packets(self, filter_expr: str, packet_count: int, 
                      timeout: int, callback: Optional[Callable]) -> None:
        """Internal method for packet sniffing."""
        try:
            def packet_handler(packet) -> Any:
                packet_info = self._extract_packet_info(packet)
                self.captured_packets.append(packet_info)
                
                if callback:
                    callback(packet_info)
                
                logger.debug("Packet captured",
                           module="scapy",
                           function="_sniff_packets",
                           source_ip=packet_info.source_ip,
                           destination_ip=packet_info.destination_ip,
                           protocol=packet_info.protocol)
            
            sniff(
                iface=self.interface,
                filter=filter_expr,
                prn=packet_handler,
                count=packet_count if packet_count > 0 else None,
                timeout=timeout if timeout > 0 else None,
                store=False
            )
            
        except Exception as e:
            logger.error("Packet sniffing failed",
                        module="scapy",
                        function="_sniff_packets",
                        error=str(e))
        finally:
            self.is_sniffing = False
    
    def stop_sniffing(self) -> None:
        """Stop packet sniffing."""
        if not self.is_sniffing:
            return
        
        self.is_sniffing = False
        if self.sniffing_thread and self.sniffing_thread.is_alive():
            self.sniffing_thread.join(timeout=5.0)
        
        logger.info("Packet sniffing stopped",
                   module="scapy",
                   function="stop_sniffing",
                   total_packets=len(self.captured_packets))
    
    def _extract_packet_info(self, packet: Packet) -> PacketInfo:
        """Extract information from a captured packet."""
        try:
            # Extract basic information
            timestamp = datetime.now()
            packet_size = len(packet)
            
            # Extract IP layer information
            if IP in packet:
                source_ip = packet[IP].src
                destination_ip = packet[IP].dst
                ttl = packet[IP].ttl
            else:
                source_ip = "N/A"
                destination_ip = "N/A"
                ttl = None
            
            # Extract transport layer information
            source_port = None
            destination_port = None
            protocol = "Unknown"
            flags = None
            window_size = None
            sequence_number = None
            acknowledgment_number = None
            
            if TCP in packet:
                protocol = "TCP"
                source_port = packet[TCP].sport
                destination_port = packet[TCP].dport
                flags = str(packet[TCP].flags)
                window_size = packet[TCP].window
                sequence_number = packet[TCP].seq
                acknowledgment_number = packet[TCP].ack
            elif UDP in packet:
                protocol = "UDP"
                source_port = packet[UDP].sport
                destination_port = packet[UDP].dport
            elif ICMP in packet:
                protocol = "ICMP"
            
            # Extract payload length
            payload_length = len(packet[Raw].load) if Raw in packet else 0
            
            return PacketInfo(
                timestamp=timestamp,
                source_ip=source_ip,
                destination_ip=destination_ip,
                source_port=source_port,
                destination_port=destination_port,
                protocol=protocol,
                packet_size=packet_size,
                payload_length=payload_length,
                flags=flags,
                ttl=ttl,
                window_size=window_size,
                sequence_number=sequence_number,
                acknowledgment_number=acknowledgment_number
            )
            
        except Exception as e:
            logger.error("Failed to extract packet information",
                        module="scapy",
                        function="_extract_packet_info",
                        error=str(e))
            # Return minimal packet info
            return PacketInfo(
                timestamp=datetime.now(),
                source_ip="Unknown",
                destination_ip="Unknown",
                source_port=None,
                destination_port=None,
                protocol="Unknown",
                packet_size=len(packet) if packet else 0,
                payload_length=0
            )
    
    def get_captured_packets(self) -> List[PacketInfo]:
        """Get all captured packets."""
        return self.captured_packets.copy()
    
    def clear_captured_packets(self) -> None:
        """Clear captured packets."""
        self.captured_packets.clear()
        logger.info("Captured packets cleared",
                   module="scapy",
                   function="clear_captured_packets")

class NetworkScanner:
    """Network scanning and reconnaissance using Scapy."""
    
    def __init__(self, interface: Optional[str] = None):
        
    """__init__ function."""
self.interface = interface or conf.iface
        self.crafting_engine = PacketCraftingEngine(interface)
        self.sniffer = PacketSniffer(interface)
        
        if not SCAPY_AVAILABLE:
            raise ImportError("Scapy is required for network scanning")
    
    def tcp_port_scan(self, target_ip: str, ports: List[int], 
                     timeout: float = 2.0) -> Dict[int, str]:
        """Perform TCP port scanning."""
        # Guard clauses - all error conditions first
        if not target_ip or not isinstance(target_ip, str):
            raise PacketCraftingError("Target IP must be a non-empty string")
        
        if not ports or not isinstance(ports, list):
            raise PacketCraftingError("Ports must be a non-empty list")
        
        if timeout <= 0:
            raise PacketCraftingError("Timeout must be positive")
        
        # Happy path - main port scanning logic
        try:
            results = {}
            
            for port in ports:
                try:
                    # Craft and send SYN packet
                    packet = self.crafting_engine.craft_tcp_syn_packet(target_ip, port)
                    responses = self.crafting_engine.send_packet(packet, timeout)
                    
                    if responses:
                        # Check for SYN-ACK response
                        for sent, received in responses:
                            if TCP in received and received[TCP].flags & 0x12:  # SYN-ACK
                                results[port] = "open"
                                break
                        else:
                            results[port] = "filtered"
                    else:
                        results[port] = "closed"
                        
                except Exception as e:
                    logger.warning("Port scan failed for port",
                                 module="scapy",
                                 function="tcp_port_scan",
                                 target_ip=target_ip,
                                 port=port,
                                 error=str(e))
                    results[port] = "error"
            
            logger.info("TCP port scan completed",
                       module="scapy",
                       function="tcp_port_scan",
                       target_ip=target_ip,
                       ports_scanned=len(ports),
                       open_ports=len([p for p, s in results.items() if s == "open"]))
            
            return results
            
        except Exception as e:
            logger.error("TCP port scan failed",
                        module="scapy",
                        function="tcp_port_scan",
                        target_ip=target_ip,
                        error=str(e))
            raise PacketCraftingError(f"TCP port scan failed: {str(e)}")
    
    def ping_sweep(self, network_range: str, timeout: float = 1.0) -> List[str]:
        """Perform ping sweep to discover hosts."""
        # Guard clauses - all error conditions first
        if not network_range or not isinstance(network_range, str):
            raise PacketCraftingError("Network range must be a non-empty string")
        
        if timeout <= 0:
            raise PacketCraftingError("Timeout must be positive")
        
        # Happy path - main ping sweep logic
        try:
            alive_hosts = []
            
            # Generate IP addresses in the range
            try:
                ip_list = [str(ip) for ip in IPNetwork(network_range)]
            except Exception:
                raise PacketCraftingError(f"Invalid network range: {network_range}")
            
            for ip in ip_list:
                try:
                    # Craft and send ICMP ping
                    packet = self.crafting_engine.craft_icmp_ping(ip)
                    responses = self.crafting_engine.send_packet(packet, timeout)
                    
                    if responses:
                        alive_hosts.append(ip)
                        logger.debug("Host discovered",
                                   module="scapy",
                                   function="ping_sweep",
                                   ip=ip)
                        
                except Exception as e:
                    logger.debug("Host not responding",
                               module="scapy",
                               function="ping_sweep",
                               ip=ip,
                               error=str(e))
            
            logger.info("Ping sweep completed",
                       module="scapy",
                       function="ping_sweep",
                       network_range=network_range,
                       total_hosts=len(ip_list),
                       alive_hosts=len(alive_hosts))
            
            return alive_hosts
            
        except Exception as e:
            logger.error("Ping sweep failed",
                        module="scapy",
                        function="ping_sweep",
                        network_range=network_range,
                        error=str(e))
            raise PacketCraftingError(f"Ping sweep failed: {str(e)}")
    
    def arp_scan(self, network_range: str, timeout: float = 2.0) -> Dict[str, str]:
        """Perform ARP scan to discover hosts and their MAC addresses."""
        # Guard clauses - all error conditions first
        if not network_range or not isinstance(network_range, str):
            raise PacketCraftingError("Network range must be a non-empty string")
        
        if timeout <= 0:
            raise PacketCraftingError("Timeout must be positive")
        
        # Happy path - main ARP scan logic
        try:
            results = {}
            
            # Generate IP addresses in the range
            try:
                ip_list = [str(ip) for ip in IPNetwork(network_range)]
            except Exception:
                raise PacketCraftingError(f"Invalid network range: {network_range}")
            
            for ip in ip_list:
                try:
                    # Craft and send ARP request
                    packet = self.crafting_engine.craft_arp_request(ip)
                    responses = self.crafting_engine.send_packet(packet, timeout)
                    
                    if responses:
                        for sent, received in responses:
                            if ARP in received and received[ARP].op == 2:  # ARP reply
                                mac_address = received[ARP].hwsrc
                                results[ip] = mac_address
                                logger.debug("Host discovered via ARP",
                                           module="scapy",
                                           function="arp_scan",
                                           ip=ip,
                                           mac=mac_address)
                                break
                        
                except Exception as e:
                    logger.debug("Host not responding to ARP",
                               module="scapy",
                               function="arp_scan",
                               ip=ip,
                               error=str(e))
            
            logger.info("ARP scan completed",
                       module="scapy",
                       function="arp_scan",
                       network_range=network_range,
                       total_hosts=len(ip_list),
                       discovered_hosts=len(results))
            
            return results
            
        except Exception as e:
            logger.error("ARP scan failed",
                        module="scapy",
                        function="arp_scan",
                        network_range=network_range,
                        error=str(e))
            raise PacketCraftingError(f"ARP scan failed: {str(e)}")

# Example usage and demonstration functions
def demonstrate_packet_crafting():
    """Demonstrate packet crafting capabilities."""
    if not SCAPY_AVAILABLE:
        print("Scapy not available. Install with: pip install scapy")
        return
    
    try:
        # Initialize packet crafting engine
        crafter = PacketCraftingEngine()
        
        print("=== Packet Crafting Demonstration ===")
        
        # Craft TCP SYN packet
        tcp_packet = crafter.craft_tcp_syn_packet("192.168.1.1", 80)
        print(f"TCP SYN packet crafted: {tcp_packet.summary()}")
        
        # Craft UDP packet
        udp_packet = crafter.craft_udp_packet("192.168.1.1", 53, "DNS query")
        print(f"UDP packet crafted: {udp_packet.summary()}")
        
        # Craft ICMP ping
        icmp_packet = crafter.craft_icmp_ping("192.168.1.1")
        print(f"ICMP ping crafted: {icmp_packet.summary()}")
        
        # Craft DNS query
        dns_packet = crafter.craft_dns_query("8.8.8.8", "example.com")
        print(f"DNS query crafted: {dns_packet.summary()}")
        
        print(f"Total packets crafted: {crafter.packet_count}")
        
    except Exception as e:
        print(f"Packet crafting demonstration failed: {e}")

def demonstrate_packet_sniffing():
    """Demonstrate packet sniffing capabilities."""
    if not SCAPY_AVAILABLE:
        print("Scapy not available. Install with: pip install scapy")
        return
    
    try:
        # Initialize packet sniffer
        sniffer = PacketSniffer()
        
        print("=== Packet Sniffing Demonstration ===")
        print("Starting packet sniffing for 10 seconds...")
        
        # Start sniffing
        sniffer.start_sniffing(filter="tcp", timeout=10)
        
        # Wait for sniffing to complete
        time.sleep(12)
        
        # Get captured packets
        packets = sniffer.get_captured_packets()
        
        print(f"Captured {len(packets)} packets:")
        for i, packet in enumerate(packets[:5]):  # Show first 5 packets
            print(f"  {i+1}. {packet.source_ip}:{packet.source_port} -> "
                  f"{packet.destination_ip}:{packet.destination_port} "
                  f"({packet.protocol})")
        
        if len(packets) > 5:
            print(f"  ... and {len(packets) - 5} more packets")
        
    except Exception as e:
        print(f"Packet sniffing demonstration failed: {e}")

def demonstrate_network_scanning():
    """Demonstrate network scanning capabilities."""
    if not SCAPY_AVAILABLE:
        print("Scapy not available. Install with: pip install scapy")
        return
    
    try:
        # Initialize network scanner
        scanner = NetworkScanner()
        
        print("=== Network Scanning Demonstration ===")
        
        # TCP port scan
        target_ip = "127.0.0.1"  # Localhost for demonstration
        ports = [22, 80, 443, 8080]
        
        print(f"Scanning {target_ip} on ports {ports}...")
        results = scanner.tcp_port_scan(target_ip, ports)
        
        print("Port scan results:")
        for port, status in results.items():
            print(f"  Port {port}: {status}")
        
        # ARP scan (local network)
        print("\nPerforming ARP scan on local network...")
        arp_results = scanner.arp_scan("192.168.1.0/24")
        
        print("ARP scan results:")
        for ip, mac in arp_results.items():
            print(f"  {ip}: {mac}")
        
    except Exception as e:
        print(f"Network scanning demonstration failed: {e}")

if __name__ == "__main__":
    # Run demonstrations
    demonstrate_packet_crafting()
    print()
    demonstrate_packet_sniffing()
    print()
    demonstrate_network_scanning() 