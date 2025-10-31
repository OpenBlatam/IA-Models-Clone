#!/usr/bin/env python3
"""
Packet Sniffer for Video-OpusClip
Network packet sniffing and capture using Scapy
"""

import threading
import time
import queue
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from enum import Enum
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging

from scapy.all import (
    sniff, srp, sr, sr1, IP, TCP, UDP, ICMP, ARP, DNS, Raw, Ether,
    conf, get_if_list, get_if_addr, get_if_hwaddr
)

logger = logging.getLogger(__name__)


class SnifferMode(Enum):
    """Sniffer modes"""
    PASSIVE = "passive"
    ACTIVE = "active"
    PROMISCUOUS = "promiscuous"


class CaptureFilter(Enum):
    """Capture filters"""
    ALL = "all"
    TCP = "tcp"
    UDP = "udp"
    ICMP = "icmp"
    ARP = "arp"
    DNS = "dns"
    HTTP = "http"
    HTTPS = "https"
    CUSTOM = "custom"


@dataclass
class SnifferConfig:
    """Sniffer configuration"""
    interface: str = "eth0"
    mode: SnifferMode = SnifferMode.PASSIVE
    filter: CaptureFilter = CaptureFilter.ALL
    custom_filter: Optional[str] = None
    count: int = 0  # 0 means unlimited
    timeout: int = 0  # 0 means no timeout
    store: bool = True
    prn: Optional[Callable] = None
    stop_filter: Optional[Callable] = None
    verbose: bool = False


@dataclass
class PacketInfo:
    """Packet information"""
    timestamp: float
    source_ip: Optional[str] = None
    destination_ip: Optional[str] = None
    source_port: Optional[int] = None
    destination_port: Optional[int] = None
    protocol: Optional[str] = None
    length: int = 0
    payload: Optional[bytes] = None
    flags: Optional[str] = None
    ttl: Optional[int] = None
    raw_packet: Any = None


@dataclass
class AnalysisResult:
    """Analysis result"""
    total_packets: int = 0
    protocols: Dict[str, int] = field(default_factory=dict)
    source_ips: Dict[str, int] = field(default_factory=dict)
    destination_ips: Dict[str, int] = field(default_factory=dict)
    ports: Dict[int, int] = field(default_factory=dict)
    packet_sizes: List[int] = field(default_factory=list)
    time_range: Tuple[float, float] = (0, 0)
    metadata: Dict[str, Any] = field(default_factory=dict)


class PacketSnifferBase(ABC):
    """Base class for packet sniffing"""
    
    def __init__(self, config: SnifferConfig):
        self.config = config
        self.packets: List[PacketInfo] = []
        self.is_running = False
        self.thread = None
        self.packet_queue = queue.Queue()
    
    @abstractmethod
    def start_capture(self) -> bool:
        """Start packet capture"""
        pass
    
    @abstractmethod
    def stop_capture(self) -> bool:
        """Stop packet capture"""
        pass
    
    def get_packets(self) -> List[PacketInfo]:
        """Get captured packets"""
        return self.packets.copy()
    
    def clear_packets(self):
        """Clear captured packets"""
        self.packets.clear()
    
    def get_packet_count(self) -> int:
        """Get number of captured packets"""
        return len(self.packets)
    
    def is_capturing(self) -> bool:
        """Check if sniffer is running"""
        return self.is_running


class NetworkSniffer(PacketSnifferBase):
    """Network packet sniffer"""
    
    def __init__(self, config: SnifferConfig):
        super().__init__(config)
        self.capture_thread = None
    
    def start_capture(self) -> bool:
        """Start packet capture"""
        try:
            if self.is_running:
                logger.warning("Sniffer is already running")
                return True
            
            # Build filter string
            filter_str = self._build_filter()
            
            # Start capture in separate thread
            self.capture_thread = threading.Thread(
                target=self._capture_packets,
                args=(filter_str,),
                daemon=True
            )
            self.capture_thread.start()
            
            self.is_running = True
            logger.info(f"Started packet capture on interface {self.config.interface}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start packet capture: {e}")
            return False
    
    def stop_capture(self) -> bool:
        """Stop packet capture"""
        try:
            if not self.is_running:
                logger.warning("Sniffer is not running")
                return True
            
            self.is_running = False
            
            if self.capture_thread and self.capture_thread.is_alive():
                self.capture_thread.join(timeout=5)
            
            logger.info("Stopped packet capture")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop packet capture: {e}")
            return False
    
    def _build_filter(self) -> str:
        """Build capture filter string"""
        if self.config.custom_filter:
            return self.config.custom_filter
        
        filter_map = {
            CaptureFilter.ALL: "",
            CaptureFilter.TCP: "tcp",
            CaptureFilter.UDP: "udp",
            CaptureFilter.ICMP: "icmp",
            CaptureFilter.ARP: "arp",
            CaptureFilter.DNS: "port 53",
            CaptureFilter.HTTP: "port 80",
            CaptureFilter.HTTPS: "port 443"
        }
        
        return filter_map.get(self.config.filter, "")
    
    def _capture_packets(self, filter_str: str):
        """Capture packets in background thread"""
        try:
            def packet_handler(packet):
                if not self.is_running:
                    return
                
                packet_info = self._extract_packet_info(packet)
                self.packets.append(packet_info)
                
                if self.config.prn:
                    self.config.prn(packet_info)
                
                # Check stop condition
                if self.config.count > 0 and len(self.packets) >= self.config.count:
                    self.is_running = False
                    return
            
            # Start sniffing
            sniff(
                iface=self.config.interface,
                filter=filter_str,
                prn=packet_handler,
                store=self.config.store,
                count=self.config.count if self.config.count > 0 else None,
                timeout=self.config.timeout if self.config.timeout > 0 else None,
                stop_filter=self.config.stop_filter,
                verbose=self.config.verbose
            )
            
        except Exception as e:
            logger.error(f"Packet capture error: {e}")
            self.is_running = False
    
    def _extract_packet_info(self, packet) -> PacketInfo:
        """Extract information from packet"""
        packet_info = PacketInfo(
            timestamp=time.time(),
            length=len(packet),
            raw_packet=packet
        )
        
        # Extract IP layer information
        if IP in packet:
            packet_info.source_ip = packet[IP].src
            packet_info.destination_ip = packet[IP].dst
            packet_info.ttl = packet[IP].ttl
        
        # Extract transport layer information
        if TCP in packet:
            packet_info.protocol = "TCP"
            packet_info.source_port = packet[TCP].sport
            packet_info.destination_port = packet[TCP].dport
            packet_info.flags = str(packet[TCP].flags)
        elif UDP in packet:
            packet_info.protocol = "UDP"
            packet_info.source_port = packet[UDP].sport
            packet_info.destination_port = packet[UDP].dport
        elif ICMP in packet:
            packet_info.protocol = "ICMP"
            packet_info.flags = f"type={packet[ICMP].type}, code={packet[ICMP].code}"
        elif ARP in packet:
            packet_info.protocol = "ARP"
            packet_info.source_ip = packet[ARP].psrc
            packet_info.destination_ip = packet[ARP].pdst
        elif DNS in packet:
            packet_info.protocol = "DNS"
        
        # Extract payload
        if Raw in packet:
            packet_info.payload = bytes(packet[Raw])
        
        return packet_info


class ProtocolSniffer(PacketSnifferBase):
    """Protocol-specific packet sniffer"""
    
    def __init__(self, config: SnifferConfig, protocol: str):
        super().__init__(config)
        self.protocol = protocol
    
    def start_capture(self) -> bool:
        """Start protocol-specific packet capture"""
        try:
            if self.is_running:
                logger.warning("Protocol sniffer is already running")
                return True
            
            # Set protocol-specific filter
            if self.protocol.upper() == "TCP":
                self.config.filter = CaptureFilter.TCP
            elif self.protocol.upper() == "UDP":
                self.config.filter = CaptureFilter.UDP
            elif self.protocol.upper() == "ICMP":
                self.config.filter = CaptureFilter.ICMP
            elif self.protocol.upper() == "ARP":
                self.config.filter = CaptureFilter.ARP
            elif self.protocol.upper() == "DNS":
                self.config.filter = CaptureFilter.DNS
            elif self.protocol.upper() == "HTTP":
                self.config.filter = CaptureFilter.HTTP
            elif self.protocol.upper() == "HTTPS":
                self.config.filter = CaptureFilter.HTTPS
            
            # Create network sniffer
            self.network_sniffer = NetworkSniffer(self.config)
            return self.network_sniffer.start_capture()
            
        except Exception as e:
            logger.error(f"Failed to start protocol sniffer: {e}")
            return False
    
    def stop_capture(self) -> bool:
        """Stop protocol-specific packet capture"""
        try:
            if hasattr(self, 'network_sniffer'):
                return self.network_sniffer.stop_capture()
            return True
        except Exception as e:
            logger.error(f"Failed to stop protocol sniffer: {e}")
            return False
    
    def get_protocol_packets(self) -> List[PacketInfo]:
        """Get packets for specific protocol"""
        if hasattr(self, 'network_sniffer'):
            return self.network_sniffer.get_packets()
        return []


class FilterSniffer(PacketSnifferBase):
    """Filter-based packet sniffer"""
    
    def __init__(self, config: SnifferConfig, custom_filter: str):
        super().__init__(config)
        self.custom_filter = custom_filter
        self.config.custom_filter = custom_filter
    
    def start_capture(self) -> bool:
        """Start filter-based packet capture"""
        try:
            if self.is_running:
                logger.warning("Filter sniffer is already running")
                return True
            
            # Create network sniffer with custom filter
            self.network_sniffer = NetworkSniffer(self.config)
            return self.network_sniffer.start_capture()
            
        except Exception as e:
            logger.error(f"Failed to start filter sniffer: {e}")
            return False
    
    def stop_capture(self) -> bool:
        """Stop filter-based packet capture"""
        try:
            if hasattr(self, 'network_sniffer'):
                return self.network_sniffer.stop_capture()
            return True
        except Exception as e:
            logger.error(f"Failed to stop filter sniffer: {e}")
            return False
    
    def get_filtered_packets(self) -> List[PacketInfo]:
        """Get filtered packets"""
        if hasattr(self, 'network_sniffer'):
            return self.network_sniffer.get_packets()
        return []


class PacketSniffer:
    """Main packet sniffer service"""
    
    def __init__(self):
        self.sniffers: Dict[str, PacketSnifferBase] = {}
        self.default_config = SnifferConfig()
    
    def start_sniffing(
        self,
        name: str,
        config: Optional[SnifferConfig] = None
    ) -> bool:
        """Start packet sniffing"""
        try:
            if name in self.sniffers:
                logger.warning(f"Sniffer '{name}' already exists")
                return False
            
            if config is None:
                config = self.default_config
            
            sniffer = NetworkSniffer(config)
            success = sniffer.start_capture()
            
            if success:
                self.sniffers[name] = sniffer
                logger.info(f"Started sniffer '{name}'")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to start sniffer '{name}': {e}")
            return False
    
    def stop_sniffing(self, name: str) -> bool:
        """Stop packet sniffing"""
        try:
            if name not in self.sniffers:
                logger.warning(f"Sniffer '{name}' not found")
                return False
            
            sniffer = self.sniffers[name]
            success = sniffer.stop_capture()
            
            if success:
                del self.sniffers[name]
                logger.info(f"Stopped sniffer '{name}'")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to stop sniffer '{name}': {e}")
            return False
    
    def start_protocol_sniffing(
        self,
        name: str,
        protocol: str,
        config: Optional[SnifferConfig] = None
    ) -> bool:
        """Start protocol-specific sniffing"""
        try:
            if name in self.sniffers:
                logger.warning(f"Sniffer '{name}' already exists")
                return False
            
            if config is None:
                config = self.default_config
            
            sniffer = ProtocolSniffer(config, protocol)
            success = sniffer.start_capture()
            
            if success:
                self.sniffers[name] = sniffer
                logger.info(f"Started protocol sniffer '{name}' for {protocol}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to start protocol sniffer '{name}': {e}")
            return False
    
    def start_filter_sniffing(
        self,
        name: str,
        custom_filter: str,
        config: Optional[SnifferConfig] = None
    ) -> bool:
        """Start filter-based sniffing"""
        try:
            if name in self.sniffers:
                logger.warning(f"Sniffer '{name}' already exists")
                return False
            
            if config is None:
                config = self.default_config
            
            sniffer = FilterSniffer(config, custom_filter)
            success = sniffer.start_capture()
            
            if success:
                self.sniffers[name] = sniffer
                logger.info(f"Started filter sniffer '{name}' with filter: {custom_filter}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to start filter sniffer '{name}': {e}")
            return False
    
    def get_sniffer(self, name: str) -> Optional[PacketSnifferBase]:
        """Get sniffer by name"""
        return self.sniffers.get(name)
    
    def get_all_sniffers(self) -> Dict[str, PacketSnifferBase]:
        """Get all sniffers"""
        return self.sniffers.copy()
    
    def stop_all_sniffing(self) -> bool:
        """Stop all sniffing"""
        try:
            success = True
            for name in list(self.sniffers.keys()):
                if not self.stop_sniffing(name):
                    success = False
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to stop all sniffing: {e}")
            return False
    
    def get_packets(self, name: str) -> List[PacketInfo]:
        """Get packets from specific sniffer"""
        sniffer = self.get_sniffer(name)
        if sniffer:
            return sniffer.get_packets()
        return []
    
    def clear_packets(self, name: str):
        """Clear packets from specific sniffer"""
        sniffer = self.get_sniffer(name)
        if sniffer:
            sniffer.clear_packets()
    
    def get_statistics(self, name: str) -> Dict[str, Any]:
        """Get statistics from specific sniffer"""
        sniffer = self.get_sniffer(name)
        if not sniffer:
            return {}
        
        packets = sniffer.get_packets()
        
        stats = {
            "total_packets": len(packets),
            "is_running": sniffer.is_running,
            "protocols": {},
            "source_ips": {},
            "destination_ips": {},
            "ports": {},
            "packet_sizes": []
        }
        
        for packet in packets:
            # Protocol statistics
            if packet.protocol:
                stats["protocols"][packet.protocol] = stats["protocols"].get(packet.protocol, 0) + 1
            
            # IP statistics
            if packet.source_ip:
                stats["source_ips"][packet.source_ip] = stats["source_ips"].get(packet.source_ip, 0) + 1
            
            if packet.destination_ip:
                stats["destination_ips"][packet.destination_ip] = stats["destination_ips"].get(packet.destination_ip, 0) + 1
            
            # Port statistics
            if packet.source_port:
                stats["ports"][packet.source_port] = stats["ports"].get(packet.source_port, 0) + 1
            
            if packet.destination_port:
                stats["ports"][packet.destination_port] = stats["ports"].get(packet.destination_port, 0) + 1
            
            # Packet size statistics
            stats["packet_sizes"].append(packet.length)
        
        return stats


# Convenience functions
def start_sniffing(
    interface: str = "eth0",
    filter_type: CaptureFilter = CaptureFilter.ALL,
    count: int = 0,
    timeout: int = 0,
    name: str = "default"
) -> bool:
    """Convenience function for starting packet sniffing"""
    config = SnifferConfig(
        interface=interface,
        filter=filter_type,
        count=count,
        timeout=timeout
    )
    
    sniffer = PacketSniffer()
    return sniffer.start_sniffing(name, config)


def stop_sniffing(name: str = "default") -> bool:
    """Convenience function for stopping packet sniffing"""
    sniffer = PacketSniffer()
    return sniffer.stop_sniffing(name)


def capture_packets(
    interface: str = "eth0",
    filter_type: CaptureFilter = CaptureFilter.ALL,
    count: int = 10,
    timeout: int = 30
) -> List[PacketInfo]:
    """Convenience function for capturing packets"""
    config = SnifferConfig(
        interface=interface,
        filter=filter_type,
        count=count,
        timeout=timeout
    )
    
    sniffer = NetworkSniffer(config)
    if sniffer.start_capture():
        time.sleep(timeout + 1)  # Wait for capture to complete
        sniffer.stop_capture()
        return sniffer.get_packets()
    
    return []


def analyze_packets(packets: List[PacketInfo]) -> AnalysisResult:
    """Convenience function for analyzing packets"""
    if not packets:
        return AnalysisResult()
    
    result = AnalysisResult(
        total_packets=len(packets),
        time_range=(packets[0].timestamp, packets[-1].timestamp)
    )
    
    for packet in packets:
        # Protocol analysis
        if packet.protocol:
            result.protocols[packet.protocol] = result.protocols.get(packet.protocol, 0) + 1
        
        # IP analysis
        if packet.source_ip:
            result.source_ips[packet.source_ip] = result.source_ips.get(packet.source_ip, 0) + 1
        
        if packet.destination_ip:
            result.destination_ips[packet.destination_ip] = result.destination_ips.get(packet.destination_ip, 0) + 1
        
        # Port analysis
        if packet.source_port:
            result.ports[packet.source_port] = result.ports.get(packet.source_port, 0) + 1
        
        if packet.destination_port:
            result.ports[packet.destination_port] = result.ports.get(packet.destination_port, 0) + 1
        
        # Packet size analysis
        result.packet_sizes.append(packet.length)
    
    # Calculate metadata
    if result.packet_sizes:
        result.metadata["avg_packet_size"] = sum(result.packet_sizes) / len(result.packet_sizes)
        result.metadata["min_packet_size"] = min(result.packet_sizes)
        result.metadata["max_packet_size"] = max(result.packet_sizes)
    
    result.metadata["duration"] = result.time_range[1] - result.time_range[0]
    result.metadata["packets_per_second"] = result.total_packets / result.metadata["duration"] if result.metadata["duration"] > 0 else 0
    
    return result


# Example usage
if __name__ == "__main__":
    # Example packet sniffing
    print("📡 Packet Sniffing Example")
    
    # Create sniffer service
    sniffer_service = PacketSniffer()
    
    # Start TCP packet sniffing
    print("\n" + "="*60)
    print("TCP PACKET SNIFFING")
    print("="*60)
    
    tcp_config = SnifferConfig(
        interface="eth0",
        filter=CaptureFilter.TCP,
        count=5,
        timeout=10
    )
    
    success = sniffer_service.start_sniffing("tcp_sniffer", tcp_config)
    if success:
        print("✅ TCP sniffer started")
        
        # Wait for packets
        time.sleep(5)
        
        # Get statistics
        stats = sniffer_service.get_statistics("tcp_sniffer")
        print(f"📊 TCP Statistics:")
        print(f"   Total packets: {stats.get('total_packets', 0)}")
        print(f"   Protocols: {stats.get('protocols', {})}")
        print(f"   Source IPs: {list(stats.get('source_ips', {}).keys())[:3]}")
        print(f"   Destination IPs: {list(stats.get('destination_ips', {}).keys())[:3]}")
        
        # Stop sniffer
        sniffer_service.stop_sniffing("tcp_sniffer")
        print("✅ TCP sniffer stopped")
    else:
        print("❌ Failed to start TCP sniffer")
    
    # Start protocol-specific sniffing
    print("\n" + "="*60)
    print("PROTOCOL SNIFFING")
    print("="*60)
    
    success = sniffer_service.start_protocol_sniffing("icmp_sniffer", "ICMP")
    if success:
        print("✅ ICMP sniffer started")
        
        # Wait for packets
        time.sleep(3)
        
        # Get packets
        packets = sniffer_service.get_packets("icmp_sniffer")
        print(f"📦 Captured {len(packets)} ICMP packets")
        
        for i, packet in enumerate(packets[:3]):
            print(f"   Packet {i+1}: {packet.source_ip} -> {packet.destination_ip} ({packet.protocol})")
        
        # Stop sniffer
        sniffer_service.stop_sniffing("icmp_sniffer")
        print("✅ ICMP sniffer stopped")
    else:
        print("❌ Failed to start ICMP sniffer")
    
    # Start filter-based sniffing
    print("\n" + "="*60)
    print("FILTER SNIFFING")
    print("="*60)
    
    success = sniffer_service.start_filter_sniffing("dns_sniffer", "port 53")
    if success:
        print("✅ DNS sniffer started")
        
        # Wait for packets
        time.sleep(3)
        
        # Get packets
        packets = sniffer_service.get_packets("dns_sniffer")
        print(f"📦 Captured {len(packets)} DNS packets")
        
        # Analyze packets
        analysis = analyze_packets(packets)
        print(f"📊 Analysis:")
        print(f"   Total packets: {analysis.total_packets}")
        print(f"   Protocols: {analysis.protocols}")
        print(f"   Duration: {analysis.metadata.get('duration', 0):.2f} seconds")
        print(f"   Packets per second: {analysis.metadata.get('packets_per_second', 0):.2f}")
        
        # Stop sniffer
        sniffer_service.stop_sniffing("dns_sniffer")
        print("✅ DNS sniffer stopped")
    else:
        print("❌ Failed to start DNS sniffer")
    
    # Capture packets directly
    print("\n" + "="*60)
    print("DIRECT CAPTURE")
    print("="*60)
    
    packets = capture_packets(
        interface="eth0",
        filter_type=CaptureFilter.ALL,
        count=3,
        timeout=5
    )
    
    print(f"📦 Directly captured {len(packets)} packets")
    for i, packet in enumerate(packets):
        print(f"   Packet {i+1}: {packet.source_ip}:{packet.source_port} -> {packet.destination_ip}:{packet.destination_port} ({packet.protocol})")
    
    print("\n✅ Packet sniffing example completed!") 