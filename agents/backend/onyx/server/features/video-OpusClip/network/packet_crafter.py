#!/usr/bin/env python3
"""
Packet Crafter for Video-OpusClip
Network packet crafting using Scapy
"""

import socket
import struct
import random
import time
from typing import Dict, List, Any, Optional, Union, Tuple
from enum import Enum
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging

from scapy.all import (
    IP, TCP, UDP, ICMP, ARP, DNS, DNSQR, DNSRR, Raw, Ether, srp, sr, sr1,
    RandShort, RandInt, RandIP, RandMAC, fragment, fragment6
)

logger = logging.getLogger(__name__)


class PacketType(Enum):
    """Packet types"""
    TCP = "tcp"
    UDP = "udp"
    ICMP = "icmp"
    ARP = "arp"
    DNS = "dns"
    HTTP = "http"
    HTTPS = "https"
    CUSTOM = "custom"


class ProtocolType(Enum):
    """Protocol types"""
    IPV4 = "ipv4"
    IPV6 = "ipv6"
    ETHERNET = "ethernet"
    RAW = "raw"


class PacketDirection(Enum):
    """Packet direction"""
    INBOUND = "inbound"
    OUTBOUND = "outbound"
    BOTH = "both"


@dataclass
class PacketConfig:
    """Packet configuration"""
    source_ip: str = "192.168.1.100"
    destination_ip: str = "192.168.1.1"
    source_port: Optional[int] = None
    destination_port: Optional[int] = None
    protocol: PacketType = PacketType.TCP
    payload: Optional[bytes] = None
    ttl: int = 64
    flags: Optional[str] = None
    window: int = 8192
    seq: Optional[int] = None
    ack: Optional[int] = None
    options: Optional[Dict[str, Any]] = None
    fragment: bool = False
    randomize: bool = True


@dataclass
class PacketResult:
    """Result of packet crafting"""
    packet: Any  # Scapy packet
    config: PacketConfig
    crafted: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class PacketCrafterBase(ABC):
    """Base class for packet crafting"""
    
    def __init__(self, protocol: PacketType):
        self.protocol = protocol
        self.default_config = PacketConfig()
    
    @abstractmethod
    def craft_packet(self, config: PacketConfig) -> PacketResult:
        """Craft a packet"""
        pass
    
    def _randomize_config(self, config: PacketConfig) -> PacketConfig:
        """Randomize packet configuration"""
        if not config.randomize:
            return config
        
        # Randomize source port if not specified
        if config.source_port is None:
            config.source_port = random.randint(1024, 65535)
        
        # Randomize sequence number if not specified
        if config.seq is None:
            config.seq = random.randint(1000000000, 2000000000)
        
        # Randomize acknowledgment number if not specified
        if config.ack is None:
            config.ack = random.randint(1000000000, 2000000000)
        
        return config
    
    def _validate_config(self, config: PacketConfig) -> bool:
        """Validate packet configuration"""
        try:
            # Validate IP addresses
            socket.inet_aton(config.source_ip)
            socket.inet_aton(config.destination_ip)
            
            # Validate ports if specified
            if config.source_port is not None:
                if not (0 <= config.source_port <= 65535):
                    return False
            
            if config.destination_port is not None:
                if not (0 <= config.destination_port <= 65535):
                    return False
            
            # Validate TTL
            if not (1 <= config.ttl <= 255):
                return False
            
            return True
            
        except Exception:
            return False


class TCPPacketCrafter(PacketCrafterBase):
    """TCP packet crafter"""
    
    def __init__(self):
        super().__init__(PacketType.TCP)
    
    def craft_packet(self, config: PacketConfig) -> PacketResult:
        """Craft a TCP packet"""
        try:
            # Validate and randomize config
            if not self._validate_config(config):
                return PacketResult(None, config, False, "Invalid configuration")
            
            config = self._randomize_config(config)
            
            # Create IP layer
            ip_layer = IP(
                src=config.source_ip,
                dst=config.destination_ip,
                ttl=config.ttl
            )
            
            # Create TCP layer
            tcp_layer = TCP(
                sport=config.source_port,
                dport=config.destination_port,
                window=config.window,
                seq=config.seq,
                ack=config.ack
            )
            
            # Set TCP flags
            if config.flags:
                tcp_layer.flags = config.flags
            
            # Add TCP options if specified
            if config.options:
                for option_name, option_value in config.options.items():
                    setattr(tcp_layer, option_name, option_value)
            
            # Create payload layer
            payload_layer = None
            if config.payload:
                payload_layer = Raw(load=config.payload)
            
            # Assemble packet
            if payload_layer:
                packet = ip_layer / tcp_layer / payload_layer
            else:
                packet = ip_layer / tcp_layer
            
            # Fragment if requested
            if config.fragment:
                fragments = fragment(packet)
                return PacketResult(fragments, config, True, metadata={"fragmented": True, "fragment_count": len(fragments)})
            
            return PacketResult(packet, config, True)
            
        except Exception as e:
            logger.error(f"TCP packet crafting failed: {e}")
            return PacketResult(None, config, False, str(e))


class UDPPacketCrafter(PacketCrafterBase):
    """UDP packet crafter"""
    
    def __init__(self):
        super().__init__(PacketType.UDP)
    
    def craft_packet(self, config: PacketConfig) -> PacketResult:
        """Craft a UDP packet"""
        try:
            # Validate and randomize config
            if not self._validate_config(config):
                return PacketResult(None, config, False, "Invalid configuration")
            
            config = self._randomize_config(config)
            
            # Create IP layer
            ip_layer = IP(
                src=config.source_ip,
                dst=config.destination_ip,
                ttl=config.ttl
            )
            
            # Create UDP layer
            udp_layer = UDP(
                sport=config.source_port,
                dport=config.destination_port
            )
            
            # Create payload layer
            payload_layer = None
            if config.payload:
                payload_layer = Raw(load=config.payload)
            
            # Assemble packet
            if payload_layer:
                packet = ip_layer / udp_layer / payload_layer
            else:
                packet = ip_layer / udp_layer
            
            # Fragment if requested
            if config.fragment:
                fragments = fragment(packet)
                return PacketResult(fragments, config, True, metadata={"fragmented": True, "fragment_count": len(fragments)})
            
            return PacketResult(packet, config, True)
            
        except Exception as e:
            logger.error(f"UDP packet crafting failed: {e}")
            return PacketResult(None, config, False, str(e))


class ICMPPacketCrafter(PacketCrafterBase):
    """ICMP packet crafter"""
    
    def __init__(self):
        super().__init__(PacketType.ICMP)
    
    def craft_packet(self, config: PacketConfig) -> PacketResult:
        """Craft an ICMP packet"""
        try:
            # Validate and randomize config
            if not self._validate_config(config):
                return PacketResult(None, config, False, "Invalid configuration")
            
            config = self._randomize_config(config)
            
            # Create IP layer
            ip_layer = IP(
                src=config.source_ip,
                dst=config.destination_ip,
                ttl=config.ttl
            )
            
            # Create ICMP layer
            icmp_layer = ICMP()
            
            # Set ICMP type and code if specified in options
            if config.options:
                if 'type' in config.options:
                    icmp_layer.type = config.options['type']
                if 'code' in config.options:
                    icmp_layer.code = config.options['code']
            
            # Create payload layer
            payload_layer = None
            if config.payload:
                payload_layer = Raw(load=config.payload)
            else:
                # Default ICMP payload
                payload_layer = Raw(load=b"ABCDEFGHIJKLMNOPQRSTUVWABCDEFGHIJKLMNOPQRSTUVWABCDEFGHIJKLMNOPQRSTUVWABCDEFGHIJKLMNOPQRSTUVW")
            
            # Assemble packet
            packet = ip_layer / icmp_layer / payload_layer
            
            # Fragment if requested
            if config.fragment:
                fragments = fragment(packet)
                return PacketResult(fragments, config, True, metadata={"fragmented": True, "fragment_count": len(fragments)})
            
            return PacketResult(packet, config, True)
            
        except Exception as e:
            logger.error(f"ICMP packet crafting failed: {e}")
            return PacketResult(None, config, False, str(e))


class ARPPacketCrafter(PacketCrafterBase):
    """ARP packet crafter"""
    
    def __init__(self):
        super().__init__(PacketType.ARP)
    
    def craft_packet(self, config: PacketConfig) -> PacketResult:
        """Craft an ARP packet"""
        try:
            # Validate and randomize config
            if not self._validate_config(config):
                return PacketResult(None, config, False, "Invalid configuration")
            
            config = self._randomize_config(config)
            
            # Create Ethernet layer
            ether_layer = Ether()
            
            # Create ARP layer
            arp_layer = ARP(
                psrc=config.source_ip,
                pdst=config.destination_ip
            )
            
            # Set ARP operation if specified in options
            if config.options and 'op' in config.options:
                arp_layer.op = config.options['op']
            
            # Create payload layer
            payload_layer = None
            if config.payload:
                payload_layer = Raw(load=config.payload)
            
            # Assemble packet
            if payload_layer:
                packet = ether_layer / arp_layer / payload_layer
            else:
                packet = ether_layer / arp_layer
            
            return PacketResult(packet, config, True)
            
        except Exception as e:
            logger.error(f"ARP packet crafting failed: {e}")
            return PacketResult(None, config, False, str(e))


class DNSPacketCrafter(PacketCrafterBase):
    """DNS packet crafter"""
    
    def __init__(self):
        super().__init__(PacketType.DNS)
    
    def craft_packet(self, config: PacketConfig) -> PacketResult:
        """Craft a DNS packet"""
        try:
            # Validate and randomize config
            if not self._validate_config(config):
                return PacketResult(None, config, False, "Invalid configuration")
            
            config = self._randomize_config(config)
            
            # Create IP layer
            ip_layer = IP(
                src=config.source_ip,
                dst=config.destination_ip,
                ttl=config.ttl
            )
            
            # Create UDP layer
            udp_layer = UDP(
                sport=config.source_port or 53,
                dport=config.destination_port or 53
            )
            
            # Create DNS layer
            dns_layer = DNS()
            
            # Set DNS query if specified in options
            if config.options and 'qd' in config.options:
                dns_layer.qd = DNSQR(qname=config.options['qd'])
            
            # Set DNS response if specified in options
            if config.options and 'an' in config.options:
                dns_layer.an = DNSRR(rrname=config.options['an'])
            
            # Create payload layer
            payload_layer = None
            if config.payload:
                payload_layer = Raw(load=config.payload)
            
            # Assemble packet
            if payload_layer:
                packet = ip_layer / udp_layer / dns_layer / payload_layer
            else:
                packet = ip_layer / udp_layer / dns_layer
            
            return PacketResult(packet, config, True)
            
        except Exception as e:
            logger.error(f"DNS packet crafting failed: {e}")
            return PacketResult(None, config, False, str(e))


class HTTPPacketCrafter(PacketCrafterBase):
    """HTTP packet crafter"""
    
    def __init__(self):
        super().__init__(PacketType.HTTP)
    
    def craft_packet(self, config: PacketConfig) -> PacketResult:
        """Craft an HTTP packet"""
        try:
            # Validate and randomize config
            if not self._validate_config(config):
                return PacketResult(None, config, False, "Invalid configuration")
            
            config = self._randomize_config(config)
            
            # Create IP layer
            ip_layer = IP(
                src=config.source_ip,
                dst=config.destination_ip,
                ttl=config.ttl
            )
            
            # Create TCP layer
            tcp_layer = TCP(
                sport=config.source_port,
                dport=config.destination_port or 80,
                window=config.window,
                seq=config.seq,
                ack=config.ack
            )
            
            # Create HTTP payload
            if config.payload:
                http_payload = config.payload
            else:
                # Default HTTP GET request
                http_payload = b"GET / HTTP/1.1\r\nHost: " + config.destination_ip.encode() + b"\r\n\r\n"
            
            # Create payload layer
            payload_layer = Raw(load=http_payload)
            
            # Assemble packet
            packet = ip_layer / tcp_layer / payload_layer
            
            return PacketResult(packet, config, True)
            
        except Exception as e:
            logger.error(f"HTTP packet crafting failed: {e}")
            return PacketResult(None, config, False, str(e))


class CustomPacketCrafter(PacketCrafterBase):
    """Custom packet crafter"""
    
    def __init__(self):
        super().__init__(PacketType.CUSTOM)
    
    def craft_packet(self, config: PacketConfig) -> PacketResult:
        """Craft a custom packet"""
        try:
            # Validate and randomize config
            if not self._validate_config(config):
                return PacketResult(None, config, False, "Invalid configuration")
            
            config = self._randomize_config(config)
            
            # Create base layers based on options
            layers = []
            
            # Add Ethernet layer if specified
            if config.options and 'ethernet' in config.options:
                ether_layer = Ether(**config.options['ethernet'])
                layers.append(ether_layer)
            
            # Add IP layer
            ip_layer = IP(
                src=config.source_ip,
                dst=config.destination_ip,
                ttl=config.ttl
            )
            layers.append(ip_layer)
            
            # Add transport layer based on protocol
            if config.protocol == PacketType.TCP:
                transport_layer = TCP(
                    sport=config.source_port,
                    dport=config.destination_port,
                    window=config.window,
                    seq=config.seq,
                    ack=config.ack
                )
                if config.flags:
                    transport_layer.flags = config.flags
            elif config.protocol == PacketType.UDP:
                transport_layer = UDP(
                    sport=config.source_port,
                    dport=config.destination_port
                )
            else:
                transport_layer = None
            
            if transport_layer:
                layers.append(transport_layer)
            
            # Add custom layers if specified
            if config.options and 'custom_layers' in config.options:
                for layer_config in config.options['custom_layers']:
                    layer_class = layer_config['class']
                    layer_params = layer_config.get('params', {})
                    custom_layer = layer_class(**layer_params)
                    layers.append(custom_layer)
            
            # Add payload layer
            if config.payload:
                payload_layer = Raw(load=config.payload)
                layers.append(payload_layer)
            
            # Assemble packet
            packet = layers[0]
            for layer in layers[1:]:
                packet = packet / layer
            
            return PacketResult(packet, config, True)
            
        except Exception as e:
            logger.error(f"Custom packet crafting failed: {e}")
            return PacketResult(None, config, False, str(e))


class PacketCrafter:
    """Main packet crafter service"""
    
    def __init__(self):
        self.crafters: Dict[PacketType, PacketCrafterBase] = {
            PacketType.TCP: TCPPacketCrafter(),
            PacketType.UDP: UDPPacketCrafter(),
            PacketType.ICMP: ICMPPacketCrafter(),
            PacketType.ARP: ARPPacketCrafter(),
            PacketType.DNS: DNSPacketCrafter(),
            PacketType.HTTP: HTTPPacketCrafter(),
            PacketType.CUSTOM: CustomPacketCrafter()
        }
    
    def craft_packet(
        self,
        packet_type: PacketType,
        config: Optional[PacketConfig] = None
    ) -> PacketResult:
        """Craft a packet of specified type"""
        if packet_type not in self.crafters:
            return PacketResult(None, config or PacketConfig(), False, f"Unsupported packet type: {packet_type}")
        
        if config is None:
            config = PacketConfig()
        
        crafter = self.crafters[packet_type]
        return crafter.craft_packet(config)
    
    def craft_multiple_packets(
        self,
        packet_type: PacketType,
        count: int,
        config: Optional[PacketConfig] = None
    ) -> List[PacketResult]:
        """Craft multiple packets of specified type"""
        results = []
        for i in range(count):
            if config is None:
                packet_config = PacketConfig()
            else:
                packet_config = PacketConfig(**config.__dict__)
            
            # Randomize source port for each packet
            if packet_config.randomize:
                packet_config.source_port = random.randint(1024, 65535)
            
            result = self.craft_packet(packet_type, packet_config)
            results.append(result)
        
        return results
    
    def get_supported_types(self) -> List[PacketType]:
        """Get list of supported packet types"""
        return list(self.crafters.keys())


# Convenience functions
def create_tcp_packet(
    source_ip: str = "192.168.1.100",
    destination_ip: str = "192.168.1.1",
    source_port: Optional[int] = None,
    destination_port: int = 80,
    payload: Optional[bytes] = None,
    flags: Optional[str] = None,
    **kwargs
) -> PacketResult:
    """Convenience function for creating TCP packets"""
    config = PacketConfig(
        source_ip=source_ip,
        destination_ip=destination_ip,
        source_port=source_port,
        destination_port=destination_port,
        protocol=PacketType.TCP,
        payload=payload,
        flags=flags,
        **kwargs
    )
    
    crafter = PacketCrafter()
    return crafter.craft_packet(PacketType.TCP, config)


def create_udp_packet(
    source_ip: str = "192.168.1.100",
    destination_ip: str = "192.168.1.1",
    source_port: Optional[int] = None,
    destination_port: int = 53,
    payload: Optional[bytes] = None,
    **kwargs
) -> PacketResult:
    """Convenience function for creating UDP packets"""
    config = PacketConfig(
        source_ip=source_ip,
        destination_ip=destination_ip,
        source_port=source_port,
        destination_port=destination_port,
        protocol=PacketType.UDP,
        payload=payload,
        **kwargs
    )
    
    crafter = PacketCrafter()
    return crafter.craft_packet(PacketType.UDP, config)


def create_icmp_packet(
    source_ip: str = "192.168.1.100",
    destination_ip: str = "192.168.1.1",
    icmp_type: int = 8,
    icmp_code: int = 0,
    payload: Optional[bytes] = None,
    **kwargs
) -> PacketResult:
    """Convenience function for creating ICMP packets"""
    config = PacketConfig(
        source_ip=source_ip,
        destination_ip=destination_ip,
        protocol=PacketType.ICMP,
        payload=payload,
        options={'type': icmp_type, 'code': icmp_code},
        **kwargs
    )
    
    crafter = PacketCrafter()
    return crafter.craft_packet(PacketType.ICMP, config)


def create_arp_packet(
    source_ip: str = "192.168.1.100",
    destination_ip: str = "192.168.1.1",
    arp_op: int = 1,
    payload: Optional[bytes] = None,
    **kwargs
) -> PacketResult:
    """Convenience function for creating ARP packets"""
    config = PacketConfig(
        source_ip=source_ip,
        destination_ip=destination_ip,
        protocol=PacketType.ARP,
        payload=payload,
        options={'op': arp_op},
        **kwargs
    )
    
    crafter = PacketCrafter()
    return crafter.craft_packet(PacketType.ARP, config)


def create_dns_packet(
    source_ip: str = "192.168.1.100",
    destination_ip: str = "8.8.8.8",
    query_name: str = "example.com",
    query_type: str = "A",
    **kwargs
) -> PacketResult:
    """Convenience function for creating DNS packets"""
    config = PacketConfig(
        source_ip=source_ip,
        destination_ip=destination_ip,
        source_port=random.randint(1024, 65535),
        destination_port=53,
        protocol=PacketType.DNS,
        options={'qd': query_name},
        **kwargs
    )
    
    crafter = PacketCrafter()
    return crafter.craft_packet(PacketType.DNS, config)


def create_http_packet(
    source_ip: str = "192.168.1.100",
    destination_ip: str = "192.168.1.1",
    method: str = "GET",
    path: str = "/",
    host: str = "192.168.1.1",
    **kwargs
) -> PacketResult:
    """Convenience function for creating HTTP packets"""
    http_payload = f"{method} {path} HTTP/1.1\r\nHost: {host}\r\n\r\n".encode()
    
    config = PacketConfig(
        source_ip=source_ip,
        destination_ip=destination_ip,
        source_port=random.randint(1024, 65535),
        destination_port=80,
        protocol=PacketType.HTTP,
        payload=http_payload,
        **kwargs
    )
    
    crafter = PacketCrafter()
    return crafter.craft_packet(PacketType.HTTP, config)


def create_custom_packet(
    source_ip: str = "192.168.1.100",
    destination_ip: str = "192.168.1.1",
    custom_layers: Optional[List[Dict[str, Any]]] = None,
    payload: Optional[bytes] = None,
    **kwargs
) -> PacketResult:
    """Convenience function for creating custom packets"""
    config = PacketConfig(
        source_ip=source_ip,
        destination_ip=destination_ip,
        protocol=PacketType.CUSTOM,
        payload=payload,
        options={'custom_layers': custom_layers or []},
        **kwargs
    )
    
    crafter = PacketCrafter()
    return crafter.craft_packet(PacketType.CUSTOM, config)


# Example usage
if __name__ == "__main__":
    # Example packet crafting
    print("📦 Packet Crafting Example")
    
    crafter = PacketCrafter()
    
    # Create TCP packet
    print("\n" + "="*60)
    print("TCP PACKET")
    print("="*60)
    
    tcp_config = PacketConfig(
        source_ip="192.168.1.100",
        destination_ip="192.168.1.1",
        source_port=12345,
        destination_port=80,
        payload=b"Hello, World!",
        flags="S"  # SYN flag
    )
    
    tcp_result = crafter.craft_packet(PacketType.TCP, tcp_config)
    if tcp_result.crafted:
        print(f"✅ TCP packet crafted: {len(tcp_result.packet)} bytes")
        print(f"   Source: {tcp_result.packet[IP].src}:{tcp_result.packet[TCP].sport}")
        print(f"   Destination: {tcp_result.packet[IP].dst}:{tcp_result.packet[TCP].dport}")
        print(f"   Flags: {tcp_result.packet[TCP].flags}")
    else:
        print(f"❌ TCP packet crafting failed: {tcp_result.error}")
    
    # Create UDP packet
    print("\n" + "="*60)
    print("UDP PACKET")
    print("="*60)
    
    udp_result = create_udp_packet(
        source_ip="192.168.1.100",
        destination_ip="8.8.8.8",
        source_port=54321,
        destination_port=53,
        payload=b"DNS query"
    )
    
    if udp_result.crafted:
        print(f"✅ UDP packet crafted: {len(udp_result.packet)} bytes")
        print(f"   Source: {udp_result.packet[IP].src}:{udp_result.packet[UDP].sport}")
        print(f"   Destination: {udp_result.packet[IP].dst}:{udp_result.packet[UDP].dport}")
    else:
        print(f"❌ UDP packet crafting failed: {udp_result.error}")
    
    # Create ICMP packet
    print("\n" + "="*60)
    print("ICMP PACKET")
    print("="*60)
    
    icmp_result = create_icmp_packet(
        source_ip="192.168.1.100",
        destination_ip="192.168.1.1",
        icmp_type=8,  # Echo request
        icmp_code=0
    )
    
    if icmp_result.crafted:
        print(f"✅ ICMP packet crafted: {len(icmp_result.packet)} bytes")
        print(f"   Source: {icmp_result.packet[IP].src}")
        print(f"   Destination: {icmp_result.packet[IP].dst}")
        print(f"   Type: {icmp_result.packet[ICMP].type}")
        print(f"   Code: {icmp_result.packet[ICMP].code}")
    else:
        print(f"❌ ICMP packet crafting failed: {icmp_result.error}")
    
    # Create ARP packet
    print("\n" + "="*60)
    print("ARP PACKET")
    print("="*60)
    
    arp_result = create_arp_packet(
        source_ip="192.168.1.100",
        destination_ip="192.168.1.1",
        arp_op=1  # ARP request
    )
    
    if arp_result.crafted:
        print(f"✅ ARP packet crafted: {len(arp_result.packet)} bytes")
        print(f"   Source IP: {arp_result.packet[ARP].psrc}")
        print(f"   Destination IP: {arp_result.packet[ARP].pdst}")
        print(f"   Operation: {arp_result.packet[ARP].op}")
    else:
        print(f"❌ ARP packet crafting failed: {arp_result.error}")
    
    # Create multiple packets
    print("\n" + "="*60)
    print("MULTIPLE PACKETS")
    print("="*60)
    
    multiple_results = crafter.craft_multiple_packets(PacketType.TCP, 3)
    print(f"✅ Created {len(multiple_results)} TCP packets")
    
    for i, result in enumerate(multiple_results):
        if result.crafted:
            print(f"   Packet {i+1}: {result.packet[TCP].sport} -> {result.packet[TCP].dport}")
        else:
            print(f"   Packet {i+1}: Failed - {result.error}")
    
    print("\n✅ Packet crafting example completed!") 