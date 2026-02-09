"""
Network Module for Video-OpusClip
Packet crafting, sniffing, and network analysis using Scapy
"""

from .packet_crafter import (
    PacketCrafter, TCPPacketCrafter, UDPPacketCrafter, ICMPPacketCrafter,
    ARPPacketCrafter, DNSPacketCrafter, HTTPPacketCrafter, CustomPacketCrafter,
    create_tcp_packet, create_udp_packet, create_icmp_packet, create_arp_packet,
    create_dns_packet, create_http_packet, create_custom_packet,
    PacketType, ProtocolType, PacketDirection
)

from .packet_sniffer import (
    PacketSniffer, NetworkSniffer, ProtocolSniffer, FilterSniffer,
    start_sniffing, stop_sniffing, capture_packets, analyze_packets,
    SnifferConfig, CaptureFilter, AnalysisResult
)

from .packet_analyzer import (
    PacketAnalyzer, ProtocolAnalyzer, TrafficAnalyzer, SecurityAnalyzer,
    analyze_packet, analyze_traffic, detect_anomalies, extract_metadata,
    AnalysisType, TrafficPattern, SecurityThreat
)

from .packet_injector import (
    PacketInjector, NetworkInjector, ProtocolInjector, CustomInjector,
    inject_packet, inject_traffic, replay_packets, flood_network,
    InjectionMode, InjectionConfig, InjectionResult
)

from .network_scanner import (
    NetworkScanner, PortScanner, ServiceScanner, VulnerabilityScanner,
    scan_network, scan_ports, scan_services, scan_vulnerabilities,
    ScanType, ScanResult, ServiceInfo, VulnerabilityInfo
)

from .protocol_handlers import (
    TCPHandler, UDPHandler, ICMPHandler, ARPHandler, DNSHandler,
    HTTPHandler, HTTPSHandler, FTPHandler, SMTPHandler, SSHHandler,
    ProtocolHandler, HandlerConfig, HandlerResult
)

from .network_utils import (
    NetworkUtils, IPUtils, PortUtils, ProtocolUtils, AddressUtils,
    get_interface_info, get_route_info, get_arp_table, get_connection_info,
    NetworkInfo, InterfaceInfo, RouteInfo, ConnectionInfo
)

from .packet_filters import (
    PacketFilter, IPFilter, PortFilter, ProtocolFilter, CustomFilter,
    create_filter, apply_filter, chain_filters, optimize_filters,
    FilterType, FilterRule, FilterChain
)

from .packet_modifiers import (
    PacketModifier, HeaderModifier, PayloadModifier, ChecksumModifier,
    modify_packet, modify_header, modify_payload, recalculate_checksum,
    ModificationType, ModificationRule, ModificationResult
)

from .network_monitor import (
    NetworkMonitor, TrafficMonitor, PerformanceMonitor, SecurityMonitor,
    start_monitoring, stop_monitoring, get_statistics, generate_report,
    MonitorConfig, MonitorResult, Statistics, Report
)

from .network_replay import (
    NetworkReplay, PacketReplay, TrafficReplay, SessionReplay,
    replay_pcap, replay_session, replay_traffic, analyze_replay,
    ReplayConfig, ReplayMode, ReplayResult
)

from .network_exceptions import (
    NetworkError, PacketError, SnifferError, InjectorError,
    ScannerError, HandlerError, FilterError, MonitorError
)

__all__ = [
    # Packet Crafting
    'PacketCrafter', 'TCPPacketCrafter', 'UDPPacketCrafter', 'ICMPPacketCrafter',
    'ARPPacketCrafter', 'DNSPacketCrafter', 'HTTPPacketCrafter', 'CustomPacketCrafter',
    'create_tcp_packet', 'create_udp_packet', 'create_icmp_packet', 'create_arp_packet',
    'create_dns_packet', 'create_http_packet', 'create_custom_packet',
    'PacketType', 'ProtocolType', 'PacketDirection',
    
    # Packet Sniffing
    'PacketSniffer', 'NetworkSniffer', 'ProtocolSniffer', 'FilterSniffer',
    'start_sniffing', 'stop_sniffing', 'capture_packets', 'analyze_packets',
    'SnifferConfig', 'CaptureFilter', 'AnalysisResult',
    
    # Packet Analysis
    'PacketAnalyzer', 'ProtocolAnalyzer', 'TrafficAnalyzer', 'SecurityAnalyzer',
    'analyze_packet', 'analyze_traffic', 'detect_anomalies', 'extract_metadata',
    'AnalysisType', 'TrafficPattern', 'SecurityThreat',
    
    # Packet Injection
    'PacketInjector', 'NetworkInjector', 'ProtocolInjector', 'CustomInjector',
    'inject_packet', 'inject_traffic', 'replay_packets', 'flood_network',
    'InjectionMode', 'InjectionConfig', 'InjectionResult',
    
    # Network Scanning
    'NetworkScanner', 'PortScanner', 'ServiceScanner', 'VulnerabilityScanner',
    'scan_network', 'scan_ports', 'scan_services', 'scan_vulnerabilities',
    'ScanType', 'ScanResult', 'ServiceInfo', 'VulnerabilityInfo',
    
    # Protocol Handlers
    'TCPHandler', 'UDPHandler', 'ICMPHandler', 'ARPHandler', 'DNSHandler',
    'HTTPHandler', 'HTTPSHandler', 'FTPHandler', 'SMTPHandler', 'SSHHandler',
    'ProtocolHandler', 'HandlerConfig', 'HandlerResult',
    
    # Network Utils
    'NetworkUtils', 'IPUtils', 'PortUtils', 'ProtocolUtils', 'AddressUtils',
    'get_interface_info', 'get_route_info', 'get_arp_table', 'get_connection_info',
    'NetworkInfo', 'InterfaceInfo', 'RouteInfo', 'ConnectionInfo',
    
    # Packet Filters
    'PacketFilter', 'IPFilter', 'PortFilter', 'ProtocolFilter', 'CustomFilter',
    'create_filter', 'apply_filter', 'chain_filters', 'optimize_filters',
    'FilterType', 'FilterRule', 'FilterChain',
    
    # Packet Modifiers
    'PacketModifier', 'HeaderModifier', 'PayloadModifier', 'ChecksumModifier',
    'modify_packet', 'modify_header', 'modify_payload', 'recalculate_checksum',
    'ModificationType', 'ModificationRule', 'ModificationResult',
    
    # Network Monitor
    'NetworkMonitor', 'TrafficMonitor', 'PerformanceMonitor', 'SecurityMonitor',
    'start_monitoring', 'stop_monitoring', 'get_statistics', 'generate_report',
    'MonitorConfig', 'MonitorResult', 'Statistics', 'Report',
    
    # Network Replay
    'NetworkReplay', 'PacketReplay', 'TrafficReplay', 'SessionReplay',
    'replay_pcap', 'replay_session', 'replay_traffic', 'analyze_replay',
    'ReplayConfig', 'ReplayMode', 'ReplayResult',
    
    # Network Exceptions
    'NetworkError', 'PacketError', 'SnifferError', 'InjectorError',
    'ScannerError', 'HandlerError', 'FilterError', 'MonitorError'
] 