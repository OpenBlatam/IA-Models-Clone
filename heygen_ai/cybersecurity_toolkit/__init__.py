from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from .scanners.port_scanner import (
from .scanners.vulnerability_scanner import (
from .scanners.web_scanner import (
from .enumerators.dns_enumerator import (
from .enumerators.smb_enumerator import (
from .enumerators.ssh_enumerator import (
from .attackers.brute_forcers import (
from .attackers.exploiters import (
from .reporting.console_reporter import (
from .reporting.html_reporter import (
from .reporting.json_reporter import (
from .utils.crypto_helpers import (
from .utils.network_helpers import (
from .cybersecurity_toolkit import (
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Cybersecurity Toolkit
====================

A comprehensive cybersecurity toolkit with modular architecture,
named exports, RORO pattern, and proper async/def usage.
"""

# Named exports for scanners
    scan_single_port,
    scan_port_range,
    scan_common_ports,
    PortScanResult,
    AsyncPortScanner
)

    scan_web_vulnerabilities,
    scan_sql_injection,
    scan_xss_vulnerabilities,
    VulnerabilityFinding,
    WebVulnerabilityScanner
)

    scan_web_application,
    check_security_headers,
    scan_ssl_certificate,
    WebScanResult,
    WebSecurityScanner
)

# Named exports for enumerators
    enumerate_dns_records,
    perform_dns_zone_transfer,
    check_dns_brute_force,
    DnsEnumerationResult,
    DnsEnumerator
)

    enumerate_smb_shares,
    check_smb_vulnerabilities,
    SmbEnumerationResult,
    SmbEnumerator
)

    enumerate_ssh_services,
    check_ssh_configuration,
    SshEnumerationResult,
    SshEnumerator
)

# Named exports for attackers
    brute_force_password,
    brute_force_ssh,
    brute_force_web_login,
    BruteForceResult,
    PasswordBruteForcer
)

    exploit_vulnerability,
    generate_exploit_payload,
    ExploitResult,
    VulnerabilityExploiter
)

# Named exports for reporting
    generate_console_report,
    ConsoleReporter
)

    generate_html_report,
    HtmlReporter
)

    generate_json_report,
    JsonReporter
)

# Named exports for utils
    hash_password,
    encrypt_data,
    decrypt_data,
    generate_secure_key,
    CryptoHelper
)

    check_host_connectivity,
    resolve_hostname,
    validate_ip_address,
    NetworkHelper
)

# Main toolkit class
    CybersecurityToolkit,
    create_toolkit_instance
)

# Version information
__version__ = "1.0.0"
__author__ = "Cybersecurity Team"
__description__ = "Comprehensive cybersecurity toolkit with modular architecture"

# Export all public interfaces
__all__ = [
    # Scanners
    "scan_single_port",
    "scan_port_range", 
    "scan_common_ports",
    "PortScanResult",
    "AsyncPortScanner",
    "scan_web_vulnerabilities",
    "scan_sql_injection",
    "scan_xss_vulnerabilities",
    "VulnerabilityFinding",
    "WebVulnerabilityScanner",
    "scan_web_application",
    "check_security_headers",
    "scan_ssl_certificate",
    "WebScanResult",
    "WebSecurityScanner",
    
    # Enumerators
    "enumerate_dns_records",
    "perform_dns_zone_transfer",
    "check_dns_brute_force",
    "DnsEnumerationResult",
    "DnsEnumerator",
    "enumerate_smb_shares",
    "check_smb_vulnerabilities",
    "SmbEnumerationResult",
    "SmbEnumerator",
    "enumerate_ssh_services",
    "check_ssh_configuration",
    "SshEnumerationResult",
    "SshEnumerator",
    
    # Attackers
    "brute_force_password",
    "brute_force_ssh",
    "brute_force_web_login",
    "BruteForceResult",
    "PasswordBruteForcer",
    "exploit_vulnerability",
    "generate_exploit_payload",
    "ExploitResult",
    "VulnerabilityExploiter",
    
    # Reporting
    "generate_console_report",
    "ConsoleReporter",
    "generate_html_report",
    "HtmlReporter",
    "generate_json_report",
    "JsonReporter",
    
    # Utils
    "hash_password",
    "encrypt_data",
    "decrypt_data",
    "generate_secure_key",
    "CryptoHelper",
    "check_host_connectivity",
    "resolve_hostname",
    "validate_ip_address",
    "NetworkHelper",
    
    # Main toolkit
    "CybersecurityToolkit",
    "create_toolkit_instance"
] 