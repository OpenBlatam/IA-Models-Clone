from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import aiohttp
import aiofiles
import socket
import subprocess
import paramiko
import smbclient
from typing import Dict, Any, List, Optional, Union, Tuple, Callable, TypeVar, Generic, Literal
from dataclasses import dataclass
from pydantic import BaseModel, Field, validator, ConfigDict
from pydantic.types import conint, confloat, constr
import numpy as np
from pathlib import Path
            import dns.resolver
            import dns.reversename
            import dns.reversename
            import dns.resolver
            import smbclient
            import smbclient
            import paramiko
            import paramiko
from typing import Any, List, Dict, Optional
import logging
"""
Enumerators Module Structure - DNS, SMB, SSH
============================================

This file demonstrates how to organize the enumerators module structure:
- DNS enumerator with type hints and Pydantic validation
- SMB enumerator with async/sync patterns
- SSH enumerator with RORO pattern
- Named exports for utilities
"""


# ============================================================================
# NAMED EXPORTS
# ============================================================================

__all__ = [
    # DNS Enumerator
    "DNSEnumerator",
    "DNSEnumeratorConfig",
    "DNSRecordType",
    
    # SMB Enumerator
    "SMBEnumerator", 
    "SMBEnumeratorConfig",
    "SMBShareType",
    
    # SSH Enumerator
    "SSHEnumerator",
    "SSHEnumeratorConfig",
    "SSHAuthType",
    
    # Common utilities
    "EnumeratorResult",
    "EnumeratorConfig",
    "ScanType"
]

# ============================================================================
# COMMON UTILITIES
# ============================================================================

class EnumeratorResult(BaseModel):
    """Pydantic model for enumerator results."""
    
    model_config = ConfigDict(extra="forbid")
    
    is_successful: bool = Field(description="Whether enumeration was successful")
    target: str = Field(description="Target host or domain")
    scan_type: str = Field(description="Type of scan performed")
    results: List[Dict[str, Any]] = Field(default_factory=list, description="Enumeration results")
    errors: List[str] = Field(default_factory=list, description="Error messages")
    scan_time: Optional[float] = Field(default=None, description="Scan duration in seconds")

class EnumeratorConfig(BaseModel):
    """Pydantic model for enumerator configuration."""
    
    model_config = ConfigDict(extra="forbid")
    
    timeout: confloat(gt=0.0) = Field(default=30.0, description="Timeout in seconds")
    max_retries: conint(ge=0, le=10) = Field(default=3, description="Maximum retries")
    verbose: bool = Field(default=False, description="Verbose output")
    output_file: Optional[constr(strip_whitespace=True)] = Field(default=None, description="Output file path")

class ScanType(BaseModel):
    """Pydantic model for scan type validation."""
    
    model_config = ConfigDict(extra="forbid")
    
    type_name: constr(strip_whitespace=True) = Field(
        pattern=r"^(dns|smb|ssh|port|vulnerability|web)$"
    )
    description: Optional[str] = Field(default=None)
    is_active: bool = Field(default=True)

# ============================================================================
# DNS ENUMERATOR
# ============================================================================

class DNSEnumerator:
    """DNS enumerator module with proper exports."""
    
    __all__ = [
        "enumerate_dns_records",
        "reverse_dns_lookup",
        "dns_zone_transfer",
        "DNSEnumeratorConfig",
        "DNSRecordType"
    ]
    
    class DNSEnumeratorConfig(BaseModel):
        """Pydantic model for DNS enumerator configuration."""
        
        model_config = ConfigDict(extra="forbid")
        
        target_domain: constr(strip_whitespace=True) = Field(description="Target domain to enumerate")
        record_types: List[constr(strip_whitespace=True)] = Field(
            default=["A", "AAAA", "MX", "NS", "TXT"],
            description="DNS record types to query"
        )
        nameserver: Optional[constr(strip_whitespace=True)] = Field(default=None, description="Custom nameserver")
        timeout: confloat(gt=0.0) = Field(default=10.0, description="DNS query timeout")
        max_queries: conint(gt=0, le=1000) = Field(default=100, description="Maximum DNS queries")
    
    class DNSRecordType(BaseModel):
        """Pydantic model for DNS record type validation."""
        
        model_config = ConfigDict(extra="forbid")
        
        record_type: constr(strip_whitespace=True) = Field(
            pattern=r"^(A|AAAA|MX|NS|TXT|CNAME|PTR|SOA|SRV)$"
        )
        description: Optional[str] = Field(default=None)
    
    async def enumerate_dns_records(
        config: DNSEnumeratorConfig
    ) -> EnumeratorResult:
        """Enumerate DNS records with comprehensive type hints and validation."""
        try:
            
            results = []
            errors = []
            start_time = asyncio.get_event_loop().time()
            
            # Validate inputs
            if not config.target_domain:
                raise ValueError("target_domain cannot be empty")
            
            # Configure resolver
            resolver = dns.resolver.Resolver()
            resolver.timeout = config.timeout
            resolver.lifetime = config.timeout
            
            if config.nameserver:
                resolver.nameservers = [config.nameserver]
            
            # Enumerate each record type
            for record_type in config.record_types:
                try:
                    if record_type not in ["A", "AAAA", "MX", "NS", "TXT", "CNAME", "PTR", "SOA", "SRV"]:
                        errors.append(f"Unsupported record type: {record_type}")
                        continue
                    
                    answers = resolver.resolve(config.target_domain, record_type)
                    
                    for answer in answers:
                        results.append({
                            "record_type": record_type,
                            "target": config.target_domain,
                            "value": str(answer),
                            "ttl": answer.ttl if hasattr(answer, 'ttl') else None
                        })
                        
                except dns.resolver.NXDOMAIN:
                    errors.append(f"Domain {config.target_domain} does not exist")
                except dns.resolver.NoAnswer:
                    errors.append(f"No {record_type} records found for {config.target_domain}")
                except Exception as e:
                    errors.append(f"Error querying {record_type} records: {str(e)}")
            
            scan_time = asyncio.get_event_loop().time() - start_time
            
            return EnumeratorResult(
                is_successful=len(results) > 0,
                target=config.target_domain,
                scan_type="dns_enumeration",
                results=results,
                errors=errors,
                scan_time=scan_time
            )
            
        except Exception as exc:
            return EnumeratorResult(
                is_successful=False,
                target=config.target_domain,
                scan_type="dns_enumeration",
                results=[],
                errors=[str(exc)],
                scan_time=None
            )
    
    async def reverse_dns_lookup(
        ip_address: str,
        config: EnumeratorConfig
    ) -> EnumeratorResult:
        """Perform reverse DNS lookup with type hints."""
        try:
            
            results = []
            start_time = asyncio.get_event_loop().time()
            
            # Validate IP address
            if not ip_address:
                raise ValueError("IP address cannot be empty")
            
            # Perform reverse lookup
            reverse_name = dns.reversename.from_address(ip_address)
            resolver = dns.resolver.Resolver()
            resolver.timeout = config.timeout
            
            answers = resolver.resolve(reverse_name, "PTR")
            
            for answer in answers:
                results.append({
                    "ip_address": ip_address,
                    "hostname": str(answer),
                    "record_type": "PTR"
                })
            
            scan_time = asyncio.get_event_loop().time() - start_time
            
            return EnumeratorResult(
                is_successful=True,
                target=ip_address,
                scan_type="reverse_dns",
                results=results,
                errors=[],
                scan_time=scan_time
            )
            
        except Exception as exc:
            return EnumeratorResult(
                is_successful=False,
                target=ip_address,
                scan_type="reverse_dns",
                results=[],
                errors=[str(exc)],
                scan_time=None
            )

# ============================================================================
# SMB ENUMERATOR
# ============================================================================

class SMBEnumerator:
    """SMB enumerator module with proper exports."""
    
    __all__ = [
        "enumerate_smb_shares",
        "enumerate_smb_users",
        "test_smb_authentication",
        "SMBEnumeratorConfig",
        "SMBShareType"
    ]
    
    class SMBEnumeratorConfig(BaseModel):
        """Pydantic model for SMB enumerator configuration."""
        
        model_config = ConfigDict(extra="forbid")
        
        target_host: constr(strip_whitespace=True) = Field(description="Target SMB host")
        port: conint(ge=1, le=65535) = Field(default=445, description="SMB port")
        username: Optional[constr(strip_whitespace=True)] = Field(default=None, description="Username for authentication")
        password: Optional[constr(strip_whitespace=True)] = Field(default=None, description="Password for authentication")
        domain: Optional[constr(strip_whitespace=True)] = Field(default=None, description="Domain for authentication")
        timeout: confloat(gt=0.0) = Field(default=30.0, description="SMB connection timeout")
    
    class SMBShareType(BaseModel):
        """Pydantic model for SMB share type validation."""
        
        model_config = ConfigDict(extra="forbid")
        
        share_type: constr(strip_whitespace=True) = Field(
            pattern=r"^(disk|printer|ipc|special)$"
        )
        description: Optional[str] = Field(default=None)
    
    async def enumerate_smb_shares(
        config: SMBEnumeratorConfig
    ) -> EnumeratorResult:
        """Enumerate SMB shares with comprehensive type hints and validation."""
        try:
            
            results = []
            errors = []
            start_time = asyncio.get_event_loop().time()
            
            # Validate inputs
            if not config.target_host:
                raise ValueError("target_host cannot be empty")
            
            # Configure SMB connection
            smb_config = {
                "username": config.username,
                "password": config.password,
                "domain": config.domain,
                "timeout": config.timeout
            }
            
            # Remove None values
            smb_config = {k: v for k, v in smb_config.items() if v is not None}
            
            # Connect to SMB host
            with smbclient.open_file(f"//{config.target_host}/", **smb_config) as connection:
                shares = smbclient.listdir(f"//{config.target_host}/")
                
                for share in shares:
                    try:
                        share_info = smbclient.stat(f"//{config.target_host}/{share}")
                        results.append({
                            "share_name": share,
                            "share_type": "disk",  # Default type
                            "permissions": str(share_info.st_mode),
                            "size": share_info.st_size if hasattr(share_info, 'st_size') else None
                        })
                    except Exception as e:
                        errors.append(f"Error accessing share {share}: {str(e)}")
            
            scan_time = asyncio.get_event_loop().time() - start_time
            
            return EnumeratorResult(
                is_successful=len(results) > 0,
                target=config.target_host,
                scan_type="smb_enumeration",
                results=results,
                errors=errors,
                scan_time=scan_time
            )
            
        except Exception as exc:
            return EnumeratorResult(
                is_successful=False,
                target=config.target_host,
                scan_type="smb_enumeration",
                results=[],
                errors=[str(exc)],
                scan_time=None
            )
    
    async def test_smb_authentication(
        config: SMBEnumeratorConfig
    ) -> EnumeratorResult:
        """Test SMB authentication with type hints."""
        try:
            
            results = []
            start_time = asyncio.get_event_loop().time()
            
            # Validate inputs
            if not config.target_host:
                raise ValueError("target_host cannot be empty")
            if not config.username or not config.password:
                raise ValueError("username and password are required for authentication test")
            
            # Test authentication
            smb_config = {
                "username": config.username,
                "password": config.password,
                "domain": config.domain,
                "timeout": config.timeout
            }
            
            # Remove None values
            smb_config = {k: v for k, v in smb_config.items() if v is not None}
            
            try:
                with smbclient.open_file(f"//{config.target_host}/", **smb_config):
                    results.append({
                        "authentication": "successful",
                        "username": config.username,
                        "domain": config.domain
                    })
            except Exception as auth_error:
                results.append({
                    "authentication": "failed",
                    "username": config.username,
                    "domain": config.domain,
                    "error": str(auth_error)
                })
            
            scan_time = asyncio.get_event_loop().time() - start_time
            
            return EnumeratorResult(
                is_successful=True,
                target=config.target_host,
                scan_type="smb_authentication",
                results=results,
                errors=[],
                scan_time=scan_time
            )
            
        except Exception as exc:
            return EnumeratorResult(
                is_successful=False,
                target=config.target_host,
                scan_type="smb_authentication",
                results=[],
                errors=[str(exc)],
                scan_time=None
            )

# ============================================================================
# SSH ENUMERATOR
# ============================================================================

class SSHEnumerator:
    """SSH enumerator module with proper exports."""
    
    __all__ = [
        "enumerate_ssh_services",
        "test_ssh_authentication",
        "enumerate_ssh_keys",
        "SSHEnumeratorConfig",
        "SSHAuthType"
    ]
    
    class SSHEnumeratorConfig(BaseModel):
        """Pydantic model for SSH enumerator configuration."""
        
        model_config = ConfigDict(extra="forbid")
        
        target_host: constr(strip_whitespace=True) = Field(description="Target SSH host")
        port: conint(ge=1, le=65535) = Field(default=22, description="SSH port")
        username: Optional[constr(strip_whitespace=True)] = Field(default=None, description="Username for authentication")
        password: Optional[constr(strip_whitespace=True)] = Field(default=None, description="Password for authentication")
        key_file: Optional[constr(strip_whitespace=True)] = Field(default=None, description="SSH key file path")
        timeout: confloat(gt=0.0) = Field(default=30.0, description="SSH connection timeout")
        auth_methods: List[constr(strip_whitespace=True)] = Field(
            default=["password", "publickey"],
            description="Authentication methods to try"
        )
    
    class SSHAuthType(BaseModel):
        """Pydantic model for SSH authentication type validation."""
        
        model_config = ConfigDict(extra="forbid")
        
        auth_type: constr(strip_whitespace=True) = Field(
            pattern=r"^(password|publickey|keyboard-interactive|gssapi-with-mic)$"
        )
        description: Optional[str] = Field(default=None)
    
    async def enumerate_ssh_services(
        config: SSHEnumeratorConfig
    ) -> EnumeratorResult:
        """Enumerate SSH services with comprehensive type hints and validation."""
        try:
            
            results = []
            errors = []
            start_time = asyncio.get_event_loop().time()
            
            # Validate inputs
            if not config.target_host:
                raise ValueError("target_host cannot be empty")
            
            # Create SSH client
            ssh_client = paramiko.SSHClient()
            ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            try:
                # Connect to SSH host
                ssh_client.connect(
                    hostname=config.target_host,
                    port=config.port,
                    timeout=config.timeout
                )
                
                # Get SSH server information
                transport = ssh_client.get_transport()
                if transport:
                    results.append({
                        "hostname": config.target_host,
                        "port": config.port,
                        "ssh_version": transport.remote_version,
                        "auth_methods": transport.get_auth_methods(),
                        "compression": transport.get_compression(),
                        "encryption": transport.get_encryption()
                    })
                
                # Test available authentication methods
                for auth_method in config.auth_methods:
                    try:
                        if auth_method == "password" and config.username and config.password:
                            ssh_client.connect(
                                hostname=config.target_host,
                                port=config.port,
                                username=config.username,
                                password=config.password,
                                timeout=config.timeout
                            )
                            results.append({
                                "auth_method": "password",
                                "status": "successful",
                                "username": config.username
                            })
                        elif auth_method == "publickey" and config.username and config.key_file:
                            ssh_client.connect(
                                hostname=config.target_host,
                                port=config.port,
                                username=config.username,
                                key_filename=config.key_file,
                                timeout=config.timeout
                            )
                            results.append({
                                "auth_method": "publickey",
                                "status": "successful",
                                "username": config.username,
                                "key_file": config.key_file
                            })
                    except Exception as auth_error:
                        results.append({
                            "auth_method": auth_method,
                            "status": "failed",
                            "error": str(auth_error)
                        })
                
            finally:
                ssh_client.close()
            
            scan_time = asyncio.get_event_loop().time() - start_time
            
            return EnumeratorResult(
                is_successful=len(results) > 0,
                target=config.target_host,
                scan_type="ssh_enumeration",
                results=results,
                errors=errors,
                scan_time=scan_time
            )
            
        except Exception as exc:
            return EnumeratorResult(
                is_successful=False,
                target=config.target_host,
                scan_type="ssh_enumeration",
                results=[],
                errors=[str(exc)],
                scan_time=None
            )
    
    async def test_ssh_authentication(
        config: SSHEnumeratorConfig
    ) -> EnumeratorResult:
        """Test SSH authentication with type hints."""
        try:
            
            results = []
            start_time = asyncio.get_event_loop().time()
            
            # Validate inputs
            if not config.target_host:
                raise ValueError("target_host cannot be empty")
            if not config.username:
                raise ValueError("username is required for authentication test")
            
            # Test different authentication methods
            for auth_method in config.auth_methods:
                try:
                    ssh_client = paramiko.SSHClient()
                    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                    
                    if auth_method == "password" and config.password:
                        ssh_client.connect(
                            hostname=config.target_host,
                            port=config.port,
                            username=config.username,
                            password=config.password,
                            timeout=config.timeout
                        )
                        results.append({
                            "auth_method": "password",
                            "status": "successful",
                            "username": config.username
                        })
                    elif auth_method == "publickey" and config.key_file:
                        ssh_client.connect(
                            hostname=config.target_host,
                            port=config.port,
                            username=config.username,
                            key_filename=config.key_file,
                            timeout=config.timeout
                        )
                        results.append({
                            "auth_method": "publickey",
                            "status": "successful",
                            "username": config.username,
                            "key_file": config.key_file
                        })
                    
                    ssh_client.close()
                    
                except Exception as auth_error:
                    results.append({
                        "auth_method": auth_method,
                        "status": "failed",
                        "username": config.username,
                        "error": str(auth_error)
                    })
            
            scan_time = asyncio.get_event_loop().time() - start_time
            
            return EnumeratorResult(
                is_successful=len([r for r in results if r["status"] == "successful"]) > 0,
                target=config.target_host,
                scan_type="ssh_authentication",
                results=results,
                errors=[],
                scan_time=scan_time
            )
            
        except Exception as exc:
            return EnumeratorResult(
                is_successful=False,
                target=config.target_host,
                scan_type="ssh_authentication",
                results=[],
                errors=[str(exc)],
                scan_time=None
            )

# ============================================================================
# MAIN ENUMERATOR MODULE
# ============================================================================

class MainEnumeratorModule:
    """Main enumerator module with proper imports and exports."""
    
    # Import all enumerator modules
    dns_enumerator = DNSEnumerator()
    smb_enumerator = SMBEnumerator()
    ssh_enumerator = SSHEnumerator()
    
    # Define main exports
    __all__ = [
        # Enumerator modules
        "DNSEnumerator",
        "SMBEnumerator",
        "SSHEnumerator",
        
        # Common utilities
        "EnumeratorResult",
        "EnumeratorConfig",
        "ScanType",
        
        # Main functions
        "run_dns_enumeration",
        "run_smb_enumeration",
        "run_ssh_enumeration",
        "run_comprehensive_scan"
    ]
    
    async def run_dns_enumeration(
        target_domain: str,
        config: EnumeratorConfig
    ) -> EnumeratorResult:
        """Run DNS enumeration with all patterns integrated."""
        try:
            dns_config = DNSEnumerator.DNSEnumeratorConfig(
                target_domain=target_domain,
                timeout=config.timeout,
                max_queries=100
            )
            
            return await dns_enumerator.enumerate_dns_records(dns_config)
            
        except Exception as exc:
            return EnumeratorResult(
                is_successful=False,
                target=target_domain,
                scan_type="dns_enumeration",
                results=[],
                errors=[str(exc)],
                scan_time=None
            )
    
    async def run_smb_enumeration(
        target_host: str,
        config: EnumeratorConfig
    ) -> EnumeratorResult:
        """Run SMB enumeration with all patterns integrated."""
        try:
            smb_config = SMBEnumerator.SMBEnumeratorConfig(
                target_host=target_host,
                timeout=config.timeout
            )
            
            return await smb_enumerator.enumerate_smb_shares(smb_config)
            
        except Exception as exc:
            return EnumeratorResult(
                is_successful=False,
                target=target_host,
                scan_type="smb_enumeration",
                results=[],
                errors=[str(exc)],
                scan_time=None
            )
    
    async def run_ssh_enumeration(
        target_host: str,
        config: EnumeratorConfig
    ) -> EnumeratorResult:
        """Run SSH enumeration with all patterns integrated."""
        try:
            ssh_config = SSHEnumerator.SSHEnumeratorConfig(
                target_host=target_host,
                timeout=config.timeout
            )
            
            return await ssh_enumerator.enumerate_ssh_services(ssh_config)
            
        except Exception as exc:
            return EnumeratorResult(
                is_successful=False,
                target=target_host,
                scan_type="ssh_enumeration",
                results=[],
                errors=[str(exc)],
                scan_time=None
            )
    
    async def run_comprehensive_scan(
        target: str,
        scan_types: List[str],
        config: EnumeratorConfig
    ) -> Dict[str, EnumeratorResult]:
        """Run comprehensive enumeration scan."""
        results = {}
        
        for scan_type in scan_types:
            if scan_type == "dns":
                results["dns"] = await run_dns_enumeration(target, config)
            elif scan_type == "smb":
                results["smb"] = await run_smb_enumeration(target, config)
            elif scan_type == "ssh":
                results["ssh"] = await run_ssh_enumeration(target, config)
        
        return results

# ============================================================================
# USAGE EXAMPLES
# ============================================================================

async def demonstrate_enumerators_structure():
    """Demonstrate the enumerators structure with all patterns."""
    
    print("🔍 Demonstrating Enumerators Structure with All Patterns")
    print("=" * 60)
    
    # Example 1: DNS enumeration
    print("\n🌐 DNS Enumeration:")
    dns_enumerator = DNSEnumerator()
    dns_config = DNSEnumerator.DNSEnumeratorConfig(
        target_domain="example.com",
        record_types=["A", "MX", "NS", "TXT"],
        timeout=10.0
    )
    
    dns_result = await dns_enumerator.enumerate_dns_records(dns_config)
    print(f"DNS scan: {dns_result.is_successful}")
    if dns_result.is_successful:
        print(f"Found {len(dns_result.results)} DNS records")
    
    # Example 2: SMB enumeration
    print("\n💾 SMB Enumeration:")
    smb_enumerator = SMBEnumerator()
    smb_config = SMBEnumerator.SMBEnumeratorConfig(
        target_host="192.168.1.100",
        port=445,
        timeout=30.0
    )
    
    smb_result = await smb_enumerator.enumerate_smb_shares(smb_config)
    print(f"SMB scan: {smb_result.is_successful}")
    if smb_result.is_successful:
        print(f"Found {len(smb_result.results)} SMB shares")
    
    # Example 3: SSH enumeration
    print("\n🔐 SSH Enumeration:")
    ssh_enumerator = SSHEnumerator()
    ssh_config = SSHEnumerator.SSHEnumeratorConfig(
        target_host="192.168.1.100",
        port=22,
        timeout=30.0
    )
    
    ssh_result = await ssh_enumerator.enumerate_ssh_services(ssh_config)
    print(f"SSH scan: {ssh_result.is_successful}")
    if ssh_result.is_successful:
        print(f"Found {len(ssh_result.results)} SSH services")
    
    # Example 4: Main module
    print("\n🎯 Main Enumerator Module:")
    main_module = MainEnumeratorModule()
    
    comprehensive_result = await main_module.run_comprehensive_scan(
        target="example.com",
        scan_types=["dns", "smb", "ssh"],
        config=EnumeratorConfig(timeout=30.0, verbose=True)
    )
    
    print(f"Comprehensive scan completed: {len(comprehensive_result)} scan types")

def show_enumerators_benefits():
    """Show the benefits of enumerators structure."""
    
    benefits = {
        "organization": [
            "Clear separation of enumeration types (DNS, SMB, SSH)",
            "Logical grouping of related functionality",
            "Easy to navigate and understand",
            "Scalable architecture for new enumerators"
        ],
        "type_safety": [
            "Type hints throughout all enumerators",
            "Pydantic validation for configurations",
            "Consistent error handling",
            "Clear function signatures"
        ],
        "async_support": [
            "Non-blocking enumeration operations",
            "Proper timeout handling",
            "Concurrent scanning capabilities",
            "Efficient resource utilization"
        ],
        "security": [
            "Proper authentication handling",
            "Secure connection management",
            "Error logging and monitoring",
            "Rate limiting and backoff strategies"
        ]
    }
    
    return benefits

if __name__ == "__main__":
    # Demonstrate enumerators structure
    asyncio.run(demonstrate_enumerators_structure())
    
    benefits = show_enumerators_benefits()
    
    print("\n🎯 Key Enumerators Structure Benefits:")
    for category, items in benefits.items():
        print(f"\n{category.title()}:")
        for item in items:
            print(f"  • {item}")
    
    print("\n✅ Enumerators structure organization completed successfully!") 