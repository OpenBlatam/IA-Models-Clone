# Enumerators Module Structure - Complete Guide
=============================================

## Overview

This document outlines how to organize the enumerators module structure with DNS, SMB, and SSH enumerators, integrating all patterns: type hints, Pydantic validation, async/sync patterns, RORO pattern, and named exports.

## Table of Contents

1. [Recommended Enumerators Structure](#recommended-enumerators-structure)
2. [DNS Enumerator](#dns-enumerator)
3. [SMB Enumerator](#smb-enumerator)
4. [SSH Enumerator](#ssh-enumerator)
5. [Common Utilities](#common-utilities)
6. [Main Enumerator Module](#main-enumerator-module)
7. [Import/Export Patterns](#importexport-patterns)
8. [Best Practices](#best-practices)
9. [Integration with Existing Patterns](#integration-with-existing-patterns)
10. [Security Considerations](#security-considerations)

## Recommended Enumerators Structure

```
enumerators/
├── __init__.py
├── dns/
│   ├── __init__.py
│   ├── dns_enumerator.py
│   ├── dns_config.py
│   └── dns_utils.py
├── smb/
│   ├── __init__.py
│   ├── smb_enumerator.py
│   ├── smb_config.py
│   └── smb_utils.py
├── ssh/
│   ├── __init__.py
│   ├── ssh_enumerator.py
│   ├── ssh_config.py
│   └── ssh_utils.py
├── common/
│   ├── __init__.py
│   ├── base_enumerator.py
│   ├── result_models.py
│   └── config_models.py
└── utils/
    ├── __init__.py
    ├── async_helpers.py
    ├── error_handlers.py
    └── logging_utils.py
```

## DNS Enumerator

### DNS Enumerator Module

**`enumerators/dns/dns_enumerator.py`:**
```python
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, ConfigDict
from pydantic.types import conint, confloat, constr
import asyncio
import dns.resolver
import dns.reversename

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
    max_queries: conint(gt Tu=0, le=1000) = Field(default=100, description="Maximum DNS queries")

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
```

### DNS Configuration Module

**`enumerators/dns/dns_config.py`:**
```python
from pydantic import BaseModel, Field, ConfigDict
from pydantic.types import conint, confloat, constr
from typing import List, Optional

class DNSConfig(BaseModel):
    """Pydantic model for DNS configuration."""
    
    model_config = ConfigDict(extra="forbid")
    
    # Basic settings
    target_domain: constr(strip_whitespace=True) = Field(description="Target domain")
    record_types: List[constr(strip_whitespace=True)] = Field(
        default=["A", "AAAA", "MX", "NS", "TXT"],
        description="DNS record types to query"
    )
    
    # Advanced settings
    nameserver: Optional[constr(strip_whitespace=True)] = Field(default=None)
    timeout: confloat(gt=0.0) = Field(default=10.0)
    max_queries: conint(gt=0, le=1000) = Field(default=100)
    retry_count: conint(ge=0, le=5) = Field(default=3)
    
    # Validation settings
    validate_records: bool = Field(default=True)
    check_zone_transfer: bool = Field(default=False)
    aggressive_scanning: bool = Field(default=False)
```

## SMB Enumerator

### SMB Enumerator Module

**`enumerators/smb/smb_enumerator.py`:**
```python
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, ConfigDict
from pydantic.types import conint, confloat, constr
import asyncio
import smbclient

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
```

### SMB Configuration Module

**`enumerators/smb/smb_config.py`:**
```python
from pydantic import BaseModel, Field, ConfigDict
from pydantic.types import conint, confloat, constr
from typing import Optional

class SMBConfig(BaseModel):
    """Pydantic model for SMB configuration."""
    
    model_config = ConfigDict(extra="forbid")
    
    # Basic settings
    target_host: constr(strip_whitespace=True) = Field(description="Target SMB host")
    port: conint(ge=1, le=65535) = Field(default=445, description="SMB port")
    
    # Authentication settings
    username: Optional[constr(strip_whitespace=True)] = Field(default=None)
    password: Optional[constr(strip_whitespace=True)] = Field(default=None)
    domain: Optional[constr(strip_whitespace=True)] = Field(default=None)
    
    # Connection settings
    timeout: confloat(gt=0.0) = Field(default=30.0)
    retry_count: conint(ge=0, le=5) = Field(default=3)
    
    # Enumeration settings
    enumerate_shares: bool = Field(default=True)
    enumerate_users: bool = Field(default=False)
    test_authentication: bool = Field(default=False)
    aggressive_scanning: bool = Field(default=False)
```

## SSH Enumerator

### SSH Enumerator Module

**`enumerators/ssh/ssh_enumerator.py`:**
```python
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, ConfigDict
from pydantic.types import conint, confloat, constr
import asyncio
import paramiko

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
```

### SSH Configuration Module

**`enumerators/ssh/ssh_config.py`:**
```python
from pydantic import BaseModel, Field, ConfigDict
from pydantic.types import conint, confloat, constr
from typing import List, Optional

class SSHConfig(BaseModel):
    """Pydantic model for SSH configuration."""
    
    model_config = ConfigDict(extra="forbid")
    
    # Basic settings
    target_host: constr(strip_whitespace=True) = Field(description="Target SSH host")
    port: conint(ge=1, le=65535) = Field(default=22, description="SSH port")
    
    # Authentication settings
    username: Optional[constr(strip_whitespace=True)] = Field(default=None)
    password: Optional[constr(strip_whitespace=True)] = Field(default=None)
    key_file: Optional[constr(strip_whitespace=True)] = Field(default=None)
    
    # Connection settings
    timeout: confloat(gt=0.0) = Field(default=30.0)
    retry_count: conint(ge=0, le=5) = Field(default=3)
    
    # Enumeration settings
    auth_methods: List[constr(strip_whitespace=True)] = Field(
        default=["password", "publickey"],
        description="Authentication methods to try"
    )
    enumerate_services: bool = Field(default=True)
    test_authentication: bool = Field(default=False)
    enumerate_keys: bool = Field(default=False)
    aggressive_scanning: bool = Field(default=False)
```

## Common Utilities

### Base Enumerator Module

**`enumerators/common/base_enumerator.py`:**
```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, ConfigDict
from pydantic.types import confloat, conint

class BaseEnumerator(ABC):
    """Base class for all enumerators."""
    
    def __init__(self, config: BaseModel):
        self.config = config
        self.results = []
        self.errors = []
    
    @abstractmethod
    async def enumerate(self) -> "EnumeratorResult":
        """Perform enumeration."""
        pass
    
    @abstractmethod
    def validate_config(self) -> bool:
        """Validate configuration."""
        pass
    
    def add_result(self, result: Dict[str, Any]) -> None:
        """Add result to enumeration."""
        self.results.append(result)
    
    def add_error(self, error: str) -> None:
        """Add error to enumeration."""
        self.errors.append(error)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get enumeration summary."""
        return {
            "total_results": len(self.results),
            "total_errors": len(self.errors),
            "success_rate": len(self.results) / (len(self.results) + len(self.errors)) if (len(self.results) + len(self.errors)) > 0 else 0
        }
```

### Result Models Module

**`enumerators/common/result_models.py`:**
```python
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, ConfigDict
from pydantic.types import confloat

class EnumeratorResult(BaseModel):
    """Pydantic model for enumerator results."""
    
    model_config = ConfigDict(extra="forbid")
    
    is_successful: bool = Field(description="Whether enumeration was successful")
    target: str = Field(description="Target host or domain")
    scan_type: str = Field(description="Type of scan performed")
    results: List[Dict[str, Any]] = Field(default_factory=list, description="Enumeration results")
    errors: List[str] = Field(default_factory=list, description="Error messages")
    scan_time: Optional[float] = Field(default=None, description="Scan duration in seconds")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")

class EnumeratorConfig(BaseModel):
    """Pydantic model for enumerator configuration."""
    
    model_config = ConfigDict(extra="forbid")
    
    timeout: confloat(gt=0.0) = Field(default=30.0, description="Timeout in seconds")
    max_retries: conint(ge=0, le=10) = Field(default=3, description="Maximum retries")
    verbose: bool = Field(default=False, description="Verbose output")
    output_file: Optional[str] = Field(default=None, description="Output file path")
    aggressive_scanning: bool = Field(default=False, description="Aggressive scanning mode")
```

## Main Enumerator Module

### Main Module Integration

**`enumerators/__init__.py`:**
```python
# Import all enumerator modules
from .dns.dns_enumerator import DNSEnumerator, DNSEnumeratorConfig
from .smb.smb_enumerator import SMBEnumerator, SMBEnumeratorConfig
from .ssh.ssh_enumerator import SSHEnumerator, SSHEnumeratorConfig
from .common.result_models import EnumeratorResult, EnumeratorConfig

# Define main exports
__all__ = [
    # Enumerator modules
    "DNSEnumerator",
    "SMBEnumerator",
    "SSHEnumerator",
    
    # Configuration models
    "DNSEnumeratorConfig",
    "SMBEnumeratorConfig", 
    "SSHEnumeratorConfig",
    
    # Common utilities
    "EnumeratorResult",
    "EnumeratorConfig",
    
    # Main functions
    "run_dns_enumeration",
    "run_smb_enumeration",
    "run_ssh_enumeration",
    "run_comprehensive_scan"
]

class MainEnumeratorModule:
    """Main enumerator module with proper imports and exports."""
    
    def __init__(self):
        self.dns_enumerator = DNSEnumerator()
        self.smb_enumerator = SMBEnumerator()
        self.ssh_enumerator = SSHEnumerator()
    
    async def run_dns_enumeration(
        target_domain: str,
        config: EnumeratorConfig
    ) -> EnumeratorResult:
        """Run DNS enumeration with all patterns integrated."""
        try:
            dns_config = DNSEnumeratorConfig(
                target_domain=target_domain,
                timeout=config.timeout,
                max_queries=100
            )
            
            return await self.dns_enumerator.enumerate_dns_records(dns_config)
            
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
            smb_config = SMBEnumeratorConfig(
                target_host=target_host,
                timeout=config.timeout
            )
            
            return await self.smb_enumerator.enumerate_smb_shares(smb_config)
            
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
            ssh_config = SSHEnarkedConfig(
                target_host=target_host,
                timeout=config MonsConfig
            )
            
            return await self.ssh_enumerator.enumerate_ssh_services(ssh_config)
            
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
                results["dns"] = await self.run_dns_enumeration(target, config)
            elif scan_type == "smb":
                results["smb"] = await self.run_smb_enumeration(target, config)
            elif scan_type == "ssh":
                results["ssh"] = await self.run_ssh_enumeration(target, config)
        
        return results
```

## Import/Export Patterns

### Proper Import Structure

**Module imports:**
```python
# Import specific enumerators
from enumerators.dns.dns_enumerator import enumerate_dns_records, DNSEnumeratorConfig
from enumerators.smb.smb_enumerator import enumerate_smb_shares, SMBEnumeratorConfig
from enumerators.ssh.ssh_enumerator import enumerate_ssh_services, SSHEnumeratorConfig

# Import common utilities
from enumerators.common.result_models import EnumeratorResult, EnumeratorConfig
from enumerators.utils.async_helpers import async_retry, async_timeout

# Import main module
from enumerators import MainEnumeratorModule
```

### Named Exports Usage

**Using named exports:**
```python
# In each module
__all__ = [
    "enumerate_dns_records",
    "DNSEnumeratorConfig",
    "DNSRecordType"
]

# When importing
from enumerators.dns.dns_enumerator import enumerate_dns_records, DNSEnumeratorConfig
```

## Best Practices

### 1. Enumerator Organization

**✅ Good:**
```python
# Clear module structure
enumerators/
├── dns/
│   ├── dns_enumerator.py
│   └── dns_config.py
├── smb/
│   ├── smb_enumerator.py
│   └── smb_config.py
└── ssh/
    ├── ssh_enumerator.py
    └── ssh_config.py
```

**❌ Avoid:**
```python
# Flat structure
dns_enumerator.py
smb_enumerator.py
ssh_enumerator.py
```

### 2. Configuration Management

**✅ Good:**
```python
class DNSEnumeratorConfig(BaseModel):
    """Pydantic model for DNS enumerator configuration."""
    
    model_config = ConfigDict(extra="forbid")
    
    target_domain: constr(strip_whitespace=True) = Field(description="Target domain")
    record_types: List[constr(strip_whitespace=True)] = Field(
        default=["A", "AAAA", "MX", "NS", "TXT"],
        description="DNS record types to query"
    )
    timeout: confloat(gt=0.0) = Field(default=10.0, description="DNS query timeout")
```

### 3. Async Operations

**✅ Good:**
```python
async def enumerate_dns_records(
    config: DNSEnumeratorConfig
) -> EnumeratorResult:
    """Enumerate DNS records asynchronously."""
    try:
        # Async DNS operations
        results = await perform_dns_queries(config)
        return EnumeratorResult(
            is_successful=True,
            results=results,
            scan_time=calculate_scan_time()
        )
    except Exception as exc:
        return EnumeratorResult(
            is_successful=False,
            errors=[str(exc)]
        )
```

### 4. Error Handling

**✅ Good:**
```python
async def enumerate_smb_shares(
    config: SMBEnumeratorConfig
) -> EnumeratorResult:
    """Enumerate SMB shares with comprehensive error handling."""
    try:
        #视 inputs
        if not config.target_host:
            raise ValueError("target_host cannot be empty")
        
        # Perform enumeration
        results = await perform_smb_enumeration(config)
        
        return EnumeratorResult(
            is_successful=True,
            results=results
        )
    except ValueError as ve:
        return EnumeratorResult(
            is_successful=False,
            errors=[f"Validation error: {ve}"]
        )
    except Exception as exc:
        return EnumeratorResult(
            is_successful=False,
            errors=[f"Enumeration error: {exc}"]
        )
```

## Integration with Existing Patterns

### 1. Type Hints Integration

```python
# In each enumerator
async def enumerate_dns_records(
    config: DNSEnumeratorConfig
) -> EnumeratorResult:
    """Enumerate DNS records with comprehensive type hints."""
    pass

async def enumerate_smb_shares(
    config: SMBEnumeratorConfig
) -> EnumeratorResult:
    """Enumerate SMB shares with comprehensive type hints."""
    pass

async def enumerate_ssh_services(
    config: SSHEnumeratorConfig
) -> EnumeratorResult:
    """Enumerate SSH services with comprehensive type hints."""
    pass
```

### 2. Pydantic Validation Integration

```python
# In configs modules
class DNSEnumeratorConfig(BaseModel):
    """Pydantic model for DNS enumerator configuration."""
    
    model_config = ConfigDict(extra="forbid")
    
    target_domain: constr(strip_whitespace=True) = Field(description="Target domain")
    record_types: List[constr(strip_whitespace=True)] = Field(
        default=["A", "AAAA", "MX", "NS", "TXT"],
        description="DNS record types to query"
    )
    timeout: confloat(gt=0.0) = Field(default=10.0, description="DNS query timeout")
```

### 3. Async/Sync Pattern Integration

```python
# CPU-bound functions
def validate_dns_record(record: str) -> bool:
    """Validate DNS record format - CPU-bound."""
    # Validation logic
    pass

# I/O-bound functions
async def query_dns_server(
    domain: str,
    record_type: str,
    nameserver: str
) -> List[str]:
    """Query DNS server - I/O-bound."""
    # DNS query logic
    pass
```

### 4. RORO Pattern Integration

```python
def enumerate_with_roro(params: Dict[str, Any]) -> Dict[str, Any]:
    """Enumerate using RORO pattern."""
    try:
        target = params["target"]
        scan_type = params["scan_type"]
        config = params["config"]
        
        # Perform enumeration
        result = perform_enumeration(target, scan_type, config)
        
        return {
            "is_successful": True,
            "result": result,
            "error": None
        }
    except Exception as exc:
        return {
            "is_successful": False,
            "result": None,
            "error": str(exc)
        }
```

## Security Considerations

### 1. Authentication Handling

```python
class SecureEnumeratorConfig(BaseModel):
    """Secure enumerator configuration."""
    
    model_config = ConfigDict(extra="forbid")
    
    # Authentication settings
    username: Optional[str] = Field(default=None)
    password: Optional[str] = Field(default=None)
    key_file: Optional[str] = Field(default=None)
    
    # Security settings
    verify_ssl: bool = Field(default=True)
    allow_insecure: bool = Field(default=False)
    max_attempts: conint(ge=1, le=10) = Field(default=3)
```

### 2. Rate Limiting

```python
class RateLimitedEnumerator:
    """Rate-limited enumerator with backoff."""
    
    def __init__(self, max_requests_per_second: int = 10):
        self.max_requests = max_requests_per_second
        self.request_times = []
    
    async def rate_limited_request(self, request_func: Callable) -> Any:
        """Perform rate-limited request."""
        current_time = asyncio.get_event_loop().time()
        
        # Remove old requests
        self.request_times = [t for t in self.request_times if current_time - t < 1.0]
        
        # Check rate limit
        if len(self.request_times) >= self.max_requests:
            wait_time = 1.0 - (current_time - self.request_times[0])
            await asyncio.sleep(wait_time)
        
        # Perform request
        self.request_times.append(current_time)
        return await request_func()
```

### 3. Error Logging

```python
import logging

class SecureEnumerator:
    """Secure enumerator with logging."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def enumerate_with_logging(
        self,
        config: EnumeratorConfig
    ) -> EnumeratorResult:
        """Enumerate with comprehensive logging."""
        try:
            self.logger.info(f"Starting enumeration of {config.target}")
            
            # Perform enumeration
            result = await self.perform_enumeration(config)
            
            self.logger.info(f"Enumeration completed successfully: {len(result.results)} results")
            return result
            
        except Exception as exc:
            self.logger.error(f"Enumeration failed: {exc}")
            return EnumeratorResult(
                is_successful=False,
                target=config.target,
                scan_type=config.scan_type,
                results=[],
                errors=[str(exc)]
            )
```

## Summary

### Key Principles

1. **Organize by enumeration type:**
   - DNS enumerator for domain enumeration
   - SMB enumerator for file sharing enumeration
   - SSH enumerator for remote access enumeration

2. **Use proper imports/exports:**
   - Named exports for clear APIs
   - Organized import statements
   - Clear module dependencies

3. **Integrate all patterns:**
   - Type hints throughout all enumerators
   - Pydantic validation for configurations
   - Async/sync patterns for operations
   - RORO pattern for interfaces

4. **Follow security best practices:**
   - Proper authentication handling
   - Rate limiting and backoff
   - Comprehensive error logging
   - Secure configuration management

### Benefits

- **Organization**: Clear separation of enumeration types
- **Maintainability**: Isolated changes and easy testing
- **Reusability**: Shared utilities and consistent interfaces
- **Type Safety**: Type hints throughout all enumerators
- **Validation**: Pydantic models for configurations
- **Security**: Proper authentication and rate limiting
- **Scalability**: Modular architecture for new enumerators

The enumerators structure provides a robust foundation for building secure, maintainable, and type-safe enumeration systems that properly integrate all the patterns we've discussed throughout our conversation. 