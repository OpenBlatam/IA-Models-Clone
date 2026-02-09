"""
Utils Module for Video-OpusClip
Utility functions and helper modules
"""

from .crypto_helpers import (
    CryptoHelpers, CryptoConfig, HashAlgorithm, EncryptionAlgorithm
)
from .network_helpers import (
    NetworkHelpers, NetworkConfig, NetworkResult, Protocol, ConnectionStatus
)

__all__ = [
    # Crypto Helpers
    'CryptoHelpers',
    'CryptoConfig',
    'HashAlgorithm',
    'EncryptionAlgorithm',
    
    # Network Helpers
    'NetworkHelpers',
    'NetworkConfig',
    'NetworkResult',
    'Protocol',
    'ConnectionStatus'
]

# Utility functions for common operations
async def encrypt_sensitive_data(data: str, password: str) -> Dict[str, str]:
    """
    Encrypt sensitive data using password-based encryption
    
    Args:
        data: Data to encrypt
        password: Password for encryption
        
    Returns:
        Dictionary containing encrypted data and salt
    """
    crypto = CryptoHelpers()
    salt = crypto.generate_salt()
    
    # Derive key from password
    key_data = crypto.hash_password(password, salt)
    key = base64.b64decode(key_data["hash"])
    
    # Encrypt data
    encrypted = crypto.encrypt_fernet(data, key)
    
    return {
        "encrypted_data": encrypted,
        "salt": key_data["salt"],
        "iterations": key_data["iterations"]
    }

async def decrypt_sensitive_data(encrypted_data: str, password: str, salt: str, iterations: int) -> str:
    """
    Decrypt sensitive data using password-based decryption
    
    Args:
        encrypted_data: Encrypted data
        password: Password for decryption
        salt: Salt used during encryption
        iterations: Number of iterations used during encryption
        
    Returns:
        Decrypted data
    """
    crypto = CryptoHelpers()
    
    # Verify password and derive key
    if not crypto.verify_password(password, encrypted_data, salt, iterations):
        raise ValueError("Invalid password")
    
    # Derive key from password
    key_data = crypto.hash_password(password, base64.b64decode(salt))
    key = base64.b64decode(key_data["hash"])
    
    # Decrypt data
    return crypto.decrypt_fernet(encrypted_data, key)

async def check_network_connectivity(targets: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Check connectivity to multiple targets
    
    Args:
        targets: List of target dictionaries with host, port, and protocol
        
    Returns:
        Dictionary with connectivity results
    """
    async with NetworkHelpers() as network:
        results = {}
        
        for target in targets:
            host = target["host"]
            port = target["port"]
            protocol = Protocol(target.get("protocol", "tcp"))
            
            result = await network.check_connectivity(host, port, protocol)
            results[f"{host}:{port}"] = {
                "success": result.success,
                "status": result.status.value,
                "response_time": result.response_time,
                "error": result.error_message
            }
        
        return results

async def scan_network_services(host: str, port_ranges: List[Tuple[int, int]] = None) -> Dict[str, Any]:
    """
    Scan network services on a host
    
    Args:
        host: Target host
        port_ranges: List of port ranges to scan (start, end)
        
    Returns:
        Dictionary with scan results
    """
    if port_ranges is None:
        port_ranges = [(1, 1024), (8080, 8090), (9000, 9010)]
    
    async with NetworkHelpers() as network:
        all_results = {}
        
        for start_port, end_port in port_ranges:
            results = await network.scan_port_range(host, start_port, end_port)
            all_results[f"{start_port}-{end_port}"] = results
        
        return {
            "host": host,
            "scan_results": all_results,
            "total_ports_scanned": sum(len(results) for results in all_results.values()),
            "open_ports": [
                port for results in all_results.values()
                for port, result in results.items()
                if result.success
            ]
        }

async def get_network_info(host: str) -> Dict[str, Any]:
    """
    Get comprehensive network information for a host
    
    Args:
        host: Target host
        
    Returns:
        Dictionary with network information
    """
    async with NetworkHelpers() as network:
        info = {
            "host": host,
            "timestamp": datetime.now().isoformat()
        }
        
        # DNS information
        try:
            dns_info = await network.get_dns_info(host)
            info["dns"] = dns_info
        except Exception as e:
            info["dns_error"] = str(e)
        
        # WHOIS information
        try:
            whois_info = await network.get_whois_info(host)
            info["whois"] = whois_info
        except Exception as e:
            info["whois_error"] = str(e)
        
        # SSL certificate (if applicable)
        try:
            ssl_info = await network.check_ssl_certificate(host, 443)
            info["ssl"] = ssl_info
        except Exception as e:
            info["ssl_error"] = str(e)
        
        # Ping test
        try:
            ping_info = await network.ping_host(host, count=3)
            info["ping"] = ping_info
        except Exception as e:
            info["ping_error"] = str(e)
        
        return info

def generate_secure_credentials(length: int = 16) -> Dict[str, str]:
    """
    Generate secure credentials
    
    Args:
        length: Length of generated credentials
        
    Returns:
        Dictionary with username and password
    """
    crypto = CryptoHelpers()
    
    return {
        "username": f"user_{crypto.generate_secure_token(8)}",
        "password": crypto.generate_secure_password(length, include_symbols=True)
    }

def validate_network_address(address: str) -> Dict[str, Any]:
    """
    Validate network address (IP or domain)
    
    Args:
        address: Network address to validate
        
    Returns:
        Dictionary with validation results
    """
    import re
    
    result = {
        "address": address,
        "is_valid": False,
        "type": "unknown",
        "details": {}
    }
    
    # Check if it's an IP address
    try:
        ip = ipaddress.ip_address(address)
        result["is_valid"] = True
        result["type"] = "ip"
        result["details"] = {
            "version": ip.version,
            "is_private": ip.is_private,
            "is_loopback": ip.is_loopback,
            "is_multicast": ip.is_multicast,
            "is_reserved": ip.is_reserved
        }
    except ValueError:
        # Check if it's a valid domain name
        domain_pattern = r'^[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)*$'
        if re.match(domain_pattern, address):
            result["is_valid"] = True
            result["type"] = "domain"
            result["details"] = {
                "parts": address.split('.'),
                "tld": address.split('.')[-1] if '.' in address else None
            }
        else:
            result["details"]["error"] = "Invalid address format"
    
    return result

async def create_network_report(targets: List[str], include_ports: bool = True) -> Dict[str, Any]:
    """
    Create comprehensive network report for multiple targets
    
    Args:
        targets: List of target addresses
        include_ports: Whether to include port scanning
        
    Returns:
        Dictionary with network report
    """
    report = {
        "timestamp": datetime.now().isoformat(),
        "targets": {},
        "summary": {
            "total_targets": len(targets),
            "reachable_targets": 0,
            "unreachable_targets": 0,
            "open_ports_found": 0
        }
    }
    
    async with NetworkHelpers() as network:
        for target in targets:
            target_info = {
                "address": target,
                "validation": validate_network_address(target),
                "connectivity": {},
                "services": {},
                "network_info": {}
            }
            
            # Basic connectivity check
            if target_info["validation"]["type"] == "ip":
                # For IP addresses, check common ports
                common_ports = [22, 80, 443, 8080]
                for port in common_ports:
                    result = await network.check_connectivity(target, port)
                    target_info["connectivity"][f"port_{port}"] = {
                        "success": result.success,
                        "status": result.status.value,
                        "response_time": result.response_time
                    }
                    
                    if result.success:
                        report["summary"]["reachable_targets"] += 1
                        report["summary"]["open_ports_found"] += 1
            else:
                # For domains, get network info
                try:
                    network_info = await get_network_info(target)
                    target_info["network_info"] = network_info
                    
                    if "ping" in network_info and network_info["ping"]["success"]:
                        report["summary"]["reachable_targets"] += 1
                except Exception as e:
                    target_info["network_info"]["error"] = str(e)
            
            # Port scanning if requested
            if include_ports and target_info["validation"]["type"] == "ip":
                try:
                    scan_results = await scan_network_services(target)
                    target_info["services"] = scan_results
                except Exception as e:
                    target_info["services"]["error"] = str(e)
            
            report["targets"][target] = target_info
        
        # Calculate summary
        report["summary"]["unreachable_targets"] = (
            report["summary"]["total_targets"] - report["summary"]["reachable_targets"]
        )
    
    return report

def create_crypto_report(data_samples: List[str]) -> Dict[str, Any]:
    """
    Create cryptographic analysis report
    
    Args:
        data_samples: List of data samples to analyze
        
    Returns:
        Dictionary with crypto analysis
    """
    crypto = CryptoHelpers()
    report = {
        "timestamp": datetime.now().isoformat(),
        "samples": {},
        "algorithms": {
            "hash_algorithms": list(HashAlgorithm),
            "encryption_algorithms": list(EncryptionAlgorithm)
        },
        "summary": {
            "total_samples": len(data_samples),
            "hash_results": {},
            "password_analysis": {}
        }
    }
    
    for i, sample in enumerate(data_samples):
        sample_id = f"sample_{i+1}"
        sample_info = {
            "data": sample[:50] + "..." if len(sample) > 50 else sample,
            "length": len(sample),
            "hashes": {},
            "password_strength": None
        }
        
        # Generate hashes for all algorithms
        for algorithm in HashAlgorithm:
            try:
                hash_result = crypto.hash_data(sample, algorithm)
                sample_info["hashes"][algorithm.value] = hash_result
            except Exception as e:
                sample_info["hashes"][algorithm.value] = {"error": str(e)}
        
        # Analyze password strength if applicable
        if len(sample) >= 6:  # Assume it might be a password
            try:
                strength_result = crypto.validate_password_strength(sample)
                sample_info["password_strength"] = strength_result
            except Exception as e:
                sample_info["password_strength"] = {"error": str(e)}
        
        report["samples"][sample_id] = sample_info
    
    # Generate summary statistics
    for algorithm in HashAlgorithm:
        report["summary"]["hash_results"][algorithm.value] = {
            "successful": sum(
                1 for sample in report["samples"].values()
                if algorithm.value in sample["hashes"] and "error" not in sample["hashes"][algorithm.value]
            ),
            "failed": sum(
                1 for sample in report["samples"].values()
                if algorithm.value in sample["hashes"] and "error" in sample["hashes"][algorithm.value]
            )
        }
    
    # Password strength summary
    strong_passwords = sum(
        1 for sample in report["samples"].values()
        if sample["password_strength"] and sample["password_strength"].get("is_strong", False)
    )
    
    report["summary"]["password_analysis"] = {
        "strong_passwords": strong_passwords,
        "weak_passwords": len(data_samples) - strong_passwords,
        "strength_percentage": (strong_passwords / len(data_samples) * 100) if data_samples else 0
    }
    
    return report

# Example usage
async def main():
    """Example usage of utils module"""
    print("🔧 Utils Module Example")
    
    # Crypto operations
    print("\n🔐 Crypto Operations:")
    crypto = CryptoHelpers()
    
    # Encrypt sensitive data
    sensitive_data = "secret_information"
    encrypted = await encrypt_sensitive_data(sensitive_data, "my_password")
    print(f"Encrypted data: {encrypted['encrypted_data'][:50]}...")
    
    # Decrypt sensitive data
    decrypted = await decrypt_sensitive_data(
        encrypted['encrypted_data'], 
        "my_password", 
        encrypted['salt'], 
        int(encrypted['iterations'])
    )
    print(f"Decrypted data: {decrypted}")
    
    # Network operations
    print("\n🌐 Network Operations:")
    
    # Check connectivity
    targets = [
        {"host": "google.com", "port": 80, "protocol": "tcp"},
        {"host": "github.com", "port": 443, "protocol": "tcp"}
    ]
    connectivity_results = await check_network_connectivity(targets)
    print(f"Connectivity results: {connectivity_results}")
    
    # Generate secure credentials
    credentials = generate_secure_credentials()
    print(f"Generated credentials: {credentials}")
    
    # Validate network addresses
    addresses = ["192.168.1.1", "google.com", "invalid-address"]
    for address in addresses:
        validation = validate_network_address(address)
        print(f"Address {address}: {validation['type']} - {validation['is_valid']}")
    
    # Create network report
    network_report = await create_network_report(["google.com", "github.com"], include_ports=False)
    print(f"Network report summary: {network_report['summary']}")
    
    # Create crypto report
    data_samples = ["password123", "MySecureP@ssw0rd!", "weak", "StrongP@ss1!"]
    crypto_report = create_crypto_report(data_samples)
    print(f"Crypto report summary: {crypto_report['summary']}")

if __name__ == "__main__":
    import asyncio
    from datetime import datetime
    import base64
    import ipaddress
    
    asyncio.run(main()) 