#!/usr/bin/env python3
"""
SSH Enumerator Module for Video-OpusClip
SSH enumeration and reconnaissance tools
"""

import asyncio
import socket
import paramiko
import subprocess
import re
import hashlib
import base64
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import asyncio
import aiofiles
import os

class SSHKeyType(str, Enum):
    """SSH key types"""
    RSA = "ssh-rsa"
    DSA = "ssh-dss"
    ECDSA = "ecdsa-sha2-nistp256"
    ED25519 = "ssh-ed25519"

class SSHAlgorithm(str, Enum):
    """SSH algorithms"""
    KEX_ALGORITHMS = "kex_algorithms"
    ENCRYPT_ALGORITHMS = "encrypt_algorithms"
    MAC_ALGORITHMS = "mac_algorithms"
    COMPRESSION_ALGORITHMS = "compression_algorithms"
    HOST_KEY_ALGORITHMS = "host_key_algorithms"

@dataclass
class SSHHostKey:
    """SSH host key information"""
    key_type: SSHKeyType
    key_data: str
    fingerprint_md5: str
    fingerprint_sha256: str
    discovered_at: datetime = None
    
    def __post_init__(self):
        if self.discovered_at is None:
            self.discovered_at = datetime.utcnow()

@dataclass
class SSHAlgorithmInfo:
    """SSH algorithm information"""
    algorithm_type: SSHAlgorithm
    algorithms: List[str]
    preferred: Optional[str] = None
    discovered_at: datetime = None
    
    def __post_init__(self):
        if self.discovered_at is None:
            self.discovered_at = datetime.utcnow()

@dataclass
class SSHUser:
    """SSH user information"""
    username: str
    password: Optional[str] = None
    key_file: Optional[str] = None
    authentication_method: str = "unknown"
    last_login: Optional[datetime] = None
    discovered_at: datetime = None
    
    def __post_init__(self):
        if self.discovered_at is None:
            self.discovered_at = datetime.utcnow()

@dataclass
class EnumerationConfig:
    """Configuration for SSH enumeration"""
    target_host: str
    target_port: int = 22
    timeout: float = 30.0
    max_concurrent: int = 10
    enable_bruteforce: bool = False
    bruteforce_users: List[str] = None
    bruteforce_passwords: List[str] = None
    key_files: List[str] = None
    enable_banner_grab: bool = True
    enable_algorithm_enumeration: bool = True
    enable_host_key_fingerprinting: bool = True
    
    def __post_init__(self):
        if self.bruteforce_users is None:
            self.bruteforce_users = [
                "root", "admin", "administrator", "user", "guest", "test",
                "demo", "ubuntu", "centos", "debian", "fedora", "pi",
                "vagrant", "docker", "jenkins", "git", "www-data", "nginx",
                "apache", "mysql", "postgres", "redis", "elasticsearch"
            ]
        if self.bruteforce_passwords is None:
            self.bruteforce_passwords = [
                "", "password", "123456", "admin", "root", "user",
                "guest", "test", "demo", "ubuntu", "centos", "debian",
                "raspberry", "pi", "vagrant", "docker", "jenkins"
            ]
        if self.key_files is None:
            self.key_files = [
                "~/.ssh/id_rsa", "~/.ssh/id_dsa", "~/.ssh/id_ecdsa",
                "~/.ssh/id_ed25519", "~/.ssh/id_rsa.pub", "~/.ssh/id_dsa.pub"
            ]

class SSHEnumerator:
    """SSH enumeration and reconnaissance tool"""
    
    def __init__(self, config: EnumerationConfig):
        self.config = config
        self.results: Dict[str, Any] = {}
        self.host_keys: List[SSHHostKey] = []
        self.algorithms: List[SSHAlgorithmInfo] = []
        self.users: List[SSHUser] = []
        self.banner: Optional[str] = None
        self.enumeration_start_time: float = 0.0
        self.enumeration_end_time: float = 0.0
    
    async def enumerate_ssh(self) -> Dict[str, Any]:
        """Perform comprehensive SSH enumeration"""
        self.enumeration_start_time = asyncio.get_event_loop().time()
        
        try:
            # Test SSH connectivity
            if not await self._test_ssh_connectivity():
                return {
                    "success": False,
                    "error": "SSH service not accessible",
                    "target_host": self.config.target_host
                }
            
            # Grab SSH banner
            if self.config.enable_banner_grab:
                await self._grab_ssh_banner()
            
            # Enumerate supported algorithms
            if self.config.enable_algorithm_enumeration:
                await self._enumerate_algorithms()
            
            # Get host key fingerprints
            if self.config.enable_host_key_fingerprinting:
                await self._get_host_key_fingerprints()
            
            # Brute force authentication
            if self.config.enable_bruteforce:
                await self._brute_force_authentication()
            
            # Test key-based authentication
            await self._test_key_authentication()
            
            # Get system information
            system_info = await self._get_system_info()
            self.results["system_info"] = system_info
            
            self.enumeration_end_time = asyncio.get_event_loop().time()
            
            return {
                "success": True,
                "target_host": self.config.target_host,
                "target_port": self.config.target_port,
                "enumeration_duration": self.enumeration_end_time - self.enumeration_start_time,
                "total_host_keys": len(self.host_keys),
                "total_algorithms": len(self.algorithms),
                "total_users": len(self.users),
                "results": {
                    "banner": self.banner,
                    "host_keys": [self._host_key_to_dict(hk) for hk in self.host_keys],
                    "algorithms": [self._algorithm_to_dict(alg) for alg in self.algorithms],
                    "users": [self._user_to_dict(u) for u in self.users],
                    "system_info": system_info
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "target_host": self.config.target_host
            }
    
    async def _test_ssh_connectivity(self) -> bool:
        """Test if SSH service is accessible"""
        try:
            # Test TCP connection
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.config.timeout)
            result = sock.connect_ex((self.config.target_host, self.config.target_port))
            sock.close()
            
            if result == 0:
                return True
            else:
                return False
                
        except Exception:
            return False
    
    async def _grab_ssh_banner(self) -> None:
        """Grab SSH banner"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.config.timeout)
            sock.connect((self.config.target_host, self.config.target_port))
            
            # Receive banner
            banner = sock.recv(1024).decode('utf-8', errors='ignore').strip()
            self.banner = banner
            
            sock.close()
            
        except Exception as e:
            self.banner = f"Error grabbing banner: {str(e)}"
    
    async def _enumerate_algorithms(self) -> None:
        """Enumerate supported SSH algorithms"""
        try:
            # Use ssh-keyscan to get algorithm information
            result = subprocess.run(
                ["ssh-keyscan", "-T", str(self.config.timeout), self.config.target_host],
                capture_output=True,
                text=True,
                timeout=self.config.timeout
            )
            
            if result.returncode == 0:
                # Parse algorithm information from ssh-keyscan output
                # This is a simplified approach - real implementation would need more sophisticated parsing
                pass
            
            # Alternative: Use paramiko to get algorithm information
            await self._get_algorithms_with_paramiko()
            
        except Exception as e:
            self.results["algorithm_enumeration_error"] = str(e)
    
    async def _get_algorithms_with_paramiko(self) -> None:
        """Get algorithms using paramiko"""
        try:
            # Create transport
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.config.timeout)
            sock.connect((self.config.target_host, self.config.target_port))
            
            transport = paramiko.Transport(sock)
            transport.start_client()
            
            # Get algorithm information
            if transport.remote_version:
                # Extract algorithms from transport
                kex_algorithms = transport.get_security_options().kex
                encryption_algorithms = transport.get_security_options().ciphers
                mac_algorithms = transport.get_security_options().macs
                compression_algorithms = transport.get_security_options().compression
                host_key_algorithms = transport.get_security_options().key_types
                
                # Store algorithm information
                self.algorithms.extend([
                    SSHAlgorithmInfo(SSHAlgorithm.KEX_ALGORITHMS, kex_algorithms),
                    SSHAlgorithmInfo(SSHAlgorithm.ENCRYPT_ALGORITHMS, encryption_algorithms),
                    SSHAlgorithmInfo(SSHAlgorithm.MAC_ALGORITHMS, mac_algorithms),
                    SSHAlgorithmInfo(SSHAlgorithm.COMPRESSION_ALGORITHMS, compression_algorithms),
                    SSHAlgorithmInfo(SSHAlgorithm.HOST_KEY_ALGORITHMS, host_key_algorithms)
                ])
            
            transport.close()
            
        except Exception as e:
            # Paramiko algorithm enumeration failed
            pass
    
    async def _get_host_key_fingerprints(self) -> None:
        """Get SSH host key fingerprints"""
        try:
            # Use ssh-keyscan to get host keys
            result = subprocess.run(
                ["ssh-keyscan", "-T", str(self.config.timeout), self.config.target_host],
                capture_output=True,
                text=True,
                timeout=self.config.timeout
            )
            
            if result.returncode == 0:
                output = result.stdout
                
                for line in output.split('\n'):
                    line = line.strip()
                    if line and not line.startswith('#'):
                        parts = line.split()
                        if len(parts) >= 3:
                            hostname = parts[0]
                            key_type = parts[1]
                            key_data = parts[2]
                            
                            # Calculate fingerprints
                            key_bytes = base64.b64decode(key_data)
                            
                            # MD5 fingerprint
                            md5_hash = hashlib.md5(key_bytes).hexdigest()
                            md5_fingerprint = ':'.join([md5_hash[i:i+2] for i in range(0, len(md5_hash), 2)])
                            
                            # SHA256 fingerprint
                            sha256_hash = hashlib.sha256(key_bytes).hexdigest()
                            sha256_fingerprint = base64.b64encode(hashlib.sha256(key_bytes).digest()).decode()
                            
                            # Determine key type
                            if key_type == "ssh-rsa":
                                ssh_key_type = SSHKeyType.RSA
                            elif key_type == "ssh-dss":
                                ssh_key_type = SSHKeyType.DSA
                            elif key_type.startswith("ecdsa"):
                                ssh_key_type = SSHKeyType.ECDSA
                            elif key_type == "ssh-ed25519":
                                ssh_key_type = SSHKeyType.ED25519
                            else:
                                ssh_key_type = SSHKeyType.RSA  # Default
                            
                            host_key = SSHHostKey(
                                key_type=ssh_key_type,
                                key_data=key_data,
                                fingerprint_md5=md5_fingerprint,
                                fingerprint_sha256=sha256_fingerprint
                            )
                            self.host_keys.append(host_key)
            
        except Exception as e:
            self.results["host_key_error"] = str(e)
    
    async def _brute_force_authentication(self) -> None:
        """Brute force SSH authentication"""
        semaphore = asyncio.Semaphore(self.config.max_concurrent)
        
        tasks = []
        for username in self.config.bruteforce_users:
            for password in self.config.bruteforce_passwords:
                tasks.append(self._test_credentials(username, password, semaphore))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, dict) and result.get("success"):
                user = SSHUser(
                    username=result["username"],
                    password=result["password"],
                    authentication_method="password"
                )
                self.users.append(user)
    
    async def _test_credentials(self, username: str, password: str, semaphore: asyncio.Semaphore) -> Dict[str, Any]:
        """Test specific username/password combination"""
        async with semaphore:
            try:
                # Create SSH client
                client = paramiko.SSHClient()
                client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                
                # Try to connect
                client.connect(
                    self.config.target_host,
                    port=self.config.target_port,
                    username=username,
                    password=password,
                    timeout=self.config.timeout,
                    banner_timeout=self.config.timeout,
                    auth_timeout=self.config.timeout
                )
                
                # If we get here, authentication was successful
                client.close()
                
                return {
                    "success": True,
                    "username": username,
                    "password": password,
                    "description": "Successful password authentication"
                }
                
            except paramiko.AuthenticationException:
                return {
                    "success": False,
                    "username": username,
                    "password": password,
                    "description": "Authentication failed"
                }
            except Exception as e:
                return {
                    "success": False,
                    "username": username,
                    "password": password,
                    "error": str(e)
                }
    
    async def _test_key_authentication(self) -> None:
        """Test key-based authentication"""
        for key_file in self.config.key_files:
            try:
                # Expand tilde in path
                expanded_key_file = os.path.expanduser(key_file)
                
                if os.path.exists(expanded_key_file):
                    # Test key authentication
                    for username in self.config.bruteforce_users[:5]:  # Test with first 5 users
                        try:
                            client = paramiko.SSHClient()
                            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                            
                            # Load private key
                            key = paramiko.RSAKey.from_private_key_file(expanded_key_file)
                            
                            client.connect(
                                self.config.target_host,
                                port=self.config.target_port,
                                username=username,
                                pkey=key,
                                timeout=self.config.timeout
                            )
                            
                            # If we get here, key authentication was successful
                            client.close()
                            
                            user = SSHUser(
                                username=username,
                                key_file=expanded_key_file,
                                authentication_method="public_key"
                            )
                            self.users.append(user)
                            
                            break  # Found working key for this user
                            
                        except Exception:
                            # Key authentication failed for this user
                            continue
                            
            except Exception:
                # Key file error
                continue
    
    async def _get_system_info(self) -> Dict[str, Any]:
        """Get system information via SSH"""
        system_info = {}
        
        # Try to get system info using successful authentication
        for user in self.users:
            if user.password or user.key_file:
                try:
                    client = paramiko.SSHClient()
                    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                    
                    if user.password:
                        client.connect(
                            self.config.target_host,
                            port=self.config.target_port,
                            username=user.username,
                            password=user.password,
                            timeout=self.config.timeout
                        )
                    else:
                        key = paramiko.RSAKey.from_private_key_file(user.key_file)
                        client.connect(
                            self.config.target_host,
                            port=self.config.target_port,
                            username=user.username,
                            pkey=key,
                            timeout=self.config.timeout
                        )
                    
                    # Execute commands to get system info
                    commands = [
                        "uname -a",
                        "cat /etc/os-release",
                        "hostname",
                        "whoami",
                        "id",
                        "ps aux | head -10"
                    ]
                    
                    for command in commands:
                        try:
                            stdin, stdout, stderr = client.exec_command(command, timeout=10)
                            output = stdout.read().decode('utf-8', errors='ignore').strip()
                            
                            if output:
                                system_info[command] = output
                                
                        except Exception:
                            continue
                    
                    client.close()
                    break  # Got system info, no need to try other users
                    
                except Exception:
                    continue
        
        return system_info
    
    def _host_key_to_dict(self, host_key: SSHHostKey) -> Dict[str, Any]:
        """Convert SSHHostKey to dictionary"""
        return {
            "key_type": host_key.key_type.value,
            "key_data": host_key.key_data,
            "fingerprint_md5": host_key.fingerprint_md5,
            "fingerprint_sha256": host_key.fingerprint_sha256,
            "discovered_at": host_key.discovered_at.isoformat() if host_key.discovered_at else None
        }
    
    def _algorithm_to_dict(self, algorithm: SSHAlgorithmInfo) -> Dict[str, Any]:
        """Convert SSHAlgorithmInfo to dictionary"""
        return {
            "algorithm_type": algorithm.algorithm_type.value,
            "algorithms": algorithm.algorithms,
            "preferred": algorithm.preferred,
            "discovered_at": algorithm.discovered_at.isoformat() if algorithm.discovered_at else None
        }
    
    def _user_to_dict(self, user: SSHUser) -> Dict[str, Any]:
        """Convert SSHUser to dictionary"""
        return {
            "username": user.username,
            "password": user.password,
            "key_file": user.key_file,
            "authentication_method": user.authentication_method,
            "last_login": user.last_login.isoformat() if user.last_login else None,
            "discovered_at": user.discovered_at.isoformat() if user.discovered_at else None
        }
    
    def get_users_by_method(self, method: str) -> List[SSHUser]:
        """Get users by authentication method"""
        return [u for u in self.users if u.authentication_method == method]
    
    def get_host_keys_by_type(self, key_type: SSHKeyType) -> List[SSHHostKey]:
        """Get host keys by type"""
        return [hk for hk in self.host_keys if hk.key_type == key_type]
    
    def get_algorithms_by_type(self, algorithm_type: SSHAlgorithm) -> List[SSHAlgorithmInfo]:
        """Get algorithms by type"""
        return [alg for alg in self.algorithms if alg.algorithm_type == algorithm_type]
    
    def generate_report(self) -> str:
        """Generate SSH enumeration report"""
        report = f"SSH Enumeration Report for {self.config.target_host}:{self.config.target_port}\n"
        report += "=" * 70 + "\n"
        report += f"Enumeration Duration: {self.enumeration_end_time - self.enumeration_start_time:.2f} seconds\n"
        report += f"Total Host Keys: {len(self.host_keys)}\n"
        report += f"Total Algorithms: {len(self.algorithms)}\n"
        report += f"Total Users: {len(self.users)}\n\n"
        
        # SSH Banner
        if self.banner:
            report += "SSH Banner:\n"
            report += "-" * 15 + "\n"
            report += f"{self.banner}\n\n"
        
        # Host Keys
        if self.host_keys:
            report += "Host Keys:\n"
            report += "-" * 12 + "\n"
            for host_key in self.host_keys:
                report += f"• {host_key.key_type.value}\n"
                report += f"  MD5: {host_key.fingerprint_md5}\n"
                report += f"  SHA256: {host_key.fingerprint_sha256}\n\n"
        
        # Algorithms
        if self.algorithms:
            report += "Supported Algorithms:\n"
            report += "-" * 25 + "\n"
            for algorithm in self.algorithms:
                report += f"• {algorithm.algorithm_type.value}:\n"
                for alg in algorithm.algorithms[:5]:  # Show first 5
                    report += f"  - {alg}\n"
                if len(algorithm.algorithms) > 5:
                    report += f"  ... and {len(algorithm.algorithms) - 5} more\n"
                report += "\n"
        
        # Users
        if self.users:
            report += "Authenticated Users:\n"
            report += "-" * 22 + "\n"
            for user in self.users:
                report += f"• {user.username} ({user.authentication_method})"
                if user.password:
                    report += f" - Password: {user.password}"
                if user.key_file:
                    report += f" - Key: {user.key_file}"
                report += "\n"
            report += "\n"
        
        # System Information
        if "system_info" in self.results and self.results["system_info"]:
            sys_info = self.results["system_info"]
            report += "System Information:\n"
            report += "-" * 20 + "\n"
            for command, output in sys_info.items():
                report += f"• {command}:\n"
                report += f"  {output[:200]}..." if len(output) > 200 else f"  {output}\n"
                report += "\n"
        
        return report

# Example usage
async def main():
    """Example usage of SSH enumerator"""
    print("🔍 SSH Enumerator Example")
    
    # Create enumeration configuration
    config = EnumerationConfig(
        target_host="192.168.1.100",
        target_port=22,
        timeout=30.0,
        max_concurrent=5,
        enable_bruteforce=False,  # Set to True for brute force testing
        enable_banner_grab=True,
        enable_algorithm_enumeration=True,
        enable_host_key_fingerprinting=True
    )
    
    # Create enumerator
    enumerator = SSHEnumerator(config)
    
    # Perform enumeration
    print(f"Enumerating SSH on {config.target_host}:{config.target_port}...")
    result = await enumerator.enumerate_ssh()
    
    if result["success"]:
        print(f"✅ Enumeration completed in {result['enumeration_duration']:.2f} seconds")
        print(f"🔑 Found {result['total_host_keys']} host keys")
        print(f"⚙️ Found {result['total_algorithms']} algorithm types")
        print(f"👥 Found {result['total_users']} authenticated users")
        
        # Print some results
        if result['results']['banner']:
            print(f"\n📋 SSH Banner: {result['results']['banner']}")
        
        if result['results']['host_keys']:
            print("\n🔑 Host Keys:")
            for host_key in result['results']['host_keys'][:3]:  # Show first 3
                print(f"  {host_key['key_type']} - MD5: {host_key['fingerprint_md5'][:20]}...")
        
        if result['results']['users']:
            print("\n👥 Users:")
            for user in result['results']['users']:
                print(f"  {user['username']} ({user['authentication_method']})")
        
        # Generate report
        print("\n📋 SSH Enumeration Report:")
        print(enumerator.generate_report())
        
    else:
        print(f"❌ Enumeration failed: {result['error']}")

if __name__ == "__main__":
    asyncio.run(main()) 