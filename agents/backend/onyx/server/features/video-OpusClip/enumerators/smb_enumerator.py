#!/usr/bin/env python3
"""
SMB Enumerator Module for Video-OpusClip
SMB/CIFS enumeration and reconnaissance tools
"""

import asyncio
import socket
import struct
import subprocess
import re
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import smbclient
import smbprotocol
from smbprotocol.connection import Connection
from smbprotocol.session import Session
from smbprotocol.tree import TreeConnect
from smbprotocol.file_info import FileInfo
from smbprotocol.share_info import ShareInfo

class SMBCommand(str, Enum):
    """SMB commands"""
    NEGOTIATE = "negotiate"
    SESSION_SETUP = "session_setup"
    TREE_CONNECT = "tree_connect"
    SHARE_ENUM = "share_enum"
    FILE_ENUM = "file_enum"
    USER_ENUM = "user_enum"
    GROUP_ENUM = "group_enum"

class ShareType(str, Enum):
    """SMB share types"""
    DISK = "disk"
    PIPE = "pipe"
    PRINTER = "printer"
    IPC = "ipc"
    SPECIAL = "special"

@dataclass
class SMBShare:
    """SMB share information"""
    name: str
    share_type: ShareType
    comment: Optional[str] = None
    permissions: Optional[str] = None
    accessible: bool = False
    discovered_at: datetime = None
    
    def __post_init__(self):
        if self.discovered_at is None:
            self.discovered_at = datetime.utcnow()

@dataclass
class SMBFile:
    """SMB file information"""
    name: str
    path: str
    size: Optional[int] = None
    is_directory: bool = False
    permissions: Optional[str] = None
    last_modified: Optional[datetime] = None
    discovered_at: datetime = None
    
    def __post_init__(self):
        if self.discovered_at is None:
            self.discovered_at = datetime.utcnow()

@dataclass
class SMBUser:
    """SMB user information"""
    username: str
    full_name: Optional[str] = None
    description: Optional[str] = None
    last_logon: Optional[datetime] = None
    account_disabled: bool = False
    password_expires: Optional[datetime] = None
    discovered_at: datetime = None
    
    def __post_init__(self):
        if self.discovered_at is None:
            self.discovered_at = datetime.utcnow()

@dataclass
class EnumerationConfig:
    """Configuration for SMB enumeration"""
    target_host: str
    target_port: int = 445
    username: Optional[str] = None
    password: Optional[str] = None
    domain: Optional[str] = None
    timeout: float = 30.0
    max_concurrent: int = 10
    enable_anonymous: bool = True
    enable_guest: bool = True
    enable_bruteforce: bool = False
    bruteforce_users: List[str] = None
    bruteforce_passwords: List[str] = None
    share_wordlist: List[str] = None
    
    def __post_init__(self):
        if self.bruteforce_users is None:
            self.bruteforce_users = [
                "admin", "administrator", "root", "guest", "test",
                "user", "demo", "backup", "service", "system"
            ]
        if self.bruteforce_passwords is None:
            self.bruteforce_passwords = [
                "", "password", "123456", "admin", "administrator",
                "root", "guest", "test", "user", "demo"
            ]
        if self.share_wordlist is None:
            self.share_wordlist = [
                "C$", "D$", "E$", "ADMIN$", "IPC$", "PRINT$",
                "FAX$", "NETLOGON", "SYSVOL", "SHARED", "PUBLIC",
                "USERS", "DATA", "BACKUP", "TEMP", "TMP", "LOG",
                "LOGS", "CONFIG", "CONF", "SETTINGS", "DOCS",
                "DOCUMENTS", "FILES", "UPLOAD", "DOWNLOAD", "MEDIA"
            ]

class SMBEnumerator:
    """SMB enumeration and reconnaissance tool"""
    
    def __init__(self, config: EnumerationConfig):
        self.config = config
        self.results: Dict[str, Any] = {}
        self.shares: List[SMBShare] = []
        self.files: List[SMBFile] = []
        self.users: List[SMBUser] = []
        self.groups: List[Dict[str, Any]] = []
        self.enumeration_start_time: float = 0.0
        self.enumeration_end_time: float = 0.0
        self.connection: Optional[Connection] = None
        self.session: Optional[Session] = None
    
    async def enumerate_smb(self) -> Dict[str, Any]:
        """Perform comprehensive SMB enumeration"""
        self.enumeration_start_time = asyncio.get_event_loop().time()
        
        try:
            # Test SMB connectivity
            if not await self._test_smb_connectivity():
                return {
                    "success": False,
                    "error": "SMB service not accessible",
                    "target_host": self.config.target_host
                }
            
            # Get system information
            system_info = await self._get_system_info()
            self.results["system_info"] = system_info
            
            # Enumerate shares
            await self._enumerate_shares()
            
            # Enumerate users (if possible)
            await self._enumerate_users()
            
            # Enumerate groups (if possible)
            await self._enumerate_groups()
            
            # Test share access
            await self._test_share_access()
            
            # Enumerate files in accessible shares
            await self._enumerate_files()
            
            # Test authentication methods
            auth_results = await self._test_authentication()
            self.results["authentication"] = auth_results
            
            self.enumeration_end_time = asyncio.get_event_loop().time()
            
            return {
                "success": True,
                "target_host": self.config.target_host,
                "enumeration_duration": self.enumeration_end_time - self.enumeration_start_time,
                "total_shares": len(self.shares),
                "accessible_shares": len([s for s in self.shares if s.accessible]),
                "total_users": len(self.users),
                "total_files": len(self.files),
                "results": {
                    "system_info": system_info,
                    "shares": [self._share_to_dict(s) for s in self.shares],
                    "users": [self._user_to_dict(u) for u in self.users],
                    "groups": self.groups,
                    "files": [self._file_to_dict(f) for f in self.files],
                    "authentication": auth_results
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "target_host": self.config.target_host
            }
        finally:
            await self._cleanup_connection()
    
    async def _test_smb_connectivity(self) -> bool:
        """Test if SMB service is accessible"""
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
    
    async def _get_system_info(self) -> Dict[str, Any]:
        """Get SMB system information"""
        try:
            # Use smbclient to get system info
            system_info = {}
            
            # Try to get computer name
            try:
                result = subprocess.run(
                    ["smbclient", "-L", self.config.target_host, "-U", ""],
                    capture_output=True,
                    text=True,
                    timeout=self.config.timeout
                )
                
                if result.returncode == 0:
                    output = result.stdout
                    
                    # Extract computer name
                    computer_match = re.search(r"Server\s+\[([^\]]+)\]", output)
                    if computer_match:
                        system_info["computer_name"] = computer_match.group(1)
                    
                    # Extract workgroup/domain
                    workgroup_match = re.search(r"Workgroup\s+\[([^\]]+)\]", output)
                    if workgroup_match:
                        system_info["workgroup"] = workgroup_match.group(1)
                    
                    # Extract OS version
                    os_match = re.search(r"OS\s+\[([^\]]+)\]", output)
                    if os_match:
                        system_info["os_version"] = os_match.group(1)
                
            except Exception:
                pass
            
            return system_info
            
        except Exception as e:
            return {"error": str(e)}
    
    async def _enumerate_shares(self) -> None:
        """Enumerate SMB shares"""
        try:
            # Method 1: Anonymous enumeration
            if self.config.enable_anonymous:
                await self._enumerate_shares_anonymous()
            
            # Method 2: Authenticated enumeration
            if self.config.username and self.config.password:
                await self._enumerate_shares_authenticated()
            
            # Method 3: Brute force share names
            if self.config.enable_bruteforce:
                await self._brute_force_shares()
                
        except Exception as e:
            self.results["share_enumeration_error"] = str(e)
    
    async def _enumerate_shares_anonymous(self) -> None:
        """Enumerate shares using anonymous access"""
        try:
            result = subprocess.run(
                ["smbclient", "-L", self.config.target_host, "-U", ""],
                capture_output=True,
                text=True,
                timeout=self.config.timeout
            )
            
            if result.returncode == 0:
                output = result.stdout
                
                # Parse share information
                share_section = False
                for line in output.split('\n'):
                    line = line.strip()
                    
                    if "Sharename" in line and "Type" in line:
                        share_section = True
                        continue
                    
                    if share_section and line:
                        if "Disk" in line or "IPC" in line or "Printer" in line:
                            parts = line.split()
                            if len(parts) >= 2:
                                share_name = parts[0]
                                share_type_str = parts[1]
                                
                                # Determine share type
                                if "Disk" in share_type_str:
                                    share_type = ShareType.DISK
                                elif "IPC" in share_type_str:
                                    share_type = ShareType.IPC
                                elif "Printer" in share_type_str:
                                    share_type = ShareType.PRINTER
                                else:
                                    share_type = ShareType.SPECIAL
                                
                                # Extract comment if available
                                comment = None
                                if len(parts) > 2:
                                    comment = ' '.join(parts[2:])
                                
                                share = SMBShare(
                                    name=share_name,
                                    share_type=share_type,
                                    comment=comment
                                )
                                self.shares.append(share)
                
        except Exception as e:
            # Anonymous enumeration failed
            pass
    
    async def _enumerate_shares_authenticated(self) -> None:
        """Enumerate shares using authenticated access"""
        try:
            auth_string = f"{self.config.username}"
            if self.config.password:
                auth_string += f"%{self.config.password}"
            
            result = subprocess.run(
                ["smbclient", "-L", self.config.target_host, "-U", auth_string],
                capture_output=True,
                text=True,
                timeout=self.config.timeout
            )
            
            if result.returncode == 0:
                # Parse authenticated share enumeration
                # Similar to anonymous but with more details
                pass
                
        except Exception as e:
            # Authenticated enumeration failed
            pass
    
    async def _brute_force_shares(self) -> None:
        """Brute force share names"""
        semaphore = asyncio.Semaphore(self.config.max_concurrent)
        
        tasks = [
            self._test_share_access_brute(share_name, semaphore)
            for share_name in self.config.share_wordlist
        ]
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _test_share_access_brute(self, share_name: str, semaphore: asyncio.Semaphore) -> None:
        """Test access to a specific share name"""
        async with semaphore:
            try:
                # Test anonymous access
                result = subprocess.run(
                    ["smbclient", f"//{self.config.target_host}/{share_name}", "-U", ""],
                    capture_output=True,
                    text=True,
                    timeout=10.0
                )
                
                if result.returncode == 0:
                    share = SMBShare(
                        name=share_name,
                        share_type=ShareType.DISK,
                        accessible=True
                    )
                    self.shares.append(share)
                    
            except Exception:
                # Share access test failed
                pass
    
    async def _enumerate_users(self) -> None:
        """Enumerate SMB users"""
        try:
            # Method 1: Using rpcclient (if available)
            await self._enumerate_users_rpcclient()
            
            # Method 2: Using smbclient
            await self._enumerate_users_smbclient()
            
        except Exception as e:
            self.results["user_enumeration_error"] = str(e)
    
    async def _enumerate_users_rpcclient(self) -> None:
        """Enumerate users using rpcclient"""
        try:
            result = subprocess.run(
                ["rpcclient", "-U", "", self.config.target_host, "-c", "enumdomusers"],
                capture_output=True,
                text=True,
                timeout=self.config.timeout
            )
            
            if result.returncode == 0:
                output = result.stdout
                
                for line in output.split('\n'):
                    line = line.strip()
                    if line and 'user:' in line.lower():
                        # Parse user information
                        user_match = re.search(r'user:\[([^\]]+)\]', line)
                        if user_match:
                            username = user_match.group(1)
                            
                            user = SMBUser(username=username)
                            self.users.append(user)
                            
        except Exception:
            # rpcclient enumeration failed
            pass
    
    async def _enumerate_users_smbclient(self) -> None:
        """Enumerate users using smbclient"""
        try:
            # Try to access IPC$ share for user enumeration
            result = subprocess.run(
                ["smbclient", f"//{self.config.target_host}/IPC$", "-U", "", "-c", "help"],
                capture_output=True,
                text=True,
                timeout=self.config.timeout
            )
            
            # This is a simplified approach - real implementation would need more sophisticated parsing
            pass
            
        except Exception:
            # smbclient user enumeration failed
            pass
    
    async def _enumerate_groups(self) -> None:
        """Enumerate SMB groups"""
        try:
            result = subprocess.run(
                ["rpcclient", "-U", "", self.config.target_host, "-c", "enumdomgroups"],
                capture_output=True,
                text=True,
                timeout=self.config.timeout
            )
            
            if result.returncode == 0:
                output = result.stdout
                
                for line in output.split('\n'):
                    line = line.strip()
                    if line and 'group:' in line.lower():
                        # Parse group information
                        group_match = re.search(r'group:\[([^\]]+)\]', line)
                        if group_match:
                            group_name = group_match.group(1)
                            
                            group_info = {
                                "name": group_name,
                                "description": None
                            }
                            self.groups.append(group_info)
                            
        except Exception:
            # Group enumeration failed
            pass
    
    async def _test_share_access(self) -> None:
        """Test access to discovered shares"""
        for share in self.shares:
            try:
                # Test anonymous access
                result = subprocess.run(
                    ["smbclient", f"//{self.config.target_host}/{share.name}", "-U", ""],
                    capture_output=True,
                    text=True,
                    timeout=10.0
                )
                
                share.accessible = (result.returncode == 0)
                
            except Exception:
                share.accessible = False
    
    async def _enumerate_files(self) -> None:
        """Enumerate files in accessible shares"""
        accessible_shares = [s for s in self.shares if s.accessible and s.share_type == ShareType.DISK]
        
        semaphore = asyncio.Semaphore(self.config.max_concurrent)
        
        tasks = [
            self._enumerate_share_files(share, semaphore)
            for share in accessible_shares
        ]
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _enumerate_share_files(self, share: SMBShare, semaphore: asyncio.Semaphore) -> None:
        """Enumerate files in a specific share"""
        async with semaphore:
            try:
                result = subprocess.run(
                    ["smbclient", f"//{self.config.target_host}/{share.name}", "-U", "", "-c", "ls"],
                    capture_output=True,
                    text=True,
                    timeout=self.config.timeout
                )
                
                if result.returncode == 0:
                    output = result.stdout
                    
                    for line in output.split('\n'):
                        line = line.strip()
                        if line and not line.startswith('.'):
                            # Parse file information
                            parts = line.split()
                            if len(parts) >= 4:
                                file_name = parts[0]
                                file_size = parts[1]
                                file_date = parts[2]
                                file_time = parts[3]
                                
                                # Determine if it's a directory
                                is_directory = file_size == "D"
                                
                                file = SMBFile(
                                    name=file_name,
                                    path=f"{share.name}/{file_name}",
                                    size=int(file_size) if not is_directory else None,
                                    is_directory=is_directory
                                )
                                self.files.append(file)
                                
            except Exception:
                # File enumeration failed for this share
                pass
    
    async def _test_authentication(self) -> Dict[str, Any]:
        """Test various authentication methods"""
        auth_results = {
            "anonymous": False,
            "guest": False,
            "null_session": False,
            "bruteforce_results": []
        }
        
        try:
            # Test anonymous access
            result = subprocess.run(
                ["smbclient", "-L", self.config.target_host, "-U", ""],
                capture_output=True,
                text=True,
                timeout=self.config.timeout
            )
            auth_results["anonymous"] = (result.returncode == 0)
            
            # Test guest access
            result = subprocess.run(
                ["smbclient", "-L", self.config.target_host, "-U", "guest"],
                capture_output=True,
                text=True,
                timeout=self.config.timeout
            )
            auth_results["guest"] = (result.returncode == 0)
            
            # Test null session
            result = subprocess.run(
                ["smbclient", "-L", self.config.target_host, "-U", "null"],
                capture_output=True,
                text=True,
                timeout=self.config.timeout
            )
            auth_results["null_session"] = (result.returncode == 0)
            
            # Brute force authentication
            if self.config.enable_bruteforce:
                auth_results["bruteforce_results"] = await self._brute_force_authentication()
                
        except Exception as e:
            auth_results["error"] = str(e)
        
        return auth_results
    
    async def _brute_force_authentication(self) -> List[Dict[str, Any]]:
        """Brute force authentication"""
        successful_logins = []
        
        semaphore = asyncio.Semaphore(self.config.max_concurrent)
        
        tasks = []
        for username in self.config.bruteforce_users:
            for password in self.config.bruteforce_passwords:
                tasks.append(self._test_credentials(username, password, semaphore))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, dict) and result.get("success"):
                successful_logins.append(result)
        
        return successful_logins
    
    async def _test_credentials(self, username: str, password: str, semaphore: asyncio.Semaphore) -> Dict[str, Any]:
        """Test specific username/password combination"""
        async with semaphore:
            try:
                auth_string = f"{username}"
                if password:
                    auth_string += f"%{password}"
                
                result = subprocess.run(
                    ["smbclient", "-L", self.config.target_host, "-U", auth_string],
                    capture_output=True,
                    text=True,
                    timeout=10.0
                )
                
                if result.returncode == 0:
                    return {
                        "success": True,
                        "username": username,
                        "password": password,
                        "description": "Successful authentication"
                    }
                else:
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
    
    async def _cleanup_connection(self) -> None:
        """Clean up SMB connections"""
        try:
            if self.session:
                self.session.disconnect()
            if self.connection:
                self.connection.disconnect()
        except Exception:
            pass
    
    def _share_to_dict(self, share: SMBShare) -> Dict[str, Any]:
        """Convert SMBShare to dictionary"""
        return {
            "name": share.name,
            "share_type": share.share_type.value,
            "comment": share.comment,
            "permissions": share.permissions,
            "accessible": share.accessible,
            "discovered_at": share.discovered_at.isoformat() if share.discovered_at else None
        }
    
    def _user_to_dict(self, user: SMBUser) -> Dict[str, Any]:
        """Convert SMBUser to dictionary"""
        return {
            "username": user.username,
            "full_name": user.full_name,
            "description": user.description,
            "last_logon": user.last_logon.isoformat() if user.last_logon else None,
            "account_disabled": user.account_disabled,
            "password_expires": user.password_expires.isoformat() if user.password_expires else None,
            "discovered_at": user.discovered_at.isoformat() if user.discovered_at else None
        }
    
    def _file_to_dict(self, file: SMBFile) -> Dict[str, Any]:
        """Convert SMBFile to dictionary"""
        return {
            "name": file.name,
            "path": file.path,
            "size": file.size,
            "is_directory": file.is_directory,
            "permissions": file.permissions,
            "last_modified": file.last_modified.isoformat() if file.last_modified else None,
            "discovered_at": file.discovered_at.isoformat() if file.discovered_at else None
        }
    
    def get_accessible_shares(self) -> List[SMBShare]:
        """Get accessible shares"""
        return [s for s in self.shares if s.accessible]
    
    def get_disk_shares(self) -> List[SMBShare]:
        """Get disk shares"""
        return [s for s in self.shares if s.share_type == ShareType.DISK]
    
    def get_files_by_share(self, share_name: str) -> List[SMBFile]:
        """Get files from a specific share"""
        return [f for f in self.files if f.path.startswith(share_name)]
    
    def generate_report(self) -> str:
        """Generate SMB enumeration report"""
        report = f"SMB Enumeration Report for {self.config.target_host}\n"
        report += "=" * 60 + "\n"
        report += f"Enumeration Duration: {self.enumeration_end_time - self.enumeration_start_time:.2f} seconds\n"
        report += f"Total Shares: {len(self.shares)}\n"
        report += f"Accessible Shares: {len(self.get_accessible_shares())}\n"
        report += f"Total Users: {len(self.users)}\n"
        report += f"Total Files: {len(self.files)}\n\n"
        
        # System Information
        if "system_info" in self.results:
            sys_info = self.results["system_info"]
            report += "System Information:\n"
            report += "-" * 20 + "\n"
            for key, value in sys_info.items():
                report += f"{key}: {value}\n"
            report += "\n"
        
        # Shares
        if self.shares:
            report += "SMB Shares:\n"
            report += "-" * 15 + "\n"
            for share in self.shares:
                report += f"• {share.name} ({share.share_type.value})"
                if share.comment:
                    report += f" - {share.comment}"
                if share.accessible:
                    report += " [ACCESSIBLE]"
                report += "\n"
            report += "\n"
        
        # Users
        if self.users:
            report += "Users:\n"
            report += "-" * 8 + "\n"
            for user in self.users:
                report += f"• {user.username}"
                if user.full_name:
                    report += f" ({user.full_name})"
                if user.description:
                    report += f" - {user.description}"
                report += "\n"
            report += "\n"
        
        # Authentication Results
        if "authentication" in self.results:
            auth = self.results["authentication"]
            report += "Authentication Results:\n"
            report += "-" * 25 + "\n"
            report += f"Anonymous Access: {'Yes' if auth.get('anonymous') else 'No'}\n"
            report += f"Guest Access: {'Yes' if auth.get('guest') else 'No'}\n"
            report += f"Null Session: {'Yes' if auth.get('null_session') else 'No'}\n"
            
            if auth.get("bruteforce_results"):
                report += "Successful Brute Force Logins:\n"
                for login in auth["bruteforce_results"]:
                    report += f"  • {login['username']}:{login['password']}\n"
            report += "\n"
        
        return report

# Example usage
async def main():
    """Example usage of SMB enumerator"""
    print("🔍 SMB Enumerator Example")
    
    # Create enumeration configuration
    config = EnumerationConfig(
        target_host="192.168.1.100",
        target_port=445,
        timeout=30.0,
        max_concurrent=5,
        enable_anonymous=True,
        enable_guest=True,
        enable_bruteforce=False  # Set to True for brute force testing
    )
    
    # Create enumerator
    enumerator = SMBEnumerator(config)
    
    # Perform enumeration
    print(f"Enumerating SMB on {config.target_host}...")
    result = await enumerator.enumerate_smb()
    
    if result["success"]:
        print(f"✅ Enumeration completed in {result['enumeration_duration']:.2f} seconds")
        print(f"📊 Found {result['total_shares']} shares")
        print(f"🔓 {result['accessible_shares']} shares are accessible")
        print(f"👥 Found {result['total_users']} users")
        print(f"📁 Found {result['total_files']} files")
        
        # Print some results
        if result['results']['shares']:
            print("\n📋 Shares:")
            for share in result['results']['shares'][:5]:  # Show first 5
                print(f"  {share['name']} ({share['share_type']}) - {'Accessible' if share['accessible'] else 'Not Accessible'}")
        
        if result['results']['users']:
            print("\n👥 Users:")
            for user in result['results']['users'][:5]:  # Show first 5
                print(f"  {user['username']}")
        
        # Generate report
        print("\n📋 SMB Enumeration Report:")
        print(enumerator.generate_report())
        
    else:
        print(f"❌ Enumeration failed: {result['error']}")

if __name__ == "__main__":
    asyncio.run(main()) 