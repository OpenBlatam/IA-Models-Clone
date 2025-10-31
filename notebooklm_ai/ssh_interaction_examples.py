from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import logging
import os
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse
    import paramiko
    from paramiko import SSHClient, AutoAddPolicy, RSAKey, Ed25519Key
    from paramiko.ssh_exception import (
    import asyncssh
    from asyncssh import connect, SSHClientConnection, SSHClientSession
    from asyncssh.connection import SSHConnection
from typing import Any, List, Dict, Optional
"""
SSH Interaction Examples - Comprehensive SSH Operations
=====================================================

This module provides robust SSH interaction capabilities using both:
- paramiko: Synchronous SSH operations
- asyncssh: Asynchronous SSH operations

Features:
- Connection management with retry logic
- Command execution with timeout handling
- File transfer operations
- SFTP operations
- Key-based and password authentication
- Comprehensive error handling and logging
- Connection pooling and resource management

Author: AI Assistant
License: MIT
"""


try:
        SSHException, AuthenticationException, NoValidConnectionsError,
        BadHostKeyException, ChannelException
    )
    PARAMIKO_AVAILABLE = True
except ImportError:
    PARAMIKO_AVAILABLE = False
    SSHClient = None
    AutoAddPolicy = None
    RSAKey = None
    Ed25519Key = None
    SSHException = Exception
    AuthenticationException = Exception
    NoValidConnectionsError = Exception
    BadHostKeyException = Exception
    ChannelException = Exception

try:
    ASYNCSSH_AVAILABLE = True
except ImportError:
    ASYNCSSH_AVAILABLE = False
    connect = None
    SSHClientConnection = None
    SSHClientSession = None
    SSHConnection = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SSHConfig:
    """SSH connection configuration."""
    hostname: str
    port: int = 22
    username: str = ""
    password: Optional[str] = None
    key_filename: Optional[str] = None
    key_data: Optional[str] = None
    timeout: int = 30
    banner_timeout: int = 60
    auth_timeout: int = 60
    max_retries: int = 3
    retry_delay: float = 1.0
    allow_agent: bool = True
    look_for_keys: bool = True
    compress: bool = False
    gss_auth: bool = False
    gss_kex: bool = False
    gss_deleg_creds: bool = False
    gss_host: Optional[str] = None
    known_hosts: Optional[str] = None
    hostkey_verify: bool = True
    hostkey_check: bool = True
    proxy_command: Optional[str] = None
    sock: Optional[socket.socket] = None
    keepalive_interval: int = 0
    keepalive_count_max: int = 3


@dataclass
class SSHResult:
    """Result of SSH command execution."""
    success: bool
    stdout: str = ""
    stderr: str = ""
    exit_code: int = -1
    execution_time: float = 0.0
    error_message: str = ""
    command: str = ""


@dataclass
class FileTransferResult:
    """Result of file transfer operation."""
    success: bool
    local_path: str = ""
    remote_path: str = ""
    bytes_transferred: int = 0
    error_message: str = ""
    transfer_time: float = 0.0


class SSHConnectionError(Exception):
    """Custom exception for SSH connection errors."""
    pass


class SSHCommandError(Exception):
    """Custom exception for SSH command execution errors."""
    pass


class SSHFileTransferError(Exception):
    """Custom exception for SSH file transfer errors."""
    pass


class ParamikoSSHManager:
    """Synchronous SSH operations using paramiko."""
    
    def __init__(self, config: SSHConfig):
        """Initialize SSH manager with configuration."""
        if not PARAMIKO_AVAILABLE:
            raise ImportError("paramiko is not available. Install with: pip install paramiko")
        
        self.config = config
        self.client: Optional[SSHClient] = None
        self.sftp: Optional[paramiko.SFTPClient] = None
        self._connected = False
        
    def __enter__(self) -> Any:
        """Context manager entry."""
        self.connect()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb) -> Any:
        """Context manager exit."""
        self.disconnect()
        
    def connect(self) -> bool:
        """Establish SSH connection with retry logic."""
        if self._connected and self.client:
            return True
            
        if not self.config.hostname:
            logger.error("Hostname is required for SSH connection")
            return False
            
        if not self.config.username:
            logger.error("Username is required for SSH connection")
            return False
            
        if not self.config.password and not self.config.key_filename and not self.config.key_data:
            logger.error("Either password or key must be provided")
            return False
        
        for attempt in range(self.config.max_retries):
            try:
                logger.info(f"Attempting SSH connection to {self.config.hostname}:{self.config.port} (attempt {attempt + 1})")
                
                self.client = SSHClient()
                self.client.set_missing_host_key_policy(AutoAddPolicy())
                
                # Prepare connection parameters
                connect_kwargs = {
                    'hostname': self.config.hostname,
                    'port': self.config.port,
                    'username': self.config.username,
                    'timeout': self.config.timeout,
                    'banner_timeout': self.config.banner_timeout,
                    'auth_timeout': self.config.auth_timeout,
                    'allow_agent': self.config.allow_agent,
                    'look_for_keys': self.config.look_for_keys,
                    'compress': self.config.compress,
                    'gss_auth': self.config.gss_auth,
                    'gss_kex': self.config.gss_kex,
                    'gss_deleg_creds': self.config.gss_deleg_creds,
                    'keepalive_interval': self.config.keepalive_interval,
                    'keepalive_count_max': self.config.keepalive_count_max
                }
                
                # Add authentication method
                if self.config.password:
                    connect_kwargs['password'] = self.config.password
                elif self.config.key_filename:
                    connect_kwargs['key_filename'] = self.config.key_filename
                elif self.config.key_data:
                    # Try to load key from data
                    try:
                        key = RSAKey.from_private_key(io.StringIO(self.config.key_data))
                        connect_kwargs['pkey'] = key
                    except Exception:
                        try:
                            key = Ed25519Key.from_private_key(io.StringIO(self.config.key_data))
                            connect_kwargs['pkey'] = key
                        except Exception as e:
                            logger.error(f"Failed to load key from data: {e}")
                            return False
                
                # Add optional parameters
                if self.config.gss_host:
                    connect_kwargs['gss_host'] = self.config.gss_host
                if self.config.known_hosts:
                    connect_kwargs['known_hosts'] = self.config.known_hosts
                if self.config.proxy_command:
                    connect_kwargs['sock'] = self._create_proxy_socket()
                if self.config.sock:
                    connect_kwargs['sock'] = self.config.sock
                
                self.client.connect(**connect_kwargs)
                self._connected = True
                
                logger.info(f"Successfully connected to {self.config.hostname}")
                return True
                
            except AuthenticationException as e:
                logger.error(f"Authentication failed: {e}")
                return False
            except NoValidConnectionsError as e:
                logger.error(f"No valid connections: {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)
                    continue
                return False
            except SSHException as e:
                logger.error(f"SSH exception: {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)
                    continue
                return False
            except Exception as e:
                logger.error(f"Unexpected error during connection: {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)
                    continue
                return False
        
        return False
    
    def disconnect(self) -> Any:
        """Close SSH connection."""
        if self.sftp:
            try:
                self.sftp.close()
            except Exception as e:
                logger.warning(f"Error closing SFTP: {e}")
            self.sftp = None
            
        if self.client:
            try:
                self.client.close()
            except Exception as e:
                logger.warning(f"Error closing SSH client: {e}")
            self.client = None
            
        self._connected = False
        logger.info("SSH connection closed")
    
    def execute_command(self, command: str, timeout: int = 30) -> SSHResult:
        """Execute command on remote host."""
        if not self._connected or not self.client:
            return SSHResult(
                success=False,
                error_message="Not connected to SSH server",
                command=command
            )
        
        if not command or not command.strip():
            return SSHResult(
                success=False,
                error_message="Command cannot be empty",
                command=command
            )
        
        start_time = time.time()
        
        try:
            logger.info(f"Executing command: {command}")
            
            stdin, stdout, stderr = self.client.exec_command(
                command,
                timeout=timeout,
                get_pty=False
            )
            
            # Read output with timeout
            stdout_data = ""
            stderr_data = ""
            
            # Read stdout
            if stdout.channel.recv_ready():
                stdout_data = stdout.read().decode('utf-8', errors='ignore')
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            
            # Read stderr
            if stderr.channel.recv_stderr_ready():
                stderr_data = stderr.read().decode('utf-8', errors='ignore')
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            
            # Wait for command to complete
            exit_code = stdout.channel.recv_exit_status()
            
            # Read any remaining output
            if stdout.channel.recv_ready():
                stdout_data += stdout.read().decode('utf-8', errors='ignore')
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            
            if stderr.channel.recv_stderr_ready():
                stderr_data += stderr.read().decode('utf-8', errors='ignore')
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            
            execution_time = time.time() - start_time
            
            success = exit_code == 0
            if not success:
                logger.warning(f"Command failed with exit code {exit_code}: {stderr_data}")
            
            return SSHResult(
                success=success,
                stdout=stdout_data,
                stderr=stderr_data,
                exit_code=exit_code,
                execution_time=execution_time,
                command=command
            )
            
        except ChannelException as e:
            execution_time = time.time() - start_time
            logger.error(f"Channel exception during command execution: {e}")
            return SSHResult(
                success=False,
                error_message=f"Channel error: {e}",
                execution_time=execution_time,
                command=command
            )
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Unexpected error during command execution: {e}")
            return SSHResult(
                success=False,
                error_message=f"Unexpected error: {e}",
                execution_time=execution_time,
                command=command
            )
    
    async def upload_file(self, local_path: str, remote_path: str) -> FileTransferResult:
        """Upload file to remote host."""
        if not self._connected or not self.client:
            return FileTransferResult(
                success=False,
                local_path=local_path,
                remote_path=remote_path,
                error_message="Not connected to SSH server"
            )
        
        if not os.path.exists(local_path):
            return FileTransferResult(
                success=False,
                local_path=local_path,
                remote_path=remote_path,
                error_message=f"Local file does not exist: {local_path}"
            )
        
        start_time = time.time()
        
        try:
            if not self.sftp:
                self.sftp = self.client.open_sftp()
            
            logger.info(f"Uploading {local_path} to {remote_path}")
            
            # Get file size for progress tracking
            file_size = os.path.getsize(local_path)
            
            # Upload with progress callback
            def progress_callback(transferred, to_be_transferred) -> Any:
                if to_be_transferred > 0:
                    percentage = (transferred / to_be_transferred) * 100
                    logger.debug(f"Upload progress: {percentage:.1f}% ({transferred}/{to_be_transferred} bytes)")
            
            self.sftp.put(local_path, remote_path, callback=progress_callback)
            
            transfer_time = time.time() - start_time
            
            logger.info(f"Successfully uploaded {local_path} to {remote_path} in {transfer_time:.2f}s")
            
            return FileTransferResult(
                success=True,
                local_path=local_path,
                remote_path=remote_path,
                bytes_transferred=file_size,
                transfer_time=transfer_time
            )
            
        except Exception as e:
            transfer_time = time.time() - start_time
            logger.error(f"Error uploading file: {e}")
            return FileTransferResult(
                success=False,
                local_path=local_path,
                remote_path=remote_path,
                error_message=f"Upload error: {e}",
                transfer_time=transfer_time
            )
    
    async def download_file(self, remote_path: str, local_path: str) -> FileTransferResult:
        """Download file from remote host."""
        if not self._connected or not self.client:
            return FileTransferResult(
                success=False,
                local_path=local_path,
                remote_path=remote_path,
                error_message="Not connected to SSH server"
            )
        
        start_time = time.time()
        
        try:
            if not self.sftp:
                self.sftp = self.client.open_sftp()
            
            logger.info(f"Downloading {remote_path} to {local_path}")
            
            # Check if remote file exists
            try:
                remote_stat = self.sftp.stat(remote_path)
                file_size = remote_stat.st_size
            except FileNotFoundError:
                return FileTransferResult(
                    success=False,
                    local_path=local_path,
                    remote_path=remote_path,
                    error_message=f"Remote file does not exist: {remote_path}"
                )
            
            # Download with progress callback
            def progress_callback(transferred, to_be_transferred) -> Any:
                if to_be_transferred > 0:
                    percentage = (transferred / to_be_transferred) * 100
                    logger.debug(f"Download progress: {percentage:.1f}% ({transferred}/{to_be_transferred} bytes)")
            
            self.sftp.get(remote_path, local_path, callback=progress_callback)
            
            transfer_time = time.time() - start_time
            
            logger.info(f"Successfully downloaded {remote_path} to {local_path} in {transfer_time:.2f}s")
            
            return FileTransferResult(
                success=True,
                local_path=local_path,
                remote_path=remote_path,
                bytes_transferred=file_size,
                transfer_time=transfer_time
            )
            
        except Exception as e:
            transfer_time = time.time() - start_time
            logger.error(f"Error downloading file: {e}")
            return FileTransferResult(
                success=False,
                local_path=local_path,
                remote_path=remote_path,
                error_message=f"Download error: {e}",
                transfer_time=transfer_time
            )
    
    def _create_proxy_socket(self) -> socket.socket:
        """Create proxy socket for proxy command."""
        # Implementation depends on proxy command format
        # This is a simplified example
        raise NotImplementedError("Proxy command support not implemented")


class AsyncSSHManager:
    """Asynchronous SSH operations using asyncssh."""
    
    def __init__(self, config: SSHConfig):
        """Initialize async SSH manager with configuration."""
        if not ASYNCSSH_AVAILABLE:
            raise ImportError("asyncssh is not available. Install with: pip install asyncssh")
        
        self.config = config
        self.connection: Optional[SSHClientConnection] = None
        self._connected = False
        
    async def __aenter__(self) -> Any:
        """Async context manager entry."""
        await self.connect()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> Any:
        """Async context manager exit."""
        await self.disconnect()
    
    async def connect(self) -> bool:
        """Establish async SSH connection with retry logic."""
        if self._connected and self.connection:
            return True
            
        if not self.config.hostname:
            logger.error("Hostname is required for SSH connection")
            return False
            
        if not self.config.username:
            logger.error("Username is required for SSH connection")
            return False
            
        if not self.config.password and not self.config.key_filename and not self.config.key_data:
            logger.error("Either password or key must be provided")
            return False
        
        for attempt in range(self.config.max_retries):
            try:
                logger.info(f"Attempting async SSH connection to {self.config.hostname}:{self.config.port} (attempt {attempt + 1})")
                
                # Prepare connection parameters
                connect_kwargs = {
                    'host': self.config.hostname,
                    'port': self.config.port,
                    'username': self.config.username,
                    'connect_timeout': self.config.timeout,
                    'login_timeout': self.config.auth_timeout,
                    'keepalive_interval': self.config.keepalive_interval,
                    'keepalive_count_max': self.config.keepalive_count_max,
                    'known_hosts': None if not self.config.hostkey_check else self.config.known_hosts,
                    'gss_host': self.config.gss_host,
                    'gss_auth': self.config.gss_auth,
                    'gss_kex': self.config.gss_kex,
                    'gss_deleg_creds': self.config.gss_deleg_creds
                }
                
                # Add authentication method
                if self.config.password:
                    connect_kwargs['password'] = self.config.password
                elif self.config.key_filename:
                    connect_kwargs['client_keys'] = [self.config.key_filename]
                elif self.config.key_data:
                    # Load key from data
                    try:
                        key = asyncssh.import_private_key(self.config.key_data)
                        connect_kwargs['client_keys'] = [key]
                    except Exception as e:
                        logger.error(f"Failed to load key from data: {e}")
                        return False
                
                self.connection = await connect(**connect_kwargs)
                self._connected = True
                
                logger.info(f"Successfully connected to {self.config.hostname}")
                return True
                
            except asyncssh.AuthenticationError as e:
                logger.error(f"Authentication failed: {e}")
                return False
            except asyncssh.ConnectionLost as e:
                logger.error(f"Connection lost: {e}")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay)
                    continue
                return False
            except asyncssh.Error as e:
                logger.error(f"SSH error: {e}")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay)
                    continue
                return False
            except Exception as e:
                logger.error(f"Unexpected error during connection: {e}")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay)
                    continue
                return False
        
        return False
    
    async def disconnect(self) -> Any:
        """Close async SSH connection."""
        if self.connection:
            try:
                self.connection.close()
                await self.connection.wait_closed()
            except Exception as e:
                logger.warning(f"Error closing async SSH connection: {e}")
            self.connection = None
            
        self._connected = False
        logger.info("Async SSH connection closed")
    
    async def execute_command(self, command: str, timeout: int = 30) -> SSHResult:
        """Execute command on remote host asynchronously."""
        if not self._connected or not self.connection:
            return SSHResult(
                success=False,
                error_message="Not connected to SSH server",
                command=command
            )
        
        if not command or not command.strip():
            return SSHResult(
                success=False,
                error_message="Command cannot be empty",
                command=command
            )
        
        start_time = time.time()
        
        try:
            logger.info(f"Executing async command: {command}")
            
            # Execute command with timeout
            async with self.connection.create_process(command) as process:
                stdout_data = await asyncio.wait_for(process.stdout.read(), timeout=timeout)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                stderr_data = await asyncio.wait_for(process.stderr.read(), timeout=timeout)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                exit_code = await asyncio.wait_for(process.wait(), timeout=timeout)
            
            execution_time = time.time() - start_time
            
            success = exit_code == 0
            if not success:
                logger.warning(f"Command failed with exit code {exit_code}: {stderr_data}")
            
            return SSHResult(
                success=success,
                stdout=stdout_data.decode('utf-8', errors='ignore'),
                stderr=stderr_data.decode('utf-8', errors='ignore'),
                exit_code=exit_code,
                execution_time=execution_time,
                command=command
            )
            
        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            logger.error(f"Command execution timed out after {timeout}s")
            return SSHResult(
                success=False,
                error_message=f"Command timed out after {timeout}s",
                execution_time=execution_time,
                command=command
            )
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Unexpected error during command execution: {e}")
            return SSHResult(
                success=False,
                error_message=f"Unexpected error: {e}",
                execution_time=execution_time,
                command=command
            )
    
    async async def upload_file(self, local_path: str, remote_path: str) -> FileTransferResult:
        """Upload file to remote host asynchronously."""
        if not self._connected or not self.connection:
            return FileTransferResult(
                success=False,
                local_path=local_path,
                remote_path=remote_path,
                error_message="Not connected to SSH server"
            )
        
        if not os.path.exists(local_path):
            return FileTransferResult(
                success=False,
                local_path=local_path,
                remote_path=remote_path,
                error_message=f"Local file does not exist: {local_path}"
            )
        
        start_time = time.time()
        
        try:
            logger.info(f"Uploading {local_path} to {remote_path}")
            
            # Get file size for progress tracking
            file_size = os.path.getsize(local_path)
            
            # Upload file
            async with self.connection.start_sftp_client() as sftp:
                await sftp.put(local_path, remote_path)
            
            transfer_time = time.time() - start_time
            
            logger.info(f"Successfully uploaded {local_path} to {remote_path} in {transfer_time:.2f}s")
            
            return FileTransferResult(
                success=True,
                local_path=local_path,
                remote_path=remote_path,
                bytes_transferred=file_size,
                transfer_time=transfer_time
            )
            
        except Exception as e:
            transfer_time = time.time() - start_time
            logger.error(f"Error uploading file: {e}")
            return FileTransferResult(
                success=False,
                local_path=local_path,
                remote_path=remote_path,
                error_message=f"Upload error: {e}",
                transfer_time=transfer_time
            )
    
    async async def download_file(self, remote_path: str, local_path: str) -> FileTransferResult:
        """Download file from remote host asynchronously."""
        if not self._connected or not self.connection:
            return FileTransferResult(
                success=False,
                local_path=local_path,
                remote_path=remote_path,
                error_message="Not connected to SSH server"
            )
        
        start_time = time.time()
        
        try:
            logger.info(f"Downloading {remote_path} to {local_path}")
            
            # Download file
            async with self.connection.start_sftp_client() as sftp:
                # Check if remote file exists
                try:
                    remote_stat = await sftp.stat(remote_path)
                    file_size = remote_stat.size
                except FileNotFoundError:
                    return FileTransferResult(
                        success=False,
                        local_path=local_path,
                        remote_path=remote_path,
                        error_message=f"Remote file does not exist: {remote_path}"
                    )
                
                await sftp.get(remote_path, local_path)
            
            transfer_time = time.time() - start_time
            
            logger.info(f"Successfully downloaded {remote_path} to {local_path} in {transfer_time:.2f}s")
            
            return FileTransferResult(
                success=True,
                local_path=local_path,
                remote_path=remote_path,
                bytes_transferred=file_size,
                transfer_time=transfer_time
            )
            
        except Exception as e:
            transfer_time = time.time() - start_time
            logger.error(f"Error downloading file: {e}")
            return FileTransferResult(
                success=False,
                local_path=local_path,
                remote_path=remote_path,
                error_message=f"Download error: {e}",
                transfer_time=transfer_time
            )


class SSHConnectionPool:
    """Connection pool for SSH connections."""
    
    def __init__(self, max_connections: int = 10):
        """Initialize connection pool."""
        self.max_connections = max_connections
        self.connections: List[Union[ParamikoSSHManager, AsyncSSHManager]] = []
        self._lock = asyncio.Lock()
    
    async def get_connection(self, config: SSHConfig, async_mode: bool = True) -> Union[ParamikoSSHManager, AsyncSSHManager]:
        """Get connection from pool or create new one."""
        async with self._lock:
            # Check for available connections
            for conn in self.connections:
                if not conn._connected:
                    # Reuse existing connection
                    if async_mode and isinstance(conn, AsyncSSHManager):
                        await conn.connect()
                        return conn
                    elif not async_mode and isinstance(conn, ParamikoSSHManager):
                        conn.connect()
                        return conn
            
            # Create new connection if pool not full
            if len(self.connections) < self.max_connections:
                if async_mode:
                    conn = AsyncSSHManager(config)
                    await conn.connect()
                else:
                    conn = ParamikoSSHManager(config)
                    conn.connect()
                
                self.connections.append(conn)
                return conn
            
            # Wait for available connection
            while True:
                for conn in self.connections:
                    if not conn._connected:
                        if async_mode and isinstance(conn, AsyncSSHManager):
                            await conn.connect()
                            return conn
                        elif not async_mode and isinstance(conn, ParamikoSSHManager):
                            conn.connect()
                            return conn
                
                await asyncio.sleep(0.1)
    
    async def return_connection(self, connection: Union[ParamikoSSHManager, AsyncSSHManager]):
        """Return connection to pool."""
        async with self._lock:
            if connection in self.connections:
                # Keep connection in pool but mark as available
                pass


# Example usage functions
def demonstrate_paramiko_usage():
    """Demonstrate paramiko SSH usage."""
    if not PARAMIKO_AVAILABLE:
        logger.error("paramiko not available")
        return
    
    config = SSHConfig(
        hostname="example.com",
        username="user",
        password="password",
        timeout=30
    )
    
    # Using context manager
    with ParamikoSSHManager(config) as ssh:
        # Execute command
        result = ssh.execute_command("ls -la")
        if result.success:
            logger.info(f"Command output: {result.stdout}")
        else:
            logger.error(f"Command failed: {result.error_message}")
        
        # Upload file
        upload_result = ssh.upload_file("local_file.txt", "/remote/path/file.txt")
        if upload_result.success:
            logger.info(f"Upload successful: {upload_result.bytes_transferred} bytes")
        else:
            logger.error(f"Upload failed: {upload_result.error_message}")
        
        # Download file
        download_result = ssh.download_file("/remote/path/file.txt", "downloaded_file.txt")
        if download_result.success:
            logger.info(f"Download successful: {download_result.bytes_transferred} bytes")
        else:
            logger.error(f"Download failed: {download_result.error_message}")


async def demonstrate_asyncssh_usage():
    """Demonstrate asyncssh usage."""
    if not ASYNCSSH_AVAILABLE:
        logger.error("asyncssh not available")
        return
    
    config = SSHConfig(
        hostname="example.com",
        username="user",
        password="password",
        timeout=30
    )
    
    # Using async context manager
    async with AsyncSSHManager(config) as ssh:
        # Execute command
        result = await ssh.execute_command("ls -la")
        if result.success:
            logger.info(f"Command output: {result.stdout}")
        else:
            logger.error(f"Command failed: {result.error_message}")
        
        # Upload file
        upload_result = await ssh.upload_file("local_file.txt", "/remote/path/file.txt")
        if upload_result.success:
            logger.info(f"Upload successful: {upload_result.bytes_transferred} bytes")
        else:
            logger.error(f"Upload failed: {upload_result.error_message}")
        
        # Download file
        download_result = await ssh.download_file("/remote/path/file.txt", "downloaded_file.txt")
        if download_result.success:
            logger.info(f"Download successful: {download_result.bytes_transferred} bytes")
        else:
            logger.error(f"Download failed: {download_result.error_message}")


async def demonstrate_connection_pool():
    """Demonstrate connection pool usage."""
    config = SSHConfig(
        hostname="example.com",
        username="user",
        password="password"
    )
    
    pool = SSHConnectionPool(max_connections=5)
    
    # Get connection from pool
    ssh = await pool.get_connection(config, async_mode=True)
    
    try:
        # Use connection
        result = await ssh.execute_command("echo 'Hello from pool'")
        logger.info(f"Pool command result: {result.success}")
    finally:
        # Return connection to pool
        await pool.return_connection(ssh)


def main():
    """Main function demonstrating SSH usage."""
    logger.info("Starting SSH interaction examples")
    
    # Demonstrate paramiko usage
    try:
        demonstrate_paramiko_usage()
    except Exception as e:
        logger.error(f"Paramiko demonstration failed: {e}")
    
    # Demonstrate asyncssh usage
    try:
        asyncio.run(demonstrate_asyncssh_usage())
    except Exception as e:
        logger.error(f"AsyncSSH demonstration failed: {e}")
    
    # Demonstrate connection pool
    try:
        asyncio.run(demonstrate_connection_pool())
    except Exception as e:
        logger.error(f"Connection pool demonstration failed: {e}")
    
    logger.info("SSH interaction examples completed")


match __name__:
    case "__main__":
    main() 