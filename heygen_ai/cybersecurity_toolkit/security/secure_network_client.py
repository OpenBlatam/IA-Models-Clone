from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import asyncio
import socket
import ssl
import subprocess
import shlex
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
import logging
from ..utils.structured_logger import get_logger, log_async_function_call, log_function_call
from .input_sanitizer import InputSanitizer, sanitize_string, validate_ip_address, validate_hostname
from .secure_defaults import SecureDefaults, create_secure_ssl_context, get_secure_headers
from ..exceptions.custom_exceptions import (
from typing import Any, List, Dict, Optional
"""
Secure Network Client Module
===========================

Secure network client with input sanitization and secure defaults:
- Input validation and sanitization
- Secure TLS configurations
- Strong cipher suites
- Safe command execution
- Comprehensive error handling
"""


# Import structured logger

# Import input sanitizer

# Import secure defaults

# Import custom exceptions
    ValidationError,
    SecurityError,
    NetworkError,
    ConnectionTimeoutError,
    ConnectionRefusedError
)

# Get logger instance
logger = get_logger("secure_network_client")

class SecureNetworkClient:
    """
    Secure network client with input sanitization and secure defaults.
    
    Provides secure network operations with comprehensive input validation,
    secure TLS configurations, and safe command execution.
    """
    
    def __init__(self, 
                 timeout: float = 10.0,
                 max_retries: int = 3,
                 verify_ssl: bool = True):
        """
        Initialize secure network client.
        
        Args:
            timeout: Connection timeout in seconds
            max_retries: Maximum retry attempts
            verify_ssl: Whether to verify SSL certificates
        """
        # Guard clause 1: Validate timeout
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError("Timeout must be a positive number")
        
        # Guard clause 2: Validate max_retries
        if not isinstance(max_retries, int) or max_retries < 0:
            raise ValueError("Max retries must be a non-negative integer")
        
        # Guard clause 3: Validate verify_ssl
        if not isinstance(verify_ssl, bool):
            raise ValueError("Verify SSL must be a boolean")
        
        # Happy path: Set secure defaults
        self.timeout = timeout
        self.max_retries = max_retries
        self.verify_ssl = verify_ssl
        
        # Initialize components
        self.input_sanitizer = InputSanitizer()
        self.secure_defaults = SecureDefaults()
        
        # Create secure SSL context
        self.ssl_context = create_secure_ssl_context()
        
        logger.log_function_entry(
            "__init__",
            {
                "timeout": timeout,
                "max_retries": max_retries,
                "verify_ssl": verify_ssl
            },
            {"event_type": "secure_network_client_initialization"}
        )
    
    @log_async_function_call
    async def secure_connect_async(self, 
                                 host: str, 
                                 port: int, 
                                 use_ssl: bool = True) -> Dict[str, Any]:
        """
        Establish secure connection with input validation.
        
        Args:
            host: Target host
            port: Target port
            use_ssl: Whether to use SSL/TLS
            
        Returns:
            Connection result dictionary
            
        Raises:
            ValidationError: When input validation fails
            SecurityError: When security checks fail
            NetworkError: When connection fails
        """
        # Guard clause 1: Validate and sanitize host
        try:
            validated_host = validate_hostname(host)
        except Exception:
            try:
                validated_host = validate_ip_address(host)
            except Exception as e:
                raise ValidationError(f"Invalid host: {host}", context={"error": str(e)})
        
        # Guard clause 2: Validate port
        if not isinstance(port, int) or port < 1 or port > 65535:
            raise ValidationError(f"Invalid port: {port}")
        
        # Guard clause 3: Attempt connection
        connection_start = asyncio.get_event_loop().time()
        
        try:
            # Create connection
            if use_ssl:
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(
                        validated_host, 
                        port, 
                        ssl=self.ssl_context if use_ssl else None
                    ),
                    timeout=self.timeout
                )
            else:
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(validated_host, port),
                    timeout=self.timeout
                )
            
            # Get connection info
            sock = writer.get_extra_info('socket')
            local_addr = sock.getsockname()
            remote_addr = sock.getpeername()
            
            # Get SSL info if using SSL
            ssl_info = None
            if use_ssl and writer.get_extra_info('ssl_object'):
                ssl_obj = writer.get_extra_info('ssl_object')
                ssl_info = {
                    "version": ssl_obj.version(),
                    "cipher": ssl_obj.cipher(),
                    "compression": ssl_obj.compression(),
                    "verify_mode": ssl_obj.verify_mode
                }
            
            connection_duration = asyncio.get_event_loop().time() - connection_start
            
            result = {
                "success": True,
                "host": validated_host,
                "port": port,
                "use_ssl": use_ssl,
                "local_address": local_addr,
                "remote_address": remote_addr,
                "ssl_info": ssl_info,
                "connection_duration": connection_duration,
                "connection_timestamp": datetime.utcnow().isoformat()
            }
            
            # Close connection
            writer.close()
            await writer.wait_closed()
            
            logger.log_function_exit(
                "secure_connect_async",
                result,
                context={"connection_result": "success"}
            )
            
            return result
            
        except asyncio.TimeoutError:
            raise ConnectionTimeoutError(
                target=validated_host,
                port=port,
                timeout=self.timeout,
                context={"operation": "secure_connect"}
            )
        except ConnectionRefusedError:
            raise ConnectionRefusedError(
                target=validated_host,
                port=port,
                context={"operation": "secure_connect"}
            )
        except Exception as e:
            raise NetworkError(
                target=validated_host,
                operation="secure_connect",
                message=f"Connection failed: {str(e)}",
                context={"port": port, "use_ssl": use_ssl},
                original_exception=e
            )
    
    @log_function_call
    def secure_connect_sync(self, 
                          host: str, 
                          port: int, 
                          use_ssl: bool = True) -> Dict[str, Any]:
        """
        Establish secure connection synchronously.
        
        Args:
            host: Target host
            port: Target port
            use_ssl: Whether to use SSL/TLS
            
        Returns:
            Connection result dictionary
        """
        # Run async function in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.secure_connect_async(host, port, use_ssl))
        finally:
            loop.close()
    
    @log_function_call
    def execute_secure_command(self, 
                             command: str, 
                             args: List[str],
                             timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Execute command securely with input validation.
        
        Args:
            command: Command to execute
            args: Command arguments
            timeout: Command timeout
            
        Returns:
            Command execution result
            
        Raises:
            SecurityError: When command is not in safe whitelist
            ValidationError: When input validation fails
        """
        # Guard clause 1: Validate timeout
        if timeout is None:
            timeout = self.timeout
        elif not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValidationError("Timeout must be a positive number")
        
        # Guard clause 2: Sanitize command and arguments
        try:
            sanitized_command, sanitized_args = self.input_sanitizer.sanitize_command_args(command, args)
        except Exception as e:
            raise SecurityError(f"Command sanitization failed: {str(e)}")
        
        # Guard clause 3: Execute command
        execution_start = datetime.utcnow()
        
        try:
            # Build command list
            cmd_list = [sanitized_command] + sanitized_args
            
            # Execute command
            result = subprocess.run(
                cmd_list,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False
            )
            
            execution_duration = (datetime.utcnow() - execution_start).total_seconds()
            
            # Prepare result
            execution_result = {
                "success": result.returncode == 0,
                "command": sanitized_command,
                "arguments": sanitized_args,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "execution_duration": execution_duration,
                "execution_timestamp": execution_start.isoformat()
            }
            
            logger.log_function_exit(
                "execute_secure_command",
                execution_result,
                context={"command_execution": "success" if result.returncode == 0 else "failed"}
            )
            
            return execution_result
            
        except subprocess.TimeoutExpired:
            raise SecurityError(
                f"Command execution timed out after {timeout} seconds",
                context={"command": sanitized_command, "timeout": timeout}
            )
        except Exception as e:
            raise SecurityError(
                f"Command execution failed: {str(e)}",
                context={"command": sanitized_command, "args": sanitized_args}
            )
    
    @log_function_call
    def ping_host_secure(self, host: str, count: int = 1) -> Dict[str, Any]:
        """
        Ping host securely with input validation.
        
        Args:
            host: Target host
            count: Number of ping attempts
            
        Returns:
            Ping result dictionary
        """
        # Guard clause 1: Validate host
        try:
            validated_host = validate_hostname(host)
        except Exception:
            try:
                validated_host = validate_ip_address(host)
            except Exception as e:
                raise ValidationError(f"Invalid host: {host}", context={"error": str(e)})
        
        # Guard clause 2: Validate count
        if not isinstance(count, int) or count < 1 or count > 10:
            raise ValidationError("Count must be between 1 and 10")
        
        # Guard clause 3: Execute ping command
        ping_args = ["-c", str(count)]
        
        if validated_host.replace('.', '').isdigit():  # IP address
            ping_args.append(validated_host)
        else:  # Hostname
            ping_args.append(validated_host)
        
        return self.execute_secure_command("ping", ping_args)
    
    @log_function_call
    def nslookup_secure(self, host: str) -> Dict[str, Any]:
        """
        Perform nslookup securely with input validation.
        
        Args:
            host: Target host
            
        Returns:
            Nslookup result dictionary
        """
        # Guard clause 1: Validate host
        try:
            validated_host = validate_hostname(host)
        except Exception:
            try:
                validated_host = validate_ip_address(host)
            except Exception as e:
                raise ValidationError(f"Invalid host: {host}", context={"error": str(e)})
        
        # Guard clause 2: Execute nslookup command
        return self.execute_secure_command("nslookup", [validated_host])
    
    @log_function_call
    def traceroute_secure(self, host: str) -> Dict[str, Any]:
        """
        Perform traceroute securely with input validation.
        
        Args:
            host: Target host
            
        Returns:
            Traceroute result dictionary
        """
        # Guard clause 1: Validate host
        try:
            validated_host = validate_hostname(host)
        except Exception:
            try:
                validated_host = validate_ip_address(host)
            except Exception as e:
                raise ValidationError(f"Invalid host: {host}", context={"error": str(e)})
        
        # Guard clause 2: Execute traceroute command
        return self.execute_secure_command("traceroute", [validated_host])
    
    @log_function_call
    def validate_network_target_secure(self, target: str) -> Dict[str, Any]:
        """
        Validate network target securely.
        
        Args:
            target: Network target to validate
            
        Returns:
            Validation result dictionary
        """
        # Guard clause 1: Sanitize input
        sanitized_target = sanitize_string(target, max_length=253)
        
        validation_result = {
            "target": sanitized_target,
            "is_valid": False,
            "target_type": None,
            "validation_timestamp": datetime.utcnow().isoformat()
        }
        
        # Guard clause 2: Try to validate as hostname
        try:
            validated_hostname = validate_hostname(sanitized_target)
            validation_result.update({
                "is_valid": True,
                "target_type": "hostname",
                "validated_value": validated_hostname
            })
        except Exception:
            pass
        
        # Guard clause 3: Try to validate as IP address
        if not validation_result["is_valid"]:
            try:
                validated_ip = validate_ip_address(sanitized_target)
                validation_result.update({
                    "is_valid": True,
                    "target_type": "ip_address",
                    "validated_value": validated_ip
                })
            except Exception:
                pass
        
        # Guard clause 4: Return validation result
        logger.log_function_exit(
            "validate_network_target_secure",
            validation_result,
            context={"validation_result": "success" if validation_result["is_valid"] else "failed"}
        )
        
        return validation_result

# Global secure network client instance
_secure_network_client = SecureNetworkClient()

# Convenience functions
def secure_connect_async(host: str, port: int, use_ssl: bool = True) -> Dict[str, Any]:
    """Establish secure connection asynchronously."""
    return asyncio.run(_secure_network_client.secure_connect_async(host, port, use_ssl))

def secure_connect_sync(host: str, port: int, use_ssl: bool = True) -> Dict[str, Any]:
    """Establish secure connection synchronously."""
    return _secure_network_client.secure_connect_sync(host, port, use_ssl)

def execute_secure_command(command: str, args: List[str], timeout: Optional[float] = None) -> Dict[str, Any]:
    """Execute command securely with input validation."""
    return _secure_network_client.execute_secure_command(command, args, timeout)

def ping_host_secure(host: str, count: int = 1) -> Dict[str, Any]:
    """Ping host securely with input validation."""
    return _secure_network_client.ping_host_secure(host, count)

def nslookup_secure(host: str) -> Dict[str, Any]:
    """Perform nslookup securely with input validation."""
    return _secure_network_client.nslookup_secure(host)

def traceroute_secure(host: str) -> Dict[str, Any]:
    """Perform traceroute securely with input validation."""
    return _secure_network_client.traceroute_secure(host)

def validate_network_target_secure(target: str) -> Dict[str, Any]:
    """Validate network target securely."""
    return _secure_network_client.validate_network_target_secure(target)

# --- Named Exports ---

__all__ = [
    'SecureNetworkClient',
    'secure_connect_async',
    'secure_connect_sync',
    'execute_secure_command',
    'ping_host_secure',
    'nslookup_secure',
    'traceroute_secure',
    'validate_network_target_secure'
] 