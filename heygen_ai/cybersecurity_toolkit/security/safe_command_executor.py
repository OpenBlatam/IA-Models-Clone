from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import subprocess
import shlex
import os
import tempfile
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
import asyncio
import signal
import time
from typing import Any, List, Dict, Optional
"""
Safe Command Executor Module
===========================

Safe command execution with protection against shell injection:
- Input sanitization and validation
- Safe subprocess execution
- Command whitelisting
- Output sanitization
- Secure error handling
"""


# Get logger
logger = logging.getLogger(__name__)

class ValidationError(Exception):
    """Custom validation error."""
    def __init__(self, field: str, value: Any, error_type: str, message: str, context: Optional[Dict] = None):
        
    """__init__ function."""
self.field = field
        self.value = value
        self.error_type = error_type
        self.message = message
        self.context = context or {}
        super().__init__(message)

class SecurityError(Exception):
    """Custom security error."""
    def __init__(self, error_type: str, message: str, severity: str = "MEDIUM", context: Optional[Dict] = None):
        
    """__init__ function."""
self.error_type = error_type
        self.message = message
        self.severity = severity
        self.context = context or {}
        super().__init__(message)

class SafeCommandExecutor:
    """
    Safe command executor with protection against shell injection attacks.
    """
    
    def __init__(self, 
                 allowed_commands: Optional[List[str]] = None,
                 timeout: int = 30,
                 working_directory: Optional[str] = None,
                 environment: Optional[Dict[str, str]] = None):
        """
        Initialize safe command executor.
        
        Args:
            allowed_commands: List of allowed commands (whitelist)
            timeout: Command execution timeout in seconds
            working_directory: Working directory for command execution
            environment: Environment variables for command execution
        """
        # Guard clause 1: Validate timeout
        if not isinstance(timeout, int) or timeout <= 0:
            raise ValidationError(
                "timeout",
                timeout,
                "invalid_timeout",
                "Timeout must be a positive integer",
                context={"operation": "command_executor_initialization"}
            )
        
        # Guard clause 2: Validate working directory
        if working_directory and not os.path.exists(working_directory):
            raise SecurityError(
                "working_directory_not_found",
                f"Working directory does not exist: {working_directory}"
            )
        
        # Happy path: Set configuration
        self.allowed_commands = allowed_commands or []
        self.timeout = timeout
        self.working_directory = working_directory
        self.environment = environment or {}
        
        # Secure environment setup
        self._setup_secure_environment()
        
        logger.info("Safe command executor initialized", extra={
            "allowed_commands_count": len(self.allowed_commands),
            "timeout": timeout,
            "working_directory": working_directory
        })
    
    def _setup_secure_environment(self) -> Any:
        """Setup secure environment variables."""
        # Remove potentially dangerous environment variables
        dangerous_vars = [
            'PATH', 'LD_LIBRARY_PATH', 'LD_PRELOAD', 'PYTHONPATH',
            'PYTHONHOME', 'PYTHONEXECUTABLE', 'PYTHONSTARTUP'
        ]
        
        for var in dangerous_vars:
            if var in self.environment:
                del self.environment[var]
        
        # Set secure defaults
        self.environment.update({
            'PATH': '/usr/local/bin:/usr/bin:/bin',
            'SHELL': '/bin/bash',
            'TERM': 'xterm-256color',
            'LANG': 'en_US.UTF-8',
            'LC_ALL': 'en_US.UTF-8'
        })
    
    def _sanitize_string(self, input_string: str, max_length: int = 1000) -> str:
        """
        Basic string sanitization.
        
        Args:
            input_string: String to sanitize
            max_length: Maximum allowed length
            
        Returns:
            Sanitized string
        """
        if not isinstance(input_string, str):
            raise ValidationError(
                "input_string",
                input_string,
                "invalid_type",
                "Input must be a string"
            )
        
        if len(input_string) > max_length:
            raise ValidationError(
                "input_string",
                len(input_string),
                "input_too_long",
                f"Input too long (max {max_length} characters)"
            )
        
        # Remove control characters except newlines and tabs
        sanitized = ''.join(char for char in input_string if char.isprintable() or char in '\n\t')
        
        return sanitized.strip()
    
    def _sanitize_command(self, command: str) -> str:
        """
        Sanitize command input.
        
        Args:
            command: Command to sanitize
            
        Returns:
            Sanitized command
        """
        sanitized = self._sanitize_string(command, 100)
        
        # Check for dangerous patterns
        dangerous_patterns = [
            'rm -rf', 'format', 'mkfs', 'dd', '>', '>>', '|', '&', ';', '`', '$(',
            'eval', 'exec', 'system', 'subprocess', 'os.system'
        ]
        
        for pattern in dangerous_patterns:
            if pattern in sanitized.lower():
                raise SecurityError(
                    "dangerous_command_pattern",
                    f"Dangerous pattern detected: {pattern}",
                    severity="CRITICAL"
                )
        
        return sanitized
    
    def validate_command(self, command: str, *args) -> Tuple[str, List[str]]:
        """
        Validate and sanitize command and arguments.
        
        Args:
            command: Command to validate
            *args: Command arguments
            
        Returns:
            Tuple of (sanitized_command, sanitized_args)
            
        Raises:
            SecurityError: When command is not allowed or dangerous
        """
        # Guard clause 1: Validate command
        if not command:
            raise ValidationError(
                "command",
                command,
                "missing_command",
                "Command is required",
                context={"operation": "command_validation"}
            )
        
        # Guard clause 2: Sanitize command
        sanitized_command = self._sanitize_command(command)
        
        # Guard clause 3: Check if command is in whitelist
        if self.allowed_commands and sanitized_command not in self.allowed_commands:
            raise SecurityError(
                "command_not_allowed",
                f"Command '{sanitized_command}' is not in allowed list",
                severity="HIGH",
                context={"operation": "command_validation", "allowed_commands": self.allowed_commands}
            )
        
        # Guard clause 4: Sanitize arguments
        sanitized_args = []
        for arg in args:
            if isinstance(arg, str):
                sanitized_arg = self._sanitize_string(arg)
                sanitized_args.append(sanitized_arg)
            else:
                sanitized_args.append(str(arg))
        
        # Guard clause 5: Check for dangerous patterns in arguments
        dangerous_patterns = [
            'rm -rf', 'format', 'mkfs', 'dd', '>', '>>', '|', '&', ';', '`', '$(',
            'eval', 'exec', 'system', 'subprocess', 'os.system'
        ]
        
        full_command = f"{sanitized_command} {' '.join(sanitized_args)}"
        for pattern in dangerous_patterns:
            if pattern in full_command.lower():
                raise SecurityError(
                    "dangerous_command_pattern",
                    f"Dangerous pattern detected: {pattern}",
                    severity="CRITICAL",
                    context={"operation": "command_validation", "command": full_command}
                )
        
        return sanitized_command, sanitized_args
    
    def execute_command(self, command: str, *args, **kwargs) -> Dict[str, Any]:
        """
        Execute command safely with subprocess.
        
        Args:
            command: Command to execute
            *args: Command arguments
            **kwargs: Additional options (input_data, capture_output, etc.)
            
        Returns:
            Command execution result dictionary
        """
        # Guard clause 1: Validate command
        sanitized_command, sanitized_args = self.validate_command(command, *args)
        
        # Guard clause 2: Prepare execution options
        input_data = kwargs.get('input_data')
        capture_output = kwargs.get('capture_output', True)
        check_returncode = kwargs.get('check_returncode', False)
        
        # Happy path: Execute command
        start_time = time.time()
        
        try:
            # Create command list
            cmd_list = [sanitized_command] + sanitized_args
            
            # Execute with subprocess
            result = subprocess.run(
                cmd_list,
                input=input_data.encode() if input_data else None,
                capture_output=capture_output,
                text=True,
                timeout=self.timeout,
                cwd=self.working_directory,
                env=self.environment,
                shell=False,  # Never use shell=True
                preexec_fn=os.setsid if os.name != 'nt' else None  # Create new process group
            )
            
            execution_time = time.time() - start_time
            
            # Prepare result
            execution_result = {
                'success': result.returncode == 0,
                'returncode': result.returncode,
                'stdout': self._sanitize_output(result.stdout) if result.stdout else '',
                'stderr': self._sanitize_output(result.stderr) if result.stderr else '',
                'execution_time': execution_time,
                'command': ' '.join(cmd_list),
                'timeout': False
            }
            
            # Check return code if requested
            if check_returncode and result.returncode != 0:
                raise subprocess.CalledProcessError(
                    result.returncode, cmd_list, result.stdout, result.stderr
                )
            
            logger.info("Command executed successfully", extra={
                "command": ' '.join(cmd_list),
                "returncode": result.returncode,
                "execution_time": execution_time
            })
            
            return execution_result
            
        except subprocess.TimeoutExpired:
            # Handle timeout
            execution_time = time.time() - start_time
            
            logger.warning("Command execution timed out", extra={
                "command": ' '.join([sanitized_command] + sanitized_args),
                "timeout": self.timeout,
                "execution_time": execution_time
            })
            
            return {
                'success': False,
                'returncode': None,
                'stdout': '',
                'stderr': f'Command execution timed out after {self.timeout} seconds',
                'execution_time': execution_time,
                'command': ' '.join([sanitized_command] + sanitized_args),
                'timeout': True
            }
            
        except subprocess.CalledProcessError as e:
            # Handle command execution error
            execution_time = time.time() - start_time
            
            logger.error("Command execution failed", extra={
                "command": ' '.join([sanitized_command] + sanitized_args),
                "returncode": e.returncode,
                "execution_time": execution_time,
                "error": str(e)
            })
            
            return {
                'success': False,
                'returncode': e.returncode,
                'stdout': self._sanitize_output(e.stdout) if e.stdout else '',
                'stderr': self._sanitize_output(e.stderr) if e.stderr else '',
                'execution_time': execution_time,
                'command': ' '.join([sanitized_command] + sanitized_args),
                'timeout': False
            }
            
        except Exception as e:
            # Handle unexpected errors
            execution_time = time.time() - start_time
            
            logger.error("Unexpected error during command execution", extra={
                "command": ' '.join([sanitized_command] + sanitized_args),
                "execution_time": execution_time,
                "error": str(e)
            })
            
            return {
                'success': False,
                'returncode': None,
                'stdout': '',
                'stderr': f'Unexpected error: {str(e)}',
                'execution_time': execution_time,
                'command': ' '.join([sanitized_command] + sanitized_args),
                'timeout': False
            }
    
    async def execute_command_async(self, command: str, *args, **kwargs) -> Dict[str, Any]:
        """
        Execute command asynchronously with asyncio.
        
        Args:
            command: Command to execute
            *args: Command arguments
            **kwargs: Additional options
            
        Returns:
            Command execution result dictionary
        """
        # Guard clause 1: Validate command
        sanitized_command, sanitized_args = self.validate_command(command, *args)
        
        # Guard clause 2: Prepare execution options
        input_data = kwargs.get('input_data')
        capture_output = kwargs.get('capture_output', True)
        check_returncode = kwargs.get('check_returncode', False)
        
        # Happy path: Execute command asynchronously
        start_time = time.time()
        
        try:
            # Create command list
            cmd_list = [sanitized_command] + sanitized_args
            
            # Execute with asyncio
            process = await asyncio.create_subprocess_exec(
                *cmd_list,
                stdin=asyncio.subprocess.PIPE if input_data else None,
                stdout=asyncio.subprocess.PIPE if capture_output else None,
                stderr=asyncio.subprocess.PIPE if capture_output else None,
                cwd=self.working_directory,
                env=self.environment,
                preexec_fn=os.setsid if os.name != 'nt' else None
            )
            
            # Send input data if provided
            if input_data:
                stdin_data = input_data.encode()
            else:
                stdin_data = None
            
            # Execute with timeout
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(stdin_data),
                    timeout=self.timeout
                )
            except asyncio.TimeoutExpired:
                # Handle timeout
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=5)
                except asyncio.TimeoutExpired:
                    process.kill()
                    await process.wait()
                
                execution_time = time.time() - start_time
                
                logger.warning("Async command execution timed out", extra={
                    "command": ' '.join(cmd_list),
                    "timeout": self.timeout,
                    "execution_time": execution_time
                })
                
                return {
                    'success': False,
                    'returncode': None,
                    'stdout': '',
                    'stderr': f'Command execution timed out after {self.timeout} seconds',
                    'execution_time': execution_time,
                    'command': ' '.join(cmd_list),
                    'timeout': True
                }
            
            execution_time = time.time() - start_time
            
            # Prepare result
            execution_result = {
                'success': process.returncode == 0,
                'returncode': process.returncode,
                'stdout': self._sanitize_output(stdout.decode()) if stdout else '',
                'stderr': self._sanitize_output(stderr.decode()) if stderr else '',
                'execution_time': execution_time,
                'command': ' '.join(cmd_list),
                'timeout': False
            }
            
            # Check return code if requested
            if check_returncode and process.returncode != 0:
                raise subprocess.CalledProcessError(
                    process.returncode, cmd_list, stdout, stderr
                )
            
            logger.info("Async command executed successfully", extra={
                "command": ' '.join(cmd_list),
                "returncode": process.returncode,
                "execution_time": execution_time
            })
            
            return execution_result
            
        except Exception as e:
            # Handle unexpected errors
            execution_time = time.time() - start_time
            
            logger.error("Unexpected error during async command execution", extra={
                "command": ' '.join([sanitized_command] + sanitized_args),
                "execution_time": execution_time,
                "error": str(e)
            })
            
            return {
                'success': False,
                'returncode': None,
                'stdout': '',
                'stderr': f'Unexpected error: {str(e)}',
                'execution_time': execution_time,
                'command': ' '.join([sanitized_command] + sanitized_args),
                'timeout': False
            }
    
    def _sanitize_output(self, output: str) -> str:
        """
        Sanitize command output to remove potentially dangerous content.
        
        Args:
            output: Command output to sanitize
            
        Returns:
            Sanitized output
        """
        if not output:
            return ''
        
        # Remove control characters except newlines and tabs
        sanitized = ''.join(char for char in output if char.isprintable() or char in '\n\t')
        
        # Limit output length
        max_length = 10000  # 10KB limit
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length] + '\n... (output truncated)'
        
        return sanitized
    
    def create_temp_file(self, content: str, suffix: str = '.tmp') -> str:
        """
        Create a temporary file with secure content.
        
        Args:
            content: File content
            suffix: File suffix
            
        Returns:
            Path to temporary file
        """
        # Sanitize content
        sanitized_content = self._sanitize_string(content)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False) as f:
            f.write(sanitized_content)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            temp_path = f.name
        
        # Set secure permissions
        os.chmod(temp_path, 0o600)
        
        return temp_path
    
    def cleanup_temp_file(self, file_path: str):
        """
        Clean up temporary file securely.
        
        Args:
            file_path: Path to temporary file
        """
        try:
            if os.path.exists(file_path):
                # Overwrite file content before deletion
                with open(file_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    f.write('0' * 1024)  # Overwrite with zeros
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                
                os.remove(file_path)
                logger.info(f"Temporary file cleaned up: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to cleanup temporary file {file_path}: {e}")
    
    def get_command_info(self, command: str) -> Dict[str, Any]:
        """
        Get information about a command without executing it.
        
        Args:
            command: Command to get info about
            
        Returns:
            Command information dictionary
        """
        # Guard clause 1: Validate command
        sanitized_command, _ = self.validate_command(command)
        
        # Guard clause 2: Check if command exists
        command_path = None
        for path in self.environment.get('PATH', '').split(':'):
            potential_path = os.path.join(path, sanitized_command)
            if os.path.isfile(potential_path) and os.access(potential_path, os.X_OK):
                command_path = potential_path
                break
        
        # Guard clause 3: Check if command is allowed
        is_allowed = sanitized_command in self.allowed_commands if self.allowed_commands else True
        
        return {
            'command': sanitized_command,
            'exists': command_path is not None,
            'path': command_path,
            'allowed': is_allowed,
            'executable': os.access(command_path, os.X_OK) if command_path else False
        }

# Global safe command executor instance
_safe_executor = SafeCommandExecutor()

def get_safe_executor() -> SafeCommandExecutor:
    """Get global safe command executor instance."""
    return _safe_executor

def execute_command(command: str, *args, **kwargs) -> Dict[str, Any]:
    """Execute command safely using global executor."""
    return _safe_executor.execute_command(command, *args, **kwargs)

async def execute_command_async(command: str, *args, **kwargs) -> Dict[str, Any]:
    """Execute command asynchronously using global executor."""
    return await _safe_executor.execute_command_async(command, *args, **kwargs)

def validate_command(command: str, *args) -> Tuple[str, List[str]]:
    """Validate and sanitize command using global executor."""
    return _safe_executor.validate_command(command, *args)

def create_temp_file(content: str, suffix: str = '.tmp') -> str:
    """Create temporary file using global executor."""
    return _safe_executor.create_temp_file(content, suffix)

def cleanup_temp_file(file_path: str):
    """Clean up temporary file using global executor."""
    _safe_executor.cleanup_temp_file(file_path)

def get_command_info(command: str) -> Dict[str, Any]:
    """Get command information using global executor."""
    return _safe_executor.get_command_info(command)

# --- Named Exports ---

__all__ = [
    'SafeCommandExecutor',
    'ValidationError',
    'SecurityError',
    'get_safe_executor',
    'execute_command',
    'execute_command_async',
    'validate_command',
    'create_temp_file',
    'cleanup_temp_file',
    'get_command_info'
] 