from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import re
import shlex
import subprocess
import asyncio
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
import urllib.parse
import html
import json
from fastapi import HTTPException, status
from pydantic import BaseModel, Field, field_validator
from fastapi import APIRouter
    import asyncio
from typing import Any, List, Dict, Optional
"""
Input Sanitization System for Cybersecurity Tools
Prevents shell command injection and other input-based attacks
"""


class SanitizationLevel(Enum):
    """Sanitization levels for different security requirements"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class InputType(Enum):
    """Types of input that need sanitization"""
    SHELL_COMMAND = "shell_command"
    FILE_PATH = "file_path"
    URL = "url"
    SQL_QUERY = "sql_query"
    HTML_CONTENT = "html_content"
    JSON_DATA = "json_data"
    NETWORK_ADDRESS = "network_address"
    USER_INPUT = "user_input"

@dataclass
class SanitizationResult:
    """Result of input sanitization"""
    original: str
    sanitized: str
    is_safe: bool
    warnings: List[str]
    sanitization_level: SanitizationLevel
    input_type: InputType
    changes_made: bool

class InputSanitizer:
    """Comprehensive input sanitization system"""
    
    def __init__(self, sanitization_level: SanitizationLevel = SanitizationLevel.HIGH):
        
    """__init__ function."""
self.sanitization_level = sanitization_level
        self.logger = logging.getLogger(__name__)
        
        # Dangerous patterns for different input types
        self.dangerous_patterns = {
            InputType.SHELL_COMMAND: [
                r'[;&|`$()<>]',  # Shell metacharacters
                r'\b(rm|del|format|mkfs|dd)\b',  # Dangerous commands
                r'(\$\(|`).*?(\$\)|`)',  # Command substitution
                r'(\$\{.*?\})',  # Variable substitution
                r'(\b(exec|eval|system|os\.system|subprocess)\b)',  # Code execution
            ],
            InputType.FILE_PATH: [
                r'\.\./',  # Directory traversal
                r'^/',  # Absolute paths
                r'[<>:"|?*]',  # Invalid filename characters
                r'(\b(proc|sys|dev|etc|var|tmp|root)\b)',  # System directories
            ],
            InputType.URL: [
                r'javascript:',  # JavaScript protocol
                r'data:',  # Data protocol
                r'vbscript:',  # VBScript protocol
                r'file:',  # File protocol
                r'(\%[0-9A-Fa-f]{2}){2,}',  # Double encoding
                r'(\b(script|onload|onerror|onclick)\b)',  # Event handlers
            ],
            InputType.SQL_QUERY: [
                r'\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b',
                r'(\b(OR|AND)\s+\d+\s*=\s*\d+)',  # Boolean injection
                r'(\b(WAITFOR|DELAY|SLEEP)\b)',  # Time-based injection
                r'(\b(INFORMATION_SCHEMA|sys\.databases)\b)',  # Information gathering
            ],
            InputType.HTML_CONTENT: [
                r'<script[^>]*>.*?</script>',  # Script tags
                r'<iframe[^>]*>.*?</iframe>',  # Iframe tags
                r'<object[^>]*>.*?</object>',  # Object tags
                r'<embed[^>]*>.*?</embed>',  # Embed tags
                r'javascript:',  # JavaScript protocol
                r'on\w+\s*=',  # Event handlers
            ],
            InputType.JSON_DATA: [
                r'(\b(function|eval|setTimeout|setInterval)\b)',  # Code execution
                r'(\b(require|import|export)\b)',  # Module loading
                r'(\b(global|window|document)\b)',  # Global objects
            ],
            InputType.NETWORK_ADDRESS: [
                r'[^0-9a-fA-F:.]',  # Invalid characters for IP/hostname
                r'(\b(127\.0\.0\.1|localhost|0\.0\.0\.0)\b)',  # Local addresses
                r'(\b(10\.|172\.(1[6-9]|2[0-9]|3[01])\.|192\.168\.)\b)',  # Private ranges
            ],
            InputType.USER_INPUT: [
                r'[<>"\']',  # HTML/XML special characters
                r'(\b(script|javascript|vbscript)\b)',  # Script keywords
                r'(\b(alert|confirm|prompt)\b)',  # Browser functions
                r'(\b(eval|setTimeout|setInterval)\b)',  # Code execution
            ]
        }
        
        # Whitelist patterns for safe content
        self.whitelist_patterns = {
            InputType.SHELL_COMMAND: [
                r'^[a-zA-Z0-9\s\-_\.\/]+$',  # Basic safe characters
            ],
            InputType.FILE_PATH: [
                r'^[a-zA-Z0-9\s\-_\.\/]+$',  # Safe filename characters
            ],
            InputType.URL: [
                r'^https?://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,}(:[0-9]+)?(/[a-zA-Z0-9\-_\./?=&%]*)?$',
            ],
            InputType.NETWORK_ADDRESS: [
                r'^([0-9]{1,3}\.){3}[0-9]{1,3}$',  # IPv4
                r'^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$',  # IPv6
                r'^[a-zA-Z0-9\-\.]+$',  # Hostname
            ]
        }
    
    def sanitize_input(
        self, 
        input_data: str, 
        input_type: InputType,
        custom_patterns: Optional[List[str]] = None
    ) -> SanitizationResult:
        """Sanitize input based on type and security level"""
        
        original = input_data
        sanitized = input_data
        warnings = []
        changes_made = False
        
        # Check for dangerous patterns
        dangerous_patterns = self.dangerous_patterns.get(input_type, [])
        if custom_patterns:
            dangerous_patterns.extend(custom_patterns)
        
        for pattern in dangerous_patterns:
            if re.search(pattern, input_data, re.IGNORECASE):
                if self.sanitization_level in [SanitizationLevel.HIGH, SanitizationLevel.CRITICAL]:
                    # Remove or replace dangerous content
                    sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE)
                    changes_made = True
                    warnings.append(f"Dangerous pattern removed: {pattern}")
                else:
                    warnings.append(f"Dangerous pattern detected: {pattern}")
        
        # Apply type-specific sanitization
        if input_type == InputType.SHELL_COMMAND:
            sanitized = self._sanitize_shell_command(sanitized)
        elif input_type == InputType.FILE_PATH:
            sanitized = self._sanitize_file_path(sanitized)
        elif input_type == InputType.URL:
            sanitized = self._sanitize_url(sanitized)
        elif input_type == InputType.SQL_QUERY:
            sanitized = self._sanitize_sql_query(sanitized)
        elif input_type == InputType.HTML_CONTENT:
            sanitized = self._sanitize_html_content(sanitized)
        elif input_type == InputType.JSON_DATA:
            sanitized = self._sanitize_json_data(sanitized)
        elif input_type == InputType.NETWORK_ADDRESS:
            sanitized = self._sanitize_network_address(sanitized)
        elif input_type == InputType.USER_INPUT:
            sanitized = self._sanitize_user_input(sanitized)
        
        # Check whitelist patterns
        whitelist_patterns = self.whitelist_patterns.get(input_type, [])
        is_safe = True
        if whitelist_patterns:
            is_safe = any(re.match(pattern, sanitized, re.IGNORECASE) for pattern in whitelist_patterns)
            if not is_safe:
                warnings.append("Input does not match whitelist patterns")
        
        # Log sanitization attempt
        self.logger.info(f"Input sanitized: {input_type.value} - Original: {original[:50]}... - Safe: {is_safe}")
        
        return SanitizationResult(
            original=original,
            sanitized=sanitized,
            is_safe=is_safe,
            warnings=warnings,
            sanitization_level=self.sanitization_level,
            input_type=input_type,
            changes_made=changes_made
        )
    
    def _sanitize_shell_command(self, command: str) -> str:
        """Sanitize shell command input"""
        # Remove dangerous characters
        dangerous_chars = [';', '|', '&', '`', '$', '(', ')', '<', '>']
        for char in dangerous_chars:
            command = command.replace(char, '')
        
        # Remove dangerous commands
        dangerous_commands = [
            'rm', 'del', 'format', 'mkfs', 'dd', 'shutdown', 'reboot',
            'exec', 'eval', 'system', 'os.system', 'subprocess'
        ]
        
        for cmd in dangerous_commands:
            # Replace with safe alternative or remove
            command = re.sub(rf'\b{cmd}\b', f'safe_{cmd}', command, flags=re.IGNORECASE)
        
        # Use shlex for proper command parsing
        try:
            parsed = shlex.split(command)
            return ' '.join(parsed)
        except ValueError:
            return ''
    
    def _sanitize_file_path(self, path: str) -> str:
        """Sanitize file path input"""
        # Remove directory traversal
        path = re.sub(r'\.\./', '', path)
        path = re.sub(r'\.\.\\', '', path)
        
        # Remove absolute paths
        if path.startswith('/') or path.startswith('\\'):
            path = path[1:]
        
        # Remove dangerous characters
        dangerous_chars = ['<', '>', ':', '"', '|', '?', '*']
        for char in dangerous_chars:
            path = path.replace(char, '_')
        
        # Remove system directories
        system_dirs = ['proc', 'sys', 'dev', 'etc', 'var', 'tmp', 'root']
        for dir_name in system_dirs:
            path = re.sub(rf'\b{dir_name}\b', f'safe_{dir_name}', path, flags=re.IGNORECASE)
        
        return path
    
    def _sanitize_url(self, url: str) -> str:
        """Sanitize URL input"""
        # Remove dangerous protocols
        dangerous_protocols = ['javascript:', 'data:', 'vbscript:', 'file:']
        for protocol in dangerous_protocols:
            if url.lower().startswith(protocol):
                return ''
        
        # URL encode special characters
        try:
            parsed = urllib.parse.urlparse(url)
            if parsed.scheme not in ['http', 'https']:
                return ''
            
            # Reconstruct URL with safe components
            safe_url = urllib.parse.urlunparse((
                parsed.scheme,
                parsed.netloc,
                urllib.parse.quote(parsed.path),
                parsed.params,
                urllib.parse.quote(parsed.query),
                parsed.fragment
            ))
            return safe_url
        except Exception:
            return ''
    
    def _sanitize_sql_query(self, query: str) -> str:
        """Sanitize SQL query input"""
        # Remove dangerous SQL keywords
        dangerous_keywords = [
            'SELECT', 'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER',
            'EXEC', 'UNION', 'WAITFOR', 'DELAY', 'SLEEP', 'INFORMATION_SCHEMA'
        ]
        
        for keyword in dangerous_keywords:
            query = re.sub(rf'\b{keyword}\b', f'safe_{keyword}', query, flags=re.IGNORECASE)
        
        # Remove boolean injection patterns
        query = re.sub(r'\b(OR|AND)\s+\d+\s*=\s*\d+', '', query, flags=re.IGNORECASE)
        
        return query
    
    def _sanitize_html_content(self, content: str) -> str:
        """Sanitize HTML content"""
        # Remove script tags and content
        content = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove iframe tags
        content = re.sub(r'<iframe[^>]*>.*?</iframe>', '', content, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove object and embed tags
        content = re.sub(r'<(object|embed)[^>]*>.*?</\1>', '', content, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove event handlers
        content = re.sub(r'on\w+\s*=\s*["\'][^"\']*["\']', '', content, flags=re.IGNORECASE)
        
        # HTML escape remaining content
        content = html.escape(content)
        
        return content
    
    def _sanitize_json_data(self, data: str) -> str:
        """Sanitize JSON data"""
        try:
            # Parse JSON to validate structure
            parsed = json.loads(data)
            
            # Remove dangerous functions
            dangerous_functions = ['function', 'eval', 'setTimeout', 'setInterval']
            for func in dangerous_functions:
                if isinstance(parsed, dict):
                    parsed = self._remove_dangerous_from_dict(parsed, func)
                elif isinstance(parsed, list):
                    parsed = self._remove_dangerous_from_list(parsed, func)
            
            return json.dumps(parsed)
        except json.JSONDecodeError:
            return '{}'
    
    def _remove_dangerous_from_dict(self, data: Dict, dangerous_func: str) -> Dict:
        """Remove dangerous functions from dictionary"""
        safe_data = {}
        for key, value in data.items():
            if isinstance(value, dict):
                safe_data[key] = self._remove_dangerous_from_dict(value, dangerous_func)
            elif isinstance(value, list):
                safe_data[key] = self._remove_dangerous_from_list(value, dangerous_func)
            elif isinstance(value, str) and dangerous_func not in value.lower():
                safe_data[key] = value
            elif not isinstance(value, str):
                safe_data[key] = value
        return safe_data
    
    def _remove_dangerous_from_list(self, data: List, dangerous_func: str) -> List:
        """Remove dangerous functions from list"""
        safe_data = []
        for item in data:
            if isinstance(item, dict):
                safe_data.append(self._remove_dangerous_from_dict(item, dangerous_func))
            elif isinstance(item, list):
                safe_data.append(self._remove_dangerous_from_list(item, dangerous_func))
            elif isinstance(item, str) and dangerous_func not in item.lower():
                safe_data.append(item)
            elif not isinstance(item, str):
                safe_data.append(item)
        return safe_data
    
    def _sanitize_network_address(self, address: str) -> str:
        """Sanitize network address input"""
        # Remove invalid characters
        address = re.sub(r'[^0-9a-fA-F:.]', '', address)
        
        # Check for local addresses
        local_patterns = [
            r'^127\.0\.0\.1$',
            r'^localhost$',
            r'^0\.0\.0\.0$',
            r'^10\.',
            r'^172\.(1[6-9]|2[0-9]|3[01])\.',
            r'^192\.168\.'
        ]
        
        for pattern in local_patterns:
            if re.match(pattern, address, re.IGNORECASE):
                return ''  # Block local addresses
        
        return address
    
    def _sanitize_user_input(self, input_text: str) -> str:
        """Sanitize general user input"""
        # HTML escape
        input_text = html.escape(input_text)
        
        # Remove script keywords
        script_keywords = ['script', 'javascript', 'vbscript']
        for keyword in script_keywords:
            input_text = re.sub(rf'\b{keyword}\b', f'safe_{keyword}', input_text, flags=re.IGNORECASE)
        
        # Remove browser functions
        browser_functions = ['alert', 'confirm', 'prompt', 'eval', 'setTimeout', 'setInterval']
        for func in browser_functions:
            input_text = re.sub(rf'\b{func}\b', f'safe_{func}', input_text, flags=re.IGNORECASE)
        
        return input_text

class SecureCommandExecutor:
    """Secure command execution with input sanitization"""
    
    def __init__(self, sanitizer: InputSanitizer):
        
    """__init__ function."""
self.sanitizer = sanitizer
        self.logger = logging.getLogger(__name__)
        self.allowed_commands = {
            'ping': ['ping', '-c', '4'],
            'nslookup': ['nslookup'],
            'traceroute': ['traceroute'],
            'whois': ['whois'],
            'dig': ['dig'],
            'netstat': ['netstat', '-an'],
            'ps': ['ps', 'aux'],
            'ls': ['ls', '-la'],
            'cat': ['cat'],
            'grep': ['grep'],
            'head': ['head'],
            'tail': ['tail'],
            'wc': ['wc'],
            'sort': ['sort'],
            'uniq': ['uniq']
        }
    
    async def execute_command(
        self, 
        command: str, 
        args: List[str] = None,
        timeout: int = 30
    ) -> Dict[str, Any]:
        """Execute command with sanitization"""
        
        # Sanitize command
        sanitized_command = self.sanitizer.sanitize_input(command, InputType.SHELL_COMMAND)
        if not sanitized_command.is_safe:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsafe command detected: {sanitized_command.warnings}"
            )
        
        # Check if command is allowed
        if sanitized_command.sanitized not in self.allowed_commands:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Command not allowed: {sanitized_command.sanitized}"
            )
        
        # Build command array
        cmd_array = self.allowed_commands[sanitized_command.sanitized].copy()
        
        # Sanitize and add arguments
        if args:
            for arg in args:
                sanitized_arg = self.sanitizer.sanitize_input(arg, InputType.USER_INPUT)
                if sanitized_arg.is_safe:
                    cmd_array.append(sanitized_arg.sanitized)
                else:
                    self.logger.warning(f"Unsafe argument removed: {arg}")
        
        # Execute command
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd_array,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), 
                timeout=timeout
            )
            
            return {
                "command": ' '.join(cmd_array),
                "return_code": process.returncode,
                "stdout": stdout.decode('utf-8', errors='ignore'),
                "stderr": stderr.decode('utf-8', errors='ignore'),
                "execution_time": timeout,
                "sanitized": True
            }
            
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=status.HTTP_408_REQUEST_TIMEOUT,
                detail="Command execution timed out"
            )
        except Exception as e:
            self.logger.error(f"Command execution failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Command execution failed"
            )

# Pydantic models for API
class SanitizationRequest(BaseModel):
    input_data: str = Field(..., min_length=1, description="Input data to sanitize")
    input_type: InputType = Field(..., description="Type of input to sanitize")
    sanitization_level: SanitizationLevel = Field(
        default=SanitizationLevel.HIGH, 
        description="Level of sanitization to apply"
    )
    custom_patterns: Optional[List[str]] = Field(
        default=None, 
        description="Custom dangerous patterns to check"
    )

class SanitizationResponse(BaseModel):
    original: str
    sanitized: str
    is_safe: bool
    warnings: List[str]
    sanitization_level: SanitizationLevel
    input_type: InputType
    changes_made: bool

class CommandExecutionRequest(BaseModel):
    command: str = Field(..., min_length=1, description="Command to execute")
    arguments: Optional[List[str]] = Field(
        default=None, 
        description="Command arguments"
    )
    timeout: int = Field(
        default=30, 
        ge=1, 
        le=300, 
        description="Command timeout in seconds"
    )

class CommandExecutionResponse(BaseModel):
    command: str
    return_code: int
    stdout: str
    stderr: str
    execution_time: int
    sanitized: bool

# FastAPI router

router = APIRouter(prefix="/sanitization", tags=["Input Sanitization"])

# Initialize sanitizer and command executor
sanitizer = InputSanitizer()
command_executor = SecureCommandExecutor(sanitizer)

@router.post("/sanitize", response_model=SanitizationResponse)
async def sanitize_input(request: SanitizationRequest) -> SanitizationResponse:
    """Sanitize input data"""
    
    # Create sanitizer with requested level
    level_sanitizer = InputSanitizer(request.sanitization_level)
    
    # Sanitize input
    result = level_sanitizer.sanitize_input(
        request.input_data,
        request.input_type,
        request.custom_patterns
    )
    
    return SanitizationResponse(
        original=result.original,
        sanitized=result.sanitized,
        is_safe=result.is_safe,
        warnings=result.warnings,
        sanitization_level=result.sanitization_level,
        input_type=result.input_type,
        changes_made=result.changes_made
    )

@router.post("/execute-command", response_model=CommandExecutionResponse)
async def execute_secure_command(request: CommandExecutionRequest) -> CommandExecutionResponse:
    """Execute command with sanitization"""
    
    result = await command_executor.execute_command(
        request.command,
        request.arguments,
        request.timeout
    )
    
    return CommandExecutionResponse(**result)

@router.get("/allowed-commands")
async def get_allowed_commands() -> Dict[str, List[str]]:
    """Get list of allowed commands"""
    return command_executor.allowed_commands

@router.get("/dangerous-patterns")
async def get_dangerous_patterns() -> Dict[str, List[str]]:
    """Get dangerous patterns for each input type"""
    return sanitizer.dangerous_patterns

# Demo function
async def demo_input_sanitization():
    """Demonstrate input sanitization features"""
    print("=== Input Sanitization Demo ===\n")
    
    # Test shell command sanitization
    print("1. Testing Shell Command Sanitization...")
    dangerous_commands = [
        "ls -la; rm -rf /",
        "ping 192.168.1.1 | grep 'bytes'",
        "echo 'hello' && rm file.txt",
        "cat /etc/passwd",
        "ping 192.168.1.1"  # Safe command
    ]
    
    for cmd in dangerous_commands:
        result = sanitizer.sanitize_input(cmd, InputType.SHELL_COMMAND)
        print(f"   Original: {cmd}")
        print(f"   Sanitized: {result.sanitized}")
        print(f"   Safe: {result.is_safe}")
        print(f"   Warnings: {result.warnings}")
        print()
    
    # Test file path sanitization
    print("2. Testing File Path Sanitization...")
    dangerous_paths = [
        "../../../etc/passwd",
        "/root/.ssh/id_rsa",
        "file<script>.txt",
        "normal_file.txt"
    ]
    
    for path in dangerous_paths:
        result = sanitizer.sanitize_input(path, InputType.FILE_PATH)
        print(f"   Original: {path}")
        print(f"   Sanitized: {result.sanitized}")
        print(f"   Safe: {result.is_safe}")
        print()
    
    # Test URL sanitization
    print("3. Testing URL Sanitization...")
    dangerous_urls = [
        "javascript:alert('xss')",
        "data:text/html,<script>alert('xss')</script>",
        "https://example.com/path?param=value",
        "file:///etc/passwd"
    ]
    
    for url in dangerous_urls:
        result = sanitizer.sanitize_input(url, InputType.URL)
        print(f"   Original: {url}")
        print(f"   Sanitized: {result.sanitized}")
        print(f"   Safe: {result.is_safe}")
        print()
    
    # Test secure command execution
    print("4. Testing Secure Command Execution...")
    try:
        result = await command_executor.execute_command("ping", ["-c", "1", "127.0.0.1"])
        print(f"   Command: {result['command']}")
        print(f"   Return Code: {result['return_code']}")
        print(f"   Output Length: {len(result['stdout'])} chars")
        print(f"   Sanitized: {result['sanitized']}")
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n=== Input Sanitization Demo Completed! ===")

match __name__:
    case "__main__":
    asyncio.run(demo_input_sanitization()) 