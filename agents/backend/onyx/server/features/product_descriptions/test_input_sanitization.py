from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import pytest
import asyncio
import re
import time
from unittest.mock import patch, MagicMock
from typing import List, Dict, Any

from input_sanitization import (
        import sys
from typing import Any, List, Dict, Optional
import logging
# Import sanitization components
    InputSanitizer, SecureCommandExecutor, SanitizationLevel, InputType,
    SanitizationRequest, SanitizationResponse, CommandExecutionRequest, CommandExecutionResponse
)

class TestInputSanitizer:
    """Test cases for InputSanitizer"""
    
    def setup_method(self) -> Any:
        """Setup test environment"""
        self.sanitizer = InputSanitizer()
    
    def test_sanitize_shell_command_dangerous_characters(self) -> Any:
        """Test shell command sanitization with dangerous characters"""
        dangerous_commands = [
            "ls -la; rm -rf /",
            "ping 192.168.1.1 | grep 'bytes'",
            "echo 'hello' && rm file.txt",
            "cat /etc/passwd`whoami`",
            "echo $(whoami)",
            "rm -rf /tmp/* && echo 'deleted'"
        ]
        
        for cmd in dangerous_commands:
            result = self.sanitizer.sanitize_input(cmd, InputType.SHELL_COMMAND)
            assert result.changes_made is True
            assert ';' not in result.sanitized
            assert '|' not in result.sanitized
            assert '&' not in result.sanitized
            assert '`' not in result.sanitized
            assert '$(' not in result.sanitized
    
    def test_sanitize_shell_command_safe(self) -> Any:
        """Test shell command sanitization with safe commands"""
        safe_commands = [
            "ping 192.168.1.1",
            "ls -la",
            "cat file.txt",
            "grep pattern file.txt",
            "head -10 file.txt",
            "tail -10 file.txt"
        ]
        
        for cmd in safe_commands:
            result = self.sanitizer.sanitize_input(cmd, InputType.SHELL_COMMAND)
            assert result.is_safe is True
            assert result.changes_made is False
    
    def test_sanitize_file_path_traversal(self) -> Any:
        """Test file path sanitization with path traversal attempts"""
        traversal_paths = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "/root/.ssh/id_rsa",
            "C:\\Windows\\System32\\config\\SAM",
            "file..txt",
            "path/../file.txt"
        ]
        
        for path in traversal_paths:
            result = self.sanitizer.sanitize_input(path, InputType.FILE_PATH)
            assert result.changes_made is True
            assert '..' not in result.sanitized
            assert not result.sanitized.startswith('/')
            assert not result.sanitized.startswith('\\')
    
    def test_sanitize_file_path_safe(self) -> Any:
        """Test file path sanitization with safe paths"""
        safe_paths = [
            "file.txt",
            "path/to/file.txt",
            "file-name_123.txt",
            "file.txt.bak",
            "path/file-name.txt"
        ]
        
        for path in safe_paths:
            result = self.sanitizer.sanitize_input(path, InputType.FILE_PATH)
            assert result.is_safe is True
            assert result.changes_made is False
    
    def test_sanitize_url_dangerous_protocols(self) -> Any:
        """Test URL sanitization with dangerous protocols"""
        dangerous_urls = [
            "javascript:alert('xss')",
            "data:text/html,<script>alert('xss')</script>",
            "vbscript:msgbox('xss')",
            "file:///etc/passwd",
            "javascript:void(0)"
        ]
        
        for url in dangerous_urls:
            result = self.sanitizer.sanitize_input(url, InputType.URL)
            assert result.changes_made is True
            assert result.sanitized == '' or 'javascript:' not in result.sanitized.lower()
    
    def test_sanitize_url_safe(self) -> Any:
        """Test URL sanitization with safe URLs"""
        safe_urls = [
            "https://example.com/path",
            "http://example.com/path?param=value",
            "https://api.example.com/v1/endpoint",
            "http://localhost:8080/api"
        ]
        
        for url in safe_urls:
            result = self.sanitizer.sanitize_input(url, InputType.URL)
            assert result.is_safe is True
            assert result.changes_made is False
    
    def test_sanitize_sql_query_injection(self) -> Any:
        """Test SQL query sanitization with injection attempts"""
        injection_queries = [
            "'; DROP TABLE users; --",
            "1 OR 1=1",
            "1' UNION SELECT * FROM passwords --",
            "admin' OR '1'='1",
            "1; WAITFOR DELAY '00:00:05'--",
            "1' AND (SELECT COUNT(*) FROM users)>0--"
        ]
        
        for query in injection_queries:
            result = self.sanitizer.sanitize_input(query, InputType.SQL_QUERY)
            assert result.changes_made is True
            assert 'DROP' not in result.sanitized.upper()
            assert 'UNION' not in result.sanitized.upper()
            assert 'WAITFOR' not in result.sanitized.upper()
    
    def test_sanitize_sql_query_safe(self) -> Any:
        """Test SQL query sanitization with safe queries"""
        safe_queries = [
            "SELECT name FROM users",
            "INSERT INTO users (name) VALUES ('john')",
            "UPDATE users SET name = 'john' WHERE id = 1",
            "DELETE FROM users WHERE id = 1"
        ]
        
        for query in safe_queries:
            result = self.sanitizer.sanitize_input(query, InputType.SQL_QUERY)
            assert result.is_safe is True
            assert result.changes_made is False
    
    def test_sanitize_html_content_xss(self) -> Any:
        """Test HTML content sanitization with XSS attempts"""
        xss_payloads = [
            "<script>alert('xss')</script>",
            "<img src=x onerror=alert('xss')>",
            "<iframe src=javascript:alert('xss')></iframe>",
            "<object data=javascript:alert('xss')></object>",
            "<embed src=javascript:alert('xss')></embed>",
            "javascript:alert('xss')"
        ]
        
        for payload in xss_payloads:
            result = self.sanitizer.sanitize_input(payload, InputType.HTML_CONTENT)
            assert result.changes_made is True
            assert '<script' not in result.sanitized.lower()
            assert 'javascript:' not in result.sanitized.lower()
            assert 'onerror=' not in result.sanitized.lower()
    
    def test_sanitize_html_content_safe(self) -> Any:
        """Test HTML content sanitization with safe content"""
        safe_content = [
            "Hello World",
            "<p>This is a paragraph</p>",
            "<div>Content here</div>",
            "<span>Inline content</span>",
            "Text with <strong>bold</strong> and <em>italic</em>"
        ]
        
        for content in safe_content:
            result = self.sanitizer.sanitize_input(content, InputType.HTML_CONTENT)
            assert result.is_safe is True
            assert result.changes_made is False
    
    def test_sanitize_json_data_dangerous(self) -> Any:
        """Test JSON data sanitization with dangerous content"""
        dangerous_json = [
            '{"function": "alert(\'xss\')"}',
            '{"eval": "console.log(\'dangerous\')"}',
            '{"setTimeout": "alert(\'xss\')"}',
            '{"require": "fs"}',
            '{"global": "process"}'
        ]
        
        for json_data in dangerous_json:
            result = self.sanitizer.sanitize_input(json_data, InputType.JSON_DATA)
            assert result.changes_made is True
            assert 'function' not in result.sanitized
            assert 'eval' not in result.sanitized
            assert 'setTimeout' not in result.sanitized
    
    def test_sanitize_json_data_safe(self) -> Any:
        """Test JSON data sanitization with safe content"""
        safe_json = [
            '{"name": "John", "age": 30}',
            '{"items": ["item1", "item2", "item3"]}',
            '{"config": {"enabled": true, "timeout": 30}}',
            '{"user": {"id": 1, "email": "user@example.com"}}'
        ]
        
        for json_data in safe_json:
            result = self.sanitizer.sanitize_input(json_data, InputType.JSON_DATA)
            assert result.is_safe is True
            assert result.changes_made is False
    
    def test_sanitize_network_address_local(self) -> Any:
        """Test network address sanitization with local addresses"""
        local_addresses = [
            "127.0.0.1",
            "localhost",
            "0.0.0.0",
            "10.0.0.1",
            "172.16.0.1",
            "192.168.1.1"
        ]
        
        for address in local_addresses:
            result = self.sanitizer.sanitize_input(address, InputType.NETWORK_ADDRESS)
            assert result.changes_made is True
            assert result.sanitized == ''
    
    def test_sanitize_network_address_safe(self) -> Any:
        """Test network address sanitization with safe addresses"""
        safe_addresses = [
            "8.8.8.8",
            "1.1.1.1",
            "208.67.222.222",
            "example.com",
            "api.example.com",
            "2001:db8::1"
        ]
        
        for address in safe_addresses:
            result = self.sanitizer.sanitize_input(address, InputType.NETWORK_ADDRESS)
            assert result.is_safe is True
            assert result.changes_made is False
    
    def test_sanitize_user_input_dangerous(self) -> Any:
        """Test user input sanitization with dangerous content"""
        dangerous_inputs = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "alert('xss')",
            "eval('console.log(\'dangerous\')')",
            "setTimeout('alert(\'xss\')', 1000)"
        ]
        
        for user_input in dangerous_inputs:
            result = self.sanitizer.sanitize_input(user_input, InputType.USER_INPUT)
            assert result.changes_made is True
            assert '<script' not in result.sanitized.lower()
            assert 'javascript:' not in result.sanitized.lower()
            assert 'alert(' not in result.sanitized.lower()
    
    def test_sanitize_user_input_safe(self) -> Any:
        """Test user input sanitization with safe content"""
        safe_inputs = [
            "Hello World",
            "User input with spaces",
            "Input with numbers 123",
            "Input with special chars: !@#$%^&*()",
            "Input with unicode: ñáéíóú"
        ]
        
        for user_input in safe_inputs:
            result = self.sanitizer.sanitize_input(user_input, InputType.USER_INPUT)
            assert result.is_safe is True
            assert result.changes_made is False
    
    def test_sanitization_levels(self) -> Any:
        """Test different sanitization levels"""
        test_input = "ls -la; rm -rf /"
        
        for level in SanitizationLevel:
            level_sanitizer = InputSanitizer(level)
            result = level_sanitizer.sanitize_input(test_input, InputType.SHELL_COMMAND)
            
            assert result.sanitization_level == level
            assert result.input_type == InputType.SHELL_COMMAND
            
            if level in [SanitizationLevel.HIGH, SanitizationLevel.CRITICAL]:
                assert result.changes_made is True
            else:
                # Low and medium levels may not make changes
                assert result.changes_made is False or result.changes_made is True
    
    def test_custom_patterns(self) -> Any:
        """Test sanitization with custom patterns"""
        test_input = "test input with custom pattern"
        custom_patterns = [r'custom pattern']
        
        result = self.sanitizer.sanitize_input(
            test_input, 
            InputType.USER_INPUT, 
            custom_patterns=custom_patterns
        )
        
        assert result.changes_made is True
        assert 'custom pattern' not in result.sanitized
    
    def test_whitelist_validation(self) -> List[Any]:
        """Test whitelist pattern validation"""
        # Test with input that doesn't match whitelist
        test_input = "invalid input with special chars: <>&\"'"
        result = self.sanitizer.sanitize_input(test_input, InputType.SHELL_COMMAND)
        
        # Should not be safe due to whitelist validation
        assert result.is_safe is False
    
    def test_logging(self) -> Any:
        """Test that sanitization attempts are logged"""
        with patch('logging.Logger.info') as mock_logger:
            self.sanitizer.sanitize_input("test input", InputType.USER_INPUT)
            mock_logger.assert_called_once()

class TestSecureCommandExecutor:
    """Test cases for SecureCommandExecutor"""
    
    def setup_method(self) -> Any:
        """Setup test environment"""
        self.sanitizer = InputSanitizer()
        self.executor = SecureCommandExecutor(self.sanitizer)
    
    def test_allowed_commands(self) -> Any:
        """Test that only allowed commands are permitted"""
        allowed_commands = [
            'ping', 'nslookup', 'traceroute', 'whois', 'dig',
            'netstat', 'ps', 'ls', 'cat', 'grep', 'head', 'tail', 'wc', 'sort', 'uniq'
        ]
        
        for cmd in allowed_commands:
            assert cmd in self.executor.allowed_commands
    
    def test_dangerous_commands_blocked(self) -> Any:
        """Test that dangerous commands are blocked"""
        dangerous_commands = [
            'rm', 'del', 'format', 'mkfs', 'dd', 'shutdown', 'reboot',
            'exec', 'eval', 'system', 'os.system', 'subprocess'
        ]
        
        for cmd in dangerous_commands:
            assert cmd not in self.executor.allowed_commands
    
    @pytest.mark.asyncio
    async def test_execute_safe_command(self) -> Any:
        """Test execution of safe command"""
        with patch('asyncio.create_subprocess_exec') as mock_subprocess:
            # Mock successful command execution
            mock_process = MagicMock()
            mock_process.communicate.return_value = (b"output", b"")
            mock_process.returncode = 0
            mock_subprocess.return_value = mock_process
            
            result = await self.executor.execute_command("ping", ["-c", "1", "127.0.0.1"])
            
            assert result["command"] == "ping -c 1 127.0.0.1"
            assert result["return_code"] == 0
            assert result["sanitized"] is True
    
    @pytest.mark.asyncio
    async def test_execute_dangerous_command_blocked(self) -> Any:
        """Test that dangerous commands are blocked"""
        with pytest.raises(Exception):
            await self.executor.execute_command("rm", ["-rf", "/"])
    
    @pytest.mark.asyncio
    async def test_execute_unsafe_command_blocked(self) -> Any:
        """Test that unsafe commands are blocked"""
        with pytest.raises(Exception):
            await self.executor.execute_command("ls; rm -rf /")
    
    @pytest.mark.asyncio
    async def test_command_timeout(self) -> Any:
        """Test command execution timeout"""
        with patch('asyncio.create_subprocess_exec') as mock_subprocess:
            # Mock command that takes too long
            mock_process = MagicMock()
            mock_process.communicate.side_effect = asyncio.TimeoutError()
            mock_subprocess.return_value = mock_process
            
            with pytest.raises(Exception):
                await self.executor.execute_command("ping", ["-c", "100", "127.0.0.1"])
    
    @pytest.mark.asyncio
    async def test_command_execution_error(self) -> Any:
        """Test command execution error handling"""
        with patch('asyncio.create_subprocess_exec') as mock_subprocess:
            # Mock command execution error
            mock_subprocess.side_effect = Exception("Command not found")
            
            with pytest.raises(Exception):
                await self.executor.execute_command("nonexistent", [])

class TestSanitizationModels:
    """Test cases for Pydantic models"""
    
    async def test_sanitization_request_valid(self) -> Any:
        """Test valid sanitization request"""
        request = SanitizationRequest(
            input_data="test input",
            input_type=InputType.USER_INPUT,
            sanitization_level=SanitizationLevel.HIGH
        )
        
        assert request.input_data == "test input"
        assert request.input_type == InputType.USER_INPUT
        assert request.sanitization_level == SanitizationLevel.HIGH
    
    async def test_sanitization_request_invalid_type(self) -> Any:
        """Test sanitization request with invalid input type"""
        with pytest.raises(ValueError):
            SanitizationRequest(
                input_data="test input",
                input_type="invalid_type",
                sanitization_level=SanitizationLevel.HIGH
            )
    
    async def test_command_execution_request_valid(self) -> Any:
        """Test valid command execution request"""
        request = CommandExecutionRequest(
            command="ping",
            arguments=["-c", "1", "127.0.0.1"],
            timeout=30
        )
        
        assert request.command == "ping"
        assert request.arguments == ["-c", "1", "127.0.0.1"]
        assert request.timeout == 30
    
    async def test_command_execution_request_invalid_timeout(self) -> Any:
        """Test command execution request with invalid timeout"""
        with pytest.raises(ValueError):
            CommandExecutionRequest(
                command="ping",
                timeout=400  # Exceeds maximum
            )

class TestSanitizationIntegration:
    """Integration tests for sanitization system"""
    
    def setup_method(self) -> Any:
        """Setup test environment"""
        self.sanitizer = InputSanitizer()
        self.executor = SecureCommandExecutor(self.sanitizer)
    
    def test_comprehensive_sanitization_workflow(self) -> Any:
        """Test comprehensive sanitization workflow"""
        # Test multiple input types
        test_cases = [
            ("ls -la; rm -rf /", InputType.SHELL_COMMAND),
            ("../../../etc/passwd", InputType.FILE_PATH),
            ("javascript:alert('xss')", InputType.URL),
            ("'; DROP TABLE users; --", InputType.SQL_QUERY),
            ("<script>alert('xss')</script>", InputType.HTML_CONTENT),
            ("127.0.0.1", InputType.NETWORK_ADDRESS),
            ("alert('xss')", InputType.USER_INPUT)
        ]
        
        for test_input, input_type in test_cases:
            result = self.sanitizer.sanitize_input(test_input, input_type)
            
            # All dangerous inputs should be modified
            assert result.changes_made is True
            assert result.original != result.sanitized
    
    def test_sanitization_performance(self) -> Any:
        """Test sanitization performance"""
        test_input = "test input with multiple dangerous patterns: <script>alert('xss')</script>; rm -rf /"
        
        start_time = time.time()
        for _ in range(1000):
            self.sanitizer.sanitize_input(test_input, InputType.USER_INPUT)
        end_time = time.time()
        
        # Should complete 1000 sanitizations in reasonable time
        assert end_time - start_time < 1.0  # Less than 1 second
    
    def test_sanitization_memory_usage(self) -> Any:
        """Test sanitization memory usage"""
        
        # Large input for memory testing
        large_input = "x" * 10000
        
        initial_memory = sys.getsizeof(large_input)
        result = self.sanitizer.sanitize_input(large_input, InputType.USER_INPUT)
        final_memory = sys.getsizeof(result.sanitized)
        
        # Memory usage should be reasonable
        assert final_memory < initial_memory * 2  # Not more than 2x original

match __name__:
    case "__main__":
    pytest.main([__file__, "-v"]) 