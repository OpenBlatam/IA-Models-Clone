from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

from fastapi import FastAPI, APIRouter, HTTPException, status, Depends
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import asyncio
import re
import time
from datetime import datetime

from input_sanitization import (
    import uvicorn
from typing import Any, List, Dict, Optional
import logging
# Import sanitization components
    InputSanitizer, SecureCommandExecutor, SanitizationLevel, InputType,
    SanitizationRequest, SanitizationResponse, CommandExecutionRequest, CommandExecutionResponse
)

# Data Models for Demo
class SanitizationTestRequest(BaseModel):
    test_type: str = Field(..., regex="^(shell|file|url|sql|html|json|network|user)$")
    test_data: List[str] = Field(..., description="Test data for sanitization")
    sanitization_level: SanitizationLevel = Field(default=SanitizationLevel.HIGH)

class SanitizationTestResponse(BaseModel):
    test_type: str
    results: List[Dict[str, Any]]
    summary: Dict[str, Any]
    timestamp: datetime

class SecurityVulnerabilityTest(BaseModel):
    vulnerability_type: str = Field(..., regex="^(injection|xss|path_traversal|command_injection)$")
    payload: str = Field(..., description="Malicious payload to test")
    expected_behavior: str = Field(..., description="Expected sanitization behavior")

class SecurityVulnerabilityResponse(BaseModel):
    vulnerability_type: str
    original_payload: str
    sanitized_payload: str
    is_blocked: bool
    sanitization_method: str
    security_level: str
    timestamp: datetime

# Demo Router
router = APIRouter(prefix="/sanitization-demo", tags=["Sanitization Demo"])

# Initialize sanitization components
sanitizer = InputSanitizer()
command_executor = SecureCommandExecutor(sanitizer)

@router.post("/test-sanitization", response_model=SanitizationTestResponse)
async def test_sanitization_features(
    request: SanitizationTestRequest
) -> SanitizationTestResponse:
    """Test various sanitization features"""
    
    results = []
    input_type_map = {
        "shell": InputType.SHELL_COMMAND,
        "file": InputType.FILE_PATH,
        "url": InputType.URL,
        "sql": InputType.SQL_QUERY,
        "html": InputType.HTML_CONTENT,
        "json": InputType.JSON_DATA,
        "network": InputType.NETWORK_ADDRESS,
        "user": InputType.USER_INPUT
    }
    
    input_type = input_type_map.get(request.test_type, InputType.USER_INPUT)
    
    for test_data in request.test_data:
        result = sanitizer.sanitize_input(
            test_data, 
            input_type, 
            sanitization_level=request.sanitization_level
        )
        
        results.append({
            "original": result.original,
            "sanitized": result.sanitized,
            "is_safe": result.is_safe,
            "warnings": result.warnings,
            "changes_made": result.changes_made,
            "sanitization_level": result.sanitization_level.value
        })
    
    # Calculate summary
    total_tests = len(results)
    safe_count = sum(1 for r in results if r["is_safe"])
    changed_count = sum(1 for r in results if r["changes_made"])
    warning_count = sum(len(r["warnings"]) for r in results)
    
    summary = {
        "total_tests": total_tests,
        "safe_inputs": safe_count,
        "unsafe_inputs": total_tests - safe_count,
        "inputs_modified": changed_count,
        "total_warnings": warning_count,
        "success_rate": (safe_count / total_tests) * 100 if total_tests > 0 else 0
    }
    
    return SanitizationTestResponse(
        test_type=request.test_type,
        results=results,
        summary=summary,
        timestamp=datetime.utcnow()
    )

@router.post("/test-vulnerability", response_model=SecurityVulnerabilityResponse)
async def test_security_vulnerability(
    request: SecurityVulnerabilityTest
) -> SecurityVulnerabilityResponse:
    """Test specific security vulnerabilities"""
    
    # Map vulnerability types to input types
    vulnerability_map = {
        "injection": InputType.SQL_QUERY,
        "xss": InputType.HTML_CONTENT,
        "path_traversal": InputType.FILE_PATH,
        "command_injection": InputType.SHELL_COMMAND
    }
    
    input_type = vulnerability_map.get(request.vulnerability_type, InputType.USER_INPUT)
    
    # Sanitize payload
    result = sanitizer.sanitize_input(
        request.payload, 
        input_type, 
        sanitization_level=SanitizationLevel.CRITICAL
    )
    
    # Determine if payload was blocked
    is_blocked = result.original != result.sanitized or not result.is_safe
    
    # Determine sanitization method
    sanitization_method = "removal" if result.changes_made else "validation"
    
    # Determine security level
    security_level = "high" if result.is_safe else "low"
    
    return SecurityVulnerabilityResponse(
        vulnerability_type=request.vulnerability_type,
        original_payload=request.payload,
        sanitized_payload=result.sanitized,
        is_blocked=is_blocked,
        sanitization_method=sanitization_method,
        security_level=security_level,
        timestamp=datetime.utcnow()
    )

@router.post("/execute-secure-command", response_model=CommandExecutionResponse)
async def execute_secure_command_demo(
    request: CommandExecutionRequest
) -> CommandExecutionResponse:
    """Execute command with sanitization demo"""
    
    try:
        result = await command_executor.execute_command(
            request.command,
            request.arguments,
            request.timeout
        )
        
        return CommandExecutionResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Command execution failed: {str(e)}"
        )

@router.get("/sanitization-stats")
async def get_sanitization_statistics() -> Dict[str, Any]:
    """Get sanitization statistics and patterns"""
    
    return {
        "dangerous_patterns": {
            input_type.value: len(patterns) 
            for input_type, patterns in sanitizer.dangerous_patterns.items()
        },
        "whitelist_patterns": {
            input_type.value: len(patterns) 
            for input_type, patterns in sanitizer.whitelist_patterns.items()
        },
        "allowed_commands": len(command_executor.allowed_commands),
        "sanitization_levels": [level.value for level in SanitizationLevel],
        "input_types": [input_type.value for input_type in InputType]
    }

@router.get("/test-patterns")
async def get_test_patterns() -> Dict[str, List[str]]:
    """Get test patterns for different vulnerability types"""
    
    return {
        "shell_injection": [
            "ls -la; rm -rf /",
            "ping 192.168.1.1 | grep 'bytes'",
            "echo 'hello' && rm file.txt",
            "cat /etc/passwd",
            "ping 192.168.1.1"  # Safe
        ],
        "path_traversal": [
            "../../../etc/passwd",
            "/root/.ssh/id_rsa",
            "file<script>.txt",
            "normal_file.txt"  # Safe
        ],
        "sql_injection": [
            "'; DROP TABLE users; --",
            "1 OR 1=1",
            "1' UNION SELECT * FROM passwords --",
            "SELECT name FROM users"  # Safe
        ],
        "xss": [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>",
            "Hello World"  # Safe
        ],
        "url_injection": [
            "javascript:alert('xss')",
            "data:text/html,<script>alert('xss')</script>",
            "https://example.com/path?param=value",  # Safe
            "file:///etc/passwd"
        ]
    }

# Demo function
async def run_sanitization_demo():
    """Run comprehensive sanitization demo"""
    print("=== Input Sanitization Demo ===\n")
    
    # Test shell command sanitization
    print("1. Testing Shell Command Sanitization...")
    shell_commands = [
        "ls -la; rm -rf /",
        "ping 192.168.1.1 | grep 'bytes'",
        "echo 'hello' && rm file.txt",
        "cat /etc/passwd",
        "ping 192.168.1.1"  # Safe command
    ]
    
    for cmd in shell_commands:
        result = sanitizer.sanitize_input(cmd, InputType.SHELL_COMMAND)
        print(f"   Original: {cmd}")
        print(f"   Sanitized: {result.sanitized}")
        print(f"   Safe: {result.is_safe}")
        print(f"   Warnings: {result.warnings}")
        print()
    
    # Test file path sanitization
    print("2. Testing File Path Sanitization...")
    file_paths = [
        "../../../etc/passwd",
        "/root/.ssh/id_rsa",
        "file<script>.txt",
        "normal_file.txt"
    ]
    
    for path in file_paths:
        result = sanitizer.sanitize_input(path, InputType.FILE_PATH)
        print(f"   Original: {path}")
        print(f"   Sanitized: {result.sanitized}")
        print(f"   Safe: {result.is_safe}")
        print()
    
    # Test SQL injection sanitization
    print("3. Testing SQL Injection Sanitization...")
    sql_queries = [
        "'; DROP TABLE users; --",
        "1 OR 1=1",
        "1' UNION SELECT * FROM passwords --",
        "SELECT name FROM users"  # Safe
    ]
    
    for query in sql_queries:
        result = sanitizer.sanitize_input(query, InputType.SQL_QUERY)
        print(f"   Original: {query}")
        print(f"   Sanitized: {result.sanitized}")
        print(f"   Safe: {result.is_safe}")
        print()
    
    # Test XSS sanitization
    print("4. Testing XSS Sanitization...")
    xss_payloads = [
        "<script>alert('xss')</script>",
        "javascript:alert('xss')",
        "<img src=x onerror=alert('xss')>",
        "Hello World"  # Safe
    ]
    
    for payload in xss_payloads:
        result = sanitizer.sanitize_input(payload, InputType.HTML_CONTENT)
        print(f"   Original: {payload}")
        print(f"   Sanitized: {result.sanitized}")
        print(f"   Safe: {result.is_safe}")
        print()
    
    # Test URL sanitization
    print("5. Testing URL Sanitization...")
    urls = [
        "javascript:alert('xss')",
        "data:text/html,<script>alert('xss')</script>",
        "https://example.com/path?param=value",  # Safe
        "file:///etc/passwd"
    ]
    
    for url in urls:
        result = sanitizer.sanitize_input(url, InputType.URL)
        print(f"   Original: {url}")
        print(f"   Sanitized: {result.sanitized}")
        print(f"   Safe: {result.is_safe}")
        print()
    
    # Test secure command execution
    print("6. Testing Secure Command Execution...")
    try:
        result = await command_executor.execute_command("ping", ["-c", "1", "127.0.0.1"])
        print(f"   Command: {result['command']}")
        print(f"   Return Code: {result['return_code']}")
        print(f"   Output Length: {len(result['stdout'])} chars")
        print(f"   Sanitized: {result['sanitized']}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test different sanitization levels
    print("\n7. Testing Different Sanitization Levels...")
    test_input = "ls -la; rm -rf /"
    
    for level in SanitizationLevel:
        level_sanitizer = InputSanitizer(level)
        result = level_sanitizer.sanitize_input(test_input, InputType.SHELL_COMMAND)
        print(f"   Level: {level.value}")
        print(f"   Sanitized: {result.sanitized}")
        print(f"   Safe: {result.is_safe}")
        print(f"   Warnings: {len(result.warnings)}")
        print()
    
    print("=== Input Sanitization Demo Completed! ===")

# FastAPI app
app = FastAPI(
    title="Input Sanitization Demo",
    description="Demonstration of comprehensive input sanitization for cybersecurity tools",
    version="1.0.0"
)

# Include demo router
app.include_router(router)

if __name__ == "__main__":
    print("Input Sanitization Demo")
    print("Access API at: http://localhost:8000")
    print("API Documentation at: http://localhost:8000/docs")
    
    # Run demo
    asyncio.run(run_sanitization_demo())
    
    # Start server
    uvicorn.run(app, host="0.0.0.0", port=8000) 