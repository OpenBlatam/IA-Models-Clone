from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from fastapi import FastAPI, APIRouter, HTTPException, status, Depends, Request, Response
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import asyncio
import hashlib
import secrets
import re
import time
from datetime import datetime, timedelta

from security_guidelines import (
    import uvicorn
from typing import Any, List, Dict, Optional
import logging
# Import security components
    SecureInputValidator, SecurityAuthenticator, SecurityCrypto,
    SecurityLogger, SecurityHeaders, SecurityMiddleware,
    SecureScanRequest, SecureScanResponse, SecurityUtils,
    SecurityConfig, SecurityBestPractices, require_authentication,
    validate_input
)

# Data Models for Demo
class SecurityTestRequest(BaseModel):
    test_type: str = Field(..., regex="^(input_validation|authentication|crypto|logging|headers)$")
    test_data: Dict[str, Any] = Field(..., description="Test data for security validation")

class SecurityTestResponse(BaseModel):
    test_type: str
    success: bool
    results: Dict[str, Any]
    timestamp: datetime
    duration: float

class VulnerabilityScanRequest(BaseModel):
    target: str = Field(..., min_length=1)
    scan_type: str = Field(..., regex="^(sql_injection|xss|csrf|path_traversal)$")
    payload: Optional[str] = None

class VulnerabilityScanResponse(BaseModel):
    target: str
    scan_type: str
    vulnerabilities_found: List[str]
    risk_level: str
    recommendations: List[str]
    scan_timestamp: datetime

# Demo Router
router = APIRouter(prefix="/security-demo", tags=["Security Demo"])

# Initialize security components
validator = SecureInputValidator()
authenticator = SecurityAuthenticator(secrets.token_urlsafe(32))
crypto = SecurityCrypto()
logger = SecurityLogger("security_demo.log")
security_config = SecurityConfig()

@router.post("/test-security", response_model=SecurityTestResponse)
async def test_security_features(
    request: SecurityTestRequest
) -> SecurityTestResponse:
    """Test various security features"""
    start_time = time.time()
    
    results = {}
    
    if request.test_type == "input_validation":
        results = await test_input_validation(request.test_data)
    elif request.test_type == "authentication":
        results = await test_authentication(request.test_data)
    elif request.test_type == "crypto":
        results = await test_cryptography(request.test_data)
    elif request.test_type == "logging":
        results = await test_secure_logging(request.test_data)
    elif request.test_type == "headers":
        results = await test_security_headers(request.test_data)
    
    duration = time.time() - start_time
    
    return SecurityTestResponse(
        test_type=request.test_type,
        success=results.get("success", False),
        results=results,
        timestamp=datetime.utcnow(),
        duration=duration
    )

async def test_input_validation(test_data: Dict[str, Any]) -> Dict[str, Any]:
    """Test input validation features"""
    results = {
        "success": True,
        "tests": {}
    }
    
    # Test IP address validation
    if "ip_addresses" in test_data:
        ip_results = {}
        for ip in test_data["ip_addresses"]:
            ip_results[ip] = validator.validate_ip_address(ip)
        results["tests"]["ip_validation"] = ip_results
    
    # Test hostname validation
    if "hostnames" in test_data:
        hostname_results = {}
        for hostname in test_data["hostnames"]:
            hostname_results[hostname] = validator.validate_hostname(hostname)
        results["tests"]["hostname_validation"] = hostname_results
    
    # Test port validation
    if "ports" in test_data:
        port_results = {}
        for port in test_data["ports"]:
            port_results[port] = validator.validate_port_range(port)
        results["tests"]["port_validation"] = port_results
    
    # Test command sanitization
    if "commands" in test_data:
        command_results = {}
        for command in test_data["commands"]:
            sanitized = validator.sanitize_command_input(command)
            command_results[command] = {
                "original": command,
                "sanitized": sanitized,
                "changed": command != sanitized
            }
        results["tests"]["command_sanitization"] = command_results
    
    return results

async def test_authentication(test_data: Dict[str, Any]) -> Dict[str, Any]:
    """Test authentication features"""
    results = {
        "success": True,
        "tests": {}
    }
    
    # Test token generation and verification
    if "user_id" in test_data and "permissions" in test_data:
        user_id = test_data["user_id"]
        permissions = test_data["permissions"]
        
        # Generate token
        token = authenticator.generate_secure_token(user_id, permissions)
        results["tests"]["token_generation"] = {
            "user_id": user_id,
            "permissions": permissions,
            "token_length": len(token)
        }
        
        # Verify token
        payload = authenticator.verify_token(token)
        results["tests"]["token_verification"] = {
            "valid": payload is not None,
            "payload": payload if payload else None
        }
        
        # Test permission checking
        if permissions:
            permission_check = authenticator.check_permission(token, permissions[0])
            results["tests"]["permission_check"] = {
                "permission": permissions[0],
                "has_permission": permission_check
            }
    
    # Test rate limiting
    if "client_id" in test_data:
        client_id = test_data["client_id"]
        rate_limit_results = []
        
        for i in range(15):  # Test rate limiting
            allowed = authenticator.check_rate_limit(client_id, "api_request", limit=10)
            rate_limit_results.append({
                "attempt": i + 1,
                "allowed": allowed
            })
        
        results["tests"]["rate_limiting"] = rate_limit_results
    
    return results

async def test_cryptography(test_data: Dict[str, Any]) -> Dict[str, Any]:
    """Test cryptography features"""
    results = {
        "success": True,
        "tests": {}
    }
    
    # Test data encryption/decryption
    if "sensitive_data" in test_data:
        data = test_data["sensitive_data"]
        
        # Encrypt data
        encrypted = crypto.encrypt_sensitive_data(data)
        results["tests"]["encryption"] = {
            "original": data,
            "encrypted": encrypted,
            "encrypted_length": len(encrypted)
        }
        
        # Decrypt data
        decrypted = crypto.decrypt_sensitive_data(encrypted)
        results["tests"]["decryption"] = {
            "decrypted": decrypted,
            "matches_original": data == decrypted
        }
    
    # Test password hashing
    if "password" in test_data:
        password = test_data["password"]
        
        # Hash password
        hash_result = crypto.hash_password(password)
        results["tests"]["password_hashing"] = {
            "password": "***REDACTED***",
            "hash": hash_result["hash"],
            "salt": hash_result["salt"],
            "hash_length": len(hash_result["hash"])
        }
        
        # Verify password
        verification = crypto.verify_password(password, hash_result["hash"], hash_result["salt"])
        results["tests"]["password_verification"] = {
            "correct_password": verification
        }
        
        # Test wrong password
        wrong_verification = crypto.verify_password("wrong_password", hash_result["hash"], hash_result["salt"])
        results["tests"]["wrong_password_verification"] = {
            "wrong_password": wrong_verification
        }
    
    # Test secure random generation
    random_string = crypto.generate_secure_random_string(32)
    results["tests"]["secure_random"] = {
        "random_string": random_string,
        "length": len(random_string)
    }
    
    return results

async def test_secure_logging(test_data: Dict[str, Any]) -> Dict[str, Any]:
    """Test secure logging features"""
    results = {
        "success": True,
        "tests": {}
    }
    
    # Test security event logging
    if "event_type" in test_data:
        event_type = test_data["event_type"]
        details = test_data.get("details", {})
        user_id = test_data.get("user_id")
        
        logger.log_security_event(event_type, details, user_id)
        results["tests"]["security_event_logging"] = {
            "event_type": event_type,
            "details": details,
            "user_id": user_id,
            "logged": True
        }
    
    # Test authentication attempt logging
    if "auth_attempts" in test_data:
        auth_results = []
        for attempt in test_data["auth_attempts"]:
            logger.log_authentication_attempt(
                attempt["user_id"],
                attempt["success"],
                attempt["ip_address"]
            )
            auth_results.append({
                "user_id": attempt["user_id"],
                "success": attempt["success"],
                "ip_address": attempt["ip_address"],
                "logged": True
            })
        results["tests"]["authentication_logging"] = auth_results
    
    # Test authorization failure logging
    if "auth_failures" in test_data:
        failure_results = []
        for failure in test_data["auth_failures"]:
            logger.log_authorization_failure(
                failure["user_id"],
                failure["action"],
                failure["resource"]
            )
            failure_results.append({
                "user_id": failure["user_id"],
                "action": failure["action"],
                "resource": failure["resource"],
                "logged": True
            })
        results["tests"]["authorization_logging"] = failure_results
    
    return results

async def test_security_headers(test_data: Dict[str, Any]) -> Dict[str, Any]:
    """Test security headers"""
    results = {
        "success": True,
        "tests": {}
    }
    
    # Get security headers
    headers = SecurityHeaders.get_security_headers()
    results["tests"]["security_headers"] = {
        "headers": headers,
        "header_count": len(headers)
    }
    
    # Test specific headers
    expected_headers = [
        'X-Content-Type-Options',
        'X-Frame-Options',
        'X-XSS-Protection',
        'Strict-Transport-Security',
        'Content-Security-Policy'
    ]
    
    header_validation = {}
    for header in expected_headers:
        header_validation[header] = header in headers
    
    results["tests"]["header_validation"] = header_validation
    
    return results

@router.post("/vulnerability-scan", response_model=VulnerabilityScanResponse)
async def perform_vulnerability_scan(
    request: VulnerabilityScanRequest
) -> VulnerabilityScanResponse:
    """Perform vulnerability scan with security validation"""
    
    # Validate target
    if not validator.validate_ip_address(request.target) and not validator.validate_hostname(request.target):
        raise HTTPException(
            status_code=400,
            detail="Invalid target address"
        )
    
    # Simulate vulnerability scan
    vulnerabilities = []
    risk_level = "LOW"
    recommendations = []
    
    if request.scan_type == "sql_injection":
        if request.payload and any(keyword in request.payload.upper() for keyword in ['SELECT', 'UNION', 'DROP']):
            vulnerabilities.append("Potential SQL injection detected")
            risk_level = "HIGH"
            recommendations.append("Use parameterized queries")
            recommendations.append("Implement input validation")
    
    elif request.scan_type == "xss":
        if request.payload and any(tag in request.payload.lower() for tag in ['<script>', '<img>', 'javascript:']):
            vulnerabilities.append("Potential XSS attack detected")
            risk_level = "MEDIUM"
            recommendations.append("Sanitize user input")
            recommendations.append("Use Content Security Policy")
    
    elif request.scan_type == "csrf":
        vulnerabilities.append("CSRF protection should be implemented")
        risk_level = "MEDIUM"
        recommendations.append("Use CSRF tokens")
        recommendations.append("Validate request origin")
    
    elif request.scan_type == "path_traversal":
        if request.payload and ('..' in request.payload or request.payload.startswith('/')):
            vulnerabilities.append("Potential path traversal attack detected")
            risk_level = "HIGH"
            recommendations.append("Validate file paths")
            recommendations.append("Use safe file operations")
    
    # Log scan attempt
    logger.log_security_event('vulnerability_scan', {
        'target': request.target,
        'scan_type': request.scan_type,
        'vulnerabilities_found': len(vulnerabilities),
        'risk_level': risk_level
    })
    
    return VulnerabilityScanResponse(
        target=request.target,
        scan_type=request.scan_type,
        vulnerabilities_found=vulnerabilities,
        risk_level=risk_level,
        recommendations=recommendations,
        scan_timestamp=datetime.utcnow()
    )

@router.get("/security-config")
async def get_security_configuration() -> Dict[str, Any]:
    """Get security configuration"""
    return {
        "max_scan_duration": security_config.max_scan_duration,
        "rate_limit_per_minute": security_config.rate_limit_per_minute,
        "max_file_size": security_config.max_file_size,
        "allowed_file_types": security_config.allowed_file_types,
        "session_timeout": security_config.session_timeout,
        "password_min_length": security_config.password_min_length,
        "require_special_chars": security_config.require_special_chars,
        "max_login_attempts": security_config.max_login_attempts,
        "lockout_duration": security_config.lockout_duration
    }

@router.get("/security-best-practices")
async def get_security_best_practices() -> Dict[str, Any]:
    """Get security best practices implementation"""
    return {
        "defense_in_depth": SecurityBestPractices.implement_defense_in_depth(),
        "least_privilege": SecurityBestPractices.implement_least_privilege(),
        "secure_by_default": SecurityBestPractices.implement_secure_by_default()
    }

@router.post("/file-upload-test")
async async def test_file_upload_security(file_content: str, filename: str) -> Dict[str, Any]:
    """Test file upload security"""
    results = {
        "success": True,
        "tests": {}
    }
    
    # Test filename sanitization
    secure_filename = SecurityUtils.generate_secure_filename(filename)
    results["tests"]["filename_sanitization"] = {
        "original": filename,
        "secure": secure_filename,
        "changed": filename != secure_filename
    }
    
    # Test file content validation
    file_bytes = file_content.encode()
    is_valid = SecurityUtils.validate_file_upload(file_bytes)
    results["tests"]["file_validation"] = {
        "file_size": len(file_bytes),
        "is_valid": is_valid
    }
    
    # Test SQL query sanitization
    if "sql_query" in file_content:
        try:
            sanitized_query = SecurityUtils.sanitize_sql_query(file_content)
            results["tests"]["sql_sanitization"] = {
                "original": file_content,
                "sanitized": sanitized_query,
                "changed": file_content != sanitized_query
            }
        except ValueError as e:
            results["tests"]["sql_sanitization"] = {
                "error": str(e),
                "blocked": True
            }
    
    return results

# Demo function
async def run_security_demo():
    """Run comprehensive security demo"""
    print("=== Security Guidelines Demo ===\n")
    
    # Test input validation
    print("1. Testing Input Validation...")
    input_test_data = {
        "ip_addresses": ["192.168.1.1", "256.256.256.256", "2001:db8::1"],
        "hostnames": ["example.com", "invalid..hostname", "test-host_123"],
        "ports": [80, 443, 65536, -1],
        "commands": ["ls -la", "rm -rf /; echo hacked", "cat /etc/passwd"]
    }
    
    input_results = await test_input_validation(input_test_data)
    print(f"   IP Validation: {input_results['tests']['ip_validation']}")
    print(f"   Hostname Validation: {input_results['tests']['hostname_validation']}")
    print(f"   Port Validation: {input_results['tests']['port_validation']}")
    
    # Test authentication
    print("\n2. Testing Authentication...")
    auth_test_data = {
        "user_id": "test_user",
        "permissions": ["read", "write", "admin"],
        "client_id": "test_client"
    }
    
    auth_results = await test_authentication(auth_test_data)
    print(f"   Token Generation: {auth_results['tests']['token_generation']['token_length']} chars")
    print(f"   Token Verification: {auth_results['tests']['token_verification']['valid']}")
    print(f"   Rate Limiting: {len([r for r in auth_results['tests']['rate_limiting'] if r['allowed']])}/15 allowed")
    
    # Test cryptography
    print("\n3. Testing Cryptography...")
    crypto_test_data = {
        "sensitive_data": "secret_password_123",
        "password": "MySecurePassword123!"
    }
    
    crypto_results = await test_cryptography(crypto_test_data)
    print(f"   Encryption/Decryption: {crypto_results['tests']['decryption']['matches_original']}")
    print(f"   Password Hashing: {crypto_results['tests']['password_verification']['correct_password']}")
    print(f"   Secure Random: {crypto_results['tests']['secure_random']['length']} chars")
    
    # Test security headers
    print("\n4. Testing Security Headers...")
    header_results = await test_security_headers({})
    print(f"   Security Headers: {header_results['tests']['header_validation']}")
    
    # Test vulnerability scanning
    print("\n5. Testing Vulnerability Scanning...")
    vuln_scan = await perform_vulnerability_scan(
        VulnerabilityScanRequest(
            target="192.168.1.1",
            scan_type="sql_injection",
            payload="'; DROP TABLE users; --"
        )
    )
    print(f"   Vulnerabilities Found: {len(vuln_scan.vulnerabilities_found)}")
    print(f"   Risk Level: {vuln_scan.risk_level}")
    
    print("\n=== Security Demo Completed Successfully! ===")

# FastAPI app
app = FastAPI(
    title="Security Guidelines Demo",
    description="Demonstration of security best practices for cybersecurity tools",
    version="1.0.0"
)

# Add security middleware
app.add_middleware(SecurityMiddleware, authenticator=authenticator, logger=logger)

# Include demo router
app.include_router(router)

if __name__ == "__main__":
    print("Security Guidelines Demo")
    print("Access API at: http://localhost:8000")
    print("API Documentation at: http://localhost:8000/docs")
    
    # Run demo
    asyncio.run(run_security_demo())
    
    # Start server
    uvicorn.run(app, host="0.0.0.0", port=8000) 