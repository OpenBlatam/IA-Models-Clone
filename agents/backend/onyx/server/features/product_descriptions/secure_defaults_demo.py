from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

from fastapi import FastAPI, APIRouter, HTTPException, status, Depends
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Tuple
import asyncio
import ssl
import socket
import time
from datetime import datetime

from secure_defaults import (
    import uvicorn
from typing import Any, List, Dict, Optional
import logging
# Import secure defaults components
    SecureDefaultsManager, SecurityLevel, CipherStrength,
    SecurityDefaultsRequest, SecurityDefaultsResponse,
    PasswordValidationRequest, PasswordValidationResponse,
    CertificateGenerationRequest, CertificateGenerationResponse
)

# Data Models for Demo
class SecurityTestRequest(BaseModel):
    test_type: str = Field(..., regex="^(tls|crypto|password|certificate|headers|cookies)$")
    security_level: SecurityLevel = Field(default=SecurityLevel.HIGH)
    test_data: Optional[Dict[str, Any]] = None

class SecurityTestResponse(BaseModel):
    test_type: str
    security_level: SecurityLevel
    results: Dict[str, Any]
    recommendations: List[str]
    timestamp: datetime

class TLSTestRequest(BaseModel):
    host: str = Field(..., min_length=1)
    port: int = Field(default=443, ge=1, le=65535)
    security_level: SecurityLevel = Field(default=SecurityLevel.HIGH)

class TLSTestResponse(BaseModel):
    host: str
    port: int
    connection_successful: bool
    tls_version: str
    cipher_suite: str
    certificate_valid: bool
    security_score: int
    recommendations: List[str]
    timestamp: datetime

# Demo Router
router = APIRouter(prefix="/secure-defaults-demo", tags=["Secure Defaults Demo"])

@router.post("/test-security", response_model=SecurityTestResponse)
async def test_security_features(
    request: SecurityTestRequest
) -> SecurityTestResponse:
    """Test various security features"""
    
    manager = SecureDefaultsManager(request.security_level)
    results = {}
    recommendations = []
    
    if request.test_type == "tls":
        results, recommendations = await test_tls_configuration(manager)
    elif request.test_type == "crypto":
        results, recommendations = await test_crypto_configuration(manager)
    elif request.test_type == "password":
        results, recommendations = await test_password_security(manager)
    elif request.test_type == "certificate":
        results, recommendations = await test_certificate_generation(manager)
    elif request.test_type == "headers":
        results, recommendations = await test_security_headers(manager)
    elif request.test_type == "cookies":
        results, recommendations = await test_cookie_settings(manager)
    
    return SecurityTestResponse(
        test_type=request.test_type,
        security_level=request.security_level,
        results=results,
        recommendations=recommendations,
        timestamp=datetime.utcnow()
    )

async def test_tls_configuration(manager: SecureDefaultsManager) -> Tuple[Dict[str, Any], List[str]]:
    """Test TLS configuration"""
    results = {
        "tls_version": {
            "min_version": manager.defaults.tls_config.min_version.name,
            "max_version": manager.defaults.tls_config.max_version.name
        },
        "cipher_suites": manager.defaults.tls_config.cipher_suites,
        "certificate_requirements": {
            "cert_required": manager.defaults.tls_config.cert_reqs == ssl.CERT_REQUIRED,
            "check_hostname": manager.defaults.tls_config.check_hostname
        },
        "session_security": {
            "session_tickets": manager.defaults.tls_config.session_tickets,
            "session_cache_size": manager.defaults.tls_config.session_cache_size,
            "session_timeout": manager.defaults.tls_config.session_timeout
        }
    }
    
    recommendations = []
    
    # Check TLS version
    if manager.defaults.tls_config.min_version.value < ssl.TLSVersion.TLSv1_2.value:
        recommendations.append("Upgrade minimum TLS version to 1.2 or higher")
    
    # Check cipher suites
    weak_ciphers = ['RC4', 'DES', '3DES', 'MD5']
    for cipher in manager.defaults.tls_config.cipher_suites:
        if any(weak in cipher for weak in weak_ciphers):
            recommendations.append(f"Remove weak cipher: {cipher}")
    
    # Check certificate requirements
    if not manager.defaults.tls_config.cert_reqs == ssl.CERT_REQUIRED:
        recommendations.append("Enable certificate verification")
    
    if not manager.defaults.tls_config.check_hostname:
        recommendations.append("Enable hostname verification")
    
    return results, recommendations

async def test_crypto_configuration(manager: SecureDefaultsManager) -> Tuple[Dict[str, Any], List[str]]:
    """Test cryptographic configuration"""
    results = {
        "hash_algorithm": manager.defaults.crypto_config.hash_algorithm,
        "key_size": manager.defaults.crypto_config.key_size,
        "curve": manager.defaults.crypto_config.curve,
        "encryption_algorithm": manager.defaults.crypto_config.encryption_algorithm,
        "pbkdf2_iterations": manager.defaults.crypto_config.pbkdf2_iterations,
        "salt_length": manager.defaults.crypto_config.salt_length,
        "iv_length": manager.defaults.crypto_config.iv_length,
        "tag_length": manager.defaults.crypto_config.tag_length
    }
    
    recommendations = []
    
    # Check hash algorithm
    weak_hashes = ['md5', 'sha1']
    if manager.defaults.crypto_config.hash_algorithm.lower() in weak_hashes:
        recommendations.append("Use SHA-256 or stronger hash algorithm")
    
    # Check key size
    if manager.defaults.crypto_config.key_size < 2048:
        recommendations.append("Use at least 2048-bit RSA keys")
    
    # Check PBKDF2 iterations
    if manager.defaults.crypto_config.pbkdf2_iterations < 100000:
        recommendations.append("Use at least 100,000 PBKDF2 iterations")
    
    return results, recommendations

async def test_password_security(manager: SecureDefaultsManager) -> Tuple[Dict[str, Any], List[str]]:
    """Test password security configuration"""
    results = {
        "password_requirements": {
            "min_length": manager.defaults.password_min_length,
            "require_special_chars": manager.defaults.require_special_chars,
            "require_numbers": manager.defaults.require_numbers,
            "require_uppercase": manager.defaults.require_uppercase,
            "require_lowercase": manager.defaults.require_lowercase
        },
        "session_security": {
            "session_timeout": manager.defaults.session_timeout,
            "max_login_attempts": manager.defaults.max_login_attempts,
            "lockout_duration": manager.defaults.lockout_duration
        }
    }
    
    recommendations = []
    
    # Check password length
    if manager.defaults.password_min_length < 12:
        recommendations.append("Increase minimum password length to 12 characters")
    
    # Check password complexity
    if not manager.defaults.require_special_chars:
        recommendations.append("Require special characters in passwords")
    
    if not manager.defaults.require_numbers:
        recommendations.append("Require numbers in passwords")
    
    if not manager.defaults.require_uppercase:
        recommendations.append("Require uppercase letters in passwords")
    
    # Check session security
    if manager.defaults.session_timeout > 3600:
        recommendations.append("Reduce session timeout to 1 hour or less")
    
    if manager.defaults.max_login_attempts > 5:
        recommendations.append("Reduce maximum login attempts to 5 or less")
    
    return results, recommendations

async def test_certificate_generation(manager: SecureDefaultsManager) -> Tuple[Dict[str, Any], List[str]]:
    """Test certificate generation"""
    try:
        cert_pem, key_pem = manager.generate_self_signed_certificate("test.example.com")
        
        results = {
            "certificate_generated": True,
            "certificate_size": len(cert_pem),
            "private_key_size": len(key_pem),
            "key_size": manager.defaults.crypto_config.key_size,
            "hash_algorithm": manager.defaults.crypto_config.hash_algorithm
        }
        
        recommendations = []
        
        # Check certificate security
        if manager.defaults.crypto_config.key_size < 2048:
            recommendations.append("Use at least 2048-bit keys for certificates")
        
        if manager.defaults.crypto_config.hash_algorithm.lower() in ['md5', 'sha1']:
            recommendations.append("Use SHA-256 or stronger for certificate signing")
        
    except Exception as e:
        results = {
            "certificate_generated": False,
            "error": str(e)
        }
        recommendations = ["Fix certificate generation issues"]
    
    return results, recommendations

async def test_security_headers(manager: SecureDefaultsManager) -> Tuple[Dict[str, Any], List[str]]:
    """Test security headers configuration"""
    headers = manager.get_security_headers()
    
    results = {
        "security_headers": headers,
        "header_count": len(headers),
        "critical_headers_present": {
            "HSTS": "Strict-Transport-Security" in headers,
            "CSP": "Content-Security-Policy" in headers,
            "X-Frame-Options": "X-Frame-Options" in headers,
            "X-Content-Type-Options": "X-Content-Type-Options" in headers,
            "X-XSS-Protection": "X-XSS-Protection" in headers
        }
    }
    
    recommendations = []
    
    # Check for missing headers
    required_headers = [
        "Strict-Transport-Security",
        "Content-Security-Policy",
        "X-Frame-Options",
        "X-Content-Type-Options",
        "X-XSS-Protection"
    ]
    
    for header in required_headers:
        if header not in headers:
            recommendations.append(f"Add missing security header: {header}")
    
    # Check HSTS configuration
    if "Strict-Transport-Security" in headers:
        hsts_value = headers["Strict-Transport-Security"]
        if "max-age=" not in hsts_value:
            recommendations.append("HSTS header should include max-age directive")
        if "preload" not in hsts_value:
            recommendations.append("Consider adding preload to HSTS header")
    
    return results, recommendations

async def test_cookie_settings(manager: SecureDefaultsManager) -> Tuple[Dict[str, Any], List[str]]:
    """Test cookie security settings"""
    cookie_settings = manager.get_cookie_settings()
    
    results = {
        "cookie_settings": cookie_settings,
        "security_features": {
            "secure": cookie_settings.get("secure", False),
            "httponly": cookie_settings.get("httponly", False),
            "samesite": cookie_settings.get("samesite", "lax")
        }
    }
    
    recommendations = []
    
    # Check secure flag
    if not cookie_settings.get("secure", False):
        recommendations.append("Enable secure flag for cookies")
    
    # Check httpOnly flag
    if not cookie_settings.get("httponly", False):
        recommendations.append("Enable httpOnly flag for cookies")
    
    # Check SameSite setting
    samesite = cookie_settings.get("samesite", "lax")
    if samesite not in ["strict", "lax"]:
        recommendations.append("Set SameSite to 'strict' or 'lax'")
    elif samesite == "lax":
        recommendations.append("Consider using SameSite 'strict' for better security")
    
    return results, recommendations

@router.post("/test-tls-connection", response_model=TLSTestResponse)
async def test_tls_connection(request: TLSTestRequest) -> TLSTestResponse:
    """Test TLS connection to a host"""
    
    manager = SecureDefaultsManager(request.security_level)
    connection_successful = False
    tls_version = "Unknown"
    cipher_suite = "Unknown"
    certificate_valid = False
    security_score = 0
    recommendations = []
    
    try:
        # Create secure socket
        sock = manager.create_secure_socket(request.host, request.port)
        
        # Get connection info
        connection_successful = True
        tls_version = sock.version()
        cipher_suite = sock.cipher()[0]
        
        # Check certificate
        cert = sock.getpeercert()
        certificate_valid = cert is not None
        
        # Calculate security score
        security_score = 0
        
        # TLS version score
        if tls_version == "TLSv1.3":
            security_score += 40
        elif tls_version == "TLSv1.2":
            security_score += 30
        elif tls_version == "TLSv1.1":
            security_score += 10
        else:
            security_score += 0
        
        # Cipher suite score
        strong_ciphers = ['AES_256_GCM', 'CHACHA20_POLY1305', 'AES_128_GCM']
        if any(cipher in cipher_suite for cipher in strong_ciphers):
            security_score += 30
        elif 'AES' in cipher_suite:
            security_score += 20
        else:
            security_score += 10
        
        # Certificate score
        if certificate_valid:
            security_score += 30
        else:
            security_score += 0
        
        # Generate recommendations
        if tls_version != "TLSv1.3":
            recommendations.append("Upgrade to TLS 1.3 for maximum security")
        
        if not certificate_valid:
            recommendations.append("Use valid SSL/TLS certificates")
        
        if security_score < 70:
            recommendations.append("Improve overall TLS security configuration")
        
        sock.close()
        
    except Exception as e:
        recommendations.append(f"Connection failed: {str(e)}")
    
    return TLSTestResponse(
        host=request.host,
        port=request.port,
        connection_successful=connection_successful,
        tls_version=tls_version,
        cipher_suite=cipher_suite,
        certificate_valid=certificate_valid,
        security_score=security_score,
        recommendations=recommendations,
        timestamp=datetime.utcnow()
    )

@router.get("/security-benchmark")
async def run_security_benchmark() -> Dict[str, Any]:
    """Run comprehensive security benchmark"""
    
    benchmark_results = {}
    
    # Test all security levels
    for level in SecurityLevel:
        manager = SecureDefaultsManager(level)
        
        # Test TLS configuration
        tls_results, tls_recommendations = await test_tls_configuration(manager)
        
        # Test crypto configuration
        crypto_results, crypto_recommendations = await test_crypto_configuration(manager)
        
        # Test password security
        password_results, password_recommendations = await test_password_security(manager)
        
        # Calculate overall score
        total_recommendations = len(tls_recommendations + crypto_recommendations + password_recommendations)
        max_recommendations = 15  # Estimated maximum recommendations
        score = max(0, 100 - (total_recommendations / max_recommendations) * 100)
        
        benchmark_results[level.value] = {
            "security_score": round(score, 1),
            "tls_configuration": {
                "score": max(0, 100 - len(tls_recommendations) * 10),
                "recommendations": tls_recommendations
            },
            "crypto_configuration": {
                "score": max(0, 100 - len(crypto_recommendations) * 10),
                "recommendations": crypto_recommendations
            },
            "password_security": {
                "score": max(0, 100 - len(password_recommendations) * 10),
                "recommendations": password_recommendations
            },
            "total_recommendations": total_recommendations
        }
    
    return {
        "benchmark_results": benchmark_results,
        "timestamp": datetime.utcnow().isoformat(),
        "summary": {
            "best_level": max(benchmark_results.keys(), 
                            key=lambda x: benchmark_results[x]["security_score"]),
            "worst_level": min(benchmark_results.keys(), 
                             key=lambda x: benchmark_results[x]["security_score"])
        }
    }

# Demo function
async def run_secure_defaults_demo():
    """Run comprehensive secure defaults demo"""
    print("=== Secure Defaults Demo ===\n")
    
    # Test different security levels
    print("1. Testing Different Security Levels...")
    for level in SecurityLevel:
        manager = SecureDefaultsManager(level)
        print(f"   Level: {level.value}")
        print(f"   TLS Min Version: {manager.defaults.tls_config.min_version.name}")
        print(f"   Key Size: {manager.defaults.crypto_config.key_size} bits")
        print(f"   Password Min Length: {manager.defaults.password_min_length}")
        print(f"   Session Timeout: {manager.defaults.session_timeout} seconds")
        print(f"   Max Login Attempts: {manager.defaults.max_login_attempts}")
        print()
    
    # Test password generation and validation
    print("2. Testing Password Security...")
    manager = SecureDefaultsManager(SecurityLevel.HIGH)
    
    # Generate secure password
    password = manager._generate_secure_password()
    print(f"   Generated Password: {password}")
    
    # Validate password
    validation = manager.validate_password_strength(password)
    print(f"   Is Valid: {validation['is_valid']}")
    print(f"   Strength Score: {validation['strength_score']}/100")
    print(f"   Errors: {validation['errors']}")
    print(f"   Warnings: {validation['warnings']}")
    print()
    
    # Test weak password
    weak_password = "password123"
    weak_validation = manager.validate_password_strength(weak_password)
    print(f"   Weak Password: {weak_password}")
    print(f"   Is Valid: {weak_validation['is_valid']}")
    print(f"   Strength Score: {weak_validation['strength_score']}/100")
    print(f"   Errors: {weak_validation['errors']}")
    print()
    
    # Test certificate generation
    print("3. Testing Certificate Generation...")
    try:
        cert_pem, key_pem = manager.generate_self_signed_certificate("example.com")
        print(f"   Certificate Generated Successfully")
        print(f"   Certificate Size: {len(cert_pem)} bytes")
        print(f"   Private Key Size: {len(key_pem)} bytes")
        print(f"   Key Size: {manager.defaults.crypto_config.key_size} bits")
        print(f"   Hash Algorithm: {manager.defaults.crypto_config.hash_algorithm}")
    except Exception as e:
        print(f"   Certificate Generation Failed: {e}")
    print()
    
    # Test security headers
    print("4. Testing Security Headers...")
    headers = manager.get_security_headers()
    for header, value in headers.items():
        print(f"   {header}: {value}")
    print()
    
    # Test cookie settings
    print("5. Testing Cookie Settings...")
    cookie_settings = manager.get_cookie_settings()
    for setting, value in cookie_settings.items():
        print(f"   {setting}: {value}")
    print()
    
    # Test TLS connection (simulated)
    print("6. Testing TLS Configuration...")
    try:
        context = manager.create_ssl_context()
        print(f"   SSL Context Created Successfully")
        print(f"   Min TLS Version: {context.minimum_version.name}")
        print(f"   Max TLS Version: {context.maximum_version.name}")
        print(f"   Verify Mode: {context.verify_mode}")
        print(f"   Check Hostname: {context.check_hostname}")
        print(f"   Cipher Suites: {len(manager.defaults.tls_config.cipher_suites)} configured")
    except Exception as e:
        print(f"   SSL Context Creation Failed: {e}")
    print()
    
    # Run security benchmark
    print("7. Running Security Benchmark...")
    benchmark_results = {}
    for level in SecurityLevel:
        manager = SecureDefaultsManager(level)
        tls_results, tls_recs = await test_tls_configuration(manager)
        crypto_results, crypto_recs = await test_crypto_configuration(manager)
        password_results, password_recs = await test_password_security(manager)
        
        total_recs = len(tls_recs + crypto_recs + password_recs)
        score = max(0, 100 - (total_recs / 15) * 100)
        
        benchmark_results[level.value] = round(score, 1)
        print(f"   {level.value}: {round(score, 1)}/100")
    
    best_level = max(benchmark_results.keys(), key=lambda x: benchmark_results[x])
    print(f"   Best Level: {best_level} ({benchmark_results[best_level]}/100)")
    print()
    
    print("=== Secure Defaults Demo Completed! ===")

# FastAPI app
app = FastAPI(
    title="Secure Defaults Demo",
    description="Demonstration of secure defaults for cybersecurity tools",
    version="1.0.0"
)

# Include demo router
app.include_router(router)

if __name__ == "__main__":
    print("Secure Defaults Demo")
    print("Access API at: http://localhost:8000")
    print("API Documentation at: http://localhost:8000/docs")
    
    # Run demo
    asyncio.run(run_secure_defaults_demo())
    
    # Start server
    uvicorn.run(app, host="0.0.0.0", port=8000) 