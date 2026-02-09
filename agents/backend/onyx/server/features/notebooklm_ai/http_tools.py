from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import asyncio
import json
import re
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import logging
    import aiohttp
    import httpx
from error_handling_core import (
from typing import Any, List, Dict, Optional
"""
HTTP Tools - Async HTTP operations for security testing
Uses aiohttp and httpx with proper error handling and guard clauses
"""


# Import HTTP libraries with error handling
try:
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    logging.warning("aiohttp not available")

try:
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    logging.warning("httpx not available")

# Import validation functions
    ValidationError, ErrorContext, ValidationResult, OperationResult,
    create_error_context, log_error_with_context, validate_ip_address
)

logger = logging.getLogger("http_tools")

# ============================================================================
# CORE DATA STRUCTURES
# ============================================================================

@dataclass(frozen=True)
class HttpRequest:
    """Immutable HTTP request configuration"""
    url: str
    method: str = "GET"
    headers: Dict[str, str] = field(default_factory=dict)
    data: Optional[Dict[str, Any]] = None
    json_data: Optional[Dict[str, Any]] = None
    timeout: int = 30
    follow_redirects: bool = True
    verify_ssl: bool = True

@dataclass(frozen=True)
class HttpResponse:
    """Immutable HTTP response with security context"""
    is_successful: bool
    status_code: Optional[int] = None
    headers: Optional[Dict[str, str]] = None
    content: Optional[str] = None
    response_time: Optional[float] = None
    security_headers: Optional[Dict[str, str]] = None
    error_message: Optional[str] = None

@dataclass(frozen=True)
class WebVulnerabilityScanRequest:
    """Immutable web vulnerability scan request"""
    target_url: str
    scan_types: List[str]  # ["sql_injection", "xss", "csrf", "headers"]
    custom_headers: Dict[str, str] = field(default_factory=dict)
    timeout: int = 30
    max_concurrent: int = 5

@dataclass(frozen=True)
class WebVulnerabilityScanResponse:
    """Immutable web vulnerability scan response"""
    is_successful: bool
    vulnerabilities: List[Dict[str, Any]]
    security_headers: Dict[str, str]
    scan_summary: Dict[str, Any]
    error_message: Optional[str] = None

# ============================================================================
# VALIDATION FUNCTIONS (CPU-bound)
# ============================================================================

def validate_url(url: str) -> ValidationResult:
    """Validate URL format - guard clauses with happy path last"""
    # Guard clause: Check if input is string
    if not isinstance(url, str):
        return ValidationResult(
            is_valid=False,
            errors=[ValidationError(
                field_name="url",
                error_type="type_error",
                error_message="URL must be a string",
                value=url,
                expected_format="string"
            )]
        )
    
    # Guard clause: Check if input is empty
    if not url.strip():
        return ValidationResult(
            is_valid=False,
            errors=[ValidationError(
                field_name="url",
                error_type="empty_value",
                error_message="URL cannot be empty",
                value=url
            )]
        )
    
    # Guard clause: Check URL format
    url_pattern = r'^https?://[^\s/$.?#].[^\s]*$'
    if not re.match(url_pattern, url):
        return ValidationResult(
            is_valid=False,
            errors=[ValidationError(
                field_name="url",
                error_type="format_error",
                error_message="Invalid URL format",
                value=url,
                expected_format="http://example.com or https://example.com"
            )]
        )
    
    # Happy path: All validations passed
    return ValidationResult(is_valid=True)

async def validate_http_method(method: str) -> ValidationResult:
    """Validate HTTP method - guard clauses with happy path last"""
    # Guard clause: Check if input is string
    if not isinstance(method, str):
        return ValidationResult(
            is_valid=False,
            errors=[ValidationError(
                field_name="method",
                error_type="type_error",
                error_message="HTTP method must be a string",
                value=method,
                expected_format="string"
            )]
        )
    
    # Guard clause: Check if method is valid
    valid_methods = ["GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS", "PATCH"]
    if method.upper() not in valid_methods:
        return ValidationResult(
            is_valid=False,
            errors=[ValidationError(
                field_name="method",
                error_type="invalid_method",
                error_message=f"Invalid HTTP method. Must be one of: {valid_methods}",
                value=method,
                expected_format="valid HTTP method"
            )]
        )
    
    # Happy path: All validations passed
    return ValidationResult(is_valid=True)

def validate_scan_types(scan_types: List[str]) -> ValidationResult:
    """Validate scan types - guard clauses with happy path last"""
    # Guard clause: Check if input is list
    if not isinstance(scan_types, list):
        return ValidationResult(
            is_valid=False,
            errors=[ValidationError(
                field_name="scan_types",
                error_type="type_error",
                error_message="Scan types must be a list",
                value=scan_types,
                expected_format="list"
            )]
        )
    
    # Guard clause: Check if list is empty
    if len(scan_types) == 0:
        return ValidationResult(
            is_valid=False,
            errors=[ValidationError(
                field_name="scan_types",
                error_type="empty_list",
                error_message="Scan types cannot be empty",
                value=scan_types
            )]
        )
    
    # Guard clause: Validate each scan type
    valid_scan_types = ["sql_injection", "xss", "csrf", "headers", "open_redirect", "file_inclusion"]
    errors = []
    
    for scan_type in scan_types:
        if scan_type not in valid_scan_types:
            errors.append(ValidationError(
                field_name="scan_type",
                error_type="invalid_scan_type",
                error_message=f"Invalid scan type: {scan_type}",
                value=scan_type,
                expected_format=f"one of: {valid_scan_types}"
            ))
    
    if errors:
        return ValidationResult(is_valid=False, errors=errors)
    
    # Happy path: All validations passed
    return ValidationResult(is_valid=True)

# ============================================================================
# HTTP OPERATIONS (I/O-bound)
# ============================================================================

async async def make_http_request_aiohttp(request: HttpRequest) -> HttpResponse:
    """Make HTTP request using aiohttp - guard clauses with happy path last"""
    # Guard clause: Check if aiohttp is available
    if not AIOHTTP_AVAILABLE:
        return HttpResponse(
            is_successful=False,
            error_message="aiohttp not available"
        )
    
    # Guard clause: Validate URL
    url_validation = validate_url(request.url)
    if not url_validation.is_valid:
        return HttpResponse(
            is_successful=False,
            error_message="Invalid URL format"
        )
    
    # Guard clause: Validate HTTP method
    method_validation = validate_http_method(request.method)
    if not method_validation.is_valid:
        return HttpResponse(
            is_successful=False,
            error_message="Invalid HTTP method"
        )
    
    # Guard clause: Validate timeout
    if request.timeout <= 0:
        return HttpResponse(
            is_successful=False,
            error_message="Timeout must be positive"
        )
    
    # Happy path: Make HTTP request
    start_time = datetime.utcnow()
    
    try:
        timeout_config = aiohttp.ClientTimeout(total=request.timeout)
        
        async with aiohttp.ClientSession(timeout=timeout_config) as session:
            # Prepare request data
            request_kwargs = {
                "headers": request.headers,
                "allow_redirects": request.follow_redirects,
                "ssl": request.verify_ssl
            }
            
            if request.data:
                request_kwargs["data"] = request.data
            elif request.json_data:
                request_kwargs["json"] = request.json_data
            
            async with session.request(request.method, request.url, **request_kwargs) as response:
                content = await response.text()
                response_time = (datetime.utcnow() - start_time).total_seconds()
                
                # Extract security headers
                security_headers = {
                    "X-Frame-Options": response.headers.get("X-Frame-Options", ""),
                    "X-Content-Type-Options": response.headers.get("X-Content-Type-Options", ""),
                    "X-XSS-Protection": response.headers.get("X-XSS-Protection", ""),
                    "Strict-Transport-Security": response.headers.get("Strict-Transport-Security", ""),
                    "Content-Security-Policy": response.headers.get("Content-Security-Policy", ""),
                    "Referrer-Policy": response.headers.get("Referrer-Policy", "")
                }
                
                return HttpResponse(
                    is_successful=True,
                    status_code=response.status,
                    headers=dict(response.headers),
                    content=content,
                    response_time=response_time,
                    security_headers=security_headers
                )
                
    except asyncio.TimeoutError:
        return HttpResponse(
            is_successful=False,
            error_message="Request timed out"
        )
    except Exception as e:
        context = create_error_context(
            module_name=__name__,
            function_name="make_http_request_aiohttp",
            parameters={"url": request.url, "method": request.method}
        )
        log_error_with_context(e, context)
        
        return HttpResponse(
            is_successful=False,
            error_message=str(e)
        )

async async def make_http_request_httpx(request: HttpRequest) -> HttpResponse:
    """Make HTTP request using httpx - guard clauses with happy path last"""
    # Guard clause: Check if httpx is available
    if not HTTPX_AVAILABLE:
        return HttpResponse(
            is_successful=False,
            error_message="httpx not available"
        )
    
    # Guard clause: Validate URL
    url_validation = validate_url(request.url)
    if not url_validation.is_valid:
        return HttpResponse(
            is_successful=False,
            error_message="Invalid URL format"
        )
    
    # Guard clause: Validate HTTP method
    method_validation = validate_http_method(request.method)
    if not method_validation.is_valid:
        return HttpResponse(
            is_successful=False,
            error_message="Invalid HTTP method"
        )
    
    # Guard clause: Validate timeout
    if request.timeout <= 0:
        return HttpResponse(
            is_successful=False,
            error_message="Timeout must be positive"
        )
    
    # Happy path: Make HTTP request
    start_time = datetime.utcnow()
    
    try:
        async with httpx.AsyncClient(
            timeout=request.timeout,
            follow_redirects=request.follow_redirects,
            verify=request.verify_ssl
        ) as client:
            # Prepare request data
            request_kwargs = {"headers": request.headers}
            
            if request.data:
                request_kwargs["data"] = request.data
            elif request.json_data:
                request_kwargs["json"] = request.json_data
            
            response = await client.request(request.method, request.url, **request_kwargs)
            content = response.text
            response_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Extract security headers
            security_headers = {
                "X-Frame-Options": response.headers.get("x-frame-options", ""),
                "X-Content-Type-Options": response.headers.get("x-content-type-options", ""),
                "X-XSS-Protection": response.headers.get("x-xss-protection", ""),
                "Strict-Transport-Security": response.headers.get("strict-transport-security", ""),
                "Content-Security-Policy": response.headers.get("content-security-policy", ""),
                "Referrer-Policy": response.headers.get("referrer-policy", "")
            }
            
            return HttpResponse(
                is_successful=True,
                status_code=response.status_code,
                headers=dict(response.headers),
                content=content,
                response_time=response_time,
                security_headers=security_headers
            )
            
    except httpx.TimeoutException:
        return HttpResponse(
            is_successful=False,
            error_message="Request timed out"
        )
    except Exception as e:
        context = create_error_context(
            module_name=__name__,
            function_name="make_http_request_httpx",
            parameters={"url": request.url, "method": request.method}
        )
        log_error_with_context(e, context)
        
        return HttpResponse(
            is_successful=False,
            error_message=str(e)
        )

# ============================================================================
# WEB VULNERABILITY SCANNING
# ============================================================================

async def scan_web_vulnerabilities_roro(request: WebVulnerabilityScanRequest) -> WebVulnerabilityScanResponse:
    """Scan web vulnerabilities - guard clauses with happy path last"""
    # Guard clause: Validate target URL
    url_validation = validate_url(request.target_url)
    if not url_validation.is_valid:
        return WebVulnerabilityScanResponse(
            is_successful=False,
            vulnerabilities=[],
            security_headers={},
            scan_summary={},
            error_message="Invalid target URL"
        )
    
    # Guard clause: Validate scan types
    scan_types_validation = validate_scan_types(request.scan_types)
    if not scan_types_validation.is_valid:
        return WebVulnerabilityScanResponse(
            is_successful=False,
            vulnerabilities=[],
            security_headers={},
            scan_summary={},
            error_message="Invalid scan types"
        )
    
    # Guard clause: Validate timeout
    if request.timeout <= 0:
        return WebVulnerabilityScanResponse(
            is_successful=False,
            vulnerabilities=[],
            security_headers={},
            scan_summary={},
            error_message="Timeout must be positive"
        )
    
    # Guard clause: Validate max concurrent
    if request.max_concurrent <= 0:
        return WebVulnerabilityScanResponse(
            is_successful=False,
            vulnerabilities=[],
            security_headers={},
            scan_summary={},
            error_message="Max concurrent must be positive"
        )
    
    # Happy path: Perform vulnerability scanning
    try:
        vulnerabilities = []
        security_headers = {}
        
        # Initial request to get security headers
        initial_request = HttpRequest(
            url=request.target_url,
            method="GET",
            headers=request.custom_headers,
            timeout=request.timeout
        )
        
        if AIOHTTP_AVAILABLE:
            initial_response = await make_http_request_aiohttp(initial_request)
        elif HTTPX_AVAILABLE:
            initial_response = await make_http_request_httpx(initial_request)
        else:
            return WebVulnerabilityScanResponse(
                is_successful=False,
                vulnerabilities=[],
                security_headers={},
                scan_summary={},
                error_message="No HTTP client available"
            )
        
        if not initial_response.is_successful:
            return WebVulnerabilityScanResponse(
                is_successful=False,
                vulnerabilities=[],
                security_headers={},
                scan_summary={},
                error_message=f"Failed to connect to target: {initial_response.error_message}"
            )
        
        security_headers = initial_response.security_headers or {}
        
        # Perform vulnerability scans
        semaphore = asyncio.Semaphore(request.max_concurrent)
        
        async def scan_with_semaphore(scan_type: str) -> List[Dict[str, Any]]:
            async with semaphore:
                if scan_type == "headers":
                    return await scan_security_headers(request.target_url, request.custom_headers, request.timeout)
                elif scan_type == "xss":
                    return await scan_xss_vulnerabilities(request.target_url, request.custom_headers, request.timeout)
                elif scan_type == "sql_injection":
                    return await scan_sql_injection(request.target_url, request.custom_headers, request.timeout)
                else:
                    return []
        
        scan_tasks = [scan_with_semaphore(scan_type) for scan_type in request.scan_types]
        scan_results = await asyncio.gather(*scan_tasks, return_exceptions=True)
        
        # Combine results
        for result in scan_results:
            if isinstance(result, list):
                vulnerabilities.extend(result)
        
        scan_summary = {
            "target_url": request.target_url,
            "scan_types": request.scan_types,
            "total_vulnerabilities": len(vulnerabilities),
            "vulnerabilities_by_type": {}
        }
        
        for vuln in vulnerabilities:
            vuln_type = vuln.get("type", "unknown")
            if vuln_type not in scan_summary["vulnerabilities_by_type"]:
                scan_summary["vulnerabilities_by_type"][vuln_type] = 0
            scan_summary["vulnerabilities_by_type"][vuln_type] += 1
        
        return WebVulnerabilityScanResponse(
            is_successful=True,
            vulnerabilities=vulnerabilities,
            security_headers=security_headers,
            scan_summary=scan_summary
        )
        
    except Exception as e:
        context = create_error_context(
            module_name=__name__,
            function_name="scan_web_vulnerabilities_roro",
            parameters={"target_url": request.target_url, "scan_types": request.scan_types}
        )
        log_error_with_context(e, context)
        
        return WebVulnerabilityScanResponse(
            is_successful=False,
            vulnerabilities=[],
            security_headers={},
            scan_summary={},
            error_message=str(e)
        )

# ============================================================================
# VULNERABILITY SCANNING FUNCTIONS
# ============================================================================

async def scan_security_headers(url: str, custom_headers: Dict[str, str], timeout: int) -> List[Dict[str, Any]]:
    """Scan for missing security headers"""
    vulnerabilities = []
    
    request = HttpRequest(
        url=url,
        method="GET",
        headers=custom_headers,
        timeout=timeout
    )
    
    if AIOHTTP_AVAILABLE:
        response = await make_http_request_aiohttp(request)
    elif HTTPX_AVAILABLE:
        response = await make_http_request_httpx(request)
    else:
        return vulnerabilities
    
    if not response.is_successful or not response.security_headers:
        return vulnerabilities
    
    security_headers = response.security_headers
    
    # Check for missing security headers
    required_headers = {
        "X-Frame-Options": "Missing X-Frame-Options header (clickjacking protection)",
        "X-Content-Type-Options": "Missing X-Content-Type-Options header (MIME sniffing protection)",
        "X-XSS-Protection": "Missing X-XSS-Protection header (XSS protection)",
        "Strict-Transport-Security": "Missing HSTS header (HTTPS enforcement)",
        "Content-Security-Policy": "Missing CSP header (content security policy)"
    }
    
    for header, description in required_headers.items():
        if not security_headers.get(header):
            vulnerabilities.append({
                "type": "missing_security_header",
                "severity": "medium",
                "description": description,
                "header": header,
                "url": url
            })
    
    return vulnerabilities

async def scan_xss_vulnerabilities(url: str, custom_headers: Dict[str, str], timeout: int) -> List[Dict[str, Any]]:
    """Scan for XSS vulnerabilities"""
    vulnerabilities = []
    
    # XSS payloads to test
    xss_payloads = [
        "<script>alert('XSS')</script>",
        "javascript:alert('XSS')",
        "<img src=x onerror=alert('XSS')>",
        "'><script>alert('XSS')</script>"
    ]
    
    for payload in xss_payloads:
        # Test reflected XSS
        test_url = f"{url}?test={payload}"
        request = HttpRequest(
            url=test_url,
            method="GET",
            headers=custom_headers,
            timeout=timeout
        )
        
        if AIOHTTP_AVAILABLE:
            response = await make_http_request_aiohttp(request)
        elif HTTPX_AVAILABLE:
            response = await make_http_request_httpx(request)
        else:
            continue
        
        if response.is_successful and response.content and payload in response.content:
            vulnerabilities.append({
                "type": "xss",
                "severity": "high",
                "description": f"Reflected XSS vulnerability found with payload: {payload}",
                "payload": payload,
                "url": test_url
            })
    
    return vulnerabilities

async def scan_sql_injection(url: str, custom_headers: Dict[str, str], timeout: int) -> List[Dict[str, Any]]:
    """Scan for SQL injection vulnerabilities"""
    vulnerabilities = []
    
    # SQL injection payloads to test
    sql_payloads = [
        "' OR '1'='1",
        "'; DROP TABLE users; --",
        "' UNION SELECT NULL--",
        "1' AND 1=1--"
    ]
    
    for payload in sql_payloads:
        # Test SQL injection
        test_url = f"{url}?id={payload}"
        request = HttpRequest(
            url=test_url,
            method="GET",
            headers=custom_headers,
            timeout=timeout
        )
        
        if AIOHTTP_AVAILABLE:
            response = await make_http_request_aiohttp(request)
        elif HTTPX_AVAILABLE:
            response = await make_http_request_httpx(request)
        else:
            continue
        
        if response.is_successful and response.content:
            # Check for SQL error messages
            sql_errors = [
                "sql syntax",
                "mysql_fetch_array",
                "oracle error",
                "postgresql error",
                "sql server error"
            ]
            
            content_lower = response.content.lower()
            for error in sql_errors:
                if error in content_lower:
                    vulnerabilities.append({
                        "type": "sql_injection",
                        "severity": "critical",
                        "description": f"SQL injection vulnerability found with payload: {payload}",
                        "payload": payload,
                        "url": test_url,
                        "error_detected": error
                    })
                    break
    
    return vulnerabilities

# ============================================================================
# NAMED EXPORTS
# ============================================================================

__all__ = [
    # Data structures
    "HttpRequest",
    "HttpResponse",
    "WebVulnerabilityScanRequest",
    "WebVulnerabilityScanResponse",
    
    # Validation functions
    "validate_url",
    "validate_http_method",
    "validate_scan_types",
    
    # HTTP operations
    "make_http_request_aiohttp",
    "make_http_request_httpx",
    
    # Vulnerability scanning
    "scan_web_vulnerabilities_roro",
    "scan_security_headers",
    "scan_xss_vulnerabilities",
    "scan_sql_injection",
    
    # Dependency flags
    "AIOHTTP_AVAILABLE",
    "HTTPX_AVAILABLE"
] 