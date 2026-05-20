from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import asyncio
import time
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass
import ssl
import urllib.parse
from urllib.parse import urlparse, urljoin
    import aiohttp
    import httpx
from ..core import BaseConfig, ScanResult, BaseScanner
from typing import Any, List, Dict, Optional
import logging
"""
HTTP scanning and security assessment utilities.
Uses aiohttp and httpx for async HTTP operations.
"""


# Optional imports for HTTP operations
try:
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

try:
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


@dataclass
class HTTPScanConfig(BaseConfig):
    """Configuration for HTTP scanning operations."""
    timeout: float = 10.0
    max_workers: int = 20
    retry_count: int = 2
    follow_redirects: bool = True
    max_redirects: int = 5
    verify_ssl: bool = True
    user_agent: str = "Mozilla/5.0 (Security Scanner) Chrome/91.0.4472.124"
    headers: Dict[str, str] = None
    check_security_headers: bool = True
    check_cors: bool = True
    check_csp: bool = True
    check_hsts: bool = True
    check_xss_protection: bool = True
    check_content_type: bool = True
    
    def __post_init__(self) -> Any:
        if self.headers is None:
            self.headers = {
                "User-Agent": self.user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate",
                "Connection": "keep-alive",
            }

@dataclass
class HTTPScanResult:
    """Result of an HTTP scan operation."""
    target: str
    url: str
    status_code: Optional[int] = None
    response_time: float = 0.0
    content_length: Optional[int] = None
    content_type: Optional[str] = None
    headers: Dict[str, str] = None
    security_headers: Dict[str, Any] = None
    ssl_info: Optional[Dict] = None
    redirects: List[str] = None
    error_message: Optional[str] = None
    success: bool = False
    metadata: Dict[str, Any] = None
    
    def __post_init__(self) -> Any:
        if self.headers is None:
            self.headers = {}
        if self.security_headers is None:
            self.security_headers = {}
        if self.redirects is None:
            self.redirects = []
        if self.metadata is None:
            self.metadata = {}

def validate_url(url: str) -> bool:
    """Validate URL format."""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False

def normalize_url(url: str) -> str:
    """Normalize URL for consistent processing."""
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    return url

def extract_domain(url: str) -> str:
    """Extract domain from URL."""
    parsed = urlparse(normalize_url(url))
    return parsed.netloc

async async def scan_http_target_aiohttp(url: str, config: HTTPScanConfig) -> HTTPScanResult:
    """Scan HTTP target using aiohttp."""
    if not AIOHTTP_AVAILABLE:
        return HTTPScanResult(
            target=url, url=url, success=False,
            error_message="aiohttp not available"
        )
    
    start_time = time.time()
    normalized_url = normalize_url(url)
    
    # Guard clause for invalid inputs
    if not validate_url(normalized_url):
        return HTTPScanResult(
            target=url, url=normalized_url, success=False,
            error_message="Invalid URL format"
        )
    
    try:
        timeout = aiohttp.ClientTimeout(total=config.timeout)
        connector = aiohttp.TCPConnector(verify_ssl=config.verify_ssl)
        
        async with aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            headers=config.headers
        ) as session:
            
            async with session.get(
                normalized_url,
                allow_redirects=config.follow_redirects,
                max_redirects=config.max_redirects
            ) as response:
                
                response_time = time.time() - start_time
                
                # Get response data
                content = await response.read()
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                headers = dict(response.headers)
                
                # Extract security headers
                security_headers = {}
                if config.check_security_headers:
                    security_headers = analyze_security_headers(headers, config)
                
                # SSL information
                ssl_info = None
                if normalized_url.startswith('https://'):
                    ssl_info = extract_ssl_info(response.connection.transport)
                
                return HTTPScanResult(
                    target=url,
                    url=normalized_url,
                    status_code=response.status,
                    response_time=response_time,
                    content_length=len(content),
                    content_type=headers.get('content-type', ''),
                    headers=headers,
                    security_headers=security_headers,
                    ssl_info=ssl_info,
                    redirects=[str(response.url)],
                    success=True
                )
                
    except Exception as e:
        return HTTPScanResult(
            target=url, url=normalized_url, success=False,
            error_message=str(e)
        )

async async def scan_http_target_httpx(url: str, config: HTTPScanConfig) -> HTTPScanResult:
    """Scan HTTP target using httpx."""
    if not HTTPX_AVAILABLE:
        return HTTPScanResult(
            target=url, url=url, success=False,
            error_message="httpx not available"
        )
    
    start_time = time.time()
    normalized_url = normalize_url(url)
    
    # Guard clause for invalid inputs
    if not validate_url(normalized_url):
        return HTTPScanResult(
            target=url, url=normalized_url, success=False,
            error_message="Invalid URL format"
        )
    
    try:
        timeout = httpx.Timeout(config.timeout)
        
        async with httpx.AsyncClient(
            timeout=timeout,
            verify=config.verify_ssl,
            headers=config.headers,
            follow_redirects=config.follow_redirects
        ) as client:
            
            response = await client.get(normalized_url)
            response_time = time.time() - start_time
            
            # Get response data
            content = response.content
            headers = dict(response.headers)
            
            # Extract security headers
            security_headers = {}
            if config.check_security_headers:
                security_headers = analyze_security_headers(headers, config)
            
            # SSL information
            ssl_info = None
            if normalized_url.startswith('https://'):
                ssl_info = extract_ssl_info_httpx(response)
            
            return HTTPScanResult(
                target=url,
                url=normalized_url,
                status_code=response.status_code,
                response_time=response_time,
                content_length=len(content),
                content_type=headers.get('content-type', ''),
                headers=headers,
                security_headers=security_headers,
                ssl_info=ssl_info,
                redirects=[str(response.url)],
                success=True
            )
            
    except Exception as e:
        return HTTPScanResult(
            target=url, url=normalized_url, success=False,
            error_message=str(e)
        )

def analyze_security_headers(headers: Dict[str, str], config: HTTPScanConfig) -> Dict[str, Any]:
    """Analyze security headers in HTTP response."""
    security_analysis = {}
    
    # Content Security Policy
    if config.check_csp and 'content-security-policy' in headers:
        security_analysis['csp'] = {
            'present': True,
            'value': headers['content-security-policy']
        }
    else:
        security_analysis['csp'] = {'present': False}
    
    # HSTS (HTTP Strict Transport Security)
    if config.check_hsts and 'strict-transport-security' in headers:
        security_analysis['hsts'] = {
            'present': True,
            'value': headers['strict-transport-security']
        }
    else:
        security_analysis['hsts'] = {'present': False}
    
    # XSS Protection
    if config.check_xss_protection and 'x-xss-protection' in headers:
        security_analysis['xss_protection'] = {
            'present': True,
            'value': headers['x-xss-protection']
        }
    else:
        security_analysis['xss_protection'] = {'present': False}
    
    # CORS
    if config.check_cors and 'access-control-allow-origin' in headers:
        security_analysis['cors'] = {
            'present': True,
            'value': headers['access-control-allow-origin']
        }
    else:
        security_analysis['cors'] = {'present': False}
    
    # Additional security headers
    security_headers = [
        'x-frame-options', 'x-content-type-options',
        'referrer-policy', 'permissions-policy',
        'cross-origin-embedder-policy', 'cross-origin-opener-policy'
    ]
    
    for header in security_headers:
        if header in headers:
            security_analysis[header.replace('-', '_')] = {
                'present': True,
                'value': headers[header]
            }
        else:
            security_analysis[header.replace('-', '_')] = {'present': False}
    
    return security_analysis

def extract_ssl_info(transport) -> Optional[Dict]:
    """Extract SSL certificate information from aiohttp transport."""
    try:
        if hasattr(transport, 'get_extra_info'):
            cert = transport.get_extra_info('ssl_object')
            if cert:
                return {
                    'version': cert.version(),
                    'cipher': cert.cipher(),
                    'compression': cert.compression(),
                    'verify_mode': cert.verify_mode
                }
    except Exception:
        pass
    return None

async def extract_ssl_info_httpx(response) -> Optional[Dict]:
    """Extract SSL certificate information from httpx response."""
    try:
        if hasattr(response, 'transport') and hasattr(response.transport, 'get_extra_info'):
            cert = response.transport.get_extra_info('ssl_object')
            if cert:
                return {
                    'version': cert.version(),
                    'cipher': cert.cipher(),
                    'compression': cert.compression(),
                    'verify_mode': cert.verify_mode
                }
    except Exception:
        pass
    return None

async async def scan_multiple_http_targets(targets: List[str], config: HTTPScanConfig, 
                                   use_httpx: bool = False) -> List[HTTPScanResult]:
    """Scan multiple HTTP targets concurrently."""
    semaphore = asyncio.Semaphore(config.max_workers)
    
    async def scan_single_target(target: str) -> HTTPScanResult:
        async with semaphore:
            if use_httpx and HTTPX_AVAILABLE:
                return await scan_http_target_httpx(target, config)
            elif AIOHTTP_AVAILABLE:
                return await scan_http_target_aiohttp(target, config)
            else:
                return HTTPScanResult(
                    target=target, url=target, success=False,
                    error_message="No HTTP client available"
                )
    
    tasks = [scan_single_target(target) for target in targets]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter out exceptions
    valid_results = []
    for result in results:
        if isinstance(result, HTTPScanResult):
            valid_results.append(result)
    
    return valid_results

async def analyze_http_results(results: List[HTTPScanResult]) -> Dict[str, Any]:
    """Analyze HTTP scan results for security patterns."""
    if not results:
        return {"total_targets": 0, "successful_scans": 0}
    
    successful_scans = [r for r in results if r.success]
    security_analysis = {
        "total_targets": len(results),
        "successful_scans": len(successful_scans),
        "average_response_time": sum(r.response_time for r in successful_scans) / len(successful_scans) if successful_scans else 0,
        "status_codes": {},
        "security_headers_summary": {},
        "ssl_enabled": 0,
        "redirects_found": 0
    }
    
    # Status code analysis
    for result in successful_scans:
        status = result.status_code
        security_analysis["status_codes"][status] = security_analysis["status_codes"].get(status, 0) + 1
    
    # Security headers analysis
    for result in successful_scans:
        for header, info in result.security_headers.items():
            if info.get('present', False):
                security_analysis["security_headers_summary"][header] = security_analysis["security_headers_summary"].get(header, 0) + 1
    
    # SSL and redirect analysis
    for result in successful_scans:
        if result.ssl_info:
            security_analysis["ssl_enabled"] += 1
        if len(result.redirects) > 1:
            security_analysis["redirects_found"] += 1
    
    return security_analysis

class HTTPScanner(BaseScanner):
    """HTTP scanner with comprehensive security assessment."""
    
    def __init__(self, config: HTTPScanConfig):
        
    """__init__ function."""
self.config = config
    
    async def comprehensive_scan(self, target: str, use_httpx: bool = False) -> Dict[str, Any]:
        """Perform comprehensive HTTP scan."""
        results = {
            "target": target,
            "timestamp": time.time(),
            "aiohttp_available": AIOHTTP_AVAILABLE,
            "httpx_available": HTTPX_AVAILABLE,
            "scan_results": None,
            "analysis": None
        }
        
        # Perform scan
        if use_httpx and HTTPX_AVAILABLE:
            scan_result = await scan_http_target_httpx(target, self.config)
        elif AIOHTTP_AVAILABLE:
            scan_result = await scan_http_target_aiohttp(target, self.config)
        else:
            scan_result = HTTPScanResult(
                target=target, url=target, success=False,
                error_message="No HTTP client available"
            )
        
        results["scan_results"] = scan_result.__dict__
        
        # Analyze results
        if scan_result.success:
            results["analysis"] = {
                "security_score": calculate_security_score(scan_result),
                "recommendations": generate_security_recommendations(scan_result)
            }
        
        return results
    
    async def scan_multiple_targets(self, targets: List[str], use_httpx: bool = False) -> Dict[str, Any]:
        """Scan multiple HTTP targets."""
        results = await scan_multiple_http_targets(targets, self.config, use_httpx)
        
        return {
            "targets": targets,
            "timestamp": time.time(),
            "results": [r.__dict__ for r in results],
            "analysis": analyze_http_results(results),
            "summary": {
                "total": len(results),
                "successful": len([r for r in results if r.success]),
                "failed": len([r for r in results if not r.success])
            }
        }

def calculate_security_score(result: HTTPScanResult) -> int:
    """Calculate security score based on HTTP scan results."""
    score = 0
    
    # SSL/TLS (30 points)
    if result.ssl_info:
        score += 30
    
    # Security headers (40 points)
    security_headers = result.security_headers
    if security_headers.get('hsts', {}).get('present'):
        score += 10
    if security_headers.get('csp', {}).get('present'):
        score += 10
    if security_headers.get('xss_protection', {}).get('present'):
        score += 5
    if security_headers.get('x_frame_options', {}).get('present'):
        score += 5
    if security_headers.get('x_content_type_options', {}).get('present'):
        score += 5
    if security_headers.get('referrer_policy', {}).get('present'):
        score += 5
    
    # Status code (20 points)
    if result.status_code in [200, 301, 302]:
        score += 20
    elif result.status_code in [404, 403]:
        score += 10
    
    # Response time (10 points)
    if result.response_time < 1.0:
        score += 10
    elif result.response_time < 3.0:
        score += 5
    
    return min(score, 100)

def generate_security_recommendations(result: HTTPScanResult) -> List[str]:
    """Generate security recommendations based on scan results."""
    recommendations = []
    
    if not result.ssl_info:
        recommendations.append("Enable HTTPS/TLS encryption")
    
    security_headers = result.security_headers
    
    if not security_headers.get('hsts', {}).get('present'):
        recommendations.append("Implement HTTP Strict Transport Security (HSTS)")
    
    if not security_headers.get('csp', {}).get('present'):
        recommendations.append("Implement Content Security Policy (CSP)")
    
    if not security_headers.get('xss_protection', {}).get('present'):
        recommendations.append("Enable XSS Protection header")
    
    if not security_headers.get('x_frame_options', {}).get('present'):
        recommendations.append("Implement X-Frame-Options to prevent clickjacking")
    
    if not security_headers.get('x_content_type_options', {}).get('present'):
        recommendations.append("Add X-Content-Type-Options: nosniff")
    
    if result.response_time > 3.0:
        recommendations.append("Optimize server response time for better security")
    
    return recommendations 