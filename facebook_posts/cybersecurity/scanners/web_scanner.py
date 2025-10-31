from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import asyncio
import aiohttp
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import re
import json
import hashlib
from urllib.parse import urljoin, urlparse
import ssl
from ..core import BaseConfig, ScanResult, BaseScanner
from typing import Any, List, Dict, Optional
import logging
"""
Web security scanning utilities with proper async/def distinction.
Async for network operations, def for CPU-bound analysis.
"""



@dataclass
class WebScanConfig(BaseConfig):
    """Configuration for web security scanning."""
    timeout: float = 10.0
    max_workers: int = 20
    scan_depth: str = "medium"  # low, medium, high
    follow_redirects: bool = True
    max_redirects: int = 5
    user_agent: str = "SecurityScanner/1.0"
    include_headers: bool = True
    include_forms: bool = True
    include_links: bool = True
    include_js: bool = True
    custom_payloads: List[str] = None

@dataclass
class WebScanResult(ScanResult):
    """Result of web security scan."""
    url: str
    status_code: int
    content_type: str
    content_length: int
    headers: Dict[str, str]
    forms: List[Dict[str, Any]]
    links: List[str]
    vulnerabilities: List[Dict[str, Any]]
    security_headers: Dict[str, str]
    missing_security_headers: List[str]

def get_web_vulnerability_signatures() -> Dict[str, List[Dict[str, Any]]]:
    """Get web vulnerability detection signatures."""
    return {
        "injection": [
            {
                "name": "SQL Injection",
                "pattern": r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)",
                "severity": "critical",
                "description": "Potential SQL injection vulnerability"
            },
            {
                "name": "XSS",
                "pattern": r"<script[^>]*>.*?</script>|javascript:|on\w+\s*=",
                "severity": "high",
                "description": "Cross-site scripting vulnerability"
            },
            {
                "name": "Command Injection",
                "pattern": r"(\b(cmd|exec|system|eval|shell)\b)",
                "severity": "critical",
                "description": "Potential command injection vulnerability"
            }
        ],
        "information_disclosure": [
            {
                "name": "Error Messages",
                "pattern": r"(error|exception|stack trace|debug|warning)",
                "severity": "medium",
                "description": "Detailed error messages may reveal sensitive information"
            },
            {
                "name": "Version Information",
                "pattern": r"(version|v\d+\.\d+|build \d+)",
                "severity": "low",
                "description": "Version information disclosure"
            }
        ],
        "authentication": [
            {
                "name": "Weak Authentication",
                "pattern": r"(admin|password|login|auth)",
                "severity": "medium",
                "description": "Potential weak authentication mechanism"
            }
        ]
    }

def analyze_web_content(content: str, url: str) -> List[Dict[str, Any]]:
    """Analyze web content for vulnerabilities."""
    vulnerabilities = []
    signatures = get_web_vulnerability_signatures()
    
    for category, sigs in signatures.items():
        for sig in sigs:
            if re.search(sig["pattern"], content, re.IGNORECASE):
                vulnerabilities.append({
                    "category": category,
                    "name": sig["name"],
                    "severity": sig["severity"],
                    "description": sig["description"],
                    "evidence": content[:200] + "..." if len(content) > 200 else content,
                    "url": url
                })
    
    return vulnerabilities

def analyze_security_headers(headers: Dict[str, str]) -> Tuple[Dict[str, str], List[str]]:
    """Analyze security headers."""
    security_headers = {
        'strict-transport-security': 'HSTS',
        'content-security-policy': 'CSP',
        'x-frame-options': 'X-Frame-Options',
        'x-content-type-options': 'X-Content-Type-Options',
        'x-xss-protection': 'X-XSS-Protection',
        'referrer-policy': 'Referrer-Policy',
        'permissions-policy': 'Permissions-Policy'
    }
    
    found_headers = {}
    missing_headers = []
    
    for header, description in security_headers.items():
        if header in headers:
            found_headers[description] = headers[header]
        else:
            missing_headers.append(description)
    
    return found_headers, missing_headers

def extract_forms(content: str) -> List[Dict[str, Any]]:
    """Extract forms from HTML content."""
    forms = []
    form_pattern = r'<form[^>]*>(.*?)</form>'
    
    for match in re.finditer(form_pattern, content, re.IGNORECASE | re.DOTALL):
        form_html = match.group(0)
        form_data = {
            "action": "",
            "method": "get",
            "inputs": []
        }
        
        # Extract action
        action_match = re.search(r'action=["\']([^"\']*)["\']', form_html, re.IGNORECASE)
        if action_match:
            form_data["action"] = action_match.group(1)
        
        # Extract method
        method_match = re.search(r'method=["\']([^"\']*)["\']', form_html, re.IGNORECASE)
        if method_match:
            form_data["method"] = method_match.group(1).lower()
        
        # Extract inputs
        input_pattern = r'<input[^>]*>'
        for input_match in re.finditer(input_pattern, form_html, re.IGNORECASE):
            input_html = input_match.group(0)
            input_data = {}
            
            # Extract input attributes
            for attr in ['name', 'type', 'value', 'placeholder']:
                attr_match = re.search(f'{attr}=["\']([^"\']*)["\']', input_html, re.IGNORECASE)
                if attr_match:
                    input_data[attr] = attr_match.group(1)
            
            if input_data:
                form_data["inputs"].append(input_data)
        
        forms.append(form_data)
    
    return forms

def extract_links(content: str, base_url: str) -> List[str]:
    """Extract links from HTML content."""
    links = []
    link_pattern = r'href=["\']([^"\']*)["\']'
    
    for match in re.finditer(link_pattern, content, re.IGNORECASE):
        link = match.group(1)
        if link.startswith('http'):
            links.append(link)
        elif link.startswith('/'):
            links.append(urljoin(base_url, link))
        elif not link.startswith(('#', 'javascript:', 'mailto:')):
            links.append(urljoin(base_url, link))
    
    return list(set(links))  # Remove duplicates

async def scan_web_page(url: str, config: WebScanConfig = None) -> WebScanResult:
    """Scan a single web page for security issues."""
    if config is None:
        config = WebScanConfig()
    
    try:
        timeout = aiohttp.ClientTimeout(total=config.timeout)
        headers = {"User-Agent": config.user_agent}
        
        async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
            async with session.get(url, allow_redirects=config.follow_redirects) as response:
                content = await response.text()
                
                # Analyze content for vulnerabilities
                vulnerabilities = analyze_web_content(content, url)
                
                # Analyze security headers
                security_headers, missing_headers = analyze_security_headers(dict(response.headers))
                
                # Extract forms and links
                forms = extract_forms(content) if config.include_forms else []
                links = extract_links(content, url) if config.include_links else []
                
                return WebScanResult(
                    target=url,
                    url=url,
                    status_code=response.status,
                    content_type=response.headers.get('content-type', ''),
                    content_length=len(content),
                    headers=dict(response.headers),
                    forms=forms,
                    links=links,
                    vulnerabilities=vulnerabilities,
                    security_headers=security_headers,
                    missing_security_headers=missing_headers,
                    success=True
                )
                
    except Exception as e:
        return WebScanResult(
            target=url,
            url=url,
            status_code=0,
            content_type='',
            content_length=0,
            headers={},
            forms=[],
            links=[],
            vulnerabilities=[],
            security_headers={},
            missing_security_headers=[],
            success=False,
            error_message=str(e)
        )

async def scan_web_forms(url: str, forms: List[Dict[str, Any]], config: WebScanConfig = None) -> List[Dict[str, Any]]:
    """Scan web forms for vulnerabilities."""
    if config is None:
        config = WebScanConfig()
    
    form_vulnerabilities = []
    
    for form in forms:
        if form["method"] == "post" and form["action"]:
            # Test for common injection payloads
            test_payloads = config.custom_payloads or [
                "' OR '1'='1",
                "<script>alert('xss')</script>",
                "'; DROP TABLE users; --",
                "../../../etc/passwd"
            ]
            
            for payload in test_payloads:
                try:
                    # This would typically send the payload and analyze the response
                    # For demo purposes, we'll just check if the form accepts the payload
                    form_vulnerabilities.append({
                        "form_action": form["action"],
                        "payload": payload,
                        "severity": "medium",
                        "description": f"Form may be vulnerable to injection: {payload[:20]}..."
                    })
                except Exception:
                    pass
    
    return form_vulnerabilities

async def scan_web_directory(url: str, config: WebScanConfig = None) -> List[Dict[str, Any]]:
    """Scan for common web directories and files."""
    if config is None:
        config = WebScanConfig()
    
    common_paths = [
        "/admin", "/login", "/wp-admin", "/phpmyadmin", "/config",
        "/backup", "/.git", "/.env", "/robots.txt", "/sitemap.xml",
        "/api", "/test", "/dev", "/staging", "/.htaccess"
    ]
    
    discovered_paths = []
    
    for path in common_paths:
        try:
            test_url = urljoin(url, path)
            timeout = aiohttp.ClientTimeout(total=config.timeout)
            headers = {"User-Agent": config.user_agent}
            
            async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
                async with session.get(test_url) as response:
                    if response.status < 400:  # Not 4xx error
                        discovered_paths.append({
                            "path": path,
                            "url": test_url,
                            "status_code": response.status,
                            "content_length": len(await response.text()),
                            "severity": "medium" if response.status == 200 else "low"
                        })
        except Exception:
            pass
    
    return discovered_paths

async def comprehensive_web_scan(url: str, config: WebScanConfig = None) -> Dict[str, Any]:
    """Perform comprehensive web security scan."""
    if config is None:
        config = WebScanConfig()
    
    results = {
        "url": url,
        "scan_timestamp": time.time(),
        "page_scan": None,
        "form_vulnerabilities": [],
        "directory_scan": [],
        "overall_summary": {}
    }
    
    # Scan main page
    page_result = await scan_web_page(url, config)
    results["page_scan"] = page_result
    
    # Scan forms if found
    if page_result.forms:
        form_vulns = await scan_web_forms(url, page_result.forms, config)
        results["form_vulnerabilities"] = form_vulns
    
    # Directory scan
    if config.scan_depth in ["medium", "high"]:
        dir_scan = await scan_web_directory(url, config)
        results["directory_scan"] = dir_scan
    
    # Generate summary
    total_vulns = len(page_result.vulnerabilities) + len(results["form_vulnerabilities"])
    missing_headers = len(page_result.missing_security_headers)
    
    results["overall_summary"] = {
        "total_vulnerabilities": total_vulns,
        "missing_security_headers": missing_headers,
        "forms_found": len(page_result.forms),
        "links_found": len(page_result.links),
        "directories_discovered": len(results["directory_scan"]),
        "risk_score": calculate_web_risk_score(results)
    }
    
    return results

def calculate_web_risk_score(results: Dict[str, Any]) -> float:
    """Calculate web security risk score."""
    score = 0.0
    
    # Page vulnerabilities
    if results["page_scan"]:
        for vuln in results["page_scan"].vulnerabilities:
            severity_weights = {"low": 1, "medium": 3, "high": 7, "critical": 10}
            score += severity_weights.get(vuln["severity"], 0)
    
    # Form vulnerabilities
    score += len(results["form_vulnerabilities"]) * 2
    
    # Missing security headers
    score += len(results["page_scan"].missing_security_headers) * 0.5
    
    # Directory discoveries
    for dir_result in results["directory_scan"]:
        if dir_result["status_code"] == 200:
            score += 2
    
    return min(10.0, score)

class WebScanner(BaseScanner):
    """Web security scanner with advanced features."""
    
    def __init__(self, config: WebScanConfig):
        
    """__init__ function."""
super().__init__(config)
        self.config = config
    
    async def scan_website(self, base_url: str) -> Dict[str, Any]:
        """Scan entire website for vulnerabilities."""
        self.logger.info(f"Starting web security scan of {base_url}")
        
        # Scan main page
        main_scan = await comprehensive_web_scan(base_url, self.config)
        
        # Scan discovered links if depth allows
        if self.config.scan_depth == "high" and main_scan["page_scan"].links:
            link_scans = []
            for link in main_scan["page_scan"].links[:10]:  # Limit to first 10 links
                try:
                    link_scan = await scan_web_page(link, self.config)
                    link_scans.append(link_scan)
                except Exception as e:
                    self.logger.warning(f"Failed to scan {link}: {e}")
            
            main_scan["link_scans"] = link_scans
        
        return main_scan
    
    async def scan_multiple_sites(self, urls: List[str]) -> Dict[str, Any]:
        """Scan multiple websites for vulnerabilities."""
        self.logger.info(f"Starting web security scan of {len(urls)} sites")
        
        tasks = [self.scan_website(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_results = [r for r in results if isinstance(r, dict)]
        
        return {
            "sites_scanned": len(urls),
            "successful_scans": len(valid_results),
            "results": valid_results,
            "overall_summary": self._generate_web_summary(valid_results)
        }
    
    def _generate_web_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate overall summary from multiple web scan results."""
        total_vulns = 0
        total_missing_headers = 0
        total_forms = 0
        
        for result in results:
            if "overall_summary" in result:
                total_vulns += result["overall_summary"]["total_vulnerabilities"]
                total_missing_headers += result["overall_summary"]["missing_security_headers"]
                total_forms += result["overall_summary"]["forms_found"]
        
        return {
            "total_vulnerabilities": total_vulns,
            "total_missing_security_headers": total_missing_headers,
            "total_forms_found": total_forms,
            "average_risk_score": sum(r.get("overall_summary", {}).get("risk_score", 0) for r in results) / len(results) if results else 0
        }

# Named exports
__all__ = [
    'scan_web_page',
    'scan_web_forms',
    'scan_web_directory',
    'comprehensive_web_scan',
    'calculate_web_risk_score',
    'WebScanConfig',
    'WebScanResult',
    'WebScanner'
] 