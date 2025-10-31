#!/usr/bin/env python3
"""
Web Scanner Module for Video-OpusClip
Scans web applications for security issues and information gathering
"""

import asyncio
import aiohttp
import re
import json
import hashlib
import ssl
import socket
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
from urllib.parse import urljoin, urlparse, parse_qs
import subprocess
import dns.resolver
import whois

class ScanCategory(str, Enum):
    """Categories of web scans"""
    INFORMATION_GATHERING = "information_gathering"
    DIRECTORY_ENUMERATION = "directory_enumeration"
    TECHNOLOGY_DETECTION = "technology_detection"
    SECURITY_HEADERS = "security_headers"
    CONTENT_ANALYSIS = "content_analysis"
    API_DISCOVERY = "api_discovery"

class HeaderType(str, Enum):
    """Types of security headers"""
    CONTENT_SECURITY_POLICY = "content-security-policy"
    X_FRAME_OPTIONS = "x-frame-options"
    X_CONTENT_TYPE_OPTIONS = "x-content-type-options"
    X_XSS_PROTECTION = "x-xss-protection"
    STRICT_TRANSPORT_SECURITY = "strict-transport-security"
    REFERRER_POLICY = "referrer-policy"

@dataclass
class WebScanResult:
    """Result of a web scan"""
    category: ScanCategory
    title: str
    description: str
    url: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    severity: str = "info"
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

@dataclass
class ScanConfig:
    """Configuration for web scanning"""
    target_url: str
    scan_categories: List[ScanCategory] = None
    max_concurrent: int = 10
    timeout: float = 30.0
    user_agent: str = "Video-OpusClip-WebScanner/1.0"
    follow_redirects: bool = True
    verify_ssl: bool = False
    custom_headers: Dict[str, str] = None
    wordlist_path: Optional[str] = None
    
    def __post_init__(self):
        if self.scan_categories is None:
            self.scan_categories = [
                ScanCategory.INFORMATION_GATHERING,
                ScanCategory.TECHNOLOGY_DETECTION,
                ScanCategory.SECURITY_HEADERS,
                ScanCategory.CONTENT_ANALYSIS
            ]
        if self.custom_headers is None:
            self.custom_headers = {}

class WebScanner:
    """Web application scanner for security assessment"""
    
    def __init__(self, config: ScanConfig):
        self.config = config
        self.results: List[WebScanResult] = []
        self.session: Optional[aiohttp.ClientSession] = None
        self.scan_start_time: float = 0.0
        self.scan_end_time: float = 0.0
        
        # Common wordlists for directory enumeration
        self.common_directories = [
            "admin", "administrator", "manage", "management",
            "api", "rest", "graphql", "swagger", "docs",
            "backup", "backups", "db", "database", "config",
            "logs", "log", "debug", "test", "dev", "stage",
            "assets", "static", "media", "uploads", "files",
            "phpinfo.php", "info.php", "server-status",
            ".env", ".git", ".svn", ".htaccess", "robots.txt",
            "sitemap.xml", "crossdomain.xml", "security.txt"
        ]
    
    async def scan_website(self) -> Dict[str, Any]:
        """Perform comprehensive web scan"""
        self.scan_start_time = asyncio.get_event_loop().time()
        
        try:
            # Initialize session
            connector = aiohttp.TCPConnector(ssl=False if not self.config.verify_ssl else None)
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout),
                headers={
                    "User-Agent": self.config.user_agent,
                    **self.config.custom_headers
                }
            )
            
            # Perform scans based on categories
            if ScanCategory.INFORMATION_GATHERING in self.config.scan_categories:
                await self._information_gathering()
            
            if ScanCategory.DIRECTORY_ENUMERATION in self.config.scan_categories:
                await self._directory_enumeration()
            
            if ScanCategory.TECHNOLOGY_DETECTION in self.config.scan_categories:
                await self._technology_detection()
            
            if ScanCategory.SECURITY_HEADERS in self.config.scan_categories:
                await self._security_headers_scan()
            
            if ScanCategory.CONTENT_ANALYSIS in self.config.scan_categories:
                await self._content_analysis()
            
            if ScanCategory.API_DISCOVERY in self.config.scan_categories:
                await self._api_discovery()
            
            self.scan_end_time = asyncio.get_event_loop().time()
            
            return {
                "success": True,
                "target": self.config.target_url,
                "total_findings": len(self.results),
                "scan_duration": self.scan_end_time - self.scan_start_time,
                "results": [self._result_to_dict(r) for r in self.results]
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "target": self.config.target_url
            }
        finally:
            if self.session:
                await self.session.close()
    
    async def _information_gathering(self) -> None:
        """Gather information about the target"""
        try:
            parsed_url = urlparse(self.config.target_url)
            domain = parsed_url.netloc
            
            # DNS information
            dns_info = await self._get_dns_info(domain)
            if dns_info:
                self.results.append(WebScanResult(
                    category=ScanCategory.INFORMATION_GATHERING,
                    title="DNS Information",
                    description="DNS records for the target domain",
                    url=self.config.target_url,
                    data=dns_info
                ))
            
            # WHOIS information
            whois_info = await self._get_whois_info(domain)
            if whois_info:
                self.results.append(WebScanResult(
                    category=ScanCategory.INFORMATION_GATHERING,
                    title="WHOIS Information",
                    description="WHOIS data for the target domain",
                    url=self.config.target_url,
                    data=whois_info
                ))
            
            # SSL certificate information
            ssl_info = await self._get_ssl_info(domain)
            if ssl_info:
                self.results.append(WebScanResult(
                    category=ScanCategory.INFORMATION_GATHERING,
                    title="SSL Certificate Information",
                    description="SSL certificate details",
                    url=self.config.target_url,
                    data=ssl_info
                ))
            
            # Server information
            server_info = await self._get_server_info()
            if server_info:
                self.results.append(WebScanResult(
                    category=ScanCategory.INFORMATION_GATHERING,
                    title="Server Information",
                    description="Web server details",
                    url=self.config.target_url,
                    data=server_info
                ))
                
        except Exception as e:
            self.results.append(WebScanResult(
                category=ScanCategory.INFORMATION_GATHERING,
                title="Information Gathering Error",
                description=f"Error during information gathering: {str(e)}",
                severity="error"
            ))
    
    async def _directory_enumeration(self) -> None:
        """Enumerate directories and files"""
        try:
            discovered_directories = []
            
            # Use semaphore for concurrency control
            semaphore = asyncio.Semaphore(self.config.max_concurrent)
            
            tasks = [
                self._check_directory(directory, semaphore)
                for directory in self.common_directories
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, dict) and result.get("found"):
                    discovered_directories.append(result)
            
            if discovered_directories:
                self.results.append(WebScanResult(
                    category=ScanCategory.DIRECTORY_ENUMERATION,
                    title="Directory Enumeration",
                    description=f"Found {len(discovered_directories)} accessible directories",
                    url=self.config.target_url,
                    data={"directories": discovered_directories}
                ))
                
        except Exception as e:
            self.results.append(WebScanResult(
                category=ScanCategory.DIRECTORY_ENUMERATION,
                title="Directory Enumeration Error",
                description=f"Error during directory enumeration: {str(e)}",
                severity="error"
            ))
    
    async def _check_directory(self, directory: str, semaphore: asyncio.Semaphore) -> Dict[str, Any]:
        """Check if a directory exists"""
        async with semaphore:
            try:
                url = urljoin(self.config.target_url, directory)
                async with self.session.get(url) as response:
                    if response.status == 200:
                        return {
                            "found": True,
                            "directory": directory,
                            "url": url,
                            "status_code": response.status,
                            "content_length": len(await response.text())
                        }
                    elif response.status in [301, 302, 403]:
                        return {
                            "found": True,
                            "directory": directory,
                            "url": url,
                            "status_code": response.status,
                            "redirect": response.headers.get("Location")
                        }
                    else:
                        return {"found": False, "directory": directory}
                        
            except Exception:
                return {"found": False, "directory": directory}
    
    async def _technology_detection(self) -> None:
        """Detect technologies used by the website"""
        try:
            technologies = {}
            
            # Get main page content
            async with self.session.get(self.config.target_url) as response:
                content = await response.text()
                headers = dict(response.headers)
            
            # Detect technologies from headers
            tech_from_headers = self._detect_tech_from_headers(headers)
            technologies.update(tech_from_headers)
            
            # Detect technologies from content
            tech_from_content = self._detect_tech_from_content(content)
            technologies.update(tech_from_content)
            
            # Detect technologies from JavaScript
            js_technologies = await self._detect_js_technologies()
            technologies.update(js_technologies)
            
            if technologies:
                self.results.append(WebScanResult(
                    category=ScanCategory.TECHNOLOGY_DETECTION,
                    title="Technology Detection",
                    description="Detected technologies and frameworks",
                    url=self.config.target_url,
                    data={"technologies": technologies}
                ))
                
        except Exception as e:
            self.results.append(WebScanResult(
                category=ScanCategory.TECHNOLOGY_DETECTION,
                title="Technology Detection Error",
                description=f"Error during technology detection: {str(e)}",
                severity="error"
            ))
    
    def _detect_tech_from_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Detect technologies from HTTP headers"""
        technologies = {}
        
        # Server header
        if "Server" in headers:
            server = headers["Server"]
            if "nginx" in server.lower():
                technologies["Web Server"] = "Nginx"
            elif "apache" in server.lower():
                technologies["Web Server"] = "Apache"
            elif "iis" in server.lower():
                technologies["Web Server"] = "IIS"
            else:
                technologies["Web Server"] = server
        
        # X-Powered-By header
        if "X-Powered-By" in headers:
            powered_by = headers["X-Powered-By"]
            if "php" in powered_by.lower():
                technologies["Backend"] = "PHP"
            elif "asp.net" in powered_by.lower():
                technologies["Backend"] = "ASP.NET"
            else:
                technologies["Backend"] = powered_by
        
        # Content-Type header
        if "Content-Type" in headers:
            content_type = headers["Content-Type"]
            if "text/html" in content_type:
                technologies["Content Type"] = "HTML"
            elif "application/json" in content_type:
                technologies["Content Type"] = "JSON"
            elif "application/xml" in content_type:
                technologies["Content Type"] = "XML"
        
        return technologies
    
    def _detect_tech_from_content(self, content: str) -> Dict[str, str]:
        """Detect technologies from HTML content"""
        technologies = {}
        
        # Framework detection
        if "wp-content" in content or "wp-includes" in content:
            technologies["CMS"] = "WordPress"
        elif "drupal" in content.lower():
            technologies["CMS"] = "Drupal"
        elif "joomla" in content.lower():
            technologies["CMS"] = "Joomla"
        
        # JavaScript frameworks
        if "react" in content.lower() or "reactjs" in content.lower():
            technologies["Frontend Framework"] = "React"
        elif "angular" in content.lower():
            technologies["Frontend Framework"] = "Angular"
        elif "vue" in content.lower():
            technologies["Frontend Framework"] = "Vue.js"
        elif "jquery" in content.lower():
            technologies["JavaScript Library"] = "jQuery"
        
        # CSS frameworks
        if "bootstrap" in content.lower():
            technologies["CSS Framework"] = "Bootstrap"
        elif "foundation" in content.lower():
            technologies["CSS Framework"] = "Foundation"
        elif "materialize" in content.lower():
            technologies["CSS Framework"] = "Materialize"
        
        # Analytics and tracking
        if "google-analytics" in content.lower() or "gtag" in content.lower():
            technologies["Analytics"] = "Google Analytics"
        elif "facebook" in content.lower() and "pixel" in content.lower():
            technologies["Analytics"] = "Facebook Pixel"
        
        return technologies
    
    async def _detect_js_technologies(self) -> Dict[str, str]:
        """Detect technologies from JavaScript files"""
        technologies = {}
        
        try:
            # Look for common JavaScript files
            js_files = [
                "/js/app.js", "/js/main.js", "/static/js/app.js",
                "/assets/js/app.js", "/dist/js/app.js"
            ]
            
            for js_file in js_files:
                url = urljoin(self.config.target_url, js_file)
                try:
                    async with self.session.get(url) as response:
                        if response.status == 200:
                            content = await response.text()
                            
                            # Detect frameworks in JS content
                            if "react" in content.lower():
                                technologies["JavaScript Framework"] = "React"
                            elif "angular" in content.lower():
                                technologies["JavaScript Framework"] = "Angular"
                            elif "vue" in content.lower():
                                technologies["JavaScript Framework"] = "Vue.js"
                            elif "jquery" in content.lower():
                                technologies["JavaScript Library"] = "jQuery"
                            
                            break
                            
                except Exception:
                    continue
                    
        except Exception:
            pass
        
        return technologies
    
    async def _security_headers_scan(self) -> None:
        """Scan for security headers"""
        try:
            async with self.session.get(self.config.target_url) as response:
                headers = dict(response.headers)
            
            security_headers = {}
            missing_headers = []
            
            # Check for security headers
            header_checks = {
                HeaderType.CONTENT_SECURITY_POLICY: "Content Security Policy",
                HeaderType.X_FRAME_OPTIONS: "X-Frame-Options",
                HeaderType.X_CONTENT_TYPE_OPTIONS: "X-Content-Type-Options",
                HeaderType.X_XSS_PROTECTION: "X-XSS-Protection",
                HeaderType.STRICT_TRANSPORT_SECURITY: "Strict-Transport-Security",
                HeaderType.REFERRER_POLICY: "Referrer-Policy"
            }
            
            for header_name, display_name in header_checks.items():
                if header_name.value in headers:
                    security_headers[display_name] = headers[header_name.value]
                else:
                    missing_headers.append(display_name)
            
            self.results.append(WebScanResult(
                category=ScanCategory.SECURITY_HEADERS,
                title="Security Headers Analysis",
                description=f"Found {len(security_headers)} security headers, missing {len(missing_headers)}",
                url=self.config.target_url,
                data={
                    "present_headers": security_headers,
                    "missing_headers": missing_headers
                },
                severity="medium" if missing_headers else "low"
            ))
            
        except Exception as e:
            self.results.append(WebScanResult(
                category=ScanCategory.SECURITY_HEADERS,
                title="Security Headers Scan Error",
                description=f"Error during security headers scan: {str(e)}",
                severity="error"
            ))
    
    async def _content_analysis(self) -> None:
        """Analyze website content"""
        try:
            async with self.session.get(self.config.target_url) as response:
                content = await response.text()
            
            analysis = {}
            
            # Email addresses
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            emails = re.findall(email_pattern, content)
            if emails:
                analysis["email_addresses"] = list(set(emails))
            
            # Phone numbers
            phone_pattern = r'(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
            phones = re.findall(phone_pattern, content)
            if phones:
                analysis["phone_numbers"] = list(set(phones))
            
            # Social media links
            social_patterns = [
                r'https?://(?:www\.)?facebook\.com/[^\s"<>]+',
                r'https?://(?:www\.)?twitter\.com/[^\s"<>]+',
                r'https?://(?:www\.)?linkedin\.com/[^\s"<>]+',
                r'https?://(?:www\.)?instagram\.com/[^\s"<>]+'
            ]
            
            social_links = []
            for pattern in social_patterns:
                matches = re.findall(pattern, content)
                social_links.extend(matches)
            
            if social_links:
                analysis["social_media_links"] = list(set(social_links))
            
            # Comments in HTML
            comment_pattern = r'<!--(.*?)-->'
            comments = re.findall(comment_pattern, content, re.DOTALL)
            if comments:
                analysis["html_comments"] = [comment.strip() for comment in comments]
            
            # JavaScript variables
            js_var_pattern = r'var\s+(\w+)\s*=\s*["\']([^"\']+)["\']'
            js_vars = re.findall(js_var_pattern, content)
            if js_vars:
                analysis["javascript_variables"] = dict(js_vars)
            
            if analysis:
                self.results.append(WebScanResult(
                    category=ScanCategory.CONTENT_ANALYSIS,
                    title="Content Analysis",
                    description="Analysis of website content",
                    url=self.config.target_url,
                    data=analysis
                ))
                
        except Exception as e:
            self.results.append(WebScanResult(
                category=ScanCategory.CONTENT_ANALYSIS,
                title="Content Analysis Error",
                description=f"Error during content analysis: {str(e)}",
                severity="error"
            ))
    
    async def _api_discovery(self) -> None:
        """Discover API endpoints"""
        try:
            discovered_apis = []
            
            # Common API endpoints
            api_endpoints = [
                "/api", "/api/", "/rest", "/rest/", "/graphql",
                "/swagger", "/swagger-ui", "/docs", "/documentation",
                "/v1", "/v2", "/v3", "/beta", "/alpha"
            ]
            
            for endpoint in api_endpoints:
                url = urljoin(self.config.target_url, endpoint)
                try:
                    async with self.session.get(url) as response:
                        if response.status == 200:
                            content = await response.text()
                            
                            api_info = {
                                "endpoint": endpoint,
                                "url": url,
                                "status_code": response.status,
                                "content_type": response.headers.get("Content-Type", ""),
                                "content_length": len(content)
                            }
                            
                            # Try to detect API type
                            if "swagger" in content.lower():
                                api_info["type"] = "Swagger/OpenAPI"
                            elif "graphql" in content.lower():
                                api_info["type"] = "GraphQL"
                            elif "json" in response.headers.get("Content-Type", "").lower():
                                api_info["type"] = "REST API"
                            else:
                                api_info["type"] = "Unknown API"
                            
                            discovered_apis.append(api_info)
                            
                except Exception:
                    continue
            
            if discovered_apis:
                self.results.append(WebScanResult(
                    category=ScanCategory.API_DISCOVERY,
                    title="API Discovery",
                    description=f"Discovered {len(discovered_apis)} API endpoints",
                    url=self.config.target_url,
                    data={"apis": discovered_apis}
                ))
                
        except Exception as e:
            self.results.append(WebScanResult(
                category=ScanCategory.API_DISCOVERY,
                title="API Discovery Error",
                description=f"Error during API discovery: {str(e)}",
                severity="error"
            ))
    
    async def _get_dns_info(self, domain: str) -> Optional[Dict[str, Any]]:
        """Get DNS information for domain"""
        try:
            dns_info = {}
            
            # A records
            try:
                a_records = dns.resolver.resolve(domain, 'A')
                dns_info["A"] = [str(record) for record in a_records]
            except Exception:
                pass
            
            # AAAA records
            try:
                aaaa_records = dns.resolver.resolve(domain, 'AAAA')
                dns_info["AAAA"] = [str(record) for record in aaaa_records]
            except Exception:
                pass
            
            # MX records
            try:
                mx_records = dns.resolver.resolve(domain, 'MX')
                dns_info["MX"] = [str(record.exchange) for record in mx_records]
            except Exception:
                pass
            
            # NS records
            try:
                ns_records = dns.resolver.resolve(domain, 'NS')
                dns_info["NS"] = [str(record) for record in ns_records]
            except Exception:
                pass
            
            # TXT records
            try:
                txt_records = dns.resolver.resolve(domain, 'TXT')
                dns_info["TXT"] = [str(record) for record in txt_records]
            except Exception:
                pass
            
            return dns_info if dns_info else None
            
        except Exception:
            return None
    
    async def _get_whois_info(self, domain: str) -> Optional[Dict[str, Any]]:
        """Get WHOIS information for domain"""
        try:
            w = whois.whois(domain)
            return {
                "registrar": w.registrar,
                "creation_date": str(w.creation_date),
                "expiration_date": str(w.expiration_date),
                "updated_date": str(w.updated_date),
                "status": w.status
            }
        except Exception:
            return None
    
    async def _get_ssl_info(self, domain: str) -> Optional[Dict[str, Any]]:
        """Get SSL certificate information"""
        try:
            context = ssl.create_default_context()
            with socket.create_connection((domain, 443), timeout=10) as sock:
                with context.wrap_socket(sock, server_hostname=domain) as ssock:
                    cert = ssock.getpeercert()
                    
                    return {
                        "subject": dict(x[0] for x in cert['subject']),
                        "issuer": dict(x[0] for x in cert['issuer']),
                        "version": cert['version'],
                        "serial_number": cert['serialNumber'],
                        "not_before": cert['notBefore'],
                        "not_after": cert['notAfter'],
                        "cipher": ssock.cipher()
                    }
        except Exception:
            return None
    
    async def _get_server_info(self) -> Optional[Dict[str, Any]]:
        """Get server information"""
        try:
            async with self.session.get(self.config.target_url) as response:
                headers = dict(response.headers)
                
                return {
                    "server": headers.get("Server"),
                    "x_powered_by": headers.get("X-Powered-By"),
                    "content_type": headers.get("Content-Type"),
                    "content_length": headers.get("Content-Length"),
                    "last_modified": headers.get("Last-Modified"),
                    "etag": headers.get("ETag")
                }
        except Exception:
            return None
    
    def _result_to_dict(self, result: WebScanResult) -> Dict[str, Any]:
        """Convert WebScanResult to dictionary"""
        return {
            "category": result.category.value,
            "title": result.title,
            "description": result.description,
            "url": result.url,
            "data": result.data,
            "severity": result.severity,
            "timestamp": result.timestamp.isoformat() if result.timestamp else None
        }
    
    def get_results_by_category(self, category: ScanCategory) -> List[WebScanResult]:
        """Get results by category"""
        return [r for r in self.results if r.category == category]
    
    def generate_report(self) -> str:
        """Generate web scan report"""
        report = f"Web Scan Report for {self.config.target_url}\n"
        report += "=" * 60 + "\n"
        report += f"Scan Duration: {self.scan_end_time - self.scan_start_time:.2f} seconds\n"
        report += f"Total Findings: {len(self.results)}\n\n"
        
        # Group by category
        for category in ScanCategory:
            category_results = self.get_results_by_category(category)
            if category_results:
                report += f"{category.value.replace('_', ' ').title()} ({len(category_results)}):\n"
                report += "-" * 40 + "\n"
                for result in category_results:
                    report += f"• {result.title}\n"
                    report += f"  Description: {result.description}\n"
                    if result.url:
                        report += f"  URL: {result.url}\n"
                    if result.data:
                        report += f"  Data: {json.dumps(result.data, indent=2)}\n"
                    report += "\n"
        
        return report

# Example usage
async def main():
    """Example usage of web scanner"""
    print("🌐 Web Scanner Example")
    
    # Create scan configuration
    config = ScanConfig(
        target_url="http://localhost:8000",
        scan_categories=[
            ScanCategory.INFORMATION_GATHERING,
            ScanCategory.TECHNOLOGY_DETECTION,
            ScanCategory.SECURITY_HEADERS,
            ScanCategory.CONTENT_ANALYSIS,
            ScanCategory.API_DISCOVERY
        ],
        max_concurrent=5,
        timeout=10.0
    )
    
    # Create scanner
    scanner = WebScanner(config)
    
    # Perform scan
    print(f"Scanning {config.target_url}...")
    result = await scanner.scan_website()
    
    if result["success"]:
        print(f"✅ Scan completed in {result['scan_duration']:.2f} seconds")
        print(f"📊 Found {result['total_findings']} findings")
        
        # Print results by category
        for category in ScanCategory:
            category_results = scanner.get_results_by_category(category)
            if category_results:
                print(f"\n📋 {category.value.replace('_', ' ').title()} ({len(category_results)}):")
                for scan_result in category_results:
                    print(f"  • {scan_result.title}")
                    print(f"    {scan_result.description}")
        
        # Generate report
        print("\n📋 Web Scan Report:")
        print(scanner.generate_report())
        
    else:
        print(f"❌ Scan failed: {result['error']}")

if __name__ == "__main__":
    asyncio.run(main()) 