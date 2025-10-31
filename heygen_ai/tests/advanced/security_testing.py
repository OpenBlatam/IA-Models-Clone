"""
Advanced Security Testing Framework for HeyGen AI system.
Comprehensive security testing including vulnerability scanning, penetration testing,
and security compliance validation.
"""

import json
import time
import hashlib
import secrets
import subprocess
import requests
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import logging
import re
import base64
import hmac
import sqlite3
from urllib.parse import urljoin, urlparse
import socket
import ssl
import threading
import concurrent.futures

@dataclass
class SecurityVulnerability:
    """Represents a security vulnerability."""
    vulnerability_id: str
    title: str
    severity: str  # critical, high, medium, low, info
    description: str
    cwe_id: Optional[str] = None
    cvss_score: Optional[float] = None
    affected_component: str = ""
    remediation: str = ""
    references: List[str] = field(default_factory=list)
    discovered_at: datetime = field(default_factory=datetime.now)

@dataclass
class SecurityTestResult:
    """Result of a security test."""
    test_name: str
    status: str  # passed, failed, warning, error
    vulnerabilities: List[SecurityVulnerability] = field(default_factory=list)
    score: float = 0.0
    recommendations: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

class VulnerabilityScanner:
    """Scans for common security vulnerabilities."""
    
    def __init__(self):
        self.vulnerability_patterns = {
            "sql_injection": [
                r"'.*OR.*'.*'",
                r"'.*UNION.*SELECT.*",
                r"'.*DROP.*TABLE.*",
                r"'.*DELETE.*FROM.*",
                r"'.*INSERT.*INTO.*"
            ],
            "xss": [
                r"<script.*>.*</script>",
                r"javascript:",
                r"on\w+\s*=",
                r"<iframe.*>",
                r"<object.*>"
            ],
            "path_traversal": [
                r"\.\./",
                r"\.\.\\",
                r"%2e%2e%2f",
                r"%2e%2e%5c",
                r"\.\.%2f"
            ],
            "command_injection": [
                r";\s*\w+",
                r"\|\s*\w+",
                r"&\s*\w+",
                r"`.*`",
                r"\$\(.*\)"
            ],
            "hardcoded_secrets": [
                r"password\s*=\s*['\"][^'\"]+['\"]",
                r"api_key\s*=\s*['\"][^'\"]+['\"]",
                r"secret\s*=\s*['\"][^'\"]+['\"]",
                r"token\s*=\s*['\"][^'\"]+['\"]"
            ]
        }
    
    def scan_code_file(self, file_path: Path) -> List[SecurityVulnerability]:
        """Scan a code file for vulnerabilities."""
        vulnerabilities = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.split('\n')
            
            for vuln_type, patterns in self.vulnerability_patterns.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1
                        
                        vuln = SecurityVulnerability(
                            vulnerability_id=f"{vuln_type}_{hash(match.group())}",
                            title=f"Potential {vuln_type.replace('_', ' ').title()}",
                            severity=self._get_severity(vuln_type),
                            description=f"Found {vuln_type} pattern in {file_path.name} at line {line_num}",
                            affected_component=str(file_path),
                            remediation=self._get_remediation(vuln_type)
                        )
                        vulnerabilities.append(vuln)
        
        except Exception as e:
            logging.error(f"Error scanning file {file_path}: {e}")
        
        return vulnerabilities
    
    def _get_severity(self, vuln_type: str) -> str:
        """Get severity level for vulnerability type."""
        severity_map = {
            "sql_injection": "critical",
            "xss": "high",
            "path_traversal": "high",
            "command_injection": "critical",
            "hardcoded_secrets": "high"
        }
        return severity_map.get(vuln_type, "medium")
    
    def _get_remediation(self, vuln_type: str) -> str:
        """Get remediation advice for vulnerability type."""
        remediation_map = {
            "sql_injection": "Use parameterized queries or prepared statements",
            "xss": "Implement proper input validation and output encoding",
            "path_traversal": "Validate and sanitize file paths",
            "command_injection": "Avoid executing user input as commands",
            "hardcoded_secrets": "Use environment variables or secure secret management"
        }
        return remediation_map.get(vuln_type, "Review and fix the identified issue")

class AuthenticationTester:
    """Tests authentication and authorization mechanisms."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def test_weak_passwords(self, username: str, password_list: List[str]) -> SecurityTestResult:
        """Test for weak password vulnerabilities."""
        start_time = time.time()
        vulnerabilities = []
        
        weak_passwords = [
            "password", "123456", "admin", "test", "guest",
            "root", "user", "qwerty", "abc123", "password123"
        ]
        
        for weak_pass in weak_passwords:
            if weak_pass in password_list:
                vuln = SecurityVulnerability(
                    vulnerability_id=f"weak_password_{hash(weak_pass)}",
                    title="Weak Password Detected",
                    severity="high",
                    description=f"Weak password '{weak_pass}' found in password list",
                    remediation="Implement strong password requirements"
                )
                vulnerabilities.append(vuln)
        
        execution_time = time.time() - start_time
        score = max(0, 100 - len(vulnerabilities) * 20)
        
        return SecurityTestResult(
            test_name="weak_password_test",
            status="failed" if vulnerabilities else "passed",
            vulnerabilities=vulnerabilities,
            score=score,
            execution_time=execution_time
        )
    
    def test_brute_force_protection(self, login_endpoint: str, username: str, 
                                  password_list: List[str]) -> SecurityTestResult:
        """Test brute force protection mechanisms."""
        start_time = time.time()
        vulnerabilities = []
        failed_attempts = 0
        
        for password in password_list[:10]:  # Test first 10 passwords
            try:
                response = self.session.post(
                    urljoin(self.base_url, login_endpoint),
                    json={"username": username, "password": password},
                    timeout=5
                )
                
                if response.status_code != 200:
                    failed_attempts += 1
                
                # Check for rate limiting
                if "rate" in response.headers.get("retry-after", "").lower():
                    break  # Rate limiting is working
                
            except Exception as e:
                logging.error(f"Error testing brute force: {e}")
        
        if failed_attempts >= 5 and "retry-after" not in str(self.session.cookies):
            vuln = SecurityVulnerability(
                vulnerability_id="brute_force_vulnerability",
                title="Brute Force Protection Missing",
                severity="high",
                description="No rate limiting detected after multiple failed attempts",
                remediation="Implement rate limiting and account lockout mechanisms"
            )
            vulnerabilities.append(vuln)
        
        execution_time = time.time() - start_time
        score = max(0, 100 - len(vulnerabilities) * 30)
        
        return SecurityTestResult(
            test_name="brute_force_protection_test",
            status="failed" if vulnerabilities else "passed",
            vulnerabilities=vulnerabilities,
            score=score,
            execution_time=execution_time
        )
    
    def test_session_management(self, session_cookie: str) -> SecurityTestResult:
        """Test session management security."""
        start_time = time.time()
        vulnerabilities = []
        
        # Check session cookie attributes
        if not session_cookie:
            vuln = SecurityVulnerability(
                vulnerability_id="no_session_cookie",
                title="No Session Cookie",
                severity="medium",
                description="No session cookie found",
                remediation="Implement proper session management"
            )
            vulnerabilities.append(vuln)
        else:
            # Check for secure attributes
            if "Secure" not in session_cookie:
                vuln = SecurityVulnerability(
                    vulnerability_id="insecure_session_cookie",
                    title="Insecure Session Cookie",
                    severity="medium",
                    description="Session cookie missing Secure flag",
                    remediation="Add Secure flag to session cookies"
                )
                vulnerabilities.append(vuln)
            
            if "HttpOnly" not in session_cookie:
                vuln = SecurityVulnerability(
                    vulnerability_id="httponly_session_cookie",
                    title="Session Cookie Missing HttpOnly",
                    severity="medium",
                    description="Session cookie missing HttpOnly flag",
                    remediation="Add HttpOnly flag to session cookies"
                )
                vulnerabilities.append(vuln)
        
        execution_time = time.time() - start_time
        score = max(0, 100 - len(vulnerabilities) * 25)
        
        return SecurityTestResult(
            test_name="session_management_test",
            status="failed" if vulnerabilities else "passed",
            vulnerabilities=vulnerabilities,
            score=score,
            execution_time=execution_time
        )

class NetworkSecurityTester:
    """Tests network security configurations."""
    
    def __init__(self, target_host: str = "localhost", target_port: int = 8000):
        self.target_host = target_host
        self.target_port = target_port
    
    def test_ssl_configuration(self) -> SecurityTestResult:
        """Test SSL/TLS configuration."""
        start_time = time.time()
        vulnerabilities = []
        
        try:
            context = ssl.create_default_context()
            with socket.create_connection((self.target_host, self.target_port), timeout=10) as sock:
                with context.wrap_socket(sock, server_hostname=self.target_host) as ssock:
                    cert = ssock.getpeercert()
                    cipher = ssock.cipher()
                    
                    # Check certificate validity
                    if not cert:
                        vuln = SecurityVulnerability(
                            vulnerability_id="no_ssl_certificate",
                            title="No SSL Certificate",
                            severity="critical",
                            description="No SSL certificate found",
                            remediation="Implement proper SSL certificate"
                        )
                        vulnerabilities.append(vuln)
                    
                    # Check cipher strength
                    if cipher and cipher[2] < 128:  # Key length
                        vuln = SecurityVulnerability(
                            vulnerability_id="weak_cipher",
                            title="Weak SSL Cipher",
                            severity="high",
                            description=f"Weak cipher detected: {cipher[0]}",
                            remediation="Use strong encryption ciphers"
                        )
                        vulnerabilities.append(vuln)
        
        except Exception as e:
            vuln = SecurityVulnerability(
                vulnerability_id="ssl_connection_failed",
                title="SSL Connection Failed",
                severity="high",
                description=f"SSL connection failed: {str(e)}",
                remediation="Fix SSL configuration"
            )
            vulnerabilities.append(vuln)
        
        execution_time = time.time() - start_time
        score = max(0, 100 - len(vulnerabilities) * 30)
        
        return SecurityTestResult(
            test_name="ssl_configuration_test",
            status="failed" if vulnerabilities else "passed",
            vulnerabilities=vulnerabilities,
            score=score,
            execution_time=execution_time
        )
    
    def test_port_scanning(self, ports: List[int] = None) -> SecurityTestResult:
        """Test for open ports and services."""
        if ports is None:
            ports = [21, 22, 23, 25, 53, 80, 110, 143, 443, 993, 995, 3389, 5432, 3306]
        
        start_time = time.time()
        vulnerabilities = []
        open_ports = []
        
        def scan_port(port):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex((self.target_host, port))
                sock.close()
                return port if result == 0 else None
            except:
                return None
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(scan_port, port) for port in ports]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result:
                    open_ports.append(result)
        
        # Check for potentially dangerous open ports
        dangerous_ports = {
            21: "FTP (unencrypted)",
            23: "Telnet (unencrypted)",
            3389: "RDP (Remote Desktop)",
            5432: "PostgreSQL",
            3306: "MySQL"
        }
        
        for port in open_ports:
            if port in dangerous_ports:
                vuln = SecurityVulnerability(
                    vulnerability_id=f"dangerous_port_{port}",
                    title=f"Dangerous Port Open: {port}",
                    severity="medium",
                    description=f"Port {port} ({dangerous_ports[port]}) is open",
                    remediation="Close unnecessary ports or secure them properly"
                )
                vulnerabilities.append(vuln)
        
        execution_time = time.time() - start_time
        score = max(0, 100 - len(vulnerabilities) * 15)
        
        return SecurityTestResult(
            test_name="port_scanning_test",
            status="failed" if vulnerabilities else "passed",
            vulnerabilities=vulnerabilities,
            score=score,
            execution_time=execution_time
        )

class DataSecurityTester:
    """Tests data security and privacy compliance."""
    
    def __init__(self, db_path: str = "test_database.db"):
        self.db_path = db_path
    
    def test_data_encryption(self) -> SecurityTestResult:
        """Test if sensitive data is encrypted."""
        start_time = time.time()
        vulnerabilities = []
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get table schema
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            for table in tables:
                table_name = table[0]
                cursor.execute(f"PRAGMA table_info({table_name});")
                columns = cursor.fetchall()
                
                for column in columns:
                    column_name = column[1].lower()
                    if any(keyword in column_name for keyword in ['password', 'secret', 'key', 'token', 'ssn', 'credit']):
                        # Check if data looks encrypted (basic check)
                        cursor.execute(f"SELECT {column[1]} FROM {table_name} LIMIT 1;")
                        sample_data = cursor.fetchone()
                        
                        if sample_data and sample_data[0]:
                            data = str(sample_data[0])
                            # Simple check: encrypted data should look random
                            if len(data) < 32 or data.isalnum():
                                vuln = SecurityVulnerability(
                                    vulnerability_id=f"unencrypted_data_{table_name}_{column[1]}",
                                    title="Unencrypted Sensitive Data",
                                    severity="high",
                                    description=f"Column {column[1]} in table {table_name} appears to contain unencrypted sensitive data",
                                    remediation="Encrypt sensitive data at rest"
                                )
                                vulnerabilities.append(vuln)
            
            conn.close()
        
        except Exception as e:
            logging.error(f"Error testing data encryption: {e}")
        
        execution_time = time.time() - start_time
        score = max(0, 100 - len(vulnerabilities) * 25)
        
        return SecurityTestResult(
            test_name="data_encryption_test",
            status="failed" if vulnerabilities else "passed",
            vulnerabilities=vulnerabilities,
            score=score,
            execution_time=execution_time
        )
    
    def test_sql_injection_protection(self, test_queries: List[str] = None) -> SecurityTestResult:
        """Test SQL injection protection."""
        if test_queries is None:
            test_queries = [
                "'; DROP TABLE users; --",
                "' OR '1'='1",
                "'; INSERT INTO users VALUES ('hacker', 'password'); --",
                "' UNION SELECT * FROM users --"
            ]
        
        start_time = time.time()
        vulnerabilities = []
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for query in test_queries:
                try:
                    # Try to execute malicious query
                    cursor.execute(f"SELECT * FROM users WHERE username = '{query}'")
                    result = cursor.fetchall()
                    
                    # If query executed without error, there might be a vulnerability
                    vuln = SecurityVulnerability(
                        vulnerability_id=f"sql_injection_{hash(query)}",
                        title="Potential SQL Injection Vulnerability",
                        severity="critical",
                        description=f"Query '{query}' executed without proper validation",
                        remediation="Use parameterized queries or prepared statements"
                    )
                    vulnerabilities.append(vuln)
                
                except sqlite3.Error:
                    # Good: query was rejected
                    pass
            
            conn.close()
        
        except Exception as e:
            logging.error(f"Error testing SQL injection: {e}")
        
        execution_time = time.time() - start_time
        score = max(0, 100 - len(vulnerabilities) * 30)
        
        return SecurityTestResult(
            test_name="sql_injection_protection_test",
            status="failed" if vulnerabilities else "passed",
            vulnerabilities=vulnerabilities,
            score=score,
            execution_time=execution_time
        )

class SecurityComplianceTester:
    """Tests compliance with security standards."""
    
    def __init__(self):
        self.owasp_top_10 = [
            "Injection",
            "Broken Authentication",
            "Sensitive Data Exposure",
            "XML External Entities (XXE)",
            "Broken Access Control",
            "Security Misconfiguration",
            "Cross-Site Scripting (XSS)",
            "Insecure Deserialization",
            "Using Components with Known Vulnerabilities",
            "Insufficient Logging & Monitoring"
        ]
    
    def test_owasp_compliance(self, test_results: List[SecurityTestResult]) -> SecurityTestResult:
        """Test compliance with OWASP Top 10."""
        start_time = time.time()
        vulnerabilities = []
        
        # Map vulnerabilities to OWASP categories
        owasp_mapping = {
            "sql_injection": "Injection",
            "xss": "Cross-Site Scripting (XSS)",
            "hardcoded_secrets": "Sensitive Data Exposure",
            "brute_force_vulnerability": "Broken Authentication",
            "insecure_session_cookie": "Broken Authentication",
            "unencrypted_data": "Sensitive Data Exposure"
        }
        
        owasp_violations = set()
        
        for result in test_results:
            for vuln in result.vulnerabilities:
                for vuln_type, owasp_category in owasp_mapping.items():
                    if vuln_type in vuln.vulnerability_id:
                        owasp_violations.add(owasp_category)
        
        # Check for missing OWASP categories
        for category in self.owasp_top_10:
            if category not in owasp_violations:
                # This is actually good - no violations found
                pass
            else:
                vuln = SecurityVulnerability(
                    vulnerability_id=f"owasp_{category.lower().replace(' ', '_')}",
                    title=f"OWASP Violation: {category}",
                    severity="high",
                    description=f"Violation of OWASP Top 10: {category}",
                    remediation=f"Address {category} security concerns"
                )
                vulnerabilities.append(vuln)
        
        execution_time = time.time() - start_time
        score = max(0, 100 - len(vulnerations) * 10)
        
        return SecurityTestResult(
            test_name="owasp_compliance_test",
            status="failed" if vulnerabilities else "passed",
            vulnerabilities=vulnerabilities,
            score=score,
            execution_time=execution_time
        )

class SecurityTestingFramework:
    """Main security testing framework."""
    
    def __init__(self, target_url: str = "http://localhost:8000", 
                 target_host: str = "localhost", target_port: int = 8000):
        self.target_url = target_url
        self.target_host = target_host
        self.target_port = target_port
        
        # Initialize testers
        self.vuln_scanner = VulnerabilityScanner()
        self.auth_tester = AuthenticationTester(target_url)
        self.network_tester = NetworkSecurityTester(target_host, target_port)
        self.data_tester = DataSecurityTester()
        self.compliance_tester = SecurityComplianceTester()
    
    def run_comprehensive_security_scan(self, code_path: str = ".") -> Dict[str, Any]:
        """Run comprehensive security scan."""
        print("🔒 Starting Comprehensive Security Scan")
        print("=" * 50)
        
        all_results = []
        
        # Code vulnerability scanning
        print("\n📁 Scanning code for vulnerabilities...")
        code_vulns = self._scan_code_directory(Path(code_path))
        all_results.extend(code_vulns)
        
        # Authentication testing
        print("\n🔐 Testing authentication security...")
        auth_results = self._test_authentication()
        all_results.extend(auth_results)
        
        # Network security testing
        print("\n🌐 Testing network security...")
        network_results = self._test_network_security()
        all_results.extend(network_results)
        
        # Data security testing
        print("\n💾 Testing data security...")
        data_results = self._test_data_security()
        all_results.extend(data_results)
        
        # Compliance testing
        print("\n📋 Testing security compliance...")
        compliance_result = self.compliance_tester.test_owasp_compliance(all_results)
        all_results.append(compliance_result)
        
        # Generate security report
        report = self._generate_security_report(all_results)
        self._print_security_summary(report)
        
        return report
    
    def _scan_code_directory(self, code_path: Path) -> List[SecurityTestResult]:
        """Scan code directory for vulnerabilities."""
        results = []
        vulnerabilities = []
        
        # Find Python files
        python_files = list(code_path.rglob("*.py"))
        
        for file_path in python_files:
            file_vulns = self.vuln_scanner.scan_code_file(file_path)
            vulnerabilities.extend(file_vulns)
        
        # Create result
        result = SecurityTestResult(
            test_name="code_vulnerability_scan",
            status="failed" if vulnerabilities else "passed",
            vulnerabilities=vulnerabilities,
            score=max(0, 100 - len(vulnerabilities) * 5),
            execution_time=0.0
        )
        results.append(result)
        
        return results
    
    def _test_authentication(self) -> List[SecurityTestResult]:
        """Test authentication security."""
        results = []
        
        # Test weak passwords
        weak_passwords = ["password", "123456", "admin", "test"]
        result = self.auth_tester.test_weak_passwords("test_user", weak_passwords)
        results.append(result)
        
        # Test brute force protection
        result = self.auth_tester.test_brute_force_protection("/login", "test_user", weak_passwords)
        results.append(result)
        
        # Test session management
        result = self.auth_tester.test_session_management("")
        results.append(result)
        
        return results
    
    def _test_network_security(self) -> List[SecurityTestResult]:
        """Test network security."""
        results = []
        
        # Test SSL configuration
        result = self.network_tester.test_ssl_configuration()
        results.append(result)
        
        # Test port scanning
        result = self.network_tester.test_port_scanning()
        results.append(result)
        
        return results
    
    def _test_data_security(self) -> List[SecurityTestResult]:
        """Test data security."""
        results = []
        
        # Test data encryption
        result = self.data_tester.test_data_encryption()
        results.append(result)
        
        # Test SQL injection protection
        result = self.data_tester.test_sql_injection_protection()
        results.append(result)
        
        return results
    
    def _generate_security_report(self, results: List[SecurityTestResult]) -> Dict[str, Any]:
        """Generate comprehensive security report."""
        total_vulnerabilities = sum(len(r.vulnerabilities) for r in results)
        critical_vulns = sum(1 for r in results for v in r.vulnerabilities if v.severity == "critical")
        high_vulns = sum(1 for r in results for v in r.vulnerabilities if v.severity == "high")
        medium_vulns = sum(1 for r in results for v in r.vulnerabilities if v.severity == "medium")
        low_vulns = sum(1 for r in results for v in r.vulnerabilities if v.severity == "low")
        
        avg_score = sum(r.score for r in results) / len(results) if results else 0
        
        return {
            "summary": {
                "total_tests": len(results),
                "total_vulnerabilities": total_vulnerabilities,
                "critical_vulnerabilities": critical_vulns,
                "high_vulnerabilities": high_vulns,
                "medium_vulnerabilities": medium_vulns,
                "low_vulnerabilities": low_vulns,
                "average_score": avg_score,
                "security_grade": self._calculate_security_grade(avg_score)
            },
            "test_results": [
                {
                    "test_name": r.test_name,
                    "status": r.status,
                    "score": r.score,
                    "vulnerability_count": len(r.vulnerabilities),
                    "execution_time": r.execution_time
                }
                for r in results
            ],
            "vulnerabilities": [
                {
                    "id": v.vulnerability_id,
                    "title": v.title,
                    "severity": v.severity,
                    "description": v.description,
                    "remediation": v.remediation,
                    "affected_component": v.affected_component
                }
                for r in results for v in r.vulnerabilities
            ]
        }
    
    def _calculate_security_grade(self, score: float) -> str:
        """Calculate security grade based on score."""
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"
    
    def _print_security_summary(self, report: Dict[str, Any]):
        """Print security scan summary."""
        summary = report["summary"]
        
        print("\n" + "=" * 60)
        print("🔒 SECURITY SCAN SUMMARY")
        print("=" * 60)
        
        print(f"📊 Overall Security Grade: {summary['security_grade']}")
        print(f"📈 Average Score: {summary['average_score']:.1f}/100")
        print(f"🧪 Total Tests: {summary['total_tests']}")
        print(f"🚨 Total Vulnerabilities: {summary['total_vulnerabilities']}")
        
        print(f"\n⚠️  Vulnerability Breakdown:")
        print(f"   🔴 Critical: {summary['critical_vulnerabilities']}")
        print(f"   🟠 High: {summary['high_vulnerabilities']}")
        print(f"   🟡 Medium: {summary['medium_vulnerabilities']}")
        print(f"   🟢 Low: {summary['low_vulnerabilities']}")
        
        print(f"\n📋 Test Results:")
        for result in report["test_results"]:
            status_icon = "✅" if result["status"] == "passed" else "❌"
            print(f"   {status_icon} {result['test_name']}: {result['score']:.1f}/100 ({result['vulnerability_count']} vulns)")
        
        print("=" * 60)

# Example usage and demo
def demo_security_testing():
    """Demonstrate security testing capabilities."""
    print("🔒 Security Testing Framework Demo")
    print("=" * 40)
    
    # Create security framework
    security_framework = SecurityTestingFramework()
    
    # Run comprehensive security scan
    report = security_framework.run_comprehensive_security_scan()
    
    # Save report
    report_file = Path("security_scan_report.json")
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n💾 Security report saved to: {report_file}")

if __name__ == "__main__":
    # Run demo
    demo_security_testing()
