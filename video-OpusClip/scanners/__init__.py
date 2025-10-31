"""
Scanners Module for Video-OpusClip
Network security scanning and assessment tools
"""

from .port_scanner import PortScanner, ScanConfig as PortScanConfig, ScanType, PortStatus, PortResult
from .vulnerability_scanner import VulnerabilityScanner, ScanConfig as VulnScanConfig, VulnerabilityType, SeverityLevel, Vulnerability
from .web_scanner import WebScanner, ScanConfig as WebScanConfig, ScanCategory, WebScanResult

__all__ = [
    # Port Scanner
    'PortScanner',
    'PortScanConfig', 
    'ScanType',
    'PortStatus',
    'PortResult',
    
    # Vulnerability Scanner
    'VulnerabilityScanner',
    'VulnScanConfig',
    'VulnerabilityType',
    'SeverityLevel',
    'Vulnerability',
    
    # Web Scanner
    'WebScanner',
    'WebScanConfig',
    'ScanCategory',
    'WebScanResult'
]

# Example usage
async def run_comprehensive_scan(target_host: str, target_url: str) -> Dict[str, Any]:
    """
    Run comprehensive security scan including port, vulnerability, and web scanning
    
    Args:
        target_host: Host to scan for ports
        target_url: URL to scan for vulnerabilities and web analysis
        
    Returns:
        Dictionary containing all scan results
    """
    results = {}
    
    # Port scan
    print("🔍 Running port scan...")
    port_config = PortScanConfig(
        target_host=target_host,
        start_port=1,
        end_port=1000,
        scan_type=ScanType.TCP_CONNECT,
        timeout=3.0,
        max_concurrent=50
    )
    port_scanner = PortScanner(port_config)
    port_results = await port_scanner.scan_ports()
    results["port_scan"] = port_results
    
    # Vulnerability scan
    print("🔍 Running vulnerability scan...")
    vuln_config = VulnScanConfig(
        target_url=target_url,
        scan_depth=2,
        max_concurrent=10,
        timeout=30.0,
        scan_types=[
            VulnerabilityType.SQL_INJECTION,
            VulnerabilityType.XSS,
            VulnerabilityType.OPEN_REDIRECT,
            VulnerabilityType.EXPOSED_ENDPOINTS
        ]
    )
    vuln_scanner = VulnerabilityScanner(vuln_config)
    vuln_results = await vuln_scanner.scan_target()
    results["vulnerability_scan"] = vuln_results
    
    # Web scan
    print("🔍 Running web scan...")
    web_config = WebScanConfig(
        target_url=target_url,
        scan_categories=[
            ScanCategory.INFORMATION_GATHERING,
            ScanCategory.TECHNOLOGY_DETECTION,
            ScanCategory.SECURITY_HEADERS,
            ScanCategory.CONTENT_ANALYSIS,
            ScanCategory.API_DISCOVERY
        ],
        max_concurrent=10,
        timeout=30.0
    )
    web_scanner = WebScanner(web_config)
    web_results = await web_scanner.scan_website()
    results["web_scan"] = web_results
    
    return results

def generate_security_report(scan_results: Dict[str, Any]) -> str:
    """
    Generate comprehensive security report from scan results
    
    Args:
        scan_results: Results from comprehensive scan
        
    Returns:
        Formatted security report
    """
    report = "🔒 COMPREHENSIVE SECURITY SCAN REPORT\n"
    report += "=" * 60 + "\n\n"
    
    # Port scan summary
    if "port_scan" in scan_results and scan_results["port_scan"]["success"]:
        port_data = scan_results["port_scan"]
        report += f"📡 PORT SCAN RESULTS\n"
        report += f"Target: {port_data['target']}\n"
        report += f"Open Ports: {port_data['open_ports']}/{port_data['total_ports']}\n"
        report += f"Scan Duration: {port_data['scan_duration']:.2f}s\n\n"
        
        # List open ports
        if port_data['open_ports'] > 0:
            report += "Open Ports:\n"
            for result in port_data['results']:
                if result['status'] == 'open':
                    report += f"  • Port {result['port']}: {result['service'] or 'unknown'}\n"
            report += "\n"
    
    # Vulnerability scan summary
    if "vulnerability_scan" in scan_results and scan_results["vulnerability_scan"]["success"]:
        vuln_data = scan_results["vulnerability_scan"]
        report += f"🚨 VULNERABILITY SCAN RESULTS\n"
        report += f"Target: {vuln_data['target']}\n"
        report += f"Vulnerabilities Found: {vuln_data['total_vulnerabilities']}\n"
        report += f"URLs Scanned: {vuln_data['scanned_urls']}\n"
        report += f"Scan Duration: {vuln_data['scan_duration']:.2f}s\n\n"
        
        # Group vulnerabilities by severity
        if vuln_data['total_vulnerabilities'] > 0:
            severity_counts = {}
            for vuln in vuln_data['vulnerabilities']:
                severity = vuln['severity']
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            report += "Vulnerabilities by Severity:\n"
            for severity in ['critical', 'high', 'medium', 'low']:
                if severity in severity_counts:
                    report += f"  • {severity.title()}: {severity_counts[severity]}\n"
            report += "\n"
    
    # Web scan summary
    if "web_scan" in scan_results and scan_results["web_scan"]["success"]:
        web_data = scan_results["web_scan"]
        report += f"🌐 WEB SCAN RESULTS\n"
        report += f"Target: {web_data['target']}\n"
        report += f"Findings: {web_data['total_findings']}\n"
        report += f"Scan Duration: {web_data['scan_duration']:.2f}s\n\n"
        
        # Group findings by category
        if web_data['total_findings'] > 0:
            category_counts = {}
            for result in web_data['results']:
                category = result['category']
                category_counts[category] = category_counts.get(category, 0) + 1
            
            report += "Findings by Category:\n"
            for category, count in category_counts.items():
                report += f"  • {category.replace('_', ' ').title()}: {count}\n"
            report += "\n"
    
    # Overall summary
    report += "📊 OVERALL SUMMARY\n"
    report += "-" * 30 + "\n"
    
    total_issues = 0
    if "port_scan" in scan_results and scan_results["port_scan"]["success"]:
        total_issues += scan_results["port_scan"]["open_ports"]
    if "vulnerability_scan" in scan_results and scan_results["vulnerability_scan"]["success"]:
        total_issues += scan_results["vulnerability_scan"]["total_vulnerabilities"]
    if "web_scan" in scan_results and scan_results["web_scan"]["success"]:
        total_issues += scan_results["web_scan"]["total_findings"]
    
    report += f"Total Security Issues: {total_issues}\n"
    
    if total_issues == 0:
        report += "✅ No security issues detected\n"
    elif total_issues < 5:
        report += "⚠️  Low number of security issues detected\n"
    elif total_issues < 10:
        report += "⚠️  Moderate number of security issues detected\n"
    else:
        report += "🚨 High number of security issues detected\n"
    
    report += "\n🔧 RECOMMENDATIONS\n"
    report += "-" * 20 + "\n"
    report += "1. Review and address all critical and high severity vulnerabilities\n"
    report += "2. Close unnecessary open ports\n"
    report += "3. Implement security headers\n"
    report += "4. Regularly update software and dependencies\n"
    report += "5. Conduct regular security assessments\n"
    
    return report 