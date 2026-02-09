"""
Reporting Module for Video-OpusClip
Comprehensive reporting and output generation tools
"""

from .console_reporter import (
    ConsoleReporter, ConsoleReport, ReportLevel, ReportType
)
from .html_reporter import (
    HTMLReporter, HTMLReport
)
from .json_reporter import (
    JSONReporter, JSONReport
)

__all__ = [
    # Console Reporter
    'ConsoleReporter',
    'ConsoleReport',
    
    # HTML Reporter
    'HTMLReporter',
    'HTMLReport',
    
    # JSON Reporter
    'JSONReporter',
    'JSONReport',
    
    # Common Enums
    'ReportLevel',
    'ReportType'
]

# Unified reporting function
async def generate_comprehensive_report(
    scan_results: Optional[Dict[str, Any]] = None,
    enumeration_results: Optional[Dict[str, Any]] = None,
    attack_results: Optional[Dict[str, Any]] = None,
    security_analysis: Optional[Dict[str, Any]] = None,
    output_formats: List[str] = ["console", "html", "json"],
    output_directory: str = "reports"
) -> Dict[str, str]:
    """
    Generate comprehensive reports in multiple formats
    
    Args:
        scan_results: Results from scanning operations
        enumeration_results: Results from enumeration operations
        attack_results: Results from attack operations
        security_analysis: Security analysis results
        output_formats: List of output formats ("console", "html", "json")
        output_directory: Directory to save reports
        
    Returns:
        Dictionary mapping format to filename
    """
    import os
    from datetime import datetime
    
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output_files = {}
    
    # Console report
    if "console" in output_formats:
        console_reporter = ConsoleReporter(enable_colors=True, enable_timestamps=True)
        
        if scan_results:
            console_reporter.print_scan_results(scan_results)
        
        if enumeration_results:
            console_reporter.print_enumeration_results(enumeration_results)
        
        if attack_results:
            console_reporter.print_attack_results(attack_results)
        
        if security_analysis:
            console_reporter.print_security_report(security_analysis)
        
        console_reporter.print_footer()
        output_files["console"] = "console_output"
    
    # HTML report
    if "html" in output_formats:
        html_reporter = HTMLReporter("Video-OpusClip Security Assessment")
        
        # Add scan results
        if scan_results:
            html_reporter.generate_scan_results_html(scan_results)
        
        # Add security analysis
        if security_analysis:
            html_reporter.generate_security_report_html(security_analysis)
        
        html_filename = os.path.join(output_directory, f"security_report_{timestamp}.html")
        html_reporter.save_report(html_filename)
        output_files["html"] = html_filename
    
    # JSON report
    if "json" in output_formats:
        json_reporter = JSONReporter("Video-OpusClip Security Assessment")
        
        # Add metadata
        json_reporter.add_metadata("generated_at", datetime.now().isoformat())
        json_reporter.add_metadata("version", "1.0.0")
        
        # Add scan results
        if scan_results:
            scan_filename = os.path.join(output_directory, f"scan_report_{timestamp}.json")
            json_reporter.save_scan_report(scan_results, scan_filename)
            output_files["json_scan"] = scan_filename
        
        # Add enumeration results
        if enumeration_results:
            enum_filename = os.path.join(output_directory, f"enumeration_report_{timestamp}.json")
            json_reporter.save_enumeration_report(enumeration_results, enum_filename)
            output_files["json_enumeration"] = enum_filename
        
        # Add attack results
        if attack_results:
            attack_filename = os.path.join(output_directory, f"attack_report_{timestamp}.json")
            json_reporter.save_attack_report(attack_results, attack_filename)
            output_files["json_attack"] = attack_filename
        
        # Add security analysis
        if security_analysis:
            security_filename = os.path.join(output_directory, f"security_report_{timestamp}.json")
            json_reporter.save_security_report(security_analysis, security_filename)
            output_files["json_security"] = security_filename
    
    return output_files

def create_report_summary(
    scan_results: Optional[Dict[str, Any]] = None,
    enumeration_results: Optional[Dict[str, Any]] = None,
    attack_results: Optional[Dict[str, Any]] = None,
    security_analysis: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create a summary of all results
    
    Args:
        scan_results: Results from scanning operations
        enumeration_results: Results from enumeration operations
        attack_results: Results from attack operations
        security_analysis: Security analysis results
        
    Returns:
        Summary dictionary
    """
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_findings": 0,
        "critical_findings": 0,
        "high_findings": 0,
        "medium_findings": 0,
        "low_findings": 0,
        "credentials_compromised": 0,
        "services_compromised": 0,
        "vulnerabilities_found": 0,
        "security_score": 100
    }
    
    # Process scan results
    if scan_results:
        if "port_scan" in scan_results and scan_results["port_scan"]["success"]:
            summary["open_ports"] = scan_results["port_scan"]["open_ports"]
        
        if "vulnerability_scan" in scan_results and scan_results["vulnerability_scan"]["success"]:
            vuln_count = scan_results["vulnerability_scan"]["total_vulnerabilities"]
            summary["vulnerabilities_found"] += vuln_count
            summary["total_findings"] += vuln_count
    
    # Process enumeration results
    if enumeration_results:
        for enum_type, enum_data in enumeration_results.items():
            if enum_data.get("success"):
                summary["total_findings"] += 1
    
    # Process attack results
    if attack_results:
        if "brute_force" in attack_results:
            brute_data = attack_results["brute_force"]
            summary["credentials_compromised"] += brute_data["total_credentials"]
            summary["services_compromised"] += brute_data["successful_attacks"]
            summary["total_findings"] += brute_data["total_credentials"]
        
        if "exploitation" in attack_results:
            exploit_data = attack_results["exploitation"]
            summary["total_findings"] += exploit_data["successful_exploits"]
    
    # Process security analysis
    if security_analysis:
        summary["security_score"] = security_analysis.get("security_score", 100)
        summary["critical_findings"] = len(security_analysis.get("critical_issues", []))
        summary["high_findings"] = len(security_analysis.get("high_issues", []))
        summary["medium_findings"] = len(security_analysis.get("medium_issues", []))
        summary["low_findings"] = len(security_analysis.get("low_issues", []))
    
    return summary

def generate_executive_summary(
    scan_results: Optional[Dict[str, Any]] = None,
    enumeration_results: Optional[Dict[str, Any]] = None,
    attack_results: Optional[Dict[str, Any]] = None,
    security_analysis: Optional[Dict[str, Any]] = None
) -> str:
    """
    Generate an executive summary of all findings
    
    Args:
        scan_results: Results from scanning operations
        enumeration_results: Results from enumeration operations
        attack_results: Results from attack operations
        security_analysis: Security analysis results
        
    Returns:
        Executive summary string
    """
    summary = create_report_summary(scan_results, enumeration_results, attack_results, security_analysis)
    
    executive_summary = f"""
EXECUTIVE SUMMARY
================

Security Assessment Date: {datetime.now().strftime('%B %d, %Y')}

OVERALL SECURITY SCORE: {summary['security_score']}/100

KEY FINDINGS:
• Total Findings: {summary['total_findings']}
• Critical Issues: {summary['critical_findings']}
• High Priority Issues: {summary['high_findings']}
• Medium Priority Issues: {summary['medium_findings']}
• Low Priority Issues: {summary['low_findings']}

SECURITY BREACHES:
• Credentials Compromised: {summary['credentials_compromised']}
• Services Compromised: {summary['services_compromised']}
• Vulnerabilities Found: {summary['vulnerabilities_found']}

RECOMMENDATIONS:
"""
    
    if summary['critical_findings'] > 0:
        executive_summary += "• IMMEDIATE ACTION REQUIRED: Address critical security issues\n"
    
    if summary['credentials_compromised'] > 0:
        executive_summary += "• Change all compromised credentials immediately\n"
    
    if summary['services_compromised'] > 0:
        executive_summary += "• Secure compromised services and implement access controls\n"
    
    if summary['vulnerabilities_found'] > 0:
        executive_summary += "• Patch identified vulnerabilities and update systems\n"
    
    executive_summary += "• Implement comprehensive security monitoring\n"
    executive_summary += "• Conduct regular security assessments\n"
    executive_summary += "• Provide security awareness training to staff\n"
    
    return executive_summary

def create_report_archive(
    scan_results: Optional[Dict[str, Any]] = None,
    enumeration_results: Optional[Dict[str, Any]] = None,
    attack_results: Optional[Dict[str, Any]] = None,
    security_analysis: Optional[Dict[str, Any]] = None,
    output_directory: str = "reports"
) -> str:
    """
    Create a comprehensive report archive with all formats
    
    Args:
        scan_results: Results from scanning operations
        enumeration_results: Results from enumeration operations
        attack_results: Results from attack operations
        security_analysis: Security analysis results
        output_directory: Directory to save reports
        
    Returns:
        Path to the archive file
    """
    import zipfile
    import os
    from datetime import datetime
    
    # Generate all reports
    output_files = await generate_comprehensive_report(
        scan_results=scan_results,
        enumeration_results=enumeration_results,
        attack_results=attack_results,
        security_analysis=security_analysis,
        output_formats=["html", "json"],
        output_directory=output_directory
    )
    
    # Create executive summary
    executive_summary = generate_executive_summary(
        scan_results, enumeration_results, attack_results, security_analysis
    )
    
    summary_filename = os.path.join(output_directory, "executive_summary.txt")
    with open(summary_filename, 'w') as f:
        f.write(executive_summary)
    
    # Create ZIP archive
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_filename = os.path.join(output_directory, f"security_report_archive_{timestamp}.zip")
    
    with zipfile.ZipFile(archive_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add all report files
        for format_type, filepath in output_files.items():
            if os.path.exists(filepath):
                zipf.write(filepath, os.path.basename(filepath))
        
        # Add executive summary
        zipf.write(summary_filename, "executive_summary.txt")
        
        # Add README
        readme_content = """
VIDEO-OPUSCLIP SECURITY REPORT ARCHIVE
=====================================

This archive contains comprehensive security assessment reports in multiple formats:

1. HTML Report: Interactive web-based report with charts and detailed findings
2. JSON Reports: Machine-readable reports for integration with other tools
3. Executive Summary: High-level overview of findings and recommendations

REPORT CONTENTS:
- scan_report_*.json: Detailed port and vulnerability scan results
- enumeration_report_*.json: DNS, SMB, and SSH enumeration findings
- attack_report_*.json: Brute force and exploitation attack results
- security_report_*.json: Comprehensive security analysis
- security_report_*.html: Interactive HTML report
- executive_summary.txt: Executive summary and recommendations

USAGE:
- Open HTML files in a web browser for interactive viewing
- Use JSON files for data analysis and integration
- Review executive summary for immediate action items

For questions or support, contact the security team.
        """
        
        zipf.writestr("README.txt", readme_content)
    
    print(f"Report archive created: {archive_filename}")
    return archive_filename

# Example usage
async def main():
    """Example usage of comprehensive reporting"""
    print("📊 Comprehensive Reporting Example")
    
    # Sample data
    scan_results = {
        "target": "192.168.1.100",
        "port_scan": {
            "success": True,
            "open_ports": 5,
            "total_ports": 1000
        },
        "vulnerability_scan": {
            "success": True,
            "total_vulnerabilities": 3,
            "scanned_urls": 15
        }
    }
    
    security_analysis = {
        "security_score": 65,
        "critical_issues": ["Weak SSH configuration"],
        "high_issues": ["Open MySQL port"],
        "medium_issues": ["Verbose error messages"],
        "low_issues": ["Information disclosure"],
        "recommendations": [
            "Change default credentials",
            "Configure SSH properly",
            "Update system packages"
        ]
    }
    
    # Generate comprehensive reports
    output_files = await generate_comprehensive_report(
        scan_results=scan_results,
        security_analysis=security_analysis,
        output_formats=["console", "html", "json"]
    )
    
    print(f"Generated reports: {output_files}")
    
    # Create report archive
    archive_path = await create_report_archive(
        scan_results=scan_results,
        security_analysis=security_analysis
    )
    
    print(f"Report archive: {archive_path}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 