#!/usr/bin/env python3
"""
Console Reporter Module for Video-OpusClip
Console-based reporting and output formatting
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import sys
import os

class ReportLevel(str, Enum):
    """Report levels for console output"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    SUCCESS = "success"

class ReportType(str, Enum):
    """Types of reports"""
    SCAN = "scan"
    ENUMERATION = "enumeration"
    ATTACK = "attack"
    SECURITY = "security"
    PERFORMANCE = "performance"
    SYSTEM = "system"

@dataclass
class ConsoleReport:
    """Console report information"""
    report_type: ReportType
    level: ReportLevel
    message: str
    data: Optional[Dict[str, Any]] = None
    timestamp: datetime = None
    duration: Optional[float] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

class ConsoleReporter:
    """Console-based reporting system"""
    
    def __init__(self, enable_colors: bool = True, enable_timestamps: bool = True):
        self.enable_colors = enable_colors and self._supports_colors()
        self.enable_timestamps = enable_timestamps
        self.reports: List[ConsoleReport] = []
        self.start_time = time.time()
        
        # ANSI color codes
        self.colors = {
            ReportLevel.DEBUG: "\033[36m",      # Cyan
            ReportLevel.INFO: "\033[34m",       # Blue
            ReportLevel.WARNING: "\033[33m",    # Yellow
            ReportLevel.ERROR: "\033[31m",      # Red
            ReportLevel.CRITICAL: "\033[35m",   # Magenta
            ReportLevel.SUCCESS: "\033[32m",    # Green
            "RESET": "\033[0m",                 # Reset
            "BOLD": "\033[1m",                  # Bold
            "UNDERLINE": "\033[4m"              # Underline
        }
    
    def _supports_colors(self) -> bool:
        """Check if terminal supports colors"""
        return hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
    
    def add_report(self, report_type: ReportType, level: ReportLevel, message: str, 
                   data: Optional[Dict[str, Any]] = None, duration: Optional[float] = None) -> None:
        """Add a new report"""
        report = ConsoleReport(
            report_type=report_type,
            level=level,
            message=message,
            data=data,
            duration=duration
        )
        self.reports.append(report)
        self._print_report(report)
    
    def _print_report(self, report: ConsoleReport) -> None:
        """Print a single report to console"""
        # Build timestamp
        timestamp_str = ""
        if self.enable_timestamps:
            timestamp_str = f"[{report.timestamp.strftime('%H:%M:%S')}] "
        
        # Build level indicator
        level_str = f"[{report.level.upper()}]"
        
        # Build message
        message_str = report.message
        
        # Apply colors if enabled
        if self.enable_colors:
            color = self.colors.get(report.level, "")
            reset = self.colors["RESET"]
            level_str = f"{color}{level_str}{reset}"
        
        # Print the report
        print(f"{timestamp_str}{level_str} {message_str}")
        
        # Print additional data if present
        if report.data:
            self._print_data(report.data, indent=2)
        
        # Print duration if present
        if report.duration:
            duration_str = f"Duration: {report.duration:.2f}s"
            if self.enable_colors:
                duration_str = f"{self.colors['BOLD']}{duration_str}{self.colors['RESET']}"
            print(f"{' ' * (len(timestamp_str) + len(level_str) + 1)}{duration_str}")
    
    def _print_data(self, data: Dict[str, Any], indent: int = 0) -> None:
        """Print data in a formatted way"""
        indent_str = " " * indent
        
        for key, value in data.items():
            if isinstance(value, dict):
                print(f"{indent_str}{key}:")
                self._print_data(value, indent + 2)
            elif isinstance(value, list):
                print(f"{indent_str}{key}:")
                for item in value:
                    if isinstance(item, dict):
                        self._print_data(item, indent + 2)
                    else:
                        print(f"{indent_str}  - {item}")
            else:
                print(f"{indent_str}{key}: {value}")
    
    def print_header(self, title: str, subtitle: Optional[str] = None) -> None:
        """Print a formatted header"""
        width = 60
        print("\n" + "=" * width)
        
        if self.enable_colors:
            title = f"{self.colors['BOLD']}{title}{self.colors['RESET']}"
        
        print(f"{title:^{width}}")
        
        if subtitle:
            if self.enable_colors:
                subtitle = f"{self.colors['UNDERLINE']}{subtitle}{self.colors['RESET']}"
            print(f"{subtitle:^{width}}")
        
        print("=" * width)
    
    def print_section(self, title: str) -> None:
        """Print a section header"""
        if self.enable_colors:
            title = f"{self.colors['BOLD']}{title}{self.colors['RESET']}"
        print(f"\n{title}")
        print("-" * len(title))
    
    def print_progress(self, current: int, total: int, description: str = "Progress") -> None:
        """Print a progress bar"""
        percentage = (current / total) * 100 if total > 0 else 0
        bar_length = 30
        filled_length = int(bar_length * current // total)
        
        bar = "█" * filled_length + "░" * (bar_length - filled_length)
        
        if self.enable_colors:
            bar = f"{self.colors['SUCCESS']}{bar}{self.colors['RESET']}"
        
        print(f"\r{description}: [{bar}] {percentage:.1f}% ({current}/{total})", end="", flush=True)
        
        if current == total:
            print()  # New line when complete
    
    def print_table(self, headers: List[str], rows: List[List[str]], title: Optional[str] = None) -> None:
        """Print a formatted table"""
        if title:
            self.print_section(title)
        
        if not headers or not rows:
            return
        
        # Calculate column widths
        col_widths = [len(header) for header in headers]
        for row in rows:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(str(cell)))
        
        # Print header
        header_str = " | ".join(header.ljust(width) for header, width in zip(headers, col_widths))
        if self.enable_colors:
            header_str = f"{self.colors['BOLD']}{header_str}{self.colors['RESET']}"
        print(header_str)
        
        # Print separator
        separator = "-" * len(header_str)
        print(separator)
        
        # Print rows
        for row in rows:
            row_str = " | ".join(str(cell).ljust(width) for cell, width in zip(row, col_widths))
            print(row_str)
    
    def print_summary(self, summary_data: Dict[str, Any]) -> None:
        """Print a summary report"""
        self.print_header("SUMMARY REPORT")
        
        # Print key metrics
        if "total_scans" in summary_data:
            self.print_section("Scan Results")
            print(f"Total Scans: {summary_data['total_scans']}")
            print(f"Successful: {summary_data.get('successful_scans', 0)}")
            print(f"Failed: {summary_data.get('failed_scans', 0)}")
        
        if "total_vulnerabilities" in summary_data:
            self.print_section("Vulnerabilities")
            print(f"Total Vulnerabilities: {summary_data['total_vulnerabilities']}")
            print(f"Critical: {summary_data.get('critical_vulns', 0)}")
            print(f"High: {summary_data.get('high_vulns', 0)}")
            print(f"Medium: {summary_data.get('medium_vulns', 0)}")
            print(f"Low: {summary_data.get('low_vulns', 0)}")
        
        if "total_credentials" in summary_data:
            self.print_section("Credentials")
            print(f"Credentials Found: {summary_data['total_credentials']}")
            print(f"Services Compromised: {summary_data.get('services_compromised', 0)}")
        
        # Print duration
        if "duration" in summary_data:
            self.print_section("Performance")
            print(f"Total Duration: {summary_data['duration']:.2f} seconds")
            print(f"Average per Scan: {summary_data.get('avg_duration', 0):.2f} seconds")
    
    def print_security_report(self, security_data: Dict[str, Any]) -> None:
        """Print a security assessment report"""
        self.print_header("SECURITY ASSESSMENT")
        
        # Security score
        if "security_score" in security_data:
            score = security_data["security_score"]
            score_str = f"Security Score: {score}/100"
            
            if score >= 80:
                level = ReportLevel.SUCCESS
                status = "SECURE"
            elif score >= 60:
                level = ReportLevel.WARNING
                status = "MODERATE"
            else:
                level = ReportLevel.CRITICAL
                status = "CRITICAL"
            
            if self.enable_colors:
                color = self.colors[level]
                reset = self.colors["RESET"]
                score_str = f"{color}{score_str} ({status}){reset}"
            
            print(score_str)
            print()
        
        # Issues by severity
        for severity in ["critical_issues", "high_issues", "medium_issues", "low_issues"]:
            if severity in security_data and security_data[severity]:
                issues = security_data[severity]
                severity_name = severity.replace("_", " ").title()
                
                if self.enable_colors:
                    color = self.colors[ReportLevel.ERROR if "critical" in severity else ReportLevel.WARNING]
                    reset = self.colors["RESET"]
                    severity_name = f"{color}{severity_name}{reset}"
                
                print(f"{severity_name}:")
                for issue in issues:
                    print(f"  • {issue}")
                print()
        
        # Recommendations
        if "recommendations" in security_data and security_data["recommendations"]:
            self.print_section("Recommendations")
            for i, recommendation in enumerate(security_data["recommendations"], 1):
                print(f"{i}. {recommendation}")
    
    def print_scan_results(self, scan_data: Dict[str, Any]) -> None:
        """Print scan results"""
        self.print_header("SCAN RESULTS")
        
        # Target information
        if "target" in scan_data:
            self.print_section("Target Information")
            print(f"Target: {scan_data['target']}")
            if "port" in scan_data:
                print(f"Port: {scan_data['port']}")
            if "scan_type" in scan_data:
                print(f"Scan Type: {scan_data['scan_type']}")
        
        # Results summary
        if "results" in scan_data:
            results = scan_data["results"]
            
            # Port scan results
            if "port_scan" in results and results["port_scan"]["success"]:
                port_data = results["port_scan"]
                self.print_section("Port Scan Results")
                print(f"Total Ports: {port_data['total_ports']}")
                print(f"Open Ports: {port_data['open_ports']}")
                
                if port_data["open_ports"] > 0:
                    print("\nOpen Ports:")
                    for result in port_data["results"]:
                        if result["status"] == "open":
                            service = result.get("service", "unknown")
                            print(f"  • Port {result['port']}: {service}")
            
            # Vulnerability scan results
            if "vulnerability_scan" in results and results["vulnerability_scan"]["success"]:
                vuln_data = results["vulnerability_scan"]
                self.print_section("Vulnerability Scan Results")
                print(f"Total Vulnerabilities: {vuln_data['total_vulnerabilities']}")
                print(f"URLs Scanned: {vuln_data['scanned_urls']}")
                
                if vuln_data["total_vulnerabilities"] > 0:
                    print("\nVulnerabilities by Severity:")
                    severity_counts = {}
                    for vuln in vuln_data["vulnerabilities"]:
                        severity = vuln["severity"]
                        severity_counts[severity] = severity_counts.get(severity, 0) + 1
                    
                    for severity in ["critical", "high", "medium", "low"]:
                        if severity in severity_counts:
                            count = severity_counts[severity]
                            severity_str = f"{severity.title()}: {count}"
                            if self.enable_colors:
                                color = self.colors[ReportLevel.ERROR if severity in ["critical", "high"] else ReportLevel.WARNING]
                                reset = self.colors["RESET"]
                                severity_str = f"{color}{severity_str}{reset}"
                            print(f"  • {severity_str}")
    
    def print_enumeration_results(self, enum_data: Dict[str, Any]) -> None:
        """Print enumeration results"""
        self.print_header("ENUMERATION RESULTS")
        
        # DNS enumeration
        if "dns_enumeration" in enum_data and enum_data["dns_enumeration"]["success"]:
            dns_data = enum_data["dns_enumeration"]
            self.print_section("DNS Enumeration")
            print(f"Target Domain: {dns_data['target_domain']}")
            print(f"DNS Records: {dns_data['total_records']}")
            print(f"Subdomains: {dns_data['total_subdomains']}")
            print(f"Zone Transfers: {dns_data['zone_transfers']}")
        
        # SMB enumeration
        if "smb_enumeration" in enum_data and enum_data["smb_enumeration"]["success"]:
            smb_data = enum_data["smb_enumeration"]
            self.print_section("SMB Enumeration")
            print(f"Target Host: {smb_data['target_host']}")
            print(f"Total Shares: {smb_data['total_shares']}")
            print(f"Accessible Shares: {smb_data['accessible_shares']}")
            print(f"Users: {smb_data['total_users']}")
        
        # SSH enumeration
        if "ssh_enumeration" in enum_data and enum_data["ssh_enumeration"]["success"]:
            ssh_data = enum_data["ssh_enumeration"]
            self.print_section("SSH Enumeration")
            print(f"Target Host: {ssh_data['target_host']}:{ssh_data['target_port']}")
            print(f"Host Keys: {ssh_data['total_host_keys']}")
            print(f"Algorithms: {ssh_data['total_algorithms']}")
            print(f"Users: {ssh_data['total_users']}")
    
    def print_attack_results(self, attack_data: Dict[str, Any]) -> None:
        """Print attack results"""
        self.print_header("ATTACK RESULTS")
        
        # Brute force results
        if "brute_force" in attack_data:
            brute_data = attack_data["brute_force"]
            self.print_section("Brute Force Attacks")
            print(f"Total Attacks: {brute_data['total_attacks']}")
            print(f"Successful Attacks: {brute_data['successful_attacks']}")
            print(f"Credentials Found: {brute_data['total_credentials']}")
            
            if brute_data["total_credentials"] > 0:
                print("\nCompromised Credentials:")
                for result in brute_data["results"]:
                    if result.credentials_found:
                        for cred in result.credentials_found:
                            cred_str = f"{cred.username}:{cred.password} ({cred.service})"
                            if self.enable_colors:
                                cred_str = f"{self.colors[ReportLevel.CRITICAL]}{cred_str}{self.colors['RESET']}"
                            print(f"  • {cred_str}")
        
        # Exploitation results
        if "exploitation" in attack_data:
            exploit_data = attack_data["exploitation"]
            self.print_section("Exploitation Attacks")
            print(f"Total Exploits: {exploit_data['total_exploits']}")
            print(f"Successful Exploits: {exploit_data['successful_exploits']}")
            
            if exploit_data["successful_exploits"] > 0:
                print("\nSuccessful Exploits:")
                for result in exploit_data["results"]:
                    if result.status == "success":
                        exploit_str = f"{result.exploit_type} on {result.target}"
                        if self.enable_colors:
                            exploit_str = f"{self.colors[ReportLevel.CRITICAL]}{exploit_str}{self.colors['RESET']}"
                        print(f"  • {exploit_str}")
    
    def print_footer(self) -> None:
        """Print a footer"""
        total_duration = time.time() - self.start_time
        footer = f"Report generated in {total_duration:.2f} seconds"
        
        if self.enable_colors:
            footer = f"{self.colors['BOLD']}{footer}{self.colors['RESET']}"
        
        print(f"\n{footer}")
        print("=" * 60)
    
    def export_to_json(self, filename: str) -> None:
        """Export reports to JSON file"""
        export_data = {
            "reports": [
                {
                    "type": report.report_type.value,
                    "level": report.level.value,
                    "message": report.message,
                    "data": report.data,
                    "timestamp": report.timestamp.isoformat(),
                    "duration": report.duration
                }
                for report in self.reports
            ],
            "summary": {
                "total_reports": len(self.reports),
                "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
                "end_time": datetime.now().isoformat(),
                "total_duration": time.time() - self.start_time
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.add_report(
            ReportType.SYSTEM,
            ReportLevel.INFO,
            f"Reports exported to {filename}"
        )

# Example usage
async def main():
    """Example usage of console reporter"""
    print("📊 Console Reporter Example")
    
    # Create reporter
    reporter = ConsoleReporter(enable_colors=True, enable_timestamps=True)
    
    # Print header
    reporter.print_header("VIDEO-OPUSCLIP SECURITY SCAN", "Comprehensive Security Assessment")
    
    # Add some reports
    reporter.add_report(
        ReportType.SCAN,
        ReportLevel.INFO,
        "Starting port scan on target 192.168.1.100"
    )
    
    # Simulate progress
    for i in range(1, 11):
        reporter.print_progress(i, 10, "Port Scan Progress")
        await asyncio.sleep(0.1)
    
    reporter.add_report(
        ReportType.SCAN,
        ReportLevel.SUCCESS,
        "Port scan completed successfully",
        data={"open_ports": 5, "total_ports": 1000},
        duration=2.5
    )
    
    reporter.add_report(
        ReportType.SECURITY,
        ReportLevel.WARNING,
        "Found 3 open ports with potential vulnerabilities"
    )
    
    # Print a table
    headers = ["Port", "Service", "Status", "Risk"]
    rows = [
        ["22", "SSH", "Open", "Medium"],
        ["80", "HTTP", "Open", "Low"],
        ["443", "HTTPS", "Open", "Low"],
        ["3306", "MySQL", "Open", "High"],
        ["8080", "HTTP-Proxy", "Open", "Medium"]
    ]
    reporter.print_table(headers, rows, "Open Ports")
    
    # Print security summary
    security_data = {
        "security_score": 65,
        "critical_issues": ["Weak SSH configuration", "Default MySQL credentials"],
        "high_issues": ["Open MySQL port", "Unpatched services"],
        "medium_issues": ["Verbose error messages"],
        "low_issues": ["Information disclosure"],
        "recommendations": [
            "Change default MySQL credentials",
            "Configure SSH properly",
            "Update system packages",
            "Implement firewall rules"
        ]
    }
    reporter.print_security_report(security_data)
    
    # Print footer
    reporter.print_footer()
    
    # Export to JSON
    reporter.export_to_json("console_report.json")

if __name__ == "__main__":
    asyncio.run(main()) 