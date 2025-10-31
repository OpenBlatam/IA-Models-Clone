#!/usr/bin/env python3
"""
JSON Reporter Module for Video-OpusClip
JSON-based reporting and data export
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime
import os
import gzip
import base64

class ReportLevel(str, Enum):
    """Report levels for JSON output"""
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
class JSONReport:
    """JSON report information"""
    report_type: ReportType
    level: ReportLevel
    message: str
    data: Optional[Dict[str, Any]] = None
    timestamp: datetime = None
    duration: Optional[float] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

class JSONReporter:
    """JSON-based reporting system"""
    
    def __init__(self, title: str = "Video-OpusClip Security Report"):
        self.title = title
        self.reports: List[JSONReport] = []
        self.start_time = time.time()
        self.metadata: Dict[str, Any] = {}
    
    def add_report(self, report_type: ReportType, level: ReportLevel, message: str, 
                   data: Optional[Dict[str, Any]] = None, duration: Optional[float] = None) -> None:
        """Add a new report"""
        report = JSONReport(
            report_type=report_type,
            level=level,
            message=message,
            data=data,
            duration=duration
        )
        self.reports.append(report)
    
    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to the report"""
        self.metadata[key] = value
    
    def _convert_datetime(self, obj: Any) -> Any:
        """Convert datetime objects to ISO format for JSON serialization"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {k: self._convert_datetime(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_datetime(item) for item in obj]
        else:
            return obj
    
    def generate_json(self, pretty: bool = True) -> str:
        """Generate JSON report"""
        total_duration = time.time() - self.start_time
        
        # Prepare report data
        report_data = {
            "title": self.title,
            "metadata": self.metadata,
            "summary": {
                "total_reports": len(self.reports),
                "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
                "end_time": datetime.now().isoformat(),
                "total_duration": total_duration,
                "reports_by_type": {},
                "reports_by_level": {}
            },
            "reports": []
        }
        
        # Process reports
        for report in self.reports:
            report_dict = {
                "type": report.report_type.value,
                "level": report.level.value,
                "message": report.message,
                "timestamp": report.timestamp.isoformat(),
                "duration": report.duration
            }
            
            if report.data:
                report_dict["data"] = self._convert_datetime(report.data)
            
            report_data["reports"].append(report_dict)
            
            # Update summary statistics
            report_type = report.report_type.value
            report_data["summary"]["reports_by_type"][report_type] = \
                report_data["summary"]["reports_by_type"].get(report_type, 0) + 1
            
            level = report.level.value
            report_data["summary"]["reports_by_level"][level] = \
                report_data["summary"]["reports_by_level"].get(level, 0) + 1
        
        # Convert to JSON
        if pretty:
            return json.dumps(report_data, indent=2, ensure_ascii=False)
        else:
            return json.dumps(report_data, ensure_ascii=False)
    
    def generate_scan_report_json(self, scan_data: Dict[str, Any]) -> str:
        """Generate scan-specific JSON report"""
        scan_report = {
            "title": "Scan Report",
            "timestamp": datetime.now().isoformat(),
            "scan_data": self._convert_datetime(scan_data),
            "summary": {
                "target": scan_data.get("target", "Unknown"),
                "scan_type": scan_data.get("scan_type", "Unknown"),
                "success": scan_data.get("success", False),
                "duration": scan_data.get("duration", 0)
            }
        }
        
        # Add port scan summary
        if "port_scan" in scan_data and scan_data["port_scan"]["success"]:
            port_data = scan_data["port_scan"]
            scan_report["port_scan"] = {
                "target": port_data["target"],
                "total_ports": port_data["total_ports"],
                "open_ports": port_data["open_ports"],
                "scan_duration": port_data["scan_duration"],
                "open_ports_list": [
                    {
                        "port": result["port"],
                        "service": result.get("service", "unknown"),
                        "status": result["status"]
                    }
                    for result in port_data["results"]
                    if result["status"] == "open"
                ]
            }
        
        # Add vulnerability scan summary
        if "vulnerability_scan" in scan_data and scan_data["vulnerability_scan"]["success"]:
            vuln_data = scan_data["vulnerability_scan"]
            scan_report["vulnerability_scan"] = {
                "target": vuln_data["target"],
                "total_vulnerabilities": vuln_data["total_vulnerabilities"],
                "scanned_urls": vuln_data["scanned_urls"],
                "scan_duration": vuln_data["scan_duration"],
                "vulnerabilities_by_severity": {}
            }
            
            # Group vulnerabilities by severity
            for vuln in vuln_data["vulnerabilities"]:
                severity = vuln["severity"]
                if severity not in scan_report["vulnerability_scan"]["vulnerabilities_by_severity"]:
                    scan_report["vulnerability_scan"]["vulnerabilities_by_severity"][severity] = []
                scan_report["vulnerability_scan"]["vulnerabilities_by_severity"][severity].append(vuln)
        
        return json.dumps(scan_report, indent=2, ensure_ascii=False)
    
    def generate_enumeration_report_json(self, enum_data: Dict[str, Any]) -> str:
        """Generate enumeration-specific JSON report"""
        enum_report = {
            "title": "Enumeration Report",
            "timestamp": datetime.now().isoformat(),
            "enumeration_data": self._convert_datetime(enum_data),
            "summary": {
                "total_enumeration_types": len([k for k, v in enum_data.items() if v.get("success")]),
                "total_duration": sum(v.get("enumeration_duration", 0) for v in enum_data.values() if v.get("success"))
            }
        }
        
        # Add DNS enumeration results
        if "dns_enumeration" in enum_data and enum_data["dns_enumeration"]["success"]:
            dns_data = enum_data["dns_enumeration"]
            enum_report["dns_enumeration"] = {
                "target_domain": dns_data["target_domain"],
                "total_records": dns_data["total_records"],
                "total_subdomains": dns_data["total_subdomains"],
                "zone_transfers": dns_data["zone_transfers"],
                "enumeration_duration": dns_data["enumeration_duration"],
                "dns_records": dns_data["results"]["dns_records"],
                "subdomains": dns_data["results"]["subdomains"]
            }
        
        # Add SMB enumeration results
        if "smb_enumeration" in enum_data and enum_data["smb_enumeration"]["success"]:
            smb_data = enum_data["smb_enumeration"]
            enum_report["smb_enumeration"] = {
                "target_host": smb_data["target_host"],
                "total_shares": smb_data["total_shares"],
                "accessible_shares": smb_data["accessible_shares"],
                "total_users": smb_data["total_users"],
                "total_files": smb_data["total_files"],
                "enumeration_duration": smb_data["enumeration_duration"],
                "shares": smb_data["results"]["shares"],
                "users": smb_data["results"]["users"]
            }
        
        # Add SSH enumeration results
        if "ssh_enumeration" in enum_data and enum_data["ssh_enumeration"]["success"]:
            ssh_data = enum_data["ssh_enumeration"]
            enum_report["ssh_enumeration"] = {
                "target_host": ssh_data["target_host"],
                "target_port": ssh_data["target_port"],
                "total_host_keys": ssh_data["total_host_keys"],
                "total_algorithms": ssh_data["total_algorithms"],
                "total_users": ssh_data["total_users"],
                "enumeration_duration": ssh_data["enumeration_duration"],
                "host_keys": ssh_data["results"]["host_keys"],
                "algorithms": ssh_data["results"]["algorithms"]
            }
        
        return json.dumps(enum_report, indent=2, ensure_ascii=False)
    
    def generate_attack_report_json(self, attack_data: Dict[str, Any]) -> str:
        """Generate attack-specific JSON report"""
        attack_report = {
            "title": "Attack Report",
            "timestamp": datetime.now().isoformat(),
            "attack_data": self._convert_datetime(attack_data),
            "summary": {
                "total_attacks": 0,
                "successful_attacks": 0,
                "total_credentials": 0,
                "total_exploits": 0,
                "successful_exploits": 0
            }
        }
        
        # Add brute force results
        if "brute_force" in attack_data:
            brute_data = attack_data["brute_force"]
            attack_report["brute_force"] = {
                "total_attacks": brute_data["total_attacks"],
                "successful_attacks": brute_data["successful_attacks"],
                "total_credentials": brute_data["total_credentials"],
                "attack_results": [
                    {
                        "attack_type": result.attack_type.value,
                        "target": result.target,
                        "status": result.status.value,
                        "credentials_found": [
                            {
                                "username": cred.username,
                                "password": cred.password,
                                "service": cred.service,
                                "discovered_at": cred.discovered_at.isoformat()
                            }
                            for cred in result.credentials_found
                        ],
                        "attempts_made": result.attempts_made,
                        "total_combinations": result.total_combinations,
                        "start_time": result.start_time.isoformat(),
                        "end_time": result.end_time.isoformat() if result.end_time else None
                    }
                    for result in brute_data["results"]
                ]
            }
            attack_report["summary"]["total_attacks"] += brute_data["total_attacks"]
            attack_report["summary"]["successful_attacks"] += brute_data["successful_attacks"]
            attack_report["summary"]["total_credentials"] += brute_data["total_credentials"]
        
        # Add exploitation results
        if "exploitation" in attack_data:
            exploit_data = attack_data["exploitation"]
            attack_report["exploitation"] = {
                "total_exploits": exploit_data["total_exploits"],
                "successful_exploits": exploit_data["successful_exploits"],
                "exploit_results": [
                    {
                        "exploit_type": result.exploit_type.value,
                        "target": result.target,
                        "status": result.status.value,
                        "payload": result.payload,
                        "response": result.response,
                        "shell_access": result.shell_access,
                        "data_extracted": result.data_extracted,
                        "start_time": result.start_time.isoformat(),
                        "end_time": result.end_time.isoformat() if result.end_time else None
                    }
                    for result in exploit_data["results"]
                ]
            }
            attack_report["summary"]["total_exploits"] += exploit_data["total_exploits"]
            attack_report["summary"]["successful_exploits"] += exploit_data["successful_exploits"]
        
        return json.dumps(attack_report, indent=2, ensure_ascii=False)
    
    def generate_security_report_json(self, security_data: Dict[str, Any]) -> str:
        """Generate security-specific JSON report"""
        security_report = {
            "title": "Security Assessment Report",
            "timestamp": datetime.now().isoformat(),
            "security_data": self._convert_datetime(security_data),
            "summary": {
                "security_score": security_data.get("security_score", 0),
                "total_issues": 0,
                "critical_issues": len(security_data.get("critical_issues", [])),
                "high_issues": len(security_data.get("high_issues", [])),
                "medium_issues": len(security_data.get("medium_issues", [])),
                "low_issues": len(security_data.get("low_issues", []))
            },
            "issues": {
                "critical": security_data.get("critical_issues", []),
                "high": security_data.get("high_issues", []),
                "medium": security_data.get("medium_issues", []),
                "low": security_data.get("low_issues", [])
            },
            "recommendations": security_data.get("recommendations", [])
        }
        
        # Calculate total issues
        security_report["summary"]["total_issues"] = sum([
            security_report["summary"]["critical_issues"],
            security_report["summary"]["high_issues"],
            security_report["summary"]["medium_issues"],
            security_report["summary"]["low_issues"]
        ])
        
        return json.dumps(security_report, indent=2, ensure_ascii=False)
    
    def save_report(self, filename: str, pretty: bool = True, compress: bool = False) -> None:
        """Save JSON report to file"""
        json_content = self.generate_json(pretty)
        
        if compress:
            # Compress with gzip
            compressed_filename = filename + ".gz"
            with gzip.open(compressed_filename, 'wt', encoding='utf-8') as f:
                f.write(json_content)
            print(f"Compressed JSON report saved to {compressed_filename}")
        else:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(json_content)
            print(f"JSON report saved to {filename}")
    
    def save_scan_report(self, scan_data: Dict[str, Any], filename: str) -> None:
        """Save scan-specific JSON report"""
        json_content = self.generate_scan_report_json(scan_data)
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(json_content)
        
        print(f"Scan JSON report saved to {filename}")
    
    def save_enumeration_report(self, enum_data: Dict[str, Any], filename: str) -> None:
        """Save enumeration-specific JSON report"""
        json_content = self.generate_enumeration_report_json(enum_data)
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(json_content)
        
        print(f"Enumeration JSON report saved to {filename}")
    
    def save_attack_report(self, attack_data: Dict[str, Any], filename: str) -> None:
        """Save attack-specific JSON report"""
        json_content = self.generate_attack_report_json(attack_data)
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(json_content)
        
        print(f"Attack JSON report saved to {filename}")
    
    def save_security_report(self, security_data: Dict[str, Any], filename: str) -> None:
        """Save security-specific JSON report"""
        json_content = self.generate_security_report_json(security_data)
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(json_content)
        
        print(f"Security JSON report saved to {filename}")
    
    def export_to_base64(self) -> str:
        """Export report as base64 encoded string"""
        json_content = self.generate_json(pretty=False)
        return base64.b64encode(json_content.encode('utf-8')).decode('utf-8')
    
    def import_from_base64(self, base64_string: str) -> None:
        """Import report from base64 encoded string"""
        json_content = base64.b64decode(base64_string.encode('utf-8')).decode('utf-8')
        data = json.loads(json_content)
        
        # Reconstruct reports
        self.reports.clear()
        for report_dict in data.get("reports", []):
            report = JSONReport(
                report_type=ReportType(report_dict["type"]),
                level=ReportLevel(report_dict["level"]),
                message=report_dict["message"],
                data=report_dict.get("data"),
                timestamp=datetime.fromisoformat(report_dict["timestamp"]),
                duration=report_dict.get("duration")
            )
            self.reports.append(report)
        
        # Restore metadata
        self.metadata = data.get("metadata", {})
    
    def merge_reports(self, other_reporter: 'JSONReporter') -> None:
        """Merge reports from another JSONReporter"""
        self.reports.extend(other_reporter.reports)
        self.metadata.update(other_reporter.metadata)
    
    def filter_reports(self, report_type: Optional[ReportType] = None, 
                      level: Optional[ReportLevel] = None) -> List[JSONReport]:
        """Filter reports by type and/or level"""
        filtered_reports = self.reports
        
        if report_type:
            filtered_reports = [r for r in filtered_reports if r.report_type == report_type]
        
        if level:
            filtered_reports = [r for r in filtered_reports if r.level == level]
        
        return filtered_reports
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get report statistics"""
        stats = {
            "total_reports": len(self.reports),
            "reports_by_type": {},
            "reports_by_level": {},
            "total_duration": time.time() - self.start_time
        }
        
        for report in self.reports:
            # Count by type
            report_type = report.report_type.value
            stats["reports_by_type"][report_type] = stats["reports_by_type"].get(report_type, 0) + 1
            
            # Count by level
            level = report.level.value
            stats["reports_by_level"][level] = stats["reports_by_level"].get(level, 0) + 1
        
        return stats

# Example usage
async def main():
    """Example usage of JSON reporter"""
    print("📊 JSON Reporter Example")
    
    # Create reporter
    reporter = JSONReporter("Video-OpusClip Security Assessment")
    
    # Add metadata
    reporter.add_metadata("version", "1.0.0")
    reporter.add_metadata("target", "192.168.1.100")
    reporter.add_metadata("scanner", "Video-OpusClip")
    
    # Add some reports
    reporter.add_report(
        ReportType.SCAN,
        ReportLevel.INFO,
        "Starting port scan on target 192.168.1.100"
    )
    
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
    
    reporter.add_report(
        ReportType.SECURITY,
        ReportLevel.ERROR,
        "Critical vulnerability detected: SQL injection",
        data={"vulnerability_type": "SQL Injection", "severity": "Critical", "affected_url": "/login"}
    )
    
    # Generate and save report
    reporter.save_report("security_report.json")
    
    # Save compressed version
    reporter.save_report("security_report_compressed.json", compress=True)
    
    # Generate scan-specific report
    scan_data = {
        "target": "192.168.1.100",
        "scan_type": "comprehensive",
        "success": True,
        "duration": 45.2,
        "port_scan": {
            "success": True,
            "target": "192.168.1.100",
            "total_ports": 1000,
            "open_ports": 5,
            "scan_duration": 12.3,
            "results": [
                {"port": 22, "status": "open", "service": "ssh"},
                {"port": 80, "status": "open", "service": "http"},
                {"port": 443, "status": "open", "service": "https"}
            ]
        },
        "vulnerability_scan": {
            "success": True,
            "target": "http://192.168.1.100",
            "total_vulnerabilities": 2,
            "scanned_urls": 15,
            "scan_duration": 32.9,
            "vulnerabilities": [
                {"type": "sql_injection", "severity": "high", "url": "/login"},
                {"type": "xss", "severity": "medium", "url": "/search"}
            ]
        }
    }
    
    reporter.save_scan_report(scan_data, "scan_report.json")
    
    # Generate security report
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
    
    reporter.save_security_report(security_data, "security_assessment.json")
    
    # Export to base64
    base64_export = reporter.export_to_base64()
    print(f"Base64 export (first 100 chars): {base64_export[:100]}...")
    
    # Get statistics
    stats = reporter.get_statistics()
    print(f"Report statistics: {stats}")

if __name__ == "__main__":
    asyncio.run(main()) 