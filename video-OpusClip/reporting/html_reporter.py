#!/usr/bin/env python3
"""
HTML Reporter Module for Video-OpusClip
HTML-based reporting and visualization
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import os
import base64

class ReportLevel(str, Enum):
    """Report levels for HTML output"""
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
class HTMLReport:
    """HTML report information"""
    report_type: ReportType
    level: ReportLevel
    message: str
    data: Optional[Dict[str, Any]] = None
    timestamp: datetime = None
    duration: Optional[float] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

class HTMLReporter:
    """HTML-based reporting system"""
    
    def __init__(self, title: str = "Video-OpusClip Security Report"):
        self.title = title
        self.reports: List[HTMLReport] = []
        self.start_time = time.time()
        self.css_styles = self._get_css_styles()
        self.js_scripts = self._get_js_scripts()
    
    def _get_css_styles(self) -> str:
        """Get CSS styles for the report"""
        return """
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
                background-color: #f8f9fa;
            }
            
            .container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }
            
            .header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 40px 20px;
                text-align: center;
                border-radius: 10px;
                margin-bottom: 30px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            
            .header h1 {
                font-size: 2.5em;
                margin-bottom: 10px;
            }
            
            .header p {
                font-size: 1.2em;
                opacity: 0.9;
            }
            
            .section {
                background: white;
                border-radius: 10px;
                padding: 25px;
                margin-bottom: 25px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }
            
            .section h2 {
                color: #2c3e50;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
                margin-bottom: 20px;
            }
            
            .section h3 {
                color: #34495e;
                margin: 20px 0 10px 0;
            }
            
            .metric-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }
            
            .metric-card {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 8px;
                text-align: center;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }
            
            .metric-card h4 {
                font-size: 2em;
                margin-bottom: 5px;
            }
            
            .metric-card p {
                opacity: 0.9;
            }
            
            .table-container {
                overflow-x: auto;
                margin: 20px 0;
            }
            
            table {
                width: 100%;
                border-collapse: collapse;
                background: white;
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }
            
            th, td {
                padding: 12px 15px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }
            
            th {
                background-color: #3498db;
                color: white;
                font-weight: 600;
            }
            
            tr:hover {
                background-color: #f5f5f5;
            }
            
            .status-badge {
                padding: 4px 8px;
                border-radius: 12px;
                font-size: 0.8em;
                font-weight: bold;
                text-transform: uppercase;
            }
            
            .status-success {
                background-color: #27ae60;
                color: white;
            }
            
            .status-warning {
                background-color: #f39c12;
                color: white;
            }
            
            .status-error {
                background-color: #e74c3c;
                color: white;
            }
            
            .status-info {
                background-color: #3498db;
                color: white;
            }
            
            .progress-bar {
                width: 100%;
                height: 20px;
                background-color: #ecf0f1;
                border-radius: 10px;
                overflow: hidden;
                margin: 10px 0;
            }
            
            .progress-fill {
                height: 100%;
                background: linear-gradient(90deg, #27ae60, #2ecc71);
                transition: width 0.3s ease;
            }
            
            .chart-container {
                height: 300px;
                margin: 20px 0;
            }
            
            .alert {
                padding: 15px;
                border-radius: 8px;
                margin: 15px 0;
                border-left: 4px solid;
            }
            
            .alert-success {
                background-color: #d4edda;
                border-color: #27ae60;
                color: #155724;
            }
            
            .alert-warning {
                background-color: #fff3cd;
                border-color: #f39c12;
                color: #856404;
            }
            
            .alert-error {
                background-color: #f8d7da;
                border-color: #e74c3c;
                color: #721c24;
            }
            
            .alert-info {
                background-color: #d1ecf1;
                border-color: #3498db;
                color: #0c5460;
            }
            
            .footer {
                text-align: center;
                padding: 20px;
                color: #7f8c8d;
                border-top: 1px solid #ecf0f1;
                margin-top: 30px;
            }
            
            .collapsible {
                background-color: #f8f9fa;
                color: #333;
                cursor: pointer;
                padding: 15px;
                width: 100%;
                border: none;
                text-align: left;
                outline: none;
                font-size: 16px;
                border-radius: 5px;
                margin: 5px 0;
            }
            
            .collapsible:hover {
                background-color: #e9ecef;
            }
            
            .collapsible:after {
                content: '\\002B';
                color: #777;
                font-weight: bold;
                float: right;
                margin-left: 5px;
            }
            
            .collapsible.active:after {
                content: "\\2212";
            }
            
            .content {
                padding: 0 18px;
                max-height: 0;
                overflow: hidden;
                transition: max-height 0.2s ease-out;
                background-color: white;
                border-radius: 5px;
            }
            
            @media (max-width: 768px) {
                .container {
                    padding: 10px;
                }
                
                .header h1 {
                    font-size: 2em;
                }
                
                .metric-grid {
                    grid-template-columns: 1fr;
                }
            }
        </style>
        """
    
    def _get_js_scripts(self) -> str:
        """Get JavaScript scripts for the report"""
        return """
        <script>
            // Collapsible sections
            function toggleCollapsible(element) {
                element.classList.toggle("active");
                var content = element.nextElementSibling;
                if (content.style.maxHeight) {
                    content.style.maxHeight = null;
                } else {
                    content.style.maxHeight = content.scrollHeight + "px";
                }
            }
            
            // Auto-refresh progress bars
            function updateProgressBars() {
                const progressBars = document.querySelectorAll('.progress-fill');
                progressBars.forEach(bar => {
                    const targetWidth = bar.getAttribute('data-width');
                    if (targetWidth) {
                        bar.style.width = targetWidth + '%';
                    }
                });
            }
            
            // Initialize charts
            function initializeCharts() {
                // This would integrate with Chart.js or similar library
                console.log('Charts initialized');
            }
            
            // Search functionality
            function searchReports() {
                const input = document.getElementById('searchInput');
                const filter = input.value.toLowerCase();
                const reports = document.querySelectorAll('.report-item');
                
                reports.forEach(report => {
                    const text = report.textContent.toLowerCase();
                    if (text.includes(filter)) {
                        report.style.display = '';
                    } else {
                        report.style.display = 'none';
                    }
                });
            }
            
            // Export functionality
            function exportToPDF() {
                window.print();
            }
            
            // Initialize when page loads
            document.addEventListener('DOMContentLoaded', function() {
                updateProgressBars();
                initializeCharts();
            });
        </script>
        """
    
    def add_report(self, report_type: ReportType, level: ReportLevel, message: str, 
                   data: Optional[Dict[str, Any]] = None, duration: Optional[float] = None) -> None:
        """Add a new report"""
        report = HTMLReport(
            report_type=report_type,
            level=level,
            message=message,
            data=data,
            duration=duration
        )
        self.reports.append(report)
    
    def _get_level_class(self, level: ReportLevel) -> str:
        """Get CSS class for report level"""
        level_classes = {
            ReportLevel.DEBUG: "status-info",
            ReportLevel.INFO: "status-info",
            ReportLevel.WARNING: "status-warning",
            ReportLevel.ERROR: "status-error",
            ReportLevel.CRITICAL: "status-error",
            ReportLevel.SUCCESS: "status-success"
        }
        return level_classes.get(level, "status-info")
    
    def _get_alert_class(self, level: ReportLevel) -> str:
        """Get CSS class for alert styling"""
        alert_classes = {
            ReportLevel.DEBUG: "alert-info",
            ReportLevel.INFO: "alert-info",
            ReportLevel.WARNING: "alert-warning",
            ReportLevel.ERROR: "alert-error",
            ReportLevel.CRITICAL: "alert-error",
            ReportLevel.SUCCESS: "alert-success"
        }
        return alert_classes.get(level, "alert-info")
    
    def generate_html(self) -> str:
        """Generate complete HTML report"""
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{self.title}</title>
            {self.css_styles}
        </head>
        <body>
            <div class="container">
                {self._generate_header()}
                {self._generate_summary()}
                {self._generate_detailed_reports()}
                {self._generate_charts()}
                {self._generate_footer()}
            </div>
            {self.js_scripts}
        </body>
        </html>
        """
        return html
    
    def _generate_header(self) -> str:
        """Generate HTML header"""
        total_duration = time.time() - self.start_time
        return f"""
        <div class="header">
            <h1>{self.title}</h1>
            <p>Generated on {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}</p>
            <p>Total Duration: {total_duration:.2f} seconds</p>
        </div>
        """
    
    def _generate_summary(self) -> str:
        """Generate summary section"""
        # Calculate summary statistics
        total_reports = len(self.reports)
        reports_by_type = {}
        reports_by_level = {}
        
        for report in self.reports:
            # Count by type
            report_type = report.report_type.value
            reports_by_type[report_type] = reports_by_type.get(report_type, 0) + 1
            
            # Count by level
            level = report.level.value
            reports_by_level[level] = reports_by_level.get(level, 0) + 1
        
        # Generate metrics cards
        metrics_html = ""
        for report_type, count in reports_by_type.items():
            metrics_html += f"""
            <div class="metric-card">
                <h4>{count}</h4>
                <p>{report_type.title()} Reports</p>
            </div>
            """
        
        return f"""
        <div class="section">
            <h2>Summary</h2>
            <div class="metric-grid">
                <div class="metric-card">
                    <h4>{total_reports}</h4>
                    <p>Total Reports</p>
                </div>
                {metrics_html}
            </div>
        </div>
        """
    
    def _generate_detailed_reports(self) -> str:
        """Generate detailed reports section"""
        if not self.reports:
            return '<div class="section"><h2>Reports</h2><p>No reports available.</p></div>'
        
        reports_html = ""
        for i, report in enumerate(self.reports):
            level_class = self._get_level_class(report.level)
            alert_class = self._get_alert_class(report.level)
            
            # Generate data HTML if present
            data_html = ""
            if report.data:
                data_html = self._generate_data_html(report.data)
            
            # Generate duration HTML if present
            duration_html = ""
            if report.duration:
                duration_html = f'<p><strong>Duration:</strong> {report.duration:.2f} seconds</p>'
            
            reports_html += f"""
            <div class="report-item">
                <button class="collapsible" onclick="toggleCollapsible(this)">
                    <span class="status-badge {level_class}">{report.level.value.upper()}</span>
                    {report.message}
                </button>
                <div class="content">
                    <div class="alert {alert_class}">
                        <p><strong>Type:</strong> {report.report_type.value}</p>
                        <p><strong>Timestamp:</strong> {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
                        {duration_html}
                        {data_html}
                    </div>
                </div>
            </div>
            """
        
        return f"""
        <div class="section">
            <h2>Detailed Reports</h2>
            <input type="text" id="searchInput" onkeyup="searchReports()" placeholder="Search reports..." style="width: 100%; padding: 10px; margin-bottom: 20px; border: 1px solid #ddd; border-radius: 5px;">
            {reports_html}
        </div>
        """
    
    def _generate_data_html(self, data: Dict[str, Any]) -> str:
        """Generate HTML for report data"""
        if not data:
            return ""
        
        html = "<div style='margin-top: 10px;'><strong>Data:</strong><ul>"
        
        for key, value in data.items():
            if isinstance(value, dict):
                html += f"<li><strong>{key}:</strong> {self._generate_data_html(value)}</li>"
            elif isinstance(value, list):
                html += f"<li><strong>{key}:</strong><ul>"
                for item in value:
                    if isinstance(item, dict):
                        html += f"<li>{self._generate_data_html(item)}</li>"
                    else:
                        html += f"<li>{item}</li>"
                html += "</ul></li>"
            else:
                html += f"<li><strong>{key}:</strong> {value}</li>"
        
        html += "</ul></div>"
        return html
    
    def _generate_charts(self) -> str:
        """Generate charts section"""
        # This would integrate with Chart.js or similar
        return """
        <div class="section">
            <h2>Charts & Analytics</h2>
            <div class="chart-container">
                <p>Charts would be rendered here using Chart.js or similar library.</p>
            </div>
        </div>
        """
    
    def _generate_footer(self) -> str:
        """Generate footer section"""
        return f"""
        <div class="footer">
            <p>Report generated by Video-OpusClip Security Scanner</p>
            <p>Generated on {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}</p>
        </div>
        """
    
    def generate_security_report_html(self, security_data: Dict[str, Any]) -> str:
        """Generate security-specific HTML report"""
        security_score = security_data.get("security_score", 0)
        
        # Determine security level
        if security_score >= 80:
            security_level = "SECURE"
            level_class = "status-success"
            alert_class = "alert-success"
        elif security_score >= 60:
            security_level = "MODERATE"
            level_class = "status-warning"
            alert_class = "alert-warning"
        else:
            security_level = "CRITICAL"
            level_class = "status-error"
            alert_class = "alert-error"
        
        # Generate issues HTML
        issues_html = ""
        for severity in ["critical_issues", "high_issues", "medium_issues", "low_issues"]:
            if severity in security_data and security_data[severity]:
                issues = security_data[severity]
                severity_name = severity.replace("_", " ").title()
                issues_html += f"""
                <div class="alert alert-error">
                    <h3>{severity_name} ({len(issues)})</h3>
                    <ul>
                        {''.join(f'<li>{issue}</li>' for issue in issues)}
                    </ul>
                </div>
                """
        
        # Generate recommendations HTML
        recommendations_html = ""
        if "recommendations" in security_data and security_data["recommendations"]:
            recommendations_html = """
            <div class="alert alert-info">
                <h3>Recommendations</h3>
                <ol>
            """
            for recommendation in security_data["recommendations"]:
                recommendations_html += f"<li>{recommendation}</li>"
            recommendations_html += "</ol></div>"
        
        return f"""
        <div class="section">
            <h2>Security Assessment</h2>
            <div class="metric-grid">
                <div class="metric-card">
                    <h4>{security_score}/100</h4>
                    <p>Security Score</p>
                </div>
                <div class="metric-card">
                    <h4>{security_level}</h4>
                    <p>Security Level</p>
                </div>
            </div>
            {issues_html}
            {recommendations_html}
        </div>
        """
    
    def generate_scan_results_html(self, scan_data: Dict[str, Any]) -> str:
        """Generate scan results HTML"""
        if not scan_data:
            return ""
        
        # Generate port scan results
        port_scan_html = ""
        if "port_scan" in scan_data and scan_data["port_scan"]["success"]:
            port_data = scan_data["port_scan"]
            port_scan_html = f"""
            <div class="alert alert-info">
                <h3>Port Scan Results</h3>
                <p><strong>Target:</strong> {port_data['target']}</p>
                <p><strong>Total Ports:</strong> {port_data['total_ports']}</p>
                <p><strong>Open Ports:</strong> {port_data['open_ports']}</p>
            </div>
            """
        
        # Generate vulnerability scan results
        vuln_scan_html = ""
        if "vulnerability_scan" in scan_data and scan_data["vulnerability_scan"]["success"]:
            vuln_data = scan_data["vulnerability_scan"]
            vuln_scan_html = f"""
            <div class="alert alert-warning">
                <h3>Vulnerability Scan Results</h3>
                <p><strong>Target:</strong> {vuln_data['target']}</p>
                <p><strong>Total Vulnerabilities:</strong> {vuln_data['total_vulnerabilities']}</p>
                <p><strong>URLs Scanned:</strong> {vuln_data['scanned_urls']}</p>
            </div>
            """
        
        return f"""
        <div class="section">
            <h2>Scan Results</h2>
            {port_scan_html}
            {vuln_scan_html}
        </div>
        """
    
    def save_report(self, filename: str) -> None:
        """Save HTML report to file"""
        html_content = self.generate_html()
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"HTML report saved to {filename}")
    
    def save_security_report(self, security_data: Dict[str, Any], filename: str) -> None:
        """Save security-specific HTML report"""
        # Create a new reporter for security report
        security_reporter = HTMLReporter("Security Assessment Report")
        
        # Add security report section
        security_html = security_reporter.generate_security_report_html(security_data)
        
        # Generate complete HTML
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Security Assessment Report</title>
            {self.css_styles}
        </head>
        <body>
            <div class="container">
                {security_reporter._generate_header()}
                {security_html}
                {security_reporter._generate_footer()}
            </div>
            {self.js_scripts}
        </body>
        </html>
        """
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"Security HTML report saved to {filename}")

# Example usage
async def main():
    """Example usage of HTML reporter"""
    print("📊 HTML Reporter Example")
    
    # Create reporter
    reporter = HTMLReporter("Video-OpusClip Security Assessment")
    
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
    reporter.save_report("security_report.html")
    
    # Generate security-specific report
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
    
    reporter.save_security_report(security_data, "security_assessment.html")

if __name__ == "__main__":
    asyncio.run(main()) 