from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import asyncio
import aiofiles
from typing import Any, List, Dict, Optional
import logging
"""
HTML reporting utilities for cybersecurity testing results.

Provides tools for:
- Web-based result visualization
- Interactive HTML reports
- Chart and graph generation
- Responsive design
"""


@dataclass
class HTMLReportConfig:
    """Configuration for HTML reporting."""
    output_directory: str = "reports"
    template_path: Optional[str] = None
    include_charts: bool = True
    include_timestamps: bool = True
    responsive_design: bool = True
    dark_mode: bool = False
    auto_open: bool = False

@dataclass
class HTMLReportResult:
    """Result of an HTML reporting operation."""
    success: bool = False
    file_path: Optional[str] = None
    file_size: int = 0
    time_taken: float = 0.0
    error_message: Optional[str] = None

# CPU-bound operations (use 'def')
def generate_html_header(title: str, config: HTMLReportConfig) -> str:
    """Generate HTML header - CPU intensive."""
    css_styles = """
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
        .summary-card { background: white; border-radius: 8px; padding: 20px; margin: 20px 0; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .vulnerability-card { background: white; border-radius: 8px; padding: 15px; margin: 10px 0; border-left: 4px solid #ff6b6b; }
        .vulnerability-critical { border-left-color: #ff4757; }
        .vulnerability-high { border-left-color: #ff6348; }
        .vulnerability-medium { border-left-color: #ffa502; }
        .vulnerability-low { border-left-color: #2ed573; }
        .vulnerability-info { border-left-color: #3742fa; }
        .progress-bar { background: #f1f2f6; border-radius: 10px; height: 20px; overflow: hidden; }
        .progress-fill { height: 100%; background: linear-gradient(90deg, #667eea, #764ba2); transition: width 0.3s ease; }
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }
        .stat-card { background: white; padding: 20px; border-radius: 8px; text-align: center; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .stat-number { font-size: 2em; font-weight: bold; color: #667eea; }
        .stat-label { color: #666; margin-top: 5px; }
        .table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        .table th, .table td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        .table th { background-color: #f8f9fa; font-weight: 600; }
        .status-success { color: #2ed573; }
        .status-error { color: #ff4757; }
        .status-warning { color: #ffa502; }
        .timestamp { color: #666; font-size: 0.9em; }
        @media (max-width: 768px) { .stats-grid { grid-template-columns: 1fr; } }
    </style>
    """
    
    if config.dark_mode:
        css_styles += """
        <style>
            body { background-color: #1a1a1a; color: #ffffff; }
            .summary-card, .vulnerability-card, .stat-card { background-color: #2d2d2d; color: #ffffff; }
            .table th { background-color: #3d3d3d; }
            .table td { border-bottom-color: #4d4d4d; }
        </style>
        """
    
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{title}</title>
        {css_styles}
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>{title}</h1>
                <p class="timestamp">Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
    """

def generate_html_footer() -> str:
    """Generate HTML footer - CPU intensive."""
    return """
        </div>
        <script>
            // Add interactive features
            document.addEventListener('DOMContentLoaded', function() {
                // Animate progress bars
                const progressBars = document.querySelectorAll('.progress-fill');
                progressBars.forEach(bar => {
                    const width = bar.style.width;
                    bar.style.width = '0%';
                    setTimeout(() => {
                        bar.style.width = width;
                    }, 500);
                });
                
                // Add click handlers for vulnerability cards
                const vulnCards = document.querySelectorAll('.vulnerability-card');
                vulnCards.forEach(card => {
                    card.addEventListener('click', function() {
                        this.style.transform = 'scale(1.02)';
                        setTimeout(() => {
                            this.style.transform = 'scale(1)';
                        }, 200);
                    });
                });
            });
        </script>
    </body>
    </html>
    """

def format_severity_badge(severity: str) -> str:
    """Format severity as HTML badge - CPU intensive."""
    severity_colors = {
        "critical": "#ff4757",
        "high": "#ff6348", 
        "medium": "#ffa502",
        "low": "#2ed573",
        "info": "#3742fa"
    }
    
    color = severity_colors.get(severity.lower(), "#666")
    return f'<span style="background-color: {color}; color: white; padding: 4px 8px; border-radius: 4px; font-size: 0.8em;">{severity.upper()}</span>'

def format_status_badge(status: str) -> str:
    """Format status as HTML badge - CPU intensive."""
    status_colors = {
        "success": "#2ed573",
        "warning": "#ffa502",
        "error": "#ff4757",
        "info": "#3742fa"
    }
    
    color = status_colors.get(status.lower(), "#666")
    return f'<span style="background-color: {color}; color: white; padding: 4px 8px; border-radius: 4px; font-size: 0.8em;">{status.upper()}</span>'

def generate_summary_section(summary: Dict[str, Any]) -> str:
    """Generate summary section HTML - CPU intensive."""
    html = '<div class="summary-card"><h2>Scan Summary</h2><div class="stats-grid">'
    
    for key, value in summary.items():
        if isinstance(value, float):
            formatted_value = f"{value:.2f}"
        elif isinstance(value, dict):
            formatted_value = f"{len(value)}"
        else:
            formatted_value = str(value)
        
        label = key.replace('_', ' ').title()
        html += f"""
        <div class="stat-card">
            <div class="stat-number">{formatted_value}</div>
            <div class="stat-label">{label}</div>
        </div>
        """
    
    html += '</div></div>'
    return html

def generate_vulnerabilities_section(vulnerabilities: List[Dict[str, Any]]) -> str:
    """Generate vulnerabilities section HTML - CPU intensive."""
    if not vulnerabilities:
        return '<div class="summary-card"><h2>Vulnerabilities</h2><p>No vulnerabilities found.</p></div>'
    
    html = '<div class="summary-card"><h2>Vulnerability Findings</h2>'
    
    # Group by severity
    severity_groups = {}
    for vuln in vulnerabilities:
        severity = vuln.get('severity', 'info').lower()
        if severity not in severity_groups:
            severity_groups[severity] = []
        severity_groups[severity].append(vuln)
    
    # Generate sections by severity
    severity_order = ['critical', 'high', 'medium', 'low', 'info']
    for severity in severity_order:
        if severity in severity_groups:
            html += f'<h3>{format_severity_badge(severity)} Vulnerabilities ({len(severity_groups[severity])})</h3>'
            
            for vuln in severity_groups[severity]:
                html += generate_vulnerability_card(vuln, severity)
    
    html += '</div>'
    return html

def generate_vulnerability_card(vuln: Dict[str, Any], severity: str) -> str:
    """Generate individual vulnerability card HTML - CPU intensive."""
    title = vuln.get('title', 'Unknown Vulnerability')
    description = vuln.get('description', 'No description available')
    target = vuln.get('target', 'Unknown Target')
    
    html = f'<div class="vulnerability-card vulnerability-{severity}">'
    html += f'<h4>{title}</h4>'
    html += f'<p><strong>Target:</strong> {target}</p>'
    html += f'<p><strong>Description:</strong> {description}</p>'
    
    if 'cvss_score' in vuln:
        html += f'<p><strong>CVSS Score:</strong> {vuln["cvss_score"]}</p>'
    
    if 'remediation' in vuln:
        html += f'<p><strong>Remediation:</strong> {vuln["remediation"]}</p>'
    
    html += '</div>'
    return html

def generate_scan_results_table(results: List[Dict[str, Any]]) -> str:
    """Generate scan results table HTML - CPU intensive."""
    if not results:
        return '<div class="summary-card"><h2>Scan Results</h2><p>No scan results available.</p></div>'
    
    html = '''
    <div class="summary-card">
        <h2>Scan Results</h2>
        <table class="table">
            <thead>
                <tr>
                    <th>Target</th>
                    <th>Status</th>
                    <th>Response Time</th>
                    <th>Details</th>
                </tr>
            </thead>
            <tbody>
    '''
    
    for result in results:
        target = result.get('target', 'Unknown')
        success = result.get('success', False)
        status = format_status_badge('success' if success else 'error')
        response_time = result.get('response_time', 0)
        
        html += f'''
        <tr>
            <td>{target}</td>
            <td>{status}</td>
            <td>{response_time:.3f}s</td>
            <td>{result.get('error_message', 'Success')}</td>
        </tr>
        '''
    
    html += '''
            </tbody>
        </table>
    </div>
    '''
    return html

# Async operations (use 'async def')
async def write_html_file_async(file_path: str, content: str) -> None:
    """Write HTML content to file asynchronously - I/O bound."""
    # Ensure directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        await f.write(content)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")

async def create_report_directory_async(directory: str) -> None:
    """Create report directory asynchronously - I/O bound."""
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

class HTMLReporter:
    """HTML-based reporting tool."""
    
    def __init__(self, config: HTMLReportConfig):
        
    """__init__ function."""
self.config = config
        self.start_time = None
    
    async def generate_report(self, data: Dict[str, Any], title: str = "Cybersecurity Report") -> HTMLReportResult:
        """Generate complete HTML report."""
        start_time = time.time()
        
        try:
            # Create output directory
            await create_report_directory_async(self.config.output_directory)
            
            # Generate HTML content
            html_content = self._generate_html_content(data, title)
            
            # Write to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"cybersecurity_report_{timestamp}.html"
            file_path = os.path.join(self.config.output_directory, filename)
            
            await write_html_file_async(file_path, html_content)
            
            # Get file size
            file_size = os.path.getsize(file_path)
            
            return HTMLReportResult(
                success=True,
                file_path=file_path,
                file_size=file_size,
                time_taken=time.time() - start_time
            )
            
        except Exception as e:
            return HTMLReportResult(
                success=False,
                time_taken=time.time() - start_time,
                error_message=str(e)
            )
    
    def _generate_html_content(self, data: Dict[str, Any], title: str) -> str:
        """Generate complete HTML content - CPU intensive."""
        html = generate_html_header(title, self.config)
        
        # Summary section
        if 'summary' in data:
            html += generate_summary_section(data['summary'])
        
        # Scan results section
        if 'scan_results' in data:
            html += generate_scan_results_table(data['scan_results'])
        
        # Vulnerabilities section
        if 'vulnerabilities' in data:
            html += generate_vulnerabilities_section(data['vulnerabilities'])
        
        # Additional sections can be added here
        
        html += generate_html_footer()
        return html
    
    async def generate_vulnerability_report(self, vulnerabilities: List[Dict[str, Any]]) -> HTMLReportResult:
        """Generate vulnerability-specific HTML report."""
        data = {
            'vulnerabilities': vulnerabilities,
            'summary': {
                'total_vulnerabilities': len(vulnerabilities),
                'critical_count': len([v for v in vulnerabilities if v.get('severity') == 'critical']),
                'high_count': len([v for v in vulnerabilities if v.get('severity') == 'high']),
                'medium_count': len([v for v in vulnerabilities if v.get('severity') == 'medium']),
                'low_count': len([v for v in vulnerabilities if v.get('severity') == 'low']),
                'info_count': len([v for v in vulnerabilities if v.get('severity') == 'info'])
            }
        }
        
        return await self.generate_report(data, "Vulnerability Assessment Report")
    
    async def generate_scan_report(self, scan_results: List[Dict[str, Any]]) -> HTMLReportResult:
        """Generate scan-specific HTML report."""
        successful_scans = [r for r in scan_results if r.get('success', False)]
        failed_scans = [r for r in scan_results if not r.get('success', False)]
        
        data = {
            'scan_results': scan_results,
            'summary': {
                'total_scans': len(scan_results),
                'successful_scans': len(successful_scans),
                'failed_scans': len(failed_scans),
                'success_rate': (len(successful_scans) / len(scan_results)) * 100 if scan_results else 0,
                'average_response_time': sum(r.get('response_time', 0) for r in scan_results) / len(scan_results) if scan_results else 0
            }
        }
        
        return await self.generate_report(data, "Security Scan Report") 