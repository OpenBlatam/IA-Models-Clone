from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, field_validator
from datetime import datetime
import structlog
import json
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
HTML reporting module for cybersecurity assessment results.
"""

logger = structlog.get_logger(__name__)

class HTMLReportInput(BaseModel):
    """Input model for HTML report generation."""
    scan_results: Dict[str, Any]
    target_info: Dict[str, str]
    scan_metadata: Dict[str, Any]
    include_charts: bool = True
    custom_css: Optional[str] = None
    template_name: str = "default"
    
    @field_validator('scan_results')
    def validate_scan_results(cls, v) -> bool:
        if not v:
            raise ValueError("Scan results cannot be empty")
        return v
    
    @field_validator('target_info')
    def validate_target_info(cls, v) -> Optional[Dict[str, Any]]:
        if not v:
            raise ValueError("Target info cannot be empty")
        return v

class HTMLReportResult(BaseModel):
    """Result model for HTML report generation."""
    html_content: str
    file_path: Optional[str] = None
    summary_stats: Dict[str, int]
    generation_time: float
    is_successful: bool
    error_message: Optional[str] = None

def generate_html_report(input_data: HTMLReportInput) -> HTMLReportResult:
    """
    RORO: Receive HTMLReportInput, return HTMLReportResult
    
    Generate formatted HTML report for cybersecurity assessment results.
    """
    start_time = datetime.now()
    
    try:
        # Generate HTML content
        html_content = create_html_report(input_data)
        
        # Calculate summary statistics
        summary_stats = calculate_summary_stats(input_data.scan_results)
        
        generation_time = (datetime.now() - start_time).total_seconds()
        
        logger.info("HTML report generated successfully", 
                   target=input_data.target_info.get('host', 'unknown'),
                   generation_time=generation_time)
        
        return HTMLReportResult(
            html_content=html_content,
            summary_stats=summary_stats,
            generation_time=generation_time,
            is_successful=True
        )
        
    except Exception as e:
        generation_time = (datetime.now() - start_time).total_seconds()
        logger.error("HTML report generation failed", error=str(e))
        
        return HTMLReportResult(
            html_content="",
            summary_stats={},
            generation_time=generation_time,
            is_successful=False,
            error_message=str(e)
        )

def create_html_report(input_data: HTMLReportInput) -> str:
    """Create complete HTML report."""
    html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cybersecurity Assessment Report</title>
    <style>
        {get_default_css()}
        {input_data.custom_css or ''}
    </style>
    {get_chart_scripts() if input_data.include_charts else ''}
</head>
<body>
    {create_header_section(input_data.target_info)}
    {create_summary_section(input_data.scan_results)}
    {create_details_section(input_data.scan_results)}
    {create_charts_section(input_data.scan_results) if input_data.include_charts else ''}
    {create_footer_section(input_data.scan_metadata)}
</body>
</html>
    """
    
    return html_template

def create_header_section(target_info: Dict[str, str]) -> str:
    """Create HTML header section."""
    return f"""
    <header class="report-header">
        <div class="container">
            <h1>🔒 Cybersecurity Assessment Report</h1>
            <div class="target-info">
                <h2>Target Information</h2>
                <table class="info-table">
                    <tr><td><strong>Host:</strong></td><td>{target_info.get('host', 'Unknown')}</td></tr>
                    <tr><td><strong>IP Address:</strong></td><td>{target_info.get('ip', 'Unknown')}</td></tr>
                    <tr><td><strong>Scan Date:</strong></td><td>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</td></tr>
                    <tr><td><strong>Ports Scanned:</strong></td><td>{target_info.get('ports', 'Unknown')}</td></tr>
                </table>
            </div>
        </div>
    </header>
    """

def create_summary_section(scan_results: Dict[str, Any]) -> str:
    """Create HTML summary section."""
    summary_html = """
    <section class="summary-section">
        <div class="container">
            <h2>📊 Scan Summary</h2>
            <div class="summary-grid">
    """
    
    for category, data in scan_results.items():
        if isinstance(data, dict):
            count = len(data.get('results', []))
        elif isinstance(data, list):
            count = len(data)
        else:
            count = 0
        
        status_class = "status-clean" if count == 0 else "status-issues"
        status_text = "✅ Clean" if count == 0 else "⚠️ Issues Found"
        
        summary_html += f"""
                <div class="summary-card {status_class}">
                    <h3>{category.title()}</h3>
                    <div class="count">{count}</div>
                    <div class="status">{status_text}</div>
                </div>
        """
    
    summary_html += """
            </div>
        </div>
    </section>
    """
    
    return summary_html

def create_details_section(scan_results: Dict[str, Any]) -> str:
    """Create HTML details section."""
    details_html = """
    <section class="details-section">
        <div class="container">
            <h2>🔍 Detailed Results</h2>
    """
    
    for category, data in scan_results.items():
        if not data or (isinstance(data, list) and len(data) == 0):
            continue
        
        details_html += f"""
            <div class="category-details">
                <h3>{category.title()}</h3>
                <div class="table-container">
        """
        
        if isinstance(data, dict) and 'results' in data:
            results = data['results']
            if results and isinstance(results[0], dict):
                details_html += create_results_table(results)
        
        elif isinstance(data, list) and data and isinstance(data[0], dict):
            details_html += create_results_table(data)
        
        details_html += """
                </div>
            </div>
        """
    
    details_html += """
        </div>
    </section>
    """
    
    return details_html

def create_results_table(results: List[Dict[str, Any]]) -> str:
    """Create HTML table for results."""
    if not results:
        return "<p>No results found.</p>"
    
    table_html = "<table class='results-table'>"
    
    # Header
    table_html += "<thead><tr>"
    for key in results[0].keys():
        table_html += f"<th>{key.title()}</th>"
    table_html += "</tr></thead>"
    
    # Body
    table_html += "<tbody>"
    for result in results:
        table_html += "<tr>"
        for key in results[0].keys():
            value = result.get(key, '')
            # Apply severity styling
            if key.lower() == 'severity':
                severity_class = f"severity-{value.lower()}" if value else ""
                table_html += f"<td class='{severity_class}'>{value}</td>"
            else:
                table_html += f"<td>{value}</td>"
        table_html += "</tr>"
    table_html += "</tbody></table>"
    
    return table_html

def create_charts_section(scan_results: Dict[str, Any]) -> str:
    """Create HTML charts section."""
    chart_data = prepare_chart_data(scan_results)
    
    return f"""
    <section class="charts-section">
        <div class="container">
            <h2>📈 Charts & Analytics</h2>
            <div class="charts-grid">
                <div class="chart-container">
                    <canvas id="findingsChart"></canvas>
                </div>
                <div class="chart-container">
                    <canvas id="severityChart"></canvas>
                </div>
            </div>
        </div>
        <script>
            const chartData = {json.dumps(chart_data)};
            createCharts(chartData);
        </script>
    </section>
    """

def create_footer_section(scan_metadata: Dict[str, Any]) -> str:
    """Create HTML footer section."""
    metadata_html = ""
    for key, value in scan_metadata.items():
        metadata_html += f"<tr><td><strong>{key.title()}:</strong></td><td>{value}</td></tr>"
    
    return f"""
    <footer class="report-footer">
        <div class="container">
            <h3>📋 Scan Metadata</h3>
            <table class="metadata-table">
                {metadata_html}
            </table>
            <p class="footer-note">🔐 Report generated by Key Messages Security Scanner</p>
        </div>
    </footer>
    """

def get_default_css() -> str:
    """Get default CSS styles."""
    return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f5f5;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 20px;
        }
        
        .report-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px 0;
        }
        
        .report-header h1 {
            font-size: 2.5em;
            margin-bottom: 20px;
        }
        
        .info-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        
        .info-table td {
            padding: 10px;
            border-bottom: 1px solid rgba(255,255,255,0.2);
        }
        
        .summary-section {
            padding: 40px 0;
            background: white;
        }
        
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        
        .summary-card {
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .status-clean {
            background: linear-gradient(135deg, #4CAF50, #45a049);
            color: white;
        }
        
        .status-issues {
            background: linear-gradient(135deg, #ff9800, #f57c00);
            color: white;
        }
        
        .summary-card .count {
            font-size: 2em;
            font-weight: bold;
            margin: 10px 0;
        }
        
        .details-section {
            padding: 40px 0;
            background: #f9f9f9;
        }
        
        .category-details {
            background: white;
            margin: 20px 0;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .results-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }
        
        .results-table th,
        .results-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        
        .results-table th {
            background-color: #f8f9fa;
            font-weight: bold;
        }
        
        .severity-critical { color: #dc3545; font-weight: bold; }
        .severity-high { color: #fd7e14; font-weight: bold; }
        .severity-medium { color: #ffc107; font-weight: bold; }
        .severity-low { color: #28a745; font-weight: bold; }
        
        .charts-section {
            padding: 40px 0;
            background: white;
        }
        
        .charts-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        
        .chart-container {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
        }
        
        .report-footer {
            background: #343a40;
            color: white;
            padding: 40px 0;
        }
        
        .metadata-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }
        
        .metadata-table td {
            padding: 10px;
            border-bottom: 1px solid rgba(255,255,255,0.2);
        }
        
        .footer-note {
            text-align: center;
            margin-top: 20px;
            opacity: 0.8;
        }
    """

def get_chart_scripts() -> str:
    """Get Chart.js scripts for charts."""
    return """
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <script>
            function createCharts(data) {
                // Findings Chart
                const findingsCtx = document.getElementById('findingsChart').getContext('2d');
                new Chart(findingsCtx, {
                    type: 'bar',
                    data: {
                        labels: data.categories,
                        datasets: [{
                            label: 'Findings',
                            data: data.counts,
                            backgroundColor: 'rgba(54, 162, 235, 0.8)'
                        }]
                    },
                    options: {
                        responsive: true,
                        plugins: {
                            title: {
                                display: true,
                                text: 'Findings by Category'
                            }
                        }
                    }
                });
                
                // Severity Chart
                const severityCtx = document.getElementById('severityChart').getContext('2d');
                new Chart(severityCtx, {
                    type: 'doughnut',
                    data: {
                        labels: data.severities,
                        datasets: [{
                            data: data.severityCounts,
                            backgroundColor: [
                                '#dc3545',
                                '#fd7e14', 
                                '#ffc107',
                                '#28a745'
                            ]
                        }]
                    },
                    options: {
                        responsive: true,
                        plugins: {
                            title: {
                                display: true,
                                text: 'Findings by Severity'
                            }
                        }
                    }
                });
            }
        </script>
    """

def prepare_chart_data(scan_results: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare data for charts."""
    categories = []
    counts = []
    severities = ['Critical', 'High', 'Medium', 'Low']
    severity_counts = [0, 0, 0, 0]
    
    for category, data in scan_results.items():
        categories.append(category.title())
        
        if isinstance(data, dict) and 'results' in data:
            results = data['results']
            counts.append(len(results))
            
            for result in results:
                if isinstance(result, dict):
                    severity = result.get('severity', '').lower()
                    if severity == 'critical':
                        severity_counts[0] += 1
                    elif severity == 'high':
                        severity_counts[1] += 1
                    elif severity == 'medium':
                        severity_counts[2] += 1
                    elif severity == 'low':
                        severity_counts[3] += 1
        
        elif isinstance(data, list):
            counts.append(len(data))
    
    return {
        'categories': categories,
        'counts': counts,
        'severities': severities,
        'severityCounts': severity_counts
    }

def calculate_summary_stats(scan_results: Dict[str, Any]) -> Dict[str, int]:
    """Calculate summary statistics from scan results."""
    stats = {
        'total_categories': len(scan_results),
        'total_findings': 0,
        'critical_findings': 0,
        'high_findings': 0,
        'medium_findings': 0,
        'low_findings': 0
    }
    
    for category, data in scan_results.items():
        if isinstance(data, dict) and 'results' in data:
            results = data['results']
            stats['total_findings'] += len(results)
            
            for result in results:
                if isinstance(result, dict):
                    severity = result.get('severity', '').lower()
                    if severity == 'critical':
                        stats['critical_findings'] += 1
                    elif severity == 'high':
                        stats['high_findings'] += 1
                    elif severity == 'medium':
                        stats['medium_findings'] += 1
                    elif severity == 'low':
                        stats['low_findings'] += 1
        
        elif isinstance(data, list):
            stats['total_findings'] += len(data)
    
    return stats 