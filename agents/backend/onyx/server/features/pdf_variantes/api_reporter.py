#!/usr/bin/env python3
"""
API Reporter
============
Generate comprehensive reports from various API tools.

⚠️ DEPRECATED: This file is deprecated. Consider migrating to the new tools structure.

For new code, use:
    from tools.manager import ToolManager
    manager = ToolManager()
    # Tools can be extended with reporting capabilities
"""
import warnings

warnings.warn(
    "api_reporter.py is deprecated. Consider migrating to the new tools structure in tools/.",
    DeprecationWarning,
    stacklevel=2
)

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import asdict


class APIReporter:
    """API report generator."""
    
    def __init__(self):
        self.reports: List[Dict[str, Any]] = []
    
    def add_health_check(self, health_file: Path):
        """Add health check results to report."""
        with open(health_file, "r") as f:
            data = json.load(f)
        
        self.reports.append({
            "type": "health_check",
            "data": data,
            "timestamp": datetime.now().isoformat()
        })
    
    def add_test_results(self, test_file: Path):
        """Add test results to report."""
        with open(test_file, "r") as f:
            data = json.load(f)
        
        self.reports.append({
            "type": "test_results",
            "data": data,
            "timestamp": datetime.now().isoformat()
        })
    
    def add_benchmark(self, benchmark_file: Path):
        """Add benchmark results to report."""
        with open(benchmark_file, "r") as f:
            data = json.load(f)
        
        self.reports.append({
            "type": "benchmark",
            "data": data,
            "timestamp": datetime.now().isoformat()
        })
    
    def add_dashboard_data(self, dashboard_file: Path):
        """Add dashboard data to report."""
        with open(dashboard_file, "r") as f:
            data = json.load(f)
        
        self.reports.append({
            "type": "dashboard",
            "data": data,
            "timestamp": datetime.now().isoformat()
        })
    
    def generate_html_report(self, output_file: Path):
        """Generate HTML report."""
        html = """
<!DOCTYPE html>
<html>
<head>
    <title>API Comprehensive Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .header { background: #4CAF50; color: white; padding: 20px; border-radius: 5px; }
        .section { background: white; margin: 20px 0; padding: 20px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .success { color: green; }
        .error { color: red; }
        .warning { color: orange; }
        table { width: 100%; border-collapse: collapse; margin-top: 10px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #4CAF50; color: white; }
        .metric { display: inline-block; margin: 10px; padding: 10px; background: #f0f0f0; border-radius: 5px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 API Comprehensive Report</h1>
        <p>Generated: {timestamp}</p>
    </div>
"""
        
        for report in self.reports:
            report_type = report["type"]
            data = report["data"]
            
            if report_type == "health_check":
                html += f"""
    <div class="section">
        <h2>🔍 Health Check Results</h2>
        <p><strong>Overall Status:</strong> <span class="{'success' if data.get('overall_status') == 'healthy' else 'error'}">{data.get('overall_status', 'unknown').upper()}</span></p>
        <p><strong>Total Checks:</strong> {data.get('results', {}).get('total', 0)}</p>
        <p><strong>Healthy:</strong> {data.get('results', {}).get('healthy', 0)}</p>
        <p><strong>Degraded:</strong> {data.get('results', {}).get('degraded', 0)}</p>
        <p><strong>Unhealthy:</strong> {data.get('results', {}).get('unhealthy', 0)}</p>
    </div>
"""
            
            elif report_type == "test_results":
                summary = data.get("summary", {})
                html += f"""
    <div class="section">
        <h2>🧪 Test Results</h2>
        <div class="metric"><strong>Total:</strong> {summary.get('total', 0)}</div>
        <div class="metric success"><strong>Passed:</strong> {summary.get('passed', 0)}</div>
        <div class="metric error"><strong>Failed:</strong> {summary.get('failed', 0)}</div>
        <div class="metric warning"><strong>Skipped:</strong> {summary.get('skipped', 0)}</div>
    </div>
"""
            
            elif report_type == "benchmark":
                html += """
    <div class="section">
        <h2>🔥 Benchmark Results</h2>
        <table>
            <tr>
                <th>Endpoint</th>
                <th>Avg Time</th>
                <th>P95</th>
                <th>Success Rate</th>
            </tr>
"""
                for result in data.get("results", []):
                    html += f"""
            <tr>
                <td>{result.get('endpoint')}</td>
                <td>{result.get('avg_time', 0):.2f}ms</td>
                <td>{result.get('p95_time', 0):.2f}ms</td>
                <td>{result.get('success_rate', 0):.2f}%</td>
            </tr>
"""
                html += """
        </table>
    </div>
"""
            
            elif report_type == "dashboard":
                stats = data.get("statistics", {})
                html += f"""
    <div class="section">
        <h2>📊 Dashboard Statistics</h2>
        <div class="metric"><strong>Total Requests:</strong> {stats.get('total_requests', 0)}</div>
        <div class="metric"><strong>Error Rate:</strong> {stats.get('error_rate', 0):.2f}%</div>
        <div class="metric"><strong>Availability:</strong> {stats.get('availability', 0):.2f}%</div>
        <div class="metric"><strong>Avg Response Time:</strong> {stats.get('response_time', {}).get('avg', 0):.2f}ms</div>
    </div>
"""
        
        html += f"""
</body>
</html>
"""
        
        html = html.replace("{timestamp}", datetime.now().isoformat())
        
        with open(output_file, "w") as f:
            f.write(html)
        
        print(f"✅ HTML report generated: {output_file}")
    
    def generate_json_report(self, output_file: Path):
        """Generate JSON report."""
        report = {
            "generated_at": datetime.now().isoformat(),
            "reports": self.reports
        }
        
        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"✅ JSON report generated: {output_file}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="API Reporter")
    parser.add_argument("--health", help="Health check JSON file")
    parser.add_argument("--tests", help="Test results JSON file")
    parser.add_argument("--benchmark", help="Benchmark JSON file")
    parser.add_argument("--dashboard", help="Dashboard JSON file")
    parser.add_argument("--output", required=True, help="Output file (html or json)")
    
    args = parser.parse_args()
    
    reporter = APIReporter()
    
    if args.health:
        reporter.add_health_check(Path(args.health))
    
    if args.tests:
        reporter.add_test_results(Path(args.tests))
    
    if args.benchmark:
        reporter.add_benchmark(Path(args.benchmark))
    
    if args.dashboard:
        reporter.add_dashboard_data(Path(args.dashboard))
    
    output_file = Path(args.output)
    
    if output_file.suffix == ".html":
        reporter.generate_html_report(output_file)
    else:
        reporter.generate_json_report(output_file)


if __name__ == "__main__":
    main()



