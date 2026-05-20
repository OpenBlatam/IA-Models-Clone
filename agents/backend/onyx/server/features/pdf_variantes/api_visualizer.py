#!/usr/bin/env python3
"""
API Visualizer
==============
Visualize API metrics and data.

⚠️ DEPRECATED: This file is deprecated. Consider migrating to the new tools structure.

For new code, use:
    from tools.manager import ToolManager
    manager = ToolManager()
    # Tools can be extended with visualization capabilities
"""
import warnings

warnings.warn(
    "api_visualizer.py is deprecated. Consider migrating to the new tools structure in tools/.",
    DeprecationWarning,
    stacklevel=2
)

import json
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime


class APIVisualizer:
    """API data visualizer."""
    
    def generate_metrics_chart(self, data: Dict[str, Any], output_file: Path):
        """Generate metrics chart HTML."""
        html = """
<!DOCTYPE html>
<html>
<head>
    <title>API Metrics Chart</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .chart-container { width: 800px; margin: 20px auto; }
    </style>
</head>
<body>
    <h1>📊 API Metrics Visualization</h1>
    <div class="chart-container">
        <canvas id="metricsChart"></canvas>
    </div>
    <script>
        const ctx = document.getElementById('metricsChart').getContext('2d');
        const chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: {labels},
                datasets: [{
                    label: 'Response Time (ms)',
                    data: {data},
                    borderColor: 'rgb(75, 192, 192)',
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
    </script>
</body>
</html>
"""
        
        # Extract data (simplified - would need actual time series data)
        labels = [f"Request {i+1}" for i in range(10)]
        data_points = [45.2, 42.1, 48.3, 44.5, 46.7, 43.2, 47.8, 45.9, 44.1, 46.3]
        
        html = html.replace("{labels}", json.dumps(labels))
        html = html.replace("{data}", json.dumps(data_points))
        
        with open(output_file, "w") as f:
            f.write(html)
        
        print(f"✅ Chart generated: {output_file}")
    
    def generate_status_dashboard(self, health_data: Dict[str, Any], output_file: Path):
        """Generate status dashboard HTML."""
        overall_status = health_data.get("overall_status", "unknown")
        status_color = {
            "healthy": "#4CAF50",
            "degraded": "#FF9800",
            "unhealthy": "#F44336"
        }.get(overall_status, "#9E9E9E")
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>API Status Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .dashboard {{ background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .status {{ font-size: 24px; padding: 20px; border-radius: 5px; color: white; background: {status_color}; }}
        .metric {{ display: inline-block; margin: 10px; padding: 15px; background: #f0f0f0; border-radius: 5px; min-width: 150px; }}
        .metric-value {{ font-size: 32px; font-weight: bold; }}
        .metric-label {{ color: #666; }}
    </style>
</head>
<body>
    <div class="dashboard">
        <h1>📊 API Status Dashboard</h1>
        <div class="status">
            Status: {overall_status.upper()}
        </div>
        <div style="margin-top: 20px;">
            <div class="metric">
                <div class="metric-value">{health_data.get('results', {}).get('healthy', 0)}</div>
                <div class="metric-label">Healthy</div>
            </div>
            <div class="metric">
                <div class="metric-value">{health_data.get('results', {}).get('degraded', 0)}</div>
                <div class="metric-label">Degraded</div>
            </div>
            <div class="metric">
                <div class="metric-value">{health_data.get('results', {}).get('unhealthy', 0)}</div>
                <div class="metric-label">Unhealthy</div>
            </div>
        </div>
    </div>
</body>
</html>
"""
        
        with open(output_file, "w") as f:
            f.write(html)
        
        print(f"✅ Dashboard generated: {output_file}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="API Visualizer")
    parser.add_argument("--health", help="Health check JSON file")
    parser.add_argument("--output", required=True, help="Output HTML file")
    
    args = parser.parse_args()
    
    visualizer = APIVisualizer()
    
    if args.health:
        with open(args.health, "r") as f:
            health_data = json.load(f)
        visualizer.generate_status_dashboard(health_data, Path(args.output))
    else:
        # Generate sample chart
        visualizer.generate_metrics_chart({}, Path(args.output))


if __name__ == "__main__":
    main()



