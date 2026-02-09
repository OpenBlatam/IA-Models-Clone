"""
Test Metrics Dashboard Generator
Generates HTML dashboard with test metrics and visualizations
"""

import json
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import html

class DashboardGenerator:
    """Generates HTML dashboard for test metrics"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
    
    def generate_dashboard(
        self,
        test_results: Dict[str, Any],
        history_data: List[Dict] = None,
        coverage_data: Dict = None,
        output_file: str = "test_dashboard.html"
    ) -> Path:
        """Generate comprehensive test dashboard"""
        output_path = self.project_root / output_file
        
        # Prepare data
        total = test_results.get('total_tests', 0)
        passed = test_results.get('passed', 0)
        failed = test_results.get('failed', 0)
        errors = test_results.get('errors', 0)
        skipped = test_results.get('skipped', 0)
        success_rate = test_results.get('success_rate', 0)
        execution_time = test_results.get('execution_time', 0)
        
        # Generate history chart data
        history_chart_data = self._prepare_history_chart(history_data) if history_data else None
        
        # Generate coverage chart data
        coverage_chart_data = self._prepare_coverage_chart(coverage_data) if coverage_data else None
        
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TruthGPT Test Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        .header {{
            background: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .header h1 {{
            color: #333;
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        .header .subtitle {{
            color: #666;
            font-size: 1.1em;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}
        .stat-card {{
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }}
        .stat-card:hover {{
            transform: translateY(-5px);
        }}
        .stat-card.passed {{
            border-top: 4px solid #4CAF50;
        }}
        .stat-card.failed {{
            border-top: 4px solid #f44336;
        }}
        .stat-card.error {{
            border-top: 4px solid #ff9800;
        }}
        .stat-card.skipped {{
            border-top: 4px solid #2196F3;
        }}
        .stat-card.total {{
            border-top: 4px solid #9C27B0;
        }}
        .stat-value {{
            font-size: 3em;
            font-weight: bold;
            color: #333;
            margin-bottom: 10px;
        }}
        .stat-label {{
            color: #666;
            font-size: 1.1em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .chart-container {{
            background: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .chart-container h2 {{
            color: #333;
            margin-bottom: 20px;
            font-size: 1.5em;
        }}
        .progress-container {{
            background: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .progress-bar {{
            width: 100%;
            height: 50px;
            background: #e0e0e0;
            border-radius: 25px;
            overflow: hidden;
            position: relative;
            margin-top: 20px;
        }}
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #4CAF50, #8BC34A);
            transition: width 1s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
            font-size: 1.2em;
        }}
        .footer {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            color: #666;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧪 TruthGPT Test Dashboard</h1>
            <div class="subtitle">Comprehensive Test Metrics and Analytics</div>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card total">
                <div class="stat-value">{total}</div>
                <div class="stat-label">Total Tests</div>
            </div>
            <div class="stat-card passed">
                <div class="stat-value">{passed}</div>
                <div class="stat-label">Passed</div>
            </div>
            <div class="stat-card failed">
                <div class="stat-value">{failed}</div>
                <div class="stat-label">Failed</div>
            </div>
            <div class="stat-card error">
                <div class="stat-value">{errors}</div>
                <div class="stat-label">Errors</div>
            </div>
            <div class="stat-card skipped">
                <div class="stat-value">{skipped}</div>
                <div class="stat-label">Skipped</div>
            </div>
        </div>
        
        <div class="progress-container">
            <h2>Success Rate</h2>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {success_rate}%">
                    {success_rate:.1f}%
                </div>
            </div>
        </div>
        
        {self._generate_history_chart(history_chart_data) if history_chart_data else ''}
        {self._generate_coverage_chart(coverage_chart_data) if coverage_chart_data else ''}
        
        <div class="chart-container">
            <h2>Test Results Distribution</h2>
            <canvas id="resultsChart"></canvas>
        </div>
        
        <div class="footer">
            <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            <p>Execution Time: {execution_time:.2f} seconds | Tests/Second: {(total / execution_time if execution_time > 0 else 0):.1f}</p>
        </div>
    </div>
    
    <script>
        // Results distribution chart
        const ctx = document.getElementById('resultsChart').getContext('2d');
        new Chart(ctx, {{
            type: 'doughnut',
            data: {{
                labels: ['Passed', 'Failed', 'Errors', 'Skipped'],
                datasets: [{{
                    data: [{passed}, {failed}, {errors}, {skipped}],
                    backgroundColor: [
                        '#4CAF50',
                        '#f44336',
                        '#ff9800',
                        '#2196F3'
                    ]
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    legend: {{
                        position: 'bottom'
                    }}
                }}
            }}
        }});
        
        {self._generate_history_chart_script(history_chart_data) if history_chart_data else ''}
        {self._generate_coverage_chart_script(coverage_chart_data) if coverage_chart_data else ''}
    </script>
</body>
</html>"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return output_path
    
    def _prepare_history_chart(self, history_data: List[Dict]) -> Dict:
        """Prepare history data for chart"""
        if not history_data:
            return None
        
        # Get last 10 runs
        recent = sorted(history_data, key=lambda x: x.get('timestamp', ''), reverse=True)[:10]
        recent.reverse()  # Oldest first
        
        return {
            'labels': [r.get('timestamp', '')[:10] for r in recent],
            'success_rates': [r.get('success_rate', 0) for r in recent],
            'execution_times': [r.get('execution_time', 0) for r in recent]
        }
    
    def _prepare_coverage_chart(self, coverage_data: Dict) -> Dict:
        """Prepare coverage data for chart"""
        if not coverage_data:
            return None
        
        return {
            'covered': coverage_data.get('covered_modules', 0),
            'uncovered': coverage_data.get('uncovered_modules', 0),
            'percentage': coverage_data.get('coverage_percentage', 0)
        }
    
    def _generate_history_chart(self, chart_data: Dict) -> str:
        """Generate history chart HTML"""
        if not chart_data:
            return ''
        
        return f"""
        <div class="chart-container">
            <h2>Test History Trend</h2>
            <canvas id="historyChart"></canvas>
        </div>
        """
    
    def _generate_coverage_chart(self, chart_data: Dict) -> str:
        """Generate coverage chart HTML"""
        if not chart_data:
            return ''
        
        return f"""
        <div class="chart-container">
            <h2>Test Coverage</h2>
            <canvas id="coverageChart"></canvas>
        </div>
        """
    
    def _generate_history_chart_script(self, chart_data: Dict) -> str:
        """Generate history chart JavaScript"""
        if not chart_data:
            return ''
        
        labels = json.dumps(chart_data['labels'])
        success_rates = json.dumps(chart_data['success_rates'])
        execution_times = json.dumps(chart_data['execution_times'])
        
        return f"""
        const historyCtx = document.getElementById('historyChart').getContext('2d');
        new Chart(historyCtx, {{
            type: 'line',
            data: {{
                labels: {labels},
                datasets: [{{
                    label: 'Success Rate (%)',
                    data: {success_rates},
                    borderColor: '#4CAF50',
                    backgroundColor: 'rgba(76, 175, 80, 0.1)',
                    yAxisID: 'y'
                }}, {{
                    label: 'Execution Time (s)',
                    data: {execution_times},
                    borderColor: '#2196F3',
                    backgroundColor: 'rgba(33, 150, 243, 0.1)',
                    yAxisID: 'y1'
                }}]
            }},
            options: {{
                responsive: true,
                scales: {{
                    y: {{
                        type: 'linear',
                        display: true,
                        position: 'left',
                    }},
                    y1: {{
                        type: 'linear',
                        display: true,
                        position: 'right',
                        grid: {{
                            drawOnChartArea: false,
                        }},
                    }}
                }}
            }}
        }});
        """
    
    def _generate_coverage_chart_script(self, chart_data: Dict) -> str:
        """Generate coverage chart JavaScript"""
        if not chart_data:
            return ''
        
        return f"""
        const coverageCtx = document.getElementById('coverageChart').getContext('2d');
        new Chart(coverageCtx, {{
            type: 'bar',
            data: {{
                labels: ['Covered', 'Uncovered'],
                datasets: [{{
                    label: 'Modules',
                    data: [{chart_data['covered']}, {chart_data['uncovered']}],
                    backgroundColor: ['#4CAF50', '#f44336']
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    legend: {{
                        display: false
                    }},
                    title: {{
                        display: true,
                        text: 'Coverage: {chart_data['percentage']:.1f}%'
                    }}
                }}
            }}
        }});
        """

def main():
    """Main function"""
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    
    # Example data
    test_results = {
        'total_tests': 204,
        'passed': 200,
        'failed': 2,
        'errors': 0,
        'skipped': 2,
        'success_rate': 98.0,
        'execution_time': 45.3
    }
    
    generator = DashboardGenerator(project_root)
    dashboard_path = generator.generate_dashboard(test_results)
    
    print(f"✅ Dashboard generated: {dashboard_path}")
    print(f"   Open in browser: file://{dashboard_path.absolute()}")

if __name__ == "__main__":
    main()







