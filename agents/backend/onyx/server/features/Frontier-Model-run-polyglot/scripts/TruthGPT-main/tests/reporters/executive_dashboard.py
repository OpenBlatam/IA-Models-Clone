"""
Executive Dashboard
High-level executive dashboard
"""

import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime, timedelta
from statistics import mean

class ExecutiveDashboard:
    """Generate executive dashboard"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.history_file = project_root / "test_history.json"
    
    def generate_executive_dashboard(
        self,
        period_days: int = 30,
        output_file: str = "executive_dashboard.html"
    ) -> Path:
        """Generate executive dashboard"""
        history = self._load_history()
        
        cutoff_date = (datetime.now() - timedelta(days=period_days)).isoformat()
        recent = [r for r in history if r.get('timestamp', '') >= cutoff_date]
        
        if not recent:
            return None
        
        # Calculate key metrics
        success_rates = [r.get('success_rate', 0) for r in recent]
        execution_times = [r.get('execution_time', 0) for r in recent]
        total_tests = [r.get('total_tests', 0) for r in recent]
        
        avg_success = mean(success_rates) if success_rates else 0
        avg_time = mean(execution_times) if execution_times else 0
        total_tests_sum = sum(total_tests)
        
        # Generate HTML
        output_path = self.project_root / output_file
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Executive Test Dashboard</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            min-height: 100vh;
            padding: 40px 20px;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        .header {{
            background: white;
            padding: 40px;
            border-radius: 15px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            text-align: center;
        }}
        .header h1 {{
            color: #1e3c72;
            font-size: 3em;
            margin-bottom: 10px;
        }}
        .header p {{
            color: #666;
            font-size: 1.2em;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 25px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            text-align: center;
            transition: transform 0.3s;
        }}
        .metric-card:hover {{
            transform: translateY(-5px);
        }}
        .metric-value {{
            font-size: 4em;
            font-weight: bold;
            color: #1e3c72;
            margin-bottom: 10px;
        }}
        .metric-label {{
            color: #666;
            font-size: 1.1em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .status-badge {{
            display: inline-block;
            padding: 10px 20px;
            border-radius: 25px;
            font-weight: bold;
            margin-top: 15px;
        }}
        .status-excellent {{
            background: #4CAF50;
            color: white;
        }}
        .status-good {{
            background: #2196F3;
            color: white;
        }}
        .status-fair {{
            background: #ff9800;
            color: white;
        }}
        .status-poor {{
            background: #f44336;
            color: white;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Executive Test Dashboard</h1>
            <p>Period: Last {period_days} days | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value">{avg_success:.1f}%</div>
                <div class="metric-label">Success Rate</div>
                <div class="status-badge status-{'excellent' if avg_success >= 95 else 'good' if avg_success >= 90 else 'fair' if avg_success >= 80 else 'poor'}">
                    {'Excellent' if avg_success >= 95 else 'Good' if avg_success >= 90 else 'Fair' if avg_success >= 80 else 'Needs Improvement'}
                </div>
            </div>
            
            <div class="metric-card">
                <div class="metric-value">{avg_time:.0f}s</div>
                <div class="metric-label">Avg Execution Time</div>
                <div class="status-badge status-{'excellent' if avg_time < 60 else 'good' if avg_time < 120 else 'fair' if avg_time < 300 else 'poor'}">
                    {'Excellent' if avg_time < 60 else 'Good' if avg_time < 120 else 'Fair' if avg_time < 300 else 'Needs Improvement'}
                </div>
            </div>
            
            <div class="metric-card">
                <div class="metric-value">{total_tests_sum:,}</div>
                <div class="metric-label">Total Tests Executed</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-value">{len(recent)}</div>
                <div class="metric-label">Test Runs</div>
            </div>
        </div>
    </div>
</body>
</html>"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        return output_path
    
    def _load_history(self) -> List[Dict]:
        """Load test history"""
        if not self.history_file.exists():
            return []
        
        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return []

def main():
    """Main function"""
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    
    dashboard = ExecutiveDashboard(project_root)
    dashboard_path = dashboard.generate_executive_dashboard(period_days=30)
    
    if dashboard_path:
        print(f"✅ Executive dashboard generated: {dashboard_path}")
    else:
        print("❌ Insufficient data for dashboard")

if __name__ == "__main__":
    main()







