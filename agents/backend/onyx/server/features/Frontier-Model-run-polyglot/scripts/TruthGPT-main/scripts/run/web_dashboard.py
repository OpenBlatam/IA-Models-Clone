"""
Web Dashboard Server
Interactive web dashboard for test results
"""

from flask import Flask, render_template_string, jsonify, request
from pathlib import Path
import sys
import json

# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from tests.test_database import TestResultDatabase
    from tests.test_history import TestHistory
    from tests.statistics_aggregator import StatisticsAggregator
except ImportError as e:
    print(f"⚠️  Warning: Some dependencies unavailable: {e}")

app = Flask(__name__)

# Initialize components
db = TestResultDatabase(project_root / "test_results.db")
history = TestHistory()
aggregator = StatisticsAggregator(project_root)

# HTML Template
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TruthGPT Test Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        .header {
            background: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .header h1 {
            color: #333;
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .stat-card {
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .stat-value {
            font-size: 3em;
            font-weight: bold;
            color: #333;
            margin-bottom: 10px;
        }
        .stat-label {
            color: #666;
            font-size: 1.1em;
        }
        .chart-container {
            background: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .chart-container h2 {
            color: #333;
            margin-bottom: 20px;
        }
        .loading {
            text-align: center;
            padding: 50px;
            color: #666;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧪 TruthGPT Test Dashboard</h1>
            <p>Comprehensive Test Analytics and Monitoring</p>
        </div>
        
        <div class="stats-grid" id="statsGrid">
            <div class="stat-card">
                <div class="stat-value" id="totalRuns">-</div>
                <div class="stat-label">Total Runs</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="avgSuccess">-</div>
                <div class="stat-label">Avg Success Rate</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="totalTests">-</div>
                <div class="stat-label">Total Tests Run</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="avgTime">-</div>
                <div class="stat-label">Avg Execution Time</div>
            </div>
        </div>
        
        <div class="chart-container">
            <h2>Success Rate Trend</h2>
            <canvas id="successChart"></canvas>
        </div>
        
        <div class="chart-container">
            <h2>Execution Time Trend</h2>
            <canvas id="timeChart"></canvas>
        </div>
    </div>
    
    <script>
        // Load data
        fetch('/api/dashboard/stats')
            .then(r => r.json())
            .then(data => {
                document.getElementById('totalRuns').textContent = data.total_runs || 0;
                document.getElementById('avgSuccess').textContent = (data.avg_success_rate || 0).toFixed(1) + '%';
                document.getElementById('totalTests').textContent = data.total_tests || 0;
                document.getElementById('avgTime').textContent = (data.avg_time || 0).toFixed(2) + 's';
            });
        
        // Load charts
        fetch('/api/dashboard/history')
            .then(r => r.json())
            .then(data => {
                const labels = data.labels || [];
                const successRates = data.success_rates || [];
                const executionTimes = data.execution_times || [];
                
                // Success rate chart
                new Chart(document.getElementById('successChart'), {
                    type: 'line',
                    data: {
                        labels: labels,
                        datasets: [{
                            label: 'Success Rate (%)',
                            data: successRates,
                            borderColor: '#4CAF50',
                            backgroundColor: 'rgba(76, 175, 80, 0.1)'
                        }]
                    },
                    options: { responsive: true }
                });
                
                // Execution time chart
                new Chart(document.getElementById('timeChart'), {
                    type: 'line',
                    data: {
                        labels: labels,
                        datasets: [{
                            label: 'Execution Time (s)',
                            data: executionTimes,
                            borderColor: '#2196F3',
                            backgroundColor: 'rgba(33, 150, 243, 0.1)'
                        }]
                    },
                    options: { responsive: true }
                });
            });
    </script>
</body>
</html>
"""

@app.route('/')
def dashboard():
    """Main dashboard page"""
    return render_template_string(DASHBOARD_HTML)

@app.route('/api/dashboard/stats')
def dashboard_stats():
    """Get dashboard statistics"""
    stats = db.get_test_statistics()
    history_stats = history.get_statistics()
    
    return jsonify({
        'total_runs': stats.get('total_runs', 0),
        'avg_success_rate': stats.get('average_success_rate', 0),
        'total_tests': stats.get('total_tests_run', 0),
        'avg_time': history_stats.get('average_execution_time', 0) if history_stats else 0
    })

@app.route('/api/dashboard/history')
def dashboard_history():
    """Get history data for charts"""
    recent = history.get_recent_runs(20)
    
    labels = [r['timestamp'][:10] for r in recent]
    success_rates = [r.get('success_rate', 0) for r in recent]
    execution_times = [r.get('execution_time', 0) for r in recent]
    
    return jsonify({
        'labels': labels,
        'success_rates': success_rates,
        'execution_times': execution_times
    })

def run_dashboard(host='127.0.0.1', port=8080, debug=False):
    """Run the web dashboard"""
    print(f"🌐 Starting Test Dashboard on http://{host}:{port}")
    print(f"📊 Open in browser: http://{host}:{port}")
    print()
    app.run(host=host, port=port, debug=debug)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Results Web Dashboard')
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8080, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    run_dashboard(host=args.host, port=args.port, debug=args.debug)







