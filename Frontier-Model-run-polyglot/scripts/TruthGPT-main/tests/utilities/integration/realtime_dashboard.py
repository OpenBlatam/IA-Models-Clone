"""
Real-time Test Metrics Dashboard
Dashboard that updates in real-time
"""

from flask import Flask, render_template_string, jsonify
from pathlib import Path
import sys
import json
import threading
import time

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from tests.test_database import TestResultDatabase
    from tests.test_history import TestHistory
except ImportError:
    pass

app = Flask(__name__)

# Initialize components
db = TestResultDatabase(project_root / "test_results.db")
history = TestHistory()

REALTIME_DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Real-time Test Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{ max-width: 1600px; margin: 0 auto; }}
        .header {{
            background: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .header h1 {{ color: #333; font-size: 2.5em; }}
        .status-indicator {{
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #4CAF50;
            margin-left: 10px;
            animation: pulse 2s infinite;
        }}
        @keyframes pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.5; }}
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}
        .stat-card {{
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .stat-value {{
            font-size: 3em;
            font-weight: bold;
            color: #333;
        }}
        .stat-label {{
            color: #666;
            margin-top: 10px;
        }}
        .chart-container {{
            background: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Real-time Test Dashboard<span class="status-indicator"></span></h1>
            <p>Live updates every 5 seconds</p>
        </div>
        
        <div class="stats-grid" id="statsGrid">
            <!-- Stats will be populated by JavaScript -->
        </div>
        
        <div class="chart-container">
            <h2>Success Rate (Live)</h2>
            <canvas id="liveChart"></canvas>
        </div>
    </div>
    
    <script>
        let chart = null;
        
        function updateDashboard() {{
            fetch('/api/realtime/stats')
                .then(r => r.json())
                .then(data => {{
                    // Update stats
                    const statsGrid = document.getElementById('statsGrid');
                    statsGrid.innerHTML = `
                        <div class="stat-card">
                            <div class="stat-value" style="color: #667eea;">${{data.total_runs || 0}}</div>
                            <div class="stat-label">Total Runs</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value" style="color: #4CAF50;">${{(data.avg_success_rate || 0).toFixed(1)}}%</div>
                            <div class="stat-label">Avg Success Rate</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value" style="color: #2196F3;">${{data.total_tests || 0}}</div>
                            <div class="stat-label">Total Tests</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value" style="color: #ff9800;">${{(data.avg_time || 0).toFixed(2)}}s</div>
                            <div class="stat-label">Avg Time</div>
                        </div>
                    `;
                    
                    // Update chart
                    if (!chart) {{
                        chart = new Chart(document.getElementById('liveChart'), {{
                            type: 'line',
                            data: {{
                                labels: data.labels || [],
                                datasets: [{{
                                    label: 'Success Rate (%)',
                                    data: data.success_rates || [],
                                    borderColor: '#4CAF50',
                                    backgroundColor: 'rgba(76, 175, 80, 0.1)'
                                }}]
                            }},
                            options: {{
                                responsive: true,
                                animation: {{ duration: 0 }},
                                scales: {{
                                    y: {{ beginAtZero: true, max: 100 }}
                                }}
                            }}
                        }});
                    }} else {{
                        chart.data.labels = data.labels || [];
                        chart.data.datasets[0].data = data.success_rates || [];
                        chart.update('none');
                    }}
                }})
                .catch(err => console.error('Error updating dashboard:', err));
        }}
        
        // Update every 5 seconds
        updateDashboard();
        setInterval(updateDashboard, 5000);
    </script>
</body>
</html>
"""

@app.route('/')
def dashboard():
    """Real-time dashboard"""
    return render_template_string(REALTIME_DASHBOARD_HTML)

@app.route('/api/realtime/stats')
def realtime_stats():
    """Get real-time statistics"""
    stats = db.get_test_statistics()
    history_stats = history.get_statistics()
    recent = history.get_recent_runs(20)
    
    labels = [r['timestamp'][:10] for r in recent]
    success_rates = [r.get('success_rate', 0) for r in recent]
    
    return jsonify({
        'total_runs': stats.get('total_runs', 0),
        'avg_success_rate': stats.get('average_success_rate', 0),
        'total_tests': stats.get('total_tests_run', 0),
        'avg_time': history_stats.get('average_execution_time', 0) if history_stats else 0,
        'labels': labels,
        'success_rates': success_rates
    })

def run_realtime_dashboard(host='127.0.0.1', port=8081, debug=False):
    """Run real-time dashboard"""
    print(f"🌐 Starting Real-time Dashboard on http://{host}:{port}")
    print(f"📊 Updates every 5 seconds")
    print()
    app.run(host=host, port=port, debug=debug)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-time Test Dashboard')
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8081, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    run_realtime_dashboard(host=args.host, port=args.port, debug=args.debug)







