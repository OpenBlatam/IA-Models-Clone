"""
BUL - Business Universal Language (Advanced Dashboard)
=====================================================

Real-time advanced dashboard for BUL system monitoring and control.
"""

import dash
from dash import dcc, html, Input, Output, State, dash_table, callback
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import requests
import json
import time
from datetime import datetime, timedelta
import threading
import queue
import asyncio
from typing import Dict, Any, List

# Initialize Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.themes.DARKLY],
    title="BUL Advanced Dashboard"
)

# Global data storage
dashboard_data = {
    "system_metrics": [],
    "ai_usage": [],
    "performance_data": [],
    "error_logs": [],
    "user_activity": []
}

# Data update queue
data_queue = queue.Queue()

def update_dashboard_data():
    """Update dashboard data from BUL system."""
    while True:
        try:
            # Fetch data from BUL system
            system_data = fetch_system_data()
            ai_data = fetch_ai_usage_data()
            performance_data = fetch_performance_data()
            error_data = fetch_error_data()
            user_data = fetch_user_activity()
            
            # Update global data
            dashboard_data["system_metrics"].append(system_data)
            dashboard_data["ai_usage"].append(ai_data)
            dashboard_data["performance_data"].append(performance_data)
            dashboard_data["error_logs"].extend(error_data)
            dashboard_data["user_activity"].append(user_data)
            
            # Keep only last 100 entries
            for key in dashboard_data:
                if isinstance(dashboard_data[key], list) and len(dashboard_data[key]) > 100:
                    dashboard_data[key] = dashboard_data[key][-100:]
            
            time.sleep(5)  # Update every 5 seconds
        except Exception as e:
            print(f"Error updating dashboard data: {e}")
            time.sleep(10)

def fetch_system_data():
    """Fetch system data from BUL API."""
    try:
        response = requests.get("http://localhost:8000/", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return {
                "timestamp": datetime.now(),
                "active_tasks": data.get("active_tasks", 0),
                "ai_models": len(data.get("ai_models", [])),
                "status": data.get("status", "unknown")
            }
    except:
        pass
    return {"timestamp": datetime.now(), "active_tasks": 0, "ai_models": 0, "status": "offline"}

def fetch_ai_usage_data():
    """Fetch AI usage data."""
    try:
        response = requests.get("http://localhost:8000/ai/divine-models", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return {
                "timestamp": datetime.now(),
                "models_available": len(data.get("models", {})),
                "default_model": data.get("default_model", "unknown")
            }
    except:
        pass
    return {"timestamp": datetime.now(), "models_available": 0, "default_model": "unknown"}

def fetch_performance_data():
    """Fetch performance data."""
    try:
        response = requests.get("http://localhost:8001/metrics/performance", timeout=5)
        if response.status_code == 200:
            data = response.json()
            current = data.get("current_metrics", {})
            return {
                "timestamp": datetime.now(),
                "cpu_percent": current.get("cpu_percent", 0),
                "memory_percent": current.get("memory_percent", 0),
                "response_time": current.get("response_time_avg", 0),
                "requests_per_second": current.get("requests_per_second", 0)
            }
    except:
        pass
    return {"timestamp": datetime.now(), "cpu_percent": 0, "memory_percent": 0, "response_time": 0, "requests_per_second": 0}

def fetch_error_data():
    """Fetch error data."""
    # Simulate error data
    return [{
        "timestamp": datetime.now(),
        "error_type": "API Error",
        "message": "Connection timeout",
        "severity": "warning"
    }]

def fetch_user_activity():
    """Fetch user activity data."""
    return {
        "timestamp": datetime.now(),
        "active_users": 5,
        "total_requests": 150,
        "success_rate": 95.5
    }

# Start data update thread
data_thread = threading.Thread(target=update_dashboard_data, daemon=True)
data_thread.start()

# Dashboard layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("BUL Advanced Dashboard", className="text-center mb-4"),
            html.Hr()
        ])
    ]),
    
    # System Status Cards
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("System Status", className="card-title"),
                    html.H2(id="system-status", className="text-success"),
                    html.P("Current system status")
                ])
            ], color="success", outline=True)
        ], width=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Active Tasks", className="card-title"),
                    html.H2(id="active-tasks", className="text-info"),
                    html.P("Currently processing")
                ])
            ], color="info", outline=True)
        ], width=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("AI Models", className="card-title"),
                    html.H2(id="ai-models", className="text-warning"),
                    html.P("Available models")
                ])
            ], color="warning", outline=True)
        ], width=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Uptime", className="card-title"),
                    html.H2(id="uptime", className="text-primary"),
                    html.P("System uptime")
                ])
            ], color="primary", outline=True)
        ], width=3)
    ], className="mb-4"),
    
    # Performance Charts
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("System Performance"),
                dbc.CardBody([
                    dcc.Graph(id="performance-chart")
                ])
            ])
        ], width=6),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("AI Usage"),
                dbc.CardBody([
                    dcc.Graph(id="ai-usage-chart")
                ])
            ])
        ], width=6)
    ], className="mb-4"),
    
    # Real-time Metrics
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Real-time Metrics"),
                dbc.CardBody([
                    dcc.Graph(id="realtime-metrics")
                ])
            ])
        ], width=12)
    ], className="mb-4"),
    
    # Error Logs
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Error Logs"),
                dbc.CardBody([
                    dash_table.DataTable(
                        id="error-logs-table",
                        columns=[
                            {"name": "Timestamp", "id": "timestamp"},
                            {"name": "Error Type", "id": "error_type"},
                            {"name": "Message", "id": "message"},
                            {"name": "Severity", "id": "severity"}
                        ],
                        data=[],
                        style_cell={'textAlign': 'left'},
                        style_header={'backgroundColor': 'rgb(30, 30, 30)', 'color': 'white'},
                        style_data={'backgroundColor': 'rgb(50, 50, 50)', 'color': 'white'}
                    )
                ])
            ])
        ], width=12)
    ], className="mb-4"),
    
    # Auto-refresh interval
    dcc.Interval(
        id='interval-component',
        interval=5*1000,  # Update every 5 seconds
        n_intervals=0
    )
], fluid=True)

# Callbacks
@app.callback(
    [Output('system-status', 'children'),
     Output('active-tasks', 'children'),
     Output('ai-models', 'children'),
     Output('uptime', 'children')],
    [Input('interval-component', 'n_intervals')]
)
def update_status_cards(n):
    """Update status cards."""
    if dashboard_data["system_metrics"]:
        latest = dashboard_data["system_metrics"][-1]
        status = latest.get("status", "offline")
        active_tasks = latest.get("active_tasks", 0)
        ai_models = latest.get("ai_models", 0)
        uptime = "24h 15m"  # Simulated uptime
    else:
        status = "offline"
        active_tasks = 0
        ai_models = 0
        uptime = "0h 0m"
    
    return status.title(), active_tasks, ai_models, uptime

@app.callback(
    Output('performance-chart', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_performance_chart(n):
    """Update performance chart."""
    if not dashboard_data["performance_data"]:
        return go.Figure()
    
    df = pd.DataFrame(dashboard_data["performance_data"])
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('CPU Usage', 'Memory Usage', 'Response Time', 'Requests/sec'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # CPU Usage
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['cpu_percent'], name='CPU %', line=dict(color='red')),
        row=1, col=1
    )
    
    # Memory Usage
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['memory_percent'], name='Memory %', line=dict(color='blue')),
        row=1, col=2
    )
    
    # Response Time
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['response_time'], name='Response Time', line=dict(color='green')),
        row=2, col=1
    )
    
    # Requests per second
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['requests_per_second'], name='RPS', line=dict(color='orange')),
        row=2, col=2
    )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        template="plotly_dark"
    )
    
    return fig

@app.callback(
    Output('ai-usage-chart', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_ai_usage_chart(n):
    """Update AI usage chart."""
    if not dashboard_data["ai_usage"]:
        return go.Figure()
    
    df = pd.DataFrame(dashboard_data["ai_usage"])
    
    fig = px.bar(
        df, x='timestamp', y='models_available',
        title='AI Models Available Over Time',
        template="plotly_dark"
    )
    
    fig.update_layout(height=400)
    return fig

@app.callback(
    Output('realtime-metrics', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_realtime_metrics(n):
    """Update real-time metrics."""
    if not dashboard_data["user_activity"]:
        return go.Figure()
    
    df = pd.DataFrame(dashboard_data["user_activity"])
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Active Users', 'Total Requests', 'Success Rate'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Active Users
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['active_users'], name='Active Users', line=dict(color='purple')),
        row=1, col=1
    )
    
    # Total Requests
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['total_requests'], name='Total Requests', line=dict(color='cyan')),
        row=1, col=2
    )
    
    # Success Rate
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['success_rate'], name='Success Rate', line=dict(color='yellow')),
        row=1, col=3
    )
    
    fig.update_layout(
        height=300,
        showlegend=False,
        template="plotly_dark"
    )
    
    return fig

@app.callback(
    Output('error-logs-table', 'data'),
    [Input('interval-component', 'n_intervals')]
)
def update_error_logs(n):
    """Update error logs table."""
    if not dashboard_data["error_logs"]:
        return []
    
    # Convert to format for DataTable
    error_data = []
    for error in dashboard_data["error_logs"][-10:]:  # Show last 10 errors
        error_data.append({
            "timestamp": error["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
            "error_type": error["error_type"],
            "message": error["message"],
            "severity": error["severity"]
        })
    
    return error_data

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)
