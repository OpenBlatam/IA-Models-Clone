"""
BUL API Dashboard
================

Dashboard web interactivo para monitorear la API BUL usando Dash.
Proporciona visualizaciones en tiempo real del estado del sistema.
"""

import dash
from dash import dcc, html, Input, Output, dash_table, callback
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import requests
import time
from datetime import datetime, timedelta
import json

# Initialize Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="BUL API Dashboard",
    update_title="Loading..."
)

# API Configuration
API_BASE_URL = "http://localhost:8000"
REFRESH_INTERVAL = 5000  # 5 seconds

# Layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("🚀 BUL API Dashboard", className="text-center mb-4"),
            html.Hr()
        ])
    ]),
    
    # Status Cards
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("System Status", className="card-title"),
                    html.H2(id="system-status", className="text-success"),
                    html.P("API Health", className="card-text")
                ])
            ])
        ], width=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Active Tasks", className="card-title"),
                    html.H2(id="active-tasks", className="text-primary"),
                    html.P("Currently Processing", className="card-text")
                ])
            ])
        ], width=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Total Requests", className="card-title"),
                    html.H2(id="total-requests", className="text-info"),
                    html.P("All Time", className="card-text")
                ])
            ])
        ], width=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Success Rate", className="card-title"),
                    html.H2(id="success-rate", className="text-warning"),
                    html.P("Completion Rate", className="card-text")
                ])
            ])
        ], width=3)
    ], className="mb-4"),
    
    # Charts Row
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Request Timeline"),
                dbc.CardBody([
                    dcc.Graph(id="request-timeline")
                ])
            ])
        ], width=6),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Task Status Distribution"),
                dbc.CardBody([
                    dcc.Graph(id="task-status-pie")
                ])
            ])
        ], width=6)
    ], className="mb-4"),
    
    # Tasks Table
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Recent Tasks"),
                dbc.CardBody([
                    dash_table.DataTable(
                        id="tasks-table",
                        columns=[
                            {"name": "Task ID", "id": "task_id"},
                            {"name": "Status", "id": "status"},
                            {"name": "Progress", "id": "progress"},
                            {"name": "Created", "id": "created_at"},
                            {"name": "User", "id": "user_id"}
                        ],
                        style_cell={'textAlign': 'left'},
                        style_data_conditional=[
                            {
                                'if': {'filter_query': '{status} = completed'},
                                'backgroundColor': '#d4edda',
                                'color': 'black',
                            },
                            {
                                'if': {'filter_query': '{status} = processing'},
                                'backgroundColor': '#d1ecf1',
                                'color': 'black',
                            },
                            {
                                'if': {'filter_query': '{status} = failed'},
                                'backgroundColor': '#f8d7da',
                                'color': 'black',
                            }
                        ]
                    )
                ])
            ])
        ], width=12)
    ], className="mb-4"),
    
    # System Metrics
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("System Metrics"),
                dbc.CardBody([
                    html.Div(id="system-metrics")
                ])
            ])
        ], width=12)
    ]),
    
    # Auto-refresh
    dcc.Interval(
        id='interval-component',
        interval=REFRESH_INTERVAL,
        n_intervals=0
    )
], fluid=True)

# Callbacks
@app.callback(
    [Output('system-status', 'children'),
     Output('active-tasks', 'children'),
     Output('total-requests', 'children'),
     Output('success-rate', 'children')],
    [Input('interval-component', 'n_intervals')]
)
def update_status_cards(n):
    """Update status cards with current API data."""
    try:
        # Get health status
        health_response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        health_data = health_response.json()
        
        # Get stats
        stats_response = requests.get(f"{API_BASE_URL}/stats", timeout=5)
        stats_data = stats_response.json()
        
        status = "🟢 Online" if health_data.get("status") == "healthy" else "🔴 Offline"
        active_tasks = health_data.get("active_tasks", 0)
        total_requests = stats_data.get("total_requests", 0)
        success_rate = f"{stats_data.get('success_rate', 0):.1%}"
        
        return status, active_tasks, total_requests, success_rate
        
    except Exception as e:
        return "🔴 Offline", "N/A", "N/A", "N/A"

@app.callback(
    Output('request-timeline', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_request_timeline(n):
    """Update request timeline chart."""
    try:
        # Get tasks data
        tasks_response = requests.get(f"{API_BASE_URL}/tasks", timeout=5)
        tasks_data = tasks_response.json()
        
        if not tasks_data.get("tasks"):
            return go.Figure()
        
        # Process data for timeline
        df = pd.DataFrame(tasks_data["tasks"])
        df['created_at'] = pd.to_datetime(df['created_at'])
        
        # Group by hour
        df['hour'] = df['created_at'].dt.floor('H')
        hourly_counts = df.groupby('hour').size().reset_index(name='count')
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hourly_counts['hour'],
            y=hourly_counts['count'],
            mode='lines+markers',
            name='Requests per Hour',
            line=dict(color='#007bff')
        ))
        
        fig.update_layout(
            title="Request Timeline",
            xaxis_title="Time",
            yaxis_title="Number of Requests",
            hovermode='x unified'
        )
        
        return fig
        
    except Exception as e:
        return go.Figure()

@app.callback(
    Output('task-status-pie', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_task_status_pie(n):
    """Update task status pie chart."""
    try:
        # Get tasks data
        tasks_response = requests.get(f"{API_BASE_URL}/tasks", timeout=5)
        tasks_data = tasks_response.json()
        
        if not tasks_data.get("tasks"):
            return go.Figure()
        
        df = pd.DataFrame(tasks_data["tasks"])
        status_counts = df['status'].value_counts()
        
        colors = {
            'completed': '#28a745',
            'processing': '#17a2b8',
            'queued': '#ffc107',
            'failed': '#dc3545',
            'cancelled': '#6c757d'
        }
        
        fig = go.Figure(data=[go.Pie(
            labels=status_counts.index,
            values=status_counts.values,
            marker_colors=[colors.get(status, '#6c757d') for status in status_counts.index]
        )])
        
        fig.update_layout(
            title="Task Status Distribution",
            showlegend=True
        )
        
        return fig
        
    except Exception as e:
        return go.Figure()

@app.callback(
    Output('tasks-table', 'data'),
    [Input('interval-component', 'n_intervals')]
)
def update_tasks_table(n):
    """Update tasks table."""
    try:
        # Get tasks data
        tasks_response = requests.get(f"{API_BASE_URL}/tasks", timeout=5)
        tasks_data = tasks_response.json()
        
        if not tasks_data.get("tasks"):
            return []
        
        # Format data for table
        tasks = tasks_data["tasks"][:20]  # Show last 20 tasks
        
        for task in tasks:
            # Format datetime
            if 'created_at' in task:
                task['created_at'] = datetime.fromisoformat(task['created_at'].replace('Z', '+00:00')).strftime('%Y-%m-%d %H:%M:%S')
            
            # Add progress bar
            task['progress'] = f"{task.get('progress', 0)}%"
        
        return tasks
        
    except Exception as e:
        return []

@app.callback(
    Output('system-metrics', 'children'),
    [Input('interval-component', 'n_intervals')]
)
def update_system_metrics(n):
    """Update system metrics display."""
    try:
        # Get stats
        stats_response = requests.get(f"{API_BASE_URL}/stats", timeout=5)
        stats_data = stats_response.json()
        
        metrics = [
            dbc.Row([
                dbc.Col([
                    html.H6("Average Processing Time"),
                    html.H4(f"{stats_data.get('average_processing_time', 0):.2f}s")
                ], width=3),
                dbc.Col([
                    html.H6("Cache Hit Rate"),
                    html.H4(f"{stats_data.get('cache_hit_rate', 0):.1%}")
                ], width=3),
                dbc.Col([
                    html.H6("System Uptime"),
                    html.H4(stats_data.get('uptime', 'N/A'))
                ], width=3),
                dbc.Col([
                    html.H6("Last Updated"),
                    html.H4(datetime.now().strftime('%H:%M:%S'))
                ], width=3)
            ])
        ]
        
        return metrics
        
    except Exception as e:
        return html.P("Unable to fetch metrics", className="text-muted")

# Run the app
if __name__ == '__main__':
    print("🚀 Starting BUL API Dashboard...")
    print(f"📊 Dashboard will be available at: http://localhost:8050")
    print(f"🔗 API should be running at: {API_BASE_URL}")
    print("=" * 50)
    
    app.run_server(
        debug=True,
        host='0.0.0.0',
        port=8050,
        dev_tools_hot_reload=True
    )
