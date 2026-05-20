"""
BUL Ultra Advanced Dashboard
============================

Dashboard ultra-avanzado con funcionalidades de colaboración en tiempo real,
notificaciones, templates y gestión de versiones.
"""

import dash
from dash import dcc, html, Input, Output, State, dash_table, callback, clientside_callback
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import requests
import time
import json
import asyncio
import websockets
from datetime import datetime, timedelta
import uuid

# Initialize Dash app with advanced features
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css"
    ],
    title="BUL Ultra Advanced Dashboard",
    update_title="Loading...",
    suppress_callback_exceptions=True
)

# API Configuration
API_BASE_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000"
REFRESH_INTERVAL = 3000  # 3 seconds for real-time updates

# Global state
app_state = {
    "user_id": "admin",
    "active_room": None,
    "notifications": [],
    "templates": [],
    "collaboration_rooms": []
}

# Layout
app.layout = dbc.Container([
    # Header with real-time indicators
    dbc.Row([
        dbc.Col([
            html.H1([
                html.I(className="fas fa-rocket me-2"),
                "BUL Ultra Advanced Dashboard"
            ], className="text-center mb-4"),
            html.Div([
                dbc.Badge("🟢 Online", color="success", className="me-2"),
                dbc.Badge(id="active-connections-badge", color="info", className="me-2"),
                dbc.Badge(id="collaboration-rooms-badge", color="warning", className="me-2"),
                dbc.Badge(id="notifications-count", color="danger", className="me-2")
            ], className="text-center mb-3")
        ])
    ]),
    
    # Navigation tabs
    dbc.Row([
        dbc.Col([
            dbc.Tabs([
                dbc.Tab(label="📊 Overview", tab_id="overview"),
                dbc.Tab(label="📄 Documents", tab_id="documents"),
                dbc.Tab(label="🤝 Collaboration", tab_id="collaboration"),
                dbc.Tab(label="📋 Templates", tab_id="templates"),
                dbc.Tab(label="🔔 Notifications", tab_id="notifications"),
                dbc.Tab(label="⚙️ Settings", tab_id="settings")
            ], id="main-tabs", active_tab="overview")
        ])
    ], className="mb-4"),
    
    # Tab content
    html.Div(id="tab-content"),
    
    # Real-time WebSocket status
    html.Div(id="websocket-status", style={"display": "none"}),
    
    # Auto-refresh
    dcc.Interval(
        id='interval-component',
        interval=REFRESH_INTERVAL,
        n_intervals=0
    ),
    
    # WebSocket connection
    dcc.Store(id="ws-store", data={"connected": False}),
    
    # Notification store
    dcc.Store(id="notification-store", data=[]),
    
    # Template store
    dcc.Store(id="template-store", data=[]),
    
    # Collaboration store
    dcc.Store(id="collaboration-store", data=[])
], fluid=True)

# Overview Tab Content
overview_content = dbc.Container([
    # System Status Cards
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
                    html.H4("WebSocket Connections", className="card-title"),
                    html.H2(id="ws-connections", className="text-info"),
                    html.P("Real-time Users", className="card-text")
                ])
            ])
        ], width=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Collaboration Rooms", className="card-title"),
                    html.H2(id="collaboration-count", className="text-warning"),
                    html.P("Active Rooms", className="card-text")
                ])
            ])
        ], width=3)
    ], className="mb-4"),
    
    # Charts Row
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Real-time Request Timeline"),
                dbc.CardBody([
                    dcc.Graph(id="realtime-timeline")
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
    
    # System Metrics
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("System Performance"),
                dbc.CardBody([
                    html.Div(id="performance-metrics")
                ])
            ])
        ], width=12)
    ])
])

# Documents Tab Content
documents_content = dbc.Container([
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H5("Document Generation"),
                    dbc.Button("Generate New Document", id="generate-doc-btn", color="primary", className="float-end")
                ]),
                dbc.CardBody([
                    # Document generation form
                    dbc.Form([
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Query"),
                                dbc.Textarea(
                                    id="doc-query",
                                    placeholder="Describe what document you want to generate...",
                                    rows=3
                                )
                            ], width=8),
                            dbc.Col([
                                dbc.Label("Business Area"),
                                dbc.Select(
                                    id="doc-business-area",
                                    options=[
                                        {"label": "Marketing", "value": "marketing"},
                                        {"label": "Sales", "value": "sales"},
                                        {"label": "Operations", "value": "operations"},
                                        {"label": "HR", "value": "hr"},
                                        {"label": "Finance", "value": "finance"},
                                        {"label": "Strategy", "value": "strategy"}
                                    ],
                                    value="marketing"
                                )
                            ], width=4)
                        ], className="mb-3"),
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Document Type"),
                                dbc.Select(
                                    id="doc-type",
                                    options=[
                                        {"label": "Strategy", "value": "strategy"},
                                        {"label": "Report", "value": "report"},
                                        {"label": "Proposal", "value": "proposal"},
                                        {"label": "Plan", "value": "plan"},
                                        {"label": "Manual", "value": "manual"}
                                    ],
                                    value="strategy"
                                )
                            ], width=4),
                            dbc.Col([
                                dbc.Label("Template"),
                                dbc.Select(
                                    id="doc-template",
                                    options=[],
                                    placeholder="Select template (optional)"
                                )
                            ], width=4),
                            dbc.Col([
                                dbc.Label("Collaboration"),
                                dbc.Checklist(
                                    id="doc-collaboration",
                                    options=[{"label": "Enable real-time collaboration", "value": "enabled"}],
                                    value=[]
                                )
                            ], width=4)
                        ], className="mb-3"),
                        dbc.Row([
                            dbc.Col([
                                dbc.Button("Generate Document", id="submit-doc-btn", color="success", size="lg")
                            ])
                        ])
                    ])
                ])
            ])
        ], width=12)
    ], className="mb-4"),
    
    # Recent Documents
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Recent Documents"),
                dbc.CardBody([
                    dash_table.DataTable(
                        id="documents-table",
                        columns=[
                            {"name": "Task ID", "id": "task_id"},
                            {"name": "Title", "id": "title"},
                            {"name": "Status", "id": "status"},
                            {"name": "Progress", "id": "progress"},
                            {"name": "Created", "id": "created_at"},
                            {"name": "Collaboration", "id": "collaboration"},
                            {"name": "Actions", "id": "actions"}
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
    ])
])

# Collaboration Tab Content
collaboration_content = dbc.Container([
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H5("Collaboration Rooms"),
                    dbc.Button("Create Room", id="create-room-btn", color="primary", className="float-end")
                ]),
                dbc.CardBody([
                    dash_table.DataTable(
                        id="rooms-table",
                        columns=[
                            {"name": "Room ID", "id": "room_id"},
                            {"name": "Name", "id": "name"},
                            {"name": "Document", "id": "document_id"},
                            {"name": "Participants", "id": "participants"},
                            {"name": "Status", "id": "status"},
                            {"name": "Actions", "id": "actions"}
                        ],
                        style_cell={'textAlign': 'left'}
                    )
                ])
            ])
        ], width=12)
    ], className="mb-4"),
    
    # Real-time collaboration area
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Real-time Collaboration"),
                dbc.CardBody([
                    html.Div(id="collaboration-area", children=[
                        html.P("Select a collaboration room to start working together in real-time.")
                    ])
                ])
            ])
        ], width=12)
    ])
])

# Templates Tab Content
templates_content = dbc.Container([
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H5("Document Templates"),
                    dbc.Button("Create Template", id="create-template-btn", color="primary", className="float-end")
                ]),
                dbc.CardBody([
                    dash_table.DataTable(
                        id="templates-table",
                        columns=[
                            {"name": "Name", "id": "name"},
                            {"name": "Description", "id": "description"},
                            {"name": "Business Area", "id": "business_area"},
                            {"name": "Document Type", "id": "document_type"},
                            {"name": "Version", "id": "version"},
                            {"name": "Public", "id": "is_public"},
                            {"name": "Actions", "id": "actions"}
                        ],
                        style_cell={'textAlign': 'left'}
                    )
                ])
            ])
        ], width=12)
    ])
])

# Notifications Tab Content
notifications_content = dbc.Container([
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H5("Notifications"),
                    dbc.Button("Mark All Read", id="mark-all-read-btn", color="secondary", className="float-end")
                ]),
                dbc.CardBody([
                    html.Div(id="notifications-list")
                ])
            ])
        ], width=12)
    ])
])

# Settings Tab Content
settings_content = dbc.Container([
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("System Settings"),
                dbc.CardBody([
                    dbc.Form([
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("User ID"),
                                dbc.Input(id="user-id-input", value="admin", type="text")
                            ], width=6),
                            dbc.Col([
                                dbc.Label("API Key"),
                                dbc.Input(id="api-key-input", value="admin_key_ultra_123", type="password")
                            ], width=6)
                        ], className="mb-3"),
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Refresh Interval (seconds)"),
                                dbc.Input(id="refresh-interval", value="3", type="number", min=1, max=60)
                            ], width=6),
                            dbc.Col([
                                dbc.Label("Auto-refresh"),
                                dbc.Checklist(
                                    id="auto-refresh",
                                    options=[{"label": "Enable auto-refresh", "value": "enabled"}],
                                    value=["enabled"]
                                )
                            ], width=6)
                        ], className="mb-3"),
                        dbc.Row([
                            dbc.Col([
                                dbc.Button("Save Settings", id="save-settings-btn", color="success")
                            ])
                        ])
                    ])
                ])
            ])
        ], width=12)
    ])
])

# Callbacks
@app.callback(
    Output('tab-content', 'children'),
    [Input('main-tabs', 'active_tab')]
)
def render_tab_content(active_tab):
    """Render content based on active tab."""
    if active_tab == "overview":
        return overview_content
    elif active_tab == "documents":
        return documents_content
    elif active_tab == "collaboration":
        return collaboration_content
    elif active_tab == "templates":
        return templates_content
    elif active_tab == "notifications":
        return notifications_content
    elif active_tab == "settings":
        return settings_content
    else:
        return overview_content

@app.callback(
    [Output('system-status', 'children'),
     Output('active-tasks', 'children'),
     Output('ws-connections', 'children'),
     Output('collaboration-count', 'children'),
     Output('active-connections-badge', 'children'),
     Output('collaboration-rooms-badge', 'children')],
    [Input('interval-component', 'n_intervals')]
)
def update_status_cards(n):
    """Update status cards with current API data."""
    try:
        # Get health status
        health_response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        health_data = health_response.json()
        
        status = "🟢 Online" if health_data.get("status") == "healthy" else "🔴 Offline"
        active_tasks = health_data.get("active_tasks", 0)
        ws_connections = health_data.get("active_connections", 0)
        collaboration_rooms = health_data.get("collaboration_rooms", 0)
        
        return (
            status,
            active_tasks,
            ws_connections,
            collaboration_rooms,
            f"🔗 {ws_connections}",
            f"🤝 {collaboration_rooms}"
        )
        
    except Exception as e:
        return "🔴 Offline", "N/A", "N/A", "N/A", "🔗 0", "🤝 0"

@app.callback(
    Output('realtime-timeline', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_realtime_timeline(n):
    """Update real-time timeline chart."""
    try:
        # Get tasks data
        tasks_response = requests.get(f"{API_BASE_URL}/tasks", timeout=5)
        tasks_data = tasks_response.json()
        
        if not tasks_data.get("tasks"):
            return go.Figure()
        
        # Process data for timeline
        df = pd.DataFrame(tasks_data["tasks"])
        df['created_at'] = pd.to_datetime(df['created_at'])
        
        # Group by minute for real-time effect
        df['minute'] = df['created_at'].dt.floor('min')
        minute_counts = df.groupby('minute').size().reset_index(name='count')
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=minute_counts['minute'],
            y=minute_counts['count'],
            mode='lines+markers',
            name='Requests per Minute',
            line=dict(color='#007bff', width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title="Real-time Request Timeline",
            xaxis_title="Time",
            yaxis_title="Number of Requests",
            hovermode='x unified',
            showlegend=False
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
            marker_colors=[colors.get(status, '#6c757d') for status in status_counts.index],
            hole=0.3
        )])
        
        fig.update_layout(
            title="Task Status Distribution",
            showlegend=True
        )
        
        return fig
        
    except Exception as e:
        return go.Figure()

@app.callback(
    Output('performance-metrics', 'children'),
    [Input('interval-component', 'n_intervals')]
)
def update_performance_metrics(n):
    """Update performance metrics display."""
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

@app.callback(
    Output('notifications-count', 'children'),
    [Input('interval-component', 'n_intervals')]
)
def update_notifications_count(n):
    """Update notifications count badge."""
    try:
        notifications_response = requests.get(f"{API_BASE_URL}/notifications/admin", params={"unread_only": True}, timeout=5)
        notifications_data = notifications_response.json()
        
        count = len(notifications_data)
        return f"🔔 {count}" if count > 0 else "🔔 0"
        
    except Exception as e:
        return "🔔 0"

@app.callback(
    Output('notifications-list', 'children'),
    [Input('interval-component', 'n_intervals')]
)
def update_notifications_list(n):
    """Update notifications list."""
    try:
        notifications_response = requests.get(f"{API_BASE_URL}/notifications/admin", timeout=5)
        notifications_data = notifications_response.json()
        
        if not notifications_data:
            return html.P("No notifications", className="text-muted")
        
        notifications = []
        for notif in notifications_data[:10]:  # Show last 10
            color = {
                'info': 'primary',
                'success': 'success',
                'warning': 'warning',
                'error': 'danger'
            }.get(notif['type'], 'primary')
            
            notifications.append(
                dbc.Alert([
                    html.H6(notif['title'], className="alert-heading"),
                    html.P(notif['message']),
                    html.Small(f"Created: {notif['created_at']}", className="text-muted")
                ], color=color, className="mb-2")
            )
        
        return notifications
        
    except Exception as e:
        return html.P("Unable to fetch notifications", className="text-muted")

@app.callback(
    Output('templates-table', 'data'),
    [Input('interval-component', 'n_intervals')]
)
def update_templates_table(n):
    """Update templates table."""
    try:
        templates_response = requests.get(f"{API_BASE_URL}/templates", timeout=5)
        templates_data = templates_response.json()
        
        return templates_data
        
    except Exception as e:
        return []

@app.callback(
    Output('doc-template', 'options'),
    [Input('doc-business-area', 'value')]
)
def update_template_options(business_area):
    """Update template options based on business area."""
    try:
        templates_response = requests.get(f"{API_BASE_URL}/templates", params={"business_area": business_area}, timeout=5)
        templates_data = templates_response.json()
        
        options = [{"label": "No template", "value": ""}]
        for template in templates_data:
            options.append({
                "label": f"{template['name']} (v{template['version']})",
                "value": template['id']
            })
        
        return options
        
    except Exception as e:
        return [{"label": "No template", "value": ""}]

# Run the app
if __name__ == '__main__':
    print("🚀 Starting BUL Ultra Advanced Dashboard...")
    print(f"📊 Dashboard will be available at: http://localhost:8050")
    print(f"🔗 API should be running at: {API_BASE_URL}")
    print(f"🔌 WebSocket will connect to: {WS_URL}")
    print("=" * 60)
    
    app.run_server(
        debug=True,
        host='0.0.0.0',
        port=8050,
        dev_tools_hot_reload=True
    )
