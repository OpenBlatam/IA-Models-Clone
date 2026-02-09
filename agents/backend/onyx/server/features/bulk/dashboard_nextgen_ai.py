"""
BUL Next-Gen AI Dashboard
========================

Dashboard de próxima generación con visualizaciones avanzadas de IA,
análisis de sentimientos, generación de imágenes y métricas en tiempo real.
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
import base64
from datetime import datetime, timedelta
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO

# Initialize Dash app with advanced features
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css",
        "https://cdnjs.cloudflare.com/ajax/libs/animate.css/4.1.1/animate.min.css"
    ],
    title="BUL Next-Gen AI Dashboard",
    update_title="Loading...",
    suppress_callback_exceptions=True
)

# API Configuration
API_BASE_URL = "http://localhost:8000"
REFRESH_INTERVAL = 2000  # 2 seconds for real-time updates

# Layout
app.layout = dbc.Container([
    # Header with AI indicators
    dbc.Row([
        dbc.Col([
            html.H1([
                html.I(className="fas fa-brain me-2"),
                "BUL Next-Gen AI Dashboard"
            ], className="text-center mb-4 animate__animated animate__fadeInDown"),
            html.Div([
                dbc.Badge("🤖 AI Active", color="success", className="me-2"),
                dbc.Badge(id="ai-models-badge", color="info", className="me-2"),
                dbc.Badge(id="sentiment-badge", color="warning", className="me-2"),
                dbc.Badge(id="image-gen-badge", color="danger", className="me-2")
            ], className="text-center mb-3")
        ])
    ]),
    
    # Navigation tabs
    dbc.Row([
        dbc.Col([
            dbc.Tabs([
                dbc.Tab(label="🧠 AI Overview", tab_id="ai-overview"),
                dbc.Tab(label="📊 Analytics", tab_id="analytics"),
                dbc.Tab(label="🎨 Image Generation", tab_id="image-gen"),
                dbc.Tab(label="📝 Documents", tab_id="documents"),
                dbc.Tab(label="🔍 AI Analysis", tab_id="ai-analysis"),
                dbc.Tab(label="⚙️ AI Settings", tab_id="ai-settings")
            ], id="main-tabs", active_tab="ai-overview")
        ])
    ], className="mb-4"),
    
    # Tab content
    html.Div(id="tab-content"),
    
    # Auto-refresh
    dcc.Interval(
        id='interval-component',
        interval=REFRESH_INTERVAL,
        n_intervals=0
    ),
    
    # Stores
    dcc.Store(id="ai-models-store", data={}),
    dcc.Store(id="sentiment-data-store", data={}),
    dcc.Store(id="analytics-store", data={})
], fluid=True)

# AI Overview Tab Content
ai_overview_content = dbc.Container([
    # AI Models Status Cards
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("GPT-4", className="card-title"),
                    html.H2(id="gpt4-status", className="text-success"),
                    html.P("OpenAI Model", className="card-text")
                ])
            ])
        ], width=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Claude", className="card-title"),
                    html.H2(id="claude-status", className="text-info"),
                    html.P("Anthropic Model", className="card-text")
                ])
            ])
        ], width=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Gemini", className="card-title"),
                    html.H2(id="gemini-status", className="text-warning"),
                    html.P("Google Model", className="card-text")
                ])
            ])
        ], width=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Llama", className="card-title"),
                    html.H2(id="llama-status", className="text-primary"),
                    html.P("Meta Model", className="card-text")
                ])
            ])
        ], width=3)
    ], className="mb-4"),
    
    # AI Performance Charts
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("AI Model Usage"),
                dbc.CardBody([
                    dcc.Graph(id="ai-model-usage-chart")
                ])
            ])
        ], width=6),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("AI Processing Time"),
                dbc.CardBody([
                    dcc.Graph(id="ai-processing-time-chart")
                ])
            ])
        ], width=6)
    ], className="mb-4"),
    
    # AI Capabilities
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("AI Capabilities"),
                dbc.CardBody([
                    html.Div(id="ai-capabilities")
                ])
            ])
        ], width=12)
    ])
])

# Analytics Tab Content
analytics_content = dbc.Container([
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Sentiment Analysis"),
                dbc.CardBody([
                    dcc.Graph(id="sentiment-pie-chart")
                ])
            ])
        ], width=6),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Sentiment Trends"),
                dbc.CardBody([
                    dcc.Graph(id="sentiment-trends-chart")
                ])
            ])
        ], width=6)
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Top Keywords"),
                dbc.CardBody([
                    dcc.Graph(id="keywords-wordcloud")
                ])
            ])
        ], width=6),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Keyword Frequency"),
                dbc.CardBody([
                    dcc.Graph(id="keywords-bar-chart")
                ])
            ])
        ], width=6)
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Document Analytics"),
                dbc.CardBody([
                    dash_table.DataTable(
                        id="document-analytics-table",
                        columns=[
                            {"name": "Document ID", "id": "document_id"},
                            {"name": "AI Model", "id": "ai_model"},
                            {"name": "Sentiment", "id": "sentiment"},
                            {"name": "Keywords", "id": "keywords"},
                            {"name": "Processing Time", "id": "processing_time"}
                        ],
                        style_cell={'textAlign': 'left'}
                    )
                ])
            ])
        ], width=12)
    ])
])

# Image Generation Tab Content
image_gen_content = dbc.Container([
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("AI Image Generation"),
                dbc.CardBody([
                    dbc.Form([
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Image Prompt"),
                                dbc.Textarea(
                                    id="image-prompt",
                                    placeholder="Describe the image you want to generate...",
                                    rows=3
                                )
                            ], width=8),
                            dbc.Col([
                                dbc.Label("Style"),
                                dbc.Select(
                                    id="image-style",
                                    options=[
                                        {"label": "Realistic", "value": "realistic"},
                                        {"label": "Artistic", "value": "artistic"},
                                        {"label": "Cartoon", "value": "cartoon"},
                                        {"label": "Abstract", "value": "abstract"},
                                        {"label": "Minimalist", "value": "minimalist"}
                                    ],
                                    value="realistic"
                                )
                            ], width=4)
                        ], className="mb-3"),
                        dbc.Row([
                            dbc.Col([
                                dbc.Button("Generate Image", id="generate-image-btn", color="success", size="lg")
                            ])
                        ])
                    ])
                ])
            ])
        ], width=12)
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Generated Images"),
                dbc.CardBody([
                    html.Div(id="generated-images")
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
                    html.H5("Next-Gen Document Generation"),
                    dbc.Button("Generate Document", id="generate-doc-btn", color="primary", className="float-end")
                ]),
                dbc.CardBody([
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
                                dbc.Label("AI Model"),
                                dbc.Select(
                                    id="ai-model-select",
                                    options=[
                                        {"label": "GPT-4", "value": "gpt4"},
                                        {"label": "Claude", "value": "claude"},
                                        {"label": "Gemini", "value": "gemini"},
                                        {"label": "Llama", "value": "llama"}
                                    ],
                                    value="gpt4"
                                )
                            ], width=4)
                        ], className="mb-3"),
                        dbc.Row([
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
                            ], width=4),
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
                                dbc.Label("AI Features"),
                                dbc.Checklist(
                                    id="ai-features",
                                    options=[
                                        {"label": "Generate Image", "value": "image"},
                                        {"label": "Sentiment Analysis", "value": "sentiment"},
                                        {"label": "Keyword Extraction", "value": "keywords"},
                                        {"label": "Blockchain", "value": "blockchain"}
                                    ],
                                    value=["sentiment", "keywords"]
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
                            {"name": "AI Model", "id": "ai_model"},
                            {"name": "Status", "id": "status"},
                            {"name": "Progress", "id": "progress"},
                            {"name": "Sentiment", "id": "sentiment"},
                            {"name": "Created", "id": "created_at"}
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

# AI Analysis Tab Content
ai_analysis_content = dbc.Container([
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("AI Text Analysis"),
                dbc.CardBody([
                    dbc.Form([
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Text to Analyze"),
                                dbc.Textarea(
                                    id="analysis-text",
                                    placeholder="Enter text for AI analysis...",
                                    rows=5
                                )
                            ], width=8),
                            dbc.Col([
                                dbc.Label("Analysis Types"),
                                dbc.Checklist(
                                    id="analysis-types",
                                    options=[
                                        {"label": "Sentiment Analysis", "value": "sentiment"},
                                        {"label": "Keyword Extraction", "value": "keywords"},
                                        {"label": "Text Embeddings", "value": "embeddings"},
                                        {"label": "Summary", "value": "summary"}
                                    ],
                                    value=["sentiment", "keywords"]
                                ),
                                dbc.Label("AI Model"),
                                dbc.Select(
                                    id="analysis-model",
                                    options=[
                                        {"label": "GPT-4", "value": "gpt4"},
                                        {"label": "Claude", "value": "claude"},
                                        {"label": "Gemini", "value": "gemini"}
                                    ],
                                    value="gpt4"
                                )
                            ], width=4)
                        ], className="mb-3"),
                        dbc.Row([
                            dbc.Col([
                                dbc.Button("Analyze Text", id="analyze-text-btn", color="primary", size="lg")
                            ])
                        ])
                    ])
                ])
            ])
        ], width=12)
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Analysis Results"),
                dbc.CardBody([
                    html.Div(id="analysis-results")
                ])
            ])
        ], width=12)
    ])
])

# AI Settings Tab Content
ai_settings_content = dbc.Container([
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("AI Model Configuration"),
                dbc.CardBody([
                    dbc.Form([
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Default AI Model"),
                                dbc.Select(
                                    id="default-ai-model",
                                    options=[
                                        {"label": "GPT-4", "value": "gpt4"},
                                        {"label": "Claude", "value": "claude"},
                                        {"label": "Gemini", "value": "gemini"},
                                        {"label": "Llama", "value": "llama"}
                                    ],
                                    value="gpt4"
                                )
                            ], width=6),
                            dbc.Col([
                                dbc.Label("Max Tokens"),
                                dbc.Input(id="max-tokens", value="2000", type="number")
                            ], width=6)
                        ], className="mb-3"),
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Temperature"),
                                dbc.Input(id="temperature", value="0.7", type="number", step="0.1", min="0", max="2")
                            ], width=6),
                            dbc.Col([
                                dbc.Label("Auto Analysis"),
                                dbc.Checklist(
                                    id="auto-analysis",
                                    options=[
                                        {"label": "Enable automatic sentiment analysis", "value": "sentiment"},
                                        {"label": "Enable automatic keyword extraction", "value": "keywords"},
                                        {"label": "Enable automatic image generation", "value": "image"}
                                    ],
                                    value=["sentiment", "keywords"]
                                )
                            ], width=6)
                        ], className="mb-3"),
                        dbc.Row([
                            dbc.Col([
                                dbc.Button("Save Settings", id="save-ai-settings-btn", color="success")
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
    if active_tab == "ai-overview":
        return ai_overview_content
    elif active_tab == "analytics":
        return analytics_content
    elif active_tab == "image-gen":
        return image_gen_content
    elif active_tab == "documents":
        return documents_content
    elif active_tab == "ai-analysis":
        return ai_analysis_content
    elif active_tab == "ai-settings":
        return ai_settings_content
    else:
        return ai_overview_content

@app.callback(
    [Output('gpt4-status', 'children'),
     Output('claude-status', 'children'),
     Output('gemini-status', 'children'),
     Output('llama-status', 'children'),
     Output('ai-models-badge', 'children')],
    [Input('interval-component', 'n_intervals')]
)
def update_ai_model_status(n):
    """Update AI model status."""
    try:
        # Get AI models info
        response = requests.get(f"{API_BASE_URL}/ai/models", timeout=5)
        models_data = response.json()
        
        models = models_data.get("models", {})
        
        gpt4_status = "🟢 Active" if "gpt4" in models else "🔴 Offline"
        claude_status = "🟢 Active" if "claude" in models else "🔴 Offline"
        gemini_status = "🟢 Active" if "gemini" in models else "🔴 Offline"
        llama_status = "🟢 Active" if "llama" in models else "🔴 Offline"
        
        active_models = len([m for m in models.values() if m])
        models_badge = f"🤖 {active_models}/4"
        
        return gpt4_status, claude_status, gemini_status, llama_status, models_badge
        
    except Exception as e:
        return "🔴 Offline", "🔴 Offline", "🔴 Offline", "🔴 Offline", "🤖 0/4"

@app.callback(
    Output('ai-model-usage-chart', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_ai_model_usage_chart(n):
    """Update AI model usage chart."""
    try:
        # Mock data for AI model usage
        models = ['GPT-4', 'Claude', 'Gemini', 'Llama']
        usage = [45, 30, 15, 10]  # Mock usage percentages
        
        fig = go.Figure(data=[go.Pie(
            labels=models,
            values=usage,
            hole=0.3,
            marker_colors=['#28a745', '#17a2b8', '#ffc107', '#6f42c1']
        )])
        
        fig.update_layout(
            title="AI Model Usage Distribution",
            showlegend=True
        )
        
        return fig
        
    except Exception as e:
        return go.Figure()

@app.callback(
    Output('sentiment-pie-chart', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_sentiment_pie_chart(n):
    """Update sentiment analysis pie chart."""
    try:
        # Get sentiment analytics
        response = requests.get(f"{API_BASE_URL}/analytics/sentiment", timeout=5)
        sentiment_data = response.json()
        
        sentiment_dist = sentiment_data.get("sentiment_distribution", {})
        
        if sentiment_dist:
            labels = list(sentiment_dist.keys())
            values = list(sentiment_dist.values())
            
            colors = {
                'POSITIVE': '#28a745',
                'NEGATIVE': '#dc3545',
                'NEUTRAL': '#6c757d'
            }
            
            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                hole=0.3,
                marker_colors=[colors.get(label, '#6c757d') for label in labels]
            )])
        else:
            fig = go.Figure()
        
        fig.update_layout(
            title="Sentiment Analysis Distribution",
            showlegend=True
        )
        
        return fig
        
    except Exception as e:
        return go.Figure()

@app.callback(
    Output('keywords-bar-chart', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_keywords_bar_chart(n):
    """Update keywords bar chart."""
    try:
        # Get keyword analytics
        response = requests.get(f"{API_BASE_URL}/analytics/keywords", timeout=5)
        keyword_data = response.json()
        
        top_keywords = keyword_data.get("top_keywords", [])[:10]
        
        if top_keywords:
            keywords = [item[0] for item in top_keywords]
            counts = [item[1] for item in top_keywords]
            
            fig = go.Figure(data=[go.Bar(
                x=keywords,
                y=counts,
                marker_color='#007bff'
            )])
        else:
            fig = go.Figure()
        
        fig.update_layout(
            title="Top Keywords Frequency",
            xaxis_title="Keywords",
            yaxis_title="Frequency"
        )
        
        return fig
        
    except Exception as e:
        return go.Figure()

@app.callback(
    Output('analysis-results', 'children'),
    [Input('analyze-text-btn', 'n_clicks')],
    [State('analysis-text', 'value'),
     State('analysis-types', 'value'),
     State('analysis-model', 'value')]
)
def analyze_text(n_clicks, text, analysis_types, model):
    """Analyze text with AI."""
    if not n_clicks or not text:
        return html.P("Enter text and click 'Analyze Text' to see results.")
    
    try:
        # Send analysis request
        response = requests.post(f"{API_BASE_URL}/ai/analyze", 
                               json={
                                   "text": text,
                                   "analysis_types": analysis_types,
                                   "model": model
                               }, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            
            results = []
            
            if data.get("sentiment"):
                sentiment = data["sentiment"]
                results.append(
                    dbc.Alert([
                        html.H6("Sentiment Analysis", className="alert-heading"),
                        html.P(f"Sentiment: {sentiment['sentiment']}"),
                        html.P(f"Confidence: {sentiment['confidence']:.2f}"),
                        html.P(f"Polarity: {sentiment['polarity']:.2f}")
                    ], color="info", className="mb-3")
                )
            
            if data.get("keywords"):
                keywords = data["keywords"]
                results.append(
                    dbc.Alert([
                        html.H6("Keywords", className="alert-heading"),
                        html.P(", ".join(keywords[:10]))
                    ], color="success", className="mb-3")
                )
            
            if data.get("summary"):
                summary = data["summary"]
                results.append(
                    dbc.Alert([
                        html.H6("Summary", className="alert-heading"),
                        html.P(summary)
                    ], color="warning", className="mb-3")
                )
            
            return results
        else:
            return dbc.Alert("Error analyzing text", color="danger")
            
    except Exception as e:
        return dbc.Alert(f"Error: {str(e)}", color="danger")

@app.callback(
    Output('generated-images', 'children'),
    [Input('generate-image-btn', 'n_clicks')],
    [State('image-prompt', 'value'),
     State('image-style', 'value')]
)
def generate_image(n_clicks, prompt, style):
    """Generate image with AI."""
    if not n_clicks or not prompt:
        return html.P("Enter a prompt and click 'Generate Image' to create an AI image.")
    
    try:
        # Send image generation request
        response = requests.post(f"{API_BASE_URL}/ai/generate-image", 
                               params={"prompt": prompt, "style": style}, timeout=60)
        
        if response.status_code == 200:
            data = response.json()
            image_data = data.get("image_data")
            
            if image_data:
                return html.Div([
                    html.H6(f"Generated Image: {prompt[:50]}..."),
                    html.Img(src=image_data, style={"width": "100%", "max-width": "500px"}),
                    html.P(f"Style: {style}", className="text-muted")
                ])
            else:
                return dbc.Alert("No image generated", color="warning")
        else:
            return dbc.Alert("Error generating image", color="danger")
            
    except Exception as e:
        return dbc.Alert(f"Error: {str(e)}", color="danger")

# Run the app
if __name__ == '__main__':
    print("🚀 Starting BUL Next-Gen AI Dashboard...")
    print(f"📊 Dashboard will be available at: http://localhost:8050")
    print(f"🔗 API should be running at: {API_BASE_URL}")
    print("🤖 AI Features:")
    print("   - GPT-4, Claude, Gemini, Llama integration")
    print("   - Sentiment analysis")
    print("   - Image generation")
    print("   - Keyword extraction")
    print("   - Text embeddings")
    print("=" * 60)
    
    app.run_server(
        debug=True,
        host='0.0.0.0',
        port=8050,
        dev_tools_hot_reload=True
    )
