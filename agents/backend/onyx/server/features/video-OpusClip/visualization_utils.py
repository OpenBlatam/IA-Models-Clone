"""
Visualization Utilities for Video-OpusClip Demos

Comprehensive visualization tools for interactive demos:
- Real-time charts and plots
- Performance visualizations
- Training metrics visualization
- Viral analysis charts
- System monitoring dashboards
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import time
from datetime import datetime, timedelta
import json

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class DemoVisualizer:
    """Main visualization class for demos."""
    
    def __init__(self):
        self.color_scheme = {
            'primary': '#667eea',
            'secondary': '#764ba2',
            'success': '#4ecdc4',
            'warning': '#ff6b6b',
            'info': '#45b7d1',
            'light': '#f8f9fa',
            'dark': '#343a40'
        }
        
        self.chart_templates = {
            'plotly': 'plotly_white',
            'matplotlib': 'seaborn-v0_8'
        }
    
    def create_performance_chart(self, metrics: Dict, chart_type: str = "line") -> go.Figure:
        """Create performance monitoring chart."""
        
        if chart_type == "line":
            fig = go.Figure()
            
            # CPU Usage
            fig.add_trace(go.Scatter(
                x=metrics.get("timestamps", []),
                y=metrics.get("cpu_usage", []),
                mode='lines+markers',
                name='CPU Usage',
                line=dict(color=self.color_scheme['primary'], width=2),
                marker=dict(size=6)
            ))
            
            # Memory Usage
            fig.add_trace(go.Scatter(
                x=metrics.get("timestamps", []),
                y=metrics.get("memory_usage", []),
                mode='lines+markers',
                name='Memory Usage',
                line=dict(color=self.color_scheme['secondary'], width=2),
                marker=dict(size=6)
            ))
            
            # GPU Usage
            if "gpu_usage" in metrics:
                fig.add_trace(go.Scatter(
                    x=metrics.get("timestamps", []),
                    y=metrics.get("gpu_usage", []),
                    mode='lines+markers',
                    name='GPU Usage',
                    line=dict(color=self.color_scheme['success'], width=2),
                    marker=dict(size=6)
                ))
            
            fig.update_layout(
                title="System Performance Over Time",
                xaxis_title="Time",
                yaxis_title="Usage (%)",
                template=self.chart_templates['plotly'],
                height=400
            )
            
        elif chart_type == "gauge":
            fig = go.Figure()
            
            # CPU Gauge
            fig.add_trace(go.Indicator(
                mode="gauge+number+delta",
                value=metrics.get("cpu_usage", [0])[-1] if metrics.get("cpu_usage") else 0,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "CPU Usage"},
                delta={'reference': 50},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': self.color_scheme['primary']},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            
            fig.update_layout(height=300)
        
        return fig
    
    def create_training_visualization(self, training_data: Dict) -> Tuple[go.Figure, go.Figure, go.Figure]:
        """Create training progress visualizations."""
        
        epochs = training_data.get("epochs", [])
        train_loss = training_data.get("train_loss", [])
        val_loss = training_data.get("val_loss", [])
        accuracy = training_data.get("accuracy", [])
        learning_rates = training_data.get("learning_rates", [])
        
        # Loss curves
        loss_fig = go.Figure()
        loss_fig.add_trace(go.Scatter(
            x=epochs,
            y=train_loss,
            mode='lines+markers',
            name='Training Loss',
            line=dict(color=self.color_scheme['primary'], width=2)
        ))
        loss_fig.add_trace(go.Scatter(
            x=epochs,
            y=val_loss,
            mode='lines+markers',
            name='Validation Loss',
            line=dict(color=self.color_scheme['secondary'], width=2)
        ))
        loss_fig.update_layout(
            title="Training & Validation Loss",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            template=self.chart_templates['plotly'],
            height=400
        )
        
        # Accuracy curve
        acc_fig = go.Figure()
        acc_fig.add_trace(go.Scatter(
            x=epochs,
            y=accuracy,
            mode='lines+markers',
            name='Accuracy',
            line=dict(color=self.color_scheme['success'], width=2),
            fill='tonexty'
        ))
        acc_fig.update_layout(
            title="Training Accuracy",
            xaxis_title="Epoch",
            yaxis_title="Accuracy",
            template=self.chart_templates['plotly'],
            height=400
        )
        
        # Learning rate schedule
        lr_fig = go.Figure()
        lr_fig.add_trace(go.Scatter(
            x=epochs,
            y=learning_rates,
            mode='lines+markers',
            name='Learning Rate',
            line=dict(color=self.color_scheme['info'], width=2)
        ))
        lr_fig.update_layout(
            title="Learning Rate Schedule",
            xaxis_title="Epoch",
            yaxis_title="Learning Rate",
            template=self.chart_templates['plotly'],
            height=400
        )
        
        return loss_fig, acc_fig, lr_fig
    
    def create_viral_analysis_chart(self, analysis_data: Dict) -> Tuple[go.Figure, go.Figure]:
        """Create viral analysis visualizations."""
        
        viral_score = analysis_data.get("viral_score", 0)
        engagement = analysis_data.get("engagement_prediction", 0)
        platform_scores = analysis_data.get("platform_scores", {})
        
        # Radar chart for viral metrics
        categories = ['Viral Potential', 'Engagement', 'Shareability', 'Timing', 'Relevance']
        scores = [
            viral_score,
            engagement,
            viral_score * 0.9,
            viral_score * 0.8,
            viral_score * 0.95
        ]
        
        radar_fig = go.Figure()
        radar_fig.add_trace(go.Scatterpolar(
            r=scores,
            theta=categories,
            fill='toself',
            name='Content Score',
            line_color=self.color_scheme['primary']
        ))
        radar_fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=False,
            title="Viral Potential Radar Chart",
            height=400
        )
        
        # Platform comparison bar chart
        platforms = list(platform_scores.keys())
        scores_values = list(platform_scores.values())
        
        bar_fig = go.Figure(data=[
            go.Bar(
                x=platforms,
                y=scores_values,
                marker_color=[self.color_scheme['primary'], self.color_scheme['secondary'], 
                            self.color_scheme['success'], self.color_scheme['info']]
            )
        ])
        bar_fig.update_layout(
            title="Platform Performance Comparison",
            xaxis_title="Platform",
            yaxis_title="Score",
            template=self.chart_templates['plotly'],
            height=400
        )
        
        return radar_fig, bar_fig
    
    def create_generation_metrics_chart(self, generation_data: Dict) -> go.Figure:
        """Create video generation metrics visualization."""
        
        parameters = generation_data.get("parameters", {})
        performance = generation_data.get("performance", {})
        
        # Create parameter comparison
        param_names = list(parameters.keys())
        param_values = list(parameters.values())
        
        fig = go.Figure(data=[
            go.Bar(
                x=param_names,
                y=param_values,
                marker_color=self.color_scheme['primary'],
                text=param_values,
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title="Generation Parameters",
            xaxis_title="Parameter",
            yaxis_title="Value",
            template=self.chart_templates['plotly'],
            height=400
        )
        
        return fig
    
    def create_system_dashboard(self, system_metrics: Dict) -> go.Figure:
        """Create comprehensive system dashboard."""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('CPU Usage', 'Memory Usage', 'GPU Usage', 'Network Activity'),
            specs=[[{"type": "indicator"}, {"type": "indicator"}],
                   [{"type": "indicator"}, {"type": "indicator"}]]
        )
        
        # CPU Gauge
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=system_metrics.get("cpu_usage", 0),
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "CPU"},
            gauge={'axis': {'range': [None, 100]},
                   'bar': {'color': self.color_scheme['primary']},
                   'steps': [
                       {'range': [0, 50], 'color': "lightgray"},
                       {'range': [50, 80], 'color': "yellow"},
                       {'range': [80, 100], 'color': "red"}
                   ]}
        ), row=1, col=1)
        
        # Memory Gauge
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=system_metrics.get("memory_usage", 0),
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Memory"},
            gauge={'axis': {'range': [None, 100]},
                   'bar': {'color': self.color_scheme['secondary']},
                   'steps': [
                       {'range': [0, 50], 'color': "lightgray"},
                       {'range': [50, 80], 'color': "yellow"},
                       {'range': [80, 100], 'color': "red"}
                   ]}
        ), row=1, col=2)
        
        # GPU Gauge
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=system_metrics.get("gpu_usage", 0),
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "GPU"},
            gauge={'axis': {'range': [None, 100]},
                   'bar': {'color': self.color_scheme['success']},
                   'steps': [
                       {'range': [0, 50], 'color': "lightgray"},
                       {'range': [50, 80], 'color': "yellow"},
                       {'range': [80, 100], 'color': "red"}
                   ]}
        ), row=2, col=1)
        
        # Network Gauge
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=system_metrics.get("network_usage", 0),
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Network"},
            gauge={'axis': {'range': [None, 100]},
                   'bar': {'color': self.color_scheme['info']},
                   'steps': [
                       {'range': [0, 50], 'color': "lightgray"},
                       {'range': [50, 80], 'color': "yellow"},
                       {'range': [80, 100], 'color': "red"}
                   ]}
        ), row=2, col=2)
        
        fig.update_layout(
            title="System Performance Dashboard",
            height=600,
            template=self.chart_templates['plotly']
        )
        
        return fig
    
    def create_realtime_chart(self, data_stream: List[Dict]) -> go.Figure:
        """Create real-time updating chart."""
        
        timestamps = [d.get("timestamp", i) for i, d in enumerate(data_stream)]
        values = [d.get("value", 0) for d in data_stream]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=values,
            mode='lines+markers',
            name='Real-time Data',
            line=dict(color=self.color_scheme['primary'], width=2),
            marker=dict(size=6)
        ))
        
        fig.update_layout(
            title="Real-time Data Stream",
            xaxis_title="Time",
            yaxis_title="Value",
            template=self.chart_templates['plotly'],
            height=400,
            uirevision=True  # Preserve zoom on updates
        )
        
        return fig
    
    def create_comparison_chart(self, data_sets: Dict[str, List]) -> go.Figure:
        """Create comparison chart for multiple datasets."""
        
        fig = go.Figure()
        
        colors = [self.color_scheme['primary'], self.color_scheme['secondary'], 
                 self.color_scheme['success'], self.color_scheme['info']]
        
        for i, (name, data) in enumerate(data_sets.items()):
            fig.add_trace(go.Scatter(
                x=list(range(len(data))),
                y=data,
                mode='lines+markers',
                name=name,
                line=dict(color=colors[i % len(colors)], width=2),
                marker=dict(size=6)
            ))
        
        fig.update_layout(
            title="Dataset Comparison",
            xaxis_title="Index",
            yaxis_title="Value",
            template=self.chart_templates['plotly'],
            height=400
        )
        
        return fig
    
    def create_heatmap(self, data: np.ndarray, labels: List[str] = None) -> go.Figure:
        """Create heatmap visualization."""
        
        if labels is None:
            labels = [f"Dim_{i}" for i in range(data.shape[1])]
        
        fig = go.Figure(data=go.Heatmap(
            z=data,
            x=labels,
            y=labels,
            colorscale='Viridis'
        ))
        
        fig.update_layout(
            title="Correlation Heatmap",
            template=self.chart_templates['plotly'],
            height=500
        )
        
        return fig
    
    def create_3d_scatter(self, x: List, y: List, z: List, labels: List = None) -> go.Figure:
        """Create 3D scatter plot."""
        
        fig = go.Figure(data=[go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='markers',
            marker=dict(
                size=8,
                color=z,
                colorscale='Viridis',
                opacity=0.8
            ),
            text=labels
        )])
        
        fig.update_layout(
            title="3D Scatter Plot",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z"
            ),
            template=self.chart_templates['plotly'],
            height=600
        )
        
        return fig

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_matplotlib_chart(data: Dict, chart_type: str = "line") -> plt.Figure:
    """Create matplotlib chart for Gradio compatibility."""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if chart_type == "line":
        x = data.get("x", [])
        y = data.get("y", [])
        ax.plot(x, y, linewidth=2, marker='o')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("Line Chart")
        
    elif chart_type == "bar":
        x = data.get("x", [])
        y = data.get("y", [])
        ax.bar(x, y, color='skyblue', alpha=0.7)
        ax.set_xlabel("Categories")
        ax.set_ylabel("Values")
        ax.set_title("Bar Chart")
        
    elif chart_type == "scatter":
        x = data.get("x", [])
        y = data.get("y", [])
        ax.scatter(x, y, alpha=0.6, s=50)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("Scatter Plot")
    
    plt.tight_layout()
    return fig

def create_plotly_chart(data: Dict, chart_type: str = "line") -> go.Figure:
    """Create plotly chart for Gradio compatibility."""
    
    visualizer = DemoVisualizer()
    
    if chart_type == "line":
        return visualizer.create_performance_chart(data, "line")
    elif chart_type == "bar":
        return visualizer.create_generation_metrics_chart(data)
    elif chart_type == "radar":
        return visualizer.create_viral_analysis_chart(data)[0]
    else:
        return go.Figure()

def generate_sample_data(data_type: str = "performance") -> Dict:
    """Generate sample data for demos."""
    
    if data_type == "performance":
        timestamps = [time.time() - i * 60 for i in range(10, 0, -1)]
        return {
            "timestamps": timestamps,
            "cpu_usage": [np.random.uniform(20, 80) for _ in range(10)],
            "memory_usage": [np.random.uniform(30, 90) for _ in range(10)],
            "gpu_usage": [np.random.uniform(10, 70) for _ in range(10)]
        }
    
    elif data_type == "training":
        epochs = list(range(1, 21))
        return {
            "epochs": epochs,
            "train_loss": [2.0 * np.exp(-e/10) + 0.1 * np.random.random() for e in epochs],
            "val_loss": [2.0 * np.exp(-e/10) + 0.2 * np.random.random() for e in epochs],
            "accuracy": [0.3 + 0.6 * (1 - np.exp(-e/8)) + 0.05 * np.random.random() for e in epochs],
            "learning_rates": [1e-4 * (0.9 ** (e // 5)) for e in epochs]
        }
    
    elif data_type == "viral":
        return {
            "viral_score": 0.75,
            "engagement_prediction": 0.68,
            "platform_scores": {
                "TikTok": 0.85,
                "YouTube": 0.72,
                "Instagram": 0.78,
                "Twitter": 0.65
            }
        }
    
    else:
        return {}

# =============================================================================
# EXPORT FUNCTIONS
# =============================================================================

def get_visualizer() -> DemoVisualizer:
    """Get the main visualizer instance."""
    return DemoVisualizer()

def create_demo_chart(data: Dict, chart_type: str = "line") -> Any:
    """Create chart for demo use."""
    if chart_type in ["line", "bar", "scatter"]:
        return create_matplotlib_chart(data, chart_type)
    else:
        return create_plotly_chart(data, chart_type) 