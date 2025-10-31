#!/usr/bin/env python3
"""
Advanced Visualization and Analysis Tools for Frontier Model Training
Provides comprehensive data visualization, model analysis, and interactive dashboards.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from plotly.utils import PlotlyJSONEncoder
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from pathlib import Path
import time
from datetime import datetime, timedelta
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import streamlit as st
import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
from sklearn.manifold import TSNE, UMAP
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, classification_report
import networkx as nx
from wordcloud import WordCloud
import cv2
from PIL import Image
import librosa
import librosa.display
from transformers import AutoTokenizer
import gradio as gr

console = Console()

class VisualizationType(Enum):
    """Types of visualizations."""
    LINE = "line"
    BAR = "bar"
    SCATTER = "scatter"
    HEATMAP = "heatmap"
    HISTOGRAM = "histogram"
    BOX = "box"
    VIOLIN = "violin"
    PIE = "pie"
    DONUT = "donut"
    SANKEY = "sankey"
    TREEMAP = "treemap"
    SUNBURST = "sunburst"
    PARALLEL = "parallel"
    RADAR = "radar"
    WATERFALL = "waterfall"
    FUNNEL = "funnel"
    GAUGE = "gauge"
    INDICATOR = "indicator"

class AnalysisType(Enum):
    """Types of analysis."""
    STATISTICAL = "statistical"
    CORRELATION = "correlation"
    CLUSTERING = "clustering"
    DIMENSIONALITY = "dimensionality"
    TIME_SERIES = "time_series"
    TEXT_ANALYSIS = "text_analysis"
    MODEL_ANALYSIS = "model_analysis"
    PERFORMANCE = "performance"

@dataclass
class VisualizationConfig:
    """Configuration for visualizations."""
    title: str
    x_label: Optional[str] = None
    y_label: Optional[str] = None
    width: int = 800
    height: int = 600
    theme: str = "plotly_white"
    colors: List[str] = None
    show_legend: bool = True
    show_grid: bool = True
    interactive: bool = True
    save_format: str = "html"
    save_path: Optional[str] = None

@dataclass
class AnalysisResult:
    """Result of data analysis."""
    analysis_type: AnalysisType
    data: Dict[str, Any]
    insights: List[str]
    recommendations: List[str]
    visualizations: List[str] = None
    metrics: Dict[str, float] = None

class DataVisualizer:
    """Advanced data visualization with multiple chart types."""
    
    def __init__(self, config: VisualizationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def create_line_chart(self, data: Union[pd.DataFrame, Dict[str, List]], 
                         x_col: str, y_cols: List[str]) -> go.Figure:
        """Create interactive line chart."""
        fig = go.Figure()
        
        if isinstance(data, pd.DataFrame):
            for y_col in y_cols:
                fig.add_trace(go.Scatter(
                    x=data[x_col],
                    y=data[y_col],
                    mode='lines+markers',
                    name=y_col,
                    line=dict(width=2)
                ))
        else:
            for y_col in y_cols:
                fig.add_trace(go.Scatter(
                    x=data[x_col],
                    y=data[y_col],
                    mode='lines+markers',
                    name=y_col,
                    line=dict(width=2)
                ))
        
        fig.update_layout(
            title=self.config.title,
            xaxis_title=self.config.x_label or x_col,
            yaxis_title=self.config.y_label or y_cols[0],
            width=self.config.width,
            height=self.config.height,
            template=self.config.theme,
            showlegend=self.config.show_legend
        )
        
        return fig
    
    def create_bar_chart(self, data: Union[pd.DataFrame, Dict[str, List]], 
                        x_col: str, y_col: str) -> go.Figure:
        """Create interactive bar chart."""
        fig = go.Figure()
        
        if isinstance(data, pd.DataFrame):
            fig.add_trace(go.Bar(
                x=data[x_col],
                y=data[y_col],
                name=y_col
            ))
        else:
            fig.add_trace(go.Bar(
                x=data[x_col],
                y=data[y_col],
                name=y_col
            ))
        
        fig.update_layout(
            title=self.config.title,
            xaxis_title=self.config.x_label or x_col,
            yaxis_title=self.config.y_label or y_col,
            width=self.config.width,
            height=self.config.height,
            template=self.config.theme,
            showlegend=self.config.show_legend
        )
        
        return fig
    
    def create_scatter_plot(self, data: Union[pd.DataFrame, Dict[str, List]], 
                           x_col: str, y_col: str, color_col: Optional[str] = None) -> go.Figure:
        """Create interactive scatter plot."""
        fig = go.Figure()
        
        if isinstance(data, pd.DataFrame):
            if color_col:
                fig.add_trace(go.Scatter(
                    x=data[x_col],
                    y=data[y_col],
                    mode='markers',
                    marker=dict(
                        color=data[color_col],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title=color_col)
                    ),
                    name=y_col
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=data[x_col],
                    y=data[y_col],
                    mode='markers',
                    name=y_col
                ))
        else:
            fig.add_trace(go.Scatter(
                x=data[x_col],
                y=data[y_col],
                mode='markers',
                name=y_col
            ))
        
        fig.update_layout(
            title=self.config.title,
            xaxis_title=self.config.x_label or x_col,
            yaxis_title=self.config.y_label or y_col,
            width=self.config.width,
            height=self.config.height,
            template=self.config.theme,
            showlegend=self.config.show_legend
        )
        
        return fig
    
    def create_heatmap(self, data: Union[pd.DataFrame, np.ndarray], 
                      title: Optional[str] = None) -> go.Figure:
        """Create interactive heatmap."""
        if isinstance(data, pd.DataFrame):
            values = data.values
            labels = data.columns.tolist()
        else:
            values = data
            labels = None
        
        fig = go.Figure(data=go.Heatmap(
            z=values,
            colorscale='Viridis',
            showscale=True
        ))
        
        fig.update_layout(
            title=title or self.config.title,
            width=self.config.width,
            height=self.config.height,
            template=self.config.theme
        )
        
        return fig
    
    def create_histogram(self, data: Union[pd.DataFrame, List], 
                        column: Optional[str] = None) -> go.Figure:
        """Create interactive histogram."""
        if isinstance(data, pd.DataFrame):
            values = data[column] if column else data.iloc[:, 0]
        else:
            values = data
        
        fig = go.Figure(data=go.Histogram(
            x=values,
            nbinsx=30
        ))
        
        fig.update_layout(
            title=self.config.title,
            xaxis_title=self.config.x_label or column,
            yaxis_title=self.config.y_label or "Count",
            width=self.config.width,
            height=self.config.height,
            template=self.config.theme
        )
        
        return fig
    
    def create_box_plot(self, data: Union[pd.DataFrame, Dict[str, List]], 
                       columns: List[str]) -> go.Figure:
        """Create interactive box plot."""
        fig = go.Figure()
        
        if isinstance(data, pd.DataFrame):
            for col in columns:
                fig.add_trace(go.Box(
                    y=data[col],
                    name=col
                ))
        else:
            for col in columns:
                fig.add_trace(go.Box(
                    y=data[col],
                    name=col
                ))
        
        fig.update_layout(
            title=self.config.title,
            yaxis_title=self.config.y_label or "Value",
            width=self.config.width,
            height=self.config.height,
            template=self.config.theme,
            showlegend=self.config.show_legend
        )
        
        return fig
    
    def create_pie_chart(self, data: Union[pd.DataFrame, Dict[str, float]], 
                        labels_col: str, values_col: str) -> go.Figure:
        """Create interactive pie chart."""
        if isinstance(data, pd.DataFrame):
            labels = data[labels_col].tolist()
            values = data[values_col].tolist()
        else:
            labels = list(data.keys())
            values = list(data.values())
        
        fig = go.Figure(data=go.Pie(
            labels=labels,
            values=values,
            hole=0.3 if self.config.title.lower().find('donut') != -1 else 0
        ))
        
        fig.update_layout(
            title=self.config.title,
            width=self.config.width,
            height=self.config.height,
            template=self.config.theme
        )
        
        return fig
    
    def create_sankey_diagram(self, source: List[str], target: List[str], 
                             value: List[float]) -> go.Figure:
        """Create Sankey diagram."""
        fig = go.Figure(data=go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=list(set(source + target)),
                color="blue"
            ),
            link=dict(
                source=[list(set(source + target)).index(s) for s in source],
                target=[list(set(source + target)).index(t) for t in target],
                value=value
            )
        ))
        
        fig.update_layout(
            title=self.config.title,
            width=self.config.width,
            height=self.config.height,
            template=self.config.theme
        )
        
        return fig
    
    def create_treemap(self, data: Dict[str, float]) -> go.Figure:
        """Create treemap visualization."""
        labels = list(data.keys())
        values = list(data.values())
        parents = [""] * len(labels)
        
        fig = go.Figure(go.Treemap(
            labels=labels,
            values=values,
            parents=parents
        ))
        
        fig.update_layout(
            title=self.config.title,
            width=self.config.width,
            height=self.config.height,
            template=self.config.theme
        )
        
        return fig
    
    def create_radar_chart(self, data: Dict[str, List[float]], 
                          categories: List[str]) -> go.Figure:
        """Create radar chart."""
        fig = go.Figure()
        
        for name, values in data.items():
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=name
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(max(values) for values in data.values())]
                )
            ),
            title=self.config.title,
            width=self.config.width,
            height=self.config.height,
            template=self.config.theme
        )
        
        return fig
    
    def create_gauge_chart(self, value: float, max_value: float = 100, 
                          title: Optional[str] = None) -> go.Figure:
        """Create gauge chart."""
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=value,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': title or self.config.title},
            delta={'reference': max_value * 0.8},
            gauge={
                'axis': {'range': [None, max_value]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, max_value * 0.5], 'color': "lightgray"},
                    {'range': [max_value * 0.5, max_value * 0.8], 'color': "gray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': max_value * 0.9
                }
            }
        ))
        
        fig.update_layout(
            width=self.config.width,
            height=self.config.height,
            template=self.config.theme
        )
        
        return fig
    
    def save_visualization(self, fig: go.Figure, filename: Optional[str] = None):
        """Save visualization to file."""
        if filename is None:
            filename = f"{self.config.title.replace(' ', '_')}.{self.config.save_format}"
        
        if self.config.save_path:
            filepath = Path(self.config.save_path) / filename
        else:
            filepath = Path(filename)
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if self.config.save_format == "html":
            fig.write_html(str(filepath))
        elif self.config.save_format == "png":
            fig.write_image(str(filepath))
        elif self.config.save_format == "pdf":
            fig.write_image(str(filepath))
        elif self.config.save_format == "json":
            with open(filepath, 'w') as f:
                json.dump(fig, f, cls=PlotlyJSONEncoder)
        
        console.print(f"[green]Visualization saved to: {filepath}[/green]")

class ModelAnalyzer:
    """Advanced model analysis and visualization."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def analyze_model_architecture(self, model: nn.Module) -> Dict[str, Any]:
        """Analyze model architecture."""
        analysis = {
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "layers": [],
            "layer_types": {},
            "parameter_distribution": {}
        }
        
        # Analyze layers
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                layer_info = {
                    "name": name,
                    "type": type(module).__name__,
                    "parameters": sum(p.numel() for p in module.parameters()),
                    "shape": None
                }
                
                # Get layer shape if possible
                if hasattr(module, 'weight') and module.weight is not None:
                    layer_info["shape"] = list(module.weight.shape)
                
                analysis["layers"].append(layer_info)
                
                # Count layer types
                layer_type = type(module).__name__
                analysis["layer_types"][layer_type] = analysis["layer_types"].get(layer_type, 0) + 1
        
        return analysis
    
    def visualize_model_architecture(self, model: nn.Module, 
                                   save_path: Optional[str] = None) -> go.Figure:
        """Visualize model architecture."""
        analysis = self.analyze_model_architecture(model)
        
        # Create layer type distribution pie chart
        fig = go.Figure(data=go.Pie(
            labels=list(analysis["layer_types"].keys()),
            values=list(analysis["layer_types"].values()),
            title="Layer Type Distribution"
        ))
        
        fig.update_layout(
            title="Model Architecture Analysis",
            width=800,
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def analyze_gradient_flow(self, model: nn.Module, 
                            loss: torch.Tensor) -> Dict[str, Any]:
        """Analyze gradient flow through the model."""
        # Backward pass
        loss.backward()
        
        gradient_analysis = {
            "gradient_norms": {},
            "gradient_stats": {},
            "vanishing_gradients": [],
            "exploding_gradients": []
        }
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                gradient_analysis["gradient_norms"][name] = grad_norm
                
                # Check for vanishing/exploding gradients
                if grad_norm < 1e-6:
                    gradient_analysis["vanishing_gradients"].append(name)
                elif grad_norm > 10.0:
                    gradient_analysis["exploding_gradients"].append(name)
        
        # Calculate statistics
        grad_norms = list(gradient_analysis["gradient_norms"].values())
        gradient_analysis["gradient_stats"] = {
            "mean": np.mean(grad_norms),
            "std": np.std(grad_norms),
            "min": np.min(grad_norms),
            "max": np.max(grad_norms),
            "median": np.median(grad_norms)
        }
        
        return gradient_analysis
    
    def visualize_gradient_flow(self, gradient_analysis: Dict[str, Any]) -> go.Figure:
        """Visualize gradient flow analysis."""
        grad_norms = list(gradient_analysis["gradient_norms"].values())
        layer_names = list(gradient_analysis["gradient_norms"].keys())
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=layer_names,
            y=grad_norms,
            name="Gradient Norms"
        ))
        
        # Add threshold lines
        fig.add_hline(y=1e-6, line_dash="dash", line_color="red", 
                     annotation_text="Vanishing Gradient Threshold")
        fig.add_hline(y=10.0, line_dash="dash", line_color="red", 
                     annotation_text="Exploding Gradient Threshold")
        
        fig.update_layout(
            title="Gradient Flow Analysis",
            xaxis_title="Layer Name",
            yaxis_title="Gradient Norm",
            yaxis_type="log",
            width=1200,
            height=600
        )
        
        return fig
    
    def analyze_attention_patterns(self, attention_weights: torch.Tensor, 
                                 tokens: List[str]) -> go.Figure:
        """Analyze and visualize attention patterns."""
        # Convert to numpy
        attention = attention_weights.detach().cpu().numpy()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=attention,
            x=tokens,
            y=tokens,
            colorscale='Blues'
        ))
        
        fig.update_layout(
            title="Attention Pattern Visualization",
            xaxis_title="Key Tokens",
            yaxis_title="Query Tokens",
            width=800,
            height=800
        )
        
        return fig
    
    def analyze_embedding_space(self, embeddings: torch.Tensor, 
                              labels: Optional[List[str]] = None) -> go.Figure:
        """Analyze embedding space using dimensionality reduction."""
        # Convert to numpy
        emb_np = embeddings.detach().cpu().numpy()
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        emb_2d = tsne.fit_transform(emb_np)
        
        fig = go.Figure()
        
        if labels:
            unique_labels = list(set(labels))
            colors = px.colors.qualitative.Set1
            
            for i, label in enumerate(unique_labels):
                mask = [l == label for l in labels]
                fig.add_trace(go.Scatter(
                    x=emb_2d[mask, 0],
                    y=emb_2d[mask, 1],
                    mode='markers',
                    name=label,
                    marker=dict(color=colors[i % len(colors)])
                ))
        else:
            fig.add_trace(go.Scatter(
                x=emb_2d[:, 0],
                y=emb_2d[:, 1],
                mode='markers',
                name='Embeddings'
            ))
        
        fig.update_layout(
            title="Embedding Space Visualization (t-SNE)",
            xaxis_title="t-SNE 1",
            yaxis_title="t-SNE 2",
            width=800,
            height=600
        )
        
        return fig

class TextAnalyzer:
    """Advanced text analysis and visualization."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def create_word_cloud(self, texts: List[str], 
                         max_words: int = 100) -> go.Figure:
        """Create word cloud visualization."""
        # Combine all texts
        combined_text = ' '.join(texts)
        
        # Generate word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            max_words=max_words,
            background_color='white'
        ).generate(combined_text)
        
        # Convert to image
        img_array = np.array(wordcloud)
        
        fig = go.Figure()
        fig.add_trace(go.Image(z=img_array))
        
        fig.update_layout(
            title="Word Cloud",
            width=800,
            height=400,
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False, showticklabels=False)
        )
        
        return fig
    
    def analyze_text_sentiment(self, texts: List[str]) -> go.Figure:
        """Analyze text sentiment distribution."""
        # Simplified sentiment analysis (in practice, use a proper sentiment analyzer)
        sentiments = []
        for text in texts:
            # Simple heuristic-based sentiment
            positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful']
            negative_words = ['bad', 'terrible', 'awful', 'horrible', 'disappointing']
            
            text_lower = text.lower()
            pos_count = sum(1 for word in positive_words if word in text_lower)
            neg_count = sum(1 for word in negative_words if word in text_lower)
            
            if pos_count > neg_count:
                sentiments.append('Positive')
            elif neg_count > pos_count:
                sentiments.append('Negative')
            else:
                sentiments.append('Neutral')
        
        # Create pie chart
        sentiment_counts = pd.Series(sentiments).value_counts()
        
        fig = go.Figure(data=go.Pie(
            labels=sentiment_counts.index,
            values=sentiment_counts.values,
            title="Sentiment Distribution"
        ))
        
        fig.update_layout(
            title="Text Sentiment Analysis",
            width=600,
            height=400
        )
        
        return fig
    
    def analyze_text_length_distribution(self, texts: List[str]) -> go.Figure:
        """Analyze text length distribution."""
        lengths = [len(text.split()) for text in texts]
        
        fig = go.Figure(data=go.Histogram(
            x=lengths,
            nbinsx=30,
            name="Text Length Distribution"
        ))
        
        fig.update_layout(
            title="Text Length Distribution",
            xaxis_title="Number of Words",
            yaxis_title="Frequency",
            width=800,
            height=400
        )
        
        return fig

class PerformanceAnalyzer:
    """Performance analysis and visualization."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def analyze_training_metrics(self, metrics: Dict[str, List[float]]) -> go.Figure:
        """Analyze training metrics over time."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Training Loss", "Validation Loss", 
                          "Learning Rate", "Accuracy"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Training loss
        if "train_loss" in metrics:
            fig.add_trace(
                go.Scatter(y=metrics["train_loss"], name="Training Loss"),
                row=1, col=1
            )
        
        # Validation loss
        if "val_loss" in metrics:
            fig.add_trace(
                go.Scatter(y=metrics["val_loss"], name="Validation Loss"),
                row=1, col=2
            )
        
        # Learning rate
        if "learning_rate" in metrics:
            fig.add_trace(
                go.Scatter(y=metrics["learning_rate"], name="Learning Rate"),
                row=2, col=1
            )
        
        # Accuracy
        if "accuracy" in metrics:
            fig.add_trace(
                go.Scatter(y=metrics["accuracy"], name="Accuracy"),
                row=2, col=2
            )
        
        fig.update_layout(
            title="Training Metrics Analysis",
            width=1200,
            height=800
        )
        
        return fig
    
    def analyze_system_performance(self, system_metrics: Dict[str, List[float]]) -> go.Figure:
        """Analyze system performance metrics."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("CPU Usage", "Memory Usage", 
                          "GPU Memory", "GPU Utilization"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # CPU usage
        if "cpu_percent" in system_metrics:
            fig.add_trace(
                go.Scatter(y=system_metrics["cpu_percent"], name="CPU %"),
                row=1, col=1
            )
        
        # Memory usage
        if "memory_percent" in system_metrics:
            fig.add_trace(
                go.Scatter(y=system_metrics["memory_percent"], name="Memory %"),
                row=1, col=2
            )
        
        # GPU memory
        if "gpu_memory_used" in system_metrics:
            fig.add_trace(
                go.Scatter(y=system_metrics["gpu_memory_used"], name="GPU Memory (MB)"),
                row=2, col=1
            )
        
        # GPU utilization
        if "gpu_utilization" in system_metrics:
            fig.add_trace(
                go.Scatter(y=system_metrics["gpu_utilization"], name="GPU Utilization %"),
                row=2, col=2
            )
        
        fig.update_layout(
            title="System Performance Analysis",
            width=1200,
            height=800
        )
        
        return fig

class DashboardCreator:
    """Create interactive dashboards."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def create_streamlit_dashboard(self, data: Dict[str, Any]):
        """Create Streamlit dashboard."""
        st.set_page_config(page_title="Frontier Model Dashboard", layout="wide")
        
        st.title("Frontier Model Training Dashboard")
        
        # Sidebar
        st.sidebar.title("Navigation")
        page = st.sidebar.selectbox("Select Page", 
                                   ["Overview", "Training Metrics", "Model Analysis", "Data Analysis"])
        
        if page == "Overview":
            self._create_overview_page(data)
        elif page == "Training Metrics":
            self._create_training_metrics_page(data)
        elif page == "Model Analysis":
            self._create_model_analysis_page(data)
        elif page == "Data Analysis":
            self._create_data_analysis_page(data)
    
    def _create_overview_page(self, data: Dict[str, Any]):
        """Create overview page."""
        st.header("Training Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Steps", data.get("total_steps", 0))
        
        with col2:
            st.metric("Current Loss", f"{data.get('current_loss', 0):.4f}")
        
        with col3:
            st.metric("Accuracy", f"{data.get('accuracy', 0):.2%}")
        
        with col4:
            st.metric("Learning Rate", f"{data.get('learning_rate', 0):.6f}")
        
        # Training progress
        if "training_loss" in data:
            st.subheader("Training Progress")
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=data["training_loss"], name="Training Loss"))
            if "validation_loss" in data:
                fig.add_trace(go.Scatter(y=data["validation_loss"], name="Validation Loss"))
            st.plotly_chart(fig, use_container_width=True)
    
    def _create_training_metrics_page(self, data: Dict[str, Any]):
        """Create training metrics page."""
        st.header("Training Metrics")
        
        # Loss curves
        st.subheader("Loss Curves")
        col1, col2 = st.columns(2)
        
        with col1:
            if "training_loss" in data:
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=data["training_loss"], name="Training Loss"))
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if "validation_loss" in data:
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=data["validation_loss"], name="Validation Loss"))
                st.plotly_chart(fig, use_container_width=True)
        
        # Learning rate schedule
        if "learning_rate" in data:
            st.subheader("Learning Rate Schedule")
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=data["learning_rate"], name="Learning Rate"))
            st.plotly_chart(fig, use_container_width=True)
    
    def _create_model_analysis_page(self, data: Dict[str, Any]):
        """Create model analysis page."""
        st.header("Model Analysis")
        
        # Model architecture
        st.subheader("Model Architecture")
        if "model_info" in data:
            st.json(data["model_info"])
        
        # Gradient analysis
        if "gradient_analysis" in data:
            st.subheader("Gradient Analysis")
            analyzer = ModelAnalyzer()
            fig = analyzer.visualize_gradient_flow(data["gradient_analysis"])
            st.plotly_chart(fig, use_container_width=True)
    
    def _create_data_analysis_page(self, data: Dict[str, Any]):
        """Create data analysis page."""
        st.header("Data Analysis")
        
        # Data distribution
        if "data_distribution" in data:
            st.subheader("Data Distribution")
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=data["data_distribution"]))
            st.plotly_chart(fig, use_container_width=True)
        
        # Text analysis
        if "texts" in data:
            st.subheader("Text Analysis")
            analyzer = TextAnalyzer()
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = analyzer.create_word_cloud(data["texts"])
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = analyzer.analyze_text_sentiment(data["texts"])
                st.plotly_chart(fig, use_container_width=True)

def main():
    """Main function for visualization CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualization and Analysis Tools")
    parser.add_argument("--data-file", type=str, help="Path to data file")
    parser.add_argument("--visualization-type", type=str,
                       choices=["line", "bar", "scatter", "heatmap", "histogram", "box"],
                       default="line", help="Type of visualization")
    parser.add_argument("--output-file", type=str, help="Output file path")
    parser.add_argument("--title", type=str, default="Visualization", help="Chart title")
    parser.add_argument("--width", type=int, default=800, help="Chart width")
    parser.add_argument("--height", type=int, default=600, help="Chart height")
    parser.add_argument("--format", type=str, choices=["html", "png", "pdf"], 
                       default="html", help="Output format")
    
    args = parser.parse_args()
    
    # Create visualization config
    config = VisualizationConfig(
        title=args.title,
        width=args.width,
        height=args.height,
        save_format=args.format,
        save_path=args.output_file
    )
    
    # Create visualizer
    visualizer = DataVisualizer(config)
    
    # Load sample data
    if args.data_file and Path(args.data_file).exists():
        # Load data from file
        if args.data_file.endswith('.csv'):
            data = pd.read_csv(args.data_file)
        elif args.data_file.endswith('.json'):
            with open(args.data_file, 'r') as f:
                data = json.load(f)
        else:
            data = None
    else:
        # Create sample data
        data = pd.DataFrame({
            'x': range(100),
            'y': np.random.randn(100).cumsum(),
            'z': np.random.randn(100)
        })
    
    # Create visualization based on type
    if args.visualization_type == "line":
        fig = visualizer.create_line_chart(data, 'x', ['y'])
    elif args.visualization_type == "bar":
        fig = visualizer.create_bar_chart(data, 'x', 'y')
    elif args.visualization_type == "scatter":
        fig = visualizer.create_scatter_plot(data, 'x', 'y', 'z')
    elif args.visualization_type == "heatmap":
        fig = visualizer.create_heatmap(data.corr())
    elif args.visualization_type == "histogram":
        fig = visualizer.create_histogram(data, 'y')
    elif args.visualization_type == "box":
        fig = visualizer.create_box_plot(data, ['y', 'z'])
    
    # Save visualization
    visualizer.save_visualization(fig)
    
    console.print(f"[green]Visualization created successfully[/green]")

if __name__ == "__main__":
    main()
