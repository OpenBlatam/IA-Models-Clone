from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import gradio as gr
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import warnings
            from sklearn.metrics import confusion_matrix
            from sklearn.metrics import roc_curve, auc
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Advanced Gradio Interfaces for Cybersecurity Model Showcase

This module provides comprehensive, user-friendly interfaces that showcase:
- Real-time model inference with confidence scores
- Interactive data visualization and analysis
- Model performance metrics and comparisons
- Batch processing capabilities
- Anomaly detection visualization
- Threat classification with explanations
- Model interpretability features
- Real-time monitoring dashboards
"""

warnings.filterwarnings('ignore')

# Configure matplotlib for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class CybersecurityModelInterface:
    """Advanced interface for cybersecurity model showcase."""
    
    def __init__(self) -> Any:
        self.model = self._create_demo_model()
        self.model.eval()
        self.prediction_history = []
        self.performance_metrics = {}
        
    def _create_demo_model(self) -> nn.Module:
        """Create a demo model for showcase."""
        class DemoCybersecurityModel(nn.Module):
            def __init__(self, input_dim=20, num_classes=4) -> Any:
                super().__init__()
                self.features = nn.Sequential(
                    nn.Linear(input_dim, 64),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(32, 16),
                    nn.ReLU()
                )
                self.classifier = nn.Linear(16, num_classes)
                
            def forward(self, x) -> Any:
                features = self.features(x)
                return self.classifier(features)
        
        return DemoCybersecurityModel(input_dim=20, num_classes=4)
    
    def _sanitize_output(self, value: Any) -> Any:
        """Sanitize outputs to handle NaN/Inf values."""
        if isinstance(value, (np.ndarray, torch.Tensor)):
            if isinstance(value, torch.Tensor):
                value = value.cpu().numpy()
            value = np.nan_to_num(value, nan=0.0, posinf=1.0, neginf=0.0)
            return value
        elif isinstance(value, (int, float)):
            if not np.isfinite(value):
                return 0.0
        return value
    
    def _generate_demo_data(self, num_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Generate demo data for visualization."""
        np.random.seed(42)
        
        # Generate features (simulating network traffic, system calls, etc.)
        features = np.random.randn(num_samples, 20)
        
        # Add some patterns for different threat types
        # Normal traffic
        normal_mask = np.random.choice([True, False], num_samples, p=[0.7, 0.3])
        features[normal_mask] += np.random.normal(0, 0.1, (normal_mask.sum(), 20))
        
        # Malware patterns
        malware_mask = ~normal_mask
        features[malware_mask, :5] += np.random.normal(2, 0.5, (malware_mask.sum(), 5))
        
        # Generate labels
        labels = np.zeros(num_samples, dtype=int)
        labels[malware_mask] = np.random.choice([1, 2, 3], malware_mask.sum(), p=[0.4, 0.3, 0.3])
        
        return features, labels
    
    def real_time_inference(self, *feature_values) -> Dict[str, Any]:
        """Real-time inference with detailed analysis."""
        try:
            # Convert inputs to tensor
            features = torch.tensor(feature_values, dtype=torch.float32).unsqueeze(0)
            
            # Get model prediction
            with torch.no_grad():
                logits = self.model(features)
                probabilities = torch.softmax(logits, dim=1)
                prediction = torch.argmax(probabilities, dim=1)
                confidence = torch.max(probabilities, dim=1)[0]
            
            # Sanitize outputs
            probs = self._sanitize_output(probabilities.cpu().numpy().flatten())
            pred_class = int(self._sanitize_output(prediction.cpu().numpy()[0]))
            conf_score = float(self._sanitize_output(confidence.cpu().numpy()[0]))
            
            # Threat classification mapping
            threat_types = {
                0: "Normal Traffic",
                1: "Malware Detection",
                2: "Network Intrusion", 
                3: "Data Exfiltration"
            }
            
            # Risk assessment
            if conf_score > 0.8:
                risk_level = "High"
                risk_color = "red"
            elif conf_score > 0.6:
                risk_level = "Medium"
                risk_color = "orange"
            else:
                risk_level = "Low"
                risk_color = "green"
            
            # Store prediction history
            self.prediction_history.append({
                'timestamp': datetime.now().isoformat(),
                'prediction': pred_class,
                'confidence': conf_score,
                'risk_level': risk_level,
                'features': feature_values
            })
            
            return {
                'prediction': threat_types[pred_class],
                'confidence': f"{conf_score:.3f}",
                'risk_level': risk_level,
                'risk_color': risk_color,
                'probabilities': {threat_types[i]: f"{prob:.3f}" for i, prob in enumerate(probs)},
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
        except Exception as e:
            return {
                'prediction': "Error",
                'confidence': "0.000",
                'risk_level': "Unknown",
                'risk_color': "gray",
                'probabilities': {},
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'error': str(e)
            }
    
    def batch_analysis(self, csv_file) -> Tuple[pd.DataFrame, go.Figure, go.Figure, go.Figure]:
        """Batch analysis with comprehensive visualizations."""
        try:
            # Read CSV file
            df = pd.read_csv(csv_file.name)
            
            # Assume last column is target, rest are features
            feature_cols = df.columns[:-1]
            target_col = df.columns[-1]
            
            # Prepare data
            features = df[feature_cols].values
            targets = df[target_col].values
            
            # Get predictions
            with torch.no_grad():
                feature_tensor = torch.tensor(features, dtype=torch.float32)
                logits = self.model(feature_tensor)
                probabilities = torch.softmax(logits, dim=1)
                predictions = torch.argmax(probabilities, dim=1)
            
            # Sanitize outputs
            predictions = self._sanitize_output(predictions.cpu().numpy())
            probabilities = self._sanitize_output(probabilities.cpu().numpy())
            
            # Add predictions to dataframe
            df['Predicted'] = predictions
            df['Confidence'] = np.max(probabilities, axis=1)
            df['Correct'] = (predictions == targets)
            
            # Create visualizations
            # 1. Confusion Matrix
            cm = confusion_matrix(targets, predictions)
            
            fig1 = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Normal', 'Malware', 'Intrusion', 'Exfiltration'],
                y=['Normal', 'Malware', 'Intrusion', 'Exfiltration'],
                colorscale='Blues',
                text=cm,
                texttemplate="%{text}",
                textfont={"size": 16}
            ))
            fig1.update_layout(
                title="Confusion Matrix",
                xaxis_title="Predicted",
                yaxis_title="Actual",
                width=500,
                height=400
            )
            
            # 2. ROC Curve
            fig2 = go.Figure()
            
            for i in range(4):
                fpr, tpr, _ = roc_curve((targets == i).astype(int), probabilities[:, i])
                roc_auc = auc(fpr, tpr)
                
                fig2.add_trace(go.Scatter(
                    x=fpr, y=tpr,
                    name=f'Class {i} (AUC = {roc_auc:.3f})',
                    mode='lines'
                ))
            
            fig2.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                name='Random',
                mode='lines',
                line=dict(dash='dash', color='gray')
            ))
            
            fig2.update_layout(
                title="ROC Curves",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                width=500,
                height=400
            )
            
            # 3. Confidence Distribution
            fig3 = go.Figure()
            
            for i in range(4):
                class_confidences = df[df['Predicted'] == i]['Confidence']
                fig3.add_trace(go.Box(
                    y=class_confidences,
                    name=f'Class {i}',
                    boxpoints='outliers'
                ))
            
            fig3.update_layout(
                title="Confidence Distribution by Class",
                yaxis_title="Confidence Score",
                width=500,
                height=400
            )
            
            return df, fig1, fig2, fig3
            
        except Exception as e:
            # Return empty results on error
            empty_df = pd.DataFrame()
            empty_fig = go.Figure()
            return empty_df, empty_fig, empty_fig, empty_fig
    
    def anomaly_detection_demo(self, num_samples: int = 1000) -> Tuple[go.Figure, go.Figure, pd.DataFrame]:
        """Anomaly detection visualization demo."""
        # Generate demo data
        features, labels = self._generate_demo_data(num_samples)
        
        # Get model predictions
        with torch.no_grad():
            feature_tensor = torch.tensor(features, dtype=torch.float32)
            logits = self.model(feature_tensor)
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            confidence_scores = torch.max(probabilities, dim=1)[0]
        
        # Sanitize outputs
        predictions = self._sanitize_output(predictions.cpu().numpy())
        confidence_scores = self._sanitize_output(confidence_scores.cpu().numpy())
        
        # Create anomaly score (inverse of confidence for non-normal predictions)
        anomaly_scores = np.where(predictions == 0, 1 - confidence_scores, confidence_scores)
        
        # 1. Anomaly Score Distribution
        fig1 = go.Figure()
        
        for i in range(4):
            mask = predictions == i
            fig1.add_trace(go.Histogram(
                x=anomaly_scores[mask],
                name=f'Class {i}',
                opacity=0.7,
                nbinsx=30
            ))
        
        fig1.update_layout(
            title="Anomaly Score Distribution",
            xaxis_title="Anomaly Score",
            yaxis_title="Frequency",
            barmode='overlay',
            width=600,
            height=400
        )
        
        # 2. Feature Importance (simplified)
        feature_importance = np.abs(features).mean(axis=0)
        top_features = np.argsort(feature_importance)[-10:]
        
        fig2 = go.Figure(data=go.Bar(
            x=[f'Feature {i+1}' for i in top_features],
            y=feature_importance[top_features],
            marker_color='lightcoral'
        ))
        
        fig2.update_layout(
            title="Top 10 Most Important Features",
            xaxis_title="Features",
            yaxis_title="Average Absolute Value",
            width=600,
            height=400
        )
        
        # 3. Results summary
        results_df = pd.DataFrame({
            'Class': ['Normal', 'Malware', 'Intrusion', 'Exfiltration'],
            'Count': [np.sum(predictions == i) for i in range(4)],
            'Percentage': [np.sum(predictions == i) / len(predictions) * 100 for i in range(4)],
            'Avg_Confidence': [np.mean(confidence_scores[predictions == i]) for i in range(4)],
            'Avg_Anomaly_Score': [np.mean(anomaly_scores[predictions == i]) for i in range(4)]
        })
        
        return fig1, fig2, results_df
    
    def model_performance_dashboard(self) -> Tuple[go.Figure, go.Figure, pd.DataFrame]:
        """Real-time performance dashboard."""
        if not self.prediction_history:
            # Return empty dashboard if no history
            empty_fig = go.Figure()
            empty_df = pd.DataFrame()
            return empty_fig, empty_fig, empty_df
        
        # Convert history to dataframe
        history_df = pd.DataFrame(self.prediction_history)
        history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
        
        # 1. Prediction trends over time
        fig1 = go.Figure()
        
        for i in range(4):
            mask = history_df['prediction'] == i
            if mask.any():
                fig1.add_trace(go.Scatter(
                    x=history_df[mask]['timestamp'],
                    y=history_df[mask]['confidence'],
                    mode='markers',
                    name=f'Class {i}',
                    marker=dict(size=8)
                ))
        
        fig1.update_layout(
            title="Prediction Confidence Over Time",
            xaxis_title="Time",
            yaxis_title="Confidence Score",
            width=600,
            height=400
        )
        
        # 2. Risk level distribution
        risk_counts = history_df['risk_level'].value_counts()
        
        fig2 = go.Figure(data=go.Pie(
            labels=risk_counts.index,
            values=risk_counts.values,
            hole=0.3
        ))
        
        fig2.update_layout(
            title="Risk Level Distribution",
            width=400,
            height=400
        )
        
        # 3. Performance summary
        performance_summary = pd.DataFrame({
            'Metric': [
                'Total Predictions',
                'High Risk Predictions',
                'Medium Risk Predictions', 
                'Low Risk Predictions',
                'Average Confidence',
                'Highest Confidence',
                'Lowest Confidence'
            ],
            'Value': [
                len(history_df),
                len(history_df[history_df['risk_level'] == 'High']),
                len(history_df[history_df['risk_level'] == 'Medium']),
                len(history_df[history_df['risk_level'] == 'Low']),
                f"{history_df['confidence'].mean():.3f}",
                f"{history_df['confidence'].max():.3f}",
                f"{history_df['confidence'].min():.3f}"
            ]
        })
        
        return fig1, fig2, performance_summary
    
    def model_interpretability(self, feature_values: List[float]) -> Tuple[go.Figure, pd.DataFrame]:
        """Model interpretability and feature importance."""
        try:
            # Create baseline (all zeros)
            baseline = torch.zeros(1, 20)
            
            # Create input
            input_tensor = torch.tensor(feature_values, dtype=torch.float32).unsqueeze(0)
            
            # Simple feature importance using gradients
            input_tensor.requires_grad_(True)
            
            with torch.enable_grad():
                output = self.model(input_tensor)
                output.backward()
                
                # Get gradients
                gradients = input_tensor.grad.abs().cpu().numpy().flatten()
            
            # Sanitize gradients
            gradients = self._sanitize_output(gradients)
            
            # Create feature importance plot
            fig = go.Figure(data=go.Bar(
                x=[f'Feature {i+1}' for i in range(20)],
                y=gradients,
                marker_color='lightblue'
            ))
            
            fig.update_layout(
                title="Feature Importance (Gradient-based)",
                xaxis_title="Features",
                yaxis_title="Importance Score",
                width=800,
                height=400
            )
            
            # Create feature analysis table
            feature_analysis = pd.DataFrame({
                'Feature': [f'Feature {i+1}' for i in range(20)],
                'Value': feature_values,
                'Importance': gradients,
                'Normalized_Value': [(v - min(feature_values)) / (max(feature_values) - min(feature_values)) for v in feature_values]
            })
            
            feature_analysis = feature_analysis.sort_values('Importance', ascending=False)
            
            return fig, feature_analysis
            
        except Exception as e:
            empty_fig = go.Figure()
            empty_df = pd.DataFrame()
            return empty_fig, empty_df


def create_advanced_interfaces():
    """Create and configure advanced Gradio interfaces."""
    
    # Initialize model interface
    model_interface = CybersecurityModelInterface()
    
    # Custom CSS for better styling
    custom_css = """
    .gradio-container {
        max-width: 1200px !important;
    }
    .main-header {
        text-align: center;
        color: #2c3e50;
        margin-bottom: 20px;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    """
    
    # 1. Real-time Inference Interface
    with gr.Blocks(css=custom_css, title="Cybersecurity AI Model Showcase") as demo:
        gr.HTML("""
        <div class="main-header">
            <h1>🛡️ Cybersecurity AI Model Showcase</h1>
            <p>Advanced machine learning models for threat detection and security analysis</p>
        </div>
        """)
        
        with gr.Tabs():
            # Tab 1: Real-time Inference
            with gr.TabItem("🔍 Real-time Inference"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Input Features")
                        feature_inputs = []
                        for i in range(20):
                            feature_inputs.append(
                                gr.Number(
                                    label=f"Feature {i+1}",
                                    value=np.random.normal(0, 1),
                                    precision=3
                                )
                            )
                        
                        analyze_btn = gr.Button("🚀 Analyze Threat", variant="primary")
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### Analysis Results")
                        
                        with gr.Row():
                            prediction_output = gr.Textbox(label="Threat Type", interactive=False)
                            confidence_output = gr.Textbox(label="Confidence", interactive=False)
                        
                        with gr.Row():
                            risk_output = gr.Textbox(label="Risk Level", interactive=False)
                            timestamp_output = gr.Textbox(label="Timestamp", interactive=False)
                        
                        gr.Markdown("### Class Probabilities")
                        probability_output = gr.JSON(label="Probability Distribution")
                
                analyze_btn.click(
                    fn=model_interface.real_time_inference,
                    inputs=feature_inputs,
                    outputs=[prediction_output, confidence_output, risk_output, timestamp_output, probability_output]
                )
            
            # Tab 2: Batch Analysis
            with gr.TabItem("📊 Batch Analysis"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Upload Data")
                        file_input = gr.File(
                            label="Upload CSV file with features and labels",
                            file_types=[".csv"]
                        )
                        analyze_batch_btn = gr.Button("📈 Analyze Batch", variant="primary")
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### Results Summary")
                        results_table = gr.Dataframe(label="Analysis Results")
                
                with gr.Row():
                    confusion_matrix_plot = gr.Plot(label="Confusion Matrix")
                    roc_plot = gr.Plot(label="ROC Curves")
                
                confidence_dist_plot = gr.Plot(label="Confidence Distribution")
                
                analyze_batch_btn.click(
                    fn=model_interface.batch_analysis,
                    inputs=file_input,
                    outputs=[results_table, confusion_matrix_plot, roc_plot, confidence_dist_plot]
                )
            
            # Tab 3: Anomaly Detection
            with gr.TabItem("🚨 Anomaly Detection"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Demo Configuration")
                        sample_size = gr.Slider(
                            minimum=100,
                            maximum=5000,
                            value=1000,
                            step=100,
                            label="Number of Samples"
                        )
                        run_anomaly_btn = gr.Button("🔍 Run Anomaly Detection", variant="primary")
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### Detection Summary")
                        anomaly_summary = gr.Dataframe(label="Anomaly Detection Results")
                
                with gr.Row():
                    anomaly_dist_plot = gr.Plot(label="Anomaly Score Distribution")
                    feature_importance_plot = gr.Plot(label="Feature Importance")
                
                run_anomaly_btn.click(
                    fn=model_interface.anomaly_detection_demo,
                    inputs=sample_size,
                    outputs=[anomaly_dist_plot, feature_importance_plot, anomaly_summary]
                )
            
            # Tab 4: Performance Dashboard
            with gr.TabItem("📈 Performance Dashboard"):
                with gr.Row():
                    refresh_btn = gr.Button("🔄 Refresh Dashboard", variant="primary")
                    clear_btn = gr.Button("🗑️ Clear History", variant="secondary")
                
                with gr.Row():
                    trend_plot = gr.Plot(label="Prediction Trends")
                    risk_dist_plot = gr.Plot(label="Risk Distribution")
                
                performance_summary = gr.Dataframe(label="Performance Metrics")
                
                refresh_btn.click(
                    fn=model_interface.model_performance_dashboard,
                    inputs=[],
                    outputs=[trend_plot, risk_dist_plot, performance_summary]
                )
                
                def clear_history():
                    
    """clear_history function."""
model_interface.prediction_history = []
                    return gr.update(), gr.update(), gr.update()
                
                clear_btn.click(
                    fn=clear_history,
                    inputs=[],
                    outputs=[trend_plot, risk_dist_plot, performance_summary]
                )
            
            # Tab 5: Model Interpretability
            with gr.TabItem("🔬 Model Interpretability"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Input Features for Analysis")
                        interpret_inputs = []
                        for i in range(20):
                            interpret_inputs.append(
                                gr.Number(
                                    label=f"Feature {i+1}",
                                    value=np.random.normal(0, 1),
                                    precision=3
                                )
                            )
                        
                        interpret_btn = gr.Button("🔍 Analyze Features", variant="primary")
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### Feature Analysis")
                        feature_analysis_table = gr.Dataframe(label="Feature Importance Analysis")
                
                feature_importance_plot = gr.Plot(label="Feature Importance")
                
                interpret_btn.click(
                    fn=model_interface.model_interpretability,
                    inputs=interpret_inputs,
                    outputs=[feature_importance_plot, feature_analysis_table]
                )
    
    return demo


if __name__ == "__main__":
    # Create and launch the interface
    demo = create_advanced_interfaces()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True,
        show_tips=True
    ) 