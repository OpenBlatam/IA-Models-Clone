#!/usr/bin/env python3
"""
Interactive Gradio Demos for Model Inference and Visualization
Advanced demonstrations of gradient clipping and NaN handling with real-time visualization.
"""

import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional, Tuple, Union
import json
import time
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

# Import our gradient clipping and NaN handling system
from gradient_clipping_nan_handling import (
    GradientClippingConfig,
    NaNHandlingConfig,
    NumericalStabilityManager,
    ClippingType,
    NaNHandlingType,
    create_training_wrapper
)


class AdvancedModelDemo:
    """Advanced model demonstration with real-time inference and visualization."""
    
    def __init__(self):
        self.models = {}
        self.stability_managers = {}
        self.inference_history = []
        self.current_demo = None
        
    def create_demo_model(self, model_type: str, complexity: str) -> str:
        """Create a demonstration model based on type and complexity."""
        try:
            if model_type == "Classification":
                if complexity == "Simple":
                    model = nn.Sequential(
                        nn.Linear(2, 64),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(64, 32),
                        nn.ReLU(),
                        nn.Linear(32, 3)
                    )
                elif complexity == "Medium":
                    model = nn.Sequential(
                        nn.Linear(2, 128),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(128, 64),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(64, 32),
                        nn.ReLU(),
                        nn.Linear(32, 3)
                    )
                else:  # Complex
                    model = nn.Sequential(
                        nn.Linear(2, 256),
                        nn.ReLU(),
                        nn.Dropout(0.4),
                        nn.Linear(256, 128),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(128, 64),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(64, 32),
                        nn.ReLU(),
                        nn.Linear(32, 3)
                    )
                    
            elif model_type == "Regression":
                if complexity == "Simple":
                    model = nn.Sequential(
                        nn.Linear(1, 32),
                        nn.ReLU(),
                        nn.Linear(32, 16),
                        nn.ReLU(),
                        nn.Linear(16, 1)
                    )
                elif complexity == "Medium":
                    model = nn.Sequential(
                        nn.Linear(1, 64),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(64, 32),
                        nn.ReLU(),
                        nn.Linear(32, 16),
                        nn.ReLU(),
                        nn.Linear(16, 1)
                    )
                else:  # Complex
                    model = nn.Sequential(
                        nn.Linear(1, 128),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(128, 64),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(64, 32),
                        nn.ReLU(),
                        nn.Linear(32, 16),
                        nn.ReLU(),
                        nn.Linear(16, 1)
                    )
                    
            elif model_type == "Autoencoder":
                if complexity == "Simple":
                    model = nn.Sequential(
                        nn.Linear(10, 32),
                        nn.ReLU(),
                        nn.Linear(32, 16),
                        nn.ReLU(),
                        nn.Linear(16, 8),
                        nn.ReLU(),
                        nn.Linear(8, 16),
                        nn.ReLU(),
                        nn.Linear(16, 32),
                        nn.ReLU(),
                        nn.Linear(32, 10)
                    )
                elif complexity == "Medium":
                    model = nn.Sequential(
                        nn.Linear(10, 64),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(64, 32),
                        nn.ReLU(),
                        nn.Linear(32, 16),
                        nn.ReLU(),
                        nn.Linear(16, 8),
                        nn.ReLU(),
                        nn.Linear(8, 16),
                        nn.ReLU(),
                        nn.Linear(16, 32),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(32, 64),
                        nn.ReLU(),
                        nn.Linear(64, 10)
                    )
                else:  # Complex
                    model = nn.Sequential(
                        nn.Linear(10, 128),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(128, 64),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(64, 32),
                        nn.ReLU(),
                        nn.Linear(32, 16),
                        nn.ReLU(),
                        nn.Linear(16, 8),
                        nn.ReLU(),
                        nn.Linear(8, 16),
                        nn.ReLU(),
                        nn.Linear(16, 32),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(32, 64),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(64, 128),
                        nn.ReLU(),
                        nn.Linear(128, 10)
                    )
            
            # Initialize weights with some variance to make training interesting
            for module in model.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.constant_(module.bias, 0.1)
            
            model_id = f"{model_type}_{complexity}_{int(time.time())}"
            self.models[model_id] = model
            
            param_count = sum(p.numel() for p in model.parameters())
            return f"✅ {model_type} model ({complexity}) created successfully!\nModel ID: {model_id}\nParameters: {param_count:,}"
            
        except Exception as e:
            return f"❌ Error creating model: {str(e)}"
    
    def configure_stability_for_demo(self, model_id: str, clipping_type: str, 
                                   max_norm: float, nan_handling: str) -> str:
        """Configure stability manager for a specific demo model."""
        try:
            if model_id not in self.models:
                return "❌ Model not found. Please create a model first."
            
            # Create stability configuration
            clipping_config = GradientClippingConfig(
                clipping_type=getattr(ClippingType, clipping_type.upper()),
                max_norm=max_norm,
                monitor_clipping=True,
                log_clipping_stats=True,
                save_clipping_history=True
            )
            
            nan_config = NaNHandlingConfig(
                handling_type=getattr(NaNHandlingType, nan_handling.upper()),
                detect_nan=True,
                detect_inf=True,
                detect_overflow=True,
                monitor_nan=True,
                log_nan_stats=True,
                save_nan_history=True
            )
            
            # Create stability manager
            stability_manager = NumericalStabilityManager(clipping_config, nan_config)
            self.stability_managers[model_id] = stability_manager
            
            return f"✅ Stability manager configured for {model_id}!\nClipping: {clipping_type}\nMax norm: {max_norm}\nNaN handling: {nan_handling}"
            
        except Exception as e:
            return f"❌ Error configuring stability: {str(e)}"
    
    def run_inference_demo(self, model_id: str, input_data: str, 
                          introduce_issues: bool, issue_probability: float) -> Tuple[str, Dict[str, Any]]:
        """Run inference demo with the specified model."""
        try:
            if model_id not in self.models:
                return "❌ Model not found.", {}
            
            model = self.models[model_id]
            model.eval()
            
            # Parse input data
            try:
                if model_id.startswith("Classification"):
                    # Parse 2D input for classification
                    coords = [float(x.strip()) for x in input_data.split(',')]
                    if len(coords) != 2:
                        return "❌ Classification model expects 2 comma-separated values (x, y)", {}
                    input_tensor = torch.tensor([coords], dtype=torch.float32)
                    
                elif model_id.startswith("Regression"):
                    # Parse 1D input for regression
                    x_val = float(input_data.strip())
                    input_tensor = torch.tensor([[x_val]], dtype=torch.float32)
                    
                elif model_id.startswith("Autoencoder"):
                    # Parse 10D input for autoencoder
                    values = [float(x.strip()) for x in input_data.split(',')]
                    if len(values) != 10:
                        return "❌ Autoencoder expects 10 comma-separated values", {}
                    input_tensor = torch.tensor([values], dtype=torch.float32)
                    
            except ValueError:
                return "❌ Invalid input format. Please check your input data.", {}
            
            # Run inference
            with torch.no_grad():
                output = model(input_tensor)
                
                if model_id.startswith("Classification"):
                    probabilities = F.softmax(output, dim=1)
                    predicted_class = torch.argmax(probabilities, dim=1).item()
                    confidence = probabilities[0][predicted_class].item()
                    
                    result = {
                        'input': input_tensor.numpy().tolist(),
                        'output': output.numpy().tolist(),
                        'probabilities': probabilities.numpy().tolist(),
                        'predicted_class': predicted_class,
                        'confidence': confidence,
                        'model_type': 'classification'
                    }
                    
                elif model_id.startswith("Regression"):
                    predicted_value = output.item()
                    
                    result = {
                        'input': input_tensor.numpy().tolist(),
                        'output': output.numpy().tolist(),
                        'predicted_value': predicted_value,
                        'model_type': 'regression'
                    }
                    
                elif model_id.startswith("Autoencoder"):
                    reconstructed = output
                    reconstruction_error = F.mse_loss(input_tensor, reconstructed).item()
                    
                    result = {
                        'input': input_tensor.numpy().tolist(),
                        'output': output.numpy().tolist(),
                        'reconstruction_error': reconstruction_error,
                        'model_type': 'autoencoder'
                    }
            
            # Introduce numerical issues if requested
            if introduce_issues and model_id in self.stability_managers:
                # Switch to training mode to introduce issues
                model.train()
                
                # Create a dummy loss for demonstration
                if model_id.startswith("Classification"):
                    target = torch.tensor([predicted_class], dtype=torch.long)
                    loss = F.cross_entropy(output, target)
                elif model_id.startswith("Regression"):
                    target = torch.tensor([[predicted_value]], dtype=torch.float32)
                    loss = F.mse_loss(output, target)
                else:  # Autoencoder
                    loss = F.mse_loss(output, input_tensor)
                
                # Backward pass to generate gradients
                loss.backward()
                
                # Introduce issues based on probability
                if np.random.random() < issue_probability:
                    for param in model.parameters():
                        if param.grad is not None:
                            if np.random.random() < 0.5:
                                param.grad.data[0, 0] = float('nan')
                            else:
                                param.grad.data *= 1e6
                
                # Apply stability measures
                stability_result = self.stability_managers[model_id].step(model, loss, None)
                result['stability_result'] = stability_result
                
                # Switch back to eval mode
                model.eval()
            
            # Store inference history
            self.inference_history.append({
                'timestamp': time.time(),
                'model_id': model_id,
                'input': input_tensor.numpy().tolist(),
                'result': result,
                'introduced_issues': introduce_issues
            })
            
            # Prepare result message
            if model_id.startswith("Classification"):
                msg = f"✅ Classification inference completed!\n"
                msg += f"Input: ({coords[0]:.3f}, {coords[1]:.3f})\n"
                msg += f"Predicted class: {predicted_class}\n"
                msg += f"Confidence: {confidence:.4f}\n"
                if introduce_issues and 'stability_result' in result:
                    msg += f"Numerical issues introduced: {result['stability_result']['nan_stats']['nan_detected']}"
                    
            elif model_id.startswith("Regression"):
                msg = f"✅ Regression inference completed!\n"
                msg += f"Input: {x_val:.3f}\n"
                msg += f"Predicted value: {predicted_value:.4f}\n"
                if introduce_issues and 'stability_result' in result:
                    msg += f"Numerical issues introduced: {result['stability_result']['nan_stats']['nan_detected']}"
                    
            else:  # Autoencoder
                msg = f"✅ Autoencoder inference completed!\n"
                msg += f"Input dimension: {len(values)}\n"
                msg += f"Reconstruction error: {reconstruction_error:.6f}\n"
                if introduce_issues and 'stability_result' in result:
                    msg += f"Numerical issues introduced: {result['stability_result']['nan_stats']['nan_detected']}"
            
            return msg, result
            
        except Exception as e:
            return f"❌ Error in inference: {str(e)}", {}
    
    def generate_interactive_plots(self, model_id: str) -> Tuple[go.Figure, go.Figure]:
        """Generate interactive Plotly plots for model visualization."""
        try:
            if model_id not in self.models:
                # Return empty plots
                fig1 = go.Figure()
                fig1.add_annotation(text="No model selected", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
                fig1.update_layout(title="Model Architecture")
                
                fig2 = go.Figure()
                fig2.add_annotation(text="No inference data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
                fig2.update_layout(title="Inference Results")
                
                return fig1, fig2
            
            model = self.models[model_id]
            
            # Plot 1: Model Architecture Visualization
            fig1 = self._create_model_architecture_plot(model, model_id)
            
            # Plot 2: Inference Results
            fig2 = self._create_inference_results_plot(model_id)
            
            return fig1, fig2
            
        except Exception as e:
            # Return error plots
            fig1 = go.Figure()
            fig1.add_annotation(text=f"Error: {str(e)}", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            fig1.update_layout(title="Model Architecture")
            
            fig2 = go.Figure()
            fig2.add_annotation(text=f"Error: {str(e)}", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            fig2.update_layout(title="Inference Results")
            
            return fig1, fig2
    
    def _create_model_architecture_plot(self, model: nn.Module, model_id: str) -> go.Figure:
        """Create a visualization of the model architecture."""
        layers = []
        layer_sizes = []
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                layers.append(f"{name}\n({module.in_features} → {module.out_features})")
                layer_sizes.append(module.out_features)
            elif isinstance(module, (nn.ReLU, nn.Dropout)):
                layers.append(f"{name}")
                layer_sizes.append(layer_sizes[-1] if layer_sizes else 1)
        
        # Create the plot
        fig = go.Figure()
        
        # Add layer nodes
        for i, (layer, size) in enumerate(zip(layers, layer_sizes)):
            fig.add_trace(go.Scatter(
                x=[i],
                y=[0],
                mode='markers+text',
                marker=dict(
                    size=min(max(size * 2, 20), 100),
                    color='lightblue',
                    line=dict(color='darkblue', width=2)
                ),
                text=[layer],
                textposition='middle center',
                name=layer,
                showlegend=False
            ))
        
        # Add connections
        for i in range(len(layers) - 1):
            fig.add_trace(go.Scatter(
                x=[i, i + 1],
                y=[0, 0],
                mode='lines',
                line=dict(color='gray', width=2),
                showlegend=False
            ))
        
        fig.update_layout(
            title=f"Model Architecture: {model_id}",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            height=400
        )
        
        return fig
    
    def _create_inference_results_plot(self, model_id: str) -> go.Figure:
        """Create a plot of inference results."""
        if not self.inference_history:
            fig = go.Figure()
            fig.add_annotation(text="No inference data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            fig.update_layout(title="Inference Results")
            return fig
        
        # Filter history for this model
        model_history = [h for h in self.inference_history if h['model_id'] == model_id]
        
        if not model_history:
            fig = go.Figure()
            fig.add_annotation(text="No inference data for this model", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            fig.update_layout(title="Inference Results")
            return fig
        
        # Create subplots based on model type
        if model_id.startswith("Classification"):
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Input Coordinates', 'Class Probabilities', 'Confidence Over Time', 'Issue Detection'),
                specs=[[{"type": "scatter"}, {"type": "bar"}],
                       [{"type": "scatter"}, {"type": "scatter"}]]
            )
            
            # Input coordinates
            x_coords = [h['input'][0][0] for h in model_history]
            y_coords = [h['input'][0][1] for h in model_history]
            fig.add_trace(
                go.Scatter(x=x_coords, y=y_coords, mode='markers', name='Input Points'),
                row=1, col=1
            )
            
            # Class probabilities (last inference)
            last_result = model_history[-1]['result']
            classes = ['Class 0', 'Class 1', 'Class 2']
            probs = last_result['probabilities'][0]
            fig.add_trace(
                go.Bar(x=classes, y=probs, name='Probabilities'),
                row=1, col=2
            )
            
            # Confidence over time
            timestamps = [h['timestamp'] for h in model_history]
            confidences = [h['result']['confidence'] for h in model_history]
            fig.add_trace(
                go.Scatter(x=timestamps, y=confidences, mode='lines+markers', name='Confidence'),
                row=2, col=1
            )
            
            # Issue detection
            issues_introduced = [h['introduced_issues'] for h in model_history]
            fig.add_trace(
                go.Scatter(x=timestamps, y=issues_introduced, mode='markers', name='Issues Introduced'),
                row=2, col=2
            )
            
        elif model_id.startswith("Regression"):
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Input vs Output', 'Prediction Error', 'Output Distribution', 'Issue Detection'),
                specs=[[{"type": "scatter"}, {"type": "histogram"}],
                       [{"type": "scatter"}, {"type": "scatter"}]]
            )
            
            # Input vs Output
            inputs = [h['input'][0][0] for h in model_history]
            outputs = [h['result']['predicted_value'] for h in model_history]
            fig.add_trace(
                go.Scatter(x=inputs, y=outputs, mode='markers', name='Input vs Output'),
                row=1, col=1
            )
            
            # Output distribution
            fig.add_trace(
                go.Histogram(x=outputs, name='Output Distribution'),
                row=1, col=2
            )
            
            # Prediction error (assuming linear relationship for demo)
            expected = [x * 0.5 for x in inputs]  # Simple expected relationship
            errors = [abs(o - e) for o, e in zip(outputs, expected)]
            fig.add_trace(
                go.Scatter(x=inputs, y=errors, mode='markers', name='Prediction Error'),
                row=2, col=1
            )
            
            # Issue detection
            timestamps = [h['timestamp'] for h in model_history]
            issues_introduced = [h['introduced_issues'] for h in model_history]
            fig.add_trace(
                go.Scatter(x=timestamps, y=issues_introduced, mode='markers', name='Issues Introduced'),
                row=2, col=2
            )
            
        else:  # Autoencoder
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Input vs Reconstructed', 'Reconstruction Error', 'Error Distribution', 'Issue Detection'),
                specs=[[{"type": "scatter"}, {"type": "histogram"}],
                       [{"type": "scatter"}, {"type": "scatter"}]]
            )
            
            # Input vs Reconstructed (using first dimension for simplicity)
            inputs = [h['input'][0][0] for h in model_history]
            reconstructed = [h['result']['output'][0][0] for h in model_history]
            fig.add_trace(
                go.Scatter(x=inputs, y=reconstructed, mode='markers', name='Input vs Reconstructed'),
                row=1, col=1
            )
            
            # Reconstruction error distribution
            errors = [h['result']['reconstruction_error'] for h in model_history]
            fig.add_trace(
                go.Histogram(x=errors, name='Error Distribution'),
                row=1, col=2
            )
            
            # Error over time
            timestamps = [h['timestamp'] for h in model_history]
            fig.add_trace(
                go.Scatter(x=timestamps, y=errors, mode='lines+markers', name='Error Over Time'),
                row=2, col=1
            )
            
            # Issue detection
            issues_introduced = [h['introduced_issues'] for h in model_history]
            fig.add_trace(
                go.Scatter(x=timestamps, y=issues_introduced, mode='markers', name='Issues Introduced'),
                row=2, col=2
            )
        
        fig.update_layout(
            title=f"Inference Results: {model_id}",
            height=600,
            showlegend=True
        )
        
        return fig
    
    def get_demo_summary(self) -> str:
        """Get a summary of all demo activities."""
        if not self.models:
            return "No models created yet."
        
        summary = f"📊 Demo Session Summary\n"
        summary += f"{'='*40}\n"
        summary += f"Models Created: {len(self.models)}\n"
        summary += f"Stability Managers: {len(self.stability_managers)}\n"
        summary += f"Total Inferences: {len(self.inference_history)}\n\n"
        
        summary += "Models:\n"
        for model_id in self.models:
            param_count = sum(p.numel() for p in self.models[model_id].parameters())
            summary += f"  • {model_id}: {param_count:,} parameters\n"
        
        if self.inference_history:
            summary += f"\nRecent Inferences:\n"
            for i, inference in enumerate(self.inference_history[-5:]):  # Last 5
                summary += f"  • {inference['model_id']}: {inference['timestamp']:.0f}s ago\n"
        
        return summary
    
    def clear_demo_session(self) -> str:
        """Clear all demo data."""
        self.models.clear()
        self.stability_managers.clear()
        self.inference_history.clear()
        return "✅ Demo session cleared successfully!"


def create_interactive_demo_interface():
    """Create the interactive demo Gradio interface."""
    demo_system = AdvancedModelDemo()
    
    with gr.Blocks(title="Interactive Model Inference & Visualization Demos", theme=gr.themes.Soft()) as interface:
        gr.Markdown("""
        # 🚀 Interactive Model Inference & Visualization Demos
        
        Advanced demonstrations of gradient clipping and NaN handling with real-time model inference,
        interactive visualizations, and comprehensive analysis tools.
        
        **Features:**
        - 🎯 Multiple model types (Classification, Regression, Autoencoder)
        - 🔧 Configurable complexity levels
        - 📊 Real-time inference with numerical issue injection
        - 📈 Interactive Plotly visualizations
        - 🚨 Stability monitoring and analysis
        """)
        
        with gr.Tabs():
            # Tab 1: Model Creation
            with gr.Tab("🏗️ Model Creation"):
                gr.Markdown("### Create demonstration models with different architectures")
                
                with gr.Row():
                    with gr.Column():
                        model_type = gr.Dropdown(
                            choices=["Classification", "Regression", "Autoencoder"],
                            value="Classification",
                            label="Model Type"
                        )
                        complexity = gr.Dropdown(
                            choices=["Simple", "Medium", "Complex"],
                            value="Simple",
                            label="Model Complexity"
                        )
                        
                        create_model_btn = gr.Button("🚀 Create Demo Model", variant="primary")
                        model_status = gr.Textbox(label="Model Creation Status", lines=3)
                    
                    with gr.Column():
                        gr.Markdown("### Model Information")
                        model_info = gr.Markdown("No models created yet.")
                
                create_model_btn.click(
                    fn=demo_system.create_demo_model,
                    inputs=[model_type, complexity],
                    outputs=model_status
                )
            
            # Tab 2: Stability Configuration
            with gr.Tab("⚙️ Stability Setup"):
                gr.Markdown("### Configure numerical stability for demo models")
                
                with gr.Row():
                    with gr.Column():
                        model_selector = gr.Dropdown(
                            choices=[],
                            label="Select Model",
                            placeholder="Choose a model to configure"
                        )
                        clipping_type = gr.Dropdown(
                            choices=["NORM", "VALUE", "GLOBAL_NORM", "ADAPTIVE", "LAYER_WISE", "PERCENTILE", "EXPONENTIAL"],
                            value="NORM",
                            label="Gradient Clipping Type"
                        )
                        max_norm = gr.Slider(minimum=0.1, maximum=10.0, value=1.0, step=0.1, label="Max Norm")
                        
                    with gr.Column():
                        nan_handling = gr.Dropdown(
                            choices=["DETECT", "REPLACE", "SKIP", "GRADIENT_ZEROING", "ADAPTIVE", "GRADIENT_SCALING"],
                            value="ADAPTIVE",
                            label="NaN/Inf Handling"
                        )
                        
                        config_btn = gr.Button("🔧 Configure Stability", variant="primary")
                        config_status = gr.Textbox(label="Configuration Status", lines=3)
                
                config_btn.click(
                    fn=demo_system.configure_stability_for_demo,
                    inputs=[model_selector, clipping_type, max_norm, nan_handling],
                    outputs=config_status
                )
            
            # Tab 3: Interactive Inference
            with gr.Tab("🎯 Interactive Inference"):
                gr.Markdown("### Run real-time inference with numerical issue injection")
                
                with gr.Row():
                    with gr.Column():
                        inference_model = gr.Dropdown(
                            choices=[],
                            label="Select Model for Inference",
                            placeholder="Choose a model to run inference"
                        )
                        
                        gr.Markdown("### Input Data Format")
                        gr.Markdown("""
                        **Classification**: Enter 2 comma-separated values (x, y)\n
                        **Regression**: Enter 1 value (x)\n
                        **Autoencoder**: Enter 10 comma-separated values
                        """)
                        
                        input_data = gr.Textbox(
                            value="0.5, 0.3",
                            label="Input Data",
                            placeholder="Enter input data according to format above"
                        )
                        
                        introduce_issues = gr.Checkbox(
                            label="Introduce Numerical Issues",
                            value=True
                        )
                        issue_probability = gr.Slider(
                            minimum=0.0, maximum=1.0, value=0.3, step=0.1,
                            label="Issue Introduction Probability"
                        )
                        
                        inference_btn = gr.Button("🎯 Run Inference", variant="primary")
                        
                    with gr.Column():
                        inference_status = gr.Textbox(label="Inference Status", lines=8)
                        inference_result = gr.JSON(label="Inference Results")
                
                inference_btn.click(
                    fn=demo_system.run_inference_demo,
                    inputs=[inference_model, input_data, introduce_issues, issue_probability],
                    outputs=[inference_status, inference_result]
                )
            
            # Tab 4: Interactive Visualization
            with gr.Tab("📊 Interactive Visualization"):
                gr.Markdown("### Real-time model visualization and inference analysis")
                
                with gr.Row():
                    plot_model = gr.Dropdown(
                        choices=[],
                        label="Select Model for Visualization",
                        placeholder="Choose a model to visualize"
                    )
                    plot_btn = gr.Button("📈 Generate Interactive Plots", variant="primary")
                
                with gr.Row():
                    architecture_plot = gr.Plot(label="Model Architecture")
                    inference_plot = gr.Plot(label="Inference Results")
                
                plot_btn.click(
                    fn=demo_system.generate_interactive_plots,
                    inputs=[plot_model],
                    outputs=[architecture_plot, inference_plot]
                )
            
            # Tab 5: Session Management
            with gr.Tab("💾 Session Management"):
                gr.Markdown("### Manage demo session and view summaries")
                
                with gr.Row():
                    summary_btn = gr.Button("📋 View Session Summary", variant="primary")
                    clear_btn = gr.Button("🗑️ Clear Session", variant="primary")
                
                with gr.Row():
                    session_summary = gr.Textbox(label="Session Summary", lines=15)
                
                summary_btn.click(
                    fn=demo_system.get_demo_summary,
                    outputs=session_summary
                )
                
                clear_btn.click(
                    fn=demo_system.clear_demo_session,
                    outputs=session_summary
                )
        
        # Footer
        gr.Markdown("""
        ---
        **Interactive Model Inference & Visualization Demos** | Built with Gradio & Plotly
        
        This interface demonstrates advanced numerical stability concepts through
        interactive model inference and real-time visualization.
        """)
    
    return interface


def main():
    """Main function to launch the interactive demo interface."""
    interface = create_interactive_demo_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7861,  # Different port from main interface
        share=True,
        show_error=True,
        show_tips=True
    )


if __name__ == "__main__":
    main() 