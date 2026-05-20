from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import gradio as gr
import asyncio
import logging
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import sys
from core.sequence_generator import EmailSequenceGenerator, GeneratorConfig
from core.evaluation_metrics import EmailSequenceEvaluator, MetricsConfig
from core.training_optimization import (
from core.gradient_management import GradientManager, GradientConfig
from models.sequence import EmailSequence, SequenceStep
from models.subscriber import Subscriber
from models.template import EmailTemplate
from typing import Any, List, Dict, Optional
"""
Gradio Web Interface for Email Sequence System

A comprehensive web interface for the email sequence AI system with
all features including sequence generation, evaluation, training,
and gradient management.
"""


# Add the parent directory to the path to import modules
sys.path.append(str(Path(__file__).parent))

    TrainingOptimizer,
    EarlyStoppingConfig,
    LRSchedulerConfig,
    GradientManagementConfig
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state
current_sequence = None
current_evaluator = None
current_training_optimizer = None
training_history = []


class GradioEmailSequenceApp:
    """Gradio application for email sequence system"""
    
    def __init__(self) -> Any:
        self.sequence_generator = None
        self.evaluator = None
        self.training_optimizer = None
        self.gradient_manager = None
        
        # Sample data for demonstration
        self.sample_subscribers = self._create_sample_subscribers()
        self.sample_templates = self._create_sample_templates()
        
        logger.info("Gradio Email Sequence App initialized")
    
    def _create_sample_subscribers(self) -> List[Subscriber]:
        """Create sample subscribers for demonstration"""
        return [
            Subscriber(
                id="sub_1",
                email="john@techcorp.com",
                name="John Doe",
                company="Tech Corp",
                interests=["AI", "machine learning", "data science"],
                industry="Technology"
            ),
            Subscriber(
                id="sub_2",
                email="jane@marketinginc.com",
                name="Jane Smith",
                company="Marketing Inc",
                interests=["marketing", "social media", "content creation"],
                industry="Marketing"
            ),
            Subscriber(
                id="sub_3",
                email="bob@financeltd.com",
                name="Bob Johnson",
                company="Finance Ltd",
                interests=["finance", "investment", "business"],
                industry="Finance"
            )
        ]
    
    def _create_sample_templates(self) -> List[EmailTemplate]:
        """Create sample email templates"""
        return [
            EmailTemplate(
                id="template_1",
                name="Welcome Series",
                subject_template="Welcome to {company}!",
                content_template="Hi {name}, welcome to our platform. We're excited to help you succeed with {interest}."
            ),
            EmailTemplate(
                id="template_2",
                name="Feature Introduction",
                subject_template="Discover our {feature} feature",
                content_template="Hello {name}, we think you'll love our new {feature} feature. It's perfect for {industry} professionals."
            ),
            EmailTemplate(
                id="template_3",
                name="Conversion",
                subject_template="Ready to get started?",
                content_template="Hi {name}, based on your interest in {interest}, we think you're ready to take the next step. Click here to get started!"
            )
        ]
    
    def generate_sequence_interface(self) -> Any:
        """Create the sequence generation interface"""
        
        with gr.Tab("Sequence Generation"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("## Sequence Generation Configuration")
                    
                    # Generator configuration
                    model_type = gr.Dropdown(
                        choices=["GPT-3.5", "GPT-4", "Claude", "Custom"],
                        value="GPT-3.5",
                        label="AI Model"
                    )
                    
                    sequence_length = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=3,
                        step=1,
                        label="Sequence Length"
                    )
                    
                    creativity_level = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.7,
                        step=0.1,
                        label="Creativity Level"
                    )
                    
                    target_audience = gr.Dropdown(
                        choices=[f"{s.name} ({s.company})" for s in self.sample_subscribers],
                        value=f"{self.sample_subscribers[0].name} ({self.sample_subscribers[0].company})",
                        label="Target Audience"
                    )
                    
                    industry_focus = gr.Textbox(
                        value="Technology",
                        label="Industry Focus"
                    )
                    
                    generate_btn = gr.Button("Generate Sequence", variant="primary")
                
                with gr.Column(scale=2):
                    gr.Markdown("## Generated Sequence")
                    
                    sequence_output = gr.JSON(
                        label="Sequence JSON",
                        visible=True
                    )
                    
                    sequence_preview = gr.Markdown(
                        label="Sequence Preview"
                    )
            
            # Connect the generate button
            generate_btn.click(
                fn=self.generate_sequence,
                inputs=[
                    model_type,
                    sequence_length,
                    creativity_level,
                    target_audience,
                    industry_focus
                ],
                outputs=[sequence_output, sequence_preview]
            )
    
    def generate_sequence(
        self,
        model_type: str,
        sequence_length: int,
        creativity_level: float,
        target_audience: str,
        industry_focus: str
    ) -> Tuple[Dict, str]:
        """Generate email sequence"""
        
        try:
            # Parse target audience
            subscriber_name = target_audience.split(" (")[0]
            subscriber = next(s for s in self.sample_subscribers if s.name == subscriber_name)
            
            # Create generator configuration
            config = GeneratorConfig(
                model_type=model_type,
                sequence_length=sequence_length,
                creativity_level=creativity_level,
                industry_focus=industry_focus
            )
            
            # Initialize generator
            self.sequence_generator = EmailSequenceGenerator(config)
            
            # Generate sequence
            sequence = asyncio.run(self.sequence_generator.generate_sequence(
                target_audience=subscriber,
                templates=self.sample_templates
            ))
            
            # Convert to JSON-serializable format
            sequence_dict = {
                "id": sequence.id,
                "name": sequence.name,
                "description": sequence.description,
                "steps": [
                    {
                        "order": step.order,
                        "subscriber_id": step.subscriber_id,
                        "template_id": step.template_id,
                        "content": step.content,
                        "delay_hours": step.delay_hours
                    }
                    for step in sequence.steps
                ]
            }
            
            # Create preview
            preview = self._create_sequence_preview(sequence)
            
            # Store current sequence
            global current_sequence
            current_sequence = sequence
            
            return sequence_dict, preview
            
        except Exception as e:
            logger.error(f"Error generating sequence: {e}")
            return {"error": str(e)}, f"Error: {str(e)}"
    
    def _create_sequence_preview(self, sequence: EmailSequence) -> str:
        """Create a markdown preview of the sequence"""
        
        preview = f"# {sequence.name}\n\n"
        preview += f"**Description:** {sequence.description}\n\n"
        preview += f"**Total Steps:** {len(sequence.steps)}\n\n"
        
        for i, step in enumerate(sequence.steps, 1):
            preview += f"## Step {i}\n\n"
            preview += f"**Delay:** {step.delay_hours} hours\n\n"
            preview += f"**Content:**\n{step.content}\n\n"
            preview += "---\n\n"
        
        return preview
    
    def evaluation_interface(self) -> Any:
        """Create the evaluation interface"""
        
        with gr.Tab("Sequence Evaluation"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("## Evaluation Configuration")
                    
                    # Evaluation metrics configuration
                    enable_content_quality = gr.Checkbox(
                        value=True,
                        label="Content Quality Metrics"
                    )
                    
                    enable_engagement = gr.Checkbox(
                        value=True,
                        label="Engagement Metrics"
                    )
                    
                    enable_business_impact = gr.Checkbox(
                        value=True,
                        label="Business Impact Metrics"
                    )
                    
                    enable_technical = gr.Checkbox(
                        value=True,
                        label="Technical Metrics"
                    )
                    
                    content_weight = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.3,
                        step=0.1,
                        label="Content Quality Weight"
                    )
                    
                    engagement_weight = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.3,
                        step=0.1,
                        label="Engagement Weight"
                    )
                    
                    evaluate_btn = gr.Button("Evaluate Sequence", variant="primary")
                
                with gr.Column(scale=2):
                    gr.Markdown("## Evaluation Results")
                    
                    evaluation_results = gr.JSON(
                        label="Evaluation Results"
                    )
                    
                    evaluation_summary = gr.Markdown(
                        label="Evaluation Summary"
                    )
                    
                    evaluation_charts = gr.Plot(
                        label="Evaluation Charts"
                    )
            
            # Connect the evaluate button
            evaluate_btn.click(
                fn=self.evaluate_sequence,
                inputs=[
                    enable_content_quality,
                    enable_engagement,
                    enable_business_impact,
                    enable_technical,
                    content_weight,
                    engagement_weight
                ],
                outputs=[evaluation_results, evaluation_summary, evaluation_charts]
            )
    
    def evaluate_sequence(
        self,
        enable_content_quality: bool,
        enable_engagement: bool,
        enable_business_impact: bool,
        enable_technical: bool,
        content_weight: float,
        engagement_weight: float
    ) -> Tuple[Dict, str, go.Figure]:
        """Evaluate the current sequence"""
        
        try:
            if current_sequence is None:
                return {"error": "No sequence to evaluate"}, "Please generate a sequence first.", go.Figure()
            
            # Create evaluation configuration
            config = MetricsConfig(
                enable_content_quality=enable_content_quality,
                enable_engagement=enable_engagement,
                enable_business_impact=enable_business_impact,
                enable_technical=enable_technical,
                content_weight=content_weight,
                engagement_weight=engagement_weight
            )
            
            # Initialize evaluator
            self.evaluator = EmailSequenceEvaluator(config)
            
            # Evaluate sequence
            results = asyncio.run(self.evaluator.evaluate_sequence(
                sequence=current_sequence,
                subscribers=self.sample_subscribers,
                templates=self.sample_templates
            ))
            
            # Create summary
            summary = self._create_evaluation_summary(results)
            
            # Create charts
            charts = self._create_evaluation_charts(results)
            
            return results, summary, charts
            
        except Exception as e:
            logger.error(f"Error evaluating sequence: {e}")
            return {"error": str(e)}, f"Error: {str(e)}", go.Figure()
    
    def _create_evaluation_summary(self, results: Dict) -> str:
        """Create a markdown summary of evaluation results"""
        
        if "error" in results:
            return f"Error: {results['error']}"
        
        overall_metrics = results.get("overall_metrics", {})
        
        summary = "# Evaluation Summary\n\n"
        summary += f"**Overall Score:** {overall_metrics.get('overall_score', 0):.3f}\n\n"
        summary += f"**Content Quality:** {overall_metrics.get('content_quality_score', 0):.3f}\n\n"
        summary += f"**Engagement Score:** {overall_metrics.get('engagement_score', 0):.3f}\n\n"
        summary += f"**Business Impact:** {overall_metrics.get('business_impact_score', 0):.3f}\n\n"
        summary += f"**Sequence Coherence:** {overall_metrics.get('sequence_coherence', 0):.3f}\n\n"
        
        # Step-by-step summary
        step_evaluations = results.get("step_evaluations", [])
        if step_evaluations:
            summary += "## Step-by-Step Analysis\n\n"
            for i, step_eval in enumerate(step_evaluations, 1):
                content_score = step_eval.get("content_metrics", {}).get("content_quality_score", 0)
                engagement_score = step_eval.get("engagement_metrics", {}).get("engagement_score", 0)
                summary += f"**Step {i}:** Content: {content_score:.3f}, Engagement: {engagement_score:.3f}\n\n"
        
        return summary
    
    def _create_evaluation_charts(self, results: Dict) -> go.Figure:
        """Create evaluation charts using Plotly"""
        
        if "error" in results:
            return go.Figure()
        
        # Create subplots
        fig = go.Figure()
        
        # Overall metrics radar chart
        overall_metrics = results.get("overall_metrics", {})
        metrics_names = ["Content Quality", "Engagement", "Business Impact", "Coherence"]
        metrics_values = [
            overall_metrics.get("content_quality_score", 0),
            overall_metrics.get("engagement_score", 0),
            overall_metrics.get("business_impact_score", 0),
            overall_metrics.get("sequence_coherence", 0)
        ]
        
        fig.add_trace(go.Scatterpolar(
            r=metrics_values,
            theta=metrics_names,
            fill='toself',
            name='Overall Metrics'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Evaluation Metrics Overview"
        )
        
        return fig
    
    def training_interface(self) -> Any:
        """Create the training interface"""
        
        with gr.Tab("Model Training"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("## Training Configuration")
                    
                    # Early stopping configuration
                    patience = gr.Slider(
                        minimum=1,
                        maximum=50,
                        value=10,
                        step=1,
                        label="Early Stopping Patience"
                    )
                    
                    min_delta = gr.Number(
                        value=0.001,
                        label="Minimum Delta"
                    )
                    
                    # Learning rate scheduler configuration
                    scheduler_type = gr.Dropdown(
                        choices=["cosine", "step", "exponential", "plateau", "onecycle"],
                        value="cosine",
                        label="Learning Rate Scheduler"
                    )
                    
                    initial_lr = gr.Number(
                        value=0.001,
                        label="Initial Learning Rate"
                    )
                    
                    # Gradient management configuration
                    max_grad_norm = gr.Number(
                        value=1.0,
                        label="Max Gradient Norm"
                    )
                    
                    enable_gradient_clipping = gr.Checkbox(
                        value=True,
                        label="Enable Gradient Clipping"
                    )
                    
                    enable_nan_inf_check = gr.Checkbox(
                        value=True,
                        label="Enable NaN/Inf Check"
                    )
                    
                    # Training parameters
                    max_epochs = gr.Slider(
                        minimum=10,
                        maximum=500,
                        value=100,
                        step=10,
                        label="Max Epochs"
                    )
                    
                    batch_size = gr.Slider(
                        minimum=8,
                        maximum=128,
                        value=32,
                        step=8,
                        label="Batch Size"
                    )
                    
                    train_btn = gr.Button("Start Training", variant="primary")
                
                with gr.Column(scale=2):
                    gr.Markdown("## Training Progress")
                    
                    training_progress = gr.Textbox(
                        label="Training Log",
                        lines=10,
                        interactive=False
                    )
                    
                    training_metrics = gr.JSON(
                        label="Training Metrics"
                    )
                    
                    training_charts = gr.Plot(
                        label="Training Charts"
                    )
            
            # Connect the train button
            train_btn.click(
                fn=self.start_training,
                inputs=[
                    patience,
                    min_delta,
                    scheduler_type,
                    initial_lr,
                    max_grad_norm,
                    enable_gradient_clipping,
                    enable_nan_inf_check,
                    max_epochs,
                    batch_size
                ],
                outputs=[training_progress, training_metrics, training_charts]
            )
    
    def start_training(
        self,
        patience: int,
        min_delta: float,
        scheduler_type: str,
        initial_lr: float,
        max_grad_norm: float,
        enable_gradient_clipping: bool,
        enable_nan_inf_check: bool,
        max_epochs: int,
        batch_size: int
    ) -> Tuple[str, Dict, go.Figure]:
        """Start model training"""
        
        try:
            # Create configurations
            early_stopping_config = EarlyStoppingConfig(
                patience=patience,
                min_delta=min_delta
            )
            
            lr_scheduler_config = LRSchedulerConfig(
                scheduler_type=scheduler_type,
                initial_lr=initial_lr
            )
            
            gradient_config = GradientManagementConfig(
                enable_gradient_management=True,
                max_grad_norm=max_grad_norm,
                enable_gradient_clipping=enable_gradient_clipping,
                enable_nan_inf_check=enable_nan_inf_check
            )
            
            # Create dummy optimizer for demonstration
            dummy_model = torch.nn.Linear(10, 1)
            optimizer = torch.optim.Adam(dummy_model.parameters(), lr=initial_lr)
            
            # Initialize training optimizer
            self.training_optimizer = TrainingOptimizer(
                early_stopping_config=early_stopping_config,
                lr_scheduler_config=lr_scheduler_config,
                gradient_config=gradient_config,
                optimizer=optimizer
            )
            
            # Simulate training (in real implementation, this would train actual model)
            training_log = self._simulate_training(max_epochs, batch_size)
            
            # Create training metrics
            metrics = {
                "final_loss": 0.123,
                "best_loss": 0.098,
                "epochs_trained": max_epochs,
                "early_stopping_triggered": False,
                "final_learning_rate": initial_lr * 0.5
            }
            
            # Create training charts
            charts = self._create_training_charts(max_epochs)
            
            return training_log, metrics, charts
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            return f"Error: {str(e)}", {"error": str(e)}, go.Figure()
    
    def _simulate_training(self, max_epochs: int, batch_size: int) -> str:
        """Simulate training for demonstration"""
        
        log = f"Starting training with {max_epochs} epochs, batch size {batch_size}\n\n"
        
        for epoch in range(max_epochs):
            train_loss = 0.5 * np.exp(-epoch / 20) + 0.1 + np.random.normal(0, 0.02)
            val_loss = 0.6 * np.exp(-epoch / 25) + 0.12 + np.random.normal(0, 0.03)
            
            log += f"Epoch {epoch+1}/{max_epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}\n"
            
            if epoch % 10 == 0:
                log += f"  - Learning rate: {0.001 * (0.95 ** (epoch // 10)):.6f}\n"
                log += f"  - Gradient norm: {0.8 + np.random.normal(0, 0.1):.3f}\n"
        
        log += "\nTraining completed successfully!"
        return log
    
    def _create_training_charts(self, max_epochs: int) -> go.Figure:
        """Create training charts using Plotly"""
        
        epochs = list(range(1, max_epochs + 1))
        train_losses = [0.5 * np.exp(-e / 20) + 0.1 + np.random.normal(0, 0.02) for e in range(max_epochs)]
        val_losses = [0.6 * np.exp(-e / 25) + 0.12 + np.random.normal(0, 0.03) for e in range(max_epochs)]
        learning_rates = [0.001 * (0.95 ** (e // 10)) for e in range(max_epochs)]
        
        fig = go.Figure()
        
        # Add training and validation loss
        fig.add_trace(go.Scatter(
            x=epochs,
            y=train_losses,
            mode='lines',
            name='Training Loss',
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            x=epochs,
            y=val_losses,
            mode='lines',
            name='Validation Loss',
            line=dict(color='red')
        ))
        
        # Add learning rate on secondary y-axis
        fig.add_trace(go.Scatter(
            x=epochs,
            y=learning_rates,
            mode='lines',
            name='Learning Rate',
            yaxis='y2',
            line=dict(color='green', dash='dash')
        ))
        
        fig.update_layout(
            title="Training Progress",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            yaxis2=dict(
                title="Learning Rate",
                overlaying='y',
                side='right'
            ),
            hovermode='x unified'
        )
        
        return fig
    
    def gradient_management_interface(self) -> Any:
        """Create the gradient management interface"""
        
        with gr.Tab("Gradient Management"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("## Gradient Management Configuration")
                    
                    # Gradient clipping
                    max_grad_norm = gr.Number(
                        value=1.0,
                        label="Max Gradient Norm"
                    )
                    
                    clip_type = gr.Dropdown(
                        choices=["norm", "value", "adaptive"],
                        value="norm",
                        label="Clipping Type"
                    )
                    
                    # NaN/Inf handling
                    enable_nan_inf_check = gr.Checkbox(
                        value=True,
                        label="Enable NaN/Inf Check"
                    )
                    
                    replace_nan_with = gr.Number(
                        value=0.0,
                        label="Replace NaN With"
                    )
                    
                    replace_inf_with = gr.Number(
                        value=1e6,
                        label="Replace Inf With"
                    )
                    
                    # Monitoring
                    enable_monitoring = gr.Checkbox(
                        value=True,
                        label="Enable Gradient Monitoring"
                    )
                    
                    verbose_logging = gr.Checkbox(
                        value=False,
                        label="Verbose Logging"
                    )
                    
                    # Adaptive clipping
                    adaptive_clipping = gr.Checkbox(
                        value=False,
                        label="Adaptive Clipping"
                    )
                    
                    adaptive_window_size = gr.Slider(
                        minimum=10,
                        maximum=200,
                        value=100,
                        step=10,
                        label="Adaptive Window Size"
                    )
                    
                    test_gradient_btn = gr.Button("Test Gradient Management", variant="primary")
                
                with gr.Column(scale=2):
                    gr.Markdown("## Gradient Management Results")
                    
                    gradient_log = gr.Textbox(
                        label="Gradient Management Log",
                        lines=10,
                        interactive=False
                    )
                    
                    gradient_metrics = gr.JSON(
                        label="Gradient Metrics"
                    )
                    
                    gradient_charts = gr.Plot(
                        label="Gradient Charts"
                    )
            
            # Connect the test button
            test_gradient_btn.click(
                fn=self.test_gradient_management,
                inputs=[
                    max_grad_norm,
                    clip_type,
                    enable_nan_inf_check,
                    replace_nan_with,
                    replace_inf_with,
                    enable_monitoring,
                    verbose_logging,
                    adaptive_clipping,
                    adaptive_window_size
                ],
                outputs=[gradient_log, gradient_metrics, gradient_charts]
            )
    
    def test_gradient_management(
        self,
        max_grad_norm: float,
        clip_type: str,
        enable_nan_inf_check: bool,
        replace_nan_with: float,
        replace_inf_with: float,
        enable_monitoring: bool,
        verbose_logging: bool,
        adaptive_clipping: bool,
        adaptive_window_size: int
    ) -> Tuple[str, Dict, go.Figure]:
        """Test gradient management functionality"""
        
        try:
            # Create gradient configuration
            config = GradientConfig(
                max_grad_norm=max_grad_norm,
                clip_type=clip_type,
                enable_nan_inf_check=enable_nan_inf_check,
                replace_nan_with=replace_nan_with,
                replace_inf_with=replace_inf_with,
                enable_gradient_monitoring=enable_monitoring,
                verbose_logging=verbose_logging,
                adaptive_clipping=adaptive_clipping,
                adaptive_window_size=adaptive_window_size
            )
            
            # Initialize gradient manager
            self.gradient_manager = GradientManager(config)
            
            # Create test model
            model = torch.nn.Sequential(
                torch.nn.Linear(10, 20),
                torch.nn.ReLU(),
                torch.nn.Linear(20, 1)
            )
            
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            
            # Simulate training with gradient management
            log = f"Testing gradient management with {clip_type} clipping\n"
            log += f"Max gradient norm: {max_grad_norm}\n"
            log += f"NaN/Inf check: {enable_nan_inf_check}\n\n"
            
            metrics_history = []
            
            for step in range(50):
                optimizer.zero_grad()
                
                # Generate some data
                x = torch.randn(32, 10)
                y = torch.randn(32, 1)
                
                # Forward pass
                output = model(x)
                loss = torch.nn.MSELoss()(output, y)
                
                # Apply gradient management
                step_info = self.gradient_manager.step(
                    model=model,
                    optimizer=optimizer,
                    loss=loss
                )
                
                if step % 10 == 0:
                    log += f"Step {step}: Loss={loss.item():.4f}, "
                    log += f"GradNorm={step_info['statistics']['total_norm']:.4f}, "
                    log += f"Clipped={step_info['clipping']['clipped']}\n"
                
                metrics_history.append({
                    "step": step,
                    "loss": loss.item(),
                    "grad_norm": step_info['statistics']['total_norm'],
                    "clipped": step_info['clipping']['clipped'],
                    "healthy": step_info['health']['healthy']
                })
                
                optimizer.step()
            
            # Get final summary
            summary = self.gradient_manager.get_training_summary()
            log += f"\nTraining Summary:\n"
            log += f"Total steps: {summary['total_steps']}\n"
            log += f"Health issues: {summary['health_issues']['unhealthy_steps']}\n"
            log += f"NaN/Inf replacements: {summary['nan_inf_summary']['total_replacements']}\n"
            
            # Create charts
            charts = self._create_gradient_charts(metrics_history)
            
            return log, summary, charts
            
        except Exception as e:
            logger.error(f"Error testing gradient management: {e}")
            return f"Error: {str(e)}", {"error": str(e)}, go.Figure()
    
    def _create_gradient_charts(self, metrics_history: List[Dict]) -> go.Figure:
        """Create gradient management charts"""
        
        steps = [m["step"] for m in metrics_history]
        losses = [m["loss"] for m in metrics_history]
        grad_norms = [m["grad_norm"] for m in metrics_history]
        health_status = [1 if m["healthy"] else 0 for m in metrics_history]
        
        fig = go.Figure()
        
        # Add loss
        fig.add_trace(go.Scatter(
            x=steps,
            y=losses,
            mode='lines',
            name='Loss',
            line=dict(color='blue')
        ))
        
        # Add gradient norms
        fig.add_trace(go.Scatter(
            x=steps,
            y=grad_norms,
            mode='lines',
            name='Gradient Norm',
            yaxis='y2',
            line=dict(color='red')
        ))
        
        # Add health status
        fig.add_trace(go.Scatter(
            x=steps,
            y=health_status,
            mode='lines',
            name='Gradient Health',
            yaxis='y3',
            line=dict(color='green')
        ))
        
        fig.update_layout(
            title="Gradient Management Results",
            xaxis_title="Step",
            yaxis_title="Loss",
            yaxis2=dict(
                title="Gradient Norm",
                overlaying='y',
                side='right'
            ),
            yaxis3=dict(
                title="Health Status",
                overlaying='y',
                side='right',
                position=0.95
            ),
            hovermode='x unified'
        )
        
        return fig
    
    def create_app(self) -> gr.Blocks:
        """Create the complete Gradio application"""
        
        with gr.Blocks(
            title="Email Sequence AI System",
            theme=gr.themes.Soft(),
            css="""
            .gradio-container {
                max-width: 1200px !important;
            }
            """
        ) as app:
            
            gr.Markdown("""
            # 📧 Email Sequence AI System
            
            A comprehensive AI-powered system for generating, evaluating, and optimizing email sequences.
            
            ---
            """)
            
            # Create all interfaces
            self.generate_sequence_interface()
            self.evaluation_interface()
            self.training_interface()
            self.gradient_management_interface()
            
            # Footer
            gr.Markdown("""
            ---
            
            **Email Sequence AI System** - Powered by advanced AI models and comprehensive evaluation metrics.
            
            Built with Gradio, PyTorch, and modern machine learning techniques.
            """)
        
        return app


def main():
    """Main function to launch the Gradio app"""
    
    # Create the application
    app_instance = GradioEmailSequenceApp()
    app = app_instance.create_app()
    
    # Launch the app
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        debug=True,
        show_error=True
    )


match __name__:
    case "__main__":
    main() 