from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
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
import time
import random
import sys
from core.error_handling import (
from core.sequence_generator import EmailSequenceGenerator, GeneratorConfig
from core.evaluation_metrics import EmailSequenceEvaluator, MetricsConfig
from core.training_optimization import (
from core.gradient_management import GradientManager, GradientConfig
from models.sequence import EmailSequence, SequenceStep
from models.subscriber import Subscriber
from models.template import EmailTemplate
from typing import Any, List, Dict, Optional
"""
Enhanced Gradio Web Interface with Error Handling

A comprehensive web interface for the email sequence AI system with
robust error handling, input validation, and debugging capabilities.
"""


# Add the parent directory to the path to import modules
sys.path.append(str(Path(__file__).parent))

    ErrorHandler, InputValidator, DataLoaderErrorHandler, 
    ModelInferenceErrorHandler, GradioErrorHandler,
    ValidationError, ModelError, DataError, ConfigurationError
)
    TrainingOptimizer,
    EarlyStoppingConfig,
    LRSchedulerConfig,
    GradientManagementConfig
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedGradioEmailSequenceApp:
    """Enhanced Gradio application with comprehensive error handling"""
    
    def __init__(self, debug_mode: bool = False):
        
    """__init__ function."""
# Initialize error handling system
        self.error_handler = ErrorHandler(debug_mode=debug_mode)
        self.validator = InputValidator()
        self.data_handler = DataLoaderErrorHandler(self.error_handler)
        self.model_handler = ModelInferenceErrorHandler(self.error_handler)
        self.gradio_handler = GradioErrorHandler(self.error_handler, debug_mode=debug_mode)
        
        # Initialize system components
        self.sequence_generator = None
        self.evaluator = None
        self.training_optimizer = None
        self.gradient_manager = None
        
        # Sample data for demonstration
        self.sample_subscribers = self._create_sample_subscribers()
        self.sample_templates = self._create_sample_templates()
        
        # Global state
        self.current_sequence = None
        self.current_evaluator = None
        self.current_training_optimizer = None
        self.training_history = []
        
        logger.info("Enhanced Gradio Email Sequence App initialized")
    
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
    
    def validate_sequence_generation_inputs(
        self,
        model_type: str,
        sequence_length: int,
        creativity_level: float,
        target_audience: str,
        industry_focus: str
    ) -> Tuple[bool, List[str]]:
        """Validate inputs for sequence generation"""
        
        errors = []
        
        # Validate model type
        is_valid, error_msg = self.validator.validate_model_type(model_type)
        if not is_valid:
            errors.append(error_msg)
        
        # Validate sequence length
        is_valid, error_msg = self.validator.validate_sequence_length(sequence_length)
        if not is_valid:
            errors.append(error_msg)
        
        # Validate creativity level
        is_valid, error_msg = self.validator.validate_creativity_level(creativity_level)
        if not is_valid:
            errors.append(error_msg)
        
        # Validate target audience
        if not target_audience or target_audience.strip() == "":
            errors.append("Target audience must be selected")
        
        # Validate industry focus
        if not industry_focus or industry_focus.strip() == "":
            errors.append("Industry focus cannot be empty")
        
        return len(errors) == 0, errors
    
    def generate_sequence_interface(self) -> Any:
        """Create the sequence generation interface with error handling"""
        
        with gr.Tab("Sequence Generation"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("## Sequence Generation Configuration")
                    
                    # Generator configuration
                    model_type = gr.Dropdown(
                        choices=["GPT-3.5", "GPT-4", "Claude", "Custom"],
                        value="GPT-3.5",
                        label="AI Model",
                        info="Choose the AI model for sequence generation"
                    )
                    
                    sequence_length = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=3,
                        step=1,
                        label="Sequence Length",
                        info="Number of emails in the sequence"
                    )
                    
                    creativity_level = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.7,
                        step=0.1,
                        label="Creativity Level",
                        info="Controls AI creativity and originality"
                    )
                    
                    target_audience = gr.Dropdown(
                        choices=[f"{s.name} ({s.company})" for s in self.sample_subscribers],
                        value=f"{self.sample_subscribers[0].name} ({self.sample_subscribers[0].company})",
                        label="Target Audience",
                        info="Select the target audience for the sequence"
                    )
                    
                    industry_focus = gr.Textbox(
                        value="Technology",
                        label="Industry Focus",
                        info="Target industry for customization"
                    )
                    
                    generate_btn = gr.Button("Generate Sequence", variant="primary")
                    
                    # Error display
                    error_display = gr.Markdown(
                        label="Error Messages",
                        visible=False
                    )
                
                with gr.Column(scale=2):
                    gr.Markdown("## Generated Sequence")
                    
                    sequence_output = gr.JSON(
                        label="Sequence JSON",
                        visible=True
                    )
                    
                    sequence_preview = gr.Markdown(
                        label="Sequence Preview"
                    )
                    
                    # Generation metrics
                    generation_metrics = gr.JSON(
                        label="Generation Metrics"
                    )
            
            # Connect the generate button with error handling
            generate_btn.click(
                fn=self.safe_generate_sequence,
                inputs=[
                    model_type,
                    sequence_length,
                    creativity_level,
                    target_audience,
                    industry_focus
                ],
                outputs=[sequence_output, sequence_preview, generation_metrics, error_display]
            )
    
    def safe_generate_sequence(
        self,
        model_type: str,
        sequence_length: int,
        creativity_level: float,
        target_audience: str,
        industry_focus: str
    ) -> Tuple[Dict, str, Dict, str]:
        """Safely generate email sequence with comprehensive error handling"""
        
        try:
            # Validate inputs
            is_valid, errors = self.validate_sequence_generation_inputs(
                model_type, sequence_length, creativity_level, target_audience, industry_focus
            )
            
            if not is_valid:
                error_message = "**Input Validation Errors:**\n" + "\n".join([f"- {error}" for error in errors])
                return (
                    {"error": "Validation failed", "details": errors},
                    "Please fix the input errors above.",
                    {"error": True},
                    error_message
                )
            
            # Parse target audience
            try:
                subscriber_name = target_audience.split(" (")[0]
                subscriber = next(s for s in self.sample_subscribers if s.name == subscriber_name)
            except (ValueError, StopIteration) as e:
                self.error_handler.log_error(e, "Parsing target audience", "safe_generate_sequence")
                return (
                    {"error": "Invalid target audience"},
                    "Error: Could not parse target audience selection.",
                    {"error": True},
                    "**Error:** Invalid target audience selection. Please try again."
                )
            
            # Create generator configuration
            try:
                config = GeneratorConfig(
                    model_type=model_type,
                    sequence_length=sequence_length,
                    creativity_level=creativity_level,
                    industry_focus=industry_focus
                )
            except Exception as e:
                self.error_handler.log_error(e, "Creating generator config", "safe_generate_sequence")
                return (
                    {"error": "Configuration error", "details": str(e)},
                    "Error: Failed to create generator configuration.",
                    {"error": True},
                    f"**Configuration Error:** {str(e)}"
                )
            
            # Initialize generator
            try:
                self.sequence_generator = EmailSequenceGenerator(config)
            except Exception as e:
                self.error_handler.log_error(e, "Initializing sequence generator", "safe_generate_sequence")
                return (
                    {"error": "Generator initialization failed", "details": str(e)},
                    "Error: Failed to initialize sequence generator.",
                    {"error": True},
                    f"**Initialization Error:** {str(e)}"
                )
            
            # Generate sequence
            try:
                sequence = asyncio.run(self.sequence_generator.generate_sequence(
                    target_audience=subscriber,
                    templates=self.sample_templates
                ))
            except Exception as e:
                self.error_handler.log_error(e, "Generating sequence", "safe_generate_sequence")
                return (
                    {"error": "Generation failed", "details": str(e)},
                    "Error: Failed to generate sequence.",
                    {"error": True},
                    f"**Generation Error:** {str(e)}"
                )
            
            # Convert to JSON-serializable format
            try:
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
            except Exception as e:
                self.error_handler.log_error(e, "Converting sequence to JSON", "safe_generate_sequence")
                return (
                    {"error": "Data conversion failed", "details": str(e)},
                    "Error: Failed to convert sequence data.",
                    {"error": True},
                    f"**Conversion Error:** {str(e)}"
                )
            
            # Create preview
            try:
                preview = self._create_sequence_preview(sequence)
            except Exception as e:
                self.error_handler.log_error(e, "Creating sequence preview", "safe_generate_sequence")
                preview = "Error: Could not create sequence preview."
            
            # Create metrics
            try:
                metrics = {
                    "model_used": model_type,
                    "sequence_length": len(sequence.steps),
                    "generation_time": datetime.now().isoformat(),
                    "creativity_level": creativity_level,
                    "industry_focus": industry_focus,
                    "target_audience": subscriber.name
                }
            except Exception as e:
                self.error_handler.log_error(e, "Creating metrics", "safe_generate_sequence")
                metrics = {"error": "Could not create metrics"}
            
            # Store current sequence
            self.current_sequence = sequence
            
            return sequence_dict, preview, metrics, ""
            
        except Exception as e:
            self.error_handler.log_error(e, "Unexpected error in sequence generation", "safe_generate_sequence")
            return (
                {"error": "Unexpected error", "details": str(e)},
                "An unexpected error occurred. Please try again.",
                {"error": True},
                f"**Unexpected Error:** {str(e)}"
            )
    
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
        """Create the evaluation interface with error handling"""
        
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
                    
                    # Error display
                    eval_error_display = gr.Markdown(
                        label="Evaluation Errors",
                        visible=False
                    )
                
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
                fn=self.safe_evaluate_sequence,
                inputs=[
                    enable_content_quality,
                    enable_engagement,
                    enable_business_impact,
                    enable_technical,
                    content_weight,
                    engagement_weight
                ],
                outputs=[evaluation_results, evaluation_summary, evaluation_charts, eval_error_display]
            )
    
    def safe_evaluate_sequence(
        self,
        enable_content_quality: bool,
        enable_engagement: bool,
        enable_business_impact: bool,
        enable_technical: bool,
        content_weight: float,
        engagement_weight: float
    ) -> Tuple[Dict, str, go.Figure, str]:
        """Safely evaluate sequence with error handling"""
        
        try:
            # Check if sequence exists
            if self.current_sequence is None:
                return (
                    {"error": "No sequence to evaluate"},
                    "Please generate a sequence first.",
                    go.Figure(),
                    "**Error:** No sequence available for evaluation. Please generate a sequence first."
                )
            
            # Validate weights
            if content_weight < 0 or content_weight > 1:
                return (
                    {"error": "Invalid content weight"},
                    "Content weight must be between 0 and 1.",
                    go.Figure(),
                    "**Validation Error:** Content weight must be between 0 and 1."
                )
            
            if engagement_weight < 0 or engagement_weight > 1:
                return (
                    {"error": "Invalid engagement weight"},
                    "Engagement weight must be between 0 and 1.",
                    go.Figure(),
                    "**Validation Error:** Engagement weight must be between 0 and 1."
                )
            
            # Create evaluation configuration
            try:
                config = MetricsConfig(
                    enable_content_quality=enable_content_quality,
                    enable_engagement=enable_engagement,
                    enable_business_impact=enable_business_impact,
                    enable_technical=enable_technical,
                    content_weight=content_weight,
                    engagement_weight=engagement_weight
                )
            except Exception as e:
                self.error_handler.log_error(e, "Creating evaluation config", "safe_evaluate_sequence")
                return (
                    {"error": "Configuration error", "details": str(e)},
                    "Error: Failed to create evaluation configuration.",
                    go.Figure(),
                    f"**Configuration Error:** {str(e)}"
                )
            
            # Initialize evaluator
            try:
                self.evaluator = EmailSequenceEvaluator(config)
            except Exception as e:
                self.error_handler.log_error(e, "Initializing evaluator", "safe_evaluate_sequence")
                return (
                    {"error": "Evaluator initialization failed", "details": str(e)},
                    "Error: Failed to initialize evaluator.",
                    go.Figure(),
                    f"**Initialization Error:** {str(e)}"
                )
            
            # Evaluate sequence
            try:
                results = asyncio.run(self.evaluator.evaluate_sequence(
                    sequence=self.current_sequence,
                    subscribers=self.sample_subscribers,
                    templates=self.sample_templates
                ))
            except Exception as e:
                self.error_handler.log_error(e, "Evaluating sequence", "safe_evaluate_sequence")
                return (
                    {"error": "Evaluation failed", "details": str(e)},
                    "Error: Failed to evaluate sequence.",
                    go.Figure(),
                    f"**Evaluation Error:** {str(e)}"
                )
            
            # Create summary
            try:
                summary = self._create_evaluation_summary(results)
            except Exception as e:
                self.error_handler.log_error(e, "Creating evaluation summary", "safe_evaluate_sequence")
                summary = "Error: Could not create evaluation summary."
            
            # Create charts
            try:
                charts = self._create_evaluation_charts(results)
            except Exception as e:
                self.error_handler.log_error(e, "Creating evaluation charts", "safe_evaluate_sequence")
                charts = go.Figure()
            
            return results, summary, charts, ""
            
        except Exception as e:
            self.error_handler.log_error(e, "Unexpected error in evaluation", "safe_evaluate_sequence")
            return (
                {"error": "Unexpected error", "details": str(e)},
                "An unexpected error occurred during evaluation.",
                go.Figure(),
                f"**Unexpected Error:** {str(e)}"
            )
    
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
        
        return summary
    
    def _create_evaluation_charts(self, results: Dict) -> go.Figure:
        """Create evaluation charts using Plotly"""
        
        if "error" in results:
            return go.Figure()
        
        # Create a simple radar chart
        overall_metrics = results.get("overall_metrics", {})
        metrics_names = ["Content Quality", "Engagement", "Business Impact", "Coherence"]
        metrics_values = [
            overall_metrics.get("content_quality_score", 0),
            overall_metrics.get("engagement_score", 0),
            overall_metrics.get("business_impact_score", 0),
            overall_metrics.get("sequence_coherence", 0)
        ]
        
        fig = go.Figure()
        
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
    
    def create_app(self) -> gr.Blocks:
        """Create the complete enhanced Gradio application"""
        
        with gr.Blocks(
            title="Email Sequence AI System - Enhanced",
            theme=gr.themes.Soft(),
            css="""
            .gradio-container {
                max-width: 1200px !important;
            }
            .error-message {
                background-color: #fee;
                border: 1px solid #fcc;
                border-radius: 4px;
                padding: 10px;
                margin: 10px 0;
                color: #c33;
            }
            .success-message {
                background-color: #efe;
                border: 1px solid #cfc;
                border-radius: 4px;
                padding: 10px;
                margin: 10px 0;
                color: #3c3;
            }
            """
        ) as app:
            
            gr.Markdown("""
            # 📧 Email Sequence AI System - Enhanced
            
            A comprehensive AI-powered system for generating, evaluating, and optimizing email sequences.
            This enhanced version includes robust error handling and input validation.
            
            ---
            """)
            
            # Create all interfaces
            self.generate_sequence_interface()
            self.evaluation_interface()
            
            # Error monitoring section
            with gr.Accordion("🔧 System Monitoring", open=False):
                gr.Markdown("### Error Summary")
                error_summary_btn = gr.Button("Get Error Summary")
                error_summary_display = gr.JSON(label="Error Summary")
                
                def get_error_summary():
                    
    """get_error_summary function."""
return self.error_handler.get_error_summary()
                
                error_summary_btn.click(
                    fn=get_error_summary,
                    outputs=[error_summary_display]
                )
            
            # Footer
            gr.Markdown("""
            ---
            
            **Email Sequence AI System - Enhanced** - Powered by advanced AI models and comprehensive error handling.
            
            Built with Gradio, PyTorch, and modern machine learning techniques.
            """)
        
        return app


def main():
    """Main function to launch the enhanced Gradio app"""
    
    # Create the enhanced application
    app_instance = EnhancedGradioEmailSequenceApp(debug_mode=True)
    app = app_instance.create_app()
    
    # Launch the app
    app.launch(
        server_name="0.0.0.0",
        server_port=7864,
        share=True,
        debug=True,
        show_error=True
    )


match __name__:
    case "__main__":
    main() 