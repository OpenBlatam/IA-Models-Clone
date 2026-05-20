from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import gradio as gr
import asyncio
import logging
import json
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import random
import time
import sys
from core.sequence_generator import EmailSequenceGenerator, GeneratorConfig
from core.evaluation_metrics import EmailSequenceEvaluator, MetricsConfig
from models.sequence import EmailSequence, SequenceStep
from models.subscriber import Subscriber
from models.template import EmailTemplate
from typing import Any, List, Dict, Optional
"""
Interactive Demos for Email Sequence AI System

A collection of interactive Gradio demos showcasing model inference,
visualization, and real-time analysis capabilities.
"""


# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent.parent))


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InteractiveDemos:
    """Interactive demos for email sequence system"""
    
    def __init__(self) -> Any:
        self.sequence_generator = None
        self.evaluator = None
        self.demo_data = self._create_demo_data()
        self.demo_history = []
        
        logger.info("Interactive Demos initialized")
    
    def _create_demo_data(self) -> Dict:
        """Create comprehensive demo data"""
        
        # Sample subscribers with diverse profiles
        subscribers = [
            Subscriber(
                id="tech_leader",
                email="sarah@techstartup.com",
                name="Sarah Chen",
                company="TechStartup Inc",
                interests=["AI", "startup growth", "product management"],
                industry="Technology"
            ),
            Subscriber(
                id="marketing_manager",
                email="mike@marketingpro.com",
                name="Mike Rodriguez",
                company="MarketingPro Solutions",
                interests=["digital marketing", "lead generation", "analytics"],
                industry="Marketing"
            ),
            Subscriber(
                id="finance_director",
                email="emma@financeltd.com",
                name="Emma Thompson",
                company="Finance Ltd",
                interests=["investment", "financial planning", "risk management"],
                industry="Finance"
            ),
            Subscriber(
                id="healthcare_admin",
                email="dr.james@healthcare.org",
                name="Dr. James Wilson",
                company="Healthcare Solutions",
                interests=["healthcare technology", "patient care", "medical innovation"],
                industry="Healthcare"
            ),
            Subscriber(
                id="education_coordinator",
                email="lisa@edutech.edu",
                name="Lisa Park",
                company="EduTech University",
                interests=["online learning", "educational technology", "student engagement"],
                industry="Education"
            )
        ]
        
        # Sample email templates
        templates = [
            EmailTemplate(
                id="welcome_series",
                name="Welcome Series",
                subject_template="Welcome to {company}! Let's get you started",
                content_template="Hi {name}, welcome to our platform! We're excited to help you succeed with {interest}. Here's what you can expect..."
            ),
            EmailTemplate(
                id="feature_intro",
                name="Feature Introduction",
                subject_template="Discover our powerful {feature} feature",
                content_template="Hello {name}, we think you'll love our new {feature} feature. It's perfect for {industry} professionals like you..."
            ),
            EmailTemplate(
                id="case_study",
                name="Case Study",
                subject_template="How {company_name} achieved {result} with our platform",
                content_template="Hi {name}, we wanted to share an inspiring story about how {company_name} achieved {result} using our platform..."
            ),
            EmailTemplate(
                id="conversion",
                name="Conversion",
                subject_template="Ready to take the next step?",
                content_template="Hi {name}, based on your interest in {interest}, we think you're ready to take the next step. Here's how to get started..."
            ),
            EmailTemplate(
                id="re_engagement",
                name="Re-engagement",
                subject_template="We miss you! Here's what's new",
                content_template="Hi {name}, we noticed you haven't been active lately. We've added some exciting new features that might interest you..."
            )
        ]
        
        return {
            "subscribers": subscribers,
            "templates": templates,
            "industries": ["Technology", "Marketing", "Finance", "Healthcare", "Education", "Retail", "Manufacturing"],
            "interests": ["AI", "marketing", "finance", "healthcare", "education", "sales", "productivity", "innovation"]
        }
    
    def create_live_inference_demo(self) -> gr.Blocks:
        """Create live model inference demo"""
        
        with gr.Blocks(title="Live Model Inference Demo", theme=gr.themes.Soft()) as demo:
            
            gr.Markdown("""
            # 🤖 Live Model Inference Demo
            
            Experience real-time email sequence generation with different AI models.
            Watch as the system generates personalized sequences based on your inputs.
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Configuration")
                    
                    # Model selection
                    model_choice = gr.Dropdown(
                        choices=["GPT-3.5", "GPT-4", "Claude", "Custom Model"],
                        value="GPT-3.5",
                        label="AI Model",
                        info="Choose the AI model for sequence generation"
                    )
                    
                    # Subscriber selection
                    subscriber_choice = gr.Dropdown(
                        choices=[f"{s.name} ({s.company})" for s in self.demo_data["subscribers"]],
                        value=f"{self.demo_data['subscribers'][0].name} ({self.demo_data['subscribers'][0].company})",
                        label="Target Subscriber",
                        info="Select the target audience for the sequence"
                    )
                    
                    # Industry focus
                    industry_focus = gr.Dropdown(
                        choices=self.demo_data["industries"],
                        value="Technology",
                        label="Industry Focus",
                        info="Target industry for customization"
                    )
                    
                    # Sequence parameters
                    sequence_length = gr.Slider(
                        minimum=2,
                        maximum=8,
                        value=4,
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
                    
                    # Real-time generation
                    generate_btn = gr.Button("🚀 Generate Live Sequence", variant="primary", size="lg")
                    
                    # Progress indicator
                    progress = gr.Progress()
                
                with gr.Column(scale=2):
                    gr.Markdown("### Live Generation Results")
                    
                    # Real-time output
                    live_output = gr.JSON(
                        label="Generated Sequence (JSON)",
                        visible=True
                    )
                    
                    # Formatted preview
                    sequence_preview = gr.Markdown(
                        label="Sequence Preview"
                    )
                    
                    # Generation metrics
                    generation_metrics = gr.JSON(
                        label="Generation Metrics"
                    )
            
            # Live generation function
            def live_generate_sequence(
                model: str,
                subscriber: str,
                industry: str,
                length: int,
                creativity: float,
                progress: gr.Progress
            ) -> Tuple[Dict, str, Dict]:
                
                try:
                    # Parse subscriber
                    subscriber_name = subscriber.split(" (")[0]
                    target_subscriber = next(
                        s for s in self.demo_data["subscribers"] 
                        if s.name == subscriber_name
                    )
                    
                    # Simulate generation progress
                    progress(0.1, desc="Initializing model...")
                    time.sleep(0.5)
                    
                    progress(0.3, desc="Analyzing subscriber profile...")
                    time.sleep(0.8)
                    
                    progress(0.5, desc="Generating sequence content...")
                    time.sleep(1.2)
                    
                    progress(0.7, desc="Optimizing timing and flow...")
                    time.sleep(0.6)
                    
                    progress(0.9, desc="Finalizing sequence...")
                    time.sleep(0.4)
                    
                    # Generate demo sequence
                    sequence = self._generate_demo_sequence(
                        target_subscriber, industry, length, creativity, model
                    )
                    
                    progress(1.0, desc="Generation complete!")
                    
                    # Convert to JSON
                    sequence_dict = {
                        "id": sequence.id,
                        "name": sequence.name,
                        "description": sequence.description,
                        "model_used": model,
                        "generation_time": datetime.now().isoformat(),
                        "steps": [
                            {
                                "order": step.order,
                                "subscriber_id": step.subscriber_id,
                                "template_id": step.template_id,
                                "content": step.content,
                                "delay_hours": step.delay_hours,
                                "estimated_open_rate": round(random.uniform(0.15, 0.45), 3),
                                "estimated_click_rate": round(random.uniform(0.02, 0.08), 3)
                            }
                            for step in sequence.steps
                        ]
                    }
                    
                    # Create preview
                    preview = self._create_sequence_preview(sequence, model)
                    
                    # Generation metrics
                    metrics = {
                        "model": model,
                        "generation_time_seconds": round(random.uniform(2.5, 4.0), 2),
                        "sequence_length": len(sequence.steps),
                        "creativity_score": creativity,
                        "personalization_score": round(random.uniform(0.7, 0.95), 3),
                        "estimated_total_engagement": round(random.uniform(0.25, 0.65), 3)
                    }
                    
                    return sequence_dict, preview, metrics
                    
                except Exception as e:
                    logger.error(f"Error in live generation: {e}")
                    return {"error": str(e)}, f"Error: {str(e)}", {"error": str(e)}
            
            # Connect the generate button
            generate_btn.click(
                fn=live_generate_sequence,
                inputs=[
                    model_choice,
                    subscriber_choice,
                    industry_focus,
                    sequence_length,
                    creativity_level,
                    progress
                ],
                outputs=[live_output, sequence_preview, generation_metrics]
            )
        
        return demo
    
    def _generate_demo_sequence(
        self,
        subscriber: Subscriber,
        industry: str,
        length: int,
        creativity: float,
        model: str
    ) -> EmailSequence:
        """Generate a demo sequence for live inference"""
        
        # Create sequence steps based on parameters
        steps = []
        templates = self.demo_data["templates"f"]
        
        for i in range(length):
            template = templates[i % len(templates)]
            
            # Customize content based on subscriber and industry
            content = template.content_template"
            
            step = SequenceStep(
                order=i + 1,
                subscriber_id=subscriber.id,
                template_id=template.id,
                content=content,
                delay_hours=24 * (i + 1)  # 24 hours between emails
            )
            
            steps.append(step)
        
        sequence = EmailSequence(
            id=f"demo_seq_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            name=f"Demo Sequence for {subscriber.name}",
            description=f"AI-generated sequence for {subscriber.company} in {industry}",
            steps=steps
        )
        
        return sequence
    
    def _create_sequence_preview(self, sequence: EmailSequence, model: str) -> str:
        """Create a formatted preview of the sequence"""
        
        preview = f"# 📧 Generated Sequence\n\n"
        preview += f"**Model:** {model}\n"
        preview += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        preview += f"**Length:** {len(sequence.steps)} emails\n\n"
        preview += f"## Sequence Overview\n\n"
        preview += f"{sequence.description}\n\n"
        
        for i, step in enumerate(sequence.steps, 1):
            preview += f"### Email {i} (Day {i})\n\n"
            preview += f"**Content:**\n{step.content}\n\n"
            preview += f"**Timing:** {step.delay_hours} hours after previous\n\n"
            preview += "---\n\n"
        
        return preview
    
    def create_visualization_demo(self) -> gr.Blocks:
        """Create interactive visualization demo"""
        
        with gr.Blocks(title="Interactive Visualization Demo", theme=gr.themes.Soft()) as demo:
            
            gr.Markdown("""
            # 📊 Interactive Visualization Demo
            
            Explore email sequence performance through interactive charts and analytics.
            Visualize engagement patterns, conversion rates, and optimization opportunities.
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Visualization Controls")
                    
                    # Chart type selection
                    chart_type = gr.Dropdown(
                        choices=[
                            "Engagement Timeline",
                            "Performance Comparison",
                            "Audience Analysis",
                            "Conversion Funnel",
                            "Heatmap Analysis",
                            "Predictive Analytics"
                        ],
                        value="Engagement Timeline",
                        label="Chart Type",
                        info="Select the type of visualization to display"
                    )
                    
                    # Data source
                    data_source = gr.Dropdown(
                        choices=["Demo Data", "Generated Sequences", "Historical Data"],
                        value="Demo Data",
                        label="Data Source",
                        info="Choose the data source for visualization"
                    )
                    
                    # Time range
                    time_range = gr.Dropdown(
                        choices=["Last 7 days", "Last 30 days", "Last 90 days", "All time"],
                        value="Last 30 days",
                        label="Time Range",
                        info="Select the time period for analysis"
                    )
                    
                    # Update button
                    update_chart_btn = gr.Button("📈 Update Visualization", variant="primary")
                
                with gr.Column(scale=2):
                    gr.Markdown("### Interactive Charts")
                    
                    # Main chart
                    main_chart = gr.Plot(
                        label="Main Visualization"
                    )
                    
                    # Secondary chart
                    secondary_chart = gr.Plot(
                        label="Secondary Analysis"
                    )
                    
                    # Chart statistics
                    chart_stats = gr.JSON(
                        label="Chart Statistics"
                    )
            
            # Visualization function
            def update_visualization(
                chart_type: str,
                data_source: str,
                time_range: str
            ) -> Tuple[go.Figure, go.Figure, Dict]:
                
                try:
                    # Generate demo data based on parameters
                    demo_data = self._generate_demo_visualization_data(
                        chart_type, data_source, time_range
                    )
                    
                    # Create main chart
                    main_fig = self._create_main_chart(chart_type, demo_data)
                    
                    # Create secondary chart
                    secondary_fig = self._create_secondary_chart(chart_type, demo_data)
                    
                    # Calculate statistics
                    stats = self._calculate_chart_statistics(demo_data)
                    
                    return main_fig, secondary_fig, stats
                    
                except Exception as e:
                    logger.error(f"Error in visualization: {e}")
                    empty_fig = go.Figure()
                    return empty_fig, empty_fig, {"error": str(e)}
            
            # Connect the update button
            update_chart_btn.click(
                fn=update_visualization,
                inputs=[chart_type, data_source, time_range],
                outputs=[main_chart, secondary_chart, chart_stats]
            )
        
        return demo
    
    def _generate_demo_visualization_data(
        self,
        chart_type: str,
        data_source: str,
        time_range: str
    ) -> Dict:
        """Generate demo data for visualizations"""
        
        # Generate time series data
        days = 30 if "30" in time_range else (7 if "7" in time_range else 90)
        dates = [datetime.now() - timedelta(days=i) for i in range(days)]
        dates.reverse()
        
        # Generate engagement data
        engagement_data = {
            "dates": [d.strftime("%Y-%m-%d") for d in dates],
            "open_rates": [round(random.uniform(0.15, 0.45), 3) for _ in range(days)],
            "click_rates": [round(random.uniform(0.02, 0.08), 3) for _ in range(days)],
            "conversion_rates": [round(random.uniform(0.005, 0.025), 4) for _ in range(days)],
            "revenue": [round(random.uniform(100, 5000), 2) for _ in range(days)]
        }
        
        # Generate audience data
        audience_data = {
            "industries": self.demo_data["industries"],
            "industry_counts": [random.randint(50, 500) for _ in self.demo_data["industries"]],
            "engagement_by_industry": [round(random.uniform(0.2, 0.6), 3) for _ in self.demo_data["industries"]]
        }
        
        # Generate sequence performance data
        sequence_data = {
            "sequence_types": ["Welcome", "Onboarding", "Feature", "Conversion", "Re-engagement"],
            "performance_scores": [round(random.uniform(0.6, 0.9), 3) for _ in range(5)],
            "engagement_rates": [round(random.uniform(0.25, 0.65), 3) for _ in range(5)]
        }
        
        return {
            "engagement": engagement_data,
            "audience": audience_data,
            "sequences": sequence_data,
            "chart_type": chart_type,
            "data_source": data_source,
            "time_range": time_range
        }
    
    def _create_main_chart(self, chart_type: str, data: Dict) -> go.Figure:
        """Create the main visualization chart"""
        
        if chart_type == "Engagement Timeline":
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=data["engagement"]["dates"],
                y=data["engagement"]["open_rates"],
                mode='lines+markers',
                name='Open Rate',
                line=dict(color='blue', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=data["engagement"]["dates"],
                y=data["engagement"]["click_rates"],
                mode='lines+markers',
                name='Click Rate',
                line=dict(color='red', width=2)
            ))
            
            fig.update_layout(
                title="Email Engagement Timeline",
                xaxis_title="Date",
                yaxis_title="Rate",
                hovermode='x unified'
            )
            
        elif chart_type == "Performance Comparison":
            fig = go.Figure(data=[
                go.Bar(
                    x=data["sequences"]["sequence_types"],
                    y=data["sequences"]["performance_scores"],
                    name='Performance Score',
                    marker_color='lightblue'
                )
            ])
            
            fig.update_layout(
                title="Sequence Performance Comparison",
                xaxis_title="Sequence Type",
                yaxis_title="Performance Score",
                showlegend=True
            )
            
        elif chart_type == "Audience Analysis":
            fig = go.Figure(data=[
                go.Pie(
                    labels=data["audience"]["industries"],
                    values=data["audience"]["industry_counts"],
                    hole=0.3
                )
            ])
            
            fig.update_layout(
                title="Audience Distribution by Industry",
                showlegend=True
            )
            
        else:
            # Default chart
            fig = go.Figure()
            fig.add_annotation(
                text=f"Chart type: {chart_type}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        return fig
    
    def _create_secondary_chart(self, chart_type: str, data: Dict) -> go.Figure:
        """Create the secondary analysis chart"""
        
        if chart_type == "Engagement Timeline":
            # Revenue chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=data["engagement"]["dates"],
                y=data["engagement"]["revenue"],
                mode='lines+markers',
                name='Revenue',
                line=dict(color='green', width=2)
            ))
            
            fig.update_layout(
                title="Revenue Timeline",
                xaxis_title="Date",
                yaxis_title="Revenue ($)",
                hovermode='x unified'
            )
            
        elif chart_type == "Performance Comparison":
            # Engagement rates
            fig = go.Figure(data=[
                go.Bar(
                    x=data["sequences"]["sequence_types"],
                    y=data["sequences"]["engagement_rates"],
                    name='Engagement Rate',
                    marker_color='lightgreen'
                )
            ])
            
            fig.update_layout(
                title="Sequence Engagement Rates",
                xaxis_title="Sequence Type",
                yaxis_title="Engagement Rate",
                showlegend=True
            )
            
        elif chart_type == "Audience Analysis":
            # Engagement by industry
            fig = go.Figure(data=[
                go.Bar(
                    x=data["audience"]["industries"],
                    y=data["audience"]["engagement_by_industry"],
                    name='Engagement Rate',
                    marker_color='orange'
                )
            ])
            
            fig.update_layout(
                title="Engagement Rate by Industry",
                xaxis_title="Industry",
                yaxis_title="Engagement Rate",
                showlegend=True
            )
            
        else:
            # Default secondary chart
            fig = go.Figure()
            fig.add_annotation(
                text=f"Secondary analysis for: {chart_type}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        return fig
    
    def _calculate_chart_statistics(self, data: Dict) -> Dict:
        """Calculate statistics for the chart data"""
        
        engagement = data["engagement"]
        
        stats = {
            "total_days": len(engagement["dates"]),
            "average_open_rate": round(np.mean(engagement["open_rates"]), 3),
            "average_click_rate": round(np.mean(engagement["click_rates"]), 3),
            "average_conversion_rate": round(np.mean(engagement["conversion_rates"]), 4),
            "total_revenue": round(sum(engagement["revenue"]), 2),
            "best_performing_day": engagement["dates"][np.argmax(engagement["open_rates"])],
            "worst_performing_day": engagement["dates"][np.argmin(engagement["open_rates"])],
            "chart_type": data["chart_type"],
            "data_source": data["data_source"]
        }
        
        return stats
    
    def create_ab_testing_demo(self) -> gr.Blocks:
        """Create A/B testing demo"""
        
        with gr.Blocks(title="A/B Testing Demo", theme=gr.themes.Soft()) as demo:
            
            gr.Markdown("""
            # 🧪 A/B Testing Demo
            
            Compare different email sequence variations and see which performs better.
            Test subject lines, content, timing, and more with statistical analysis.
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Test Configuration")
                    
                    # Test type
                    test_type = gr.Dropdown(
                        choices=[
                            "Subject Line Test",
                            "Content Variation Test",
                            "Timing Test",
                            "Sequence Length Test",
                            "Personalization Test"
                        ],
                        value="Subject Line Test",
                        label="Test Type",
                        info="Choose what to test"
                    )
                    
                    # Sample size
                    sample_size = gr.Slider(
                        minimum=100,
                        maximum=10000,
                        value=1000,
                        step=100,
                        label="Sample Size",
                        info="Number of subscribers in each group"
                    )
                    
                    # Test duration
                    test_duration = gr.Slider(
                        minimum=1,
                        maximum=30,
                        value=7,
                        step=1,
                        label="Test Duration (days)",
                        info="How long to run the test"
                    )
                    
                    # Confidence level
                    confidence_level = gr.Slider(
                        minimum=0.8,
                        maximum=0.99,
                        value=0.95,
                        step=0.01,
                        label="Confidence Level",
                        info="Statistical confidence level"
                    )
                    
                    # Run test button
                    run_test_btn = gr.Button("🧪 Run A/B Test", variant="primary", size="lg")
                
                with gr.Column(scale=2):
                    gr.Markdown("### Test Results")
                    
                    # Test results
                    test_results = gr.JSON(
                        label="A/B Test Results"
                    )
                    
                    # Results visualization
                    results_chart = gr.Plot(
                        label="Test Results Visualization"
                    )
                    
                    # Statistical analysis
                    statistical_analysis = gr.Markdown(
                        label="Statistical Analysis"
                    )
            
            # A/B testing function
            def run_ab_test(
                test_type: str,
                sample_size: int,
                test_duration: int,
                confidence_level: float
            ) -> Tuple[Dict, go.Figure, str]:
                
                try:
                    # Generate A/B test data
                    test_data = self._generate_ab_test_data(
                        test_type, sample_size, test_duration
                    )
                    
                    # Perform statistical analysis
                    analysis = self._perform_statistical_analysis(
                        test_data, confidence_level
                    )
                    
                    # Create visualization
                    chart = self._create_ab_test_chart(test_data, analysis)
                    
                    # Create analysis report
                    report = self._create_ab_test_report(test_data, analysis)
                    
                    return test_data, chart, report
                    
                except Exception as e:
                    logger.error(f"Error in A/B test: {e}")
                    return {"error": str(e)}, go.Figure(), f"Error: {str(e)}"
            
            # Connect the run test button
            run_test_btn.click(
                fn=run_ab_test,
                inputs=[test_type, sample_size, test_duration, confidence_level],
                outputs=[test_results, results_chart, statistical_analysis]
            )
        
        return demo
    
    def _generate_ab_test_data(
        self,
        test_type: str,
        sample_size: int,
        test_duration: int
    ) -> Dict:
        """Generate A/B test data"""
        
        # Generate control group data
        control_metrics = {
            "open_rate": round(random.uniform(0.25, 0.35), 3),
            "click_rate": round(random.uniform(0.03, 0.06), 3),
            "conversion_rate": round(random.uniform(0.008, 0.015), 4),
            "revenue_per_subscriber": round(random.uniform(5, 15), 2)
        }
        
        # Generate variation group data (with some improvement)
        improvement_factor = random.uniform(1.05, 1.25)  # 5-25% improvement
        variation_metrics = {
            "open_rate": round(control_metrics["open_rate"] * improvement_factor, 3),
            "click_rate": round(control_metrics["click_rate"] * improvement_factor, 3),
            "conversion_rate": round(control_metrics["conversion_rate"] * improvement_factor, 4),
            "revenue_per_subscriber": round(control_metrics["revenue_per_subscriber"] * improvement_factor, 2)
        }
        
        # Calculate totals
        control_totals = {
            "subscribers": sample_size,
            "opens": int(sample_size * control_metrics["open_rate"]),
            "clicks": int(sample_size * control_metrics["click_rate"]),
            "conversions": int(sample_size * control_metrics["conversion_rate"]),
            "total_revenue": sample_size * control_metrics["revenue_per_subscriber"]
        }
        
        variation_totals = {
            "subscribers": sample_size,
            "opens": int(sample_size * variation_metrics["open_rate"]),
            "clicks": int(sample_size * variation_metrics["click_rate"]),
            "conversions": int(sample_size * variation_metrics["conversion_rate"]),
            "total_revenue": sample_size * variation_metrics["revenue_per_subscriber"]
        }
        
        return {
            "test_type": test_type,
            "sample_size": sample_size,
            "test_duration": test_duration,
            "control": {
                "metrics": control_metrics,
                "totals": control_totals
            },
            "variation": {
                "metrics": variation_metrics,
                "totals": variation_totals
            },
            "test_date": datetime.now().isoformat()
        }
    
    def _perform_statistical_analysis(self, test_data: Dict, confidence_level: float) -> Dict:
        """Perform statistical analysis on A/B test results"""
        
        control = test_data["control"]["metrics"]
        variation = test_data["variation"]["metrics"]
        
        # Calculate improvements
        improvements = {
            "open_rate": round((variation["open_rate"] - control["open_rate"]) / control["open_rate"] * 100, 2),
            "click_rate": round((variation["click_rate"] - control["click_rate"]) / control["click_rate"] * 100, 2),
            "conversion_rate": round((variation["conversion_rate"] - control["conversion_rate"]) / control["conversion_rate"] * 100, 2),
            "revenue_per_subscriber": round((variation["revenue_per_subscriber"] - control["revenue_per_subscriber"]) / control["revenue_per_subscriber"] * 100, 2)
        }
        
        # Determine winner (simplified statistical test)
        winner = "Variation" if improvements["revenue_per_subscriber"] > 0 else "Control"
        
        # Calculate confidence intervals (simplified)
        confidence_intervals = {
            "open_rate": round(random.uniform(0.02, 0.08), 3),
            "click_rate": round(random.uniform(0.005, 0.015), 4),
            "conversion_rate": round(random.uniform(0.001, 0.003), 5),
            "revenue_per_subscriber": round(random.uniform(0.5, 2.0), 2)
        }
        
        # Statistical significance (simplified)
        is_significant = random.choice([True, False]) if abs(improvements["revenue_per_subscriber"]) > 5 else False
        
        return {
            "improvements": improvements,
            "winner": winner,
            "confidence_intervals": confidence_intervals,
            "is_statistically_significant": is_significant,
            "confidence_level": confidence_level,
            "recommendation": self._generate_ab_test_recommendation(improvements, is_significant)
        }
    
    def _generate_ab_test_recommendation(self, improvements: Dict, is_significant: bool) -> str:
        """Generate recommendation based on A/B test results"""
        
        if not is_significant:
            return "Continue testing - results are not statistically significant yet."
        
        if improvements["revenue_per_subscriber"] > 10:
            return "Strong recommendation to implement the variation - significant revenue improvement."
        elif improvements["revenue_per_subscriber"] > 5:
            return "Recommend implementing the variation - moderate improvement observed."
        elif improvements["revenue_per_subscriber"] > 0:
            return "Consider implementing the variation - slight improvement observed."
        else:
            return "Recommend keeping the control - variation did not improve performance."
    
    def _create_ab_test_chart(self, test_data: Dict, analysis: Dict) -> go.Figure:
        """Create A/B test results visualization"""
        
        metrics = ["Open Rate", "Click Rate", "Conversion Rate", "Revenue/Subscriber"]
        control_values = [
            test_data["control"]["metrics"]["open_rate"],
            test_data["control"]["metrics"]["click_rate"],
            test_data["control"]["metrics"]["conversion_rate"],
            test_data["control"]["metrics"]["revenue_per_subscriber"]
        ]
        variation_values = [
            test_data["variation"]["metrics"]["open_rate"],
            test_data["variation"]["metrics"]["click_rate"],
            test_data["variation"]["metrics"]["conversion_rate"],
            test_data["variation"]["metrics"]["revenue_per_subscriber"]
        ]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Control',
            x=metrics,
            y=control_values,
            marker_color='lightblue'
        ))
        
        fig.add_trace(go.Bar(
            name='Variation',
            x=metrics,
            y=variation_values,
            marker_color='lightgreen'
        ))
        
        fig.update_layout(
            title=f"A/B Test Results: {test_data['test_type']}",
            xaxis_title="Metrics",
            yaxis_title="Values",
            barmode='group',
            showlegend=True
        )
        
        return fig
    
    def _create_ab_test_report(self, test_data: Dict, analysis: Dict) -> str:
        """Create A/B test analysis report"""
        
        report = f"# A/B Test Analysis Report\n\n"
        report += f"**Test Type:** {test_data['test_type']}\n"
        report += f"**Sample Size:** {test_data['sample_size']} subscribers per group\n"
        report += f"**Test Duration:** {test_data['test_duration']} days\n"
        report += f"**Confidence Level:** {analysis['confidence_level']:.0%}\n\n"
        
        report += "## Results Summary\n\n"
        report += f"**Winner:** {analysis['winner']}\n"
        report += f"**Statistical Significance:** {'Yes' if analysis['is_statistically_significant'] else 'No'}\n\n"
        
        report += "## Performance Improvements\n\n"
        for metric, improvement in analysis['improvements'].items():
            report += f"- **{metric.replace('_', ' ').title()}:** {improvement:+.1f}%\n"
        
        report += f"\n## Recommendation\n\n{analysis['recommendation']}\n\n"
        
        report += "## Detailed Metrics\n\n"
        report += "| Metric | Control | Variation | Improvement |\n"
        report += "|--------|---------|-----------|-------------|\n"
        
        control = test_data["control"]["metrics"]
        variation = test_data["variation"]["metrics"]
        
        for metric, improvement in analysis['improvements'].items():
            control_val = control[metric]
            variation_val = variation[metric]
            report += f"| {metric.replace('_', ' ').title()} | {control_val:.3f} | {variation_val:.3f} | {improvement:+.1f}% |\n"
        
        return report


def create_demo_launcher():
    """Create a launcher for all interactive demos"""
    
    demos = InteractiveDemos()
    
    with gr.Blocks(title="Email Sequence AI - Interactive Demos", theme=gr.themes.Soft()) as app:
        
        gr.Markdown("""
        # 🚀 Email Sequence AI - Interactive Demos
        
        Explore the power of AI-driven email sequence generation through interactive demonstrations.
        Each demo showcases different aspects of the system's capabilities.
        """)
        
        with gr.Tabs():
            with gr.Tab("🤖 Live Inference"):
                live_demo = demos.create_live_inference_demo()
            
            with gr.Tab("📊 Visualizations"):
                viz_demo = demos.create_visualization_demo()
            
            with gr.Tab("🧪 A/B Testing"):
                ab_demo = demos.create_ab_testing_demo()
        
        gr.Markdown("""
        ---
        
        **Note:** These demos use simulated data for demonstration purposes.
        In production, the system would use real AI models and actual data.
        """)
    
    return app


def main():
    """Launch the interactive demos"""
    
    app = create_demo_launcher()
    
    app.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=True,
        debug=True,
        show_error=True
    )


match __name__:
    case "__main__":
    main() 