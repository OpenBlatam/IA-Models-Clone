from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import gradio as gr
import argparse
import logging
import sys
from pathlib import Path
from interactive_demos import InteractiveDemos
from performance_monitoring_demo import PerformanceMonitoringDemo
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
Comprehensive Demo Launcher

A unified launcher for all interactive demos showcasing the email sequence AI system.
Provides easy access to all demonstration features through a single interface.
"""


# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent.parent))


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ComprehensiveDemoLauncher:
    """Comprehensive demo launcher for all interactive demos"""
    
    def __init__(self) -> Any:
        self.interactive_demos = InteractiveDemos()
        self.performance_demo = PerformanceMonitoringDemo()
        
        logger.info("Comprehensive Demo Launcher initialized")
    
    def create_main_demo_interface(self) -> gr.Blocks:
        """Create the main demo interface with all demos"""
        
        with gr.Blocks(
            title="Email Sequence AI - Interactive Demos",
            theme=gr.themes.Soft(),
            css="""
            .gradio-container {
                max-width: 1400px !important;
            }
            .demo-header {
                text-align: center;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border-radius: 10px;
                margin-bottom: 20px;
            }
            .demo-description {
                background: #f8f9fa;
                padding: 15px;
                border-radius: 8px;
                margin: 10px 0;
                border-left: 4px solid #667eea;
            }
            """
        ) as app:
            
            # Header
            gr.HTML("""
            <div class="demo-header">
                <h1>🚀 Email Sequence AI - Interactive Demos</h1>
                <p>Experience the power of AI-driven email sequence generation through interactive demonstrations</p>
            </div>
            """)
            
            # Main tabs
            with gr.Tabs():
                
                # Live Inference Demo
                with gr.Tab("🤖 Live Model Inference"):
                    gr.HTML("""
                    <div class="demo-description">
                        <h3>Real-time AI Sequence Generation</h3>
                        <p>Watch as the system generates personalized email sequences using different AI models. 
                        Experience the power of GPT-3.5, GPT-4, Claude, and custom models in real-time.</p>
                    </div>
                    """)
                    live_demo = self.interactive_demos.create_live_inference_demo()
                
                # Visualization Demo
                with gr.Tab("📊 Interactive Visualizations"):
                    gr.HTML("""
                    <div class="demo-description">
                        <h3>Data Visualization & Analytics</h3>
                        <p>Explore email sequence performance through interactive charts and analytics. 
                        Visualize engagement patterns, conversion rates, and optimization opportunities.</p>
                    </div>
                    """)
                    viz_demo = self.interactive_demos.create_visualization_demo()
                
                # A/B Testing Demo
                with gr.Tab("🧪 A/B Testing"):
                    gr.HTML("""
                    <div class="demo-description">
                        <h3>Statistical A/B Testing</h3>
                        <p>Compare different email sequence variations and see which performs better. 
                        Test subject lines, content, timing, and more with statistical analysis.</p>
                    </div>
                    """)
                    ab_demo = self.interactive_demos.create_ab_testing_demo()
                
                # Performance Monitoring
                with gr.Tab("📈 Performance Monitoring"):
                    gr.HTML("""
                    <div class="demo-description">
                        <h3>Real-time System Monitoring</h3>
                        <p>Monitor system performance, model accuracy, and user activity in real-time. 
                        Track key metrics and receive alerts for performance issues.</p>
                    </div>
                    """)
                    monitoring_demo = self.performance_demo.create_performance_dashboard()
                
                # Predictive Analytics
                with gr.Tab("🔮 Predictive Analytics"):
                    gr.HTML("""
                    <div class="demo-description">
                        <h3>Future Performance Prediction</h3>
                        <p>Explore predictive analytics for email sequence performance, user behavior forecasting, 
                        and system optimization predictions.</p>
                    </div>
                    """)
                    predictive_demo = self.performance_demo.create_predictive_analytics_demo()
                
                # System Overview
                with gr.Tab("🏠 System Overview"):
                    self._create_system_overview()
            
            # Footer
            gr.HTML("""
            <div style="text-align: center; padding: 20px; margin-top: 30px; border-top: 1px solid #eee;">
                <p><strong>Email Sequence AI System</strong> - Powered by advanced AI models and comprehensive analytics</p>
                <p>Built with Gradio, PyTorch, and modern machine learning techniques</p>
                <p style="font-size: 0.9em; color: #666;">
                    Note: These demos use simulated data for demonstration purposes. 
                    In production, the system uses real AI models and actual data.
                </p>
            </div>
            """)
        
        return app
    
    def _create_system_overview(self) -> Any:
        """Create system overview tab"""
        
        gr.Markdown("""
        # 🏠 System Overview
        
        Welcome to the Email Sequence AI system! This comprehensive platform combines
        advanced AI models with sophisticated analytics to create, optimize, and
        manage email sequences.
        
        ## 🎯 Key Features
        
        ### 🤖 AI-Powered Generation
        - **Multiple AI Models**: GPT-3.5, GPT-4, Claude, and custom models
        - **Personalization**: Tailored content based on subscriber profiles
        - **Industry Focus**: Specialized sequences for different industries
        - **Creativity Control**: Adjustable AI creativity levels
        
        ### 📊 Advanced Analytics
        - **Multi-metric Evaluation**: Content quality, engagement, business impact
        - **Real-time Monitoring**: Live performance tracking
        - **Predictive Analytics**: Future performance forecasting
        - **A/B Testing**: Statistical comparison of variations
        
        ### 🚀 Performance Optimization
        - **Gradient Management**: Advanced training optimization
        - **Early Stopping**: Intelligent training termination
        - **Learning Rate Scheduling**: Adaptive learning rates
        - **Memory Optimization**: Efficient resource usage
        
        ## 📈 System Architecture
        
        ```
        ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
        │   User Interface│    │  AI Generation  │    │   Analytics     │
        │   (Gradio Web)  │◄──►│   Engine        │◄──►│   & Monitoring  │
        └─────────────────┘    └─────────────────┘    └─────────────────┘
                │                       │                       │
                ▼                       ▼                       ▼
        ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
        │   Data Models   │    │  Training &     │    │   Performance   │
        │   & Templates   │    │  Optimization   │    │   Tracking      │
        └─────────────────┘    └─────────────────┘    └─────────────────┘
        ```
        
        ## 🔧 Technical Stack
        
        - **Frontend**: Gradio (Python web interface)
        - **AI Models**: OpenAI GPT, Anthropic Claude, Custom models
        - **Machine Learning**: PyTorch, Transformers, Scikit-learn
        - **Analytics**: Plotly, Pandas, NumPy
        - **Monitoring**: Real-time metrics, alerting, logging
        
        ## 📊 Demo Capabilities
        
        | Demo | Description | Key Features |
        |------|-------------|--------------|
        | **Live Inference** | Real-time sequence generation | Multiple AI models, personalization, progress tracking |
        | **Visualizations** | Interactive data analysis | Charts, metrics, performance comparison |
        | **A/B Testing** | Statistical comparison | Hypothesis testing, confidence intervals, recommendations |
        | **Performance Monitoring** | Real-time system tracking | Metrics, alerts, health monitoring |
        | **Predictive Analytics** | Future performance prediction | Forecasting, trend analysis, insights |
        
        ## 🎮 Getting Started
        
        1. **Start with Live Inference**: Generate your first AI-powered sequence
        2. **Explore Visualizations**: Understand performance patterns
        3. **Run A/B Tests**: Compare different approaches
        4. **Monitor Performance**: Track system health and metrics
        5. **Predict Trends**: Plan for future optimization
        
        ## 📚 Documentation
        
        - **API Reference**: Complete system documentation
        - **User Guide**: Step-by-step instructions
        - **Examples**: Code examples and use cases
        - **Troubleshooting**: Common issues and solutions
        
        ## 🔗 Integration
        
        The system can be integrated with:
        - Email marketing platforms (Mailchimp, SendGrid, etc.)
        - CRM systems (Salesforce, HubSpot, etc.)
        - Analytics platforms (Google Analytics, Mixpanel, etc.)
        - Custom applications via REST API
        
        ## 🚀 Deployment Options
        
        - **Local Development**: Run on your machine
        - **Cloud Deployment**: Deploy to AWS, GCP, or Azure
        - **Containerized**: Docker containers for easy deployment
        - **Serverless**: Function-based deployment
        
        ## 📞 Support
        
        - **Documentation**: Comprehensive guides and tutorials
        - **Community**: Active user community and forums
        - **Support**: Technical support and consulting
        - **Training**: Workshops and training sessions
        
        ---
        
        **Ready to explore?** Navigate to any of the demo tabs above to start experiencing
        the power of AI-driven email sequence generation!
        """)
    
    def create_quick_start_demo(self) -> gr.Blocks:
        """Create a quick start demo for new users"""
        
        with gr.Blocks(title="Quick Start Demo", theme=gr.themes.Soft()) as demo:
            
            gr.Markdown("""
            # 🚀 Quick Start Demo
            
            Get started with the Email Sequence AI system in just a few minutes.
            This guided demo will walk you through the basic features.
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Step 1: Choose Your Demo")
                    
                    demo_choice = gr.Dropdown(
                        choices=[
                            "Generate Your First Sequence",
                            "View Performance Analytics",
                            "Run a Quick A/B Test",
                            "Monitor System Health"
                        ],
                        value="Generate Your First Sequence",
                        label="Select Demo Type",
                        info="Choose what you'd like to try first"
                    )
                    
                    start_demo_btn = gr.Button("🎯 Start Demo", variant="primary", size="lg")
                
                with gr.Column(scale=2):
                    gr.Markdown("### Demo Results")
                    
                    demo_output = gr.JSON(
                        label="Demo Results"
                    )
                    
                    demo_preview = gr.Markdown(
                        label="Demo Preview"
                    )
            
            def run_quick_demo(demo_type: str) -> Tuple[Dict, str]:
                """Run a quick demo based on user selection"""
                
                if demo_type == "Generate Your First Sequence":
                    # Simulate sequence generation
                    result = {
                        "demo_type": "Sequence Generation",
                        "status": "success",
                        "sequence": {
                            "id": "quick_demo_001",
                            "name": "Welcome Sequence for New User",
                            "length": 3,
                            "model_used": "GPT-3.5",
                            "generation_time": "2.3 seconds"
                        },
                        "steps": [
                            {
                                "order": 1,
                                "content": "Hi there! Welcome to our platform. We're excited to help you get started...",
                                "delay": "24 hours"
                            },
                            {
                                "order": 2,
                                "content": "Here are some tips to help you make the most of our features...",
                                "delay": "48 hours"
                            },
                            {
                                "order": 3,
                                "content": "Ready to take the next step? Here's how to get started...",
                                "delay": "72 hours"
                            }
                        ]
                    }
                    
                    preview = """
                    # 🎉 Your First AI-Generated Sequence!
                    
                    **Sequence Name:** Welcome Sequence for New User
                    **Length:** 3 emails
                    **Model:** GPT-3.5
                    **Generation Time:** 2.3 seconds
                    
                    ## Email 1 (Day 1)
                    Hi there! Welcome to our platform. We're excited to help you get started...
                    
                    ## Email 2 (Day 2)
                    Here are some tips to help you make the most of our features...
                    
                    ## Email 3 (Day 3)
                    Ready to take the next step? Here's how to get started...
                    
                    ---
                    
                    **Next Steps:**
                    1. Try the Live Inference demo for more customization
                    2. Explore the Visualization demo to see performance metrics
                    3. Run an A/B test to optimize your sequences
                    """
                
                elif demo_type == "View Performance Analytics":
                    result = {
                        "demo_type": "Performance Analytics",
                        "status": "success",
                        "metrics": {
                            "average_open_rate": 0.35,
                            "average_click_rate": 0.05,
                            "average_conversion_rate": 0.012,
                            "total_sequences": 150,
                            "total_subscribers": 2500
                        }
                    }
                    
                    preview = """
                    # 📊 Performance Analytics Overview
                    
                    **Average Open Rate:** 35%
                    **Average Click Rate:** 5%
                    **Average Conversion Rate:** 1.2%
                    **Total Sequences:** 150
                    **Total Subscribers:** 2,500
                    
                    ## Key Insights
                    - Your sequences are performing above industry average
                    - Click rates show good engagement
                    - Conversion rates indicate effective sequences
                    
                    ---
                    
                    **Next Steps:**
                    1. Explore the Visualization demo for detailed charts
                    2. Use A/B testing to improve performance
                    3. Monitor real-time metrics in the Performance tab
                    """
                
                elif demo_type == "Run a Quick A/B Test":
                    result = {
                        "demo_type": "A/B Testing",
                        "status": "success",
                        "test_results": {
                            "control_performance": 0.35,
                            "variation_performance": 0.42,
                            "improvement": "+20%",
                            "statistical_significance": True,
                            "recommendation": "Implement variation"
                        }
                    }
                    
                    preview = """
                    # 🧪 A/B Test Results
                    
                    **Control Performance:** 35%
                    **Variation Performance:** 42%
                    **Improvement:** +20%
                    **Statistical Significance:** Yes
                    **Recommendation:** Implement variation
                    
                    ## Analysis
                    The variation performed significantly better than the control.
                    The improvement is statistically significant, indicating a real effect.
                    
                    ---
                    
                    **Next Steps:**
                    1. Implement the winning variation
                    2. Continue monitoring performance
                    3. Run additional tests for further optimization
                    """
                
                else:  # Monitor System Health
                    result = {
                        "demo_type": "System Health",
                        "status": "success",
                        "health_metrics": {
                            "cpu_usage": 45.2,
                            "memory_usage": 62.8,
                            "gpu_usage": 23.1,
                            "response_time": 1.8,
                            "error_rate": 0.001
                        }
                    }
                    
                    preview = """
                    # 📈 System Health Overview
                    
                    **CPU Usage:** 45.2% ✅
                    **Memory Usage:** 62.8% ✅
                    **GPU Usage:** 23.1% ✅
                    **Response Time:** 1.8s ✅
                    **Error Rate:** 0.1% ✅
                    
                    ## Status: All Systems Operational
                    All metrics are within normal ranges.
                    System performance is optimal.
                    
                    ---
                    
                    **Next Steps:**
                    1. Monitor real-time metrics in the Performance tab
                    2. Set up alerts for performance thresholds
                    3. Use predictive analytics to plan for scaling
                    """
                
                return result, preview
            
            # Connect the start demo button
            start_demo_btn.click(
                fn=run_quick_demo,
                inputs=[demo_choice],
                outputs=[demo_output, demo_preview]
            )
        
        return demo


def create_comprehensive_launcher():
    """Create the comprehensive demo launcher"""
    
    launcher = ComprehensiveDemoLauncher()
    
    with gr.Blocks(
        title="Email Sequence AI - Comprehensive Demos",
        theme=gr.themes.Soft()
    ) as app:
        
        # Main interface
        main_interface = launcher.create_main_demo_interface()
        
        # Quick start option
        with gr.Accordion("🚀 Quick Start (New Users)", open=False):
            quick_start = launcher.create_quick_start_demo()
    
    return app


def main():
    """Main function to launch the comprehensive demo"""
    
    parser = argparse.ArgumentParser(
        description="Launch Email Sequence AI Interactive Demos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo_launcher.py                    # Launch with default settings
  python demo_launcher.py --port 7863        # Launch on custom port
  python demo_launcher.py --share            # Enable public sharing
  python demo_launcher.py --debug            # Enable debug mode
        """
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=7863,
        help='Port to run the server on (default: 7863)'
    )
    
    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='Host to bind the server to (default: 0.0.0.0)'
    )
    
    parser.add_argument(
        '--share',
        action='store_true',
        help='Enable public sharing (creates a public URL)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )
    
    args = parser.parse_args()
    
    # Display startup information
    logger.info("=" * 60)
    logger.info("Email Sequence AI - Comprehensive Interactive Demos")
    logger.info("=" * 60)
    logger.info(f"Server: {args.host}:{args.port}")
    logger.info(f"Share: {args.share}")
    logger.info(f"Debug: {args.debug}")
    logger.info("=" * 60)
    
    # Create and launch the app
    app = create_comprehensive_launcher()
    
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        debug=args.debug,
        show_error=True
    )


match __name__:
    case "__main__":
    main() 