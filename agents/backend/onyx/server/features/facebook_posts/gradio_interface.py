import gradio as gr
import torch
import numpy as np
import json
import time
import asyncio
from typing import Dict, Any, List, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Import our integrated advanced system
from .integrated_advanced_system import IntegratedAdvancedSystem, IntegratedSystemConfig
from .performance_optimization_engine import PerformanceConfig
from .ai_agent_system import AgentConfig, AgentType


class FacebookOptimizationInterface:
    """Advanced Gradio interface for Facebook content optimization"""
    
    def __init__(self):
        self.system = None
        self.config = None
        self.optimization_history = []
        self.performance_metrics = {}
        
        # Initialize the system
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize the integrated advanced system"""
        try:
            # Create configuration
            self.config = IntegratedSystemConfig(
                enable_mixed_precision=True,
                enable_caching=True,
                cache_size=20000,
                max_workers=16,
                enable_ai_agents=True,
                agent_autonomous_mode=True,
                enable_real_time_optimization=True,
                enable_ab_testing=True,
                enable_performance_monitoring=True
            )
            
            # Create system
            self.system = IntegratedAdvancedSystem(self.config)
            
            print("✅ Integrated Advanced System initialized successfully")
            
        except Exception as e:
            print(f"❌ Error initializing system: {e}")
            self.system = None
    
    def optimize_content(self, text: str, content_type: str, 
                        enable_ai_agents: bool, enable_caching: bool,
                        max_workers: int, cache_size: int) -> Dict[str, Any]:
        """Optimize content using the integrated system"""
        if not self.system:
            return {"error": "System not initialized"}
        
        try:
            # Update configuration if changed
            if (self.config.max_workers != max_workers or 
                self.config.cache_size != cache_size or
                self.config.enable_ai_agents != enable_ai_agents or
                self.config.enable_caching != enable_caching):
                
                self.config.max_workers = max_workers
                self.config.cache_size = cache_size
                self.config.enable_ai_agents = enable_ai_agents
                self.config.enable_caching = enable_caching
                
                # Reinitialize system with new config
                self.system.cleanup()
                self.system = IntegratedAdvancedSystem(self.config)
            
            # Optimize content
            result = self.system.optimize_content(text, content_type)
            
            # Store in history
            self.optimization_history.append({
                'timestamp': time.time(),
                'text': text,
                'content_type': content_type,
                'result': result
            })
            
            return result
            
        except Exception as e:
            return {"error": f"Optimization failed: {str(e)}"}
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        if not self.system:
            return {"error": "System not initialized"}
        
        try:
            status = self.system.get_system_status()
            return status
        except Exception as e:
            return {"error": f"Failed to get status: {str(e)}"}
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics and statistics"""
        if not self.system:
            return {"error": "System not initialized"}
        
        try:
            # Get performance engine stats
            perf_stats = self.system.performance_engine.get_performance_stats()
            
            # Get AI agent stats if available
            agent_stats = {}
            if self.system.ai_agent_system:
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    agent_stats = loop.run_until_complete(
                        self.system.ai_agent_system.get_system_status()
                    )
                    loop.close()
                except Exception as e:
                    agent_stats = {"error": str(e)}
            
            metrics = {
                'performance_engine': perf_stats,
                'ai_agents': agent_stats,
                'optimization_history_count': len(self.optimization_history),
                'system_uptime': time.time() - self.optimization_history[0]['timestamp'] if self.optimization_history else 0
            }
            
            return metrics
            
        except Exception as e:
            return {"error": f"Failed to get metrics: {str(e)}"}
    
    def create_performance_dashboard(self) -> go.Figure:
        """Create performance dashboard visualization"""
        if not self.optimization_history:
            # Return empty chart
            fig = go.Figure()
            fig.add_annotation(
                text="No optimization data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Extract data
        timestamps = [entry['timestamp'] for entry in self.optimization_history]
        engagement_scores = [entry['result'].get('engagement_score', 0) for entry in self.optimization_history]
        quality_scores = [entry['result'].get('content_quality', 0) for entry in self.optimization_history]
        viral_potentials = [entry['result'].get('viral_potential', 0) for entry in self.optimization_history]
        processing_times = [entry['result'].get('processing_time', 0) for entry in self.optimization_history]
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Engagement Scores', 'Content Quality', 'Viral Potential', 'Processing Time'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Engagement scores over time
        fig.add_trace(
            go.Scatter(x=timestamps, y=engagement_scores, mode='lines+markers', 
                      name='Engagement', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Content quality over time
        fig.add_trace(
            go.Scatter(x=timestamps, y=quality_scores, mode='lines+markers', 
                      name='Quality', line=dict(color='green')),
            row=1, col=2
        )
        
        # Viral potential over time
        fig.add_trace(
            go.Scatter(x=timestamps, y=viral_potentials, mode='lines+markers', 
                      name='Viral Potential', line=dict(color='red')),
            row=2, col=1
        )
        
        # Processing time over time
        fig.add_trace(
            go.Scatter(x=timestamps, y=processing_times, mode='lines+markers', 
                      name='Processing Time', line=dict(color='orange')),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title="Performance Dashboard - Facebook Content Optimization",
            height=600,
            showlegend=True
        )
        
        return fig
    
    def create_ai_agent_visualization(self) -> go.Figure:
        """Create AI agent system visualization"""
        if not self.system or not self.system.ai_agent_system:
            # Return empty chart
            fig = go.Figure()
            fig.add_annotation(
                text="AI Agent System not available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        try:
            # Get agent statuses
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            agent_statuses = loop.run_until_complete(
                self.system.ai_agent_system.get_system_status()
            )
            loop.close()
            
            # Extract agent data
            agents = agent_statuses.get('agent_statuses', {})
            
            if not agents:
                fig = go.Figure()
                fig.add_annotation(
                    text="No agent data available",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )
                return fig
            
            # Create agent performance chart
            agent_names = list(agents.keys())
            confidence_scores = [agents[name].get('confidence', 0) for name in agent_names]
            experience_counts = [agents[name].get('experience_count', 0) for name in agent_names]
            success_rates = [agents[name].get('performance_metrics', {}).get('success_rate', 0) for name in agent_names]
            
            fig = make_subplots(
                rows=1, cols=3,
                subplot_titles=('Agent Confidence', 'Experience Count', 'Success Rate'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Confidence scores
            fig.add_trace(
                go.Bar(x=agent_names, y=confidence_scores, name='Confidence', marker_color='lightblue'),
                row=1, col=1
            )
            
            # Experience counts
            fig.add_trace(
                go.Bar(x=agent_names, y=experience_counts, name='Experience', marker_color='lightgreen'),
                row=1, col=2
            )
            
            # Success rates
            fig.add_trace(
                go.Bar(x=agent_names, y=success_rates, name='Success Rate', marker_color='lightcoral'),
                row=1, col=3
            )
            
            fig.update_layout(
                title="AI Agent System Performance",
                height=400,
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error visualizing agents: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
    
    def create_interface(self) -> gr.Blocks:
        """Create the complete Gradio interface"""
        
        with gr.Blocks(
            title="Facebook Content Optimization - Integrated Advanced System",
            theme=gr.themes.Soft(),
            css="""
            .gradio-container {
                max-width: 1200px !important;
            }
            .performance-metrics {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 10px;
                margin: 10px 0;
            }
            """
        ) as interface:
            
            gr.Markdown("""
            # 🚀 Facebook Content Optimization - Integrated Advanced System
            
            **Advanced AI-powered content optimization with performance optimization and AI agents**
            
            This system combines:
            - 🤖 AI-powered content analysis and optimization
            - ⚡ High-performance optimization engine with caching
            - 🧠 Intelligent AI agent system for autonomous optimization
            - 📊 Real-time performance monitoring and analytics
            """)
            
            with gr.Tabs():
                
                # Tab 1: Content Analysis & Optimization
                with gr.Tab("🎯 Content Analysis & Optimization"):
                    
                    with gr.Row():
                        with gr.Column(scale=2):
                            content_input = gr.Textbox(
                                label="📝 Facebook Content",
                                placeholder="Enter your Facebook post content here...",
                                lines=8,
                                max_lines=15
                            )
                            
                            content_type = gr.Dropdown(
                                choices=["post", "story", "reel", "ad", "page_post"],
                                value="post",
                                label="📱 Content Type"
                            )
                        
                        with gr.Column(scale=1):
                            gr.Markdown("### ⚙️ System Configuration")
                            
                            enable_ai_agents = gr.Checkbox(
                                label="🤖 Enable AI Agents",
                                value=True,
                                info="Enable intelligent AI agent system"
                            )
                            
                            enable_caching = gr.Checkbox(
                                label="💾 Enable Caching",
                                value=True,
                                info="Enable high-performance caching"
                            )
                            
                            max_workers = gr.Slider(
                                minimum=1,
                                maximum=32,
                                value=16,
                                step=1,
                                label="🔧 Max Workers",
                                info="Number of parallel workers"
                            )
                            
                            cache_size = gr.Slider(
                                minimum=1000,
                                maximum=50000,
                                value=20000,
                                step=1000,
                                label="📦 Cache Size",
                                info="Cache size for optimization results"
                            )
                    
                    optimize_btn = gr.Button("🚀 Optimize Content", variant="primary", size="lg")
                    
                    with gr.Row():
                        with gr.Column():
                            results_output = gr.JSON(
                                label="📊 Optimization Results",
                                interactive=False
                            )
                        
                        with gr.Column():
                            suggestions_output = gr.Textbox(
                                label="💡 AI Agent Suggestions",
                                lines=6,
                                interactive=False
                            )
                
                # Tab 2: Performance Monitoring
                with gr.Tab("📊 Performance Monitoring"):
                    
                    with gr.Row():
                        refresh_btn = gr.Button("🔄 Refresh Metrics", variant="secondary")
                        system_status_btn = gr.Button("🔍 System Status", variant="secondary")
                    
                    with gr.Row():
                        with gr.Column():
                            performance_metrics = gr.JSON(
                                label="⚡ Performance Metrics",
                                interactive=False
                            )
                        
                        with gr.Column():
                            system_status = gr.JSON(
                                label="🏗️ System Status",
                                interactive=False
                            )
                    
                    gr.Markdown("### 📈 Performance Dashboard")
                    performance_chart = gr.Plotly(
                        label="Performance Trends",
                        interactive=True
                    )
                
                # Tab 3: AI Agent System
                with gr.Tab("🤖 AI Agent System"):
                    
                    gr.Markdown("### 🧠 AI Agent Performance & Collaboration")
                    
                    with gr.Row():
                        agent_visualization = gr.Plotly(
                            label="AI Agent Performance",
                            interactive=True
                        )
                    
                    with gr.Row():
                        agent_metrics = gr.JSON(
                            label="🤖 Agent Metrics",
                            interactive=False
                        )
                
                # Tab 4: System Configuration
                with gr.Tab("⚙️ System Configuration"):
                    
                    gr.Markdown("### 🔧 Advanced Configuration Options")
                    
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("#### 🚀 Performance Settings")
                            
                            enable_mixed_precision = gr.Checkbox(
                                label="🎯 Mixed Precision Training",
                                value=True,
                                info="Enable mixed precision for faster training"
                            )
                            
                            enable_gradient_checkpointing = gr.Checkbox(
                                label="💾 Gradient Checkpointing",
                                value=False,
                                info="Enable gradient checkpointing for memory efficiency"
                            )
                            
                            enable_memory_optimization = gr.Checkbox(
                                label="🧠 Memory Optimization",
                                value=True,
                                info="Enable automatic memory optimization"
                            )
                        
                        with gr.Column():
                            gr.Markdown("#### 🤖 AI Agent Settings")
                            
                            agent_learning_rate = gr.Slider(
                                minimum=0.0001,
                                maximum=0.01,
                                value=0.001,
                                step=0.0001,
                                label="📚 Agent Learning Rate",
                                info="Learning rate for AI agents"
                            )
                            
                            agent_memory_size = gr.Slider(
                                minimum=100,
                                maximum=10000,
                                value=1000,
                                step=100,
                                label="🧠 Agent Memory Size",
                                info="Memory size for agent learning"
                            )
                            
                            agent_autonomous_mode = gr.Checkbox(
                                label="🔄 Autonomous Mode",
                                value=True,
                                info="Enable autonomous agent operation"
                            )
                    
                    apply_config_btn = gr.Button("✅ Apply Configuration", variant="primary")
                    reset_config_btn = gr.Button("🔄 Reset to Defaults", variant="secondary")
            
            # Event handlers
            optimize_btn.click(
                fn=self.optimize_content,
                inputs=[content_input, content_type, enable_ai_agents, enable_caching, max_workers, cache_size],
                outputs=[results_output, suggestions_output]
            )
            
            refresh_btn.click(
                fn=self.get_performance_metrics,
                outputs=[performance_metrics]
            )
            
            system_status_btn.click(
                fn=self.get_system_status,
                outputs=[system_status]
            )
            
            # Auto-refresh performance chart
            interface.load(
                fn=self.create_performance_dashboard,
                outputs=[performance_chart]
            )
            
            # Auto-refresh agent visualization
            interface.load(
                fn=self.create_ai_agent_visualization,
                outputs=[agent_visualization]
            )
            
            # Update suggestions when results change
            results_output.change(
                fn=lambda x: x.get('ai_agent_suggestions', []) if isinstance(x, dict) else [],
                inputs=[results_output],
                outputs=[suggestions_output]
            )
            
            # Update performance metrics when results change
            results_output.change(
                fn=self.get_performance_metrics,
                outputs=[performance_metrics]
            )
            
            # Update agent metrics when results change
            results_output.change(
                fn=lambda: self.get_system_status().get('ai_agent_system_stats', {}),
                outputs=[agent_metrics]
            )
        
        return interface


def main():
    """Main function to launch the interface"""
    try:
        # Create interface
        interface = FacebookOptimizationInterface()
        gradio_interface = interface.create_interface()
        
        # Launch
        gradio_interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            debug=True,
            show_error=True
        )
        
    except Exception as e:
        print(f"❌ Error launching interface: {e}")
        raise


if __name__ == "__main__":
    main()

