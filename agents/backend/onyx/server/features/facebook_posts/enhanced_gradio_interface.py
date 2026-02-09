import gradio as gr
import torch
import numpy as np
import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import logging

# Import our enhanced components
from enhanced_integrated_system import EnhancedIntegratedSystem, EnhancedIntegratedSystemConfig
from enhanced_performance_engine import EnhancedPerformanceOptimizationEngine, EnhancedPerformanceConfig
from enhanced_ai_agent_system import EnhancedAIAgentSystem, EnhancedAgentConfig


class EnhancedGradioInterface:
    """Enhanced Gradio interface with advanced features"""
    
    def __init__(self):
        self.system = None
        self.is_initialized = False
        self.demo_data = self._load_demo_data()
        
        # Performance tracking
        self.request_history = []
        self.performance_metrics = {}
        
        # Initialize system
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize the enhanced system"""
        try:
            # Create configuration
            config = EnhancedIntegratedSystemConfig(
                environment="development",
                enable_ai_agents=True,
                enable_performance_monitoring=True,
                enable_health_checks=True,
                log_level="INFO"
            )
            
            # Initialize system
            self.system = EnhancedIntegratedSystem(config)
            self.system.start()
            
            self.is_initialized = True
            logging.info("✅ Enhanced system initialized successfully!")
            
        except Exception as e:
            logging.error(f"❌ Failed to initialize system: {e}")
            self.is_initialized = False
    
    def _load_demo_data(self) -> Dict[str, Any]:
        """Load demo data for the interface"""
        return {
            'sample_content': [
                "🚀 Discover our revolutionary AI-powered content optimization platform!",
                "💡 Transform your social media strategy with cutting-edge technology",
                "🔥 Boost engagement and reach with intelligent content recommendations",
                "📈 Maximize your ROI with data-driven content optimization",
                "🎯 Target the right audience with AI-powered insights"
            ],
            'content_types': ['Post', 'Story', 'Reel', 'Video', 'Image'],
            'audience_sizes': ['Small (1K-10K)', 'Medium (10K-100K)', 'Large (100K-1M)', 'Enterprise (1M+)'],
            'time_periods': ['Morning (6AM-12PM)', 'Afternoon (12PM-6PM)', 'Evening (6PM-12AM)', 'Night (12AM-6AM)']
        }
    
    def create_interface(self) -> gr.Blocks:
        """Create the enhanced Gradio interface"""
        with gr.Blocks(
            title="🚀 Enhanced Facebook Content Optimization System",
            theme=gr.themes.Soft(),
            css="""
            .gradio-container {
                max-width: 1200px !important;
                margin: 0 auto !important;
            }
            .header {
                text-align: center;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border-radius: 10px;
                margin-bottom: 20px;
            }
            .metric-card {
                background: white;
                padding: 15px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                margin: 10px 0;
            }
            .status-indicator {
                display: inline-block;
                width: 12px;
                height: 12px;
                border-radius: 50%;
                margin-right: 8px;
            }
            .status-healthy { background-color: #4CAF50; }
            .status-degraded { background-color: #FF9800; }
            .status-unhealthy { background-color: #F44336; }
            """
        ) as interface:
            
            # Header
            with gr.Row():
                gr.HTML("""
                <div class="header">
                    <h1>🚀 Enhanced Facebook Content Optimization System</h1>
                    <p>AI-Powered Content Analysis, Optimization, and Performance Prediction</p>
                    <p><strong>Version 2.0.0</strong> | Advanced AI Agents | Real-time Monitoring</p>
                </div>
                """)
            
            # Main content area
            with gr.Tabs():
                
                # Content Optimization Tab
                with gr.Tab("🎯 Content Optimization", id=0):
                    self._create_content_optimization_tab()
                
                # AI Agents Tab
                with gr.Tab("🤖 AI Agents", id=1):
                    self._create_ai_agents_tab()
                
                # Performance Monitoring Tab
                with gr.Tab("📊 Performance Monitoring", id=2):
                    self._create_performance_monitoring_tab()
                
                # System Health Tab
                with gr.Tab("🏥 System Health", id=3):
                    self._create_system_health_tab()
                
                # Analytics Tab
                with gr.Tab("📈 Analytics", id=4):
                    self._create_analytics_tab()
            
            # Footer
            with gr.Row():
                gr.HTML("""
                <div style="text-align: center; padding: 20px; color: #666;">
                    <p>Enhanced Facebook Content Optimization System v2.0.0</p>
                    <p>Powered by Advanced AI, Machine Learning, and Real-time Optimization</p>
                </div>
                """)
        
        return interface
    
    def _create_content_optimization_tab(self):
        """Create the content optimization tab"""
        with gr.Row():
            with gr.Column(scale=2):
                # Content input
                content_input = gr.Textbox(
                    label="📝 Content to Optimize",
                    placeholder="Enter your Facebook content here...",
                    lines=4,
                    max_lines=10
                )
                
                # Content type selection
                content_type = gr.Dropdown(
                    choices=self.demo_data['content_types'],
                    value="Post",
                    label="📱 Content Type",
                    info="Select the type of content you're creating"
                )
                
                # Context inputs
                with gr.Row():
                    audience_size = gr.Dropdown(
                        choices=self.demo_data['audience_sizes'],
                        value="Medium (10K-100K)",
                        label="👥 Target Audience Size"
                    )
                    
                    time_period = gr.Dropdown(
                        choices=self.demo_data['time_periods'],
                        value="Evening (6PM-12AM)",
                        label="⏰ Posting Time"
                    )
                
                # Optimization options
                with gr.Row():
                    enable_ai_agents = gr.Checkbox(
                        label="🤖 Enable AI Agents",
                        value=True,
                        info="Use advanced AI agents for enhanced analysis"
                    )
                    
                    enable_real_time = gr.Checkbox(
                        label="⚡ Real-time Optimization",
                        value=True,
                        info="Enable real-time content optimization"
                    )
                
                # Action buttons
                with gr.Row():
                    optimize_btn = gr.Button(
                        "🚀 Optimize Content",
                        variant="primary",
                        size="lg"
                    )
                    
                    clear_btn = gr.Button(
                        "🗑️ Clear",
                        variant="secondary"
                    )
                    
                    demo_btn = gr.Button(
                        "🎭 Load Demo",
                        variant="secondary"
                    )
                
                # Status indicator
                status_indicator = gr.HTML(
                    value="<div class='status-indicator status-healthy'></div> System Ready",
                    label="System Status"
                )
            
            with gr.Column(scale=2):
                # Results display
                results_header = gr.HTML(
                    value="<h3>📊 Optimization Results</h3>",
                    label="Results"
                )
                
                # Combined score
                combined_score = gr.Slider(
                    minimum=0,
                    maximum=1,
                    value=0.5,
                    step=0.01,
                    label="🎯 Combined Optimization Score",
                    interactive=False
                )
                
                # Detailed metrics
                with gr.Row():
                    engagement_score = gr.Slider(
                        minimum=0,
                        maximum=1,
                        value=0.5,
                        step=0.01,
                        label="❤️ Engagement Score",
                        interactive=False
                    )
                    
                    viral_potential = gr.Slider(
                        minimum=0,
                        maximum=1,
                        value=0.5,
                        step=0.01,
                        label="🔥 Viral Potential",
                        interactive=False
                    )
                
                # Recommendations
                recommendations = gr.Textbox(
                    label="💡 Optimization Recommendations",
                    lines=6,
                    max_lines=10,
                    interactive=False
                )
                
                # Processing time
                processing_time = gr.Textbox(
                    label="⏱️ Processing Time",
                    value="Not processed yet",
                    interactive=False
                )
                
                # Component usage
                components_used = gr.Textbox(
                    label="🔧 Components Used",
                    value="None",
                    interactive=False
                )
        
        # Event handlers
        optimize_btn.click(
            fn=self._optimize_content,
            inputs=[content_input, content_type, audience_size, time_period, enable_ai_agents, enable_real_time],
            outputs=[combined_score, engagement_score, viral_potential, recommendations, processing_time, components_used, status_indicator]
        )
        
        clear_btn.click(
            fn=self._clear_content_optimization,
            outputs=[content_input, combined_score, engagement_score, viral_potential, recommendations, processing_time, components_used]
        )
        
        demo_btn.click(
            fn=self._load_demo_content,
            outputs=[content_input, content_type, audience_size, time_period]
        )
    
    def _create_ai_agents_tab(self):
        """Create the AI agents tab"""
        with gr.Row():
            with gr.Column(scale=1):
                # Agent overview
                gr.HTML("<h3>🤖 AI Agent System Overview</h3>")
                
                # Agent status
                agent_status = gr.HTML(
                    value="<div class='status-indicator status-healthy'></div> All agents active",
                    label="Agent Status"
                )
                
                # Agent types
                agent_types = gr.HTML("""
                <div class='metric-card'>
                    <h4>🎯 Content Optimizer</h4>
                    <p>Specializes in content optimization strategies</p>
                </div>
                <div class='metric-card'>
                    <h4>📊 Engagement Analyzer</h4>
                    <p>Analyzes engagement patterns and factors</p>
                </div>
                <div class='metric-card'>
                    <h4>🔮 Trend Predictor</h4>
                    <p>Predicts content trends and viral potential</p>
                </div>
                <div class='metric-card'>
                    <h4>👥 Audience Targeter</h4>
                    <p>Optimizes audience targeting strategies</p>
                </div>
                <div class='metric-card'>
                    <h4>📈 Performance Monitor</h4>
                    <p>Monitors and tracks performance metrics</p>
                </div>
                """)
                
                # Agent controls
                with gr.Row():
                    refresh_agents_btn = gr.Button("🔄 Refresh Status", variant="secondary")
                    agent_stats_btn = gr.Button("📊 Agent Statistics", variant="secondary")
            
            with gr.Column(scale=2):
                # Agent performance
                gr.HTML("<h3>📊 Agent Performance Metrics</h3>")
                
                # Agent decision history
                agent_decisions = gr.Dataframe(
                    headers=["Agent", "Decisions", "Success Rate", "Specialization"],
                    datatype=["str", "int", "float", "str"],
                    col_count=(4, "fixed"),
                    label="Agent Decision History"
                )
                
                # Agent communication
                agent_communication = gr.Textbox(
                    label="💬 Agent Communication Log",
                    lines=8,
                    max_lines=15,
                    interactive=False
                )
                
                # Agent learning progress
                learning_progress = gr.Plot(
                    label="🧠 Agent Learning Progress"
                )
        
        # Event handlers
        refresh_agents_btn.click(
            fn=self._refresh_agent_status,
            outputs=[agent_status, agent_decisions, agent_communication]
        )
        
        agent_stats_btn.click(
            fn=self._get_agent_statistics,
            outputs=[learning_progress]
        )
    
    def _create_performance_monitoring_tab(self):
        """Create the performance monitoring tab"""
        with gr.Row():
            with gr.Column(scale=1):
                # Performance overview
                gr.HTML("<h3>📊 Performance Overview</h3>")
                
                # Cache performance
                cache_hit_rate = gr.Slider(
                    minimum=0,
                    maximum=1,
                    value=0.75,
                    step=0.01,
                    label="💾 Cache Hit Rate",
                    interactive=False
                )
                
                # Memory usage
                memory_usage = gr.Slider(
                    minimum=0,
                    maximum=16,
                    value=8.5,
                    step=0.1,
                    label="🧠 Memory Usage (GB)",
                    interactive=False
                )
                
                # GPU usage
                gpu_usage = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=65,
                    step=1,
                    label="🎮 GPU Usage (%)",
                    interactive=False
                )
                
                # Refresh button
                refresh_perf_btn = gr.Button("🔄 Refresh Metrics", variant="primary")
            
            with gr.Column(scale=2):
                # Performance charts
                gr.HTML("<h3>📈 Performance Trends</h3>")
                
                # Response time chart
                response_time_chart = gr.Plot(
                    label="⏱️ Response Time Trends"
                )
                
                # Throughput chart
                throughput_chart = gr.Plot(
                    label="🚀 Throughput Trends"
                )
        
        # Event handlers
        refresh_perf_btn.click(
            fn=self._refresh_performance_metrics,
            outputs=[cache_hit_rate, memory_usage, gpu_usage, response_time_chart, throughput_chart]
        )
    
    def _create_system_health_tab(self):
        """Create the system health tab"""
        with gr.Row():
            with gr.Column(scale=1):
                # System overview
                gr.HTML("<h3>🏥 System Health Overview</h3>")
                
                # Overall health
                overall_health = gr.HTML(
                    value="<div class='status-indicator status-healthy'></div> Healthy",
                    label="Overall Health"
                )
                
                # Health score
                health_score = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=95,
                    step=1,
                    label="🏆 Health Score",
                    interactive=False
                )
                
                # Uptime
                uptime = gr.Textbox(
                    label="⏰ System Uptime",
                    value="0 hours",
                    interactive=False
                )
                
                # Component status
                component_status = gr.HTML(
                    value="<p>All components operational</p>",
                    label="Component Status"
                )
                
                # Refresh button
                refresh_health_btn = gr.Button("🔄 Refresh Health", variant="primary")
            
            with gr.Column(scale=2):
                # Health details
                gr.HTML("<h3>🔍 Health Details</h3>")
                
                # Health history chart
                health_history_chart = gr.Plot(
                    label="📈 Health History"
                )
                
                # Error log
                error_log = gr.Textbox(
                    label="❌ Recent Errors",
                    lines=10,
                    max_lines=15,
                    interactive=False
                )
        
        # Event handlers
        refresh_health_btn.click(
            fn=self._refresh_system_health,
            outputs=[overall_health, health_score, uptime, component_status, health_history_chart, error_log]
        )
    
    def _create_analytics_tab(self):
        """Create the analytics tab"""
        with gr.Row():
            with gr.Column(scale=1):
                # Analytics controls
                gr.HTML("<h3>📊 Analytics Controls</h3>")
                
                # Date range
                date_range = gr.Dropdown(
                    choices=["Last 24 hours", "Last 7 days", "Last 30 days", "Custom"],
                    value="Last 7 days",
                    label="📅 Time Range"
                )
                
                # Metric selection
                metric_selection = gr.CheckboxGroup(
                    choices=["Engagement Rate", "Viral Potential", "Response Time", "Cache Performance", "Agent Accuracy"],
                    value=["Engagement Rate", "Viral Potential"],
                    label="📈 Metrics to Display"
                )
                
                # Generate button
                generate_analytics_btn = gr.Button("📊 Generate Analytics", variant="primary")
            
            with gr.Column(scale=2):
                # Analytics results
                gr.HTML("<h3>📈 Analytics Results</h3>")
                
                # Main analytics chart
                main_analytics_chart = gr.Plot(
                    label="📊 Main Analytics Chart"
                )
                
                # Summary statistics
                summary_stats = gr.Dataframe(
                    headers=["Metric", "Current", "Average", "Trend"],
                    datatype=["str", "float", "float", "str"],
                    col_count=(4, "fixed"),
                    label="Summary Statistics"
                )
        
        # Event handlers
        generate_analytics_btn.click(
            fn=self._generate_analytics,
            inputs=[date_range, metric_selection],
            outputs=[main_analytics_chart, summary_stats]
        )
    
    def _optimize_content(self, content: str, content_type: str, audience_size: str, 
                         time_period: str, enable_ai_agents: bool, enable_real_time: bool) -> tuple:
        """Optimize content using the enhanced system"""
        if not self.is_initialized or not self.system:
            return (0.5, 0.5, 0.5, "System not initialized", "N/A", "None", 
                   "<div class='status-indicator status-unhealthy'></div> System Error")
        
        try:
            # Create context
            context = {
                'audience_size': self._parse_audience_size(audience_size),
                'time_of_day': self._parse_time_period(time_period),
                'enable_ai_agents': enable_ai_agents,
                'enable_real_time': enable_real_time
            }
            
            # Process content
            start_time = time.time()
            result = self.system.process_content(content, content_type, context)
            processing_time = time.time() - start_time
            
            if result['status'] == 'success':
                # Extract results
                combined_score = result['result']['combined_score']
                engagement_score = result['result']['content_optimization'].get('engagement_score', 0.5)
                viral_potential = result['result']['content_optimization'].get('viral_potential', 0.5)
                
                # Format recommendations
                recommendations = "\n".join(result['result']['recommendations'][:5])
                
                # Format components used
                components_used = ", ".join(result['components_used'])
                
                # Update status
                status = "<div class='status-indicator status-healthy'></div> Optimization Complete"
                
                # Record request
                self._record_request(content, content_type, result, processing_time)
                
                return (combined_score, engagement_score, viral_potential, recommendations, 
                       f"{processing_time:.3f}s", components_used, status)
            else:
                # Error case
                return (0.5, 0.5, 0.5, f"Error: {result['error']}", 
                       f"{processing_time:.3f}s", "Error", 
                       "<div class='status-indicator status-unhealthy'></div> Processing Error")
                
        except Exception as e:
            logging.error(f"Error in content optimization: {e}")
            return (0.5, 0.5, 0.5, f"System error: {str(e)}", "N/A", "Error", 
                   "<div class='status-indicator status-unhealthy'></div> System Error")
    
    def _clear_content_optimization(self) -> tuple:
        """Clear the content optimization form"""
        return ("", 0.5, 0.5, 0.5, "", "", "")
    
    def _load_demo_content(self) -> tuple:
        """Load demo content"""
        import random
        demo_content = random.choice(self.demo_data['sample_content'])
        return (demo_content, "Post", "Medium (10K-100K)", "Evening (6PM-12AM)")
    
    def _refresh_agent_status(self) -> tuple:
        """Refresh agent status"""
        if not self.is_initialized or not self.system:
            return ("<div class='status-indicator status-unhealthy'></div> System Error", 
                   None, "System not initialized")
        
        try:
            # Get agent stats
            agent_stats = self.system.ai_agent_system.get_system_stats()
            
            # Update status
            overall_status = agent_stats['system_health']['overall_status']
            status_class = f"status-{overall_status}"
            status_html = f"<div class='status-indicator {status_class}'></div> {overall_status.title()}"
            
            # Create agent decisions dataframe
            agent_data = []
            for agent_id, stats in agent_stats['agent_performance'].items():
                agent_data.append([
                    stats['agent_type'],
                    stats['total_decisions'],
                    f"{stats['success_rate']:.3f}",
                    stats['specialization'].get('expertise_level', 'N/A')
                ])
            
            # Create communication log
            communication_log = f"Total agents: {len(agent_stats['agent_performance'])}\n"
            communication_log += f"Communication enabled: {agent_stats['system_health']['communication_enabled']}\n"
            communication_log += f"Autonomous mode: {agent_stats['system_health']['autonomous_mode']}\n"
            
            return (status_html, agent_data, communication_log)
            
        except Exception as e:
            logging.error(f"Error refreshing agent status: {e}")
            return ("<div class='status-indicator status-unhealthy'></div> Error", None, f"Error: {str(e)}")
    
    def _get_agent_statistics(self):
        """Get agent learning statistics"""
        try:
            # Create sample learning progress chart
            fig = go.Figure()
            
            # Sample data
            agents = ['Content Optimizer', 'Engagement Analyzer', 'Trend Predictor', 'Audience Targeter', 'Performance Monitor']
            success_rates = [0.85, 0.92, 0.78, 0.81, 0.89]
            
            fig.add_trace(go.Bar(
                x=agents,
                y=success_rates,
                name='Success Rate',
                marker_color='lightblue'
            ))
            
            fig.update_layout(
                title="Agent Learning Progress",
                xaxis_title="Agent Type",
                yaxis_title="Success Rate",
                yaxis_range=[0, 1]
            )
            
            return fig
            
        except Exception as e:
            logging.error(f"Error generating agent statistics: {e}")
            return None
    
    def _refresh_performance_metrics(self) -> tuple:
        """Refresh performance metrics"""
        if not self.is_initialized or not self.system:
            return (0.75, 8.5, 65, None, None)
        
        try:
            # Get performance stats
            perf_stats = self.system.performance_engine.get_system_stats()
            
            # Extract metrics
            cache_hit_rate = perf_stats['cache_stats']['hit_rate']
            memory_usage = perf_stats['memory_info']['process_rss_gb']
            gpu_usage = 0.0
            
            if perf_stats['gpu_info']:
                gpu_usage = (perf_stats['gpu_info']['used_gb'] / max(perf_stats['gpu_info']['total_gb'], 1)) * 100
            
            # Create response time chart
            response_fig = go.Figure()
            response_fig.add_trace(go.Scatter(
                x=list(range(len(self.request_history))),
                y=[req['processing_time'] for req in self.request_history],
                mode='lines+markers',
                name='Response Time'
            ))
            response_fig.update_layout(title="Response Time Trends", xaxis_title="Request", yaxis_title="Time (s)")
            
            # Create throughput chart
            throughput_fig = go.Figure()
            if self.request_history:
                # Calculate throughput (requests per minute)
                window_size = 10
                throughput_data = []
                for i in range(window_size, len(self.request_history)):
                    window_requests = self.request_history[i-window_size:i]
                    total_time = sum(req['processing_time'] for req in window_requests)
                    throughput = len(window_requests) / (total_time / 60) if total_time > 0 else 0
                    throughput_data.append(throughput)
                
                throughput_fig.add_trace(go.Scatter(
                    x=list(range(len(throughput_data))),
                    y=throughput_data,
                    mode='lines+markers',
                    name='Throughput'
                ))
                throughput_fig.update_layout(title="Throughput Trends", xaxis_title="Window", yaxis_title="Requests/Minute")
            
            return (cache_hit_rate, memory_usage, gpu_usage, response_fig, throughput_fig)
            
        except Exception as e:
            logging.error(f"Error refreshing performance metrics: {e}")
            return (0.75, 8.5, 65, None, None)
    
    def _refresh_system_health(self) -> tuple:
        """Refresh system health"""
        if not self.is_initialized or not self.system:
            return ("<div class='status-indicator status-unhealthy'></div> System Error", 0, "0 hours", 
                   "<p>System not initialized</p>", None, "System not initialized")
        
        try:
            # Get system status
            status = self.system.get_system_status()
            health_report = self.system.health_monitor.get_health_report()
            
            # Update overall health
            overall_status = health_report['current_status']['overall_status']
            status_class = f"status-{overall_status}"
            status_html = f"<div class='status-indicator {status_class}'></div> {overall_status.title()}"
            
            # Health score
            health_score = health_report['health_score']
            
            # Uptime
            uptime = f"{status['system_info']['uptime_hours']:.1f} hours"
            
            # Component status
            components = status['component_status']
            component_html = "<p>"
            for component, is_active in components.items():
                status_icon = "✅" if is_active else "❌"
                component_html += f"{status_icon} {component.replace('_', ' ').title()}<br>"
            component_html += "</p>"
            
            # Health history chart
            health_fig = go.Figure()
            if health_report['health_history']:
                timestamps = [datetime.fromtimestamp(check['timestamp']) for check in health_report['health_history']]
                scores = [check['overall_status'] for check in health_report['health_history']]
                
                # Convert status to numeric values
                score_values = []
                for score in scores:
                    if score == 'healthy':
                        score_values.append(100)
                    elif score == 'degraded':
                        score_values.append(50)
                    else:
                        score_values.append(0)
                
                health_fig.add_trace(go.Scatter(
                    x=timestamps,
                    y=score_values,
                    mode='lines+markers',
                    name='Health Score'
                ))
                health_fig.update_layout(title="System Health History", xaxis_title="Time", yaxis_title="Health Score")
            
            # Error log
            error_report = self.system.error_handler.get_error_report()
            if error_report['total_errors'] > 0:
                error_log = f"Total errors: {error_report['total_errors']}\n"
                error_log += f"Recovery success rate: {error_report['recovery_success_rate']:.2f}\n\n"
                
                for error in error_report['recent_errors'][:5]:
                    error_log += f"Error: {error['error_type']} - {error['error_message']}\n"
                    error_log += f"Component: {error['component']}\n"
                    error_log += f"Time: {datetime.fromtimestamp(error['timestamp'])}\n\n"
            else:
                error_log = "No errors recorded"
            
            return (status_html, health_score, uptime, component_html, health_fig, error_log)
            
        except Exception as e:
            logging.error(f"Error refreshing system health: {e}")
            return ("<div class='status-indicator status-unhealthy'></div> Error", 0, "0 hours", 
                   "<p>Error occurred</p>", None, f"Error: {str(e)}")
    
    def _generate_analytics(self, date_range: str, metric_selection: List[str]) -> tuple:
        """Generate analytics"""
        try:
            # Create sample analytics chart
            fig = go.Figure()
            
            # Sample data based on date range
            if "24 hours" in date_range:
                x_data = [f"{i}:00" for i in range(24)]
                y_data = [np.random.uniform(0.6, 0.9) for _ in range(24)]
            elif "7 days" in date_range:
                x_data = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
                y_data = [np.random.uniform(0.6, 0.9) for _ in range(7)]
            else:
                x_data = [f"Day {i+1}" for i in range(30)]
                y_data = [np.random.uniform(0.6, 0.9) for _ in range(30)]
            
            # Add traces for selected metrics
            colors = ['blue', 'red', 'green', 'orange', 'purple']
            for i, metric in enumerate(metric_selection):
                if i < len(colors):
                    fig.add_trace(go.Scatter(
                        x=x_data,
                        y=[y * np.random.uniform(0.8, 1.2) for y in y_data],
                        mode='lines+markers',
                        name=metric,
                        line=dict(color=colors[i])
                    ))
            
            fig.update_layout(
                title=f"Analytics for {date_range}",
                xaxis_title="Time",
                yaxis_title="Metric Value"
            )
            
            # Create summary statistics
            summary_data = []
            for metric in metric_selection:
                current = np.random.uniform(0.6, 0.9)
                average = np.random.uniform(0.6, 0.9)
                trend = "↗️ Increasing" if current > average else "↘️ Decreasing"
                
                summary_data.append([metric, f"{current:.3f}", f"{average:.3f}", trend])
            
            return (fig, summary_data)
            
        except Exception as e:
            logging.error(f"Error generating analytics: {e}")
            return (None, [])
    
    def _record_request(self, content: str, content_type: str, result: Dict[str, Any], processing_time: float):
        """Record a request for analytics"""
        self.request_history.append({
            'timestamp': time.time(),
            'content': content[:100],  # Truncate for storage
            'content_type': content_type,
            'result': result,
            'processing_time': processing_time
        })
        
        # Keep only last 1000 requests
        if len(self.request_history) > 1000:
            self.request_history = self.request_history[-1000:]
    
    def _parse_audience_size(self, audience_size: str) -> float:
        """Parse audience size string to numeric value"""
        if "Small" in audience_size:
            return 0.25
        elif "Medium" in audience_size:
            return 0.5
        elif "Large" in audience_size:
            return 0.75
        elif "Enterprise" in audience_size:
            return 1.0
        else:
            return 0.5
    
    def _parse_time_period(self, time_period: str) -> float:
        """Parse time period string to numeric value"""
        if "Morning" in time_period:
            return 0.25
        elif "Afternoon" in time_period:
            return 0.5
        elif "Evening" in time_period:
            return 0.75
        elif "Night" in time_period:
            return 0.0
        else:
            return 0.5
    
    def launch(self, **kwargs):
        """Launch the Gradio interface"""
        try:
            # Create and launch the interface
            interface = self.create_interface()
            return interface.launch(**kwargs)
        except Exception as e:
            logging.error(f"Failed to launch interface: {e}")
            raise


# Main function to launch the interface
def launch_enhanced_interface():
    """Launch the enhanced Gradio interface"""
    try:
        # Create interface
        interface = EnhancedGradioInterface()
        
        # Launch
        interface.create_interface().launch(
            server_name="0.0.0.0",
            server_port=7861,
            share=True,
            debug=True,
            show_error=True
        )
        
    except Exception as e:
        logging.error(f"Failed to launch enhanced interface: {e}")
        raise


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    print("🚀 Launching Enhanced Facebook Content Optimization Interface...")
    
    try:
        launch_enhanced_interface()
    except Exception as e:
        print(f"❌ Failed to launch interface: {e}")
        import traceback
        traceback.print_exc()
