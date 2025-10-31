#!/usr/bin/env python3
"""
Enhanced Gradio Interface for SEO Engine
Modern, user-friendly interface with advanced features and real-time monitoring
"""

import gradio as gr
import asyncio
import threading
import time
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any, Optional
import numpy as np
from pathlib import Path
import logging

# Import our enhanced engine
from enhanced_seo_engine import EnhancedSEOEngine, EnhancedSEOConfig

# ============================================================================
# CONFIGURATION
# ============================================================================

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_CONFIG = EnhancedSEOConfig(
    model_name="microsoft/DialoGPT-medium",
    enable_caching=True,
    enable_async=True,
    enable_profiling=True,
    batch_size=4,
    max_concurrent_requests=5,
    enable_logging=True,
    log_level="INFO"
)

# ============================================================================
# GLOBAL STATE MANAGEMENT
# ============================================================================

class GlobalState:
    """Global state management for the application."""
    
    def __init__(self):
        self.engine = None
        self.config = DEFAULT_CONFIG
        self.metrics_history = []
        self.processing_queue = []
        self.is_processing = False
        self.lock = threading.RLock()
    
    def initialize_engine(self, config: Optional[EnhancedSEOConfig] = None):
        """Initialize the SEO engine."""
        with self.lock:
            if self.engine is not None:
                self.engine.cleanup()
            
            self.config = config or DEFAULT_CONFIG
            self.engine = EnhancedSEOEngine(self.config)
            logger.info("SEO Engine initialized successfully")
    
    def get_engine(self) -> Optional[EnhancedSEOEngine]:
        """Get the current engine instance."""
        with self.lock:
            return self.engine
    
    def update_metrics_history(self, metrics: Dict[str, Any]):
        """Update metrics history."""
        with self.lock:
            self.metrics_history.append({
                'timestamp': time.time(),
                'metrics': metrics
            })
            
            # Keep only last 100 entries
            if len(self.metrics_history) > 100:
                self.metrics_history = self.metrics_history[-100:]
    
    def cleanup(self):
        """Cleanup resources."""
        with self.lock:
            if self.engine:
                self.engine.cleanup()
                self.engine = None

# Global state instance
global_state = GlobalState()

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def format_metrics(metrics: Dict[str, Any]) -> str:
    """Format metrics for display."""
    if not metrics:
        return "No metrics available"
    
    formatted = []
    
    # System info
    if 'system_info' in metrics:
        sys_info = metrics['system_info']
        formatted.append("## System Information")
        formatted.append(f"- CPU Count: {sys_info.get('cpu_count', 'N/A')}")
        formatted.append(f"- GPU Available: {sys_info.get('gpu_available', False)}")
        formatted.append(f"- GPU Count: {sys_info.get('gpu_count', 0)}")
        
        memory = sys_info.get('memory_usage', {})
        if memory:
            formatted.append(f"- Memory Usage: {memory.get('percent', 0):.1f}%")
            formatted.append(f"- Available Memory: {memory.get('available', 0) / (1024**3):.1f} GB")
    
    # Processor metrics
    if 'processor_metrics' in metrics:
        proc_metrics = metrics['processor_metrics']
        formatted.append("\n## Processing Metrics")
        
        counters = proc_metrics.get('counters', {})
        formatted.append(f"- Processed Texts: {counters.get('processed_texts', 0)}")
        formatted.append(f"- Cache Hits: {counters.get('cache_hits', 0)}")
        formatted.append(f"- Cache Misses: {counters.get('cache_misses', 0)}")
        formatted.append(f"- Processing Errors: {counters.get('processing_errors', 0)}")
        
        # Cache stats
        cache_stats = proc_metrics.get('cache_stats', {})
        formatted.append(f"- Cache Size: {cache_stats.get('size', 0)} / {cache_stats.get('max_size', 0)}")
    
    # Timing metrics
    if 'processor_metrics' in metrics:
        proc_metrics = metrics['processor_metrics']
        formatted.append("\n## Performance Metrics")
        
        for metric_name, stats in proc_metrics.items():
            if isinstance(stats, dict) and 'mean' in stats:
                formatted.append(f"- {metric_name}: {stats['mean']:.3f}s (avg), {stats['p95']:.3f}s (95th percentile)")
    
    return "\n".join(formatted)

def create_performance_chart(metrics_history: List[Dict[str, Any]]) -> go.Figure:
    """Create performance visualization chart."""
    if not metrics_history:
        # Return empty chart
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    # Extract timing data
    timestamps = []
    processing_times = []
    cache_hit_rates = []
    
    for entry in metrics_history:
        timestamps.append(entry['timestamp'])
        metrics = entry['metrics']
        
        # Get processing time
        proc_metrics = metrics.get('processor_metrics', {})
        seo_timings = proc_metrics.get('seo_processing_timings', {})
        if seo_timings and 'mean' in seo_timings:
            processing_times.append(seo_timings['mean'])
        else:
            processing_times.append(0)
        
        # Calculate cache hit rate
        counters = proc_metrics.get('counters', {})
        hits = counters.get('cache_hits', 0)
        misses = counters.get('cache_misses', 0)
        total = hits + misses
        if total > 0:
            cache_hit_rates.append(hits / total * 100)
        else:
            cache_hit_rates.append(0)
    
    # Create figure with secondary y-axis
    fig = go.Figure()
    
    # Add processing time line
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=processing_times,
        name="Processing Time (s)",
        line=dict(color='blue'),
        yaxis='y'
    ))
    
    # Add cache hit rate line
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=cache_hit_rates,
        name="Cache Hit Rate (%)",
        line=dict(color='red'),
        yaxis='y2'
    ))
    
    # Update layout
    fig.update_layout(
        title="Performance Metrics Over Time",
        xaxis_title="Time",
        yaxis=dict(title="Processing Time (s)", side="left"),
        yaxis2=dict(title="Cache Hit Rate (%)", side="right", overlaying="y"),
        hovermode='x unified',
        showlegend=True
    )
    
    return fig

def create_seo_score_distribution(results: List[Dict[str, Any]]) -> go.Figure:
    """Create SEO score distribution chart."""
    if not results:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    # Extract SEO scores
    seo_scores = [result.get('seo_score', 0) for result in results if isinstance(result, dict)]
    
    if not seo_scores:
        fig = go.Figure()
        fig.add_annotation(
            text="No valid SEO scores",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    # Create histogram
    fig = go.Figure(data=[go.Histogram(x=seo_scores, nbinsx=20)])
    
    fig.update_layout(
        title="SEO Score Distribution",
        xaxis_title="SEO Score",
        yaxis_title="Frequency",
        showlegend=False
    )
    
    return fig

# ============================================================================
# CORE PROCESSING FUNCTIONS
# ============================================================================

def analyze_single_text(text: str, progress=gr.Progress()) -> Dict[str, Any]:
    """Analyze a single text for SEO optimization."""
    try:
        engine = global_state.get_engine()
        if not engine:
            return {"error": "Engine not initialized"}
        
        progress(0.2, desc="Validating input...")
        
        # Analyze text
        progress(0.5, desc="Analyzing SEO metrics...")
        result = engine.analyze_text(text)
        
        progress(0.8, desc="Updating metrics...")
        
        # Update metrics
        metrics = engine.get_system_metrics()
        global_state.update_metrics_history(metrics)
        
        progress(1.0, desc="Analysis complete!")
        
        return result
        
    except Exception as e:
        logger.error(f"Error in single text analysis: {str(e)}")
        return {"error": str(e)}

def analyze_multiple_texts(texts: str, progress=gr.Progress()) -> Dict[str, Any]:
    """Analyze multiple texts for SEO optimization."""
    try:
        engine = global_state.get_engine()
        if not engine:
            return {"error": "Engine not initialized"}
        
        # Parse texts (assuming one per line)
        text_list = [text.strip() for text in texts.split('\n') if text.strip()]
        
        if not text_list:
            return {"error": "No valid texts provided"}
        
        progress(0.1, desc=f"Processing {len(text_list)} texts...")
        
        # Analyze texts
        progress(0.3, desc="Analyzing SEO metrics...")
        results = engine.analyze_texts(text_list)
        
        progress(0.7, desc="Updating metrics...")
        
        # Update metrics
        metrics = engine.get_system_metrics()
        global_state.update_metrics_history(metrics)
        
        progress(0.9, desc="Creating summary...")
        
        # Create summary
        summary = {
            "total_texts": len(text_list),
            "successful_analyses": len([r for r in results if 'error' not in r]),
            "failed_analyses": len([r for r in results if 'error' in r]),
            "average_seo_score": np.mean([r.get('seo_score', 0) for r in results if 'error' not in r]),
            "results": results
        }
        
        progress(1.0, desc="Analysis complete!")
        
        return summary
        
    except Exception as e:
        logger.error(f"Error in multiple text analysis: {str(e)}")
        return {"error": str(e)}

async def analyze_text_async(text: str, progress=gr.Progress()) -> Dict[str, Any]:
    """Async text analysis."""
    try:
        engine = global_state.get_engine()
        if not engine:
            return {"error": "Engine not initialized"}
        
        progress(0.2, desc="Starting async analysis...")
        
        # Analyze text asynchronously
        progress(0.5, desc="Analyzing SEO metrics...")
        result = await engine.analyze_text_async(text)
        
        progress(0.8, desc="Updating metrics...")
        
        # Update metrics
        metrics = engine.get_system_metrics()
        global_state.update_metrics_history(metrics)
        
        progress(1.0, desc="Async analysis complete!")
        
        return result
        
    except Exception as e:
        logger.error(f"Error in async text analysis: {str(e)}")
        return {"error": str(e)}

def get_system_metrics() -> str:
    """Get current system metrics."""
    try:
        engine = global_state.get_engine()
        if not engine:
            return "Engine not initialized"
        
        metrics = engine.get_system_metrics()
        return format_metrics(metrics)
        
    except Exception as e:
        logger.error(f"Error getting metrics: {str(e)}")
        return f"Error retrieving metrics: {str(e)}"

def get_performance_chart() -> go.Figure:
    """Get performance visualization."""
    try:
        return create_performance_chart(global_state.metrics_history)
    except Exception as e:
        logger.error(f"Error creating performance chart: {str(e)}")
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating chart: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

def initialize_engine(model_name: str, enable_caching: bool, enable_async: bool, 
                     batch_size: int, max_concurrent: int) -> str:
    """Initialize the SEO engine with custom configuration."""
    try:
        config = EnhancedSEOConfig(
            model_name=model_name,
            enable_caching=enable_caching,
            enable_async=enable_async,
            batch_size=batch_size,
            max_concurrent_requests=max_concurrent,
            enable_logging=True,
            log_level="INFO"
        )
        
        global_state.initialize_engine(config)
        return "Engine initialized successfully!"
        
    except Exception as e:
        logger.error(f"Error initializing engine: {str(e)}")
        return f"Error initializing engine: {str(e)}"

# ============================================================================
# GRADIO INTERFACE
# ============================================================================

def create_interface():
    """Create the enhanced Gradio interface."""
    
    # Initialize engine on startup
    global_state.initialize_engine()
    
    with gr.Blocks(
        title="Enhanced SEO Engine",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
        }
        """
    ) as interface:
        
        gr.Markdown("""
        # 🚀 Enhanced SEO Engine
        
        Advanced SEO optimization system with real-time monitoring, caching, and performance analytics.
        
        ---
        """)
        
        with gr.Tabs():
            
            # ========================================================================
            # ANALYSIS TAB
            # ========================================================================
            
            with gr.Tab("📊 SEO Analysis"):
                
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("### Single Text Analysis")
                        
                        single_text_input = gr.Textbox(
                            label="Enter text for SEO analysis",
                            placeholder="Paste your text here...",
                            lines=8,
                            max_lines=20
                        )
                        
                        with gr.Row():
                            analyze_single_btn = gr.Button("🔍 Analyze Text", variant="primary")
                            analyze_async_btn = gr.Button("⚡ Async Analysis", variant="secondary")
                        
                        single_result = gr.JSON(label="Analysis Results")
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### Quick Stats")
                        quick_stats = gr.Markdown("No analysis performed yet")
                
                gr.Markdown("---")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("### Batch Analysis")
                        
                        batch_text_input = gr.Textbox(
                            label="Enter multiple texts (one per line)",
                            placeholder="Text 1\nText 2\nText 3\n...",
                            lines=10,
                            max_lines=30
                        )
                        
                        analyze_batch_btn = gr.Button("📦 Analyze Batch", variant="primary")
                        
                        batch_result = gr.JSON(label="Batch Analysis Results")
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### Batch Summary")
                        batch_summary = gr.Markdown("No batch analysis performed yet")
            
            # ========================================================================
            # MONITORING TAB
            # ========================================================================
            
            with gr.Tab("📈 Monitoring"):
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### System Metrics")
                        
                        refresh_metrics_btn = gr.Button("🔄 Refresh Metrics", variant="primary")
                        metrics_display = gr.Markdown("Click 'Refresh Metrics' to view current system status")
                    
                    with gr.Column(scale=2):
                        gr.Markdown("### Performance Chart")
                        
                        refresh_chart_btn = gr.Button("📊 Update Chart", variant="secondary")
                        performance_chart = gr.Plot(label="Performance Over Time")
                
                gr.Markdown("---")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### SEO Score Distribution")
                        score_distribution = gr.Plot(label="SEO Score Distribution")
                    
                    with gr.Column():
                        gr.Markdown("### Recent Activity")
                        recent_activity = gr.Markdown("No recent activity")
            
            # ========================================================================
            # CONFIGURATION TAB
            # ========================================================================
            
            with gr.Tab("⚙️ Configuration"):
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Engine Configuration")
                        
                        model_name_input = gr.Dropdown(
                            choices=[
                                "microsoft/DialoGPT-medium",
                                "microsoft/DialoGPT-large",
                                "gpt2",
                                "distilgpt2"
                            ],
                            value="microsoft/DialoGPT-medium",
                            label="Model Name"
                        )
                        
                        enable_caching = gr.Checkbox(
                            label="Enable Caching",
                            value=True
                        )
                        
                        enable_async = gr.Checkbox(
                            label="Enable Async Processing",
                            value=True
                        )
                        
                        batch_size = gr.Slider(
                            minimum=1,
                            maximum=16,
                            value=4,
                            step=1,
                            label="Batch Size"
                        )
                        
                        max_concurrent = gr.Slider(
                            minimum=1,
                            maximum=20,
                            value=5,
                            step=1,
                            label="Max Concurrent Requests"
                        )
                        
                        init_engine_btn = gr.Button("🚀 Initialize Engine", variant="primary")
                        init_status = gr.Markdown("Engine ready")
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### System Information")
                        
                        system_info = gr.Markdown("Loading system information...")
                        
                        cleanup_btn = gr.Button("🧹 Cleanup Resources", variant="secondary")
                        cleanup_status = gr.Markdown("")
        
        # ========================================================================
        # EVENT HANDLERS
        # ========================================================================
        
        # Single text analysis
        analyze_single_btn.click(
            fn=analyze_single_text,
            inputs=[single_text_input],
            outputs=[single_result]
        )
        
        analyze_async_btn.click(
            fn=analyze_text_async,
            inputs=[single_text_input],
            outputs=[single_result]
        )
        
        # Batch analysis
        analyze_batch_btn.click(
            fn=analyze_multiple_texts,
            inputs=[batch_text_input],
            outputs=[batch_result]
        )
        
        # Metrics and monitoring
        refresh_metrics_btn.click(
            fn=get_system_metrics,
            inputs=[],
            outputs=[metrics_display]
        )
        
        refresh_chart_btn.click(
            fn=get_performance_chart,
            inputs=[],
            outputs=[performance_chart]
        )
        
        # Configuration
        init_engine_btn.click(
            fn=initialize_engine,
            inputs=[model_name_input, enable_caching, enable_async, batch_size, max_concurrent],
            outputs=[init_status]
        )
        
        cleanup_btn.click(
            fn=lambda: global_state.cleanup(),
            inputs=[],
            outputs=[cleanup_status]
        )
        
        # Auto-refresh metrics every 30 seconds
        interface.load(
            fn=get_system_metrics,
            outputs=[metrics_display],
            every=30
        )
        
        # Update performance chart every 60 seconds
        interface.load(
            fn=get_performance_chart,
            outputs=[performance_chart],
            every=60
        )
    
    return interface

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main function to launch the interface."""
    try:
        interface = create_interface()
        
        # Launch with enhanced settings
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            debug=True,
            show_error=True,
            quiet=False
        )
        
    except Exception as e:
        logger.error(f"Error launching interface: {str(e)}")
        print(f"Error: {e}")
    
    finally:
        # Cleanup on exit
        global_state.cleanup()

if __name__ == "__main__":
    main()
