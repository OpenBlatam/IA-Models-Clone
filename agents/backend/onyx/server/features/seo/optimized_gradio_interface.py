#!/usr/bin/env python3
"""
Optimized Gradio Interface for SEO Engine
Advanced UI with real-time monitoring and performance optimization
"""

import gradio as gr
import asyncio
import threading
import time
import json
import logging
import psutil
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings

# Local imports
from core_config import SEOConfig, get_config, get_container
from optimized_seo_engine import OptimizedSEOEngine, create_optimized_seo_engine
from advanced_monitoring import MonitoringSystem, MetricsVisualizer

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class OptimizedGradioInterface:
    """Advanced Gradio interface with real-time monitoring and optimization."""
    
    def __init__(self, config: Optional[SEOConfig] = None):
        self.config = config or SEOConfig()
        self.engine: Optional[OptimizedSEOEngine] = None
        self.monitoring: Optional[MonitoringSystem] = None
        self.visualizer: Optional[MetricsVisualizer] = None
        
        # Performance tracking
        self.interface_metrics = {
            'requests_processed': 0,
            'total_processing_time': 0,
            'cache_hits': 0,
            'errors': 0
        }
        
        # Initialize components
        self._initialize_engine()
        self._setup_interface()
        
        # Start background tasks
        self._start_background_tasks()
    
    def _initialize_engine(self) -> None:
        """Initialize the SEO engine."""
        try:
            self.engine = create_optimized_seo_engine()
            self.monitoring = self.engine.monitoring
            
            # Get container and register interface
            container = get_container()
            container.register('interface', self)
            
            logging.info("SEO Engine initialized successfully")
            
        except Exception as e:
            logging.error(f"Failed to initialize SEO engine: {e}")
            self.engine = None
    
    def _setup_interface(self) -> None:
        """Set up the Gradio interface components."""
        # Create interface components
        self._create_input_components()
        self._create_output_components()
        self._create_monitoring_components()
        self._create_control_components()
        
        # Build the interface
        self.interface = self._build_interface()
    
    def _create_input_components(self) -> None:
        """Create input components."""
        self.text_input = gr.Textbox(
            label="📝 Text to Analyze",
            placeholder="Enter your text here for SEO analysis...",
            lines=10,
            max_lines=20,
            scale=2
        )
        
        self.analysis_type = gr.Dropdown(
            choices=[
                "comprehensive",
                "keywords", 
                "content",
                "readability",
                "technical"
            ],
            value="comprehensive",
            label="🔍 Analysis Type",
            scale=1
        )
        
        self.batch_input = gr.File(
            file_count="multiple",
            label="📁 Batch Analysis (Upload multiple text files)",
            file_types=[".txt", ".md", ".html"],
            scale=1
        )
    
    def _create_output_components(self) -> None:
        """Create output components."""
        self.seo_score = gr.Number(
            label="🎯 Overall SEO Score",
            precision=1,
            scale=1
        )
        
        self.analysis_results = gr.JSON(
            label="📊 Detailed Analysis Results",
            scale=2
        )
        
        self.recommendations = gr.Markdown(
            label="💡 SEO Recommendations",
            scale=2
        )
        
        self.performance_metrics = gr.JSON(
            label="⚡ Performance Metrics",
            scale=1
        )
    
    def _create_monitoring_components(self) -> None:
        """Create monitoring and visualization components."""
        self.system_health = gr.JSON(
            label="🏥 System Health Status",
            scale=1
        )
        
        self.real_time_metrics = gr.Plot(
            label="📈 Real-time System Metrics",
            scale=2
        )
        
        self.cache_stats = gr.JSON(
            label="💾 Cache Statistics",
            scale=1
        )
        
        self.model_info = gr.JSON(
            label="🤖 Model Information",
            scale=1
        )
    
    def _create_control_components(self) -> None:
        """Create control and utility components."""
        self.analyze_btn = gr.Button(
            "🚀 Analyze Text",
            variant="primary",
            size="lg",
            scale=1
        )
        
        self.batch_analyze_btn = gr.Button(
            "📊 Batch Analyze",
            variant="secondary",
            size="lg",
            scale=1
        )
        
        self.optimize_btn = gr.Button(
            "⚡ Optimize Performance",
            variant="secondary",
            size="sm",
            scale=1
        )
        
        self.export_btn = gr.Button(
            "💾 Export Report",
            variant="secondary",
            size="sm",
            scale=1
        )
        
        self.clear_btn = gr.Button(
            "🗑️ Clear Results",
            variant="secondary",
            size="sm",
            scale=1
        )
    
    def _build_interface(self) -> gr.Blocks:
        """Build the complete Gradio interface."""
        with gr.Blocks(
            title="🚀 Optimized SEO Analysis Engine",
            theme=gr.themes.Soft(),
            css="""
            .gradio-container {
                max-width: 1400px !important;
            }
            .main-header {
                text-align: center;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border-radius: 10px;
                margin-bottom: 20px;
            }
            .metric-card {
                background: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 8px;
                padding: 15px;
                margin: 10px 0;
            }
            """
        ) as interface:
            
            # Header
            gr.HTML("""
            <div class="main-header">
                <h1>🚀 Optimized SEO Analysis Engine</h1>
                <p>Advanced SEO optimization with real-time monitoring and intelligent analysis</p>
            </div>
            """)
            
            with gr.Row():
                with gr.Column(scale=2):
                    # Input section
                    gr.HTML("<h3>📝 Input</h3>")
                    self.text_input
                    
                    with gr.Row():
                        self.analysis_type
                        self.analyze_btn
                    
                    gr.HTML("<h4>📁 Batch Analysis</h4>")
                    self.batch_input
                    self.batch_analyze_btn
                
                with gr.Column(scale=1):
                    # Control section
                    gr.HTML("<h3>🎛️ Controls</h3>")
                    with gr.Row():
                        self.optimize_btn
                        self.export_btn
                    self.clear_btn
            
            with gr.Row():
                with gr.Column(scale=1):
                    # Results section
                    gr.HTML("<h3>📊 Results</h3>")
                    self.seo_score
                    self.recommendations
                    self.performance_metrics
                
                with gr.Column(scale=2):
                    # Detailed analysis
                    gr.HTML("<h3>🔍 Detailed Analysis</h3>")
                    self.analysis_results
            
            with gr.Row():
                with gr.Column(scale=1):
                    # System monitoring
                    gr.HTML("<h3>🏥 System Health</h3>")
                    self.system_health
                    self.cache_stats
                    self.model_info
                
                with gr.Column(scale=2):
                    # Real-time metrics
                    gr.HTML("<h3>📈 Real-time Metrics</h3>")
                    self.real_time_metrics
            
            # Event handlers
            self._setup_event_handlers()
            
            # Auto-refresh monitoring
            interface.load(self._refresh_monitoring_data, outputs=[
                self.system_health, self.cache_stats, self.model_info
            ])
        
        return interface
    
    def _setup_event_handlers(self) -> None:
        """Set up event handlers for the interface."""
        # Main analysis
        self.analyze_btn.click(
            fn=self._analyze_text,
            inputs=[self.text_input, self.analysis_type],
            outputs=[self.seo_score, self.analysis_results, self.recommendations, self.performance_metrics]
        )
        
        # Batch analysis
        self.batch_analyze_btn.click(
            fn=self._batch_analyze,
            inputs=[self.batch_input, self.analysis_type],
            outputs=[self.analysis_results, self.performance_metrics]
        )
        
        # Performance optimization
        self.optimize_btn.click(
            fn=self._optimize_performance,
            outputs=[self.system_health, self.cache_stats]
        )
        
        # Export report
        self.export_btn.click(
            fn=self._export_report,
            outputs=[gr.HTML()]
        )
        
        # Clear results
        self.clear_btn.click(
            fn=self._clear_results,
            outputs=[self.text_input, self.seo_score, self.analysis_results, self.recommendations]
        )
    
    def _analyze_text(self, text: str, analysis_type: str) -> Tuple[float, Dict, str, Dict]:
        """Analyze text and return results."""
        if not text.strip():
            return 0.0, {}, "Please enter some text to analyze.", {}
        
        if not self.engine:
            return 0.0, {}, "❌ SEO Engine not available. Please check initialization.", {}
        
        try:
            start_time = time.time()
            
            # Perform analysis
            result = self.engine.analyze_text(text, analysis_type)
            
            # Update metrics
            processing_time = time.time() - start_time
            self._update_interface_metrics(processing_time, result.get('metadata', {}).get('cache_hit', False))
            
            # Format recommendations
            recommendations = self._format_recommendations(result.get('recommendations', []))
            
            # Get performance stats
            performance_stats = self.engine.seo_processor.get_performance_stats()
            
            return (
                result.get('seo_score', 0.0),
                result,
                recommendations,
                performance_stats
            )
            
        except Exception as e:
            logging.error(f"Text analysis failed: {e}")
            self.interface_metrics['errors'] += 1
            return 0.0, {'error': str(e)}, f"❌ Analysis failed: {str(e)}", {}
    
    def _batch_analyze(self, files: List[Any], analysis_type: str) -> Tuple[Dict, Dict]:
        """Analyze multiple files in batch."""
        if not files:
            return {}, {}
        
        if not self.engine:
            return {'error': 'SEO Engine not available'}, {}
        
        try:
            results = []
            total_time = 0
            
            for file in files:
                try:
                    # Read file content
                    content = file.decode('utf-8') if hasattr(file, 'decode') else str(file)
                    
                    # Analyze content
                    start_time = time.time()
                    result = self.engine.analyze_text(content, analysis_type)
                    analysis_time = time.time() - start_time
                    
                    # Add file info
                    result['file_info'] = {
                        'name': getattr(file, 'name', 'Unknown'),
                        'size': len(content),
                        'analysis_time': analysis_time
                    }
                    
                    results.append(result)
                    total_time += analysis_time
                    
                except Exception as e:
                    logging.error(f"File analysis failed: {e}")
                    results.append({
                        'error': str(e),
                        'file_info': {
                            'name': getattr(file, 'name', 'Unknown'),
                            'size': 0,
                            'analysis_time': 0
                        }
                    })
            
            # Update metrics
            self._update_interface_metrics(total_time, False)
            
            # Get performance stats
            performance_stats = self.engine.seo_processor.get_performance_stats()
            
            return {
                'batch_results': results,
                'summary': {
                    'total_files': len(files),
                    'successful_analyses': len([r for r in results if 'error' not in r]),
                    'total_processing_time': total_time,
                    'average_time_per_file': total_time / len(files) if files else 0
                }
            }, performance_stats
            
        except Exception as e:
            logging.error(f"Batch analysis failed: {e}")
            self.interface_metrics['errors'] += 1
            return {'error': str(e)}, {}
    
    def _optimize_performance(self) -> Tuple[Dict, Dict]:
        """Apply performance optimizations."""
        if not self.engine:
            return {'status': 'error', 'message': 'Engine not available'}, {}
        
        try:
            optimizations = self.engine.optimize_performance()
            
            # Get updated metrics
            system_health = self.engine.get_system_metrics()
            cache_stats = self.engine.seo_processor.cache.get_stats()
            
            return system_health['system_health'], cache_stats
            
        except Exception as e:
            logging.error(f"Performance optimization failed: {e}")
            return {'status': 'error', 'message': str(e)}, {}
    
    def _export_report(self) -> str:
        """Export analysis report."""
        if not self.engine:
            return "❌ SEO Engine not available"
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"seo_analysis_report_{timestamp}.json"
            
            self.engine.export_analysis_report(filename)
            
            return f"✅ Report exported successfully: {filename}"
            
        except Exception as e:
            logging.error(f"Report export failed: {e}")
            return f"❌ Export failed: {str(e)}"
    
    def _clear_results(self) -> Tuple[str, float, Dict, str]:
        """Clear all results."""
        return "", 0.0, {}, ""
    
    def _refresh_monitoring_data(self) -> Tuple[Dict, Dict, Dict]:
        """Refresh monitoring data."""
        if not self.engine:
            return {}, {}, {}
        
        try:
            system_metrics = self.engine.get_system_metrics()
            
            return (
                system_metrics['system_health'],
                system_metrics['cache_stats'],
                system_metrics['model_info']
            )
            
        except Exception as e:
            logging.error(f"Failed to refresh monitoring data: {e}")
            return {}, {}, {}
    
    def _update_interface_metrics(self, processing_time: float, cache_hit: bool) -> None:
        """Update interface performance metrics."""
        self.interface_metrics['requests_processed'] += 1
        self.interface_metrics['total_processing_time'] += processing_time
        
        if cache_hit:
            self.interface_metrics['cache_hits'] += 1
    
    def _format_recommendations(self, recommendations: List[str]) -> str:
        """Format recommendations for display."""
        if not recommendations:
            return "🎉 No specific recommendations at this time. Your content looks good!"
        
        formatted = "## 📋 SEO Recommendations\n\n"
        for i, rec in enumerate(recommendations, 1):
            formatted += f"{i}. {rec}\n\n"
        
        return formatted
    
    def _start_background_tasks(self) -> None:
        """Start background monitoring and visualization tasks."""
        if not self.monitoring:
            return
        
        # Start real-time metrics visualization
        def update_visualization():
            while True:
                try:
                    if self.monitoring and self.monitoring.visualizer:
                        self.monitoring.visualizer.update_plots()
                    time.sleep(5)  # Update every 5 seconds
                except Exception as e:
                    logging.error(f"Visualization update failed: {e}")
                    time.sleep(10)
        
        threading.Thread(target=update_visualization, daemon=True).start()
    
    def launch(self, **kwargs) -> None:
        """Launch the Gradio interface."""
        if not self.interface:
            raise RuntimeError("Interface not properly initialized")
        
        # Set default launch parameters
        default_kwargs = {
            'server_name': '0.0.0.0',
            'server_port': 7860,
            'share': False,
            'debug': False,
            'show_error': True,
            'height': 800,
            'width': '100%'
        }
        
        # Update with provided kwargs
        default_kwargs.update(kwargs)
        
        # Launch interface
        self.interface.launch(**default_kwargs)
    
    def get_interface_metrics(self) -> Dict[str, Any]:
        """Get interface performance metrics."""
        metrics = self.interface_metrics.copy()
        
        if metrics['requests_processed'] > 0:
            metrics['average_processing_time'] = (
                metrics['total_processing_time'] / metrics['requests_processed']
            )
            metrics['cache_hit_rate'] = (
                metrics['cache_hits'] / metrics['requests_processed']
            )
        else:
            metrics['average_processing_time'] = 0
            metrics['cache_hit_rate'] = 0
        
        return metrics

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_optimized_interface(config_path: Optional[str] = None) -> OptimizedGradioInterface:
    """Create and configure an optimized Gradio interface."""
    if config_path:
        config = SEOConfig.load_from_file(config_path)
    else:
        config = SEOConfig()
    
    return OptimizedGradioInterface(config)

def quick_launch(config_path: Optional[str] = None, **kwargs) -> None:
    """Quick launch with default configuration."""
    interface = create_optimized_interface(config_path)
    interface.launch(**kwargs)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Create and launch interface
    interface = create_optimized_interface()
    
    try:
        print("🚀 Launching Optimized SEO Analysis Interface...")
        print("📍 Interface will be available at: http://localhost:7860")
        print("📊 Real-time monitoring enabled")
        print("⚡ Performance optimizations active")
        
        interface.launch(
            server_name='0.0.0.0',
            server_port=7860,
            share=False,
            debug=False,
            show_error=True
        )
        
    except KeyboardInterrupt:
        print("\n🛑 Interface stopped by user")
    except Exception as e:
        print(f"❌ Failed to launch interface: {e}")
        logging.error(f"Interface launch failed: {e}")
    finally:
        if interface.engine:
            interface.engine.cleanup()
        print("🧹 Cleanup completed")


