#!/usr/bin/env python3
"""
Advanced Error Handling and Debugging Tools for Gradio Apps
Comprehensive debugging with real-time monitoring, error tracking, and troubleshooting
"""

import gradio as gr
import torch
import numpy as np
import pandas as pd
import logging
import traceback
import sys
import os
import time
import psutil
import gc
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from pathlib import Path
from datetime import datetime, timedelta
import json
import threading
import queue
import warnings
from contextlib import contextmanager
import inspect
import functools

# Setup advanced logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler('gradio_debug.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class GradioDebugger:
    """Advanced debugging and error handling for Gradio applications."""
    
    def __init__(self, enable_monitoring: bool = True):
        self.debug_log = []
        self.error_tracker = {}
        self.performance_metrics = {}
        self.memory_snapshots = []
        self.enable_monitoring = enable_monitoring
        self.monitoring_thread = None
        self.monitoring_queue = queue.Queue()
        self.debug_mode = False
        self.breakpoints = set()
        
        if enable_monitoring:
            self._start_monitoring()
    
    def _start_monitoring(self):
        """Start background monitoring thread."""
        self.monitoring_thread = threading.Thread(target=self._monitor_system, daemon=True)
        self.monitoring_thread.start()
        logger.info("System monitoring started")
    
    def _monitor_system(self):
        """Background system monitoring."""
        while self.enable_monitoring:
            try:
                # System metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                # GPU metrics if available
                gpu_metrics = {}
                if torch.cuda.is_available():
                    gpu_metrics = {
                        'gpu_memory_used': torch.cuda.memory_allocated() / 1024**3,  # GB
                        'gpu_memory_cached': torch.cuda.memory_reserved() / 1024**3,  # GB
                        'gpu_utilization': 0  # Would need nvidia-ml-py3 for this
                    }
                
                # Store metrics
                timestamp = datetime.now()
                self.performance_metrics[timestamp] = {
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'memory_available_gb': memory.available / 1024**3,
                    'disk_percent': disk.percent,
                    'gpu_metrics': gpu_metrics
                }
                
                # Memory snapshot
                if len(self.memory_snapshots) < 100:  # Keep last 100 snapshots
                    self.memory_snapshots.append({
                        'timestamp': timestamp,
                        'memory_usage': dict(memory._asdict()),
                        'gpu_metrics': gpu_metrics
                    })
                
                time.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(10)
    
    def enable_debug_mode(self, enabled: bool = True):
        """Enable or disable debug mode."""
        self.debug_mode = enabled
        if enabled:
            logging.getLogger().setLevel(logging.DEBUG)
            logger.info("Debug mode enabled")
        else:
            logging.getLogger().setLevel(logging.INFO)
            logger.info("Debug mode disabled")
    
    def add_breakpoint(self, function_name: str, condition: Optional[Callable] = None):
        """Add a breakpoint for a specific function."""
        self.breakpoints.add((function_name, condition))
        logger.info(f"Breakpoint added for function: {function_name}")
    
    def remove_breakpoint(self, function_name: str):
        """Remove a breakpoint."""
        self.breakpoints = {bp for bp in self.breakpoints if bp[0] != function_name}
        logger.info(f"Breakpoint removed for function: {function_name}")
    
    def debug_decorator(self, func: Callable) -> Callable:
        """Decorator to add debugging capabilities to functions."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            function_name = func.__name__
            
            # Check breakpoints
            for bp_name, bp_condition in self.breakpoints:
                if bp_name == function_name:
                    if bp_condition is None or bp_condition(*args, **kwargs):
                        logger.debug(f"BREAKPOINT HIT: {function_name}")
                        # In a real debugger, this would pause execution
                        break
            
            try:
                # Log function call
                logger.debug(f"Calling {function_name} with args: {args}, kwargs: {kwargs}")
                
                # Execute function
                result = func(*args, **kwargs)
                
                # Log success
                execution_time = time.time() - start_time
                logger.debug(f"{function_name} completed successfully in {execution_time:.4f}s")
                
                # Store performance metric
                self._store_performance_metric(function_name, execution_time, True)
                
                return result
                
            except Exception as e:
                # Log error
                execution_time = time.time() - start_time
                error_info = self._capture_error_info(e, function_name, args, kwargs)
                
                logger.error(f"{function_name} failed after {execution_time:.4f}s: {e}")
                logger.error(f"Error details: {error_info}")
                
                # Store error tracking
                self._store_error_info(function_name, e, error_info, execution_time)
                
                # Store performance metric
                self._store_performance_metric(function_name, execution_time, False)
                
                # Re-raise if in debug mode
                if self.debug_mode:
                    raise
                else:
                    return self._create_error_response(e, error_info)
        
        return wrapper
    
    def _capture_error_info(self, error: Exception, function_name: str, args: tuple, kwargs: dict) -> Dict:
        """Capture comprehensive error information."""
        return {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback.format_exc(),
            'function_name': function_name,
            'args': str(args),
            'kwargs': str(kwargs),
            'timestamp': datetime.now(),
            'system_info': self._get_system_info(),
            'memory_info': self._get_memory_info(),
            'gpu_info': self._get_gpu_info() if torch.cuda.is_available() else None
        }
    
    def _get_system_info(self) -> Dict:
        """Get current system information."""
        try:
            return {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': psutil.disk_usage('/').percent,
                'python_version': sys.version,
                'platform': sys.platform,
                'process_id': os.getpid()
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _get_memory_info(self) -> Dict:
        """Get detailed memory information."""
        try:
            memory = psutil.virtual_memory()
            return {
                'total_gb': memory.total / 1024**3,
                'available_gb': memory.available / 1024**3,
                'used_gb': memory.used / 1024**3,
                'percent': memory.percent
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _get_gpu_info(self) -> Dict:
        """Get GPU information if available."""
        try:
            if torch.cuda.is_available():
                return {
                    'device_count': torch.cuda.device_count(),
                    'current_device': torch.cuda.current_device(),
                    'device_name': torch.cuda.get_device_name(),
                    'memory_allocated_gb': torch.cuda.memory_allocated() / 1024**3,
                    'memory_reserved_gb': torch.cuda.memory_reserved() / 1024**3
                }
            return None
        except Exception as e:
            return {'error': str(e)}
    
    def _store_error_info(self, function_name: str, error: Exception, error_info: Dict, execution_time: float):
        """Store error information for analysis."""
        if function_name not in self.error_tracker:
            self.error_tracker[function_name] = []
        
        self.error_tracker[function_name].append({
            'timestamp': datetime.now(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'execution_time': execution_time,
            'error_info': error_info
        })
        
        # Keep only last 50 errors per function
        if len(self.error_tracker[function_name]) > 50:
            self.error_tracker[function_name] = self.error_tracker[function_name][-50:]
    
    def _store_performance_metric(self, function_name: str, execution_time: float, success: bool):
        """Store performance metrics."""
        if function_name not in self.performance_metrics:
            self.performance_metrics[function_name] = []
        
        self.performance_metrics[function_name].append({
            'timestamp': datetime.now(),
            'execution_time': execution_time,
            'success': success
        })
        
        # Keep only last 100 metrics per function
        if len(self.performance_metrics[function_name]) > 100:
            self.performance_metrics[function_name] = self.performance_metrics[function_name][-100:]
    
    def _create_error_response(self, error: Exception, error_info: Dict) -> str:
        """Create user-friendly error response."""
        error_type = type(error).__name__
        
        if "CUDA out of memory" in str(error):
            return f"❌ **GPU Memory Error**\n\nGPU memory is insufficient. Try reducing input size or batch size."
        elif "timeout" in str(error).lower():
            return f"⏰ **Timeout Error**\n\nOperation took too long. Try with smaller inputs or check system performance."
        elif "connection" in str(error).lower():
            return f"🌐 **Connection Error**\n\nNetwork connection issue. Please check your internet connection."
        else:
            return f"💥 **{error_type}**\n\nAn unexpected error occurred: {str(error)}\n\n**Debug Info**: {error_info.get('system_info', {})}"
    
    def get_error_summary(self) -> str:
        """Get comprehensive error summary."""
        if not self.error_tracker:
            return "✅ No errors recorded."
        
        summary = "📊 **Error Summary**\n\n"
        
        for function_name, errors in self.error_tracker.items():
            error_count = len(errors)
            success_rate = self._calculate_success_rate(function_name)
            
            summary += f"**{function_name}**\n"
            summary += f"• Total Errors: {error_count}\n"
            summary += f"• Success Rate: {success_rate:.1f}%\n"
            
            # Most common error
            if errors:
                error_types = [e['error_type'] for e in errors]
                most_common = max(set(error_types), key=error_types.count)
                summary += f"• Most Common Error: {most_common}\n"
            
            summary += "\n"
        
        return summary
    
    def _calculate_success_rate(self, function_name: str) -> float:
        """Calculate success rate for a function."""
        if function_name not in self.performance_metrics:
            return 100.0
        
        metrics = self.performance_metrics[function_name]
        if not metrics:
            return 100.0
        
        successful = sum(1 for m in metrics if m['success'])
        return (successful / len(metrics)) * 100
    
    def get_performance_report(self) -> str:
        """Get performance analysis report."""
        if not self.performance_metrics:
            return "📊 No performance data available."
        
        report = "📈 **Performance Report**\n\n"
        
        for function_name, metrics in self.performance_metrics.items():
            if not metrics:
                continue
            
            execution_times = [m['execution_time'] for m in metrics]
            avg_time = np.mean(execution_times)
            max_time = max(execution_times)
            min_time = min(execution_times)
            
            report += f"**{function_name}**\n"
            report += f"• Average Time: {avg_time:.4f}s\n"
            report += f"• Fastest: {min_time:.4f}s\n"
            report += f"• Slowest: {max_time:.4f}s\n"
            report += f"• Total Calls: {len(metrics)}\n\n"
        
        return report
    
    def get_system_health(self) -> str:
        """Get current system health status."""
        try:
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            health = "🏥 **System Health Status**\n\n"
            
            # CPU Status
            if cpu_percent < 50:
                health += f"🟢 **CPU**: {cpu_percent:.1f}% (Healthy)\n"
            elif cpu_percent < 80:
                health += f"🟡 **CPU**: {cpu_percent:.1f}% (Moderate)\n"
            else:
                health += f"🔴 **CPU**: {cpu_percent:.1f}% (High Load)\n"
            
            # Memory Status
            if memory.percent < 70:
                health += f"🟢 **Memory**: {memory.percent:.1f}% (Healthy)\n"
            elif memory.percent < 90:
                health += f"🟡 **Memory**: {memory.percent:.1f}% (Moderate)\n"
            else:
                health += f"🔴 **Memory**: {memory.percent:.1f}% (Critical)\n"
            
            # Disk Status
            if disk.percent < 80:
                health += f"🟢 **Disk**: {disk.percent:.1f}% (Healthy)\n"
            elif disk.percent < 95:
                health += f"🟡 **Disk**: {disk.percent:.1f}% (Moderate)\n"
            else:
                health += f"🔴 **Disk**: {disk.percent:.1f}% (Critical)\n"
            
            # GPU Status
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1024**3
                gpu_reserved = torch.cuda.memory_reserved() / 1024**3
                health += f"🎮 **GPU Memory**: {gpu_memory:.2f}GB used, {gpu_reserved:.2f}GB reserved\n"
            
            return health
            
        except Exception as e:
            return f"❌ **System Health Check Failed**: {e}"
    
    def clear_debug_data(self):
        """Clear all debug data."""
        self.debug_log.clear()
        self.error_tracker.clear()
        self.performance_metrics.clear()
        self.memory_snapshots.clear()
        logger.info("Debug data cleared")
    
    def export_debug_data(self, filepath: str):
        """Export debug data to JSON file."""
        try:
            export_data = {
                'timestamp': datetime.now().isoformat(),
                'error_tracker': self.error_tracker,
                'performance_metrics': self.performance_metrics,
                'system_info': self._get_system_info()
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Debug data exported to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export debug data: {e}")
            return False

class GradioErrorHandler:
    """Enhanced error handler with debugging capabilities."""
    
    def __init__(self, debugger: Optional[GradioDebugger] = None):
        self.debugger = debugger or GradioDebugger()
        self.error_log = []
        self.recovery_strategies = self._setup_recovery_strategies()
    
    def _setup_recovery_strategies(self) -> Dict[str, List[str]]:
        """Setup recovery strategies for different error types."""
        return {
            'memory_error': [
                "💡 **Reduce input size** - Try with shorter text or smaller batch size",
                "💡 **Close other applications** - Free up system memory",
                "💡 **Restart the interface** - Clear memory cache",
                "💡 **Use CPU mode** - Switch to CPU if GPU memory is insufficient",
                "💡 **Garbage collection** - Force memory cleanup"
            ],
            'validation_error': [
                "💡 **Check input format** - Ensure input meets requirements",
                "💡 **Remove special characters** - Avoid unsupported symbols",
                "💡 **Adjust input length** - Stay within size limits",
                "💡 **Verify file type** - Use supported file formats",
                "💡 **Check input encoding** - Ensure proper text encoding"
            ],
            'model_error': [
                "💡 **Reinitialize model** - Click the initialize button again",
                "💡 **Check model status** - Verify model is ready",
                "💡 **Try different parameters** - Adjust batch size or model type",
                "💡 **Restart interface** - Fresh start often helps",
                "💡 **Check model files** - Verify model weights are intact"
            ],
            'system_error': [
                "💡 **Refresh the page** - Reload the interface",
                "💡 **Check system resources** - Ensure sufficient memory/CPU",
                "💡 **Try again later** - System might be temporarily busy",
                "💡 **Contact support** - If problem persists",
                "💡 **Check system logs** - Review error logs for details"
            ]
        }
    
    def handle_error(self, error: Exception, context: str = "", enable_recovery: bool = True) -> str:
        """Handle errors with debugging and recovery options."""
        error_type = type(error).__name__
        timestamp = datetime.now()
        
        # Log error
        error_info = {
            'timestamp': timestamp,
            'error_type': error_type,
            'error_message': str(error),
            'context': context,
            'traceback': traceback.format_exc()
        }
        
        self.error_log.append(error_info)
        logger.error(f"Error in {context}: {error}")
        
        # Create error response
        if self.debugger.debug_mode:
            response = self._create_debug_error_response(error, error_info)
        else:
            response = self._create_user_error_response(error, error_info)
        
        # Add recovery suggestions
        if enable_recovery:
            recovery = self._get_recovery_suggestions(error_type)
            if recovery:
                response += f"\n\n🔄 **Recovery Suggestions**\n{recovery}"
        
        return response
    
    def _create_debug_error_response(self, error: Exception, error_info: Dict) -> str:
        """Create detailed error response for debug mode."""
        return f"""🚨 **DEBUG ERROR REPORT**

**Error Type**: {error_info['error_type']}
**Error Message**: {error_info['error_message']}
**Context**: {error_info['context']}
**Timestamp**: {error_info['timestamp']}

**Full Traceback**:
```
{error_info['traceback']}
```

**System Info**: {self.debugger._get_system_info()}
**Memory Info**: {self.debugger._get_memory_info()}
**GPU Info**: {self.debugger._get_gpu_info() if torch.cuda.is_available() else 'N/A'}"""
    
    def _create_user_error_response(self, error: Exception, error_info: Dict) -> str:
        """Create user-friendly error response."""
        error_type = error_info['error_type']
        
        if "CUDA out of memory" in str(error):
            return f"💾 **Memory Error**\n\nGPU memory is insufficient for this operation.\n**Context**: {error_info['context']}"
        elif "timeout" in str(error).lower():
            return f"⏰ **Timeout Error**\n\nOperation took too long to complete.\n**Context**: {error_info['context']}"
        elif "connection" in str(error).lower():
            return f"🌐 **Connection Error**\n\nNetwork connection issue detected.\n**Context**: {error_info['context']}"
        else:
            return f"💥 **{error_type}**\n\nAn unexpected error occurred.\n**Context**: {error_info['context']}\n**Message**: {error_info['error_message']}"
    
    def _get_recovery_suggestions(self, error_type: str) -> str:
        """Get recovery suggestions for error type."""
        suggestions = self.recovery_strategies.get(error_type, self.recovery_strategies['system_error'])
        return "\n".join(suggestions)
    
    def get_error_analysis(self) -> str:
        """Get comprehensive error analysis."""
        if not self.error_log:
            return "✅ No errors recorded."
        
        # Analyze error patterns
        error_types = [e['error_type'] for e in self.error_log]
        error_counts = {}
        for error_type in error_types:
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        # Most common errors
        most_common = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        analysis = "🔍 **Error Analysis Report**\n\n"
        analysis += f"**Total Errors**: {len(self.error_log)}\n"
        analysis += f"**Unique Error Types**: {len(error_counts)}\n\n"
        
        analysis += "**Most Common Errors**:\n"
        for error_type, count in most_common:
            percentage = (count / len(self.error_log)) * 100
            analysis += f"• {error_type}: {count} times ({percentage:.1f}%)\n"
        
        # Recent errors
        recent_errors = self.error_log[-5:]
        analysis += f"\n**Recent Errors** (Last 5):\n"
        for error in recent_errors:
            analysis += f"• {error['timestamp'].strftime('%H:%M:%S')} - {error['error_type']}: {error['context']}\n"
        
        return analysis

class GradioDebugInterface:
    """Gradio interface for debugging and error handling."""
    
    def __init__(self, debugger: Optional[GradioDebugger] = None, error_handler: Optional[GradioErrorHandler] = None):
        self.debugger = debugger or GradioDebugger()
        self.error_handler = error_handler or GradioErrorHandler(self.debugger)
    
    def create_debug_interface(self) -> gr.Blocks:
        """Create the debugging interface."""
        with gr.Blocks(title="Gradio Debug & Error Handling Tools", theme=gr.themes.Soft()) as demo:
            gr.HTML("""
            <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 15px; margin-bottom: 20px;">
                <h1>🔧 Gradio Debug & Error Handling Tools</h1>
                <p>Advanced debugging, error tracking, and system monitoring</p>
            </div>
            """)
            
            with gr.Tabs():
                # System Monitoring Tab
                with gr.Tab("📊 System Monitoring"):
                    gr.Markdown("### Real-time System Health Monitoring")
                    
                    with gr.Row():
                        with gr.Column():
                            refresh_health_btn = gr.Button("🔄 Refresh System Health", variant="primary")
                            start_monitoring_btn = gr.Button("▶️ Start Monitoring", variant="secondary")
                            stop_monitoring_btn = gr.Button("⏹️ Stop Monitoring", variant="secondary")
                        
                        with gr.Column():
                            system_health_display = gr.Markdown("**System Health**\n\nClick 'Refresh System Health' to view current status.")
                    
                    # Performance metrics
                    gr.Markdown("### Performance Metrics")
                    performance_report_display = gr.Markdown("**Performance Report**\n\nClick refresh to view performance data.")
                
                # Error Tracking Tab
                with gr.Tab("🚨 Error Tracking"):
                    gr.Markdown("### Error Analysis and Recovery")
                    
                    with gr.Row():
                        with gr.Column():
                            refresh_errors_btn = gr.Button("🔄 Refresh Error Summary", variant="primary")
                            error_analysis_btn = gr.Button("🔍 Error Analysis", variant="secondary")
                            clear_errors_btn = gr.Button("🗑️ Clear Errors", variant="secondary")
                        
                        with gr.Column():
                            error_summary_display = gr.Markdown("**Error Summary**\n\nClick refresh to view error data.")
                    
                    # Error analysis
                    error_analysis_display = gr.Markdown("**Error Analysis**\n\nClick 'Error Analysis' for detailed insights.")
                
                # Debug Tools Tab
                with gr.Tab("🔧 Debug Tools"):
                    gr.Markdown("### Advanced Debugging Tools")
                    
                    with gr.Row():
                        with gr.Column():
                            debug_mode_toggle = gr.Checkbox(
                                label="Enable Debug Mode",
                                value=False,
                                info="Enable detailed error reporting and debugging"
                            )
                            
                            add_breakpoint_btn = gr.Button("📍 Add Breakpoint", variant="primary")
                            remove_breakpoint_btn = gr.Button("❌ Remove Breakpoint", variant="secondary")
                            
                            function_name_input = gr.Textbox(
                                label="Function Name",
                                placeholder="Enter function name for breakpoint",
                                info="Function to add/remove breakpoint from"
                            )
                        
                        with gr.Column():
                            debug_status_display = gr.Markdown("**Debug Status**\n\nConfigure debug settings above.")
                    
                    # Data export
                    gr.Markdown("### Data Export")
                    export_btn = gr.Button("📤 Export Debug Data", variant="primary")
                    export_status_display = gr.Markdown("**Export Status**\n\nClick export to save debug data.")
                
                # Testing Tab
                with gr.Tab("🧪 Testing"):
                    gr.Markdown("### Test Error Handling and Debugging")
                    
                    with gr.Row():
                        with gr.Column():
                            test_error_type = gr.Radio(
                                choices=["memory_error", "validation_error", "model_error", "system_error"],
                                label="Error Type to Test",
                                value="memory_error"
                            )
                            
                            test_context = gr.Textbox(
                                label="Test Context",
                                placeholder="Enter test context",
                                value="Debug testing function"
                            )
                            
                            test_error_btn = gr.Button("🚨 Test Error Handling", variant="primary")
                        
                        with gr.Column():
                            test_result_display = gr.Markdown("**Test Results**\n\nClick 'Test Error Handling' to simulate errors.")
            
            # Event handlers
            refresh_health_btn.click(
                fn=self.debugger.get_system_health,
                outputs=[system_health_display]
            )
            
            start_monitoring_btn.click(
                fn=lambda: self.debugger.enable_monitoring,
                outputs=[]
            )
            
            stop_monitoring_btn.click(
                fn=lambda: setattr(self.debugger, 'enable_monitoring', False),
                outputs=[]
            )
            
            refresh_errors_btn.click(
                fn=self.error_handler.get_error_analysis,
                outputs=[error_summary_display]
            )
            
            error_analysis_btn.click(
                fn=self.error_handler.get_error_analysis,
                outputs=[error_analysis_display]
            )
            
            clear_errors_btn.click(
                fn=self.error_handler.error_log.clear,
                outputs=[error_summary_display, error_analysis_display]
            )
            
            debug_mode_toggle.change(
                fn=self.debugger.enable_debug_mode,
                inputs=[debug_mode_toggle],
                outputs=[]
            )
            
            add_breakpoint_btn.click(
                fn=self.debugger.add_breakpoint,
                inputs=[function_name_input],
                outputs=[]
            )
            
            remove_breakpoint_btn.click(
                fn=self.debugger.remove_breakpoint,
                inputs=[function_name_input],
                outputs=[]
            )
            
            test_error_btn.click(
                fn=self._test_error_handling,
                inputs=[test_error_type, test_context],
                outputs=[test_result_display]
            )
            
            export_btn.click(
                fn=self._export_debug_data,
                outputs=[export_status_display]
            )
        
        return demo
    
    def _test_error_handling(self, error_type: str, context: str) -> str:
        """Test error handling by simulating errors."""
        try:
            if error_type == "memory_error":
                raise torch.cuda.OutOfMemoryError("Simulated CUDA out of memory error")
            elif error_type == "validation_error":
                raise ValueError("Simulated validation error")
            elif error_type == "model_error":
                raise RuntimeError("Simulated model error")
            elif error_type == "system_error":
                raise Exception("Simulated system error")
            else:
                return "Please select an error type to test."
                
        except Exception as e:
            return self.error_handler.handle_error(e, context)
    
    def _export_debug_data(self) -> str:
        """Export debug data to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"debug_data_{timestamp}.json"
        
        if self.debugger.export_debug_data(filename):
            return f"✅ Debug data exported successfully to {filename}"
        else:
            return "❌ Failed to export debug data"

def create_debugging_interface():
    """Create and return the debugging interface."""
    debugger = GradioDebugger(enable_monitoring=True)
    error_handler = GradioErrorHandler(debugger)
    debug_interface = GradioDebugInterface(debugger, error_handler)
    
    return debug_interface.create_debug_interface()

if __name__ == "__main__":
    # Create and launch the debugging interface
    demo = create_debugging_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7864,
        share=True,
        debug=True
    )
