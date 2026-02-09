from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import os
import sys
import logging
import traceback
import asyncio
import time
import threading
import psutil
import gc
import json
import inspect
import linecache
from typing import List, Dict, Optional, Tuple, Any, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn as nn
import torch.optim as optim.nn as nn
from memory_profiler import profile
import gradio as gr
from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from production_code import MultiGPUTrainer, TrainingConfiguration
from error_handling_gradio import GradioErrorHandler
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
Advanced Debugging and Error Handling System
===========================================

This module provides comprehensive debugging and error handling capabilities:
- Advanced error analysis and classification
- Real-time debugging tools
- Performance profiling and optimization
- Memory leak detection
- System health monitoring
- Automated error recovery
- Debug logging and tracing
- Interactive debugging interfaces
"""


# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DebugConfiguration:
    """Configuration for debugging system"""
    enable_debugging: bool = True
    enable_profiling: bool = True
    enable_memory_tracking: bool = True
    enable_performance_monitoring: bool = True
    enable_error_analysis: bool = True
    enable_auto_recovery: bool = True
    debug_log_level: str = "DEBUG"
    max_debug_log_size: int = 10000
    profiling_interval: float = 1.0
    memory_check_interval: float = 5.0
    performance_threshold: float = 0.8
    error_analysis_depth: int = 10
    auto_recovery_attempts: int = 3


@dataclass
class DebugEvent:
    """Debug event structure"""
    timestamp: datetime
    event_type: str
    event_id: str
    severity: str  # INFO, WARNING, ERROR, CRITICAL
    message: str
    context: Dict[str, Any]
    stack_trace: Optional[str] = None
    performance_metrics: Optional[Dict[str, float]] = None
    memory_usage: Optional[Dict[str, float]] = None
    error_details: Optional[Dict[str, Any]] = None


class AdvancedDebugger:
    """Advanced debugging system with comprehensive error handling"""
    
    def __init__(self, config: DebugConfiguration = None):
        
    """__init__ function."""
self.config = config or DebugConfiguration()
        self.debug_events = deque(maxlen=self.config.max_debug_log_size)
        self.error_patterns = defaultdict(int)
        self.performance_history = deque(maxlen=1000)
        self.memory_history = deque(maxlen=1000)
        self.active_profilers = {}
        self.debug_session_id = self._generate_session_id()
        
        # Initialize monitoring
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Error handler integration
        self.error_handler = GradioErrorHandler()
        
        # Performance tracking
        self.performance_start_time = time.time()
        self.function_timings = defaultdict(list)
        self.memory_snapshots = []
        
        logger.info(f"Advanced Debugger initialized with session ID: {self.debug_session_id}")
    
    def _generate_session_id(self) -> str:
        """Generate unique debug session ID"""
        return f"debug_{int(time.time())}_{os.getpid()}"
    
    def log_debug_event(self, event_type: str, message: str, severity: str = "INFO", 
                       context: Dict[str, Any] = None, error: Exception = None):
        """Log a debug event with comprehensive information"""
        try:
            event = DebugEvent(
                timestamp=datetime.now(),
                event_type=event_type,
                event_id=f"{event_type}_{len(self.debug_events)}",
                severity=severity,
                message=message,
                context=context or {},
                stack_trace=traceback.format_exc() if error else None,
                performance_metrics=self._get_current_performance_metrics(),
                memory_usage=self._get_current_memory_usage(),
                error_details=self._analyze_error(error) if error else None
            )
            
            self.debug_events.append(event)
            
            # Update error patterns
            if error:
                error_type = type(error).__name__
                self.error_patterns[error_type] += 1
            
            # Log to standard logger
            log_level = getattr(logging, severity.upper(), logging.INFO)
            logger.log(log_level, f"[{event_type}] {message}")
            
        except Exception as e:
            logger.error(f"Error logging debug event: {e}")
    
    def _get_current_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_gb': memory.available / (1024**3),
                'memory_used_gb': memory.used / (1024**3),
                'uptime_seconds': time.time() - self.performance_start_time
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _get_current_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage details"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                'rss_mb': memory_info.rss / (1024**2),
                'vms_mb': memory_info.vms / (1024**2),
                'percent': process.memory_percent(),
                'num_threads': process.num_threads(),
                'num_fds': process.num_fds() if hasattr(process, 'num_fds') else 0
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _analyze_error(self, error: Exception) -> Dict[str, Any]:
        """Analyze error and extract detailed information"""
        if not error:
            return {}
        
        try:
            error_info = {
                'error_type': type(error).__name__,
                'error_message': str(error),
                'error_args': error.args,
                'error_module': error.__class__.__module__,
                'error_class': error.__class__.__name__,
                'stack_trace': traceback.format_exc(),
                'frame_info': self._get_frame_info(),
                'context_info': self._get_context_info()
            }
            
            # Add specific error analysis
            if isinstance(error, (ValueError, TypeError)):
                error_info['validation_error'] = True
            elif isinstance(error, (MemoryError, OSError)):
                error_info['system_error'] = True
            elif isinstance(error, (RuntimeError, AttributeError)):
                error_info['runtime_error'] = True
            
            return error_info
            
        except Exception as e:
            return {'analysis_error': str(e)}
    
    def _get_frame_info(self) -> Dict[str, Any]:
        """Get current frame information"""
        try:
            frame = inspect.currentframe()
            if frame:
                return {
                    'filename': frame.f_code.co_filename,
                    'function': frame.f_code.co_name,
                    'line_number': frame.f_lineno,
                    'locals': {k: str(v)[:100] for k, v in frame.f_locals.items() if not k.startswith('_')}
                }
        except:
            pass
        return {}
    
    def _get_context_info(self) -> Dict[str, Any]:
        """Get context information for debugging"""
        try:
            return {
                'python_version': sys.version,
                'platform': sys.platform,
                'working_directory': os.getcwd(),
                'environment_variables': {k: v for k, v in os.environ.items() if 'PATH' not in k},
                'torch_version': torch.__version__ if torch else None,
                'cuda_available': torch.cuda.is_available() if torch else False,
                'gpu_count': torch.cuda.device_count() if torch and torch.cuda.is_available() else 0
            }
        except Exception as e:
            return {'context_error': str(e)}
    
    def start_performance_monitoring(self) -> Any:
        """Start continuous performance monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        self.monitoring_thread.start()
        
        self.log_debug_event("MONITORING_START", "Performance monitoring started", "INFO")
    
    def stop_performance_monitoring(self) -> Any:
        """Stop performance monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        
        self.log_debug_event("MONITORING_STOP", "Performance monitoring stopped", "INFO")
    
    def _monitoring_loop(self) -> Any:
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect performance metrics
                metrics = self._get_current_performance_metrics()
                self.performance_history.append(metrics)
                
                # Collect memory metrics
                memory = self._get_current_memory_usage()
                self.memory_history.append(memory)
                
                # Check for performance issues
                if metrics.get('cpu_percent', 0) > 90:
                    self.log_debug_event("PERFORMANCE_WARNING", "High CPU usage detected", "WARNING", metrics)
                
                if metrics.get('memory_percent', 0) > 90:
                    self.log_debug_event("MEMORY_WARNING", "High memory usage detected", "WARNING", memory)
                
                # Memory leak detection
                if len(self.memory_history) > 10:
                    recent_memory = [m.get('rss_mb', 0) for m in list(self.memory_history)[-10:]]
                    if all(recent_memory[i] > recent_memory[i-1] for i in range(1, len(recent_memory))):
                        self.log_debug_event("MEMORY_LEAK", "Potential memory leak detected", "WARNING", {
                            'memory_trend': recent_memory
                        })
                
                time.sleep(self.config.profiling_interval)
                
            except Exception as e:
                self.log_debug_event("MONITORING_ERROR", f"Error in monitoring loop: {e}", "ERROR", error=e)
                time.sleep(1.0)
    
    def profile_function(self, func: Callable, *args, **kwargs):
        """Profile a function execution"""
        start_time = time.time()
        start_memory = self._get_current_memory_usage()
        
        try:
            result = func(*args, **kwargs)
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = e
        
        end_time = time.time()
        end_memory = self._get_current_memory_usage()
        
        execution_time = end_time - start_time
        memory_delta = end_memory.get('rss_mb', 0) - start_memory.get('rss_mb', 0)
        
        # Store timing information
        self.function_timings[func.__name__].append(execution_time)
        
        # Log profiling information
        self.log_debug_event("FUNCTION_PROFILE", f"Function {func.__name__} profiled", "INFO", {
            'function_name': func.__name__,
            'execution_time': execution_time,
            'memory_delta_mb': memory_delta,
            'success': success,
            'args_count': len(args),
            'kwargs_count': len(kwargs)
        }, error=error)
        
        return result, execution_time, memory_delta
    
    def memory_profile(self, func: Callable, *args, **kwargs):
        """Detailed memory profiling of a function"""
        try:
            # Take memory snapshot before
            gc.collect()
            before_memory = self._get_current_memory_usage()
            
            # Execute function
            result = self.profile_function(func, *args, **kwargs)
            
            # Take memory snapshot after
            gc.collect()
            after_memory = self._get_current_memory_usage()
            
            # Calculate memory differences
            memory_diff = {
                'rss_delta_mb': after_memory.get('rss_mb', 0) - before_memory.get('rss_mb', 0),
                'vms_delta_mb': after_memory.get('vms_mb', 0) - before_memory.get('vms_mb', 0),
                'percent_delta': after_memory.get('percent', 0) - before_memory.get('percent', 0)
            }
            
            self.log_debug_event("MEMORY_PROFILE", f"Memory profile for {func.__name__}", "INFO", {
                'function_name': func.__name__,
                'memory_differences': memory_diff,
                'before_memory': before_memory,
                'after_memory': after_memory
            })
            
            return result, memory_diff
            
        except Exception as e:
            self.log_debug_event("MEMORY_PROFILE_ERROR", f"Error in memory profiling: {e}", "ERROR", error=e)
            return None, {}
    
    def analyze_error_patterns(self) -> Dict[str, Any]:
        """Analyze error patterns and provide insights"""
        try:
            error_events = [event for event in self.debug_events if event.severity in ['ERROR', 'CRITICAL']]
            
            if not error_events:
                return {'message': 'No errors found in debug log'}
            
            # Analyze error types
            error_types = defaultdict(int)
            error_contexts = defaultdict(list)
            error_timeline = []
            
            for event in error_events:
                error_types[event.event_type] += 1
                error_contexts[event.event_type].append(event.context)
                error_timeline.append({
                    'timestamp': event.timestamp.isoformat(),
                    'event_type': event.event_type,
                    'message': event.message
                })
            
            # Find most common errors
            most_common_errors = sorted(error_types.items(), key=lambda x: x[1], reverse=True)
            
            # Analyze error frequency
            recent_errors = [e for e in error_events if e.timestamp > datetime.now() - timedelta(hours=1)]
            error_frequency = len(recent_errors)
            
            return {
                'total_errors': len(error_events),
                'error_frequency_per_hour': error_frequency,
                'most_common_errors': most_common_errors[:5],
                'error_timeline': error_timeline[-10:],  # Last 10 errors
                'error_contexts': dict(error_contexts),
                'error_patterns': dict(self.error_patterns)
            }
            
        except Exception as e:
            return {'analysis_error': str(e)}
    
    def get_performance_analysis(self) -> Dict[str, Any]:
        """Analyze performance metrics"""
        try:
            if not self.performance_history:
                return {'message': 'No performance data available'}
            
            recent_metrics = list(self.performance_history)[-100:]  # Last 100 measurements
            
            cpu_values = [m.get('cpu_percent', 0) for m in recent_metrics]
            memory_values = [m.get('memory_percent', 0) for m in recent_metrics]
            
            return {
                'cpu_analysis': {
                    'current': cpu_values[-1] if cpu_values else 0,
                    'average': np.mean(cpu_values) if cpu_values else 0,
                    'max': np.max(cpu_values) if cpu_values else 0,
                    'min': np.min(cpu_values) if cpu_values else 0
                },
                'memory_analysis': {
                    'current': memory_values[-1] if memory_values else 0,
                    'average': np.mean(memory_values) if memory_values else 0,
                    'max': np.max(memory_values) if memory_values else 0,
                    'min': np.min(memory_values) if memory_values else 0
                },
                'function_timings': {
                    func: {
                        'count': len(timings),
                        'average_time': np.mean(timings) if timings else 0,
                        'max_time': np.max(timings) if timings else 0,
                        'min_time': np.min(timings) if timings else 0
                    }
                    for func, timings in self.function_timings.items()
                },
                'uptime_seconds': time.time() - self.performance_start_time
            }
            
        except Exception as e:
            return {'analysis_error': str(e)}
    
    def get_memory_analysis(self) -> Dict[str, Any]:
        """Analyze memory usage patterns"""
        try:
            if not self.memory_history:
                return {'message': 'No memory data available'}
            
            recent_memory = list(self.memory_history)[-100:]  # Last 100 measurements
            
            rss_values = [m.get('rss_mb', 0) for m in recent_memory]
            vms_values = [m.get('vms_mb', 0) for m in recent_memory]
            percent_values = [m.get('percent', 0) for m in recent_memory]
            
            # Detect memory trends
            if len(rss_values) > 10:
                recent_trend = rss_values[-10:]
                memory_growing = all(recent_trend[i] >= recent_trend[i-1] for i in range(1, len(recent_trend)))
                memory_stable = all(abs(recent_trend[i] - recent_trend[i-1]) < 1.0 for i in range(1, len(recent_trend)))
            else:
                memory_growing = False
                memory_stable = False
            
            return {
                'rss_analysis': {
                    'current_mb': rss_values[-1] if rss_values else 0,
                    'average_mb': np.mean(rss_values) if rss_values else 0,
                    'max_mb': np.max(rss_values) if rss_values else 0,
                    'min_mb': np.min(rss_values) if rss_values else 0
                },
                'vms_analysis': {
                    'current_mb': vms_values[-1] if vms_values else 0,
                    'average_mb': np.mean(vms_values) if vms_values else 0,
                    'max_mb': np.max(vms_values) if vms_values else 0,
                    'min_mb': np.min(vms_values) if vms_values else 0
                },
                'percent_analysis': {
                    'current': percent_values[-1] if percent_values else 0,
                    'average': np.mean(percent_values) if percent_values else 0,
                    'max': np.max(percent_values) if percent_values else 0,
                    'min': np.min(percent_values) if percent_values else 0
                },
                'memory_trends': {
                    'growing': memory_growing,
                    'stable': memory_stable,
                    'potential_leak': memory_growing and not memory_stable
                }
            }
            
        except Exception as e:
            return {'analysis_error': str(e)}
    
    def create_debug_report(self) -> Dict[str, Any]:
        """Create comprehensive debug report"""
        try:
            return {
                'session_info': {
                    'session_id': self.debug_session_id,
                    'start_time': self.performance_start_time,
                    'uptime_seconds': time.time() - self.performance_start_time,
                    'total_events': len(self.debug_events),
                    'monitoring_active': self.monitoring_active
                },
                'error_analysis': self.analyze_error_patterns(),
                'performance_analysis': self.get_performance_analysis(),
                'memory_analysis': self.get_memory_analysis(),
                'recent_events': [
                    {
                        'timestamp': event.timestamp.isoformat(),
                        'event_type': event.event_type,
                        'severity': event.severity,
                        'message': event.message
                    }
                    for event in list(self.debug_events)[-20:]  # Last 20 events
                ],
                'system_info': self._get_context_info()
            }
            
        except Exception as e:
            return {'report_error': str(e)}
    
    def export_debug_data(self, filename: str = None) -> str:
        """Export debug data to file"""
        try:
            if not filename:
                filename = f"debug_report_{self.debug_session_id}_{int(time.time())}.json"
            
            report = self.create_debug_report()
            
            with open(filename, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                json.dump(report, f, indent=2, default=str)
            
            self.log_debug_event("DEBUG_EXPORT", f"Debug data exported to {filename}", "INFO")
            return filename
            
        except Exception as e:
            self.log_debug_event("DEBUG_EXPORT_ERROR", f"Error exporting debug data: {e}", "ERROR", error=e)
            return None
    
    def clear_debug_data(self) -> Any:
        """Clear all debug data"""
        self.debug_events.clear()
        self.error_patterns.clear()
        self.performance_history.clear()
        self.memory_history.clear()
        self.function_timings.clear()
        self.memory_snapshots.clear()
        
        self.log_debug_event("DEBUG_CLEAR", "Debug data cleared", "INFO")


class DebuggingInterface:
    """Gradio interface for advanced debugging"""
    
    def __init__(self) -> Any:
        self.debugger = AdvancedDebugger()
        self.config = TrainingConfiguration(
            enable_gradio_demo=True,
            gradio_port=7867,
            gradio_share=False
        )
        
        logger.info("Debugging Interface initialized")
    
    def create_debugging_interface(self) -> gr.Interface:
        """Create comprehensive debugging interface"""
        
        def get_debug_report():
            """Get comprehensive debug report"""
            return self.debugger.create_debug_report()
        
        def export_debug_data():
            """Export debug data"""
            filename = self.debugger.export_debug_data()
            return f"Debug data exported to: {filename}" if filename else "Export failed"
        
        def clear_debug_data():
            """Clear debug data"""
            self.debugger.clear_debug_data()
            return "Debug data cleared"
        
        def start_monitoring():
            """Start performance monitoring"""
            self.debugger.start_performance_monitoring()
            return "Performance monitoring started"
        
        def stop_monitoring():
            """Stop performance monitoring"""
            self.debugger.stop_performance_monitoring()
            return "Performance monitoring stopped"
        
        def test_error_handling():
            """Test error handling capabilities"""
            try:
                # Simulate different types of errors
                self.debugger.log_debug_event("TEST_ERROR", "Testing error handling", "ERROR", 
                                            error=ValueError("Test validation error"))
                self.debugger.log_debug_event("TEST_WARNING", "Testing warning handling", "WARNING")
                self.debugger.log_debug_event("TEST_INFO", "Testing info handling", "INFO")
                
                return "Error handling test completed"
            except Exception as e:
                return f"Error in test: {e}"
        
        # Create interface
        with gr.Blocks(
            title="Advanced Debugging System",
            theme=gr.themes.Soft(),
            css="""
            .debug-section {
                background: #f8f9fa;
                border-radius: 10px;
                padding: 1rem;
                margin: 1rem 0;
                border: 1px solid #dee2e6;
            }
            .error-section {
                background: #fff3cd;
                border: 1px solid #ffeaa7;
                border-radius: 10px;
                padding: 1rem;
                margin: 1rem 0;
            }
            .performance-section {
                background: #d1ecf1;
                border: 1px solid #bee5eb;
                border-radius: 10px;
                padding: 1rem;
                margin: 1rem 0;
            }
            """
        ) as interface:
            
            gr.Markdown("# 🔍 Advanced Debugging System")
            gr.Markdown("Comprehensive debugging and error handling tools for AI systems")
            
            with gr.Tabs():
                with gr.TabItem("📊 Debug Overview"):
                    gr.Markdown("### System Debug Overview")
                    
                    with gr.Row():
                        with gr.Column():
                            debug_report_btn = gr.Button("📊 Generate Debug Report", variant="primary")
                            debug_report_output = gr.JSON(label="Debug Report")
                        
                        with gr.Column():
                            gr.Markdown("### Quick Actions")
                            test_btn = gr.Button("🧪 Test Error Handling")
                            export_btn = gr.Button("📁 Export Debug Data")
                            clear_btn = gr.Button("🗑️ Clear Debug Data")
                            
                            test_output = gr.Textbox(label="Test Results")
                            export_output = gr.Textbox(label="Export Results")
                            clear_output = gr.Textbox(label="Clear Results")
                
                with gr.TabItem("📈 Performance Monitoring"):
                    gr.Markdown("### Performance Monitoring and Analysis")
                    
                    with gr.Row():
                        with gr.Column():
                            start_monitor_btn = gr.Button("▶️ Start Monitoring", variant="primary")
                            stop_monitor_btn = gr.Button("⏹️ Stop Monitoring")
                            
                            monitor_status = gr.Textbox(label="Monitoring Status")
                        
                        with gr.Column():
                            gr.Markdown("### Performance Metrics")
                            performance_analysis_btn = gr.Button("📊 Performance Analysis")
                            performance_output = gr.JSON(label="Performance Analysis")
                    
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Memory Analysis")
                            memory_analysis_btn = gr.Button("🧠 Memory Analysis")
                            memory_output = gr.JSON(label="Memory Analysis")
                        
                        with gr.Column():
                            gr.Markdown("### Function Profiling")
                            function_timings_btn = gr.Button("⏱️ Function Timings")
                            timings_output = gr.JSON(label="Function Timings")
                
                with gr.TabItem("🚨 Error Analysis"):
                    gr.Markdown("### Error Analysis and Patterns")
                    
                    with gr.Row():
                        with gr.Column():
                            error_analysis_btn = gr.Button("🔍 Error Analysis", variant="primary")
                            error_analysis_output = gr.JSON(label="Error Analysis")
                        
                        with gr.Column():
                            gr.Markdown("### Error Patterns")
                            error_patterns_btn = gr.Button("📊 Error Patterns")
                            patterns_output = gr.JSON(label="Error Patterns")
                    
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Recent Events")
                            recent_events_btn = gr.Button("📋 Recent Events")
                            events_output = gr.JSON(label="Recent Events")
                        
                        with gr.Column():
                            gr.Markdown("### System Health")
                            health_btn = gr.Button("💚 System Health")
                            health_output = gr.JSON(label="System Health")
                
                with gr.TabItem("🔧 Debug Tools"):
                    gr.Markdown("### Advanced Debug Tools")
                    
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Memory Profiling")
                            memory_profile_btn = gr.Button("🧠 Memory Profile")
                            memory_profile_output = gr.JSON(label="Memory Profile")
                        
                        with gr.Column():
                            gr.Markdown("### Function Profiling")
                            function_profile_btn = gr.Button("⚡ Function Profile")
                            function_profile_output = gr.JSON(label="Function Profile")
                    
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### System Information")
                            system_info_btn = gr.Button("💻 System Info")
                            system_info_output = gr.JSON(label="System Information")
                        
                        with gr.Column():
                            gr.Markdown("### Context Information")
                            context_info_btn = gr.Button("📋 Context Info")
                            context_info_output = gr.JSON(label="Context Information")
                
                with gr.TabItem("📚 Documentation"):
                    gr.Markdown("### Debugging Documentation")
                    gr.Markdown("""
                    ## Advanced Debugging System Features
                    
                    **Error Handling:**
                    - Comprehensive error analysis and classification
                    - Error pattern detection and analysis
                    - Stack trace analysis and context extraction
                    - Error recovery suggestions
                    
                    **Performance Monitoring:**
                    - Real-time CPU and memory monitoring
                    - Function execution time profiling
                    - Memory leak detection
                    - Performance bottleneck identification
                    
                    **Debug Tools:**
                    - Memory profiling and analysis
                    - Function timing and optimization
                    - System health monitoring
                    - Context information extraction
                    
                    **Reporting:**
                    - Comprehensive debug reports
                    - Data export capabilities
                    - Error pattern analysis
                    - Performance trend analysis
                    
                    **Usage:**
                    1. Start monitoring to track system performance
                    2. Use error analysis to identify and fix issues
                    3. Profile functions to optimize performance
                    4. Export debug data for offline analysis
                    5. Monitor system health and memory usage
                    """)
            
            # Event handlers
            debug_report_btn.click(
                fn=get_debug_report,
                inputs=[],
                outputs=[debug_report_output]
            )
            
            test_btn.click(
                fn=test_error_handling,
                inputs=[],
                outputs=[test_output]
            )
            
            export_btn.click(
                fn=export_debug_data,
                inputs=[],
                outputs=[export_output]
            )
            
            clear_btn.click(
                fn=clear_debug_data,
                inputs=[],
                outputs=[clear_output]
            )
            
            start_monitor_btn.click(
                fn=start_monitoring,
                inputs=[],
                outputs=[monitor_status]
            )
            
            stop_monitor_btn.click(
                fn=stop_monitoring,
                inputs=[],
                outputs=[monitor_status]
            )
            
            performance_analysis_btn.click(
                fn=self.debugger.get_performance_analysis,
                inputs=[],
                outputs=[performance_output]
            )
            
            memory_analysis_btn.click(
                fn=self.debugger.get_memory_analysis,
                inputs=[],
                outputs=[memory_output]
            )
            
            error_analysis_btn.click(
                fn=self.debugger.analyze_error_patterns,
                inputs=[],
                outputs=[error_analysis_output]
            )
            
            system_info_btn.click(
                fn=self.debugger._get_context_info,
                inputs=[],
                outputs=[system_info_output]
            )
        
        return interface
    
    def launch_debugging_interface(self, port: int = 7867, share: bool = False):
        """Launch the debugging interface"""
        print("🔍 Launching Advanced Debugging System...")
        
        interface = self.create_debugging_interface()
        interface.launch(
            server_name="0.0.0.0",
            server_port=port,
            share=share,
            show_error=True,
            quiet=False
        )


def main():
    """Main function to run the debugging system"""
    print("🔍 Starting Advanced Debugging System...")
    
    interface = DebuggingInterface()
    interface.launch_debugging_interface(port=7867, share=False)


match __name__:
    case "__main__":
    main() 