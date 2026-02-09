"""
Advanced Error Handling and Debugging System
Extended error handling with debugging capabilities, performance profiling, and advanced error analysis
"""

import gradio as gr
import torch
import numpy as np
import logging
import traceback
import time
import json
import re
import sys
import os
import psutil
import gc
import threading
import queue
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from functools import wraps
from enum import Enum
import warnings
import inspect
import linecache
from contextlib import contextmanager
import cProfile
import pstats
from io import StringIO
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DebugLevel(Enum):
    """Debug levels for different types of debugging"""
    BASIC = "basic"
    DETAILED = "detailed"
    PROFILING = "profiling"
    MEMORY = "memory"
    THREADING = "threading"
    FULL = "full"


class ErrorCategory(Enum):
    """Error categories for better classification"""
    INPUT_VALIDATION = "input_validation"
    MODEL_LOADING = "model_loading"
    INFERENCE = "inference"
    MEMORY = "memory"
    NETWORK = "network"
    SYSTEM = "system"
    UNKNOWN = "unknown"


@dataclass
class DebugInfo:
    """Debug information structure"""
    function_name: str
    execution_time: float
    memory_usage: Dict[str, float]
    cpu_usage: float
    call_stack: List[str]
    local_variables: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    debug_level: DebugLevel = DebugLevel.BASIC


@dataclass
class PerformanceProfile:
    """Performance profiling information"""
    function_name: str
    total_calls: int
    total_time: float
    average_time: float
    min_time: float
    max_time: float
    memory_peak: float
    cpu_peak: float
    profile_data: Dict[str, Any] = field(default_factory=dict)


class AdvancedErrorAnalyzer:
    """Advanced error analysis and pattern detection"""
    
    def __init__(self):
        self.error_patterns = {}
        self.error_frequency = {}
        self.error_correlations = {}
        self._setup_error_patterns()
    
    def _setup_error_patterns(self):
        """Setup common error patterns and their solutions"""
        self.error_patterns = {
            "CUDA_OUT_OF_MEMORY": {
                "category": ErrorCategory.MEMORY,
                "solutions": [
                    "Reduce batch size",
                    "Use gradient checkpointing",
                    "Clear GPU cache",
                    "Use CPU instead of GPU",
                    "Close other GPU applications"
                ],
                "prevention": [
                    "Monitor GPU memory usage",
                    "Use memory-efficient models",
                    "Implement dynamic batch sizing"
                ]
            },
            "MODEL_NOT_FOUND": {
                "category": ErrorCategory.MODEL_LOADING,
                "solutions": [
                    "Check model name spelling",
                    "Download model from Hugging Face",
                    "Verify internet connection",
                    "Check model cache directory"
                ],
                "prevention": [
                    "Pre-download required models",
                    "Use model validation",
                    "Implement fallback models"
                ]
            },
            "INVALID_INPUT": {
                "category": ErrorCategory.INPUT_VALIDATION,
                "solutions": [
                    "Validate input format",
                    "Check input length limits",
                    "Sanitize input data",
                    "Provide input examples"
                ],
                "prevention": [
                    "Implement comprehensive validation",
                    "Use input sanitization",
                    "Provide clear error messages"
                ]
            },
            "NETWORK_TIMEOUT": {
                "category": ErrorCategory.NETWORK,
                "solutions": [
                    "Increase timeout settings",
                    "Check internet connection",
                    "Use local models",
                    "Implement retry logic"
                ],
                "prevention": [
                    "Use connection pooling",
                    "Implement circuit breakers",
                    "Cache network responses"
                ]
            }
        }
    
    def analyze_error(self, error: Exception, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze error and provide detailed insights"""
        error_type = type(error).__name__
        error_message = str(error)
        
        # Basic error analysis
        analysis = {
            "error_type": error_type,
            "error_message": error_message,
            "category": self._categorize_error(error_type, error_message),
            "severity": self._assess_severity(error_type, error_message),
            "frequency": self.error_frequency.get(error_type, 0),
            "patterns": self._detect_patterns(error_type, error_message),
            "solutions": self._get_solutions(error_type, error_message),
            "prevention": self._get_prevention(error_type, error_message),
            "context": context or {}
        }
        
        # Update frequency
        self.error_frequency[error_type] = self.error_frequency.get(error_type, 0) + 1
        
        return analysis
    
    def _categorize_error(self, error_type: str, error_message: str) -> ErrorCategory:
        """Categorize error based on type and message"""
        error_lower = error_message.lower()
        
        if "cuda" in error_lower or "memory" in error_lower:
            return ErrorCategory.MEMORY
        elif "model" in error_lower or "load" in error_lower:
            return ErrorCategory.MODEL_LOADING
        elif "input" in error_lower or "validation" in error_lower:
            return ErrorCategory.INPUT_VALIDATION
        elif "network" in error_lower or "connection" in error_lower:
            return ErrorCategory.NETWORK
        elif "system" in error_lower or "os" in error_lower:
            return ErrorCategory.SYSTEM
        elif "inference" in error_lower or "forward" in error_lower:
            return ErrorCategory.INFERENCE
        else:
            return ErrorCategory.UNKNOWN
    
    def _assess_severity(self, error_type: str, error_message: str) -> str:
        """Assess error severity"""
        critical_keywords = ["memory", "cuda", "system", "critical"]
        high_keywords = ["model", "network", "timeout", "connection"]
        medium_keywords = ["input", "validation", "format"]
        
        error_lower = error_message.lower()
        
        if any(keyword in error_lower for keyword in critical_keywords):
            return "CRITICAL"
        elif any(keyword in error_lower for keyword in high_keywords):
            return "HIGH"
        elif any(keyword in error_lower for keyword in medium_keywords):
            return "MEDIUM"
        else:
            return "LOW"
    
    def _detect_patterns(self, error_type: str, error_message: str) -> List[str]:
        """Detect error patterns"""
        patterns = []
        
        for pattern_name, pattern_info in self.error_patterns.items():
            if pattern_name.lower() in error_type.lower() or pattern_name.lower() in error_message.lower():
                patterns.append(pattern_name)
        
        return patterns
    
    def _get_solutions(self, error_type: str, error_message: str) -> List[str]:
        """Get solutions for the error"""
        solutions = []
        
        for pattern_name, pattern_info in self.error_patterns.items():
            if pattern_name.lower() in error_type.lower() or pattern_name.lower() in error_message.lower():
                solutions.extend(pattern_info["solutions"])
        
        return list(set(solutions))  # Remove duplicates
    
    def _get_prevention(self, error_type: str, error_message: str) -> List[str]:
        """Get prevention strategies for the error"""
        prevention = []
        
        for pattern_name, pattern_info in self.error_patterns.items():
            if pattern_name.lower() in error_type.lower() or pattern_name.lower() in error_message.lower():
                prevention.extend(pattern_info["prevention"])
        
        return list(set(prevention))  # Remove duplicates
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics"""
        return {
            "total_errors": sum(self.error_frequency.values()),
            "error_frequency": self.error_frequency,
            "most_common_errors": sorted(self.error_frequency.items(), key=lambda x: x[1], reverse=True)[:10],
            "error_categories": self._get_category_distribution(),
            "error_trends": self._analyze_error_trends()
        }
    
    def _get_category_distribution(self) -> Dict[str, int]:
        """Get error distribution by category"""
        category_counts = {}
        
        for error_type, count in self.error_frequency.items():
            # This is a simplified version - in practice, you'd store category with each error
            category = ErrorCategory.UNKNOWN
            category_counts[category.value] = category_counts.get(category.value, 0) + count
        
        return category_counts
    
    def _analyze_error_trends(self) -> Dict[str, Any]:
        """Analyze error trends over time"""
        # This would analyze error patterns over time
        return {
            "trend_analysis": "Not implemented in this version",
            "suggestion": "Implement time-based error tracking for trend analysis"
        }


class PerformanceProfiler:
    """Advanced performance profiling and monitoring"""
    
    def __init__(self):
        self.profiles = {}
        self.active_profiles = {}
        self.memory_tracker = MemoryTracker()
        self.cpu_tracker = CPUTracker()
    
    @contextmanager
    def profile_function(self, function_name: str, debug_level: DebugLevel = DebugLevel.BASIC):
        """Context manager for profiling functions"""
        profile_id = f"{function_name}_{time.time()}"
        
        # Start profiling
        start_time = time.time()
        start_memory = self.memory_tracker.get_memory_usage()
        start_cpu = self.cpu_tracker.get_cpu_usage()
        
        # Setup profiler if detailed profiling is requested
        profiler = None
        if debug_level in [DebugLevel.PROFILING, DebugLevel.FULL]:
            profiler = cProfile.Profile()
            profiler.enable()
        
        try:
            yield profile_id
        finally:
            # End profiling
            end_time = time.time()
            end_memory = self.memory_tracker.get_memory_usage()
            end_cpu = self.cpu_tracker.get_cpu_usage()
            
            if profiler:
                profiler.disable()
                stats = self._get_profiler_stats(profiler)
            else:
                stats = {}
            
            # Calculate metrics
            execution_time = end_time - start_time
            memory_delta = end_memory["total"] - start_memory["total"]
            cpu_peak = max(start_cpu, end_cpu)
            
            # Store profile data
            profile_data = PerformanceProfile(
                function_name=function_name,
                total_calls=1,
                total_time=execution_time,
                average_time=execution_time,
                min_time=execution_time,
                max_time=execution_time,
                memory_peak=memory_delta,
                cpu_peak=cpu_peak,
                profile_data=stats
            )
            
            if function_name not in self.profiles:
                self.profiles[function_name] = []
            
            self.profiles[function_name].append(profile_data)
    
    def _get_profiler_stats(self, profiler: cProfile.Profile) -> Dict[str, Any]:
        """Get profiler statistics"""
        s = StringIO()
        stats = pstats.Stats(profiler, stream=s)
        stats.sort_stats('cumulative')
        stats.print_stats(20)  # Top 20 functions
        
        return {
            "stats_output": s.getvalue(),
            "function_calls": stats.total_calls,
            "primitive_calls": stats.prim_calls
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        summary = {}
        
        for function_name, profiles in self.profiles.items():
            if not profiles:
                continue
            
            total_calls = len(profiles)
            total_time = sum(p.total_time for p in profiles)
            times = [p.total_time for p in profiles]
            memory_peaks = [p.memory_peak for p in profiles]
            cpu_peaks = [p.cpu_peak for p in profiles]
            
            summary[function_name] = {
                "total_calls": total_calls,
                "total_time": total_time,
                "average_time": total_time / total_calls,
                "min_time": min(times),
                "max_time": max(times),
                "std_dev_time": np.std(times) if len(times) > 1 else 0,
                "average_memory_peak": np.mean(memory_peaks),
                "average_cpu_peak": np.mean(cpu_peaks),
                "recent_performance": profiles[-1].__dict__ if profiles else {}
            }
        
        return summary


class MemoryTracker:
    """Memory usage tracking and analysis"""
    
    def __init__(self):
        self.memory_history = []
        self.gc_stats = {}
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        # Get GPU memory if available
        gpu_memory = self._get_gpu_memory()
        
        memory_data = {
            "rss": memory_info.rss / 1024 / 1024,  # MB
            "vms": memory_info.vms / 1024 / 1024,  # MB
            "percent": process.memory_percent(),
            "gpu_memory": gpu_memory,
            "total": memory_info.rss / 1024 / 1024  # MB
        }
        
        self.memory_history.append({
            "timestamp": time.time(),
            "memory": memory_data
        })
        
        # Keep only last 1000 entries
        if len(self.memory_history) > 1000:
            self.memory_history = self.memory_history[-1000:]
        
        return memory_data
    
    def _get_gpu_memory(self) -> Dict[str, float]:
        """Get GPU memory usage"""
        try:
            if torch.cuda.is_available():
                return {
                    "allocated": torch.cuda.memory_allocated() / 1024 / 1024,  # MB
                    "cached": torch.cuda.memory_reserved() / 1024 / 1024,  # MB
                    "total": torch.cuda.get_device_properties(0).total_memory / 1024 / 1024  # MB
                }
        except Exception as e:
            logger.warning(f"Could not get GPU memory info: {e}")
        
        return {"allocated": 0, "cached": 0, "total": 0}
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get memory usage statistics"""
        if not self.memory_history:
            return {"error": "No memory history available"}
        
        memory_values = [entry["memory"]["total"] for entry in self.memory_history]
        
        return {
            "current_memory": self.memory_history[-1]["memory"],
            "peak_memory": max(memory_values),
            "average_memory": np.mean(memory_values),
            "memory_trend": self._analyze_memory_trend(),
            "memory_leak_detection": self._detect_memory_leaks()
        }
    
    def _analyze_memory_trend(self) -> str:
        """Analyze memory usage trend"""
        if len(self.memory_history) < 10:
            return "Insufficient data for trend analysis"
        
        recent_memory = [entry["memory"]["total"] for entry in self.memory_history[-10:]]
        
        if recent_memory[-1] > recent_memory[0] * 1.1:
            return "INCREASING - Potential memory leak"
        elif recent_memory[-1] < recent_memory[0] * 0.9:
            return "DECREASING - Memory cleanup detected"
        else:
            return "STABLE - Normal memory usage"
    
    def _detect_memory_leaks(self) -> Dict[str, Any]:
        """Detect potential memory leaks"""
        if len(self.memory_history) < 20:
            return {"detected": False, "reason": "Insufficient data"}
        
        # Simple memory leak detection
        memory_values = [entry["memory"]["total"] for entry in self.memory_history[-20:]]
        
        # Check for consistent increase
        increasing_count = sum(1 for i in range(1, len(memory_values)) 
                             if memory_values[i] > memory_values[i-1])
        
        leak_probability = increasing_count / (len(memory_values) - 1)
        
        return {
            "detected": leak_probability > 0.7,
            "probability": leak_probability,
            "suggestion": "Run garbage collection" if leak_probability > 0.7 else "Memory usage normal"
        }
    
    def force_garbage_collection(self) -> Dict[str, Any]:
        """Force garbage collection and return results"""
        before_memory = self.get_memory_usage()
        
        # Collect garbage
        collected = gc.collect()
        
        after_memory = self.get_memory_usage()
        
        return {
            "objects_collected": collected,
            "memory_freed": before_memory["total"] - after_memory["total"],
            "before_memory": before_memory,
            "after_memory": after_memory
        }


class CPUTracker:
    """CPU usage tracking and analysis"""
    
    def __init__(self):
        self.cpu_history = []
    
    def get_cpu_usage(self) -> float:
        """Get current CPU usage"""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        self.cpu_history.append({
            "timestamp": time.time(),
            "cpu_percent": cpu_percent
        })
        
        # Keep only last 1000 entries
        if len(self.cpu_history) > 1000:
            self.cpu_history = self.cpu_history[-1000:]
        
        return cpu_percent
    
    def get_cpu_statistics(self) -> Dict[str, Any]:
        """Get CPU usage statistics"""
        if not self.cpu_history:
            return {"error": "No CPU history available"}
        
        cpu_values = [entry["cpu_percent"] for entry in self.cpu_history]
        
        return {
            "current_cpu": cpu_values[-1],
            "peak_cpu": max(cpu_values),
            "average_cpu": np.mean(cpu_values),
            "cpu_trend": self._analyze_cpu_trend(),
            "cpu_cores": psutil.cpu_count(),
            "cpu_frequency": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
        }
    
    def _analyze_cpu_trend(self) -> str:
        """Analyze CPU usage trend"""
        if len(self.cpu_history) < 10:
            return "Insufficient data for trend analysis"
        
        recent_cpu = [entry["cpu_percent"] for entry in self.cpu_history[-10:]]
        
        if recent_cpu[-1] > recent_cpu[0] * 1.2:
            return "INCREASING - High CPU usage"
        elif recent_cpu[-1] < recent_cpu[0] * 0.8:
            return "DECREASING - CPU usage dropping"
        else:
            return "STABLE - Normal CPU usage"


class AdvancedDebugger:
    """Advanced debugging capabilities"""
    
    def __init__(self, debug_level: DebugLevel = DebugLevel.BASIC):
        self.debug_level = debug_level
        self.error_analyzer = AdvancedErrorAnalyzer()
        self.performance_profiler = PerformanceProfiler()
        self.memory_tracker = MemoryTracker()
        self.cpu_tracker = CPUTracker()
        self.debug_log = []
    
    def debug_function(self, func: Callable) -> Callable:
        """Decorator for debugging functions"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            function_name = func.__name__
            
            # Start debugging
            debug_info = self._create_debug_info(function_name, args, kwargs)
            
            try:
                with self.performance_profiler.profile_function(function_name, self.debug_level):
                    result = func(*args, **kwargs)
                
                # Log successful execution
                self._log_debug_info(debug_info, success=True, result=result)
                return result
            
            except Exception as e:
                # Analyze error
                error_analysis = self.error_analyzer.analyze_error(e, {
                    "function_name": function_name,
                    "args": str(args),
                    "kwargs": str(kwargs)
                })
                
                # Log error
                self._log_debug_info(debug_info, success=False, error=e, error_analysis=error_analysis)
                
                # Re-raise with enhanced error information
                raise self._enhance_exception(e, error_analysis)
        
        return wrapper
    
    def _create_debug_info(self, function_name: str, args: tuple, kwargs: dict) -> DebugInfo:
        """Create debug information"""
        return DebugInfo(
            function_name=function_name,
            execution_time=0,  # Will be updated after execution
            memory_usage=self.memory_tracker.get_memory_usage(),
            cpu_usage=self.cpu_tracker.get_cpu_usage(),
            call_stack=self._get_call_stack(),
            local_variables=self._get_local_variables(args, kwargs),
            debug_level=self.debug_level
        )
    
    def _get_call_stack(self) -> List[str]:
        """Get current call stack"""
        stack = []
        for frame_info in inspect.stack()[1:]:  # Skip current frame
            stack.append(f"{frame_info.function} at {frame_info.filename}:{frame_info.lineno}")
        return stack
    
    def _get_local_variables(self, args: tuple, kwargs: dict) -> Dict[str, Any]:
        """Get local variables (simplified version)"""
        return {
            "args_count": len(args),
            "kwargs_count": len(kwargs),
            "args_types": [type(arg).__name__ for arg in args],
            "kwargs_keys": list(kwargs.keys())
        }
    
    def _log_debug_info(self, debug_info: DebugInfo, success: bool, 
                       result: Any = None, error: Exception = None, 
                       error_analysis: Dict[str, Any] = None):
        """Log debug information"""
        log_entry = {
            "timestamp": time.time(),
            "debug_info": debug_info,
            "success": success,
            "result_type": type(result).__name__ if result is not None else None,
            "error": str(error) if error else None,
            "error_analysis": error_analysis
        }
        
        self.debug_log.append(log_entry)
        
        # Keep only last 1000 entries
        if len(self.debug_log) > 1000:
            self.debug_log = self.debug_log[-1000:]
    
    def _enhance_exception(self, original_error: Exception, error_analysis: Dict[str, Any]) -> Exception:
        """Enhance exception with debugging information"""
        enhanced_message = f"""
Original Error: {str(original_error)}
Error Type: {error_analysis['error_type']}
Category: {error_analysis['category'].value}
Severity: {error_analysis['severity']}
Solutions: {', '.join(error_analysis['solutions'][:3])}
        """.strip()
        
        # Create new exception with enhanced message
        enhanced_error = type(original_error)(enhanced_message)
        enhanced_error.original_error = original_error
        enhanced_error.error_analysis = error_analysis
        
        return enhanced_error
    
    def get_debug_summary(self) -> Dict[str, Any]:
        """Get comprehensive debug summary"""
        return {
            "debug_level": self.debug_level.value,
            "total_debug_entries": len(self.debug_log),
            "successful_executions": sum(1 for entry in self.debug_log if entry["success"]),
            "failed_executions": sum(1 for entry in self.debug_log if not entry["success"]),
            "error_statistics": self.error_analyzer.get_error_statistics(),
            "performance_summary": self.performance_profiler.get_performance_summary(),
            "memory_statistics": self.memory_tracker.get_memory_statistics(),
            "cpu_statistics": self.cpu_tracker.get_cpu_statistics(),
            "recent_debug_entries": self.debug_log[-10:] if self.debug_log else []
        }
    
    def clear_debug_log(self):
        """Clear debug log"""
        self.debug_log.clear()
        self.performance_profiler.profiles.clear()
        self.memory_tracker.memory_history.clear()
        self.cpu_tracker.cpu_history.clear()


class AdvancedErrorHandlingGradioInterface:
    """Advanced error handling and debugging Gradio interface"""
    
    def __init__(self, debug_level: DebugLevel = DebugLevel.DETAILED):
        self.debugger = AdvancedDebugger(debug_level)
        self.setup_logging()
    
    def setup_logging(self):
        """Setup advanced logging"""
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)
        
        # Setup file handlers
        file_handler = logging.FileHandler("logs/advanced_debug.log")
        file_handler.setLevel(logging.DEBUG)
        
        # Setup formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(file_handler)
        logger.setLevel(logging.DEBUG)
    
    @AdvancedDebugger.debug_function
    def debug_text_generation(self, prompt: str, max_length: int, temperature: float, model_name: str) -> Tuple[str, Optional[Any], str]:
        """Debug text generation with comprehensive error handling"""
        try:
            # Simulate text generation with potential errors
            if not prompt.strip():
                raise ValueError("Empty prompt provided")
            
            if max_length > 1000:
                raise ValueError("Max length too high")
            
            if temperature > 2.0:
                raise ValueError("Temperature too high")
            
            # Simulate successful generation
            generated_text = f"Generated text based on: '{prompt[:50]}...' with length {max_length} and temperature {temperature} using {model_name}"
            
            return generated_text, None, "✅ Generation Complete"
        
        except Exception as e:
            raise e
    
    @AdvancedDebugger.debug_function
    def debug_sentiment_analysis(self, text: str) -> Tuple[str, Optional[Any], str]:
        """Debug sentiment analysis with comprehensive error handling"""
        try:
            # Simulate sentiment analysis with potential errors
            if len(text) < 10:
                raise ValueError("Text too short for analysis")
            
            if len(text) > 2000:
                raise ValueError("Text too long for analysis")
            
            # Simulate successful analysis
            sentiment_result = {
                "label": "positive",
                "score": 0.85,
                "positive": 0.85,
                "neutral": 0.10,
                "negative": 0.05
            }
            
            result_text = f"🎭 **Sentiment Analysis Results**\n\n"
            result_text += f"**Overall Sentiment:** {sentiment_result['label']}\n"
            result_text += f"**Confidence:** {sentiment_result['score']:.3f}\n\n"
            result_text += f"**Analyzed Text:**\n{text[:200]}{'...' if len(text) > 200 else ''}\n\n"
            result_text += f"📊 **Detailed Breakdown:**\n"
            result_text += f"• Positive: {sentiment_result['positive']:.3f}\n"
            result_text += f"• Neutral: {sentiment_result['neutral']:.3f}\n"
            result_text += f"• Negative: {sentiment_result['negative']:.3f}"
            
            return result_text, None, "✅ Analysis Complete"
        
        except Exception as e:
            raise e
    
    def get_debug_dashboard(self) -> Dict[str, Any]:
        """Get debug dashboard data"""
        return self.debugger.get_debug_summary()
    
    def force_garbage_collection(self) -> Dict[str, Any]:
        """Force garbage collection"""
        return self.debugger.memory_tracker.force_garbage_collection()
    
    def create_advanced_interface(self) -> gr.Blocks:
        """Create advanced debugging Gradio interface"""
        
        with gr.Blocks(title="🔧 Advanced Error Handling & Debugging", theme="default") as interface:
            
            gr.HTML("""
            <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 15px; margin-bottom: 20px;">
                <h1>🔧 Advanced Error Handling & Debugging</h1>
                <p>Comprehensive error analysis, performance profiling, and debugging capabilities</p>
            </div>
            """)
            
            with gr.Tabs():
                
                # Debug Testing Tab
                with gr.Tab("🧪 Debug Testing"):
                    gr.Markdown("### 🧪 Test Functions with Advanced Debugging")
                    
                    with gr.Row():
                        with gr.Column(scale=2):
                            debug_prompt_input = gr.Textbox(
                                label="🎯 Test Prompt",
                                placeholder="Enter text for testing...",
                                lines=4
                            )
                            
                            with gr.Row():
                                debug_max_length = gr.Slider(
                                    minimum=10, maximum=2000, value=100,
                                    step=10, label="📏 Max Length"
                                )
                                debug_temperature = gr.Slider(
                                    minimum=0.1, maximum=3.0, value=0.7,
                                    step=0.1, label="🌡️ Temperature"
                                )
                            
                            debug_model_name = gr.Dropdown(
                                choices=["gpt2", "bert-base-uncased", "roberta-base"],
                                value="gpt2", 
                                label="🤖 Model"
                            )
                            
                            debug_test_btn = gr.Button("🧪 Test Generation", variant="primary")
                            debug_sentiment_btn = gr.Button("🧪 Test Sentiment", variant="primary")
                        
                        with gr.Column(scale=2):
                            debug_output = gr.Textbox(
                                label="📊 Debug Output",
                                lines=12,
                                interactive=False
                            )
                            debug_status = gr.HTML(label="📈 Status")
                
                # Performance Monitoring Tab
                with gr.Tab("📊 Performance"):
                    gr.Markdown("### 📊 Performance Monitoring & Profiling")
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            performance_refresh_btn = gr.Button("🔄 Refresh Performance", variant="primary")
                            gc_btn = gr.Button("🗑️ Force Garbage Collection", variant="secondary")
                        
                        with gr.Column(scale=2):
                            performance_output = gr.JSON(label="📊 Performance Data")
                            gc_output = gr.JSON(label="🗑️ Garbage Collection Results")
                
                # Error Analysis Tab
                with gr.Tab("🔍 Error Analysis"):
                    gr.Markdown("### 🔍 Error Analysis & Pattern Detection")
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            error_refresh_btn = gr.Button("🔄 Refresh Error Analysis", variant="primary")
                            clear_logs_btn = gr.Button("🧹 Clear Debug Logs", variant="secondary")
                        
                        with gr.Column(scale=2):
                            error_analysis_output = gr.JSON(label="🔍 Error Analysis")
                            debug_log_output = gr.JSON(label="📝 Debug Log")
                
                # System Monitoring Tab
                with gr.Tab("🖥️ System"):
                    gr.Markdown("### 🖥️ System Resource Monitoring")
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            system_refresh_btn = gr.Button("🔄 Refresh System Info", variant="primary")
                        
                        with gr.Column(scale=2):
                            system_info_output = gr.JSON(label="🖥️ System Information")
                            memory_info_output = gr.JSON(label="💾 Memory Information")
                            cpu_info_output = gr.JSON(label="⚡ CPU Information")
            
            # Event handlers
            debug_test_btn.click(
                fn=self.debug_text_generation,
                inputs=[debug_prompt_input, debug_max_length, debug_temperature, debug_model_name],
                outputs=[debug_output, None, debug_status]
            )
            
            debug_sentiment_btn.click(
                fn=self.debug_sentiment_analysis,
                inputs=[debug_prompt_input],
                outputs=[debug_output, None, debug_status]
            )
            
            performance_refresh_btn.click(
                fn=self.get_debug_dashboard,
                inputs=[],
                outputs=[performance_output]
            )
            
            gc_btn.click(
                fn=self.force_garbage_collection,
                inputs=[],
                outputs=[gc_output]
            )
            
            error_refresh_btn.click(
                fn=self.get_debug_dashboard,
                inputs=[],
                outputs=[error_analysis_output]
            )
            
            clear_logs_btn.click(
                fn=self.debugger.clear_debug_log,
                inputs=[],
                outputs=[debug_log_output]
            )
            
            system_refresh_btn.click(
                fn=self.get_debug_dashboard,
                inputs=[],
                outputs=[system_info_output]
            )
        
        return interface


def create_advanced_debugging_app():
    """Create and launch the advanced debugging app"""
    advanced_interface = AdvancedErrorHandlingGradioInterface(DebugLevel.DETAILED)
    interface = advanced_interface.create_advanced_interface()
    return interface


if __name__ == "__main__":
    # Create and launch the advanced debugging app
    app = create_advanced_debugging_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7862,
        share=True,
        debug=True
    )




