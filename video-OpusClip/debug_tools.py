"""
Debug Tools for Video-OpusClip

Comprehensive debugging and troubleshooting tools:
- Interactive debugger
- Performance profiler
- Memory analyzer
- Error analyzer
- System diagnostics
- Log analyzer
- Troubleshooting wizard
"""

import sys
import os
import time
import traceback
import logging
import json
import psutil
import gc
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from functools import wraps
import inspect
import linecache

# Import existing components
from optimized_config import get_config
from error_handling import ErrorHandler, ErrorType, ErrorSeverity
from logging_config import setup_logging
from performance_monitor import PerformanceMonitor

# =============================================================================
# INTERACTIVE DEBUGGER
# =============================================================================

class VideoOpusClipDebugger:
    """Interactive debugger for Video-OpusClip system."""
    
    def __init__(self):
        self.config = get_config()
        self.logger = setup_logging("debugger")
        self.breakpoints = {}
        self.watch_variables = {}
        self.debug_history = []
        self.is_debugging = False
        
        # Debug configuration
        self.debug_config = {
            "enable_trace": True,
            "enable_profiling": True,
            "enable_memory_tracking": True,
            "enable_error_tracking": True,
            "log_level": "DEBUG",
            "max_history": 1000
        }
    
    def set_breakpoint(self, function_name: str, condition: Optional[Callable] = None):
        """Set a breakpoint on a function."""
        self.breakpoints[function_name] = {
            "condition": condition,
            "hit_count": 0,
            "last_hit": None
        }
        self.logger.info(f"Breakpoint set on function: {function_name}")
    
    def remove_breakpoint(self, function_name: str):
        """Remove a breakpoint."""
        if function_name in self.breakpoints:
            del self.breakpoints[function_name]
            self.logger.info(f"Breakpoint removed from function: {function_name}")
    
    def add_watch_variable(self, variable_name: str, expression: str):
        """Add a variable to watch during debugging."""
        self.watch_variables[variable_name] = {
            "expression": expression,
            "values": [],
            "last_value": None
        }
        self.logger.info(f"Watch variable added: {variable_name}")
    
    def debug_function(self, func: Callable) -> Callable:
        """Decorator to add debugging to a function."""
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            
            # Check if breakpoint is set
            if func_name in self.breakpoints:
                bp = self.breakpoints[func_name]
                bp["hit_count"] += 1
                bp["last_hit"] = datetime.now()
                
                if bp["condition"] is None or bp["condition"](*args, **kwargs):
                    self._handle_breakpoint(func_name, args, kwargs)
            
            # Start debugging session
            self.is_debugging = True
            start_time = time.time()
            
            try:
                # Execute function with tracing
                if self.debug_config["enable_trace"]:
                    result = self._trace_execution(func, args, kwargs)
                else:
                    result = func(*args, **kwargs)
                
                execution_time = time.time() - start_time
                
                # Record debug information
                debug_info = {
                    "function": func_name,
                    "args": str(args),
                    "kwargs": str(kwargs),
                    "result": str(result),
                    "execution_time": execution_time,
                    "timestamp": datetime.now(),
                    "success": True
                }
                
                self.debug_history.append(debug_info)
                
                # Check watch variables
                self._check_watch_variables(locals())
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                
                # Record error information
                error_info = {
                    "function": func_name,
                    "args": str(args),
                    "kwargs": str(kwargs),
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "execution_time": execution_time,
                    "timestamp": datetime.now(),
                    "success": False
                }
                
                self.debug_history.append(error_info)
                self.logger.error(f"Error in {func_name}: {e}")
                
                raise
            finally:
                self.is_debugging = False
        
        return wrapper
    
    def _handle_breakpoint(self, func_name: str, args: tuple, kwargs: dict):
        """Handle breakpoint hit."""
        print(f"\n🔍 BREAKPOINT HIT: {func_name}")
        print(f"Arguments: {args}")
        print(f"Keyword arguments: {kwargs}")
        
        # Interactive debugging
        while True:
            command = input("\nDebug> ").strip().lower()
            
            if command in ['c', 'continue']:
                break
            elif command in ['s', 'step']:
                # Step through execution
                break
            elif command in ['p', 'print']:
                var_name = input("Variable name: ").strip()
                if var_name in locals():
                    print(f"{var_name} = {locals()[var_name]}")
                else:
                    print(f"Variable '{var_name}' not found")
            elif command in ['h', 'help']:
                self._show_debug_help()
            elif command in ['q', 'quit']:
                sys.exit(0)
            else:
                print("Unknown command. Type 'help' for available commands.")
    
    def _trace_execution(self, func: Callable, args: tuple, kwargs: dict):
        """Trace function execution."""
        self.logger.debug(f"Entering function: {func.__name__}")
        
        # Execute function
        result = func(*args, **kwargs)
        
        self.logger.debug(f"Exiting function: {func.__name__}")
        return result
    
    def _check_watch_variables(self, local_vars: dict):
        """Check watch variables for changes."""
        for var_name, watch_info in self.watch_variables.items():
            if var_name in local_vars:
                current_value = local_vars[var_name]
                if current_value != watch_info["last_value"]:
                    watch_info["values"].append({
                        "value": current_value,
                        "timestamp": datetime.now()
                    })
                    watch_info["last_value"] = current_value
                    
                    self.logger.debug(f"Watch variable '{var_name}' changed: {current_value}")
    
    def _show_debug_help(self):
        """Show debug command help."""
        help_text = """
        Debug Commands:
        c, continue  - Continue execution
        s, step      - Step through execution
        p, print     - Print variable value
        h, help      - Show this help
        q, quit      - Quit debugging
        """
        print(help_text)
    
    def get_debug_report(self) -> Dict[str, Any]:
        """Get comprehensive debug report."""
        return {
            "timestamp": datetime.now().isoformat(),
            "breakpoints": self.breakpoints,
            "watch_variables": self.watch_variables,
            "debug_history": self.debug_history[-100:],  # Last 100 entries
            "config": self.debug_config,
            "statistics": self._calculate_debug_statistics()
        }
    
    def _calculate_debug_statistics(self) -> Dict[str, Any]:
        """Calculate debug statistics."""
        if not self.debug_history:
            return {}
        
        successful_calls = [h for h in self.debug_history if h["success"]]
        failed_calls = [h for h in self.debug_history if not h["success"]]
        
        return {
            "total_calls": len(self.debug_history),
            "successful_calls": len(successful_calls),
            "failed_calls": len(failed_calls),
            "success_rate": len(successful_calls) / len(self.debug_history) if self.debug_history else 0,
            "average_execution_time": sum(h["execution_time"] for h in self.debug_history) / len(self.debug_history) if self.debug_history else 0,
            "most_called_function": self._get_most_called_function(),
            "most_error_prone_function": self._get_most_error_prone_function()
        }
    
    def _get_most_called_function(self) -> Optional[str]:
        """Get the most frequently called function."""
        if not self.debug_history:
            return None
        
        function_counts = {}
        for entry in self.debug_history:
            func_name = entry["function"]
            function_counts[func_name] = function_counts.get(func_name, 0) + 1
        
        return max(function_counts.items(), key=lambda x: x[1])[0] if function_counts else None
    
    def _get_most_error_prone_function(self) -> Optional[str]:
        """Get the function with the most errors."""
        if not self.debug_history:
            return None
        
        error_counts = {}
        for entry in self.debug_history:
            if not entry["success"]:
                func_name = entry["function"]
                error_counts[func_name] = error_counts.get(func_name, 0) + 1
        
        return max(error_counts.items(), key=lambda x: x[1])[0] if error_counts else None

# =============================================================================
# PERFORMANCE PROFILER
# =============================================================================

class PerformanceProfiler:
    """Performance profiler for Video-OpusClip system."""
    
    def __init__(self):
        self.config = get_config()
        self.logger = setup_logging("profiler")
        self.profiles = {}
        self.active_profiles = {}
        self.profiling_enabled = True
        
    def start_profile(self, profile_name: str):
        """Start profiling a specific operation."""
        if not self.profiling_enabled:
            return
        
        self.active_profiles[profile_name] = {
            "start_time": time.time(),
            "start_memory": psutil.Process().memory_info().rss,
            "start_cpu": psutil.cpu_percent(),
            "calls": 0,
            "errors": 0
        }
        
        self.logger.debug(f"Started profiling: {profile_name}")
    
    def end_profile(self, profile_name: str, success: bool = True):
        """End profiling and record results."""
        if not self.profiling_enabled or profile_name not in self.active_profiles:
            return
        
        profile = self.active_profiles[profile_name]
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss
        end_cpu = psutil.cpu_percent()
        
        profile_data = {
            "duration": end_time - profile["start_time"],
            "memory_delta": end_memory - profile["start_memory"],
            "cpu_usage": (profile["start_cpu"] + end_cpu) / 2,
            "success": success,
            "timestamp": datetime.now()
        }
        
        if profile_name not in self.profiles:
            self.profiles[profile_name] = []
        
        self.profiles[profile_name].append(profile_data)
        
        if not success:
            profile["errors"] += 1
        
        profile["calls"] += 1
        
        del self.active_profiles[profile_name]
        
        self.logger.debug(f"Ended profiling: {profile_name} - Duration: {profile_data['duration']:.3f}s")
    
    def profile_function(self, profile_name: str = None):
        """Decorator to profile a function."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                name = profile_name or func.__name__
                self.start_profile(name)
                
                try:
                    result = func(*args, **kwargs)
                    self.end_profile(name, success=True)
                    return result
                except Exception as e:
                    self.end_profile(name, success=False)
                    raise
            
            return wrapper
        return decorator
    
    def get_profile_report(self, profile_name: str = None) -> Dict[str, Any]:
        """Get profiling report."""
        if profile_name:
            if profile_name not in self.profiles:
                return {"error": f"Profile '{profile_name}' not found"}
            
            profile_data = self.profiles[profile_name]
            return self._analyze_profile(profile_name, profile_data)
        else:
            return {
                "profiles": list(self.profiles.keys()),
                "active_profiles": list(self.active_profiles.keys()),
                "summary": self._get_summary_statistics()
            }
    
    def _analyze_profile(self, name: str, data: List[Dict]) -> Dict[str, Any]:
        """Analyze profile data."""
        if not data:
            return {"error": "No profile data available"}
        
        durations = [d["duration"] for d in data]
        memory_deltas = [d["memory_delta"] for d in data]
        success_count = sum(1 for d in data if d["success"])
        
        return {
            "name": name,
            "total_calls": len(data),
            "successful_calls": success_count,
            "error_count": len(data) - success_count,
            "success_rate": success_count / len(data),
            "duration": {
                "min": min(durations),
                "max": max(durations),
                "avg": sum(durations) / len(durations),
                "median": sorted(durations)[len(durations) // 2]
            },
            "memory": {
                "min": min(memory_deltas),
                "max": max(memory_deltas),
                "avg": sum(memory_deltas) / len(memory_deltas)
            },
            "recent_calls": data[-10:]  # Last 10 calls
        }
    
    def _get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics for all profiles."""
        all_durations = []
        all_memory_deltas = []
        total_calls = 0
        total_errors = 0
        
        for profile_data in self.profiles.values():
            for data in profile_data:
                all_durations.append(data["duration"])
                all_memory_deltas.append(data["memory_delta"])
                total_calls += 1
                if not data["success"]:
                    total_errors += 1
        
        if not all_durations:
            return {"error": "No profiling data available"}
        
        return {
            "total_calls": total_calls,
            "total_errors": total_errors,
            "error_rate": total_errors / total_calls if total_calls > 0 else 0,
            "average_duration": sum(all_durations) / len(all_durations),
            "average_memory_delta": sum(all_memory_deltas) / len(all_memory_deltas),
            "total_profiles": len(self.profiles)
        }

# =============================================================================
# MEMORY ANALYZER
# =============================================================================

class MemoryAnalyzer:
    """Memory usage analyzer for Video-OpusClip system."""
    
    def __init__(self):
        self.config = get_config()
        self.logger = setup_logging("memory_analyzer")
        self.memory_snapshots = []
        self.memory_threshold = 1024 * 1024 * 1024  # 1GB
        self.analysis_enabled = True
        
    def take_snapshot(self, label: str = None):
        """Take a memory snapshot."""
        if not self.analysis_enabled:
            return
        
        process = psutil.Process()
        memory_info = process.memory_info()
        
        snapshot = {
            "timestamp": datetime.now(),
            "label": label or "snapshot",
            "rss": memory_info.rss,  # Resident Set Size
            "vms": memory_info.vms,  # Virtual Memory Size
            "percent": process.memory_percent(),
            "available": psutil.virtual_memory().available,
            "total": psutil.virtual_memory().total,
            "gc_stats": self._get_gc_stats()
        }
        
        self.memory_snapshots.append(snapshot)
        
        # Check for memory threshold
        if memory_info.rss > self.memory_threshold:
            self.logger.warning(f"Memory usage exceeded threshold: {memory_info.rss / (1024**3):.2f}GB")
        
        self.logger.debug(f"Memory snapshot taken: {snapshot['rss'] / (1024**2):.2f}MB")
    
    def _get_gc_stats(self) -> Dict[str, Any]:
        """Get garbage collector statistics."""
        return {
            "collections": gc.get_stats(),
            "counts": gc.get_count(),
            "thresholds": gc.get_threshold()
        }
    
    def analyze_memory_usage(self) -> Dict[str, Any]:
        """Analyze memory usage patterns."""
        if not self.memory_snapshots:
            return {"error": "No memory snapshots available"}
        
        rss_values = [s["rss"] for s in self.memory_snapshots]
        vms_values = [s["vms"] for s in self.memory_snapshots]
        
        return {
            "snapshots_count": len(self.memory_snapshots),
            "rss": {
                "min": min(rss_values),
                "max": max(rss_values),
                "avg": sum(rss_values) / len(rss_values),
                "current": rss_values[-1] if rss_values else 0
            },
            "vms": {
                "min": min(vms_values),
                "max": max(vms_values),
                "avg": sum(vms_values) / len(vms_values),
                "current": vms_values[-1] if vms_values else 0
            },
            "memory_growth": self._calculate_memory_growth(),
            "leak_detection": self._detect_memory_leaks(),
            "recommendations": self._generate_memory_recommendations()
        }
    
    def _calculate_memory_growth(self) -> Dict[str, Any]:
        """Calculate memory growth rate."""
        if len(self.memory_snapshots) < 2:
            return {"error": "Insufficient snapshots for growth analysis"}
        
        first_snapshot = self.memory_snapshots[0]
        last_snapshot = self.memory_snapshots[-1]
        
        time_diff = (last_snapshot["timestamp"] - first_snapshot["timestamp"]).total_seconds()
        memory_diff = last_snapshot["rss"] - first_snapshot["rss"]
        
        growth_rate = memory_diff / time_diff if time_diff > 0 else 0
        
        return {
            "total_growth": memory_diff,
            "growth_rate": growth_rate,  # bytes per second
            "time_period": time_diff,
            "is_growing": growth_rate > 0
        }
    
    def _detect_memory_leaks(self) -> Dict[str, Any]:
        """Detect potential memory leaks."""
        if len(self.memory_snapshots) < 10:
            return {"error": "Insufficient snapshots for leak detection"}
        
        # Simple leak detection: check for consistent growth
        recent_snapshots = self.memory_snapshots[-10:]
        rss_values = [s["rss"] for s in recent_snapshots]
        
        # Calculate trend
        trend = self._calculate_trend(rss_values)
        
        return {
            "trend": trend,
            "potential_leak": trend > 0.1,  # 10% growth threshold
            "growth_percentage": trend * 100,
            "recommendation": "Check for memory leaks" if trend > 0.1 else "Memory usage stable"
        }
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend of values."""
        if len(values) < 2:
            return 0
        
        n = len(values)
        x_sum = sum(range(n))
        y_sum = sum(values)
        xy_sum = sum(i * val for i, val in enumerate(values))
        x2_sum = sum(i * i for i in range(n))
        
        slope = (n * xy_sum - x_sum * y_sum) / (n * x2_sum - x_sum * x_sum)
        return slope
    
    def _generate_memory_recommendations(self) -> List[str]:
        """Generate memory optimization recommendations."""
        recommendations = []
        
        if not self.memory_snapshots:
            return ["Take memory snapshots to generate recommendations"]
        
        current_memory = self.memory_snapshots[-1]["rss"]
        
        if current_memory > 2 * 1024 * 1024 * 1024:  # 2GB
            recommendations.append("High memory usage detected. Consider optimizing data structures.")
        
        if current_memory > 4 * 1024 * 1024 * 1024:  # 4GB
            recommendations.append("Very high memory usage. Consider implementing memory pooling.")
        
        # Check for memory growth
        growth_analysis = self._calculate_memory_growth()
        if growth_analysis.get("is_growing", False):
            recommendations.append("Memory usage is growing. Check for memory leaks.")
        
        # Check GC stats
        gc_stats = self.memory_snapshots[-1]["gc_stats"]
        if gc_stats["counts"][0] > 100:  # High collection count
            recommendations.append("High garbage collection activity. Consider object pooling.")
        
        return recommendations
    
    def force_garbage_collection(self):
        """Force garbage collection and take snapshot."""
        self.logger.info("Forcing garbage collection...")
        
        # Collect all generations
        collected = gc.collect()
        
        self.logger.info(f"Garbage collection completed. Collected {collected} objects.")
        
        # Take snapshot after GC
        self.take_snapshot("after_gc")
        
        return collected

# =============================================================================
# ERROR ANALYZER
# =============================================================================

class ErrorAnalyzer:
    """Error analysis and pattern detection for Video-OpusClip system."""
    
    def __init__(self):
        self.config = get_config()
        self.logger = setup_logging("error_analyzer")
        self.errors = []
        self.error_patterns = {}
        self.analysis_enabled = True
        
    def record_error(self, error: Exception, context: str = "", stack_trace: str = ""):
        """Record an error for analysis."""
        if not self.analysis_enabled:
            return
        
        error_info = {
            "timestamp": datetime.now(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
            "stack_trace": stack_trace or traceback.format_exc(),
            "severity": self._determine_severity(error)
        }
        
        self.errors.append(error_info)
        
        # Update error patterns
        self._update_error_patterns(error_info)
        
        self.logger.debug(f"Error recorded: {error_info['error_type']} in {context}")
    
    def _determine_severity(self, error: Exception) -> str:
        """Determine error severity."""
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            return "CRITICAL"
        elif isinstance(error, (MemoryError, OSError)):
            return "HIGH"
        elif isinstance(error, (ValueError, TypeError, AttributeError)):
            return "MEDIUM"
        else:
            return "LOW"
    
    def _update_error_patterns(self, error_info: Dict[str, Any]):
        """Update error pattern analysis."""
        error_key = f"{error_info['error_type']}:{error_info['context']}"
        
        if error_key not in self.error_patterns:
            self.error_patterns[error_key] = {
                "count": 0,
                "first_occurrence": error_info["timestamp"],
                "last_occurrence": error_info["timestamp"],
                "severities": [],
                "examples": []
            }
        
        pattern = self.error_patterns[error_key]
        pattern["count"] += 1
        pattern["last_occurrence"] = error_info["timestamp"]
        pattern["severities"].append(error_info["severity"])
        
        # Keep only recent examples
        if len(pattern["examples"]) < 5:
            pattern["examples"].append(error_info["error_message"])
    
    def analyze_errors(self) -> Dict[str, Any]:
        """Analyze recorded errors."""
        if not self.errors:
            return {"error": "No errors recorded"}
        
        return {
            "total_errors": len(self.errors),
            "error_types": self._analyze_error_types(),
            "error_patterns": self._analyze_error_patterns(),
            "severity_distribution": self._analyze_severity_distribution(),
            "temporal_analysis": self._analyze_temporal_patterns(),
            "recommendations": self._generate_error_recommendations()
        }
    
    def _analyze_error_types(self) -> Dict[str, Any]:
        """Analyze error types."""
        type_counts = {}
        for error in self.errors:
            error_type = error["error_type"]
            type_counts[error_type] = type_counts.get(error_type, 0) + 1
        
        return {
            "distribution": type_counts,
            "most_common": max(type_counts.items(), key=lambda x: x[1]) if type_counts else None,
            "total_types": len(type_counts)
        }
    
    def _analyze_error_patterns(self) -> Dict[str, Any]:
        """Analyze error patterns."""
        patterns = []
        for pattern_key, pattern_data in self.error_patterns.items():
            patterns.append({
                "pattern": pattern_key,
                "count": pattern_data["count"],
                "frequency": pattern_data["count"] / len(self.errors),
                "first_seen": pattern_data["first_occurrence"],
                "last_seen": pattern_data["last_occurrence"],
                "most_common_severity": max(set(pattern_data["severities"]), key=pattern_data["severities"].count) if pattern_data["severities"] else "UNKNOWN"
            })
        
        # Sort by frequency
        patterns.sort(key=lambda x: x["count"], reverse=True)
        
        return {
            "patterns": patterns,
            "most_frequent": patterns[0] if patterns else None,
            "recent_patterns": [p for p in patterns if (datetime.now() - p["last_seen"]).days < 1]
        }
    
    def _analyze_severity_distribution(self) -> Dict[str, Any]:
        """Analyze error severity distribution."""
        severity_counts = {}
        for error in self.errors:
            severity = error["severity"]
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return {
            "distribution": severity_counts,
            "critical_errors": severity_counts.get("CRITICAL", 0),
            "high_severity_errors": severity_counts.get("HIGH", 0),
            "total_errors": len(self.errors)
        }
    
    def _analyze_temporal_patterns(self) -> Dict[str, Any]:
        """Analyze temporal error patterns."""
        if len(self.errors) < 2:
            return {"error": "Insufficient data for temporal analysis"}
        
        # Group errors by hour
        hourly_distribution = {}
        for error in self.errors:
            hour = error["timestamp"].hour
            hourly_distribution[hour] = hourly_distribution.get(hour, 0) + 1
        
        # Find peak error hours
        peak_hour = max(hourly_distribution.items(), key=lambda x: x[1]) if hourly_distribution else None
        
        return {
            "hourly_distribution": hourly_distribution,
            "peak_error_hour": peak_hour,
            "error_rate": len(self.errors) / ((self.errors[-1]["timestamp"] - self.errors[0]["timestamp"]).total_seconds() / 3600)  # errors per hour
        }
    
    def _generate_error_recommendations(self) -> List[str]:
        """Generate error handling recommendations."""
        recommendations = []
        
        if not self.errors:
            return ["No errors recorded. System appears stable."]
        
        # Analyze patterns
        patterns = self._analyze_error_patterns()
        if patterns["most_frequent"]:
            most_frequent = patterns["most_frequent"]
            if most_frequent["count"] > 10:
                recommendations.append(f"High frequency error pattern detected: {most_frequent['pattern']}. Consider implementing specific handling.")
        
        # Check severity
        severity_analysis = self._analyze_severity_distribution()
        if severity_analysis["critical_errors"] > 0:
            recommendations.append("Critical errors detected. Review system stability and implement fail-safes.")
        
        if severity_analysis["high_severity_errors"] > 5:
            recommendations.append("Multiple high-severity errors. Implement comprehensive error recovery mechanisms.")
        
        # Check temporal patterns
        temporal_analysis = self._analyze_temporal_patterns()
        if temporal_analysis.get("error_rate", 0) > 10:  # More than 10 errors per hour
            recommendations.append("High error rate detected. Investigate root causes and implement monitoring.")
        
        return recommendations

# =============================================================================
# SYSTEM DIAGNOSTICS
# =============================================================================

class SystemDiagnostics:
    """System diagnostics and health check for Video-OpusClip."""
    
    def __init__(self):
        self.config = get_config()
        self.logger = setup_logging("system_diagnostics")
        self.diagnostics_history = []
        
    def run_full_diagnostics(self) -> Dict[str, Any]:
        """Run comprehensive system diagnostics."""
        self.logger.info("Running full system diagnostics...")
        
        diagnostics = {
            "timestamp": datetime.now().isoformat(),
            "system_info": self._get_system_info(),
            "resource_usage": self._check_resource_usage(),
            "disk_space": self._check_disk_space(),
            "network_status": self._check_network_status(),
            "python_environment": self._check_python_environment(),
            "dependencies": self._check_dependencies(),
            "configuration": self._check_configuration(),
            "health_score": 0,
            "recommendations": []
        }
        
        # Calculate health score
        health_score = self._calculate_health_score(diagnostics)
        diagnostics["health_score"] = health_score
        
        # Generate recommendations
        diagnostics["recommendations"] = self._generate_diagnostics_recommendations(diagnostics)
        
        # Store in history
        self.diagnostics_history.append(diagnostics)
        
        self.logger.info(f"Diagnostics completed. Health score: {health_score}/100")
        
        return diagnostics
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        import platform
        
        return {
            "platform": platform.system(),
            "platform_version": platform.version(),
            "architecture": platform.architecture(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "hostname": platform.node()
        }
    
    def _check_resource_usage(self) -> Dict[str, Any]:
        """Check system resource usage."""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            "cpu_usage": cpu_percent,
            "memory_usage": memory.percent,
            "memory_available": memory.available,
            "memory_total": memory.total,
            "disk_usage": disk.percent,
            "disk_free": disk.free,
            "disk_total": disk.total
        }
    
    def _check_disk_space(self) -> Dict[str, Any]:
        """Check disk space and performance."""
        disk_io = psutil.disk_io_counters()
        
        return {
            "read_bytes": disk_io.read_bytes if disk_io else 0,
            "write_bytes": disk_io.write_bytes if disk_io else 0,
            "read_count": disk_io.read_count if disk_io else 0,
            "write_count": disk_io.write_count if disk_io else 0
        }
    
    def _check_network_status(self) -> Dict[str, Any]:
        """Check network connectivity and performance."""
        try:
            import socket
            import urllib.request
            
            # Test basic connectivity
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            network_status = "Connected"
        except OSError:
            network_status = "Disconnected"
        
        network_io = psutil.net_io_counters()
        
        return {
            "status": network_status,
            "bytes_sent": network_io.bytes_sent if network_io else 0,
            "bytes_recv": network_io.bytes_recv if network_io else 0,
            "packets_sent": network_io.packets_sent if network_io else 0,
            "packets_recv": network_io.packets_recv if network_io else 0
        }
    
    def _check_python_environment(self) -> Dict[str, Any]:
        """Check Python environment."""
        return {
            "version": sys.version,
            "executable": sys.executable,
            "path": sys.path,
            "modules_loaded": len(sys.modules),
            "threads": threading.active_count()
        }
    
    def _check_dependencies(self) -> Dict[str, Any]:
        """Check critical dependencies."""
        dependencies = {}
        
        critical_packages = [
            "torch", "gradio", "numpy", "opencv-python", 
            "transformers", "diffusers", "fastapi"
        ]
        
        for package in critical_packages:
            try:
                module = __import__(package.replace("-", "_"))
                dependencies[package] = {
                    "installed": True,
                    "version": getattr(module, "__version__", "Unknown")
                }
            except ImportError:
                dependencies[package] = {
                    "installed": False,
                    "version": None
                }
        
        return dependencies
    
    def _check_configuration(self) -> Dict[str, Any]:
        """Check system configuration."""
        return {
            "config_loaded": self.config is not None,
            "gpu_enabled": getattr(self.config, "gpu_enabled", False),
            "debug_mode": getattr(self.config, "debug_mode", False),
            "cache_enabled": getattr(self.config, "cache_enabled", False)
        }
    
    def _calculate_health_score(self, diagnostics: Dict[str, Any]) -> int:
        """Calculate system health score (0-100)."""
        score = 100
        
        # Resource usage penalties
        resource_usage = diagnostics["resource_usage"]
        if resource_usage["cpu_usage"] > 90:
            score -= 20
        elif resource_usage["cpu_usage"] > 80:
            score -= 10
        
        if resource_usage["memory_usage"] > 90:
            score -= 20
        elif resource_usage["memory_usage"] > 80:
            score -= 10
        
        if resource_usage["disk_usage"] > 90:
            score -= 15
        elif resource_usage["disk_usage"] > 80:
            score -= 5
        
        # Network penalties
        if diagnostics["network_status"]["status"] == "Disconnected":
            score -= 30
        
        # Dependency penalties
        dependencies = diagnostics["dependencies"]
        missing_deps = sum(1 for dep in dependencies.values() if not dep["installed"])
        score -= missing_deps * 10
        
        # Configuration penalties
        if not diagnostics["configuration"]["config_loaded"]:
            score -= 20
        
        return max(0, score)
    
    def _generate_diagnostics_recommendations(self, diagnostics: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on diagnostics."""
        recommendations = []
        
        resource_usage = diagnostics["resource_usage"]
        
        if resource_usage["cpu_usage"] > 80:
            recommendations.append("High CPU usage detected. Consider optimizing processing or scaling resources.")
        
        if resource_usage["memory_usage"] > 80:
            recommendations.append("High memory usage detected. Consider implementing memory optimization.")
        
        if resource_usage["disk_usage"] > 80:
            recommendations.append("Low disk space. Consider cleanup or expanding storage.")
        
        if diagnostics["network_status"]["status"] == "Disconnected":
            recommendations.append("Network connectivity issues detected. Check network configuration.")
        
        dependencies = diagnostics["dependencies"]
        missing_deps = [name for name, info in dependencies.items() if not info["installed"]]
        if missing_deps:
            recommendations.append(f"Missing dependencies: {', '.join(missing_deps)}. Install required packages.")
        
        if diagnostics["health_score"] < 50:
            recommendations.append("System health is poor. Review all recommendations and take action.")
        
        return recommendations

# =============================================================================
# DEBUG UTILITIES
# =============================================================================

def debug_print(*args, **kwargs):
    """Enhanced debug print function."""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] DEBUG:", *args, **kwargs)

def debug_function_call(func_name: str, args: tuple, kwargs: dict):
    """Debug function call information."""
    debug_print(f"Calling {func_name}")
    debug_print(f"  Args: {args}")
    debug_print(f"  Kwargs: {kwargs}")

def debug_function_return(func_name: str, result: Any, execution_time: float):
    """Debug function return information."""
    debug_print(f"Returning from {func_name}")
    debug_print(f"  Result: {result}")
    debug_print(f"  Execution time: {execution_time:.3f}s")

def debug_error(func_name: str, error: Exception, execution_time: float):
    """Debug error information."""
    debug_print(f"Error in {func_name}")
    debug_print(f"  Error: {error}")
    debug_print(f"  Execution time: {execution_time:.3f}s")
    debug_print(f"  Traceback: {traceback.format_exc()}")

# =============================================================================
# MAIN DEBUG MANAGER
# =============================================================================

class DebugManager:
    """Main debug manager that coordinates all debugging tools."""
    
    def __init__(self):
        self.config = get_config()
        self.logger = setup_logging("debug_manager")
        
        # Initialize all debug tools
        self.debugger = VideoOpusClipDebugger()
        self.profiler = PerformanceProfiler()
        self.memory_analyzer = MemoryAnalyzer()
        self.error_analyzer = ErrorAnalyzer()
        self.system_diagnostics = SystemDiagnostics()
        
        self.debug_enabled = True
        
    def enable_debugging(self):
        """Enable all debugging features."""
        self.debug_enabled = True
        self.logger.info("Debugging enabled")
    
    def disable_debugging(self):
        """Disable all debugging features."""
        self.debug_enabled = False
        self.logger.info("Debugging disabled")
    
    def run_comprehensive_debug(self) -> Dict[str, Any]:
        """Run comprehensive debugging analysis."""
        if not self.debug_enabled:
            return {"error": "Debugging is disabled"}
        
        self.logger.info("Running comprehensive debug analysis...")
        
        # Take memory snapshot
        self.memory_analyzer.take_snapshot("comprehensive_debug_start")
        
        # Run system diagnostics
        diagnostics = self.system_diagnostics.run_full_diagnostics()
        
        # Analyze errors
        error_analysis = self.error_analyzer.analyze_errors()
        
        # Analyze memory
        memory_analysis = self.memory_analyzer.analyze_memory_usage()
        
        # Get profiling data
        profile_report = self.profiler.get_profile_report()
        
        # Get debug report
        debug_report = self.debugger.get_debug_report()
        
        # Take final memory snapshot
        self.memory_analyzer.take_snapshot("comprehensive_debug_end")
        
        comprehensive_report = {
            "timestamp": datetime.now().isoformat(),
            "system_diagnostics": diagnostics,
            "error_analysis": error_analysis,
            "memory_analysis": memory_analysis,
            "performance_profiling": profile_report,
            "debug_information": debug_report,
            "summary": self._generate_debug_summary(diagnostics, error_analysis, memory_analysis)
        }
        
        self.logger.info("Comprehensive debug analysis completed")
        
        return comprehensive_report
    
    def _generate_debug_summary(self, diagnostics: Dict, error_analysis: Dict, memory_analysis: Dict) -> Dict[str, Any]:
        """Generate debug summary."""
        return {
            "system_health": diagnostics.get("health_score", 0),
            "total_errors": error_analysis.get("total_errors", 0),
            "memory_usage": memory_analysis.get("rss", {}).get("current", 0),
            "recommendations": diagnostics.get("recommendations", []) + error_analysis.get("recommendations", []) + memory_analysis.get("recommendations", [])
        }
    
    def get_debug_status(self) -> Dict[str, Any]:
        """Get current debug status."""
        return {
            "debug_enabled": self.debug_enabled,
            "debugger_active": self.debugger.is_debugging,
            "profiling_enabled": self.profiler.profiling_enabled,
            "memory_analysis_enabled": self.memory_analyzer.analysis_enabled,
            "error_analysis_enabled": self.error_analyzer.analysis_enabled,
            "active_profiles": list(self.profiler.active_profiles.keys()),
            "memory_snapshots": len(self.memory_analyzer.memory_snapshots),
            "recorded_errors": len(self.error_analyzer.errors)
        }

# =============================================================================
# USAGE EXAMPLES
# =============================================================================

def example_debug_usage():
    """Example usage of debug tools."""
    
    # Initialize debug manager
    debug_manager = DebugManager()
    
    # Enable debugging
    debug_manager.enable_debugging()
    
    # Example function with debugging
    @debug_manager.debugger.debug_function
    @debug_manager.profiler.profile_function("example_function")
    def example_function(input_data):
        # Take memory snapshot
        debug_manager.memory_analyzer.take_snapshot("function_start")
        
        try:
            # Simulate some processing
            result = input_data * 2
            
            # Take memory snapshot
            debug_manager.memory_analyzer.take_snapshot("function_success")
            
            return result
            
        except Exception as e:
            # Record error
            debug_manager.error_analyzer.record_error(e, "example_function")
            
            # Take memory snapshot
            debug_manager.memory_analyzer.take_snapshot("function_error")
            
            raise
    
    # Run example
    try:
        result = example_function(5)
        print(f"Result: {result}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Run comprehensive debug analysis
    debug_report = debug_manager.run_comprehensive_debug()
    print("Debug report generated")

if __name__ == "__main__":
    example_debug_usage() 