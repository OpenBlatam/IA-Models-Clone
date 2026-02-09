# Debugging Guide for Video-OpusClip

## Overview

This guide covers comprehensive debugging and troubleshooting for the Video-OpusClip system. The debugging tools provide interactive debugging, performance profiling, memory analysis, error tracking, and system diagnostics to help identify and resolve issues quickly.

## Debugging Architecture

### 🛠️ Core Debugging Components

#### 1. **VideoOpusClipDebugger**
Interactive debugger with breakpoints, watch variables, and execution tracing.

#### 2. **PerformanceProfiler**
Performance profiling with timing analysis and bottleneck detection.

#### 3. **MemoryAnalyzer**
Memory usage tracking, leak detection, and optimization recommendations.

#### 4. **ErrorAnalyzer**
Error pattern analysis, severity classification, and trend detection.

#### 5. **SystemDiagnostics**
Comprehensive system health checks and resource monitoring.

#### 6. **DebugManager**
Centralized debug management that coordinates all debugging tools.

## Getting Started with Debugging

### 🚀 Basic Setup

```python
from debug_tools import DebugManager

# Initialize debug manager
debug_manager = DebugManager()

# Enable debugging
debug_manager.enable_debugging()

# Run comprehensive debug analysis
debug_report = debug_manager.run_comprehensive_debug()
print(json.dumps(debug_report, indent=2))
```

### 🔍 Quick Debug Status Check

```python
# Check current debug status
status = debug_manager.get_debug_status()
print(f"Debug enabled: {status['debug_enabled']}")
print(f"Active profiles: {status['active_profiles']}")
print(f"Recorded errors: {status['recorded_errors']}")
```

## Interactive Debugger

### 🎯 Setting Breakpoints

```python
from debug_tools import VideoOpusClipDebugger

debugger = VideoOpusClipDebugger()

# Set breakpoint on function
debugger.set_breakpoint("generate_video")

# Set conditional breakpoint
def error_condition(*args, **kwargs):
    return "error" in str(args).lower()

debugger.set_breakpoint("process_video", error_condition)
```

### 👀 Watch Variables

```python
# Add variables to watch
debugger.add_watch_variable("video_data", "video_data")
debugger.add_watch_variable("processing_time", "time.time() - start_time")

# Function with debugging
@debugger.debug_function
def process_video(video_data):
    start_time = time.time()
    # Processing logic here
    return processed_video
```

### 🔧 Interactive Commands

When a breakpoint is hit, you can use these commands:

- `c` or `continue` - Continue execution
- `s` or `step` - Step through execution
- `p` or `print` - Print variable value
- `h` or `help` - Show available commands
- `q` or `quit` - Quit debugging

## Performance Profiling

### ⏱️ Function Profiling

```python
from debug_tools import PerformanceProfiler

profiler = PerformanceProfiler()

# Profile a specific function
@profiler.profile_function("video_generation")
def generate_video(prompt, duration):
    # Video generation logic
    return video

# Manual profiling
profiler.start_profile("custom_operation")
# ... your code here ...
profiler.end_profile("custom_operation", success=True)
```

### 📊 Profile Analysis

```python
# Get profile report
report = profiler.get_profile_report("video_generation")
print(f"Total calls: {report['total_calls']}")
print(f"Average duration: {report['duration']['avg']:.3f}s")
print(f"Success rate: {report['success_rate']:.2%}")

# Get all profiles summary
summary = profiler.get_profile_report()
print(f"Total profiles: {summary['total_profiles']}")
```

## Memory Analysis

### 📸 Memory Snapshots

```python
from debug_tools import MemoryAnalyzer

analyzer = MemoryAnalyzer()

# Take memory snapshots
analyzer.take_snapshot("before_processing")
# ... your code here ...
analyzer.take_snapshot("after_processing")

# Analyze memory usage
analysis = analyzer.analyze_memory_usage()
print(f"Current memory: {analysis['rss']['current'] / (1024**2):.2f}MB")
print(f"Memory growth: {analysis['memory_growth']['total_growth'] / (1024**2):.2f}MB")
```

### 🐛 Memory Leak Detection

```python
# Force garbage collection and analyze
collected = analyzer.force_garbage_collection()
print(f"Collected {collected} objects")

# Check for memory leaks
leak_analysis = analyzer.analyze_memory_usage()
if leak_analysis['leak_detection']['potential_leak']:
    print("⚠️ Potential memory leak detected!")
    print(f"Growth rate: {leak_analysis['leak_detection']['growth_percentage']:.2f}%")
```

## Error Analysis

### 📝 Recording Errors

```python
from debug_tools import ErrorAnalyzer

analyzer = ErrorAnalyzer()

# Record errors automatically
try:
    result = risky_operation()
except Exception as e:
    analyzer.record_error(e, "risky_operation", traceback.format_exc())
    raise

# Manual error recording
analyzer.record_error(
    ValueError("Custom error"),
    "custom_context",
    "Custom stack trace"
)
```

### 📈 Error Analysis

```python
# Analyze recorded errors
analysis = analyzer.analyze_errors()
print(f"Total errors: {analysis['total_errors']}")
print(f"Most common error: {analysis['error_types']['most_common']}")

# Check error patterns
patterns = analysis['error_patterns']
if patterns['patterns']:
    most_frequent = patterns['most_frequent']
    print(f"Most frequent pattern: {most_frequent['pattern']}")
    print(f"Occurrences: {most_frequent['count']}")
```

## System Diagnostics

### 🔍 Full System Check

```python
from debug_tools import SystemDiagnostics

diagnostics = SystemDiagnostics()

# Run comprehensive diagnostics
health_report = diagnostics.run_full_diagnostics()
print(f"System health score: {health_report['health_score']}/100")

# Check specific areas
system_info = health_report['system_info']
resource_usage = health_report['resource_usage']
print(f"CPU usage: {resource_usage['cpu_usage']:.1f}%")
print(f"Memory usage: {resource_usage['memory_usage']:.1f}%")
```

### 🚨 Health Monitoring

```python
# Check system health
if health_report['health_score'] < 50:
    print("⚠️ System health is poor!")
    for recommendation in health_report['recommendations']:
        print(f"  - {recommendation}")

# Monitor specific resources
if resource_usage['cpu_usage'] > 80:
    print("⚠️ High CPU usage detected")
if resource_usage['memory_usage'] > 90:
    print("⚠️ Critical memory usage")
```

## Comprehensive Debugging

### 🔧 Complete Debug Workflow

```python
def comprehensive_debug_workflow():
    """Complete debugging workflow example."""
    
    debug_manager = DebugManager()
    debug_manager.enable_debugging()
    
    # Set up debugging for critical functions
    @debug_manager.debugger.debug_function
    @debug_manager.profiler.profile_function("critical_operation")
    def critical_operation(data):
        # Take memory snapshot
        debug_manager.memory_analyzer.take_snapshot("operation_start")
        
        try:
            # Your critical operation here
            result = process_data(data)
            
            # Record success
            debug_manager.memory_analyzer.take_snapshot("operation_success")
            return result
            
        except Exception as e:
            # Record error
            debug_manager.error_analyzer.record_error(e, "critical_operation")
            debug_manager.memory_analyzer.take_snapshot("operation_error")
            raise
    
    # Run operation
    try:
        result = critical_operation(test_data)
    except Exception as e:
        print(f"Operation failed: {e}")
    
    # Generate comprehensive report
    debug_report = debug_manager.run_comprehensive_debug()
    return debug_report
```

### 📊 Debug Report Analysis

```python
def analyze_debug_report(report):
    """Analyze comprehensive debug report."""
    
    print("🔍 Debug Report Analysis")
    print("=" * 50)
    
    # System health
    health_score = report['system_diagnostics']['health_score']
    print(f"System Health: {health_score}/100")
    
    # Error analysis
    error_count = report['error_analysis']['total_errors']
    print(f"Total Errors: {error_count}")
    
    # Memory analysis
    memory_usage = report['memory_analysis']['rss']['current']
    print(f"Memory Usage: {memory_usage / (1024**2):.2f}MB")
    
    # Performance analysis
    if report['performance_profiling']['profiles']:
        print(f"Profiled Functions: {len(report['performance_profiling']['profiles'])}")
    
    # Recommendations
    recommendations = report['summary']['recommendations']
    if recommendations:
        print("\n📋 Recommendations:")
        for rec in recommendations:
            print(f"  - {rec}")
```

## Common Debugging Scenarios

### 🎬 Video Generation Issues

```python
def debug_video_generation():
    """Debug video generation issues."""
    
    debug_manager = DebugManager()
    debug_manager.enable_debugging()
    
    # Set breakpoint on video generation
    debug_manager.debugger.set_breakpoint("generate_video")
    
    # Watch critical variables
    debug_manager.debugger.add_watch_variable("prompt", "prompt")
    debug_manager.debugger.add_watch_variable("duration", "duration")
    debug_manager.debugger.add_watch_variable("quality", "quality")
    
    @debug_manager.debugger.debug_function
    @debug_manager.profiler.profile_function("video_generation")
    def generate_video(prompt, duration, quality):
        debug_manager.memory_analyzer.take_snapshot("generation_start")
        
        try:
            # Video generation logic
            result = ai_model.generate(prompt, duration, quality)
            
            debug_manager.memory_analyzer.take_snapshot("generation_success")
            return result
            
        except Exception as e:
            debug_manager.error_analyzer.record_error(e, "video_generation")
            debug_manager.memory_analyzer.take_snapshot("generation_error")
            raise
    
    return generate_video
```

### 🐛 Memory Leak Investigation

```python
def investigate_memory_leak():
    """Investigate potential memory leaks."""
    
    analyzer = MemoryAnalyzer()
    
    # Take baseline snapshot
    analyzer.take_snapshot("baseline")
    
    # Run operations that might cause leaks
    for i in range(100):
        process_large_data()
        analyzer.take_snapshot(f"iteration_{i}")
    
    # Analyze memory growth
    analysis = analyzer.analyze_memory_usage()
    
    if analysis['leak_detection']['potential_leak']:
        print("🚨 Memory leak detected!")
        print(f"Growth rate: {analysis['leak_detection']['growth_percentage']:.2f}%")
        
        # Force garbage collection
        collected = analyzer.force_garbage_collection()
        print(f"Collected {collected} objects")
        
        # Take final snapshot
        analyzer.take_snapshot("after_gc")
        
        # Re-analyze
        final_analysis = analyzer.analyze_memory_usage()
        print(f"Memory after GC: {final_analysis['rss']['current'] / (1024**2):.2f}MB")
    
    return analysis
```

### ⚡ Performance Bottleneck Analysis

```python
def analyze_performance_bottlenecks():
    """Analyze performance bottlenecks."""
    
    profiler = PerformanceProfiler()
    
    # Profile multiple operations
    @profiler.profile_function("data_loading")
    def load_data():
        # Data loading logic
        pass
    
    @profiler.profile_function("processing")
    def process_data():
        # Data processing logic
        pass
    
    @profiler.profile_function("output_generation")
    def generate_output():
        # Output generation logic
        pass
    
    # Run operations
    load_data()
    process_data()
    generate_output()
    
    # Analyze performance
    report = profiler.get_profile_report()
    
    # Find bottlenecks
    for profile_name in report['profiles']:
        profile_data = profiler.get_profile_report(profile_name)
        avg_duration = profile_data['duration']['avg']
        
        if avg_duration > 1.0:  # More than 1 second
            print(f"⚠️ Slow operation: {profile_name} ({avg_duration:.3f}s)")
        
        if profile_data['success_rate'] < 0.95:  # Less than 95% success
            print(f"⚠️ Unreliable operation: {profile_name} ({profile_data['success_rate']:.2%} success)")
    
    return report
```

## Debugging Best Practices

### 🎯 General Guidelines

1. **Start Early**: Enable debugging at the beginning of development
2. **Use Breakpoints**: Set strategic breakpoints for critical functions
3. **Monitor Resources**: Track memory and CPU usage regularly
4. **Profile Performance**: Identify bottlenecks before they become problems
5. **Analyze Errors**: Understand error patterns and root causes
6. **Document Issues**: Keep detailed records of debugging sessions

### 🔍 Debugging Workflow

1. **Reproduce the Issue**: Ensure you can consistently reproduce the problem
2. **Enable Debugging**: Turn on relevant debugging tools
3. **Set Breakpoints**: Place breakpoints at suspected problem areas
4. **Run and Monitor**: Execute the code and monitor debug output
5. **Analyze Results**: Review debug reports and identify issues
6. **Implement Fixes**: Apply fixes based on debug findings
7. **Verify Resolution**: Test that the issue is resolved

### 📊 Performance Debugging

1. **Profile First**: Always profile before optimizing
2. **Measure Impact**: Quantify the impact of changes
3. **Focus on Bottlenecks**: Optimize the slowest parts first
4. **Monitor Memory**: Watch for memory leaks and excessive usage
5. **Test Scalability**: Ensure performance scales with load

### 🐛 Error Debugging

1. **Categorize Errors**: Group errors by type and severity
2. **Track Patterns**: Look for recurring error patterns
3. **Analyze Context**: Understand the context where errors occur
4. **Implement Logging**: Add detailed logging for error scenarios
5. **Test Recovery**: Verify error recovery mechanisms work

## Advanced Debugging Techniques

### 🔬 Conditional Debugging

```python
def conditional_debugging():
    """Use conditional debugging for specific scenarios."""
    
    debugger = VideoOpusClipDebugger()
    
    # Debug only when specific conditions are met
    def debug_condition(*args, **kwargs):
        return len(args) > 0 and isinstance(args[0], str) and len(args[0]) > 100
    
    debugger.set_breakpoint("process_text", debug_condition)
    
    # Debug only during peak hours
    def peak_hour_condition(*args, **kwargs):
        current_hour = datetime.now().hour
        return 9 <= current_hour <= 17  # Business hours
    
    debugger.set_breakpoint("critical_operation", peak_hour_condition)
```

### 📈 Trend Analysis

```python
def analyze_trends():
    """Analyze trends in system behavior."""
    
    debug_manager = DebugManager()
    
    # Run diagnostics over time
    trends = []
    for i in range(10):
        report = debug_manager.run_comprehensive_debug()
        trends.append({
            "timestamp": report['timestamp'],
            "health_score": report['system_diagnostics']['health_score'],
            "error_count": report['error_analysis']['total_errors'],
            "memory_usage": report['memory_analysis']['rss']['current']
        })
        time.sleep(60)  # Wait 1 minute
    
    # Analyze trends
    health_scores = [t['health_score'] for t in trends]
    error_counts = [t['error_count'] for t in trends]
    
    print(f"Health score trend: {min(health_scores)} -> {max(health_scores)}")
    print(f"Error count trend: {min(error_counts)} -> {max(error_counts)}")
    
    return trends
```

### 🔄 Automated Debugging

```python
def automated_debugging():
    """Set up automated debugging for continuous monitoring."""
    
    debug_manager = DebugManager()
    debug_manager.enable_debugging()
    
    def auto_debug_check():
        """Automated debug check that runs periodically."""
        try:
            report = debug_manager.run_comprehensive_debug()
            
            # Check for critical issues
            if report['system_diagnostics']['health_score'] < 30:
                print("🚨 CRITICAL: System health is very poor!")
                # Send alert or take action
            
            if report['error_analysis']['total_errors'] > 50:
                print("⚠️ WARNING: High error count detected!")
                # Log for investigation
            
            if report['memory_analysis']['leak_detection']['potential_leak']:
                print("⚠️ WARNING: Memory leak detected!")
                # Force garbage collection
            
            return report
            
        except Exception as e:
            print(f"❌ Debug check failed: {e}")
            return None
    
    # Set up periodic checks
    import threading
    import time
    
    def run_periodic_checks():
        while True:
            auto_debug_check()
            time.sleep(300)  # Check every 5 minutes
    
    # Start automated monitoring
    monitor_thread = threading.Thread(target=run_periodic_checks, daemon=True)
    monitor_thread.start()
    
    return auto_debug_check
```

## Troubleshooting Common Issues

### 🔧 Debug Tool Issues

#### Import Errors
```python
# Check if debug tools are properly installed
try:
    from debug_tools import DebugManager
    print("✅ Debug tools imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Make sure debug_tools.py is in your Python path")
```

#### Performance Impact
```python
# Disable debugging if performance is critical
debug_manager = DebugManager()
debug_manager.disable_debugging()

# Or disable specific components
debug_manager.debugger.debug_config["enable_trace"] = False
debug_manager.profiler.profiling_enabled = False
```

#### Memory Overhead
```python
# Limit debug history to prevent memory issues
debug_manager.debugger.debug_config["max_history"] = 100

# Clear debug history periodically
debug_manager.debugger.debug_history.clear()
```

### 🐛 System Issues

#### High CPU Usage
```python
# Check what's causing high CPU usage
diagnostics = SystemDiagnostics()
health_report = diagnostics.run_full_diagnostics()

if health_report['resource_usage']['cpu_usage'] > 80:
    print("🔍 High CPU usage detected")
    print("💡 Check for infinite loops or heavy computations")
    print("💡 Consider optimizing algorithms or adding caching")
```

#### Memory Issues
```python
# Investigate memory problems
analyzer = MemoryAnalyzer()
analysis = analyzer.analyze_memory_usage()

if analysis['leak_detection']['potential_leak']:
    print("🔍 Memory leak detected")
    print("💡 Check for unclosed files, database connections")
    print("💡 Review object lifecycle management")
    print("💡 Consider using context managers")
```

#### Network Issues
```python
# Check network connectivity
diagnostics = SystemDiagnostics()
health_report = diagnostics.run_full_diagnostics()

if health_report['network_status']['status'] == "Disconnected":
    print("🔍 Network connectivity issues")
    print("💡 Check internet connection")
    print("💡 Verify firewall settings")
    print("💡 Test DNS resolution")
```

## Debug Output and Logging

### 📝 Debug Logging

```python
from debug_tools import debug_print, debug_function_call, debug_function_return, debug_error

def debug_logging_example():
    """Example of using debug logging functions."""
    
    def process_data(data):
        debug_function_call("process_data", (data,), {})
        start_time = time.time()
        
        try:
            result = complex_processing(data)
            execution_time = time.time() - start_time
            debug_function_return("process_data", result, execution_time)
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            debug_error("process_data", e, execution_time)
            raise
    
    return process_data
```

### 📊 Debug Report Generation

```python
def generate_debug_report():
    """Generate comprehensive debug report."""
    
    debug_manager = DebugManager()
    debug_manager.enable_debugging()
    
    # Run comprehensive analysis
    report = debug_manager.run_comprehensive_debug()
    
    # Save report to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"debug_report_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"📄 Debug report saved to: {filename}")
    
    return report
```

## Conclusion

The comprehensive debugging system for Video-OpusClip provides powerful tools for identifying and resolving issues quickly. By following the patterns and best practices outlined in this guide, you can effectively debug complex problems, optimize performance, and maintain system reliability.

Remember to:
- Enable debugging early in development
- Use appropriate debugging tools for different scenarios
- Monitor system health regularly
- Analyze trends and patterns
- Document debugging sessions
- Implement automated monitoring where appropriate

For more information, refer to the individual component documentation and the main Video-OpusClip documentation. 