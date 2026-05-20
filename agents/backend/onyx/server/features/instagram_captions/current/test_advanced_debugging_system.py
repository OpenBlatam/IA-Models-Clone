#!/usr/bin/env python3
"""
Test Suite for Advanced Error Handling and Debugging System
Comprehensive tests for all components and functionality
"""

import unittest
import time
import gc
import sys
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from advanced_error_handling_debugging_system import (
    DebugLevel,
    ErrorCategory,
    DebugInfo,
    PerformanceProfile,
    AdvancedErrorAnalyzer,
    PerformanceProfiler,
    MemoryTracker,
    CPUTracker,
    AdvancedDebugger,
    AdvancedErrorHandlingGradioInterface
)


class TestDebugLevel(unittest.TestCase):
    """Test DebugLevel enum"""
    
    def test_debug_levels(self):
        """Test all debug levels exist"""
        self.assertEqual(DebugLevel.BASIC.value, "basic")
        self.assertEqual(DebugLevel.DETAILED.value, "detailed")
        self.assertEqual(DebugLevel.PROFILING.value, "profiling")
        self.assertEqual(DebugLevel.MEMORY.value, "memory")
        self.assertEqual(DebugLevel.THREADING.value, "threading")
        self.assertEqual(DebugLevel.FULL.value, "full")
    
    def test_debug_level_count(self):
        """Test correct number of debug levels"""
        self.assertEqual(len(DebugLevel), 6)


class TestErrorCategory(unittest.TestCase):
    """Test ErrorCategory enum"""
    
    def test_error_categories(self):
        """Test all error categories exist"""
        self.assertEqual(ErrorCategory.INPUT_VALIDATION.value, "input_validation")
        self.assertEqual(ErrorCategory.MODEL_LOADING.value, "model_loading")
        self.assertEqual(ErrorCategory.INFERENCE.value, "inference")
        self.assertEqual(ErrorCategory.MEMORY.value, "memory")
        self.assertEqual(ErrorCategory.NETWORK.value, "network")
        self.assertEqual(ErrorCategory.SYSTEM.value, "system")
        self.assertEqual(ErrorCategory.UNKNOWN.value, "unknown")
    
    def test_error_category_count(self):
        """Test correct number of error categories"""
        self.assertEqual(len(ErrorCategory), 7)


class TestDebugInfo(unittest.TestCase):
    """Test DebugInfo dataclass"""
    
    def test_debug_info_creation(self):
        """Test DebugInfo creation with all fields"""
        debug_info = DebugInfo(
            function_name="test_function",
            execution_time=1.5,
            memory_usage={"total": 100.0},
            cpu_usage=50.0,
            call_stack=["test_function"],
            local_variables={"arg1": "value1"},
            debug_level=DebugLevel.DETAILED
        )
        
        self.assertEqual(debug_info.function_name, "test_function")
        self.assertEqual(debug_info.execution_time, 1.5)
        self.assertEqual(debug_info.memory_usage["total"], 100.0)
        self.assertEqual(debug_info.cpu_usage, 50.0)
        self.assertEqual(debug_info.call_stack, ["test_function"])
        self.assertEqual(debug_info.local_variables["arg1"], "value1")
        self.assertEqual(debug_info.debug_level, DebugLevel.DETAILED)
        self.assertIsInstance(debug_info.timestamp, float)


class TestPerformanceProfile(unittest.TestCase):
    """Test PerformanceProfile dataclass"""
    
    def test_performance_profile_creation(self):
        """Test PerformanceProfile creation with all fields"""
        profile = PerformanceProfile(
            function_name="test_function",
            total_calls=10,
            total_time=5.0,
            average_time=0.5,
            min_time=0.1,
            max_time=1.0,
            memory_peak=200.0,
            cpu_peak=75.0,
            profile_data={"key": "value"}
        )
        
        self.assertEqual(profile.function_name, "test_function")
        self.assertEqual(profile.total_calls, 10)
        self.assertEqual(profile.total_time, 5.0)
        self.assertEqual(profile.average_time, 0.5)
        self.assertEqual(profile.min_time, 0.1)
        self.assertEqual(profile.max_time, 1.0)
        self.assertEqual(profile.memory_peak, 200.0)
        self.assertEqual(profile.cpu_peak, 75.0)
        self.assertEqual(profile.profile_data["key"], "value")


class TestAdvancedErrorAnalyzer(unittest.TestCase):
    """Test AdvancedErrorAnalyzer class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.analyzer = AdvancedErrorAnalyzer()
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization"""
        self.assertIsInstance(self.analyzer.error_patterns, dict)
        self.assertIsInstance(self.analyzer.error_frequency, dict)
        self.assertIsInstance(self.analyzer.error_correlations, dict)
        self.assertGreater(len(self.analyzer.error_patterns), 0)
    
    def test_error_patterns_structure(self):
        """Test error patterns have correct structure"""
        for pattern_name, pattern_info in self.analyzer.error_patterns.items():
            self.assertIn("category", pattern_info)
            self.assertIn("solutions", pattern_info)
            self.assertIn("prevention", pattern_info)
            self.assertIsInstance(pattern_info["category"], ErrorCategory)
            self.assertIsInstance(pattern_info["solutions"], list)
            self.assertIsInstance(pattern_info["prevention"], list)
    
    def test_analyze_error_basic(self):
        """Test basic error analysis"""
        error = ValueError("Test error message")
        analysis = self.analyzer.analyze_error(error)
        
        self.assertEqual(analysis["error_type"], "ValueError")
        self.assertEqual(analysis["error_message"], "Test error message")
        self.assertIn("category", analysis)
        self.assertIn("severity", analysis)
        self.assertIn("frequency", analysis)
        self.assertIn("patterns", analysis)
        self.assertIn("solutions", analysis)
        self.assertIn("prevention", analysis)
        self.assertIn("context", analysis)
    
    def test_analyze_error_with_context(self):
        """Test error analysis with context"""
        error = ValueError("Test error message")
        context = {"function_name": "test_func", "args": "test_args"}
        analysis = self.analyzer.analyze_error(error, context)
        
        self.assertEqual(analysis["context"]["function_name"], "test_func")
        self.assertEqual(analysis["context"]["args"], "test_args")
    
    def test_error_categorization(self):
        """Test error categorization"""
        # Test memory errors
        memory_error = RuntimeError("CUDA out of memory")
        analysis = self.analyzer.analyze_error(memory_error)
        self.assertEqual(analysis["category"], ErrorCategory.MEMORY)
        
        # Test model loading errors
        model_error = FileNotFoundError("Model not found")
        analysis = self.analyzer.analyze_error(model_error)
        self.assertEqual(analysis["category"], ErrorCategory.MODEL_LOADING)
        
        # Test input validation errors
        input_error = ValueError("Invalid input format")
        analysis = self.analyzer.analyze_error(input_error)
        self.assertEqual(analysis["category"], ErrorCategory.INPUT_VALIDATION)
    
    def test_severity_assessment(self):
        """Test severity assessment"""
        # Test critical errors
        critical_error = RuntimeError("CUDA out of memory")
        analysis = self.analyzer.analyze_error(critical_error)
        self.assertEqual(analysis["severity"], "CRITICAL")
        
        # Test high severity errors
        high_error = ConnectionError("Network timeout")
        analysis = self.analyzer.analyze_error(high_error)
        self.assertEqual(analysis["severity"], "HIGH")
        
        # Test medium severity errors
        medium_error = ValueError("Invalid input")
        analysis = self.analyzer.analyze_error(medium_error)
        self.assertEqual(analysis["severity"], "MEDIUM")
    
    def test_error_frequency_tracking(self):
        """Test error frequency tracking"""
        error = ValueError("Test error")
        
        # First occurrence
        analysis1 = self.analyzer.analyze_error(error)
        self.assertEqual(analysis1["frequency"], 0)
        
        # Second occurrence
        analysis2 = self.analyzer.analyze_error(error)
        self.assertEqual(analysis2["frequency"], 1)
        
        # Check frequency in analyzer
        self.assertEqual(self.analyzer.error_frequency["ValueError"], 2)
    
    def test_get_error_statistics(self):
        """Test error statistics generation"""
        # Generate some errors
        self.analyzer.analyze_error(ValueError("Error 1"))
        self.analyzer.analyze_error(ValueError("Error 2"))
        self.analyzer.analyze_error(RuntimeError("Error 3"))
        
        stats = self.analyzer.get_error_statistics()
        
        self.assertIn("total_errors", stats)
        self.assertIn("error_frequency", stats)
        self.assertIn("most_common_errors", stats)
        self.assertIn("error_categories", stats)
        self.assertIn("error_trends", stats)
        
        self.assertEqual(stats["total_errors"], 3)
        self.assertEqual(stats["error_frequency"]["ValueError"], 2)
        self.assertEqual(stats["error_frequency"]["RuntimeError"], 1)


class TestPerformanceProfiler(unittest.TestCase):
    """Test PerformanceProfiler class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.profiler = PerformanceProfiler()
    
    def test_profiler_initialization(self):
        """Test profiler initialization"""
        self.assertIsInstance(self.profiler.profiles, dict)
        self.assertIsInstance(self.profiler.active_profiles, dict)
        self.assertIsInstance(self.profiler.memory_tracker, MemoryTracker)
        self.assertIsInstance(self.profiler.cpu_tracker, CPUTracker)
    
    def test_profile_function_context_manager(self):
        """Test profile_function context manager"""
        with self.profiler.profile_function("test_function", DebugLevel.BASIC):
            time.sleep(0.1)  # Simulate work
        
        # Check that profile was created
        self.assertIn("test_function", self.profiler.profiles)
        self.assertGreater(len(self.profiler.profiles["test_function"]), 0)
        
        profile = self.profiler.profiles["test_function"][0]
        self.assertEqual(profile.function_name, "test_function")
        self.assertGreater(profile.total_time, 0.1)
        self.assertGreater(profile.memory_peak, 0)
        self.assertGreater(profile.cpu_peak, 0)
    
    def test_profile_function_with_profiling(self):
        """Test profile_function with detailed profiling"""
        with self.profiler.profile_function("test_function", DebugLevel.PROFILING):
            time.sleep(0.1)  # Simulate work
        
        profile = self.profiler.profiles["test_function"][0]
        self.assertIn("profile_data", profile.__dict__)
    
    def test_get_performance_summary(self):
        """Test performance summary generation"""
        # Generate some profiles
        with self.profiler.profile_function("func1", DebugLevel.BASIC):
            time.sleep(0.05)
        
        with self.profiler.profile_function("func2", DebugLevel.BASIC):
            time.sleep(0.05)
        
        summary = self.profiler.get_performance_summary()
        
        self.assertIn("func1", summary)
        self.assertIn("func2", summary)
        
        func1_stats = summary["func1"]
        self.assertIn("total_calls", func1_stats)
        self.assertIn("total_time", func1_stats)
        self.assertIn("average_time", func1_stats)
        self.assertIn("min_time", func1_stats)
        self.assertIn("max_time", func1_stats)
        self.assertIn("std_dev_time", func1_stats)
        self.assertIn("average_memory_peak", func1_stats)
        self.assertIn("average_cpu_peak", func1_stats)
        
        self.assertEqual(func1_stats["total_calls"], 1)
        self.assertGreater(func1_stats["total_time"], 0.05)


class TestMemoryTracker(unittest.TestCase):
    """Test MemoryTracker class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.memory_tracker = MemoryTracker()
    
    def test_memory_tracker_initialization(self):
        """Test memory tracker initialization"""
        self.assertIsInstance(self.memory_tracker.memory_history, list)
        self.assertIsInstance(self.memory_tracker.gc_stats, dict)
    
    def test_get_memory_usage(self):
        """Test memory usage retrieval"""
        memory_data = self.memory_tracker.get_memory_usage()
        
        self.assertIn("rss", memory_data)
        self.assertIn("vms", memory_data)
        self.assertIn("percent", memory_data)
        self.assertIn("gpu_memory", memory_data)
        self.assertIn("total", memory_data)
        
        self.assertGreater(memory_data["rss"], 0)
        self.assertGreater(memory_data["vms"], 0)
        self.assertGreaterEqual(memory_data["percent"], 0)
        self.assertIsInstance(memory_data["gpu_memory"], dict)
        self.assertGreater(memory_data["total"], 0)
    
    def test_memory_history_tracking(self):
        """Test memory history tracking"""
        initial_length = len(self.memory_tracker.memory_history)
        
        self.memory_tracker.get_memory_usage()
        
        self.assertEqual(len(self.memory_tracker.memory_history), initial_length + 1)
        
        # Test history limit
        for _ in range(1005):  # Exceed limit
            self.memory_tracker.get_memory_usage()
        
        self.assertLessEqual(len(self.memory_tracker.memory_history), 1000)
    
    def test_get_memory_statistics(self):
        """Test memory statistics generation"""
        # Generate some memory data
        for _ in range(5):
            self.memory_tracker.get_memory_usage()
        
        stats = self.memory_tracker.get_memory_statistics()
        
        self.assertIn("current_memory", stats)
        self.assertIn("peak_memory", stats)
        self.assertIn("average_memory", stats)
        self.assertIn("memory_trend", stats)
        self.assertIn("memory_leak_detection", stats)
        
        self.assertGreater(stats["peak_memory"], 0)
        self.assertGreater(stats["average_memory"], 0)
    
    def test_memory_trend_analysis(self):
        """Test memory trend analysis"""
        # Generate memory data
        for _ in range(15):
            self.memory_tracker.get_memory_usage()
        
        stats = self.memory_tracker.get_memory_statistics()
        self.assertIsInstance(stats["memory_trend"], str)
        self.assertIn("INSUFFICIENT", stats["memory_trend"])
    
    def test_memory_leak_detection(self):
        """Test memory leak detection"""
        # Generate memory data
        for _ in range(25):
            self.memory_tracker.get_memory_usage()
        
        stats = self.memory_tracker.get_memory_statistics()
        leak_info = stats["memory_leak_detection"]
        
        self.assertIn("detected", leak_info)
        self.assertIn("probability", leak_info)
        self.assertIn("suggestion", leak_info)
        self.assertIsInstance(leak_info["detected"], bool)
        self.assertGreaterEqual(leak_info["probability"], 0)
        self.assertLessEqual(leak_info["probability"], 1)
    
    def test_force_garbage_collection(self):
        """Test forced garbage collection"""
        # Create some objects
        large_list = [i for i in range(100000)]
        
        before_memory = self.memory_tracker.get_memory_usage()
        
        # Force garbage collection
        gc_results = self.memory_tracker.force_garbage_collection()
        
        after_memory = self.memory_tracker.get_memory_usage()
        
        self.assertIn("objects_collected", gc_results)
        self.assertIn("memory_freed", gc_results)
        self.assertIn("before_memory", gc_results)
        self.assertIn("after_memory", gc_results)
        
        self.assertIsInstance(gc_results["objects_collected"], int)
        self.assertIsInstance(gc_results["memory_freed"], float)
        
        # Clean up
        del large_list


class TestCPUTracker(unittest.TestCase):
    """Test CPUTracker class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.cpu_tracker = CPUTracker()
    
    def test_cpu_tracker_initialization(self):
        """Test CPU tracker initialization"""
        self.assertIsInstance(self.cpu_tracker.cpu_history, list)
    
    def test_get_cpu_usage(self):
        """Test CPU usage retrieval"""
        cpu_usage = self.cpu_tracker.get_cpu_usage()
        
        self.assertIsInstance(cpu_usage, float)
        self.assertGreaterEqual(cpu_usage, 0)
        self.assertLessEqual(cpu_usage, 100)
    
    def test_cpu_history_tracking(self):
        """Test CPU history tracking"""
        initial_length = len(self.cpu_tracker.cpu_history)
        
        self.cpu_tracker.get_cpu_usage()
        
        self.assertEqual(len(self.cpu_tracker.cpu_history), initial_length + 1)
        
        # Test history limit
        for _ in range(1005):  # Exceed limit
            self.cpu_tracker.get_cpu_usage()
        
        self.assertLessEqual(len(self.cpu_tracker.cpu_history), 1000)
    
    def test_get_cpu_statistics(self):
        """Test CPU statistics generation"""
        # Generate some CPU data
        for _ in range(5):
            self.cpu_tracker.get_cpu_usage()
        
        stats = self.cpu_tracker.get_cpu_statistics()
        
        self.assertIn("current_cpu", stats)
        self.assertIn("peak_cpu", stats)
        self.assertIn("average_cpu", stats)
        self.assertIn("cpu_trend", stats)
        self.assertIn("cpu_cores", stats)
        
        self.assertGreaterEqual(stats["current_cpu"], 0)
        self.assertLessEqual(stats["current_cpu"], 100)
        self.assertGreaterEqual(stats["peak_cpu"], 0)
        self.assertLessEqual(stats["peak_cpu"], 100)
        self.assertGreaterEqual(stats["average_cpu"], 0)
        self.assertLessEqual(stats["average_cpu"], 100)
        self.assertGreater(stats["cpu_cores"], 0)
    
    def test_cpu_trend_analysis(self):
        """Test CPU trend analysis"""
        # Generate CPU data
        for _ in range(15):
            self.cpu_tracker.get_cpu_usage()
        
        stats = self.cpu_tracker.get_cpu_statistics()
        self.assertIsInstance(stats["cpu_trend"], str)
        self.assertIn("INSUFFICIENT", stats["cpu_trend"])


class TestAdvancedDebugger(unittest.TestCase):
    """Test AdvancedDebugger class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.debugger = AdvancedDebugger(DebugLevel.DETAILED)
    
    def test_debugger_initialization(self):
        """Test debugger initialization"""
        self.assertEqual(self.debugger.debug_level, DebugLevel.DETAILED)
        self.assertIsInstance(self.debugger.error_analyzer, AdvancedErrorAnalyzer)
        self.assertIsInstance(self.debugger.performance_profiler, PerformanceProfiler)
        self.assertIsInstance(self.debugger.memory_tracker, MemoryTracker)
        self.assertIsInstance(self.debugger.cpu_tracker, CPUTracker)
        self.assertIsInstance(self.debugger.debug_log, list)
    
    def test_debug_function_decorator_success(self):
        """Test debug function decorator with successful execution"""
        @self.debugger.debug_function
        def test_function():
            return "success"
        
        result = test_function()
        
        self.assertEqual(result, "success")
        self.assertGreater(len(self.debugger.debug_log), 0)
        
        log_entry = self.debugger.debug_log[-1]
        self.assertTrue(log_entry["success"])
        self.assertEqual(log_entry["result_type"], "str")
        self.assertIsNone(log_entry["error"])
    
    def test_debug_function_decorator_failure(self):
        """Test debug function decorator with failed execution"""
        @self.debugger.debug_function
        def failing_function():
            raise ValueError("Test error")
        
        with self.assertRaises(ValueError):
            failing_function()
        
        self.assertGreater(len(self.debugger.debug_log), 0)
        
        log_entry = self.debugger.debug_log[-1]
        self.assertFalse(log_entry["success"])
        self.assertIsNone(log_entry["result_type"])
        self.assertIsNotNone(log_entry["error"])
        self.assertIn("error_analysis", log_entry)
    
    def test_get_debug_summary(self):
        """Test debug summary generation"""
        # Generate some debug data
        @self.debugger.debug_function
        def test_function():
            return "success"
        
        test_function()
        
        summary = self.debugger.get_debug_summary()
        
        self.assertIn("debug_level", summary)
        self.assertIn("total_debug_entries", summary)
        self.assertIn("successful_executions", summary)
        self.assertIn("failed_executions", summary)
        self.assertIn("error_statistics", summary)
        self.assertIn("performance_summary", summary)
        self.assertIn("memory_statistics", summary)
        self.assertIn("cpu_statistics", summary)
        self.assertIn("recent_debug_entries", summary)
        
        self.assertEqual(summary["debug_level"], "detailed")
        self.assertGreater(summary["total_debug_entries"], 0)
        self.assertGreater(summary["successful_executions"], 0)
    
    def test_clear_debug_log(self):
        """Test debug log clearing"""
        # Generate some debug data
        @self.debugger.debug_function
        def test_function():
            return "success"
        
        test_function()
        
        self.assertGreater(len(self.debugger.debug_log), 0)
        
        self.debugger.clear_debug_log()
        
        self.assertEqual(len(self.debugger.debug_log), 0)
        self.assertEqual(len(self.debugger.performance_profiler.profiles), 0)
        self.assertEqual(len(self.debugger.memory_tracker.memory_history), 0)
        self.assertEqual(len(self.debugger.cpu_tracker.cpu_history), 0)


class TestAdvancedErrorHandlingGradioInterface(unittest.TestCase):
    """Test AdvancedErrorHandlingGradioInterface class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.interface = AdvancedErrorHandlingGradioInterface(DebugLevel.DETAILED)
    
    def test_interface_initialization(self):
        """Test interface initialization"""
        self.assertIsInstance(self.interface.debugger, AdvancedDebugger)
        self.assertEqual(self.interface.debugger.debug_level, DebugLevel.DETAILED)
    
    def test_debug_text_generation_success(self):
        """Test successful text generation debugging"""
        result, _, status = self.interface.debug_text_generation(
            "Test prompt", 100, 0.7, "gpt2"
        )
        
        self.assertIsInstance(result, str)
        self.assertIn("Generated text", result)
        self.assertIn("✅ Generation Complete", status)
    
    def test_debug_text_generation_failure(self):
        """Test failed text generation debugging"""
        with self.assertRaises(ValueError):
            self.interface.debug_text_generation("", 100, 0.7, "gpt2")
    
    def test_debug_sentiment_analysis_success(self):
        """Test successful sentiment analysis debugging"""
        result, _, status = self.interface.debug_sentiment_analysis(
            "This is a test text for sentiment analysis"
        )
        
        self.assertIsInstance(result, str)
        self.assertIn("Sentiment Analysis Results", result)
        self.assertIn("✅ Analysis Complete", status)
    
    def test_debug_sentiment_analysis_failure(self):
        """Test failed sentiment analysis debugging"""
        with self.assertRaises(ValueError):
            self.interface.debug_sentiment_analysis("short")
    
    def test_get_debug_dashboard(self):
        """Test debug dashboard data retrieval"""
        dashboard = self.interface.get_debug_dashboard()
        
        self.assertIsInstance(dashboard, dict)
        self.assertIn("debug_level", dashboard)
        self.assertIn("total_debug_entries", dashboard)
        self.assertIn("successful_executions", dashboard)
        self.assertIn("failed_executions", dashboard)
    
    def test_force_garbage_collection(self):
        """Test forced garbage collection"""
        gc_results = self.interface.force_garbage_collection()
        
        self.assertIn("objects_collected", gc_results)
        self.assertIn("memory_freed", gc_results)
        self.assertIn("before_memory", gc_results)
        self.assertIn("after_memory", gc_results)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.debugger = AdvancedDebugger(DebugLevel.DETAILED)
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_complete_workflow(self):
        """Test complete debugging workflow"""
        # Test function with debug decorator
        @self.debugger.debug_function
        def complex_function(input_data, iterations=5):
            results = []
            for i in range(iterations):
                # Simulate some work
                time.sleep(0.01)
                results.append(f"Result {i}: {input_data}")
            return results
        
        # Execute function
        result = complex_function("test_data", 3)
        
        # Verify result
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], "Result 0: test_data")
        
        # Verify debug information
        summary = self.debugger.get_debug_summary()
        self.assertGreater(summary["total_debug_entries"], 0)
        self.assertGreater(summary["successful_executions"], 0)
        
        # Verify performance data
        perf_summary = summary["performance_summary"]
        self.assertIn("complex_function", perf_summary)
        
        # Verify memory data
        memory_stats = summary["memory_statistics"]
        self.assertIn("current_memory", memory_stats)
        
        # Verify CPU data
        cpu_stats = summary["cpu_statistics"]
        self.assertIn("current_cpu", cpu_stats)
    
    def test_error_handling_workflow(self):
        """Test error handling workflow"""
        @self.debugger.debug_function
        def error_prone_function(should_fail=False):
            if should_fail:
                raise RuntimeError("Simulated error")
            return "success"
        
        # Successful execution
        result = error_prone_function(False)
        self.assertEqual(result, "success")
        
        # Failed execution
        with self.assertRaises(RuntimeError):
            error_prone_function(True)
        
        # Verify error analysis
        summary = self.debugger.get_debug_summary()
        self.assertGreater(summary["total_debug_entries"], 0)
        self.assertGreater(summary["successful_executions"], 0)
        self.assertGreater(summary["failed_executions"], 0)
        
        # Verify error statistics
        error_stats = summary["error_statistics"]
        self.assertGreater(error_stats["total_errors"], 0)
        self.assertIn("RuntimeError", error_stats["error_frequency"])
    
    def test_memory_leak_detection(self):
        """Test memory leak detection"""
        # Create objects to simulate memory usage
        large_objects = []
        
        for i in range(10):
            large_objects.append([j for j in range(10000)])
            self.debugger.memory_tracker.get_memory_usage()
        
        # Check memory statistics
        memory_stats = self.debugger.memory_tracker.get_memory_statistics()
        self.assertIn("memory_leak_detection", memory_stats)
        
        # Clean up
        del large_objects
        gc.collect()
    
    def test_performance_profiling(self):
        """Test performance profiling"""
        # Profile multiple function calls
        for i in range(5):
            with self.debugger.performance_profiler.profile_function(f"test_func_{i}", DebugLevel.PROFILING):
                time.sleep(0.01)
        
        # Check performance summary
        perf_summary = self.debugger.performance_profiler.get_performance_summary()
        self.assertEqual(len(perf_summary), 5)
        
        for i in range(5):
            func_name = f"test_func_{i}"
            self.assertIn(func_name, perf_summary)
            self.assertEqual(perf_summary[func_name]["total_calls"], 1)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.debugger = AdvancedDebugger(DebugLevel.BASIC)
    
    def test_empty_inputs(self):
        """Test handling of empty inputs"""
        @self.debugger.debug_function
        def test_function():
            return "success"
        
        # Test with no arguments
        result = test_function()
        self.assertEqual(result, "success")
        
        # Verify debug log entry
        log_entry = self.debugger.debug_log[-1]
        self.assertTrue(log_entry["success"])
    
    def test_large_inputs(self):
        """Test handling of large inputs"""
        large_data = "x" * 10000
        
        @self.debugger.debug_function
        def test_function(data):
            return len(data)
        
        result = test_function(large_data)
        self.assertEqual(result, 10000)
        
        # Verify debug log entry
        log_entry = self.debugger.debug_log[-1]
        self.assertTrue(log_entry["success"])
    
    def test_nested_function_calls(self):
        """Test nested function calls with debugging"""
        @self.debugger.debug_function
        def outer_function():
            return inner_function()
        
        @self.debugger.debug_function
        def inner_function():
            return "nested_result"
        
        result = outer_function()
        self.assertEqual(result, "nested_result")
        
        # Verify both functions were logged
        summary = self.debugger.get_debug_summary()
        self.assertGreaterEqual(summary["total_debug_entries"], 2)
    
    def test_concurrent_access(self):
        """Test concurrent access to debugger"""
        import threading
        
        results = []
        errors = []
        
        def worker_function(worker_id):
            try:
                @self.debugger.debug_function
                def test_function():
                    return f"worker_{worker_id}"
                
                result = test_function()
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker_function, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify results
        self.assertEqual(len(results), 5)
        self.assertEqual(len(errors), 0)
        
        # Verify debug log entries
        summary = self.debugger.get_debug_summary()
        self.assertGreaterEqual(summary["total_debug_entries"], 5)


def run_performance_tests():
    """Run performance tests"""
    print("\n" + "="*60)
    print("PERFORMANCE TESTS")
    print("="*60)
    
    debugger = AdvancedDebugger(DebugLevel.FULL)
    
    # Test performance impact of different debug levels
    debug_levels = [DebugLevel.BASIC, DebugLevel.DETAILED, DebugLevel.PROFILING, DebugLevel.FULL]
    
    for level in debug_levels:
        debugger.debug_level = level
        start_time = time.time()
        
        @debugger.debug_function
        def performance_test_function():
            # Simulate some work
            sum(range(10000))
            return "done"
        
        for _ in range(10):
            performance_test_function()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"Debug Level: {level.value:12} | Time: {total_time:.3f}s | Overhead: {((total_time - 0.01) / 0.01 * 100):6.1f}%")


def run_memory_tests():
    """Run memory tests"""
    print("\n" + "="*60)
    print("MEMORY TESTS")
    print("="*60)
    
    debugger = AdvancedDebugger(DebugLevel.FULL)
    
    # Test memory usage over time
    initial_memory = debugger.memory_tracker.get_memory_usage()["total"]
    
    print(f"Initial memory usage: {initial_memory:.2f} MB")
    
    for i in range(100):
        @debugger.debug_function
        def memory_test_function():
            return [i for i in range(1000)]
        
        memory_test_function()
        
        if i % 20 == 0:
            current_memory = debugger.memory_tracker.get_memory_usage()["total"]
            print(f"After {i:3d} calls: {current_memory:.2f} MB (delta: {current_memory - initial_memory:+.2f} MB)")
    
    final_memory = debugger.memory_tracker.get_memory_usage()["total"]
    print(f"Final memory usage: {final_memory:.2f} MB (total delta: {final_memory - initial_memory:+.2f} MB)")
    
    # Force garbage collection
    gc_results = debugger.memory_tracker.force_garbage_collection()
    print(f"Garbage collection: {gc_results['objects_collected']} objects collected, {gc_results['memory_freed']:.2f} MB freed")


if __name__ == "__main__":
    # Run unit tests
    print("Running Advanced Error Handling and Debugging System Tests")
    print("="*70)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestDebugLevel,
        TestErrorCategory,
        TestDebugInfo,
        TestPerformanceProfile,
        TestAdvancedErrorAnalyzer,
        TestPerformanceProfiler,
        TestMemoryTracker,
        TestCPUTracker,
        TestAdvancedDebugger,
        TestAdvancedErrorHandlingGradioInterface,
        TestIntegration,
        TestEdgeCases
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
    
    # Run performance tests if all unit tests pass
    if len(result.failures) == 0 and len(result.errors) == 0:
        run_performance_tests()
        run_memory_tests()
    
    # Exit with appropriate code
    if len(result.failures) > 0 or len(result.errors) > 0:
        sys.exit(1)
    else:
        print("\n✅ All tests passed successfully!")
        sys.exit(0)



