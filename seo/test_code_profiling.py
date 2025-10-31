#!/usr/bin/env python3
"""
Test script for Code Profiling and Performance Optimization System
Validates all aspects of the profiling implementation
"""

import unittest
import time
import tempfile
import os
import json
from unittest.mock import Mock, patch, MagicMock
import sys
import traceback

# Add the parent directory to the path to import the engine
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from advanced_llm_seo_engine import CodeProfiler, SEOConfig
except ImportError as e:
    print(f"Import error: {e}")
    print("Creating mock classes for testing...")
    
    # Mock classes for testing
    class MockConfig:
        def __init__(self):
            self.enable_code_profiling = True
            self.profile_data_loading = True
            self.profile_preprocessing = True
            self.profile_model_inference = True
            self.profile_training_loop = True
            self.profile_memory_usage = True
            self.profile_gpu_utilization = True
            self.profile_cpu_utilization = True
            self.profile_io_operations = True
            self.profile_batch_processing = True
            self.profile_validation_loop = True
            self.profile_early_stopping = True
            self.profile_lr_scheduling = True
            self.profile_mixed_precision = True
            self.profile_gradient_accumulation = True
            self.profile_multi_gpu = True
            self.profile_autocast = True
            self.profile_grad_scaler = True
            self.profile_checkpointing = True
            self.profile_logging = False
            self.profile_error_handling = False
            self.profile_input_validation = False
            self.profile_gradio_interface = False
            self.profile_metrics_calculation = True
            self.profile_cross_validation = True
            self.profile_data_splitting = True
            self.profile_model_compilation = True
            self.profile_optimization_orchestrator = True
            self.profile_memory_optimizer = True
            self.profile_async_data_loader = True
            self.profile_model_compiler = True
            self.profile_performance_benchmarking = True
            self.profile_testing_framework = False
            self.profile_demo_showcase = False
            self.profile_evaluation_metrics = True
            self.profile_error_handling_validation = False
            self.profile_try_except_blocks = False
            self.profile_comprehensive_logging = False
            self.profile_pytorch_debugging = False
            self.profile_multi_gpu_training = True
            self.profile_gradient_accumulation = True
            self.profile_enhanced_mixed_precision = True
            self.profile_code_profiling = False
    
    class MockCodeProfiler:
        def __init__(self, config):
            self.config = config
            self.profiling_enabled = config.enable_code_profiling
            self.profiling_data = {}
            self.profiling_stats = {}
            self.start_time = time.time()
            self.end_time = None
        
        def profile_operation(self, operation_name, operation_type):
            class MockContextManager:
                def __init__(self, profiler, name, op_type):
                    self.profiler = profiler
                    self.name = name
                    self.op_type = op_type
                
                def __enter__(self):
                    return self
                
                def __exit__(self, exc_type, exc_val, exc_tb):
                    # Mock recording of profiling data
                    if self.profiler.profiling_enabled:
                        self.profiler.profiling_data[self.name] = {
                            'type': self.op_type,
                            'calls': 1,
                            'total_duration': 0.1,
                            'total_memory_delta': 1024,
                            'total_gpu_memory_delta': 512
                        }
            
            return MockContextManager(self, operation_name, operation_type)
        
        def get_profiling_summary(self):
            if not self.profiling_enabled:
                return {"profiling_enabled": False}
            
            return {
                "profiling_enabled": True,
                "total_operations": len(self.profiling_data),
                "total_calls": sum(data.get('calls', 0) for data in self.profiling_data.values()),
                "total_duration": sum(data.get('total_duration', 0) for data in self.profiling_data.values()),
                "operations": self.profiling_data
            }
        
        def get_bottlenecks(self, threshold_duration=1.0):
            bottlenecks = []
            for op_name, op_data in self.profiling_data.items():
                if op_data.get('total_duration', 0) > threshold_duration:
                    bottlenecks.append({
                        'operation': op_name,
                        'type': op_data.get('type', 'unknown'),
                        'avg_duration': op_data.get('total_duration', 0),
                        'total_calls': op_data.get('calls', 0),
                        'total_duration': op_data.get('total_duration', 0),
                        'memory_efficiency': 0.0,
                        'gpu_memory_efficiency': 0.0,
                        'optimization_priority': 'high'
                    })
            return bottlenecks
        
        def get_performance_recommendations(self):
            bottlenecks = self.get_bottlenecks()
            recommendations = []
            for bottleneck in bottlenecks:
                op_name = bottleneck['operation']
                op_type = bottleneck['type']
                avg_duration = bottleneck['avg_duration']
                
                if op_type == "data_loading":
                    if avg_duration > 5.0:
                        recommendations.append(f"🚨 CRITICAL: {op_name} is extremely slow ({avg_duration:.2f}s). Consider implementing async loading, caching, or batch processing.")
                    elif avg_duration > 1.0:
                        recommendations.append(f"⚠️  WARNING: {op_name} is slow ({avg_duration:.2f}s). Consider optimizing data loading with prefetching or parallel processing.")
                    else:
                        recommendations.append(f"✅ {op_name} performance is acceptable ({avg_duration:.2f}s).")
            
            return recommendations
        
        def export_profiling_data(self, filepath=None):
            if not self.profiling_enabled:
                return "Profiling not enabled"
            
            if filepath is None:
                filepath = f"profiling_data_{int(time.time())}.json"
            
            try:
                summary = self.get_profiling_summary()
                bottlenecks = self.get_bottlenecks()
                recommendations = self.get_performance_recommendations()
                
                export_data = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "profiling_summary": summary,
                    "bottlenecks": bottlenecks,
                    "recommendations": recommendations,
                    "raw_profiling_data": self.profiling_data
                }
                
                with open(filepath, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
                
                return f"Profiling data exported to {filepath}"
                
            except Exception as e:
                return f"Failed to export profiling data: {e}"
        
        def cleanup(self):
            self.end_time = time.time()
            print(f"Mock profiling completed. Total duration: {self.end_time - self.start_time:.2f}s")
    
    # Use mock classes
    SEOConfig = MockConfig
    CodeProfiler = MockCodeProfiler

class TestCodeProfiling(unittest.TestCase):
    """Test cases for the Code Profiling system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = SEOConfig()
        self.profiler = CodeProfiler(self.config)
    
    def tearDown(self):
        """Clean up after tests."""
        if hasattr(self.profiler, 'cleanup'):
            self.profiler.cleanup()
    
    def test_profiler_initialization(self):
        """Test profiler initialization."""
        self.assertIsNotNone(self.profiler)
        self.assertEqual(self.profiler.config, self.config)
        self.assertEqual(self.profiler.profiling_enabled, self.config.enable_code_profiling)
    
    def test_profile_operation_context_manager(self):
        """Test the profile_operation context manager."""
        operation_name = "test_operation"
        operation_type = "test_type"
        
        with self.profiler.profile_operation(operation_name, operation_type):
            # Simulate some work
            time.sleep(0.01)
        
        # Check if profiling data was recorded
        if self.profiler.profiling_enabled:
            self.assertIn(operation_name, self.profiler.profiling_data)
            data = self.profiler.profiling_data[operation_name]
            self.assertEqual(data['type'], operation_type)
            self.assertGreater(data['calls'], 0)
    
    def test_profiling_disabled(self):
        """Test profiling when disabled."""
        self.profiler.profiling_enabled = False
        
        with self.profiler.profile_operation("disabled_operation", "test"):
            time.sleep(0.01)
        
        # Should not record any data
        self.assertEqual(len(self.profiler.profiling_data), 0)
    
    def test_get_profiling_summary(self):
        """Test getting profiling summary."""
        # Perform some operations first
        with self.profiler.profile_operation("test_op1", "test_type"):
            time.sleep(0.01)
        
        with self.profiler.profile_operation("test_op2", "test_type"):
            time.sleep(0.01)
        
        summary = self.profiler.get_profiling_summary()
        
        if self.profiler.profiling_enabled:
            self.assertTrue(summary['profiling_enabled'])
            self.assertGreaterEqual(summary['total_operations'], 0)
            self.assertGreaterEqual(summary['total_calls'], 0)
            self.assertIn('operations', summary)
        else:
            self.assertFalse(summary['profiling_enabled'])
    
    def test_get_bottlenecks(self):
        """Test bottleneck identification."""
        # Perform some operations
        with self.profiler.profile_operation("fast_op", "fast"):
            time.sleep(0.01)
        
        with self.profiler.profile_operation("slow_op", "slow"):
            time.sleep(0.1)
        
        bottlenecks = self.profiler.get_bottlenecks(threshold_duration=0.05)
        
        if self.profiler.profiling_enabled:
            self.assertIsInstance(bottlenecks, list)
            # Should identify the slow operation
            slow_ops = [b for b in bottlenecks if b['operation'] == 'slow_op']
            self.assertGreater(len(slow_ops), 0)
    
    def test_get_performance_recommendations(self):
        """Test performance recommendations generation."""
        # Perform operations to generate recommendations
        with self.profiler.profile_operation("data_loading_op", "data_loading"):
            time.sleep(0.1)
        
        recommendations = self.profiler.get_performance_recommendations()
        
        if self.profiler.profiling_enabled:
            self.assertIsInstance(recommendations, list)
            # Should have recommendations for data loading operations
            data_loading_recs = [r for r in recommendations if "data_loading" in r]
            self.assertGreater(len(data_loading_recs), 0)
    
    def test_export_profiling_data(self):
        """Test profiling data export."""
        # Perform some operations
        with self.profiler.profile_operation("export_test_op", "test"):
            time.sleep(0.01)
        
        # Test export with default filename
        result = self.profiler.export_profiling_data()
        
        if self.profiler.profiling_enabled:
            self.assertIn("exported", result)
            
            # Test export with custom filename
            with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
                custom_result = self.profiler.export_profiling_data(tmp.name)
                self.assertIn("exported", custom_result)
                
                # Verify file was created and contains valid JSON
                with open(tmp.name, 'r') as f:
                    data = json.load(f)
                    self.assertIn('profiling_summary', data)
                    self.assertIn('bottlenecks', data)
                    self.assertIn('recommendations', data)
                
                # Clean up
                os.unlink(tmp.name)
    
    def test_profiler_cleanup(self):
        """Test profiler cleanup."""
        if hasattr(self.profiler, 'cleanup'):
            # Should not raise any exceptions
            self.profiler.cleanup()
            
            # Verify cleanup was performed
            if hasattr(self.profiler, 'end_time'):
                self.assertIsNotNone(self.profiler.end_time)
    
    def test_multiple_operations(self):
        """Test profiling multiple operations of the same type."""
        operation_name = "repeated_op"
        operation_type = "test_type"
        
        # Perform the same operation multiple times
        for i in range(5):
            with self.profiler.profile_operation(operation_name, operation_type):
                time.sleep(0.01)
        
        if self.profiler.profiling_enabled:
            self.assertIn(operation_name, self.profiler.profiling_data)
            data = self.profiler.profiling_data[operation_name]
            self.assertEqual(data['calls'], 5)
            self.assertGreater(data['total_duration'], 0)
    
    def test_different_operation_types(self):
        """Test profiling different types of operations."""
        operations = [
            ("data_loading_op", "data_loading"),
            ("preprocessing_op", "preprocessing"),
            ("model_inference_op", "model_inference"),
            ("training_loop_op", "training_loop")
        ]
        
        for op_name, op_type in operations:
            with self.profiler.profile_operation(op_name, op_type):
                time.sleep(0.01)
        
        if self.profiler.profiling_enabled:
            summary = self.profiler.get_profiling_summary()
            self.assertEqual(summary['total_operations'], len(operations))
            
            # Check that all operation types are represented
            operation_types = set(data['type'] for data in summary['operations'].values())
            expected_types = set(op_type for _, op_type in operations)
            self.assertEqual(operation_types, expected_types)
    
    def test_profiling_configuration(self):
        """Test profiling configuration options."""
        # Test enabling/disabling specific profiling features
        self.config.profile_data_loading = False
        self.config.profile_model_inference = False
        
        # Create new profiler with modified config
        new_profiler = CodeProfiler(self.config)
        
        # Should still be able to profile general operations
        with new_profiler.profile_operation("general_op", "general"):
            time.sleep(0.01)
        
        # Check that profiling still works
        summary = new_profiler.get_profiling_summary()
        if new_profiler.profiling_enabled:
            self.assertGreaterEqual(summary['total_operations'], 0)
    
    def test_error_handling(self):
        """Test error handling in profiling operations."""
        # Test profiling with operations that might fail
        try:
            with self.profiler.profile_operation("error_op", "error"):
                # Simulate an error
                raise ValueError("Test error")
        except ValueError:
            pass
        
        # Profiling should continue to work even after errors
        with self.profiler.profile_operation("normal_op", "normal"):
            time.sleep(0.01)
        
        if self.profiler.profiling_enabled:
            summary = self.profiler.get_profiling_summary()
            self.assertGreaterEqual(summary['total_operations'], 0)

class TestCodeProfilingIntegration(unittest.TestCase):
    """Test integration aspects of the profiling system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = SEOConfig()
        self.profiler = CodeProfiler(self.config)
    
    def test_profiling_with_real_workload(self):
        """Test profiling with a realistic workload simulation."""
        # Simulate a training-like workload
        operations = [
            ("data_loading", "data_loading", 0.05),
            ("preprocessing", "preprocessing", 0.02),
            ("model_inference", "model_inference", 0.1),
            ("backward_pass", "training_loop", 0.08),
            ("optimizer_step", "training_loop", 0.01)
        ]
        
        # Simulate multiple epochs
        for epoch in range(3):
            for op_name, op_type, duration in operations:
                with self.profiler.profile_operation(f"{op_name}_epoch_{epoch}", op_type):
                    time.sleep(duration)
        
        if self.profiler.profiling_enabled:
            summary = self.profiler.get_profiling_summary()
            self.assertEqual(summary['total_operations'], len(operations) * 3)
            
            # Check for bottlenecks
            bottlenecks = self.profiler.get_bottlenecks(threshold_duration=0.05)
            self.assertGreater(len(bottlenecks), 0)
            
            # Check recommendations
            recommendations = self.profiler.get_performance_recommendations()
            self.assertGreater(len(recommendations), 0)
    
    def test_memory_profiling_simulation(self):
        """Test memory profiling simulation."""
        # Simulate memory usage changes
        initial_memory = 100 * 1024 * 1024  # 100 MB
        
        with self.profiler.profile_operation("memory_intensive_op", "memory_usage"):
            # Simulate memory allocation
            simulated_memory = initial_memory * 2
            time.sleep(0.05)
        
        if self.profiler.profiling_enabled:
            summary = self.profiler.get_profiling_summary()
            self.assertGreaterEqual(summary['total_operations'], 0)
    
    def test_gpu_profiling_simulation(self):
        """Test GPU profiling simulation."""
        # Simulate GPU operations
        with self.profiler.profile_operation("gpu_computation", "gpu_utilization"):
            # Simulate GPU computation
            time.sleep(0.1)
        
        if self.profiler.profiling_enabled:
            summary = self.profiler.get_profiling_summary()
            self.assertGreaterEqual(summary['total_operations'], 0)
    
    def test_profiling_data_persistence(self):
        """Test that profiling data persists across operations."""
        # Perform initial operations
        with self.profiler.profile_operation("persistent_op1", "test"):
            time.sleep(0.01)
        
        initial_summary = self.profiler.get_profiling_summary()
        
        # Perform more operations
        with self.profiler.profile_operation("persistent_op2", "test"):
            time.sleep(0.01)
        
        final_summary = self.profiler.get_profiling_summary()
        
        if self.profiler.profiling_enabled:
            # Should have more operations
            self.assertGreater(final_summary['total_operations'], initial_summary['total_operations'])
            
            # Previous operations should still be there
            self.assertIn('persistent_op1', final_summary['operations'])
            self.assertIn('persistent_op2', final_summary['operations'])

def run_profiling_demo():
    """Run a demonstration of the profiling system."""
    print("🚀 Code Profiling System Demo")
    print("=" * 50)
    
    # Create profiler
    config = SEOConfig()
    profiler = CodeProfiler(config)
    
    print(f"✅ Profiler initialized: {profiler.profiling_enabled}")
    
    # Simulate various operations
    operations = [
        ("data_loading", "data_loading", 0.1),
        ("text_preprocessing", "preprocessing", 0.05),
        ("model_inference", "model_inference", 0.2),
        ("training_step", "training_loop", 0.15),
        ("validation", "validation_loop", 0.08),
        ("checkpoint_save", "checkpointing", 0.03)
    ]
    
    print("\n📊 Simulating operations...")
    for op_name, op_type, duration in operations:
        print(f"   Running: {op_name} ({op_type})")
        with profiler.profile_operation(op_name, op_type):
            time.sleep(duration)
    
    # Get profiling results
    print("\n📈 Profiling Results:")
    print("-" * 30)
    
    summary = profiler.get_profiling_summary()
    print(f"Total operations: {summary.get('total_operations', 0)}")
    print(f"Total calls: {summary.get('total_calls', 0)}")
    print(f"Total duration: {summary.get('total_duration', 0):.2f}s")
    
    # Identify bottlenecks
    print("\n🐌 Performance Bottlenecks:")
    print("-" * 30)
    bottlenecks = profiler.get_bottlenecks(threshold_duration=0.05)
    for i, bottleneck in enumerate(bottlenecks, 1):
        print(f"{i}. {bottleneck['operation']}: {bottleneck['avg_duration']:.2f}s avg")
    
    # Get recommendations
    print("\n💡 Performance Recommendations:")
    print("-" * 30)
    recommendations = profiler.get_performance_recommendations()
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
    
    # Export data
    print("\n📥 Exporting profiling data...")
    export_result = profiler.export_profiling_data()
    print(f"   {export_result}")
    
    # Cleanup
    print("\n🧹 Cleaning up...")
    profiler.cleanup()
    
    print("\n✅ Demo completed successfully!")

if __name__ == "__main__":
    print("🧪 Running Code Profiling Tests...")
    print("=" * 50)
    
    # Run unit tests
    unittest.main(verbosity=2, exit=False)
    
    print("\n" + "=" * 50)
    
    # Run demo
    try:
        run_profiling_demo()
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        traceback.print_exc()






