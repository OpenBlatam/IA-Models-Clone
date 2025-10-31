"""
Performance Optimization for TruthGPT API
=========================================

Performance optimization utilities for TruthGPT API.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Union
import time
import psutil
import gc


class PerformanceOptimizer:
    """
    Performance optimizer for TruthGPT API models.
    
    Provides various optimization techniques to improve
    model performance and efficiency.
    """
    
    def __init__(self):
        """Initialize performance optimizer."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.optimization_history = []
    
    def optimize_model(self, 
                      model: Any,
                      optimization_level: str = 'medium',
                      target_metrics: List[str] = None) -> Any:
        """
        Optimize model for performance.
        
        Args:
            model: Model to optimize
            optimization_level: Level of optimization ('low', 'medium', 'high', 'extreme')
            target_metrics: Target metrics to optimize for
            
        Returns:
            Optimized model
        """
        print(f"🚀 Optimizing model with {optimization_level} optimization level...")
        
        start_time = time.time()
        
        # Move model to optimal device
        model = self._move_to_device(model)
        
        # Apply optimizations based on level
        if optimization_level == 'low':
            model = self._apply_low_optimizations(model)
        elif optimization_level == 'medium':
            model = self._apply_medium_optimizations(model)
        elif optimization_level == 'high':
            model = self._apply_high_optimizations(model)
        elif optimization_level == 'extreme':
            model = self._apply_extreme_optimizations(model)
        
        # Record optimization time
        optimization_time = time.time() - start_time
        
        # Store optimization info
        self.optimization_history.append({
            'level': optimization_level,
            'time': optimization_time,
            'target_metrics': target_metrics
        })
        
        print(f"✅ Model optimization completed in {optimization_time:.2f} seconds!")
        
        return model
    
    def _move_to_device(self, model: Any) -> Any:
        """Move model to optimal device."""
        if hasattr(model, 'to'):
            model = model.to(self.device)
        return model
    
    def _apply_low_optimizations(self, model: Any) -> Any:
        """Apply low-level optimizations."""
        print("   Applying low-level optimizations...")
        
        # Enable mixed precision if available
        if hasattr(torch.cuda, 'amp') and torch.cuda.is_available():
            model = torch.cuda.amp.autocast()(model)
        
        return model
    
    def _apply_medium_optimizations(self, model: Any) -> Any:
        """Apply medium-level optimizations."""
        print("   Applying medium-level optimizations...")
        
        # Apply low optimizations
        model = self._apply_low_optimizations(model)
        
        # Enable JIT compilation if possible
        try:
            if hasattr(model, 'eval'):
                model.eval()
                model = torch.jit.script(model)
        except Exception as e:
            print(f"   JIT compilation failed: {e}")
        
        return model
    
    def _apply_high_optimizations(self, model: Any) -> Any:
        """Apply high-level optimizations."""
        print("   Applying high-level optimizations...")
        
        # Apply medium optimizations
        model = self._apply_medium_optimizations(model)
        
        # Optimize memory usage
        model = self._optimize_memory_usage(model)
        
        # Enable gradient checkpointing if available
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        
        return model
    
    def _apply_extreme_optimizations(self, model: Any) -> Any:
        """Apply extreme optimizations."""
        print("   Applying extreme optimizations...")
        
        # Apply high optimizations
        model = self._apply_high_optimizations(model)
        
        # Enable all available optimizations
        model = self._enable_all_optimizations(model)
        
        return model
    
    def _optimize_memory_usage(self, model: Any) -> Any:
        """Optimize memory usage."""
        print("   Optimizing memory usage...")
        
        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Force garbage collection
        gc.collect()
        
        return model
    
    def _enable_all_optimizations(self, model: Any) -> Any:
        """Enable all available optimizations."""
        print("   Enabling all optimizations...")
        
        # Enable cuDNN optimizations
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        
        return model
    
    def benchmark_model(self, 
                       model: Any,
                       x_test: torch.Tensor,
                       y_test: torch.Tensor,
                       num_runs: int = 10) -> Dict[str, Any]:
        """
        Benchmark model performance.
        
        Args:
            model: Model to benchmark
            x_test: Test data
            y_test: Test labels
            num_runs: Number of benchmark runs
            
        Returns:
            Benchmark results
        """
        print(f"📊 Benchmarking model performance ({num_runs} runs)...")
        
        # Move data to device
        x_test = x_test.to(self.device)
        y_test = y_test.to(self.device)
        
        # Warmup
        model.eval()
        with torch.no_grad():
            for _ in range(3):
                _ = model(x_test[:10])
        
        # Benchmark inference
        inference_times = []
        memory_usage = []
        
        for i in range(num_runs):
            # Measure inference time
            start_time = time.time()
            
            with torch.no_grad():
                predictions = model(x_test)
            
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            # Measure memory usage
            if torch.cuda.is_available():
                memory_usage.append(torch.cuda.memory_allocated())
            else:
                memory_usage.append(psutil.Process().memory_info().rss)
        
        # Calculate statistics
        avg_inference_time = np.mean(inference_times)
        std_inference_time = np.std(inference_times)
        min_inference_time = np.min(inference_times)
        max_inference_time = np.max(inference_times)
        
        avg_memory_usage = np.mean(memory_usage)
        max_memory_usage = np.max(memory_usage)
        
        results = {
            'inference_time': {
                'mean': avg_inference_time,
                'std': std_inference_time,
                'min': min_inference_time,
                'max': max_inference_time
            },
            'memory_usage': {
                'mean': avg_memory_usage,
                'max': max_memory_usage
            },
            'throughput': 1.0 / avg_inference_time,
            'device': str(self.device)
        }
        
        print(f"✅ Benchmark completed!")
        print(f"   Average inference time: {avg_inference_time:.4f}s")
        print(f"   Throughput: {results['throughput']:.2f} samples/s")
        print(f"   Memory usage: {avg_memory_usage / 1024 / 1024:.2f} MB")
        
        return results
    
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get optimization history."""
        return self.optimization_history
    
    def clear_optimization_history(self):
        """Clear optimization history."""
        self.optimization_history = []
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        info = {
            'device': str(self.device),
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'pytorch_version': torch.__version__,
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total,
            'memory_available': psutil.virtual_memory().available
        }
        
        if torch.cuda.is_available():
            info.update({
                'cuda_device_count': torch.cuda.device_count(),
                'cuda_device_name': torch.cuda.get_device_name(0),
                'cuda_memory_total': torch.cuda.get_device_properties(0).total_memory
            })
        
        return info


def optimize_model_performance(model: Any, 
                              optimization_level: str = 'medium',
                              target_metrics: List[str] = None) -> Any:
    """
    Convenience function to optimize model performance.
    
    Args:
        model: Model to optimize
        optimization_level: Level of optimization
        target_metrics: Target metrics to optimize for
        
    Returns:
        Optimized model
    """
    optimizer = PerformanceOptimizer()
    return optimizer.optimize_model(model, optimization_level, target_metrics)


def benchmark_model_performance(model: Any,
                               x_test: torch.Tensor,
                               y_test: torch.Tensor,
                               num_runs: int = 10) -> Dict[str, Any]:
    """
    Convenience function to benchmark model performance.
    
    Args:
        model: Model to benchmark
        x_test: Test data
        y_test: Test labels
        num_runs: Number of benchmark runs
        
    Returns:
        Benchmark results
    """
    optimizer = PerformanceOptimizer()
    return optimizer.benchmark_model(model, x_test, y_test, num_runs)









