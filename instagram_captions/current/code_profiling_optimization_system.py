"""
Code Profiling and Optimization System
Comprehensive profiling to identify and optimize bottlenecks in data loading, preprocessing, and model training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.profiler as profiler
import cProfile
import pstats
import io
import time
import psutil
import gc
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass
from contextlib import contextmanager
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ProfilingConfig:
    """Configuration for profiling and optimization"""
    # PyTorch profiler settings
    use_torch_profiler: bool = True
    profiler_schedule: profiler.ProfilerAction = profiler.ProfilerAction.RECORD_AND_SAVE
    record_shapes: bool = True
    profile_memory: bool = True
    with_stack: bool = True
    
    # Custom profiling settings
    profile_data_loading: bool = True
    profile_preprocessing: bool = True
    profile_model_forward: bool = True
    profile_model_backward: bool = True
    
    # Performance monitoring
    monitor_memory: bool = True
    monitor_gpu: bool = True
    monitor_cpu: bool = True
    
    # Optimization settings
    enable_optimizations: bool = True
    use_mixed_precision: bool = True
    use_compile: bool = False
    use_channels_last: bool = False
    
    # Output settings
    save_profiles: bool = True
    output_dir: str = "profiling_results"
    verbose: bool = True

class PerformanceMonitor:
    """Monitor system performance metrics"""
    
    def __init__(self, config: ProfilingConfig):
        self.config = config
        self.metrics = {
            'cpu_usage': [],
            'memory_usage': [],
            'gpu_memory': [],
            'timestamps': []
        }
        self.start_time = None
        
    def start_monitoring(self):
        """Start performance monitoring"""
        self.start_time = time.time()
        logger.info("Performance monitoring started")
        
    def record_metrics(self):
        """Record current performance metrics"""
        if self.start_time is None:
            return
            
        timestamp = time.time() - self.start_time
        
        if self.config.monitor_cpu:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self.metrics['cpu_usage'].append(cpu_percent)
            
        if self.config.monitor_memory:
            memory = psutil.virtual_memory()
            self.metrics['memory_usage'].append(memory.percent)
            
        if self.config.monitor_gpu and torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
            self.metrics['gpu_memory'].append(gpu_memory)
            
        self.metrics['timestamps'].append(timestamp)
        
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        summary = {}
        
        if self.metrics['cpu_usage']:
            summary['cpu'] = {
                'mean': np.mean(self.metrics['cpu_usage']),
                'max': np.max(self.metrics['cpu_usage']),
                'min': np.min(self.metrics['cpu_usage'])
            }
            
        if self.metrics['memory_usage']:
            summary['memory'] = {
                'mean': np.mean(self.metrics['memory_usage']),
                'max': np.max(self.metrics['memory_usage']),
                'min': np.min(self.metrics['memory_usage'])
            }
            
        if self.metrics['gpu_memory']:
            summary['gpu_memory'] = {
                'mean': np.mean(self.metrics['gpu_memory']),
                'max': np.max(self.metrics['gpu_memory']),
                'min': np.min(self.metrics['gpu_memory'])
            }
            
        summary['duration'] = self.metrics['timestamps'][-1] if self.metrics['timestamps'] else 0
        
        return summary
        
    def plot_metrics(self, save_path: Optional[str] = None):
        """Plot performance metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # CPU Usage
        if self.metrics['cpu_usage']:
            axes[0, 0].plot(self.metrics['timestamps'], self.metrics['cpu_usage'])
            axes[0, 0].set_title('CPU Usage (%)')
            axes[0, 0].set_xlabel('Time (s)')
            axes[0, 0].set_ylabel('CPU %')
            
        # Memory Usage
        if self.metrics['memory_usage']:
            axes[0, 1].plot(self.metrics['timestamps'], self.metrics['memory_usage'])
            axes[0, 1].set_title('Memory Usage (%)')
            axes[0, 1].set_xlabel('Time (s)')
            axes[0, 1].set_ylabel('Memory %')
            
        # GPU Memory
        if self.metrics['gpu_memory']:
            axes[1, 0].plot(self.metrics['timestamps'], self.metrics['gpu_memory'])
            axes[1, 0].set_title('GPU Memory Usage (GB)')
            axes[1, 0].set_xlabel('Time (s)')
            axes[1, 0].set_ylabel('Memory (GB)')
            
        # Combined view
        if self.metrics['timestamps']:
            axes[1, 1].text(0.1, 0.5, f"Duration: {self.metrics['timestamps'][-1]:.2f}s", 
                           transform=axes[1, 1].transAxes, fontsize=12)
            axes[1, 1].set_title('Summary')
            axes[1, 1].axis('off')
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Performance metrics plot saved to {save_path}")
        else:
            plt.show()

class PyTorchProfiler:
    """PyTorch profiler wrapper for model profiling"""
    
    def __init__(self, config: ProfilingConfig):
        self.config = config
        self.profiler = None
        self.profile_results = {}
        
    def start_profiling(self, model: nn.Module, dataloader, num_steps: int = 10):
        """Start PyTorch profiling"""
        if not self.config.use_torch_profiler:
            return
            
        logger.info("Starting PyTorch profiling...")
        
        # Create profiler
        self.profiler = profiler.profile(
            activities=[
                profiler.ProfilerActivity.CPU,
                profiler.ProfilerActivity.CUDA,
            ],
            schedule=self.config.profiler_schedule,
            on_trace_ready=profiler.tensorboard_trace_handler(self.config.output_dir),
            record_shapes=self.config.record_shapes,
            profile_memory=self.config.profile_memory,
            with_stack=self.config.with_stack,
        )
        
        # Profile model
        self.profiler.start()
        
        model.train()
        step_count = 0
        
        for batch in dataloader:
            if step_count >= num_steps:
                break
                
            # Forward pass
            if hasattr(model, 'forward'):
                with torch.no_grad():
                    _ = model(batch)
            else:
                # Handle different model types
                if isinstance(batch, (list, tuple)):
                    _ = model(*batch)
                else:
                    _ = model(batch)
                    
            step_count += 1
            
        self.profiler.stop()
        logger.info("PyTorch profiling completed")
        
    def get_profile_summary(self) -> Dict[str, Any]:
        """Get profiling summary"""
        if not self.profiler:
            return {}
            
        # Get key metrics
        key_averages = self.profiler.key_averages()
        
        summary = {
            'total_time': 0,
            'cpu_time': 0,
            'cuda_time': 0,
            'top_operations': [],
            'memory_usage': {}
        }
        
        for event in key_averages:
            summary['total_time'] += event.self_cpu_time_total / 1000  # Convert to seconds
            
            if event.device_type == 'cpu':
                summary['cpu_time'] += event.self_cpu_time_total / 1000
            elif event.device_type == 'cuda':
                summary['cuda_time'] += event.self_cuda_time_total / 1000
                
            # Top operations by time
            summary['top_operations'].append({
                'name': event.name,
                'cpu_time': event.self_cpu_time_total / 1000,
                'cuda_time': event.self_cuda_time_total / 1000,
                'count': event.count
            })
            
        # Sort by total time
        summary['top_operations'].sort(key=lambda x: x['cpu_time'] + x['cuda_time'], reverse=True)
        summary['top_operations'] = summary['top_operations'][:10]  # Top 10
        
        return summary

class DataLoadingProfiler:
    """Profile data loading and preprocessing bottlenecks"""
    
    def __init__(self, config: ProfilingConfig):
        self.config = config
        self.loading_times = []
        self.preprocessing_times = []
        self.batch_sizes = []
        
    def profile_dataloader(self, dataloader, num_batches: int = 50) -> Dict[str, Any]:
        """Profile dataloader performance"""
        if not self.config.profile_data_loading:
            return {}
            
        logger.info("Profiling dataloader performance...")
        
        start_time = time.time()
        batch_count = 0
        
        for batch in dataloader:
            if batch_count >= num_batches:
                break
                
            batch_start = time.time()
            
            # Measure loading time
            if hasattr(dataloader.dataset, '__getitem__'):
                # Simulate data access
                _ = dataloader.dataset[0]
                
            loading_time = time.time() - batch_start
            self.loading_times.append(loading_time)
            
            # Measure preprocessing time
            preprocess_start = time.time()
            
            if isinstance(batch, (list, tuple)):
                batch_size = len(batch[0]) if batch[0] is not None else 1
            else:
                batch_size = batch.shape[0] if hasattr(batch, 'shape') else 1
                
            self.batch_sizes.append(batch_size)
            
            # Simulate preprocessing
            if isinstance(batch, torch.Tensor):
                _ = batch.to('cpu')  # Simulate device transfer
                _ = batch.float()     # Simulate dtype conversion
                
            preprocessing_time = time.time() - preprocess_start
            self.preprocessing_times.append(preprocessing_time)
            
            batch_count += 1
            
        total_time = time.time() - start_time
        
        # Calculate statistics
        stats = {
            'total_batches': batch_count,
            'total_time': total_time,
            'avg_loading_time': np.mean(self.loading_times),
            'avg_preprocessing_time': np.mean(self.preprocessing_times),
            'total_loading_time': np.sum(self.loading_times),
            'total_preprocessing_time': np.sum(self.preprocessing_times),
            'loading_bottleneck': np.mean(self.loading_times) > np.mean(self.preprocessing_times),
            'throughput': batch_count / total_time,  # batches per second
            'avg_batch_size': np.mean(self.batch_sizes)
        }
        
        return stats
        
    def identify_bottlenecks(self, stats: Dict[str, Any]) -> List[str]:
        """Identify data loading bottlenecks"""
        bottlenecks = []
        
        # Loading time bottleneck
        if stats['avg_loading_time'] > 0.1:  # More than 100ms
            bottlenecks.append("Slow data loading - consider using multiple workers or faster storage")
            
        # Preprocessing bottleneck
        if stats['avg_preprocessing_time'] > 0.05:  # More than 50ms
            bottlenecks.append("Slow preprocessing - consider optimizing transforms or using GPU preprocessing")
            
        # Throughput bottleneck
        if stats['throughput'] < 10:  # Less than 10 batches per second
            bottlenecks.append("Low throughput - consider increasing batch size or optimizing data pipeline")
            
        # Memory bottleneck
        if stats['avg_batch_size'] > 100:
            bottlenecks.append("Large batch sizes - consider reducing batch size or using gradient accumulation")
            
        return bottlenecks
        
    def optimize_dataloader(self, dataloader, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest dataloader optimizations"""
        optimizations = {}
        
        # Worker optimization
        if stats['avg_loading_time'] > 0.05:
            optimizations['num_workers'] = min(8, os.cpu_count())
            optimizations['pin_memory'] = True
            optimizations['persistent_workers'] = True
            
        # Batch size optimization
        if stats['throughput'] < 20:
            optimizations['batch_size'] = min(stats['avg_batch_size'] * 2, 128)
            
        # Memory optimization
        if stats['avg_batch_size'] > 50:
            optimizations['gradient_accumulation'] = 2
            optimizations['mixed_precision'] = True
            
        return optimizations

class ModelOptimizer:
    """Optimize model performance based on profiling results"""
    
    def __init__(self, config: ProfilingConfig):
        self.config = config
        self.optimizations_applied = []
        
    def optimize_model(self, model: nn.Module, profile_summary: Dict[str, Any]) -> nn.Module:
        """Apply optimizations to model based on profiling"""
        if not self.config.enable_optimizations:
            return model
            
        logger.info("Applying model optimizations...")
        
        # Mixed precision optimization
        if self.config.use_mixed_precision:
            model = self._apply_mixed_precision(model)
            
        # JIT compilation
        if self.config.use_compile and hasattr(torch, 'compile'):
            model = self._apply_jit_compilation(model)
            
        # Memory format optimization
        if self.config.use_channels_last:
            model = self._apply_channels_last(model)
            
        # Additional optimizations based on profiling
        if profile_summary.get('memory_usage', {}).get('peak', 0) > 8:  # More than 8GB
            model = self._apply_memory_optimizations(model)
            
        return model
        
    def _apply_mixed_precision(self, model: nn.Module) -> nn.Module:
        """Apply mixed precision optimization"""
        try:
            model = model.half()  # Convert to FP16
            self.optimizations_applied.append("Mixed precision (FP16)")
            logger.info("Applied mixed precision optimization")
        except Exception as e:
            logger.warning(f"Failed to apply mixed precision: {e}")
            
        return model
        
    def _apply_jit_compilation(self, model: nn.Module) -> nn.Module:
        """Apply JIT compilation optimization"""
        try:
            if hasattr(torch, 'compile'):
                model = torch.compile(model, mode="reduce-overhead")
                self.optimizations_applied.append("JIT compilation")
                logger.info("Applied JIT compilation optimization")
        except Exception as e:
            logger.warning(f"Failed to apply JIT compilation: {e}")
            
        return model
        
    def _apply_channels_last(self, model: nn.Module) -> nn.Module:
        """Apply channels last memory format optimization"""
        try:
            model = model.to(memory_format=torch.channels_last)
            self.optimizations_applied.append("Channels last memory format")
            logger.info("Applied channels last memory format optimization")
        except Exception as e:
            logger.warning(f"Failed to apply channels last: {e}")
            
        return model
        
    def _apply_memory_optimizations(self, model: nn.Module) -> nn.Module:
        """Apply memory optimizations"""
        try:
            # Enable gradient checkpointing
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
                self.optimizations_applied.append("Gradient checkpointing")
                
            # Enable memory efficient attention
            if hasattr(model, 'enable_memory_efficient_attention'):
                model.enable_memory_efficient_attention()
                self.optimizations_applied.append("Memory efficient attention")
                
            logger.info("Applied memory optimizations")
        except Exception as e:
            logger.warning(f"Failed to apply memory optimizations: {e}")
            
        return model
        
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of applied optimizations"""
        return {
            'optimizations_applied': self.optimizations_applied,
            'total_optimizations': len(self.optimizations_applied),
            'mixed_precision': "Mixed precision (FP16)" in self.optimizations_applied,
            'jit_compilation': "JIT compilation" in self.optimizations_applied,
            'channels_last': "Channels last memory format" in self.optimizations_applied,
            'memory_optimizations': any("memory" in opt.lower() for opt in self.optimizations_applied)
        }

class CodeProfiler:
    """Main code profiling and optimization system"""
    
    def __init__(self, config: ProfilingConfig):
        self.config = config
        self.performance_monitor = PerformanceMonitor(config)
        self.pytorch_profiler = PyTorchProfiler(config)
        self.data_loading_profiler = DataLoadingProfiler(config)
        self.model_optimizer = ModelOptimizer(config)
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
    def profile_training_pipeline(self, model: nn.Module, dataloader, 
                                num_steps: int = 50) -> Dict[str, Any]:
        """Profile entire training pipeline"""
        logger.info("Starting comprehensive training pipeline profiling...")
        
        # Start performance monitoring
        self.performance_monitor.start_monitoring()
        
        # Profile data loading
        data_stats = self.data_loading_profiler.profile_dataloader(dataloader, num_steps)
        
        # Profile model with PyTorch profiler
        self.pytorch_profiler.start_profiling(model, dataloader, num_steps)
        profile_summary = self.pytorch_profiler.get_profile_summary()
        
        # Stop monitoring
        self.performance_monitor.record_metrics()
        
        # Identify bottlenecks
        data_bottlenecks = self.data_loading_profiler.identify_bottlenecks(data_stats)
        data_optimizations = self.data_loading_profiler.optimize_dataloader(dataloader, data_stats)
        
        # Apply model optimizations
        optimized_model = self.model_optimizer.optimize_model(model, profile_summary)
        optimization_summary = self.model_optimizer.get_optimization_summary()
        
        # Get performance summary
        performance_summary = self.performance_monitor.get_summary()
        
        # Compile comprehensive report
        report = {
            'data_loading': data_stats,
            'data_bottlenecks': data_bottlenecks,
            'data_optimizations': data_optimizations,
            'model_profiling': profile_summary,
            'model_optimizations': optimization_summary,
            'performance_monitoring': performance_summary,
            'recommendations': self._generate_recommendations(
                data_stats, profile_summary, performance_summary
            )
        }
        
        # Save results
        if self.config.save_profiles:
            self._save_profiling_results(report)
            
        return report
        
    def _generate_recommendations(self, data_stats: Dict, profile_summary: Dict, 
                                performance_summary: Dict) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        # Data loading recommendations
        if data_stats.get('loading_bottleneck', False):
            recommendations.append("Increase num_workers in DataLoader")
            recommendations.append("Use pin_memory=True for faster GPU transfer")
            recommendations.append("Consider using persistent_workers=True")
            
        if data_stats.get('throughput', 0) < 20:
            recommendations.append("Increase batch size for better throughput")
            recommendations.append("Use gradient accumulation for large effective batch sizes")
            
        # Model optimization recommendations
        if profile_summary.get('cuda_time', 0) > profile_summary.get('cpu_time', 0):
            recommendations.append("Model is GPU-bound - consider mixed precision training")
            recommendations.append("Use gradient accumulation to reduce memory pressure")
            
        if profile_summary.get('memory_usage', {}).get('peak', 0) > 8:
            recommendations.append("High memory usage - enable gradient checkpointing")
            recommendations.append("Consider using smaller batch sizes")
            
        # Performance recommendations
        if performance_summary.get('gpu_memory', {}).get('max', 0) > 0.8:  # More than 80% GPU memory
            recommendations.append("GPU memory usage is high - optimize batch size")
            recommendations.append("Enable mixed precision training")
            
        if performance_summary.get('cpu', {}).get('max', 0) > 80:  # More than 80% CPU
            recommendations.append("CPU usage is high - optimize data preprocessing")
            recommendations.append("Use more workers in DataLoader")
            
        return recommendations
        
    def _save_profiling_results(self, report: Dict[str, Any]):
        """Save profiling results to files"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save JSON report
        json_path = os.path.join(self.config.output_dir, f"profiling_report_{timestamp}.json")
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Profiling report saved to {json_path}")
        
        # Save performance plots
        plot_path = os.path.join(self.config.output_dir, f"performance_metrics_{timestamp}.png")
        self.performance_monitor.plot_metrics(plot_path)
        
        # Save PyTorch profiler traces
        if self.config.use_torch_profiler:
            trace_path = os.path.join(self.config.output_dir, f"torch_profiler_traces_{timestamp}")
            logger.info(f"PyTorch profiler traces saved to {trace_path}")
            
    def profile_function(self, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """Profile a specific function using cProfile"""
        logger.info(f"Profiling function: {func.__name__}")
        
        # Create profiler
        pr = cProfile.Profile()
        pr.enable()
        
        # Record start time and memory
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024**2  # MB
        
        # Run function
        result = func(*args, **kwargs)
        
        # Record end time and memory
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024**2  # MB
        
        pr.disable()
        
        # Get profiling stats
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats(20)  # Top 20 functions
        
        # Compile results
        profile_results = {
            'function_name': func.__name__,
            'execution_time': end_time - start_time,
            'memory_delta': end_memory - start_memory,
            'start_memory': start_memory,
            'end_memory': end_memory,
            'profiling_stats': s.getvalue(),
            'result': result
        }
        
        logger.info(f"Function {func.__name__} executed in {profile_results['execution_time']:.4f}s")
        
        return profile_results
        
    def benchmark_optimizations(self, model: nn.Module, dataloader, 
                              num_steps: int = 10) -> Dict[str, Any]:
        """Benchmark model with and without optimizations"""
        logger.info("Benchmarking model optimizations...")
        
        # Benchmark original model
        original_times = []
        original_memory = []
        
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= num_steps:
                    break
                    
                start_time = time.time()
                start_memory = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
                
                if isinstance(batch, (list, tuple)):
                    _ = model(*batch)
                else:
                    _ = model(batch)
                    
                end_time = time.time()
                end_memory = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
                
                original_times.append(end_time - start_time)
                original_memory.append(end_memory - start_memory)
                
        # Apply optimizations
        optimized_model = self.model_optimizer.optimize_model(model, {})
        
        # Benchmark optimized model
        optimized_times = []
        optimized_memory = []
        
        optimized_model.eval()
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= num_steps:
                    break
                    
                start_time = time.time()
                start_memory = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
                
                if isinstance(batch, (list, tuple)):
                    _ = optimized_model(*batch)
                else:
                    _ = optimized_model(batch)
                    
                end_time = time.time()
                end_memory = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
                
                optimized_times.append(end_time - start_time)
                optimized_memory.append(end_memory - start_memory)
                
        # Calculate improvements
        time_improvement = (np.mean(original_times) - np.mean(optimized_times)) / np.mean(original_times) * 100
        memory_improvement = (np.mean(original_memory) - np.mean(optimized_memory)) / np.mean(original_memory) * 100
        
        benchmark_results = {
            'original': {
                'avg_time': np.mean(original_times),
                'avg_memory': np.mean(original_memory),
                'std_time': np.std(original_times),
                'std_memory': np.std(original_memory)
            },
            'optimized': {
                'avg_time': np.mean(optimized_times),
                'avg_memory': np.mean(optimized_memory),
                'std_time': np.std(optimized_times),
                'std_memory': np.std(optimized_memory)
            },
            'improvements': {
                'time_improvement_percent': time_improvement,
                'memory_improvement_percent': memory_improvement,
                'speedup_factor': np.mean(original_times) / np.mean(optimized_times)
            }
        }
        
        logger.info(f"Time improvement: {time_improvement:.2f}%")
        logger.info(f"Memory improvement: {memory_improvement:.2f}%")
        
        return benchmark_results

def create_profiling_config(**kwargs) -> ProfilingConfig:
    """Create profiling configuration with default values"""
    return ProfilingConfig(**kwargs)

def main():
    """Main function to demonstrate code profiling and optimization"""
    
    # Create configuration
    config = create_profiling_config(
        use_torch_profiler=True,
        profile_data_loading=True,
        profile_preprocessing=True,
        enable_optimizations=True,
        save_profiles=True
    )
    
    # Initialize profiler
    profiler = CodeProfiler(config)
    
    # Create dummy model and data for demonstration
    model = nn.Sequential(
        nn.Linear(1000, 500),
        nn.ReLU(),
        nn.Linear(500, 100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    
    # Create dummy dataloader
    dummy_data = torch.randn(100, 1000)
    dummy_labels = torch.randint(0, 10, (100,))
    dataset = torch.utils.data.TensorDataset(dummy_data, dummy_labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Profile training pipeline
    logger.info("=== Starting Training Pipeline Profiling ===")
    pipeline_report = profiler.profile_training_pipeline(model, dataloader, num_steps=20)
    
    # Benchmark optimizations
    logger.info("=== Starting Optimization Benchmarking ===")
    benchmark_results = profiler.benchmark_optimizations(model, dataloader, num_steps=10)
    
    # Profile specific function
    logger.info("=== Starting Function Profiling ===")
    
    def example_function(data, iterations=1000):
        result = 0
        for i in range(iterations):
            result += torch.sum(data * i)
        return result
    
    function_profile = profiler.profile_function(example_function, dummy_data, 1000)
    
    # Print summary
    logger.info("=== Profiling Summary ===")
    logger.info(f"Data loading bottlenecks: {len(pipeline_report['data_bottlenecks'])}")
    logger.info(f"Model optimizations applied: {len(pipeline_report['model_optimizations']['optimizations_applied'])}")
    logger.info(f"Performance recommendations: {len(pipeline_report['recommendations'])}")
    logger.info(f"Time improvement: {benchmark_results['improvements']['time_improvement_percent']:.2f}%")
    logger.info(f"Memory improvement: {benchmark_results['improvements']['memory_improvement_percent']:.2f}%")
    
    logger.info("Code profiling and optimization system test completed successfully!")

if __name__ == "__main__":
    main()


