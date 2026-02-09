from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import structlog
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from advanced_code_profiling_optimization import (
from typing import Any, List, Dict, Optional
"""
Advanced Code Profiling and Optimization Demo

This demo showcases comprehensive code profiling and optimization capabilities:

- Multi-level profiling (CPU, GPU, Memory, I/O)
- Bottleneck identification and analysis
- Automatic optimization suggestions
- Data loading and preprocessing optimization
- Performance monitoring and alerting
- Real-world optimization examples
- Performance comparison and visualization
"""



    ProfilingConfig, ProfilingLevel, OptimizationTarget, BottleneckType,
    AdvancedProfiler, CodeOptimizer, PerformanceMonitor,
    profile_function, profile_context
)

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


class LargeTransformerModel(nn.Module):
    """Large transformer model for profiling demonstration."""
    
    def __init__(self, input_dim: int = 768, hidden_dim: int = 1024, num_layers: int = 6):
        
    """__init__ function."""
super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        # Output layers
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 10)
        self.dropout = nn.Dropout(0.1)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module) -> Any:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, x) -> Any:
        # Add sequence dimension if needed
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # Embedding
        x = self.embedding(x)
        
        # Transformer layers
        for layer in self.transformer_layers:
            x = layer(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        # Output
        x = self.layer_norm(x)
        x = self.dropout(x)
        logits = self.classifier(x)
        
        return logits


class SlowDataset(Dataset):
    """Dataset with intentionally slow operations for profiling."""
    
    def __init__(self, num_samples: int = 5000, input_dim: int = 768, slow_loading: bool = True):
        
    """__init__ function."""
self.num_samples = num_samples
        self.input_dim = input_dim
        self.slow_loading = slow_loading
        
        # Generate data
        self.data = torch.randn(num_samples, input_dim)
        self.labels = torch.randint(0, 10, (num_samples,))
        
        logger.info(f"Created dataset with {num_samples} samples")
    
    def __len__(self) -> Any:
        return self.num_samples
    
    def __getitem__(self, idx) -> Optional[Dict[str, Any]]:
        # Simulate slow data loading
        if self.slow_loading:
            time.sleep(0.001)  # 1ms delay
        
        # Simulate slow preprocessing
        data = self.data[idx].clone()
        label = self.labels[idx].clone()
        
        # Additional slow operations
        if self.slow_loading:
            # Simulate complex preprocessing
            data = data.float() / 255.0
            data = torch.clamp(data, 0, 1)
            data = (data - data.mean()) / (data.std() + 1e-8)
        
        return data, label


class FastDataset(Dataset):
    """Optimized dataset for comparison."""
    
    def __init__(self, num_samples: int = 5000, input_dim: int = 768):
        
    """__init__ function."""
self.num_samples = num_samples
        self.input_dim = input_dim
        
        # Pre-process data
        self.data = torch.randn(num_samples, input_dim)
        self.labels = torch.randint(0, 10, (num_samples,))
        
        # Pre-compute normalized data
        self.data = self.data.float() / 255.0
        self.data = torch.clamp(self.data, 0, 1)
        self.data = (self.data - self.data.mean(dim=0)) / (self.data.std(dim=0) + 1e-8)
        
        logger.info(f"Created optimized dataset with {num_samples} samples")
    
    def __len__(self) -> Any:
        return self.num_samples
    
    def __getitem__(self, idx) -> Optional[Dict[str, Any]]:
        # Fast access to pre-processed data
        return self.data[idx], self.labels[idx]


class CodeProfilingOptimizationDemo:
    """Comprehensive demo for code profiling and optimization."""
    
    def __init__(self) -> Any:
        self.results = {}
        self.profiling_data = {}
        
    async def run_data_loading_profiling_demo(self) -> Dict:
        """Demonstrate data loading profiling and optimization."""
        logger.info("Starting Data Loading Profiling Demo")
        
        config = ProfilingConfig(
            enabled=True,
            level=ProfilingLevel.DETAILED,
            auto_optimize=True
        )
        
        profiler = AdvancedProfiler(config)
        optimizer = CodeOptimizer(profiler)
        
        # Create slow dataset
        slow_dataset = SlowDataset(2000, input_dim=512, slow_loading=True)
        slow_dataloader = DataLoader(
            slow_dataset,
            batch_size=32,
            shuffle=True,
            num_workers=0,  # Intentionally slow
            pin_memory=False
        )
        
        # Profile slow data loader
        logger.info("Profiling slow data loader")
        profiler.profile_dataloader(slow_dataloader, num_batches=20)
        
        # Optimize data loader
        logger.info("Optimizing data loader")
        optimized_dataloader = optimizer.optimize_data_loading(slow_dataloader)
        
        # Profile optimized data loader
        logger.info("Profiling optimized data loader")
        profiler.profile_dataloader(optimized_dataloader, num_batches=20)
        
        # Create fast dataset for comparison
        fast_dataset = FastDataset(2000, input_dim=512)
        fast_dataloader = DataLoader(
            fast_dataset,
            batch_size=32,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        # Profile fast data loader
        logger.info("Profiling fast data loader")
        profiler.profile_dataloader(fast_dataloader, num_batches=20)
        
        # Get results
        profiling_summary = profiler.get_profiling_summary()
        optimization_report = optimizer.get_optimization_report()
        
        return {
            'profiling_summary': profiling_summary,
            'optimization_report': optimization_report,
            'comparison': {
                'slow_dataloader': 'Baseline with slow operations',
                'optimized_dataloader': 'Optimized with profiling suggestions',
                'fast_dataloader': 'Pre-optimized for comparison'
            }
        }
    
    async def run_preprocessing_profiling_demo(self) -> Dict:
        """Demonstrate preprocessing profiling and optimization."""
        logger.info("Starting Preprocessing Profiling Demo")
        
        config = ProfilingConfig(
            enabled=True,
            level=ProfilingLevel.DETAILED,
            auto_optimize=True
        )
        
        profiler = AdvancedProfiler(config)
        optimizer = CodeOptimizer(profiler)
        
        # Define slow preprocessing function
        def slow_preprocessing(data) -> Any:
            """Intentionally slow preprocessing function."""
            # Simulate slow operations
            time.sleep(0.002)  # 2ms delay
            
            # Multiple slow operations
            result = data.float()
            result = result / 255.0
            result = torch.clamp(result, 0, 1)
            result = (result - result.mean()) / (result.std() + 1e-8)
            
            # Additional slow operations
            result = torch.nn.functional.normalize(result, p=2, dim=1)
            result = torch.tanh(result)
            
            return result
        
        # Define fast preprocessing function
        def fast_preprocessing(data) -> Any:
            """Optimized preprocessing function."""
            # Vectorized operations
            result = data.float() / 255.0
            result = torch.clamp(result, 0, 1)
            result = (result - result.mean()) / (result.std() + 1e-8)
            result = torch.nn.functional.normalize(result, p=2, dim=1)
            result = torch.tanh(result)
            
            return result
        
        # Generate test data
        test_data = torch.randn(1000, 512)
        
        # Profile slow preprocessing
        logger.info("Profiling slow preprocessing")
        profiler.profile_preprocessing(slow_preprocessing, test_data)
        
        # Optimize preprocessing
        logger.info("Optimizing preprocessing")
        optimized_preprocessing = optimizer.optimize_preprocessing(slow_preprocessing)
        profiler.profile_preprocessing(optimized_preprocessing, test_data)
        
        # Profile fast preprocessing
        logger.info("Profiling fast preprocessing")
        profiler.profile_preprocessing(fast_preprocessing, test_data)
        
        # Get results
        profiling_summary = profiler.get_profiling_summary()
        optimization_report = optimizer.get_optimization_report()
        
        return {
            'profiling_summary': profiling_summary,
            'optimization_report': optimization_report,
            'comparison': {
                'slow_preprocessing': 'Baseline with slow operations',
                'optimized_preprocessing': 'Optimized with profiling suggestions',
                'fast_preprocessing': 'Pre-optimized for comparison'
            }
        }
    
    async def run_training_profiling_demo(self) -> Dict:
        """Demonstrate training loop profiling and optimization."""
        logger.info("Starting Training Loop Profiling Demo")
        
        config = ProfilingConfig(
            enabled=True,
            level=ProfilingLevel.COMPREHENSIVE,
            auto_optimize=True
        )
        
        profiler = AdvancedProfiler(config)
        
        # Create model and data
        model = LargeTransformerModel(input_dim=384, hidden_dim=512, num_layers=4)
        dataset = FastDataset(1000, input_dim=384)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)
        
        # Define training functions
        def slow_training_step(model, batch, optimizer) -> Any:
            """Training step with profiling."""
            data, labels = batch
            
            # Forward pass
            outputs = model(data)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            return loss.item()
        
        def fast_training_step(model, batch, optimizer) -> Any:
            """Optimized training step."""
            data, labels = batch
            
            # Use mixed precision if available
            if torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    outputs = model(data)
                    loss = nn.CrossEntropyLoss()(outputs, labels)
            else:
                outputs = model(data)
                loss = nn.CrossEntropyLoss()(outputs, labels)
            
            # Optimized backward pass
            optimizer.zero_grad(set_to_none=True)  # More efficient
            loss.backward()
            optimizer.step()
            
            return loss.item()
        
        # Profile slow training
        logger.info("Profiling slow training")
        optimizer_slow = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        profiler.start_profiling()
        for i, batch in enumerate(dataloader):
            if i >= 10:  # Limit training steps
                break
            loss = slow_training_step(model, batch, optimizer_slow)
        slow_results = profiler.stop_profiling()
        
        # Profile fast training
        logger.info("Profiling fast training")
        model_fast = LargeTransformerModel(input_dim=384, hidden_dim=512, num_layers=4)
        optimizer_fast = torch.optim.Adam(model_fast.parameters(), lr=1e-4)
        
        profiler.start_profiling()
        for i, batch in enumerate(dataloader):
            if i >= 10:  # Limit training steps
                break
            loss = fast_training_step(model_fast, batch, optimizer_fast)
        fast_results = profiler.stop_profiling()
        
        return {
            'slow_training': slow_results,
            'fast_training': fast_results,
            'comparison': {
                'slow_training': 'Baseline training loop',
                'fast_training': 'Optimized training loop with mixed precision'
            }
        }
    
    async def run_memory_profiling_demo(self) -> Dict:
        """Demonstrate memory profiling and optimization."""
        logger.info("Starting Memory Profiling Demo")
        
        config = ProfilingConfig(
            enabled=True,
            level=ProfilingLevel.DETAILED,
            auto_optimize=True
        )
        
        profiler = AdvancedProfiler(config)
        
        # Memory-intensive operations
        def memory_intensive_operation():
            """Operation that uses a lot of memory."""
            # Create large tensors
            large_tensor = torch.randn(10000, 1000)
            processed_data = []
            
            for i in range(100):
                # Process data and store results
                result = torch.matmul(large_tensor, large_tensor.T)
                processed_data.append(result)
                
                # Simulate some computation
                time.sleep(0.001)
            
            return processed_data
        
        def memory_efficient_operation():
            """Memory-efficient version of the operation."""
            # Create large tensor
            large_tensor = torch.randn(10000, 1000)
            
            for i in range(100):
                # Process data without storing all results
                result = torch.matmul(large_tensor, large_tensor.T)
                
                # Use result immediately or discard
                _ = result.sum()  # Just use the result
                
                # Clear memory
                del result
                
                # Simulate some computation
                time.sleep(0.001)
            
            return "Memory efficient processing completed"
        
        # Profile memory-intensive operation
        logger.info("Profiling memory-intensive operation")
        profiler.start_profiling()
        memory_intensive_operation()
        memory_intensive_results = profiler.stop_profiling()
        
        # Profile memory-efficient operation
        logger.info("Profiling memory-efficient operation")
        profiler.start_profiling()
        memory_efficient_operation()
        memory_efficient_results = profiler.stop_profiling()
        
        return {
            'memory_intensive': memory_intensive_results,
            'memory_efficient': memory_efficient_results,
            'comparison': {
                'memory_intensive': 'Baseline with high memory usage',
                'memory_efficient': 'Optimized with reduced memory usage'
            }
        }
    
    async def run_real_time_monitoring_demo(self) -> Dict:
        """Demonstrate real-time performance monitoring."""
        logger.info("Starting Real-Time Monitoring Demo")
        
        config = ProfilingConfig(
            enabled=True,
            level=ProfilingLevel.REAL_TIME,
            monitoring_interval=0.5,
            alert_threshold=0.7
        )
        
        monitor = PerformanceMonitor(config)
        
        # Add alert callback
        async def alert_callback(alert) -> Any:
            logger.warning(f"Performance alert: {alert['message']}")
        
        monitor.add_alert_callback(alert_callback)
        
        # Start monitoring
        logger.info("Starting performance monitoring")
        monitor.start_monitoring()
        
        # Simulate various workloads
        workloads = [
            ("CPU Intensive", self._cpu_intensive_workload),
            ("Memory Intensive", self._memory_intensive_workload),
            ("I/O Intensive", self._io_intensive_workload),
            ("Mixed Workload", self._mixed_workload)
        ]
        
        monitoring_results = {}
        
        for workload_name, workload_func in workloads:
            logger.info(f"Running {workload_name} workload")
            
            # Run workload for 10 seconds
            start_time = time.time()
            while time.time() - start_time < 10:
                workload_func()
                await asyncio.sleep(0.1)
            
            # Get monitoring summary
            summary = monitor.get_monitoring_summary()
            monitoring_results[workload_name] = summary
        
        # Stop monitoring
        monitor.stop_monitoring()
        
        return {
            'monitoring_results': monitoring_results,
            'workloads': [name for name, _ in workloads]
        }
    
    def _cpu_intensive_workload(self) -> Any:
        """CPU-intensive workload."""
        # Perform CPU-intensive computations
        for _ in range(10000):
            _ = np.random.rand(100, 100).sum()
    
    def _memory_intensive_workload(self) -> Any:
        """Memory-intensive workload."""
        # Allocate and deallocate memory
        large_array = np.random.rand(1000, 1000)
        _ = large_array.sum()
        del large_array
    
    def _io_intensive_workload(self) -> Any:
        """I/O-intensive workload."""
        # Simulate I/O operations
        time.sleep(0.01)
    
    def _mixed_workload(self) -> Any:
        """Mixed workload."""
        # Combine different types of operations
        self._cpu_intensive_workload()
        self._memory_intensive_workload()
        self._io_intensive_workload()
    
    async def run_comprehensive_demo(self) -> Dict:
        """Run comprehensive profiling and optimization demo."""
        logger.info("Starting Comprehensive Code Profiling and Optimization Demo")
        
        results = {}
        
        try:
            # Run individual demos
            results['data_loading'] = await self.run_data_loading_profiling_demo()
            results['preprocessing'] = await self.run_preprocessing_profiling_demo()
            results['training'] = await self.run_training_profiling_demo()
            results['memory'] = await self.run_memory_profiling_demo()
            results['monitoring'] = await self.run_real_time_monitoring_demo()
            
            # Generate comparison report
            comparison = self._generate_comparison_report(results)
            results['comparison'] = comparison
            
            # Save results
            self._save_results(results)
            
            # Plot results
            self.plot_results(results)
            
        except Exception as e:
            logger.error(f"Demo failed: {e}")
            results['error'] = str(e)
        
        return results
    
    def _generate_comparison_report(self, results: Dict) -> Dict:
        """Generate comparison report between different approaches."""
        report = {
            'data_loading_comparison': {},
            'preprocessing_comparison': {},
            'training_comparison': {},
            'memory_comparison': {},
            'recommendations': []
        }
        
        # Analyze data loading results
        if 'data_loading' in results:
            data_loading = results['data_loading']
            if 'profiling_summary' in data_loading:
                summary = data_loading['profiling_summary']
                report['data_loading_comparison'] = {
                    'total_executions': summary.get('total_executions', 0),
                    'avg_execution_time': summary.get('avg_execution_time', 0),
                    'most_common_bottleneck': summary.get('most_common_bottleneck', None)
                }
        
        # Analyze preprocessing results
        if 'preprocessing' in results:
            preprocessing = results['preprocessing']
            if 'profiling_summary' in preprocessing:
                summary = preprocessing['profiling_summary']
                report['preprocessing_comparison'] = {
                    'total_executions': summary.get('total_executions', 0),
                    'avg_execution_time': summary.get('avg_execution_time', 0),
                    'most_common_bottleneck': summary.get('most_common_bottleneck', None)
                }
        
        # Generate recommendations
        if 'data_loading' in results and 'optimization_report' in results['data_loading']:
            optimizations = results['data_loading']['optimization_report']
            if optimizations.get('total_optimizations', 0) > 0:
                report['recommendations'].append(
                    "Data loading optimizations applied successfully"
                )
        
        if 'preprocessing' in results and 'optimization_report' in results['preprocessing']:
            optimizations = results['preprocessing']['optimization_report']
            if optimizations.get('total_optimizations', 0) > 0:
                report['recommendations'].append(
                    "Preprocessing optimizations applied successfully"
                )
        
        return report
    
    def plot_results(self, results: Dict, save_path: str = "code_profiling_results.png"):
        """Plot comprehensive results."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot 1: Data Loading Performance
        if 'data_loading' in results:
            data_loading = results['data_loading']
            if 'profiling_summary' in data_loading:
                summary = data_loading['profiling_summary']
                axes[0, 0].bar(['Slow', 'Optimized', 'Fast'], 
                              [summary.get('avg_execution_time', 0)] * 3)
                axes[0, 0].set_title('Data Loading Performance')
                axes[0, 0].set_ylabel('Execution Time (s)')
        
        # Plot 2: Preprocessing Performance
        if 'preprocessing' in results:
            preprocessing = results['preprocessing']
            if 'profiling_summary' in preprocessing:
                summary = preprocessing['profiling_summary']
                axes[0, 1].bar(['Slow', 'Optimized', 'Fast'], 
                              [summary.get('avg_execution_time', 0)] * 3)
                axes[0, 1].set_title('Preprocessing Performance')
                axes[0, 1].set_ylabel('Execution Time (s)')
        
        # Plot 3: Training Performance
        if 'training' in results:
            training = results['training']
            if 'slow_training' in training and 'fast_training' in training:
                slow_time = training['slow_training']['combined'].execution_time
                fast_time = training['fast_training']['combined'].execution_time
                axes[0, 2].bar(['Slow Training', 'Fast Training'], [slow_time, fast_time])
                axes[0, 2].set_title('Training Performance')
                axes[0, 2].set_ylabel('Execution Time (s)')
        
        # Plot 4: Memory Usage
        if 'memory' in results:
            memory = results['memory']
            if 'memory_intensive' in memory and 'memory_efficient' in memory:
                intensive_memory = memory['memory_intensive']['combined'].memory_usage
                efficient_memory = memory['memory_efficient']['combined'].memory_usage
                axes[1, 0].bar(['Memory Intensive', 'Memory Efficient'], 
                              [intensive_memory, efficient_memory])
                axes[1, 0].set_title('Memory Usage Comparison')
                axes[1, 0].set_ylabel('Memory Usage (GB)')
        
        # Plot 5: Real-time Monitoring
        if 'monitoring' in results:
            monitoring = results['monitoring']
            if 'monitoring_results' in monitoring:
                workloads = list(monitoring['monitoring_results'].keys())
                cpu_usage = [monitoring['monitoring_results'][w].get('avg_cpu_usage', 0) 
                           for w in workloads]
                axes[1, 1].bar(workloads, cpu_usage)
                axes[1, 1].set_title('CPU Usage by Workload')
                axes[1, 1].set_ylabel('CPU Usage (%)')
                axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Plot 6: Optimization Summary
        optimization_counts = []
        optimization_labels = []
        
        for demo_name in ['data_loading', 'preprocessing']:
            if demo_name in results and 'optimization_report' in results[demo_name]:
                optimizations = results[demo_name]['optimization_report']
                count = optimizations.get('total_optimizations', 0)
                optimization_counts.append(count)
                optimization_labels.append(demo_name.replace('_', ' ').title())
        
        if optimization_counts:
            axes[1, 2].bar(optimization_labels, optimization_counts)
            axes[1, 2].set_title('Optimizations Applied')
            axes[1, 2].set_ylabel('Number of Optimizations')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Results plot saved to {save_path}")
    
    def _save_results(self, results: Dict):
        """Save demo results to file."""
        output_path = Path("code_profiling_optimization_results.json")
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj) -> Any:
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Recursively convert numpy types
        def recursive_convert(obj) -> Any:
            if isinstance(obj, dict):
                return {k: recursive_convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [recursive_convert(v) for v in obj]
            else:
                return convert_numpy(obj)
        
        serializable_results = recursive_convert(results)
        
        with open(output_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Demo results saved to {output_path}")


# Example usage with decorators
@profile_function()
def example_function_with_profiling():
    """Example function with automatic profiling."""
    # Simulate some work
    time.sleep(0.1)
    return "Function completed"


async def example_context_profiling():
    """Example of using profiling context manager."""
    with profile_context("example_workload") as profiler:
        # Simulate some work
        time.sleep(0.2)
        
        # Access profiler during execution
        logger.info("Workload in progress...")
        
        # More work
        time.sleep(0.1)


async def main():
    """Main demo function."""
    logger.info("Advanced Code Profiling and Optimization Demo")
    
    # Create demo instance
    demo = CodeProfilingOptimizationDemo()
    
    # Run comprehensive demo
    results = await demo.run_comprehensive_demo()
    
    # Run example profiling
    logger.info("Running example profiling")
    example_function_with_profiling()
    await example_context_profiling()
    
    # Print summary
    logger.info("Demo completed successfully!")
    
    if 'comparison' in results:
        comparison = results['comparison']
        logger.info("Performance Summary:")
        
        if 'data_loading_comparison' in comparison:
            data_loading = comparison['data_loading_comparison']
            logger.info(
                f"Data Loading: {data_loading.get('total_executions', 0)} executions, "
                f"Avg time: {data_loading.get('avg_execution_time', 0):.4f}s"
            )
        
        if 'recommendations' in comparison:
            logger.info("Recommendations:")
            for rec in comparison['recommendations']:
                logger.info(f"- {rec}")


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main()) 