#!/usr/bin/env python3
"""
Performance Comparison: Original vs Optimized SEO Evaluation System
Demonstrates the performance improvements achieved through optimization
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import asyncio
import psutil
import os
from typing import Dict, List, Tuple
import warnings

# Import both systems for comparison
from evaluation_metrics import SEOModelEvaluator, SEOMetricsConfig, ClassificationMetricsConfig, RegressionMetricsConfig
from evaluation_metrics_optimized import (
    OptimizedSEOModelEvaluator, SEOMetricsConfig as OptimizedSEOMetricsConfig,
    ClassificationMetricsConfig as OptimizedClassificationMetricsConfig,
    RegressionMetricsConfig as OptimizedRegressionMetricsConfig
)

warnings.filterwarnings('ignore')

class PerformanceBenchmark:
    """Benchmark class to compare original vs optimized systems."""
    
    def __init__(self):
        self.results = {
            'original': {},
            'optimized': {},
            'improvements': {}
        }
        self.dataset_sizes = [1000, 5000, 10000, 50000, 100000]
        
    def generate_test_data(self, size: int) -> Dict:
        """Generate test data of specified size."""
        return {
            'y_true': np.random.randint(0, 2, size),
            'y_pred': np.random.randint(0, 2, size),
            'y_prob': np.random.random(size),
            'content_data': {
                'content_length': np.random.randint(200, 2000, size),
                'keyword_density': np.random.random(size) * 0.05,
                'readability_score': np.random.random(size) * 100
            },
            'engagement_data': {
                'time_on_page': np.random.random(size) * 300,
                'click_through_rate': np.random.random(size) * 0.1,
                'bounce_rate': np.random.random(size)
            },
            'technical_data': {
                'core_web_vitals': {
                    'lcp': np.random.random(size) * 5000,
                    'fid': np.random.random(size) * 200,
                    'cls': np.random.random(size) * 0.2
                },
                'mobile_friendly': np.random.random(size),
                'page_load_speed': np.random.random(size) * 5000
            }
        }
    
    def benchmark_original_system(self, data: Dict, task_type: str) -> Tuple[float, float]:
        """Benchmark the original system."""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Create original evaluator
        seo_config = SEOMetricsConfig()
        classification_config = ClassificationMetricsConfig()
        regression_config = RegressionMetricsConfig()
        
        evaluator = SEOModelEvaluator(
            seo_config=seo_config,
            classification_config=classification_config,
            regression_config=regression_config
        )
        
        # Run evaluation
        if task_type == "classification":
            results = evaluator.classification_metrics.calculate_metrics(
                data['y_true'], data['y_pred'], data['y_prob']
            )
        elif task_type == "ranking":
            results = evaluator.seo_metrics.calculate_ranking_metrics(
                data['y_true'], data['y_pred'], data['y_prob']
            )
        else:  # regression
            results = evaluator.regression_metrics.calculate_metrics(
                data['y_true'], data['y_pred']
            )
        
        # Calculate SEO metrics
        content_metrics = evaluator.seo_metrics.calculate_content_quality_metrics(data['content_data'])
        engagement_metrics = evaluator.seo_metrics.calculate_user_engagement_metrics(data['engagement_data'])
        technical_metrics = evaluator.seo_metrics.calculate_technical_seo_metrics(data['technical_data'])
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        execution_time = end_time - start_time
        memory_usage = end_memory - start_memory
        
        return execution_time, memory_usage
    
    async def benchmark_optimized_system(self, data: Dict, task_type: str) -> Tuple[float, float]:
        """Benchmark the optimized system."""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Create optimized evaluator
        seo_config = OptimizedSEOMetricsConfig(
            use_vectorization=True,
            use_caching=True,
            batch_size=10000,
            max_workers=os.cpu_count()
        )
        classification_config = OptimizedClassificationMetricsConfig(use_vectorization=True)
        regression_config = OptimizedRegressionMetricsConfig(use_vectorization=True)
        
        evaluator = OptimizedSEOModelEvaluator(
            seo_config=seo_config,
            classification_config=classification_config,
            regression_config=regression_config
        )
        
        try:
            # Run evaluation
            results = await evaluator.evaluate_seo_model(
                model=None,
                test_data=data,
                task_type=task_type
            )
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            execution_time = end_time - start_time
            memory_usage = end_memory - start_memory
            
            return execution_time, memory_usage
            
        finally:
            evaluator.cleanup()
    
    def run_benchmarks(self, task_type: str = "classification") -> Dict:
        """Run comprehensive benchmarks for all dataset sizes."""
        print(f"🚀 Running benchmarks for task type: {task_type}")
        print("=" * 60)
        
        for size in self.dataset_sizes:
            print(f"\n📊 Testing dataset size: {size:,}")
            
            # Generate test data
            test_data = self.generate_test_data(size)
            
            # Benchmark original system
            print("  🔄 Testing original system...")
            orig_time, orig_memory = self.benchmark_original_system(test_data, task_type)
            
            # Benchmark optimized system
            print("  ⚡ Testing optimized system...")
            opt_time, opt_memory = asyncio.run(self.benchmark_optimized_system(test_data, task_type))
            
            # Store results
            self.results['original'][size] = {
                'execution_time': orig_time,
                'memory_usage': orig_memory
            }
            
            self.results['optimized'][size] = {
                'execution_time': opt_time,
                'memory_usage': opt_memory
            }
            
            # Calculate improvements
            time_improvement = ((orig_time - opt_time) / orig_time) * 100
            memory_improvement = ((orig_memory - opt_memory) / orig_memory) * 100 if orig_memory > 0 else 0
            
            self.results['improvements'][size] = {
                'time_improvement': time_improvement,
                'memory_improvement': memory_improvement
            }
            
            print(f"    ⏱️  Original: {orig_time:.4f}s, {orig_memory:.2f}MB")
            print(f"    ⚡ Optimized: {opt_time:.4f}s, {opt_memory:.2f}MB")
            print(f"    📈 Time improvement: {time_improvement:.1f}%")
            print(f"    💾 Memory improvement: {memory_improvement:.1f}%")
        
        return self.results
    
    def generate_performance_report(self) -> str:
        """Generate a comprehensive performance report."""
        report = []
        report.append("# SEO Evaluation System Performance Report")
        report.append("=" * 50)
        report.append("")
        
        # Summary statistics
        report.append("## Performance Summary")
        report.append("")
        
        for size in self.dataset_sizes:
            if size in self.results['improvements']:
                improvements = self.results['improvements'][size]
                report.append(f"### Dataset Size: {size:,}")
                report.append(f"- **Time Improvement**: {improvements['time_improvement']:.1f}%")
                report.append(f"- **Memory Improvement**: {improvements['memory_improvement']:.1f}%")
                report.append("")
        
        # Detailed results table
        report.append("## Detailed Results")
        report.append("")
        report.append("| Dataset Size | Original Time (s) | Optimized Time (s) | Time Improvement | Original Memory (MB) | Optimized Memory (MB) | Memory Improvement |")
        report.append("|--------------|-------------------|-------------------|------------------|---------------------|----------------------|-------------------|")
        
        for size in self.dataset_sizes:
            if size in self.results['original'] and size in self.results['optimized']:
                orig = self.results['original'][size]
                opt = self.results['optimized'][size]
                improvements = self.results['improvements'][size]
                
                report.append(f"| {size:,} | {orig['execution_time']:.4f} | {opt['execution_time']:.4f} | {improvements['time_improvement']:.1f}% | {orig['memory_usage']:.2f} | {opt['memory_usage']:.2f} | {improvements['memory_improvement']:.1f}% |")
        
        report.append("")
        
        # Recommendations
        report.append("## Optimization Benefits")
        report.append("")
        report.append("### Performance Improvements")
        report.append("- **Vectorized Operations**: NumPy vectorization for faster computation")
        report.append("- **Caching**: LRU cache for expensive calculations")
        report.append("- **Async Processing**: Concurrent execution of independent metrics")
        report.append("- **Batch Processing**: Efficient memory management for large datasets")
        report.append("")
        
        report.append("### Memory Optimizations")
        report.append("- **Data Type Optimization**: Automatic conversion to efficient data types")
        report.append("- **Batch Processing**: Process data in chunks to reduce memory footprint")
        report.append("- **Resource Management**: Proper cleanup of resources and executors")
        report.append("")
        
        report.append("### Scalability Features")
        report.append("- **Multi-threading**: Configurable worker threads for parallel processing")
        report.append("- **Adaptive Batch Sizes**: Automatic batch size optimization")
        report.append("- **Memory Monitoring**: Built-in memory usage tracking")
        
        return "\n".join(report)
    
    def plot_performance_comparison(self, save_path: str = "performance_comparison.png"):
        """Generate performance comparison plots."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('SEO Evaluation System Performance Comparison', fontsize=16, fontweight='bold')
        
        sizes = list(self.results['original'].keys())
        
        # Execution time comparison
        orig_times = [self.results['original'][size]['execution_time'] for size in sizes]
        opt_times = [self.results['optimized'][size]['execution_time'] for size in sizes]
        
        ax1.plot(sizes, orig_times, 'o-', label='Original', linewidth=2, markersize=8)
        ax1.plot(sizes, opt_times, 's-', label='Optimized', linewidth=2, markersize=8)
        ax1.set_xlabel('Dataset Size')
        ax1.set_ylabel('Execution Time (seconds)')
        ax1.set_title('Execution Time Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')
        
        # Memory usage comparison
        orig_memory = [self.results['original'][size]['memory_usage'] for size in sizes]
        opt_memory = [self.results['optimized'][size]['memory_usage'] for size in sizes]
        
        ax2.plot(sizes, orig_memory, 'o-', label='Original', linewidth=2, markersize=8)
        ax2.plot(sizes, opt_memory, 's-', label='Optimized', linewidth=2, markersize=8)
        ax2.set_xlabel('Dataset Size')
        ax2.set_ylabel('Memory Usage (MB)')
        ax2.set_title('Memory Usage Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log')
        
        # Time improvement percentage
        time_improvements = [self.results['improvements'][size]['time_improvement'] for size in sizes]
        
        ax3.bar(range(len(sizes)), time_improvements, color='lightgreen', alpha=0.7)
        ax3.set_xlabel('Dataset Size Index')
        ax3.set_ylabel('Time Improvement (%)')
        ax3.set_title('Time Improvement Percentage')
        ax3.set_xticks(range(len(sizes)))
        ax3.set_xticklabels([f'{size:,}' for size in sizes], rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Memory improvement percentage
        memory_improvements = [self.results['improvements'][size]['memory_improvement'] for size in sizes]
        
        ax4.bar(range(len(sizes)), memory_improvements, color='lightblue', alpha=0.7)
        ax4.set_xlabel('Dataset Size Index')
        ax4.set_ylabel('Memory Improvement (%)')
        ax4.set_title('Memory Improvement Percentage')
        ax4.set_xticks(range(len(sizes)))
        ax4.set_xticklabels([f'{size:,}' for size in sizes], rotation=45)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Performance comparison plots saved to: {save_path}")

async def main():
    """Main function to run the performance benchmark."""
    print("🚀 SEO Evaluation System Performance Benchmark")
    print("=" * 60)
    
    # Create benchmark instance
    benchmark = PerformanceBenchmark()
    
    # Run benchmarks for different task types
    task_types = ["classification", "ranking", "regression"]
    
    for task_type in task_types:
        print(f"\n🎯 Benchmarking {task_type.upper()} task")
        print("-" * 40)
        
        try:
            results = benchmark.run_benchmarks(task_type)
            
            # Generate and save report
            report = benchmark.generate_performance_report()
            report_filename = f"performance_report_{task_type}.md"
            
            with open(report_filename, 'w') as f:
                f.write(report)
            
            print(f"\n📄 Performance report saved to: {report_filename}")
            
        except Exception as e:
            print(f"❌ Error benchmarking {task_type}: {e}")
            continue
    
    # Generate overall performance comparison plots
    try:
        benchmark.plot_performance_comparison()
        print("\n📊 Performance comparison plots generated successfully!")
    except Exception as e:
        print(f"❌ Error generating plots: {e}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("🎉 Performance benchmark completed!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
