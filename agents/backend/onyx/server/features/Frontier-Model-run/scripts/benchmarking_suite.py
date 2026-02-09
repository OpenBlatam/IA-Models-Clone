#!/usr/bin/env python3
"""
Advanced Benchmarking Suite for Frontier Model Training
Provides comprehensive performance evaluation, comparison, and analysis capabilities.
"""

import os
import json
import time
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import seaborn as sns
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
import sqlite3
from contextlib import contextmanager
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import psutil
import GPUtil
import joblib
import pickle
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

console = Console()

class BenchmarkType(Enum):
    """Benchmark types."""
    PERFORMANCE = "performance"
    ACCURACY = "accuracy"
    SPEED = "speed"
    MEMORY = "memory"
    SCALABILITY = "scalability"
    ROBUSTNESS = "robustness"
    EFFICIENCY = "efficiency"
    COMPREHENSIVE = "comprehensive"

class MetricType(Enum):
    """Metric types."""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    AUC = "auc"
    MSE = "mse"
    MAE = "mae"
    RMSE = "rmse"
    R2_SCORE = "r2_score"
    INFERENCE_TIME = "inference_time"
    TRAINING_TIME = "training_time"
    MEMORY_USAGE = "memory_usage"
    GPU_UTILIZATION = "gpu_utilization"
    CPU_UTILIZATION = "cpu_utilization"
    THROUGHPUT = "throughput"
    LATENCY = "latency"

class DatasetType(Enum):
    """Dataset types."""
    SYNTHETIC = "synthetic"
    REAL_WORLD = "real_world"
    BENCHMARK = "benchmark"
    CUSTOM = "custom"

class HardwareConfig(Enum):
    """Hardware configurations."""
    CPU_ONLY = "cpu_only"
    SINGLE_GPU = "single_gpu"
    MULTI_GPU = "multi_gpu"
    DISTRIBUTED = "distributed"
    EDGE_DEVICE = "edge_device"
    CLOUD = "cloud"

@dataclass
class BenchmarkConfig:
    """Benchmark configuration."""
    benchmark_type: BenchmarkType = BenchmarkType.COMPREHENSIVE
    metric_types: List[MetricType] = None
    dataset_type: DatasetType = DatasetType.SYNTHETIC
    hardware_config: HardwareConfig = HardwareConfig.SINGLE_GPU
    num_iterations: int = 10
    warmup_iterations: int = 3
    batch_sizes: List[int] = None
    input_sizes: List[Tuple[int, ...]] = None
    enable_profiling: bool = True
    enable_memory_tracking: bool = True
    enable_gpu_monitoring: bool = True
    enable_cpu_monitoring: bool = True
    enable_distributed_benchmarking: bool = False
    enable_comparison_mode: bool = True
    enable_statistical_analysis: bool = True
    enable_visualization: bool = True
    device: str = "auto"

@dataclass
class BenchmarkResult:
    """Benchmark result."""
    result_id: str
    benchmark_type: BenchmarkType
    model_name: str
    metrics: Dict[str, List[float]]
    statistics: Dict[str, Dict[str, float]]
    hardware_info: Dict[str, Any]
    execution_time: float
    created_at: datetime = None

class DataGenerator:
    """Benchmark data generator."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def generate_synthetic_data(self, input_shape: Tuple[int, ...], 
                              num_samples: int = 1000, 
                              num_classes: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate synthetic benchmark data."""
        if len(input_shape) == 1:
            # 1D data (e.g., tabular)
            X = torch.randn(num_samples, input_shape[0])
        elif len(input_shape) == 2:
            # 2D data (e.g., images)
            X = torch.randn(num_samples, input_shape[0], input_shape[1])
        elif len(input_shape) == 3:
            # 3D data (e.g., RGB images)
            X = torch.randn(num_samples, input_shape[0], input_shape[1], input_shape[2])
        else:
            # Higher dimensional data
            X = torch.randn(num_samples, *input_shape)
        
        # Generate labels
        y = torch.randint(0, num_classes, (num_samples,))
        
        return X, y
    
    def generate_benchmark_datasets(self) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """Generate various benchmark datasets."""
        datasets = {}
        
        # Image classification datasets
        datasets['cifar10_like'] = self.generate_synthetic_data((3, 32, 32), 1000, 10)
        datasets['imagenet_like'] = self.generate_synthetic_data((3, 224, 224), 1000, 1000)
        
        # Tabular datasets
        datasets['tabular_small'] = self.generate_synthetic_data((10,), 1000, 2)
        datasets['tabular_large'] = self.generate_synthetic_data((100,), 1000, 5)
        
        # Time series datasets
        datasets['time_series'] = self.generate_synthetic_data((100, 10), 1000, 3)
        
        # Text datasets (simplified)
        datasets['text_embeddings'] = self.generate_synthetic_data((512,), 1000, 20)
        
        return datasets

class PerformanceProfiler:
    """Performance profiling engine."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize monitoring
        self.cpu_monitor = CPUMonitor() if config.enable_cpu_monitoring else None
        self.gpu_monitor = GPUMonitor() if config.enable_gpu_monitoring else None
        self.memory_monitor = MemoryMonitor() if config.enable_memory_tracking else None
    
    def profile_model(self, model: nn.Module, data_loader: DataLoader, 
                     device: torch.device) -> Dict[str, Any]:
        """Profile model performance."""
        console.print("[blue]Profiling model performance...[/blue]")
        
        model = model.to(device)
        model.eval()
        
        # Warmup
        self._warmup_model(model, data_loader, device)
        
        # Profile inference
        inference_metrics = self._profile_inference(model, data_loader, device)
        
        # Profile training
        training_metrics = self._profile_training(model, data_loader, device)
        
        # System metrics
        system_metrics = self._get_system_metrics()
        
        return {
            'inference': inference_metrics,
            'training': training_metrics,
            'system': system_metrics
        }
    
    def _warmup_model(self, model: nn.Module, data_loader: DataLoader, device: torch.device):
        """Warmup model for accurate profiling."""
        model.eval()
        with torch.no_grad():
            for i, (data, _) in enumerate(data_loader):
                if i >= self.config.warmup_iterations:
                    break
                data = data.to(device)
                _ = model(data)
    
    def _profile_inference(self, model: nn.Module, data_loader: DataLoader, 
                          device: torch.device) -> Dict[str, List[float]]:
        """Profile inference performance."""
        model.eval()
        inference_times = []
        memory_usage = []
        cpu_usage = []
        gpu_usage = []
        
        with torch.no_grad():
            for i, (data, _) in enumerate(data_loader):
                if i >= self.config.num_iterations:
                    break
                
                # Start monitoring
                if self.memory_monitor:
                    self.memory_monitor.start()
                if self.cpu_monitor:
                    self.cpu_monitor.start()
                if self.gpu_monitor:
                    self.gpu_monitor.start()
                
                # Measure inference time
                start_time = time.time()
                data = data.to(device)
                _ = model(data)
                
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                
                end_time = time.time()
                inference_times.append((end_time - start_time) * 1000)  # Convert to ms
                
                # Stop monitoring
                if self.memory_monitor:
                    memory_usage.append(self.memory_monitor.stop())
                if self.cpu_monitor:
                    cpu_usage.append(self.cpu_monitor.stop())
                if self.gpu_monitor:
                    gpu_usage.append(self.gpu_monitor.stop())
        
        return {
            'inference_time_ms': inference_times,
            'memory_usage_mb': memory_usage,
            'cpu_usage_percent': cpu_usage,
            'gpu_usage_percent': gpu_usage
        }
    
    def _profile_training(self, model: nn.Module, data_loader: DataLoader, 
                         device: torch.device) -> Dict[str, List[float]]:
        """Profile training performance."""
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        training_times = []
        memory_usage = []
        
        for i, (data, target) in enumerate(data_loader):
            if i >= self.config.num_iterations:
                break
            
            # Start monitoring
            if self.memory_monitor:
                self.memory_monitor.start()
            
            # Measure training time
            start_time = time.time()
            
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.time()
            training_times.append((end_time - start_time) * 1000)  # Convert to ms
            
            # Stop monitoring
            if self.memory_monitor:
                memory_usage.append(self.memory_monitor.stop())
        
        return {
            'training_time_ms': training_times,
            'memory_usage_mb': memory_usage
        }
    
    def _get_system_metrics(self) -> Dict[str, Any]:
        """Get system metrics."""
        system_info = {
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'memory_available_gb': psutil.virtual_memory().available / (1024**3)
        }
        
        # GPU info
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_info = []
                for gpu in gpus:
                    gpu_info.append({
                        'name': gpu.name,
                        'memory_total_mb': gpu.memoryTotal,
                        'memory_free_mb': gpu.memoryFree,
                        'utilization_percent': gpu.load * 100
                    })
                system_info['gpus'] = gpu_info
        except:
            system_info['gpus'] = []
        
        return system_info

class CPUMonitor:
    """CPU monitoring."""
    
    def __init__(self):
        self.start_cpu_percent = None
    
    def start(self):
        """Start CPU monitoring."""
        self.start_cpu_percent = psutil.cpu_percent()
    
    def stop(self) -> float:
        """Stop CPU monitoring and return average usage."""
        if self.start_cpu_percent is not None:
            return psutil.cpu_percent()
        return 0.0

class GPUMonitor:
    """GPU monitoring."""
    
    def __init__(self):
        self.start_gpu_percent = None
    
    def start(self):
        """Start GPU monitoring."""
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                self.start_gpu_percent = gpus[0].load * 100
        except:
            self.start_gpu_percent = 0.0
    
    def stop(self) -> float:
        """Stop GPU monitoring and return usage."""
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                return gpus[0].load * 100
        except:
            pass
        return 0.0

class MemoryMonitor:
    """Memory monitoring."""
    
    def __init__(self):
        self.start_memory = None
    
    def start(self):
        """Start memory monitoring."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.start_memory = torch.cuda.memory_allocated() / (1024**2)  # MB
        else:
            process = psutil.Process()
            self.start_memory = process.memory_info().rss / (1024**2)  # MB
    
    def stop(self) -> float:
        """Stop memory monitoring and return peak usage."""
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / (1024**2)  # MB
            torch.cuda.reset_peak_memory_stats()
            return peak_memory
        else:
            process = psutil.Process()
            current_memory = process.memory_info().rss / (1024**2)  # MB
            return current_memory - (self.start_memory or 0)

class MetricCalculator:
    """Metric calculation engine."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def calculate_metrics(self, model: nn.Module, data_loader: DataLoader, 
                        device: torch.device) -> Dict[str, float]:
        """Calculate performance metrics."""
        console.print("[blue]Calculating performance metrics...[/blue]")
        
        model = model.to(device)
        model.eval()
        
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                
                # Get predictions
                _, predicted = torch.max(output, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                
                # Get probabilities
                probabilities = F.softmax(output, dim=1)
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate metrics
        metrics = {}
        
        if MetricType.ACCURACY in self.config.metric_types:
            metrics['accuracy'] = accuracy_score(all_targets, all_predictions)
        
        if MetricType.PRECISION in self.config.metric_types:
            metrics['precision'] = precision_score(all_targets, all_predictions, average='weighted')
        
        if MetricType.RECALL in self.config.metric_types:
            metrics['recall'] = recall_score(all_targets, all_predictions, average='weighted')
        
        if MetricType.F1_SCORE in self.config.metric_types:
            metrics['f1_score'] = f1_score(all_targets, all_predictions, average='weighted')
        
        if MetricType.AUC in self.config.metric_types:
            try:
                # For multi-class, use one-vs-rest
                metrics['auc'] = roc_auc_score(all_targets, all_probabilities, multi_class='ovr')
            except:
                metrics['auc'] = 0.0
        
        return metrics
    
    def calculate_statistics(self, values: List[float]) -> Dict[str, float]:
        """Calculate statistical measures."""
        if not values:
            return {}
        
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values),
            'q25': np.percentile(values, 25),
            'q75': np.percentile(values, 75),
            'cv': np.std(values) / np.mean(values) if np.mean(values) != 0 else 0
        }

class BenchmarkRunner:
    """Main benchmark runner."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.data_generator = DataGenerator(config)
        self.profiler = PerformanceProfiler(config)
        self.metric_calculator = MetricCalculator(config)
        
        # Initialize database
        self.db_path = self._init_database()
        
        # Results storage
        self.benchmark_results: Dict[str, BenchmarkResult] = {}
    
    def _init_database(self) -> str:
        """Initialize benchmark database."""
        db_path = Path("./benchmarking_suite.db")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS benchmark_results (
                    result_id TEXT PRIMARY KEY,
                    benchmark_type TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    metrics TEXT NOT NULL,
                    statistics TEXT NOT NULL,
                    hardware_info TEXT NOT NULL,
                    execution_time REAL NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
        
        return str(db_path)
    
    def run_benchmark(self, model: nn.Module, model_name: str = None) -> BenchmarkResult:
        """Run comprehensive benchmark."""
        console.print(f"[blue]Running {self.config.benchmark_type.value} benchmark...[/blue]")
        
        start_time = time.time()
        result_id = f"bench_{int(time.time())}"
        model_name = model_name or f"model_{result_id}"
        
        # Initialize device
        if self.config.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(self.config.device)
        
        # Generate datasets
        datasets = self.data_generator.generate_benchmark_datasets()
        
        # Run benchmarks on different datasets
        all_metrics = defaultdict(list)
        all_statistics = {}
        
        for dataset_name, (X, y) in datasets.items():
            console.print(f"[blue]Benchmarking on {dataset_name}...[/blue]")
            
            # Create data loader
            dataset = torch.utils.data.TensorDataset(X, y)
            data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
            
            # Profile performance
            profile_results = self.profiler.profile_model(model, data_loader, device)
            
            # Calculate metrics
            metrics = self.metric_calculator.calculate_metrics(model, data_loader, device)
            
            # Combine results
            for metric_name, values in profile_results['inference'].items():
                all_metrics[metric_name].extend(values)
            
            for metric_name, value in metrics.items():
                all_metrics[metric_name].append(value)
        
        # Calculate statistics
        for metric_name, values in all_metrics.items():
            all_statistics[metric_name] = self.metric_calculator.calculate_statistics(values)
        
        # Get hardware info
        hardware_info = self.profiler._get_system_metrics()
        
        execution_time = time.time() - start_time
        
        # Create benchmark result
        benchmark_result = BenchmarkResult(
            result_id=result_id,
            benchmark_type=self.config.benchmark_type,
            model_name=model_name,
            metrics=dict(all_metrics),
            statistics=all_statistics,
            hardware_info=hardware_info,
            execution_time=execution_time,
            created_at=datetime.now()
        )
        
        # Store result
        self.benchmark_results[result_id] = benchmark_result
        
        # Save to database
        self._save_benchmark_result(benchmark_result)
        
        console.print(f"[green]Benchmark completed in {execution_time:.2f} seconds[/green]")
        
        return benchmark_result
    
    def compare_models(self, models: Dict[str, nn.Module]) -> Dict[str, Any]:
        """Compare multiple models."""
        console.print("[blue]Comparing multiple models...[/blue]")
        
        comparison_results = {}
        
        for model_name, model in models.items():
            result = self.run_benchmark(model, model_name)
            comparison_results[model_name] = result
        
        # Generate comparison report
        comparison_report = self._generate_comparison_report(comparison_results)
        
        return {
            'results': comparison_results,
            'report': comparison_report
        }
    
    def _generate_comparison_report(self, results: Dict[str, BenchmarkResult]) -> Dict[str, Any]:
        """Generate comparison report."""
        report = {
            'summary': {},
            'rankings': {},
            'recommendations': []
        }
        
        # Summary statistics
        for model_name, result in results.items():
            report['summary'][model_name] = {
                'avg_inference_time': result.statistics.get('inference_time_ms', {}).get('mean', 0),
                'avg_accuracy': result.statistics.get('accuracy', {}).get('mean', 0),
                'avg_memory_usage': result.statistics.get('memory_usage_mb', {}).get('mean', 0)
            }
        
        # Rankings
        inference_times = {name: result.statistics.get('inference_time_ms', {}).get('mean', 0) 
                          for name, result in results.items()}
        accuracies = {name: result.statistics.get('accuracy', {}).get('mean', 0) 
                    for name, result in results.items()}
        
        report['rankings']['fastest'] = min(inference_times, key=inference_times.get)
        report['rankings']['most_accurate'] = max(accuracies, key=accuracies.get)
        
        # Recommendations
        best_overall = max(results.keys(), key=lambda x: 
                          results[x].statistics.get('accuracy', {}).get('mean', 0) * 0.7 +
                          (1 / (results[x].statistics.get('inference_time_ms', {}).get('mean', 1))) * 0.3)
        
        report['recommendations'].append(f"Best overall model: {best_overall}")
        
        return report
    
    def _save_benchmark_result(self, result: BenchmarkResult):
        """Save benchmark result to database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO benchmark_results 
                (result_id, benchmark_type, model_name, metrics,
                 statistics, hardware_info, execution_time, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result.result_id,
                result.benchmark_type.value,
                result.model_name,
                json.dumps(result.metrics),
                json.dumps(result.statistics),
                json.dumps(result.hardware_info),
                result.execution_time,
                result.created_at.isoformat()
            ))
    
    def visualize_benchmark_results(self, result: BenchmarkResult, 
                                   output_path: str = None) -> str:
        """Visualize benchmark results."""
        if output_path is None:
            output_path = f"benchmark_analysis_{result.result_id}.png"
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Performance metrics
        performance_metrics = ['inference_time_ms', 'memory_usage_mb', 'cpu_usage_percent', 'gpu_usage_percent']
        metric_data = {}
        
        for metric in performance_metrics:
            if metric in result.metrics:
                metric_data[metric] = result.statistics[metric]['mean']
        
        if metric_data:
            axes[0, 0].bar(metric_data.keys(), metric_data.values())
            axes[0, 0].set_title('Performance Metrics')
            axes[0, 0].set_ylabel('Value')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Accuracy metrics
        accuracy_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        accuracy_data = {}
        
        for metric in accuracy_metrics:
            if metric in result.metrics:
                accuracy_data[metric] = result.statistics[metric]['mean']
        
        if accuracy_data:
            axes[0, 1].bar(accuracy_data.keys(), accuracy_data.values())
            axes[0, 1].set_title('Accuracy Metrics')
            axes[0, 1].set_ylabel('Value')
            axes[0, 1].set_ylim(0, 1)
        
        # Hardware information
        hardware_info = result.hardware_info
        hw_data = {
            'CPU Cores': hardware_info.get('cpu_count', 0),
            'Memory (GB)': hardware_info.get('memory_total_gb', 0),
            'GPUs': len(hardware_info.get('gpus', []))
        }
        
        axes[1, 0].bar(hw_data.keys(), hw_data.values())
        axes[1, 0].set_title('Hardware Information')
        axes[1, 0].set_ylabel('Value')
        
        # Benchmark statistics
        stats_data = {
            'Execution Time (s)': result.execution_time,
            'Model Name': len(result.model_name),
            'Benchmark Type': len(result.benchmark_type.value),
            'Total Metrics': len(result.metrics)
        }
        
        axes[1, 1].bar(stats_data.keys(), stats_data.values())
        axes[1, 1].set_title('Benchmark Statistics')
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        console.print(f"[green]Benchmark visualization saved: {output_path}[/green]")
        return output_path
    
    def get_benchmark_summary(self) -> Dict[str, Any]:
        """Get benchmark system summary."""
        if not self.benchmark_results:
            return {'total_benchmarks': 0}
        
        total_benchmarks = len(self.benchmark_results)
        
        # Calculate average metrics
        avg_inference_time = np.mean([result.statistics.get('inference_time_ms', {}).get('mean', 0) 
                                    for result in self.benchmark_results.values()])
        avg_accuracy = np.mean([result.statistics.get('accuracy', {}).get('mean', 0) 
                              for result in self.benchmark_results.values()])
        avg_memory = np.mean([result.statistics.get('memory_usage_mb', {}).get('mean', 0) 
                            for result in self.benchmark_results.values()])
        
        # Best performing benchmark
        best_result = max(self.benchmark_results.values(), 
                         key=lambda x: x.statistics.get('accuracy', {}).get('mean', 0))
        
        return {
            'total_benchmarks': total_benchmarks,
            'average_inference_time_ms': avg_inference_time,
            'average_accuracy': avg_accuracy,
            'average_memory_usage_mb': avg_memory,
            'best_accuracy': best_result.statistics.get('accuracy', {}).get('mean', 0),
            'best_benchmark_id': best_result.result_id,
            'benchmark_types': list(set(result.benchmark_type.value for result in self.benchmark_results.values())),
            'models_benchmarked': list(set(result.model_name for result in self.benchmark_results.values()))
        }

def main():
    """Main function for Benchmarking Suite CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Advanced Benchmarking Suite")
    parser.add_argument("--benchmark-type", type=str,
                       choices=["performance", "accuracy", "speed", "memory", "comprehensive"],
                       default="comprehensive", help="Benchmark type")
    parser.add_argument("--metric-types", type=str, nargs='+',
                       choices=["accuracy", "precision", "recall", "f1_score", "auc"],
                       default=["accuracy", "precision", "recall", "f1_score"], help="Metric types")
    parser.add_argument("--dataset-type", type=str,
                       choices=["synthetic", "real_world", "benchmark"],
                       default="synthetic", help="Dataset type")
    parser.add_argument("--hardware-config", type=str,
                       choices=["cpu_only", "single_gpu", "multi_gpu"],
                       default="single_gpu", help="Hardware configuration")
    parser.add_argument("--num-iterations", type=int, default=10,
                       help="Number of iterations")
    parser.add_argument("--warmup-iterations", type=int, default=3,
                       help="Warmup iterations")
    parser.add_argument("--enable-profiling", action="store_true", default=True,
                       help="Enable profiling")
    parser.add_argument("--enable-memory-tracking", action="store_true", default=True,
                       help="Enable memory tracking")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use")
    
    args = parser.parse_args()
    
    # Create benchmark configuration
    config = BenchmarkConfig(
        benchmark_type=BenchmarkType(args.benchmark_type),
        metric_types=[MetricType(mt) for mt in args.metric_types],
        dataset_type=DatasetType(args.dataset_type),
        hardware_config=HardwareConfig(args.hardware_config),
        num_iterations=args.num_iterations,
        warmup_iterations=args.warmup_iterations,
        enable_profiling=args.enable_profiling,
        enable_memory_tracking=args.enable_memory_tracking,
        device=args.device
    )
    
    # Create benchmark runner
    benchmark_runner = BenchmarkRunner(config)
    
    # Create sample models
    class SampleModel1(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.fc1 = nn.Linear(64 * 8 * 8, 128)
            self.fc2 = nn.Linear(128, 10)
        
        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2)
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    class SampleModel2(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
            self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
            self.fc1 = nn.Linear(128 * 8 * 8, 256)
            self.fc2 = nn.Linear(256, 10)
        
        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2)
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    # Run single model benchmark
    model1 = SampleModel1()
    result1 = benchmark_runner.run_benchmark(model1, "SampleModel1")
    
    # Run model comparison
    models = {
        "Model1": SampleModel1(),
        "Model2": SampleModel2()
    }
    comparison = benchmark_runner.compare_models(models)
    
    # Show results
    console.print(f"[green]Benchmark completed[/green]")
    console.print(f"[blue]Benchmark type: {args.benchmark_type}[/blue]")
    console.print(f"[blue]Model: {result1.model_name}[/blue]")
    console.print(f"[blue]Execution time: {result1.execution_time:.2f} seconds[/blue]")
    
    # Show comparison results
    console.print(f"[blue]Comparison results:[/blue]")
    for model_name, result in comparison['results'].items():
        accuracy = result.statistics.get('accuracy', {}).get('mean', 0)
        inference_time = result.statistics.get('inference_time_ms', {}).get('mean', 0)
        console.print(f"[blue]{model_name}: Accuracy={accuracy:.4f}, Time={inference_time:.2f}ms[/blue]")
    
    # Create visualization
    benchmark_runner.visualize_benchmark_results(result1)
    
    # Show summary
    summary = benchmark_runner.get_benchmark_summary()
    console.print(f"[blue]Summary: {summary}[/blue]")

if __name__ == "__main__":
    main()