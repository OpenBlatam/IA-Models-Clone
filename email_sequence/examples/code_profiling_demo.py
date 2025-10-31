from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import asyncio
import psutil
from core.code_profiler import (
from core.training_logger import create_training_logger
from core.optimized_training_optimizer import create_optimized_training_optimizer
from core.performance_optimization import PerformanceConfig
from core.multi_gpu_training import MultiGPUConfig
from core.gradient_accumulation import GradientAccumulationConfig
        import traceback
from typing import Any, List, Dict, Optional
import logging
"""
Code Profiling Demo

Comprehensive demonstration of code profiling for identifying and optimizing bottlenecks
in data loading, preprocessing, and training pipelines.
"""


    CodeProfiler, ProfilerConfig, create_code_profiler,
    ProfilerMetrics
)


class EmailSequenceModel(nn.Module):
    """Email sequence model for profiling demonstration"""
    
    def __init__(self, vocab_size=1000, embedding_dim=128, hidden_dim=256, num_classes=2) -> Any:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, x) -> Any:
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        # Use the last output
        last_output = lstm_out[:, -1, :]
        dropped = self.dropout(last_output)
        output = self.classifier(dropped)
        return output


def create_dummy_dataset(num_samples=1000, seq_length=50, vocab_size=1000) -> Any:
    """Create dummy dataset for demonstration"""
    
    # Generate random sequences
    sequences = torch.randint(0, vocab_size, (num_samples, seq_length))
    
    # Generate random labels (0 or 1)
    labels = torch.randint(0, 2, (num_samples,))
    
    # Create dataset
    dataset = TensorDataset(sequences, labels)
    
    return dataset


def demonstrate_basic_profiling():
    """Demonstrate basic profiling functionality"""
    
    print("="*80)
    print("BASIC PROFILING DEMONSTRATION")
    print("="*80)
    
    # Create profiler
    profiler = create_code_profiler(
        enable_profiling=True,
        profile_level="detailed",
        save_profiles=True,
        profile_dir="profiles/basic_demo"
    )
    
    # Profile simple operations
    with profiler.profile_section("data_creation", "data_loading"):
        data = torch.randn(1000, 10)
        labels = torch.randint(0, 2, (1000,))
        dataset = TensorDataset(data, labels)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Profile model creation
    with profiler.profile_section("model_creation", "model_setup"):
        model = EmailSequenceModel()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Profile training steps
    for i in range(5):
        with profiler.profile_section(f"training_step_{i}", "training"):
            batch = next(iter(dataloader))
            inputs, targets = batch
            
            # Forward pass
            outputs = model(inputs)
            loss = nn.CrossEntropyLoss()(outputs, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    
    # Generate report
    report = profiler.generate_profiling_report()
    
    # Cleanup
    profiler.cleanup()
    
    return report


def demonstrate_data_loading_profiling():
    """Demonstrate data loading profiling"""
    
    print("\n" + "="*80)
    print("DATA LOADING PROFILING DEMONSTRATION")
    print("="*80)
    
    # Create profiler
    profiler = create_code_profiler(
        enable_profiling=True,
        profile_level="detailed",
        save_profiles=True,
        profile_dir="profiles/data_loading_demo"
    )
    
    # Create dataset
    dataset = create_dummy_dataset(num_samples=2000, seq_length=50, vocab_size=1000)
    
    # Test different DataLoader configurations
    configurations = [
        {"num_workers": 0, "pin_memory": False, "persistent_workers": False},
        {"num_workers": 2, "pin_memory": True, "persistent_workers": False},
        {"num_workers": 4, "pin_memory": True, "persistent_workers": True},
    ]
    
    results = {}
    
    for i, config in enumerate(configurations):
        print(f"\nTesting configuration {i+1}: {config}")
        
        # Profile DataLoader creation
        with profiler.profile_section(f"dataloader_creation_{i}", "data_loading"):
            dataloader = DataLoader(
                dataset,
                batch_size=32,
                shuffle=True,
                **config
            )
        
        # Profile data loading
        loading_metrics = profiler.profile_data_loading(dataloader, num_batches=20)
        results[f"config_{i}"] = {
            "config": config,
            "metrics": loading_metrics
        }
    
    # Generate report
    report = profiler.generate_profiling_report()
    
    # Print comparison
    print(f"\n{'='*60}")
    print("DATA LOADING COMPARISON")
    print(f"{'='*60}")
    
    for config_name, result in results.items():
        metrics = result["metrics"]
        config = result["config"]
        print(f"\nConfiguration: {config}")
        print(f"  Average batch time: {metrics.get('avg_batch_time', 0):.4f}s")
        print(f"  Throughput: {metrics.get('throughput', 0):.1f} samples/second")
        print(f"  Average memory per batch: {metrics.get('avg_memory_per_batch', 0):.3f}MB")
    
    # Cleanup
    profiler.cleanup()
    
    return report


def demonstrate_preprocessing_profiling():
    """Demonstrate preprocessing profiling"""
    
    print("\n" + "="*80)
    print("PREPROCESSING PROFILING DEMONSTRATION")
    print("="*80)
    
    # Create profiler
    profiler = create_code_profiler(
        enable_profiling=True,
        profile_level="detailed",
        save_profiles=True,
        profile_dir="profiles/preprocessing_demo"
    )
    
    # Sample data
    sample_data = torch.randint(0, 1000, (50,))
    
    # Define different preprocessing functions
    def simple_preprocessing(data) -> Any:
        """Simple preprocessing"""
        return data.float() / 1000.0
    
    def complex_preprocessing(data) -> Any:
        """Complex preprocessing with multiple operations"""
        # Simulate complex preprocessing
        processed = data.float()
        processed = processed / 1000.0
        processed = torch.clamp(processed, 0, 1)
        processed = torch.where(processed > 0.5, processed * 2, processed / 2)
        return processed
    
    def memory_intensive_preprocessing(data) -> Any:
        """Memory intensive preprocessing"""
        # Simulate memory intensive operations
        processed = data.float()
        for _ in range(10):
            processed = torch.cat([processed, processed.clone()], dim=0)
            processed = processed[:len(data)]  # Keep original size
        return processed / 1000.0
    
    # Profile different preprocessing functions
    preprocessing_functions = [
        ("simple", simple_preprocessing),
        ("complex", complex_preprocessing),
        ("memory_intensive", memory_intensive_preprocessing)
    ]
    
    results = {}
    
    for name, func in preprocessing_functions:
        print(f"\nProfiling {name} preprocessing...")
        
        # Profile preprocessing
        metrics = profiler.profile_preprocessing(func, sample_data, num_samples=100)
        results[name] = metrics
    
    # Print comparison
    print(f"\n{'='*60}")
    print("PREPROCESSING COMPARISON")
    print(f"{'='*60}")
    
    for name, metrics in results.items():
        print(f"\n{name.title()} preprocessing:")
        print(f"  Average processing time: {metrics.get('avg_processing_time', 0):.6f}s")
        print(f"  Throughput: {metrics.get('throughput', 0):.1f} samples/second")
        print(f"  Average memory usage: {metrics.get('avg_memory_usage', 0):.3f}MB")
    
    # Generate report
    report = profiler.generate_profiling_report()
    
    # Cleanup
    profiler.cleanup()
    
    return report


def demonstrate_model_training_profiling():
    """Demonstrate model training profiling"""
    
    print("\n" + "="*80)
    print("MODEL TRAINING PROFILING DEMONSTRATION")
    print("="*80)
    
    # Create profiler
    profiler = create_code_profiler(
        enable_profiling=True,
        profile_level="detailed",
        save_profiles=True,
        profile_dir="profiles/training_demo"
    )
    
    # Create dataset and model
    dataset = create_dummy_dataset(num_samples=1000, seq_length=50, vocab_size=1000)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)
    model = EmailSequenceModel()
    
    # Profile model training
    training_metrics = profiler.profile_model_training(model, dataloader, num_batches=10)
    
    # Generate report
    report = profiler.generate_profiling_report()
    
    # Cleanup
    profiler.cleanup()
    
    return report


def demonstrate_integrated_profiling():
    """Demonstrate integrated profiling with the optimized training optimizer"""
    
    print("\n" + "="*80)
    print("INTEGRATED PROFILING DEMONSTRATION")
    print("="*80)
    
    # Create dataset
    dataset = create_dummy_dataset(num_samples=1000, seq_length=50, vocab_size=1000)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)
    
    # Create model
    model = EmailSequenceModel()
    
    # Create configurations
    performance_config = PerformanceConfig(
        enable_mixed_precision=True,
        enable_memory_optimization=True,
        enable_computational_optimization=True
    )
    
    multi_gpu_config = MultiGPUConfig(
        enable_multi_gpu=False,
        strategy="data_parallel"
    )
    
    gradient_accumulation_config = GradientAccumulationConfig(
        accumulation_steps=2,
        memory_efficient=True
    )
    
    # Create optimized training optimizer with profiling
    optimizer = create_optimized_training_optimizer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        experiment_name="integrated_profiling_demo",
        debug_mode=False,
        enable_pytorch_debugging=False,
        performance_config=performance_config,
        multi_gpu_config=multi_gpu_config,
        gradient_accumulation_config=gradient_accumulation_config,
        # Profiling configuration
        enable_profiling=True,
        profile_level="detailed",
        save_profiles=True,
        enable_performance_monitoring=True,
        track_memory=True,
        track_cpu=True,
        track_gpu=True,
        profile_data_loading=True,
        profile_preprocessing=True,
        profile_forward_pass=True,
        profile_backward_pass=True,
        profile_optimizer_step=True,
        enable_memory_profiling=True,
        enable_gpu_profiling=True,
        generate_reports=True,
        create_visualizations=True
    )
    
    print("Optimized training optimizer with profiling created successfully!")
    print(f"Code profiler enabled: {optimizer.code_profiler is not None}")
    
    # Get training summary
    summary = optimizer.get_training_summary()
    print(f"\nTraining configuration: {json.dumps(summary, indent=2)}")
    
    # Cleanup
    optimizer.cleanup()
    
    print("Integrated profiling demonstration completed!")


def demonstrate_bottleneck_analysis():
    """Demonstrate bottleneck analysis and optimization recommendations"""
    
    print("\n" + "="*80)
    print("BOTTLENECK ANALYSIS DEMONSTRATION")
    print("="*80)
    
    # Create profiler
    profiler = create_code_profiler(
        enable_profiling=True,
        profile_level="comprehensive",
        save_profiles=True,
        profile_dir="profiles/bottleneck_analysis"
    )
    
    # Simulate different scenarios with bottlenecks
    
    # Scenario 1: Slow data loading
    print("\nScenario 1: Slow Data Loading")
    with profiler.profile_section("slow_data_loading", "data_loading"):
        # Simulate slow data loading
        time.sleep(0.1)  # Simulate 100ms delay
        data = torch.randn(1000, 10)
    
    # Scenario 2: Memory intensive operation
    print("\nScenario 2: Memory Intensive Operation")
    with profiler.profile_section("memory_intensive", "processing"):
        # Simulate memory intensive operation
        large_tensor = torch.randn(10000, 1000)
        result = torch.matmul(large_tensor, large_tensor.T)
        del large_tensor, result  # Clean up
    
    # Scenario 3: CPU intensive operation
    print("\nScenario 3: CPU Intensive Operation")
    with profiler.profile_section("cpu_intensive", "processing"):
        # Simulate CPU intensive operation
        for _ in range(1000000):
            _ = 1 + 1
    
    # Scenario 4: Frequent function calls
    print("\nScenario 4: Frequent Function Calls")
    for i in range(1000):
        with profiler.profile_section("frequent_calls", "function_calls"):
            _ = torch.randn(10, 10).sum()
    
    # Generate report with bottleneck analysis
    report = profiler.generate_profiling_report()
    
    # Print bottleneck analysis
    print(f"\n{'='*60}")
    print("BOTTLENECK ANALYSIS RESULTS")
    print(f"{'='*60}")
    
    if report.get("bottlenecks"):
        print(f"\nFound {len(report['bottlenecks'])} bottlenecks:")
        for i, bottleneck in enumerate(report["bottlenecks"], 1):
            print(f"\n{i}. {bottleneck['description']}")
            print(f"   Severity: {bottleneck['severity']}")
            print(f"   Recommendation: {bottleneck['recommendation']}")
    else:
        print("No significant bottlenecks detected.")
    
    if report.get("recommendations"):
        print(f"\nOptimization Recommendations:")
        for i, rec in enumerate(report["recommendations"], 1):
            print(f"{i}. {rec}")
    
    # Cleanup
    profiler.cleanup()
    
    return report


def create_profiling_visualizations(reports) -> Any:
    """Create comprehensive visualizations for profiling results"""
    
    print("\n" + "="*80)
    print("CREATING PROFILING VISUALIZATIONS")
    print("="*80)
    
    # Create output directory
    output_dir = Path("profiles/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Combine all reports for comparison
    all_metrics = {}
    
    for report_name, report in reports.items():
        if "detailed_metrics" in report:
            all_metrics[report_name] = report["detailed_metrics"]
    
    # Create execution time comparison
    if all_metrics:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Execution times
        ax1 = axes[0, 0]
        for report_name, metrics in all_metrics.items():
            if "execution_times" in metrics:
                for section, times in metrics["execution_times"].items():
                    if times:
                        ax1.hist(times, alpha=0.7, label=f"{report_name}_{section}", bins=20)
        ax1.set_title("Execution Time Distribution")
        ax1.set_xlabel("Time (seconds)")
        ax1.set_ylabel("Frequency")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Memory usage
        ax2 = axes[0, 1]
        for report_name, metrics in all_metrics.items():
            if "memory_usage" in metrics:
                for section, memory in metrics["memory_usage"].items():
                    if memory:
                        ax2.hist(memory, alpha=0.7, label=f"{report_name}_{section}", bins=20)
        ax2.set_title("Memory Usage Distribution")
        ax2.set_xlabel("Memory (MB)")
        ax2.set_ylabel("Frequency")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Call frequencies
        ax3 = axes[1, 0]
        call_frequencies = {}
        for report_name, metrics in all_metrics.items():
            if "call_frequencies" in metrics:
                call_frequencies.update(metrics["call_frequencies"])
        
        if call_frequencies:
            sections = list(call_frequencies.keys())
            frequencies = list(call_frequencies.values())
            bars = ax3.bar(range(len(sections)), frequencies)
            ax3.set_title("Function Call Frequencies")
            ax3.set_xlabel("Section")
            ax3.set_ylabel("Calls per Second")
            ax3.set_xticks(range(len(sections)))
            ax3.set_xticklabels(sections, rotation=45, ha='right')
            
            # Add value labels
            for bar, freq in zip(bars, frequencies):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{freq:.2f}', ha='center', va='bottom')
        
        # Performance summary
        ax4 = axes[1, 1]
        summary_data = []
        summary_labels = []
        
        for report_name, report in reports.items():
            if "summary" in report:
                summary = report["summary"]
                summary_data.append(summary.get("total_profiling_time", 0))
                summary_labels.append(report_name)
        
        if summary_data:
            bars = ax4.bar(summary_labels, summary_data)
            ax4.set_title("Total Profiling Time by Demo")
            ax4.set_ylabel("Time (seconds)")
            ax4.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, time_val in zip(bars, summary_data):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{time_val:.2f}s', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_dir / "profiling_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # Create bottleneck analysis visualization
    all_bottlenecks = []
    for report_name, report in reports.items():
        if "bottlenecks" in report:
            for bottleneck in report["bottlenecks"]:
                bottleneck["demo"] = report_name
                all_bottlenecks.append(bottleneck)
    
    if all_bottlenecks:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Count bottlenecks by type and severity
        bottleneck_counts = {}
        for bottleneck in all_bottlenecks:
            key = f"{bottleneck['type']}_{bottleneck['severity']}"
            bottleneck_counts[key] = bottleneck_counts.get(key, 0) + 1
        
        if bottleneck_counts:
            labels = list(bottleneck_counts.keys())
            counts = list(bottleneck_counts.values())
            
            bars = ax.bar(labels, counts)
            ax.set_title("Bottleneck Analysis Summary")
            ax.set_ylabel("Number of Bottlenecks")
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        str(count), ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(output_dir / "bottleneck_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    print(f"Visualizations saved to {output_dir}")


async def main():
    """Main demonstration function"""
    
    print("Code Profiling Demo")
    print("="*50)
    
    try:
        # Run all demonstrations
        reports = {}
        
        # Basic profiling
        reports["basic"] = demonstrate_basic_profiling()
        
        # Data loading profiling
        reports["data_loading"] = demonstrate_data_loading_profiling()
        
        # Preprocessing profiling
        reports["preprocessing"] = demonstrate_preprocessing_profiling()
        
        # Model training profiling
        reports["training"] = demonstrate_model_training_profiling()
        
        # Integrated profiling
        demonstrate_integrated_profiling()
        
        # Bottleneck analysis
        reports["bottleneck_analysis"] = demonstrate_bottleneck_analysis()
        
        # Create comprehensive visualizations
        create_profiling_visualizations(reports)
        
        # Save combined results
        combined_results = {
            "reports": reports,
            "summary": {
                "total_demos": len(reports),
                "total_bottlenecks": sum(len(r.get("bottlenecks", [])) for r in reports.values()),
                "total_recommendations": sum(len(r.get("recommendations", [])) for r in reports.values())
            }
        }
        
        with open("profiles/combined_results.json", 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(combined_results, f, indent=2)
        
        print("\n" + "="*50)
        print("PROFILING DEMO COMPLETED SUCCESSFULLY!")
        print("="*50)
        print("\nKey Features Demonstrated:")
        print("✓ Basic profiling with context managers")
        print("✓ Data loading performance analysis")
        print("✓ Preprocessing optimization profiling")
        print("✓ Model training performance profiling")
        print("✓ Integrated profiling with training optimizer")
        print("✓ Bottleneck identification and analysis")
        print("✓ Comprehensive reporting and visualizations")
        print("✓ Performance recommendations")
        
        print(f"\nResults saved to: profiles/")
        print(f"Combined results: profiles/combined_results.json")
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main()) 