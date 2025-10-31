from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

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
from core.mixed_precision_training import (
from core.training_logger import create_training_logger
from core.optimized_training_optimizer import create_optimized_training_optimizer
from core.performance_optimization import PerformanceConfig
from core.multi_gpu_training import MultiGPUConfig
from core.gradient_accumulation import GradientAccumulationConfig
        import traceback
from typing import Any, List, Dict, Optional
import logging
"""
Mixed Precision Training Demo

Comprehensive demonstration of mixed precision training with PyTorch's Automatic Mixed Precision (AMP)
using torch.cuda.amp for improved training speed and reduced memory usage.
"""


    MixedPrecisionConfig, MixedPrecisionTrainer, MixedPrecisionOptimizer,
    create_mixed_precision_trainer, create_mixed_precision_optimizer,
    check_amp_compatibility
)


class EmailSequenceModel(nn.Module):
    """Simple email sequence model for demonstration"""
    
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


def benchmark_training_speeds(model, train_loader, device, num_iterations=100) -> Any:
    """Benchmark training speeds with and without mixed precision"""
    
    print("\n" + "="*60)
    print("BENCHMARKING TRAINING SPEEDS")
    print("="*60)
    
    # Standard precision training
    print("\n1. Standard Precision Training:")
    model_std = EmailSequenceModel().to(device)
    optimizer_std = optim.Adam(model_std.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    
    # Warmup
    for _ in range(10):
        batch = next(iter(train_loader))
        inputs, targets = batch[0].to(device), batch[1].to(device)
        optimizer_std.zero_grad()
        outputs = model_std(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer_std.step()
    
    # Benchmark standard precision
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(num_iterations):
        batch = next(iter(train_loader))
        inputs, targets = batch[0].to(device), batch[1].to(device)
        optimizer_std.zero_grad()
        outputs = model_std(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer_std.step()
    
    torch.cuda.synchronize()
    std_time = time.time() - start_time
    
    print(f"   Standard precision time: {std_time:.4f}s")
    print(f"   Average time per iteration: {std_time/num_iterations:.6f}s")
    
    # Mixed precision training
    print("\n2. Mixed Precision Training:")
    model_amp = EmailSequenceModel().to(device)
    optimizer_amp = optim.Adam(model_amp.parameters(), lr=0.001)
    
    # Create mixed precision trainer
    mp_trainer = create_mixed_precision_trainer(
        model=model_amp,
        optimizer=optimizer_amp,
        enable_amp=True,
        dtype=torch.float16,
        enable_grad_scaler=True
    )
    
    # Warmup
    for _ in range(10):
        batch = next(iter(train_loader))
        inputs, targets = batch[0].to(device), batch[1].to(device)
        mp_trainer.train_step(
            (inputs, targets),
            lambda outputs, targets: nn.CrossEntropyLoss()(outputs, targets),
            device
        )
    
    # Benchmark mixed precision
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(num_iterations):
        batch = next(iter(train_loader))
        inputs, targets = batch[0].to(device), batch[1].to(device)
        mp_trainer.train_step(
            (inputs, targets),
            lambda outputs, targets: nn.CrossEntropyLoss()(outputs, targets),
            device
        )
    
    torch.cuda.synchronize()
    amp_time = time.time() - start_time
    
    print(f"   Mixed precision time: {amp_time:.4f}s")
    print(f"   Average time per iteration: {amp_time/num_iterations:.6f}s")
    
    # Calculate speedup
    speedup = std_time / amp_time
    print(f"\n3. Results:")
    print(f"   Speedup: {speedup:.2f}x")
    print(f"   Time savings: {((std_time - amp_time) / std_time * 100):.1f}%")
    
    return {
        "standard_time": std_time,
        "mixed_precision_time": amp_time,
        "speedup": speedup,
        "time_savings_percent": (std_time - amp_time) / std_time * 100
    }


def benchmark_memory_usage(model, train_loader, device, num_iterations=50) -> Any:
    """Benchmark memory usage with and without mixed precision"""
    
    print("\n" + "="*60)
    print("BENCHMARKING MEMORY USAGE")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping memory benchmark")
        return {}
    
    # Standard precision training
    print("\n1. Standard Precision Memory Usage:")
    model_std = EmailSequenceModel().to(device)
    optimizer_std = optim.Adam(model_std.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    
    # Clear cache
    torch.cuda.empty_cache()
    
    # Record initial memory
    initial_memory = torch.cuda.memory_allocated() / 1024**3  # GB
    
    # Training loop
    max_memory = 0
    for _ in range(num_iterations):
        batch = next(iter(train_loader))
        inputs, targets = batch[0].to(device), batch[1].to(device)
        optimizer_std.zero_grad()
        outputs = model_std(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer_std.step()
        
        current_memory = torch.cuda.memory_allocated() / 1024**3
        max_memory = max(max_memory, current_memory)
    
    std_max_memory = max_memory
    print(f"   Peak memory usage: {std_max_memory:.3f} GB")
    
    # Mixed precision training
    print("\n2. Mixed Precision Memory Usage:")
    model_amp = EmailSequenceModel().to(device)
    optimizer_amp = optim.Adam(model_amp.parameters(), lr=0.001)
    
    # Create mixed precision trainer
    mp_trainer = create_mixed_precision_trainer(
        model=model_amp,
        optimizer=optimizer_amp,
        enable_amp=True,
        dtype=torch.float16,
        enable_grad_scaler=True
    )
    
    # Clear cache
    torch.cuda.empty_cache()
    
    # Training loop
    max_memory = 0
    for _ in range(num_iterations):
        batch = next(iter(train_loader))
        inputs, targets = batch[0].to(device), batch[1].to(device)
        mp_trainer.train_step(
            (inputs, targets),
            lambda outputs, targets: nn.CrossEntropyLoss()(outputs, targets),
            device
        )
        
        current_memory = torch.cuda.memory_allocated() / 1024**3
        max_memory = max(max_memory, current_memory)
    
    amp_max_memory = max_memory
    print(f"   Peak memory usage: {amp_max_memory:.3f} GB")
    
    # Calculate memory savings
    memory_savings = (std_max_memory - amp_max_memory) / std_max_memory * 100
    print(f"\n3. Results:")
    print(f"   Memory savings: {memory_savings:.1f}%")
    print(f"   Memory reduction: {std_max_memory - amp_max_memory:.3f} GB")
    
    return {
        "standard_memory": std_max_memory,
        "mixed_precision_memory": amp_max_memory,
        "memory_savings_percent": memory_savings,
        "memory_reduction_gb": std_max_memory - amp_max_memory
    }


def demonstrate_mixed_precision_training():
    """Demonstrate mixed precision training with comprehensive metrics"""
    
    print("="*80)
    print("MIXED PRECISION TRAINING DEMONSTRATION")
    print("="*80)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Create dataset
    print("\nCreating dataset...")
    dataset = create_dummy_dataset(num_samples=2000, seq_length=50, vocab_size=1000)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)
    
    # Create model
    model = EmailSequenceModel()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Check AMP compatibility
    print("\nChecking AMP compatibility...")
    compatibility = check_amp_compatibility(model)
    print(f"Compatibility: {compatibility}")
    
    # Benchmark training speeds
    speed_benchmark = benchmark_training_speeds(model, train_loader, device)
    
    # Benchmark memory usage
    memory_benchmark = benchmark_memory_usage(model, train_loader, device)
    
    # Demonstrate mixed precision training with detailed monitoring
    print("\n" + "="*60)
    print("DETAILED MIXED PRECISION TRAINING")
    print("="*60)
    
    # Create mixed precision configuration
    mp_config = MixedPrecisionConfig(
        enable_amp=True,
        dtype=torch.float16,
        enable_grad_scaler=True,
        init_scale=2**16,
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=2000,
        enable_monitoring=True,
        log_amp_stats=True,
        track_memory_usage=True,
        validate_amp=True,
        check_compatibility=True
    )
    
    # Create training logger
    logger = create_training_logger(
        experiment_name="mixed_precision_demo",
        log_dir="logs/mixed_precision_demo",
        log_level="INFO",
        enable_visualization=True,
        enable_metrics_logging=True
    )
    
    # Create mixed precision trainer
    model_mp = EmailSequenceModel().to(device)
    optimizer_mp = optim.Adam(model_mp.parameters(), lr=0.001)
    
    mp_trainer = MixedPrecisionTrainer(
        config=mp_config,
        model=model_mp,
        optimizer=optimizer_mp,
        logger=logger
    )
    
    # Training loop with detailed monitoring
    print("\nTraining with mixed precision...")
    num_epochs = 3
    total_steps = 0
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        num_batches = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Training step
            step_metrics = mp_trainer.train_step(
                (inputs, targets),
                lambda outputs, targets: nn.CrossEntropyLoss()(outputs, targets),
                device
            )
            
            # Accumulate metrics
            epoch_loss += step_metrics["loss"]
            epoch_accuracy += step_metrics["accuracy"]
            num_batches += 1
            total_steps += 1
            
            # Log progress
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}: "
                      f"Loss={step_metrics['loss']:.6f}, "
                      f"Accuracy={step_metrics['accuracy']:.4f}, "
                      f"Time={step_metrics['step_time']:.4f}s, "
                      f"Scale={step_metrics.get('scaler_scale', 'N/A')}")
            
            # Get AMP statistics periodically
            if total_steps % 50 == 0:
                amp_stats = mp_trainer.get_amp_stats()
                print(f"AMP Stats: Scale={amp_stats.get('current_scale', 'N/A')}, "
                      f"Overflow={amp_stats.get('overflow_count', 0)}, "
                      f"Memory Savings={amp_stats.get('avg_memory_savings', 0):.3f}GB")
        
        # Epoch summary
        avg_loss = epoch_loss / num_batches
        avg_accuracy = epoch_accuracy / num_batches
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Average Loss: {avg_loss:.6f}")
        print(f"  Average Accuracy: {avg_accuracy:.4f}")
    
    # Final AMP statistics
    print("\n" + "="*60)
    print("FINAL MIXED PRECISION STATISTICS")
    print("="*60)
    
    final_stats = mp_trainer.get_amp_stats()
    print(json.dumps(final_stats, indent=2))
    
    # Save AMP statistics
    mp_trainer.save_amp_stats("logs/mixed_precision_demo/amp_final_stats.json")
    
    # Cleanup
    mp_trainer.cleanup()
    
    # Combined results
    results = {
        "speed_benchmark": speed_benchmark,
        "memory_benchmark": memory_benchmark,
        "final_amp_stats": final_stats,
        "compatibility": compatibility
    }
    
    # Save results
    with open("logs/mixed_precision_demo/demo_results.json", 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        json.dump(results, f, indent=2)
    
    print(f"\nDemo completed! Results saved to logs/mixed_precision_demo/")
    
    return results


def demonstrate_integrated_training():
    """Demonstrate mixed precision training integrated with the optimized training optimizer"""
    
    print("\n" + "="*80)
    print("INTEGRATED MIXED PRECISION TRAINING")
    print("="*80)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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
        enable_multi_gpu=False,  # Set to True for multi-GPU
        strategy="data_parallel"
    )
    
    gradient_accumulation_config = GradientAccumulationConfig(
        accumulation_steps=4,
        memory_efficient=True
    )
    
    # Create optimized training optimizer with mixed precision
    optimizer = create_optimized_training_optimizer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        experiment_name="integrated_mixed_precision_demo",
        debug_mode=False,
        enable_pytorch_debugging=False,
        performance_config=performance_config,
        multi_gpu_config=multi_gpu_config,
        gradient_accumulation_config=gradient_accumulation_config,
        # Mixed precision configuration
        enable_amp=True,
        amp_dtype=torch.float16,
        enable_grad_scaler=True,
        amp_init_scale=2**16,
        amp_growth_factor=2.0,
        amp_backoff_factor=0.5,
        amp_growth_interval=2000,
        amp_monitoring=True,
        amp_log_stats=True,
        amp_track_memory=True,
        amp_validate=True,
        amp_check_compatibility=True
    )
    
    print("Optimized training optimizer with mixed precision created successfully!")
    print(f"Mixed precision enabled: {optimizer.mp_trainer is not None}")
    
    # Get training summary
    summary = optimizer.get_training_summary()
    print(f"\nTraining configuration: {json.dumps(summary, indent=2)}")
    
    # Cleanup
    optimizer.cleanup()
    
    print("Integrated training demonstration completed!")


def create_visualizations(results) -> Any:
    """Create visualizations for mixed precision training results"""
    
    print("\nCreating visualizations...")
    
    # Create output directory
    output_dir = Path("logs/mixed_precision_demo/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Speed comparison
    if "speed_benchmark" in results:
        speed_data = results["speed_benchmark"]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Training time comparison
        methods = ["Standard", "Mixed Precision"]
        times = [speed_data["standard_time"], speed_data["mixed_precision_time"]]
        
        bars1 = ax1.bar(methods, times, color=['#ff7f0e', '#2ca02c'])
        ax1.set_title("Training Time Comparison")
        ax1.set_ylabel("Time (seconds)")
        ax1.set_ylim(0, max(times) * 1.1)
        
        # Add value labels on bars
        for bar, time in zip(bars1, times):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{time:.3f}s', ha='center', va='bottom')
        
        # Speedup
        speedup = speed_data["speedup"]
        ax2.bar(["Speedup"], [speedup], color='#d62728')
        ax2.set_title(f"Training Speedup: {speedup:.2f}x")
        ax2.set_ylabel("Speedup Factor")
        ax2.set_ylim(0, speedup * 1.1)
        
        # Add value label
        ax2.text(0, speedup + 0.1, f'{speedup:.2f}x', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_dir / "speed_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # Memory comparison
    if "memory_benchmark" in results:
        memory_data = results["memory_benchmark"]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Memory usage comparison
        methods = ["Standard", "Mixed Precision"]
        memory = [memory_data["standard_memory"], memory_data["mixed_precision_memory"]]
        
        bars2 = ax1.bar(methods, memory, color=['#ff7f0e', '#2ca02c'])
        ax1.set_title("Memory Usage Comparison")
        ax1.set_ylabel("Memory (GB)")
        ax1.set_ylim(0, max(memory) * 1.1)
        
        # Add value labels on bars
        for bar, mem in zip(bars2, memory):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{mem:.3f}GB', ha='center', va='bottom')
        
        # Memory savings
        savings = memory_data["memory_savings_percent"]
        ax2.bar(["Memory Savings"], [savings], color='#d62728')
        ax2.set_title(f"Memory Savings: {savings:.1f}%")
        ax2.set_ylabel("Savings (%)")
        ax2.set_ylim(0, savings * 1.1)
        
        # Add value label
        ax2.text(0, savings + 1, f'{savings:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_dir / "memory_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # AMP statistics visualization
    if "final_amp_stats" in results:
        amp_stats = results["final_amp_stats"]
        
        if "scale_values" in amp_stats and amp_stats["scale_values"]:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            scale_values = amp_stats["scale_values"]
            steps = range(len(scale_values))
            
            ax.plot(steps, scale_values, 'b-', linewidth=2)
            ax.set_title("Gradient Scaler Scale Over Time")
            ax.set_xlabel("Training Steps")
            ax.set_ylabel("Scale Value")
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / "scaler_scale.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    print(f"Visualizations saved to {output_dir}")


async def main():
    """Main demonstration function"""
    
    print("Mixed Precision Training Demo")
    print("="*50)
    
    try:
        # Run basic mixed precision demonstration
        results = demonstrate_mixed_precision_training()
        
        # Run integrated training demonstration
        demonstrate_integrated_training()
        
        # Create visualizations
        create_visualizations(results)
        
        print("\n" + "="*50)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("="*50)
        print("\nKey Benefits Demonstrated:")
        print("✓ Training speed improvement")
        print("✓ Memory usage reduction")
        print("✓ Automatic gradient scaling")
        print("✓ Overflow detection and handling")
        print("✓ Comprehensive monitoring and logging")
        print("✓ Integration with optimized training pipeline")
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main()) 