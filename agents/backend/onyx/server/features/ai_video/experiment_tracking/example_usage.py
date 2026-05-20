from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import os
import sys
import time
import logging
import numpy as np
from pathlib import Path
from experiment_tracking import (
        import matplotlib.pyplot as plt
from typing import Any, List, Dict, Optional
import asyncio
"""
Example Usage of Experiment Tracking System
==========================================

This script demonstrates various ways to use the experiment tracking system
for AI video generation experiments.

Scenarios covered:
1. Basic experiment tracking
2. Video-specific metrics tracking
3. Performance monitoring
4. Checkpoint management
5. Integration with training loops
6. Experiment comparison and analysis
7. Data export and visualization
"""


# Add the parent directory to the path to import modules
sys.path.append(str(Path(__file__).parent.parent))

    setup_experiment_tracking,
    create_video_experiment_tracker,
    log_training_step,
    log_video_generation,
    get_experiment_summary,
    export_experiment_data,
    ExperimentTracker,
    VideoMetricsTracker,
    CheckpointManager,
    PerformanceMonitor
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_1_basic_experiment_tracking():
    """Example 1: Basic experiment tracking."""
    print("\n" + "="*50)
    print("EXAMPLE 1: Basic Experiment Tracking")
    print("="*50)
    
    # Set up tracking system
    tracker, metrics, checkpoint_mgr = setup_experiment_tracking(
        "basic_experiment",
        use_wandb=False,
        use_tensorboard=True,
        save_frequency=50,
        max_checkpoints=3
    )
    
    # Mock model and optimizer
    class MockModel:
        def __init__(self) -> Any:
            self.weights = np.random.randn(100, 100)
        
        def state_dict(self) -> Any:
            return {"weights": self.weights}
        
        def load_state_dict(self, state_dict) -> Any:
            self.weights = state_dict["weights"]
    
    class MockOptimizer:
        def __init__(self) -> Any:
            self.lr = 1e-4
        
        def state_dict(self) -> Any:
            return {"lr": self.lr}
        
        def load_state_dict(self, state_dict) -> Any:
            self.lr = state_dict["lr"]
    
    model = MockModel()
    optimizer = MockOptimizer()
    
    # Simulate training
    print("🚀 Starting training simulation...")
    for step in range(200):
        # Simulate training
        loss = 1.0 / (step + 1) + np.random.normal(0, 0.01)
        accuracy = 0.5 + 0.4 * (1 - np.exp(-step / 50)) + np.random.normal(0, 0.02)
        learning_rate = 1e-4 * (0.95 ** (step // 20))
        
        # Log training step
        log_training_step(
            tracker=tracker,
            metrics_tracker=metrics,
            checkpoint_manager=checkpoint_mgr,
            model=model,
            optimizer=optimizer,
            loss=loss,
            step=step,
            epoch=step // 50,
            additional_metrics={
                "accuracy": accuracy,
                "learning_rate": learning_rate
            },
            save_checkpoint=(step % 50 == 0)
        )
        
        if step % 20 == 0:
            print(f"   Step {step}: Loss={loss:.4f}, Accuracy={accuracy:.4f}")
    
    # Get experiment summary
    summary = get_experiment_summary(tracker, metrics, checkpoint_mgr)
    print(f"\n📊 Experiment Summary:")
    print(f"   Duration: {summary['experiment']['duration_seconds']:.2f} seconds")
    print(f"   Total steps: {summary['experiment']['total_steps']}")
    print(f"   Total checkpoints: {summary['checkpoints']['total_checkpoints']}")
    print(f"   Best loss: {summary['metrics']['loss']['statistics']['min']:.4f}")
    
    # Export data
    export_experiment_data(tracker, metrics, checkpoint_mgr, "exports/basic_experiment")
    
    # Close tracker
    tracker.close()
    print("✅ Basic experiment tracking completed!")


def example_2_video_specific_tracking():
    """Example 2: Video-specific metrics tracking."""
    print("\n" + "="*50)
    print("EXAMPLE 2: Video-Specific Metrics Tracking")
    print("="*50)
    
    # Set up video experiment tracking
    tracker, video_metrics, checkpoint_mgr = create_video_experiment_tracker(
        "video_generation_experiment",
        use_wandb=False,
        use_tensorboard=True
    )
    
    # Mock model
    class MockVideoModel:
        def __init__(self) -> Any:
            self.weights = np.random.randn(100, 100)
        
        def state_dict(self) -> Any:
            return {"weights": self.weights}
        
        def load_state_dict(self, state_dict) -> Any:
            self.weights = state_dict["weights"]
    
    model = MockVideoModel()
    optimizer = MockOptimizer()
    
    # Simulate video generation training
    print("🎬 Starting video generation training...")
    for step in range(100):
        # Simulate training metrics
        loss = 0.5 / (step + 1) + np.random.normal(0, 0.005)
        
        # Simulate video generation metrics (every 10 steps)
        if step % 10 == 0:
            # Simulate video quality metrics
            psnr = 25 + 5 * (1 - np.exp(-step / 30)) + np.random.normal(0, 0.5)
            ssim = 0.7 + 0.2 * (1 - np.exp(-step / 25)) + np.random.normal(0, 0.02)
            lpips = 0.3 * np.exp(-step / 40) + np.random.normal(0, 0.01)
            generation_time = 2.0 + np.random.normal(0, 0.1)
            
            # Log video metrics
            log_video_generation(
                video_metrics_tracker=video_metrics,
                psnr=psnr,
                ssim=ssim,
                lpips=lpips,
                generation_time=generation_time,
                step=step
            )
            
            print(f"   Step {step}: PSNR={psnr:.2f}, SSIM={ssim:.3f}, LPIPS={lpips:.3f}")
        
        # Log training step
        log_training_step(
            tracker=tracker,
            metrics_tracker=video_metrics.base_tracker,
            checkpoint_manager=checkpoint_mgr,
            model=model,
            optimizer=optimizer,
            loss=loss,
            step=step,
            save_checkpoint=(step % 25 == 0)
        )
    
    # Get video quality summary
    quality_summary = video_metrics.get_video_quality_summary()
    print(f"\n🎥 Video Quality Summary:")
    for metric, values in quality_summary.items():
        if values['latest'] is not None:
            print(f"   {metric.upper()}: Latest={values['latest']:.3f}, Best={values['best']:.3f}")
    
    # Export data
    export_experiment_data(tracker, video_metrics.base_tracker, checkpoint_mgr, "exports/video_experiment")
    
    # Close tracker
    tracker.close()
    print("✅ Video-specific tracking completed!")


def example_3_performance_monitoring():
    """Example 3: Performance monitoring."""
    print("\n" + "="*50)
    print("EXAMPLE 3: Performance Monitoring")
    print("="*50)
    
    # Set up tracking with performance monitoring
    tracker, metrics, checkpoint_mgr = setup_experiment_tracking(
        "performance_monitoring_experiment",
        use_wandb=False,
        use_tensorboard=True
    )
    
    # Create performance monitor
    perf_monitor = PerformanceMonitor(metrics)
    
    # Mock model
    model = MockVideoModel()
    optimizer = MockOptimizer()
    
    # Simulate training with performance monitoring
    print("⚡ Starting performance monitoring...")
    for step in range(50):
        # Simulate training
        loss = 1.0 / (step + 1) + np.random.normal(0, 0.01)
        
        # Check performance every 10 steps
        if step % 10 == 0:
            perf_monitor.check_performance(step)
        
        # Log training step
        log_training_step(
            tracker=tracker,
            metrics_tracker=metrics,
            checkpoint_manager=checkpoint_mgr,
            model=model,
            optimizer=optimizer,
            loss=loss,
            step=step,
            save_checkpoint=(step % 25 == 0)
        )
        
        # Simulate some processing time
        time.sleep(0.01)
    
    # Get performance metrics
    perf_metrics = metrics.get_metric_summary()
    print(f"\n⚡ Performance Metrics:")
    for metric_name in ["gpu_utilization", "gpu_memory_used", "cpu_utilization", "throughput"]:
        if metric_name in perf_metrics:
            stats = perf_metrics[metric_name]["statistics"]
            print(f"   {metric_name}: Mean={stats['mean']:.2f}, Max={stats['max']:.2f}")
    
    # Export data
    export_experiment_data(tracker, metrics, checkpoint_mgr, "exports/performance_experiment")
    
    # Close tracker
    tracker.close()
    print("✅ Performance monitoring completed!")


def example_4_checkpoint_management():
    """Example 4: Advanced checkpoint management."""
    print("\n" + "="*50)
    print("EXAMPLE 4: Advanced Checkpoint Management")
    print("="*50)
    
    # Set up tracking with checkpoint management
    tracker, metrics, checkpoint_mgr = setup_experiment_tracking(
        "checkpoint_management_experiment",
        use_wandb=False,
        use_tensorboard=True,
        save_frequency=20,
        max_checkpoints=3
    )
    
    # Mock model
    model = MockVideoModel()
    optimizer = MockOptimizer()
    
    # Simulate training with checkpoint management
    print("💾 Starting checkpoint management...")
    for step in range(100):
        # Simulate training
        loss = 1.0 / (step + 1) + np.random.normal(0, 0.01)
        val_loss = 1.2 / (step + 1) + np.random.normal(0, 0.01)
        
        # Log training step
        log_training_step(
            tracker=tracker,
            metrics_tracker=metrics,
            checkpoint_manager=checkpoint_mgr,
            model=model,
            optimizer=optimizer,
            loss=loss,
            step=step,
            additional_metrics={"val_loss": val_loss},
            save_checkpoint=(step % 20 == 0)
        )
        
        if step % 20 == 0:
            print(f"   Step {step}: Checkpoint saved")
    
    # List checkpoints
    checkpoints = checkpoint_mgr.list_checkpoints()
    print(f"\n📁 Checkpoints ({len(checkpoints)} total):")
    for checkpoint in checkpoints:
        print(f"   {checkpoint.checkpoint_id}: Step {checkpoint.step}, Loss {checkpoint.metrics.get('loss', 0):.4f}")
    
    # Get best checkpoint
    best_checkpoint = checkpoint_mgr.get_best_checkpoint("val_loss")
    if best_checkpoint:
        print(f"\n🏆 Best checkpoint: {best_checkpoint.checkpoint_id}")
        print(f"   Step: {best_checkpoint.step}")
        print(f"   Val Loss: {best_checkpoint.metrics.get('val_loss', 0):.4f}")
    
    # Load best checkpoint
    if best_checkpoint:
        checkpoint_data = checkpoint_mgr.load_checkpoint(
            best_checkpoint.checkpoint_path,
            model,
            optimizer
        )
        if checkpoint_data:
            print(f"   ✅ Best checkpoint loaded successfully")
    
    # Get checkpoint summary
    summary = checkpoint_mgr.get_checkpoint_summary()
    print(f"\n💾 Checkpoint Summary:")
    print(f"   Total size: {summary['total_size_mb']:.2f} MB")
    print(f"   Average compression: {summary['average_compression_ratio']:.2f}")
    
    # Export data
    export_experiment_data(tracker, metrics, checkpoint_mgr, "exports/checkpoint_experiment")
    
    # Close tracker
    tracker.close()
    print("✅ Checkpoint management completed!")


def example_5_experiment_comparison():
    """Example 5: Experiment comparison and analysis."""
    print("\n" + "="*50)
    print("EXAMPLE 5: Experiment Comparison and Analysis")
    print("="*50)
    
    # Run multiple experiments
    experiments = [
        ("experiment_1", {"learning_rate": 1e-4, "batch_size": 8}),
        ("experiment_2", {"learning_rate": 5e-5, "batch_size": 16}),
        ("experiment_3", {"learning_rate": 2e-4, "batch_size": 4})
    ]
    
    experiment_results = []
    
    for exp_name, config in experiments:
        print(f"\n🔬 Running {exp_name}...")
        
        # Set up tracking
        tracker, metrics, checkpoint_mgr = setup_experiment_tracking(
            exp_name,
            use_wandb=False,
            use_tensorboard=True,
            save_frequency=25,
            max_checkpoints=2
        )
        
        # Mock model
        model = MockVideoModel()
        optimizer = MockOptimizer()
        
        # Simulate training with different configs
        for step in range(75):
            # Adjust loss based on config
            lr_factor = config["learning_rate"] / 1e-4
            batch_factor = config["batch_size"] / 8
            loss = (1.0 / (step + 1)) * lr_factor * batch_factor + np.random.normal(0, 0.01)
            
            # Log training step
            log_training_step(
                tracker=tracker,
                metrics_tracker=metrics,
                checkpoint_manager=checkpoint_mgr,
                model=model,
                optimizer=optimizer,
                loss=loss,
                step=step,
                additional_metrics={"learning_rate": config["learning_rate"]},
                save_checkpoint=(step % 25 == 0)
            )
        
        # Get experiment summary
        summary = get_experiment_summary(tracker, metrics, checkpoint_mgr)
        experiment_results.append({
            "name": exp_name,
            "config": config,
            "summary": summary
        })
        
        # Export data
        export_experiment_data(tracker, metrics, checkpoint_mgr, f"exports/{exp_name}")
        
        # Close tracker
        tracker.close()
    
    # Compare experiments
    print(f"\n📊 Experiment Comparison:")
    print(f"{'Experiment':<15} {'LR':<10} {'Batch':<8} {'Best Loss':<10} {'Duration':<10}")
    print("-" * 60)
    
    for result in experiment_results:
        name = result["name"]
        config = result["config"]
        summary = result["summary"]
        
        best_loss = summary["metrics"]["loss"]["statistics"]["min"]
        duration = summary["experiment"]["duration_seconds"]
        
        print(f"{name:<15} {config['learning_rate']:<10.0e} {config['batch_size']:<8} {best_loss:<10.4f} {duration:<10.2f}")
    
    # Find best experiment
    best_experiment = min(experiment_results, 
                         key=lambda x: x["summary"]["metrics"]["loss"]["statistics"]["min"])
    
    print(f"\n🏆 Best experiment: {best_experiment['name']}")
    print(f"   Config: {best_experiment['config']}")
    print(f"   Best loss: {best_experiment['summary']['metrics']['loss']['statistics']['min']:.4f}")
    
    print("✅ Experiment comparison completed!")


def example_6_data_visualization():
    """Example 6: Data visualization and analysis."""
    print("\n" + "="*50)
    print("EXAMPLE 6: Data Visualization and Analysis")
    print("="*50)
    
    # Set up tracking
    tracker, metrics, checkpoint_mgr = setup_experiment_tracking(
        "visualization_experiment",
        use_wandb=False,
        use_tensorboard=True
    )
    
    # Mock model
    model = MockVideoModel()
    optimizer = MockOptimizer()
    
    # Simulate training with various metrics
    print("📈 Starting visualization experiment...")
    for step in range(100):
        # Simulate various metrics
        loss = 1.0 / (step + 1) + np.random.normal(0, 0.01)
        accuracy = 0.5 + 0.4 * (1 - np.exp(-step / 30)) + np.random.normal(0, 0.02)
        learning_rate = 1e-4 * (0.95 ** (step // 20))
        gradient_norm = 1.0 / (step + 1) + np.random.normal(0, 0.1)
        
        # Log training step
        log_training_step(
            tracker=tracker,
            metrics_tracker=metrics,
            checkpoint_manager=checkpoint_mgr,
            model=model,
            optimizer=optimizer,
            loss=loss,
            step=step,
            additional_metrics={
                "accuracy": accuracy,
                "learning_rate": learning_rate,
                "gradient_norm": gradient_norm
            },
            save_checkpoint=(step % 25 == 0)
        )
    
    # Export metrics for visualization
    metrics.export_metrics("exports/visualization_experiment/metrics.json", "json")
    
    # Create plots if matplotlib is available
    try:
        
        # Plot metrics
        metrics.plot_all_metrics("exports/visualization_experiment/plots")
        print("📊 Plots saved to exports/visualization_experiment/plots/")
        
    except ImportError:
        print("📊 Matplotlib not available for plotting")
    
    # Get metric statistics
    metric_stats = metrics.get_metric_summary()
    print(f"\n📊 Metric Statistics:")
    for metric_name, stats in metric_stats.items():
        if "statistics" in stats:
            stat = stats["statistics"]
            print(f"   {metric_name}:")
            print(f"     Mean: {stat['mean']:.4f}")
            print(f"     Std: {stat['std']:.4f}")
            print(f"     Min: {stat['min']:.4f}")
            print(f"     Max: {stat['max']:.4f}")
    
    # Export data
    export_experiment_data(tracker, metrics, checkpoint_mgr, "exports/visualization_experiment")
    
    # Close tracker
    tracker.close()
    print("✅ Data visualization completed!")


def main():
    """Run all examples."""
    print("🔬 Experiment Tracking System Examples")
    print("=" * 60)
    
    try:
        # Create exports directory
        Path("exports").mkdir(exist_ok=True)
        
        # Run examples
        example_1_basic_experiment_tracking()
        example_2_video_specific_tracking()
        example_3_performance_monitoring()
        example_4_checkpoint_management()
        example_5_experiment_comparison()
        example_6_data_visualization()
        
        print("\n" + "="*60)
        print("✅ All examples completed successfully!")
        print("\n📁 Generated files:")
        
        # List generated files
        export_dirs = list(Path("exports").glob("*"))
        for export_dir in export_dirs:
            if export_dir.is_dir():
                files = list(export_dir.rglob("*"))
                print(f"   {export_dir.name}/ ({len(files)} files)")
        
        print("\n🎯 Next steps:")
        print("   1. Review generated experiment data")
        print("   2. Open TensorBoard: tensorboard --logdir runs")
        print("   3. Analyze metrics and checkpoints")
        print("   4. Compare different experiments")
        print("   5. Use the tracking system in your training scripts")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        logger.exception("Exception occurred")


match __name__:
    case "__main__":
    main() 