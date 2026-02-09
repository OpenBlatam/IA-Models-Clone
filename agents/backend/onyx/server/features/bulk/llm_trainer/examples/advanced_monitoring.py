"""
Advanced Monitoring Example
==========================

Example showing experiment tracking and performance profiling.

Author: BUL System
Date: 2024
"""

from llm_trainer import CustomLLMTrainer, ExperimentTracker, PerformanceProfiler

# Example: Training with experiment tracking and profiling
print("=" * 80)
print("Advanced Monitoring Example")
print("=" * 80)

trainer = CustomLLMTrainer(
    model_name="gpt2",
    dataset_path="data/training.json",
    output_dir="./checkpoints",
    # Enable monitoring
    enable_experiment_tracking=True,
    experiments_dir="./experiments",
    enable_profiling=True,
    enable_distributed=False,  # Set to True for multi-GPU
)

# Get profiling report after training
results = trainer.train()

if "profiling" in results:
    prof = results["profiling"]
    print(f"\n📊 Performance Report:")
    print(f"   Steps/sec: {prof.get('steps_per_second', 0):.2f}")
    print(f"   Total time: {prof.get('total_time', 0):.1f}s")
    
    bottlenecks = prof.get("bottlenecks", [])
    if bottlenecks:
        print(f"\n🔍 Bottlenecks:")
        for bottleneck in bottlenecks[:3]:  # Top 3
            print(f"   {bottleneck['operation']}: {bottleneck['percentage']:.1f}%")

# List experiments
if trainer.experiment_tracker:
    experiments = trainer.experiment_tracker.list_experiments()
    print(f"\n📝 Experiments: {len(experiments)}")
    for exp in experiments[:5]:  # Show first 5
        print(f"   - {exp['name']}: {exp['status']}")

# Compare experiments
if trainer.experiment_tracker and len(experiments) > 1:
    exp_ids = [exp["experiment_id"] for exp in experiments[:2]]
    comparison = trainer.experiment_tracker.compare_experiments(exp_ids)
    print(f"\n📊 Comparison:")
    print(f"   Metrics: {list(comparison.get('metrics_comparison', {}).keys())}")

