"""
Optimization Example
===================

Example showing optimization features.

Author: BUL System
Date: 2024
"""

from llm_trainer import (
    CustomLLMTrainer,
    DistributedTrainingHelper,
    MemoryOptimizer,
)

# Example 1: Memory Optimization
print("=" * 80)
print("Example 1: Memory Optimization")
print("=" * 80)

memory_optimizer = MemoryOptimizer()

# Estimate memory usage
model_size_gb = 1.5  # GPT-2 medium
memory_est = memory_optimizer.estimate_memory_usage(
    model_size_gb=model_size_gb,
    batch_size=8,
    sequence_length=512,
    precision="fp32"
)

print(f"Estimated memory usage:")
print(f"  Model: {memory_est['model_memory_gb']:.2f} GB")
print(f"  Activations: {memory_est['activation_memory_gb']:.2f} GB")
print(f"  Total: {memory_est['total_memory_gb']:.2f} GB")

# Get recommendations
recommendations = memory_optimizer.get_optimization_recommendations(
    model_size_gb=model_size_gb,
    batch_size=8,
    sequence_length=512
)

print(f"\nRecommendations:")
for rec in recommendations["recommendations"]:
    print(f"  {rec}")

print(f"\nSuggested batch size: {recommendations['suggested_batch_size']}")
print(f"Suggested precision: {recommendations['suggested_precision']}")

# Example 2: Distributed Training
print("\n" + "=" * 80)
print("Example 2: Distributed Training")
print("=" * 80)

distributed_helper = DistributedTrainingHelper()

config = distributed_helper.get_config()
print(f"Distributed config: {config}")

if distributed_helper.is_available():
    print("Distributed training is available")
    if not distributed_helper.is_distributed():
        print("To enable distributed training:")
        print("  export WORLD_SIZE=4")
        print("  export RANK=0")
        print("  python -m torch.distributed.launch --nproc_per_node=4 train.py")
else:
    print("Distributed training not available")

# Get recommendations
dist_recs = distributed_helper.get_recommendations()
for rec in dist_recs:
    print(f"  {rec}")

# Example 3: Optimized Training Setup
print("\n" + "=" * 80)
print("Example 3: Optimized Training Setup")
print("=" * 80)

# Get memory recommendations
memory_recs = memory_optimizer.get_optimization_recommendations(
    model_size_gb=1.5,
    batch_size=8
)

# Create trainer with optimizations
trainer = CustomLLMTrainer(
    model_name="gpt2",
    dataset_path="data.json",
    output_dir="./checkpoints",
    batch_size=memory_recs["suggested_batch_size"],
    fp16=(memory_recs["suggested_precision"] == "fp16"),
    gradient_checkpointing=True,
)

print("Trainer configured with optimizations:")
print(f"  Batch size: {trainer.training_args.per_device_train_batch_size}")
print(f"  FP16: {trainer.training_args.fp16}")
print(f"  Gradient checkpointing: {trainer.training_args.gradient_checkpointing}")

