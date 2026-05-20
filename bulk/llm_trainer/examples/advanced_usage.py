"""
Advanced Usage Example
=====================

Example showing advanced features of CustomLLMTrainer.

Author: BUL System
Date: 2024
"""

from llm_trainer import (
    CustomLLMTrainer,
    CheckpointManager,
    ResumeManager,
    DatasetValidator,
    DatasetProcessor,
)

# Example 1: Pre-process dataset before training
print("=" * 80)
print("Example 1: Pre-processing Dataset")
print("=" * 80)

validator = DatasetValidator()
is_valid, errors, data = validator.validate_file("raw_data.json")
if is_valid:
    processor = DatasetProcessor()
    cleaned = processor.clean_dataset(data)
    filtered = processor.filter_by_length(
        cleaned,
        min_prompt_length=10,
        max_prompt_length=500,
        min_response_length=20,
        max_response_length=1000
    )
    print(f"Processed: {len(data)} -> {len(filtered)} examples")

# Example 2: Training with all features
print("\n" + "=" * 80)
print("Example 2: Full-Featured Training")
print("=" * 80)

trainer = CustomLLMTrainer(
    model_name="gpt2",
    dataset_path="processed_data.json",
    output_dir="./checkpoints",
    learning_rate=3e-5,
    num_train_epochs=3,
    batch_size=8,
    # Advanced features
    evaluation_strategy="steps",
    eval_steps=100,
    early_stopping_patience=3,
    load_best_model_at_end=True,
    gradient_checkpointing=True,
    fp16=True,
)

# Get recommendations
recommendations = trainer.get_training_recommendations()
print("\nRecommendations:")
for rec in recommendations:
    print(f"  • {rec}")

# Get time estimate
time_est = trainer.get_estimated_training_time()
print(f"\nEstimated training time: {time_est['total_hours']:.2f} hours")

# Train
print("\nStarting training...")
results = trainer.train()

print(f"\n✅ Training completed!")
print(f"   Final loss: {results['training_loss']:.4f}")
print(f"   Checkpoint: {results['checkpoint_path']}")

# Example 3: Checkpoint management
print("\n" + "=" * 80)
print("Example 3: Checkpoint Management")
print("=" * 80)

checkpoint_manager = CheckpointManager("./checkpoints")
checkpoints = checkpoint_manager.list_checkpoints()
print(f"Available checkpoints: {len(checkpoints)}")

if checkpoints:
    best = checkpoint_manager.get_best_checkpoint(metric="eval_loss")
    print(f"Best checkpoint: {best}")
    
    # Cleanup old checkpoints
    deleted = checkpoint_manager.cleanup_old_checkpoints(keep=3)
    print(f"Deleted {len(deleted)} old checkpoints")

# Example 4: Resume training
print("\n" + "=" * 80)
print("Example 4: Resume Training")
print("=" * 80)

resume_manager = ResumeManager("./checkpoints")
resume_info = resume_manager.get_resume_info()

if resume_info["can_resume"]:
    print(f"Can resume from: {resume_info['checkpoint_path']}")
    print(f"Step: {resume_info.get('step', 'unknown')}")
    
    # Resume training
    trainer.resume_from_latest()
else:
    print("No checkpoint available to resume from")

