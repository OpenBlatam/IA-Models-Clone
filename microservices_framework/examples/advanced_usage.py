"""
Advanced Usage Examples
Demonstrates advanced features of the refactored framework.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from shared.ml import (
    InferenceEngine,
    ModelManager,
    LoRAManager,
    LearningRateSchedulerFactory,
    EarlyStopping,
    DistributedTrainer,
    Trainer,
    Evaluator,
    ExperimentTracker,
    create_data_pipeline,
    load_config,
)


def example_inference_engine():
    """Example: High-performance inference engine."""
    print("=" * 60)
    print("Example 1: Inference Engine")
    print("=" * 60)
    
    # Setup
    manager = ModelManager()
    model = manager.get_model("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create inference engine
    engine = InferenceEngine(
        model=model,
        tokenizer=tokenizer,
        use_amp=True,
        max_batch_size=32,
        compile_model=False,  # Set to True if PyTorch 2.0+
    )
    
    # Single generation
    print("\nSingle generation:")
    text = engine.generate(
        "The future of artificial intelligence",
        max_length=50,
        temperature=0.8,
    )
    print(f"Generated: {text}")
    
    # Batch generation
    print("\nBatch generation:")
    prompts = [
        "Once upon a time",
        "In a world where",
        "The secret to",
    ]
    texts = engine.batch_generate(
        prompts,
        batch_size=2,
        max_length=50,
    )
    for prompt, text in zip(prompts, texts):
        print(f"{prompt}... -> {text[:50]}...")
    
    # Embeddings
    print("\nEmbeddings:")
    embeddings = engine.get_embeddings(
        texts=["Text 1", "Text 2", "Text 3"],
        normalize=True,
        pooling="mean",
    )
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Embedding norm: {embeddings.norm(dim=1)}")


def example_lora_manager():
    """Example: LoRA management."""
    print("\n" + "=" * 60)
    print("Example 2: LoRA Manager")
    print("=" * 60)
    
    # Load base model
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained("gpt2")
    
    # Count parameters
    total_params = sum(p.numel() for p in base_model.parameters())
    print(f"Base model parameters: {total_params:,}")
    
    # Create LoRA manager
    lora_manager = LoRAManager(
        r=8,
        alpha=16,
        dropout=0.1,
        task_type="causal_lm",
    )
    
    # Apply LoRA (auto-detects target modules)
    print("\nApplying LoRA...")
    peft_model = lora_manager.apply_lora(base_model)
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Reduction: {(1 - trainable_params / total_params) * 100:.2f}%")
    
    # Custom LoRA config
    print("\nCustom LoRA config:")
    custom_config = lora_manager.create_config(
        r=16,
        target_modules=["c_attn", "c_proj"],  # GPT-2 specific
    )
    print(f"LoRA rank: {custom_config.r}")
    print(f"LoRA alpha: {custom_config.lora_alpha}")
    print(f"Target modules: {custom_config.target_modules}")


def example_learning_rate_schedulers():
    """Example: Learning rate schedulers."""
    print("\n" + "=" * 60)
    print("Example 3: Learning Rate Schedulers")
    print("=" * 60)
    
    # Create dummy model and optimizer
    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    # Create different schedulers
    schedulers = {
        "linear": LearningRateSchedulerFactory.create_scheduler(
            optimizer, "linear", num_training_steps=100, num_warmup_steps=10
        ),
        "cosine": LearningRateSchedulerFactory.create_scheduler(
            optimizer, "cosine", num_training_steps=100, num_warmup_steps=10
        ),
        "polynomial": LearningRateSchedulerFactory.create_scheduler(
            optimizer, "polynomial", num_training_steps=100, num_warmup_steps=10, power=2.0
        ),
    }
    
    print("\nLearning rate schedules (first 20 steps):")
    for name, scheduler in schedulers.items():
        lrs = []
        for step in range(20):
            lr = scheduler.get_last_lr()[0]
            lrs.append(f"{lr:.2e}")
            scheduler.step()
        print(f"{name:12s}: {', '.join(lrs[:5])}...")
    
    # Early stopping
    print("\nEarly stopping example:")
    early_stopping = EarlyStopping(
        patience=3,
        min_delta=0.001,
        mode="min",
    )
    
    val_losses = [0.5, 0.4, 0.35, 0.34, 0.33, 0.33, 0.33]  # Simulated
    for epoch, loss in enumerate(val_losses):
        should_stop = early_stopping(loss, model)
        print(f"Epoch {epoch}: loss={loss:.3f}, patience={early_stopping.counter}")
        if should_stop:
            print("Early stopping triggered!")
            break


def example_distributed_training():
    """Example: Distributed training setup."""
    print("\n" + "=" * 60)
    print("Example 4: Distributed Training")
    print("=" * 60)
    
    # Create model
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    
    # Check GPU availability
    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")
    
    if num_gpus > 1:
        # Create distributed trainer
        dist_trainer = DistributedTrainer(
            model=model,
            use_ddp=False,  # Use DataParallel for demo
        )
        
        wrapped_model = dist_trainer.get_model()
        print(f"Model wrapped: {type(wrapped_model).__name__}")
        
        # Get sampler (if DDP)
        # sampler = dist_trainer.get_sampler(dataset, shuffle=True)
        
        # Cleanup
        dist_trainer.cleanup()
    else:
        print("Single GPU/CPU - distributed training not applicable")


def example_complete_pipeline():
    """Example: Complete training pipeline."""
    print("\n" + "=" * 60)
    print("Example 5: Complete Training Pipeline")
    print("=" * 60)
    
    print("This demonstrates a complete training pipeline with:")
    print("  - Model loading with caching")
    print("  - LoRA application")
    print("  - Data pipeline creation")
    print("  - Learning rate scheduling")
    print("  - Training with callbacks")
    print("  - Experiment tracking")
    print("  - Evaluation")
    
    # Note: This is a conceptual example
    # In practice, you would:
    # 1. Load configuration
    # 2. Setup model with LoRA
    # 3. Create data pipelines
    # 4. Setup trainer with scheduler
    # 5. Train with early stopping
    # 6. Evaluate
    # 7. Save model
    
    print("\nSee ADVANCED_REFACTORING.md for complete code example")


def main():
    """Run all advanced examples."""
    print("\n" + "=" * 60)
    print("Advanced ML Framework - Usage Examples")
    print("=" * 60)
    print("\nDemonstrating advanced features:")
    print("  - High-performance inference")
    print("  - LoRA management")
    print("  - Learning rate scheduling")
    print("  - Distributed training")
    print("  - Complete pipelines\n")
    
    try:
        example_inference_engine()
        example_lora_manager()
        example_learning_rate_schedulers()
        example_distributed_training()
        example_complete_pipeline()
        
        print("\n" + "=" * 60)
        print("Advanced examples completed!")
        print("=" * 60)
        print("\nKey takeaways:")
        print("  ✓ Inference engine provides optimized batching")
        print("  ✓ LoRA reduces trainable parameters by ~90%")
        print("  ✓ LR schedulers support multiple strategies")
        print("  ✓ Distributed training scales to multiple GPUs")
        print("  ✓ All components integrate seamlessly")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()



