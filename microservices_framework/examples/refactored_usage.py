"""
Example: Using the Refactored ML Framework
Demonstrates best practices with the new modular structure.
"""

import torch
from transformers import AutoTokenizer
from shared.ml import (
    ModelManager,
    Trainer,
    Evaluator,
    ExperimentTracker,
    create_data_pipeline,
    load_config,
    ModelConfig,
    TrainingConfig,
    MLServiceSettings,
)


def example_configuration():
    """Example: Using YAML configuration."""
    print("=" * 60)
    print("Example 1: Configuration Management")
    print("=" * 60)
    
    # Load configuration from YAML
    config = load_config("configs/llm_config.yaml")
    print(f"Default model: {config['model']['default_model']}")
    print(f"Use FP16: {config['model']['use_fp16']}")
    
    # Use Pydantic models
    model_config = ModelConfig(**config["model"])
    print(f"Model config: {model_config}")
    
    # Use settings
    settings = MLServiceSettings()
    print(f"Device: {settings.device}")
    print(f"Use FP16: {settings.use_fp16}")


def example_model_management():
    """Example: Using ModelManager for model loading and caching."""
    print("\n" + "=" * 60)
    print("Example 2: Model Management")
    print("=" * 60)
    
    # Create model manager
    manager = ModelManager(
        cache_size=5,
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_fp16=True,
    )
    
    # Load model (will be cached)
    print("Loading model...")
    model = manager.get_model("gpt2")
    
    # Get model info
    from shared.ml import get_model_summary
    summary = get_model_summary(model)
    print(f"Model parameters: {summary['total_parameters']:,}")
    print(f"Model size: {summary['model_size_mb']:.2f} MB")
    
    # Load again (from cache)
    print("\nLoading same model again (from cache)...")
    model2 = manager.get_model("gpt2")
    print("Model loaded from cache!")


def example_data_pipeline():
    """Example: Using functional data pipeline."""
    print("\n" + "=" * 60)
    print("Example 3: Functional Data Pipeline")
    print("=" * 60)
    
    # Sample texts
    texts = [
        "The future of artificial intelligence",
        "Machine learning is transforming industries",
        "Deep learning models are becoming more powerful",
    ] * 10  # Repeat for demo
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create data pipeline
    data_loaders = create_data_pipeline(
        texts=texts,
        tokenizer=tokenizer,
        max_length=128,
        batch_size=4,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
    )
    
    print(f"Train batches: {len(data_loaders['train'])}")
    print(f"Val batches: {len(data_loaders['val'])}")
    print(f"Test batches: {len(data_loaders['test'])}")
    
    # Get a sample batch
    sample_batch = next(iter(data_loaders["train"]))
    print(f"\nSample batch shape: {sample_batch['input_ids'].shape}")


def example_training():
    """Example: Using Trainer for model training."""
    print("\n" + "=" * 60)
    print("Example 4: Training with Trainer")
    print("=" * 60)
    
    # This is a simplified example - in practice, you'd load real data
    print("Note: This is a conceptual example")
    print("Trainer features:")
    print("  - Mixed precision training (AMP)")
    print("  - Gradient clipping")
    print("  - Gradient accumulation")
    print("  - NaN/Inf detection")
    print("  - Automatic checkpointing")
    print("  - Validation during training")
    
    # Example trainer setup (commented out as it requires actual model/data)
    """
    from shared.ml import Trainer
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        use_amp=True,
        max_grad_norm=1.0,
        gradient_accumulation_steps=4,
        save_dir="./checkpoints",
    )
    
    trainer.train(num_epochs=3, save_best=True)
    """


def example_evaluation():
    """Example: Using Evaluator for model evaluation."""
    print("\n" + "=" * 60)
    print("Example 5: Evaluation with Evaluator")
    print("=" * 60)
    
    print("Evaluator features:")
    print("  - Multiple metrics (accuracy, precision, recall, F1)")
    print("  - Perplexity for language models")
    print("  - Confusion matrices")
    print("  - Proper validation procedures")
    
    # Example evaluator setup (commented out as it requires actual model/data)
    """
    from shared.ml import Evaluator
    
    evaluator = Evaluator(model=model, use_amp=True)
    metrics = evaluator.evaluate(
        data_loader=val_loader,
        metrics=["loss", "accuracy", "f1"]
    )
    print(f"Validation metrics: {metrics}")
    
    perplexity = evaluator.compute_perplexity(val_loader)
    print(f"Perplexity: {perplexity:.2f}")
    """


def example_experiment_tracking():
    """Example: Using ExperimentTracker."""
    print("\n" + "=" * 60)
    print("Example 6: Experiment Tracking")
    print("=" * 60)
    
    # Initialize tracker
    tracker = ExperimentTracker(
        use_wandb=False,  # Set to True if you have W&B configured
        use_tensorboard=True,
        project_name="ml-experiments",
        run_name="example-run",
        log_dir="./logs/tensorboard",
    )
    
    # Log metrics
    tracker.log({"loss": 0.5, "accuracy": 0.9}, step=100)
    tracker.log({"loss": 0.4, "accuracy": 0.92}, step=200)
    
    # Log hyperparameters
    tracker.log_hyperparameters({
        "learning_rate": 5e-5,
        "batch_size": 32,
        "num_epochs": 3,
    })
    
    print("Metrics logged to TensorBoard")
    print("View with: tensorboard --logdir=./logs/tensorboard")
    
    # Finish tracking
    tracker.finish()


def example_error_handling():
    """Example: Using custom error handling."""
    print("\n" + "=" * 60)
    print("Example 7: Error Handling")
    print("=" * 60)
    
    from shared.ml.errors import (
        ModelLoadError,
        InferenceError,
        TrainingError,
    )
    
    print("Custom exceptions available:")
    print("  - ModelLoadError: Model loading failures")
    print("  - InferenceError: Inference failures")
    print("  - TrainingError: Training failures")
    print("  - ConfigurationError: Configuration issues")
    print("  - DataError: Data processing failures")
    print("  - GPUError: GPU operation failures")
    
    # Example usage
    try:
        # This would raise ModelLoadError if model doesn't exist
        manager = ModelManager()
        model = manager.get_model("non-existent-model")
    except Exception as e:
        print(f"\nCaught error: {type(e).__name__}")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Refactored ML Framework - Usage Examples")
    print("=" * 60)
    print("\nThis demonstrates the new modular structure following")
    print("deep learning best practices.\n")
    
    try:
        example_configuration()
        example_model_management()
        example_data_pipeline()
        example_training()
        example_evaluation()
        example_experiment_tracking()
        example_error_handling()
        
        print("\n" + "=" * 60)
        print("Examples completed!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Review REFACTORING_SUMMARY.md for details")
        print("2. Check configs/ for configuration examples")
        print("3. Integrate new modules into your services")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()



