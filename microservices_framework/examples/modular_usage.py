"""
Modular Usage Examples
Demonstrates the highly modular architecture with builders and factories.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from shared.ml import (
    # Core interfaces and builders
    TrainingPipelineBuilder,
    InferencePipelineBuilder,
    ModelOptimizationBuilder,
    # Factories
    OptimizerFactory,
    LossFunctionFactory,
    DeviceFactory,
    ComponentFactory,
    # Custom losses
    FocalLoss,
    LabelSmoothingLoss,
    # Other components
    create_data_pipeline,
    ModelManager,
)


def example_builder_pattern():
    """Example: Using builders for pipeline construction."""
    print("=" * 60)
    print("Example 1: Builder Pattern")
    print("=" * 60)
    
    # Load model
    manager = ModelManager()
    model = manager.get_model("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create sample data
    texts = ["Sample text"] * 10
    data_loaders = create_data_pipeline(
        texts=texts,
        tokenizer=tokenizer,
        max_length=128,
        batch_size=2,
    )
    
    # Build training pipeline using builder
    print("\nBuilding training pipeline...")
    trainer = (
        TrainingPipelineBuilder()
        .with_model(model)
        .with_data_loaders(data_loaders["train"], data_loaders["val"])
        .with_optimizer("adamw", learning_rate=5e-5, weight_decay=0.01)
        .with_loss_function("cross_entropy", ignore_index=-100)
        .with_scheduler("cosine", num_training_steps=100, num_warmup_steps=10)
        .with_trainer_config(use_amp=True, max_grad_norm=1.0)
        .build()
    )
    
    print("Training pipeline built successfully!")
    print(f"Trainer type: {type(trainer).__name__}")
    
    # Build inference pipeline
    print("\nBuilding inference pipeline...")
    inference_engine = (
        InferencePipelineBuilder()
        .with_model(model)
        .with_tokenizer(tokenizer)
        .with_config(use_amp=True, max_batch_size=32, compile_model=False)
        .build()
    )
    
    print("Inference pipeline built successfully!")
    print(f"Inference engine type: {type(inference_engine).__name__}")


def example_factory_pattern():
    """Example: Using factories for component creation."""
    print("\n" + "=" * 60)
    print("Example 2: Factory Pattern")
    print("=" * 60)
    
    # Create dummy model for examples
    model = torch.nn.Linear(10, 1)
    
    # Create optimizers using factory
    print("\nCreating optimizers:")
    optimizers = {
        "adam": OptimizerFactory.create("adam", model, learning_rate=1e-3),
        "adamw": OptimizerFactory.create("adamw", model, learning_rate=1e-3),
        "sgd": OptimizerFactory.create("sgd", model, learning_rate=1e-3, momentum=0.9),
    }
    
    for name, opt in optimizers.items():
        print(f"  {name}: {type(opt).__name__}")
    
    # Create loss functions using factory
    print("\nCreating loss functions:")
    losses = {
        "cross_entropy": LossFunctionFactory.create("cross_entropy"),
        "mse": LossFunctionFactory.create("mse"),
        "focal": LossFunctionFactory.create("focal", alpha=1.0, gamma=2.0),
    }
    
    for name, loss in losses.items():
        print(f"  {name}: {type(loss).__name__}")
    
    # Get device using factory
    print("\nDevice management:")
    device = DeviceFactory.get_optimal_device()
    print(f"  Optimal device: {device}")
    
    device_cuda = DeviceFactory.get_device("cuda")
    print(f"  CUDA device: {device_cuda}")


def example_model_optimization_builder():
    """Example: Using optimization builder."""
    print("\n" + "=" * 60)
    print("Example 3: Model Optimization Builder")
    print("=" * 60)
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    
    print("Original model parameters:", sum(p.numel() for p in model.parameters()))
    
    # Build optimized model
    print("\nBuilding optimized model with LoRA...")
    optimized_model = (
        ModelOptimizationBuilder()
        .with_model(model)
        .add_lora(r=8, alpha=16, task_type="causal_lm")
        .build()
    )
    
    trainable_params = sum(p.numel() for p in optimized_model.parameters() if p.requires_grad)
    print(f"Trainable parameters after LoRA: {trainable_params:,}")
    print(f"Reduction: {(1 - trainable_params / sum(p.numel() for p in model.parameters())) * 100:.2f}%")


def example_custom_losses():
    """Example: Using custom loss functions."""
    print("\n" + "=" * 60)
    print("Example 4: Custom Loss Functions")
    print("=" * 60)
    
    # Create sample data
    batch_size = 4
    num_classes = 10
    logits = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    # Focal Loss
    focal_loss = FocalLoss(alpha=1.0, gamma=2.0)
    focal_value = focal_loss(logits, targets)
    print(f"Focal Loss: {focal_value.item():.4f}")
    
    # Label Smoothing Loss
    label_smoothing_loss = LabelSmoothingLoss(num_classes=num_classes, smoothing=0.1)
    smoothing_value = label_smoothing_loss(logits, targets)
    print(f"Label Smoothing Loss: {smoothing_value.item():.4f}")
    
    # Standard Cross Entropy for comparison
    ce_loss = torch.nn.CrossEntropyLoss()
    ce_value = ce_loss(logits, targets)
    print(f"Cross Entropy Loss: {ce_value.item():.4f}")


def example_component_factory():
    """Example: Using component factory."""
    print("\n" + "=" * 60)
    print("Example 5: Component Factory")
    print("=" * 60)
    
    factory = ComponentFactory()
    
    # Create quantizer
    print("Creating quantizer...")
    quantizer = factory.create_quantizer("int8")
    print(f"Quantizer type: {type(quantizer).__name__}")
    
    # Create profiler
    print("\nCreating profiler...")
    model = torch.nn.Linear(10, 1)
    profiler = factory.create_profiler(model=model, device="cpu")
    print(f"Profiler type: {type(profiler).__name__}")


def example_fluent_interface():
    """Example: Fluent interface for complex pipelines."""
    print("\n" + "=" * 60)
    print("Example 6: Fluent Interface")
    print("=" * 60)
    
    # This demonstrates how the builder pattern enables fluent, readable code
    print("""
    # Fluent interface example:
    
    trainer = (
        TrainingPipelineBuilder()
        .with_model(model)
        .with_data_loaders(train_loader, val_loader)
        .with_optimizer("adamw", learning_rate=5e-5)
        .with_loss_function("cross_entropy")
        .with_scheduler("cosine", num_training_steps=1000)
        .with_trainer_config(use_amp=True)
        .build()
    )
    
    # Clean, readable, and modular!
    """)


def main():
    """Run all modular examples."""
    print("\n" + "=" * 60)
    print("Modular ML Framework - Usage Examples")
    print("=" * 60)
    print("\nDemonstrating highly modular architecture:")
    print("  - Builder pattern for pipelines")
    print("  - Factory pattern for components")
    print("  - Custom loss functions")
    print("  - Fluent interfaces")
    print("  - Separation of concerns\n")
    
    try:
        example_builder_pattern()
        example_factory_pattern()
        example_model_optimization_builder()
        example_custom_losses()
        example_component_factory()
        example_fluent_interface()
        
        print("\n" + "=" * 60)
        print("Modular examples completed!")
        print("=" * 60)
        print("\nKey benefits of modular architecture:")
        print("  ✓ Separation of concerns")
        print("  ✓ Reusable components")
        print("  ✓ Easy to test")
        print("  ✓ Easy to extend")
        print("  ✓ Clean, readable code")
        print("  ✓ Type-safe interfaces")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()



