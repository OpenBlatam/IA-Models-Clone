"""
Example: Using Ultra-Granular Micro-Modules
Demonstrates the new micro-module architecture with single-responsibility components
"""

import torch
import torch.nn as nn

# Import micro-modules
from addiction_recovery_ai.core.layers.micro_modules import (
    # Data Processors
    StandardNormalizer,
    MinMaxNormalizer,
    ZeroPadder,
    TensorValidator,
    # Model Components
    ModelInitializer,
    ModelCompiler,
    ModelOptimizer,
    # Training Components
    LossCalculator,
    GradientManager,
    LearningRateManager,
    CheckpointManager,
    # Inference Components
    BatchProcessor,
    CacheManager,
    OutputFormatter,
    PostProcessor,
)


# ============================================================================
# Example 1: Data Processing with Micro-Modules
# ============================================================================

def example_data_processing():
    """Example of granular data processing"""
    print("\n=== Example 1: Data Processing ===")
    
    # Create individual processors
    normalizer = StandardNormalizer()
    padder = ZeroPadder(pad_value=0)
    validator = TensorValidator(check_nan=True, check_inf=True)
    
    # Process data step by step
    data = torch.randn(10) * 5 + 2  # Random data with mean ~2
    
    print(f"Original data mean: {data.mean():.3f}, std: {data.std():.3f}")
    
    # Normalize
    normalized = normalizer.normalize(data)
    print(f"Normalized mean: {normalized.mean():.3f}, std: {normalized.std():.3f}")
    
    # Pad
    padded = padder.pad(normalized, target_length=20)
    print(f"Padded shape: {padded.shape}")
    
    # Validate
    is_valid = validator.validate(padded)
    print(f"Data is valid: {is_valid}")


# ============================================================================
# Example 2: Model Management with Micro-Modules
# ============================================================================

def example_model_management():
    """Example of granular model management"""
    print("\n=== Example 2: Model Management ===")
    
    # Create simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 1)
        
        def forward(self, x):
            return self.linear(x)
    
    model = SimpleModel()
    
    # Initialize weights
    ModelInitializer.initialize(model, method='xavier')
    print("Model initialized with Xavier method")
    
    # Compile for faster inference
    compiled_model = ModelCompiler.compile(model, mode="reduce-overhead")
    print("Model compiled")
    
    # Optimize for inference
    optimized_model = ModelOptimizer.enable_mixed_precision(compiled_model)
    print("Model optimized for mixed precision")


# ============================================================================
# Example 3: Training with Micro-Modules
# ============================================================================

def example_training_components():
    """Example of granular training components"""
    print("\n=== Example 3: Training Components ===")
    
    # Create model and optimizer
    model = nn.Linear(10, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Loss calculator
    criterion = LossCalculator.create('mse')
    print(f"Loss function created: {type(criterion).__name__}")
    
    # Simulate training step
    predictions = model(torch.randn(5, 10))
    targets = torch.randn(5, 1)
    loss = LossCalculator.calculate(predictions, targets, criterion)
    
    # Check loss
    loss_stats = LossCalculator.check_loss(loss)
    print(f"Loss: {loss.item():.4f}, Valid: {loss_stats['is_valid']}")
    
    # Gradient management
    loss.backward()
    grad_norm = GradientManager.clip_gradients(model, max_norm=1.0)
    print(f"Gradient norm after clipping: {grad_norm:.4f}")
    
    # Learning rate management
    scheduler = LearningRateManager.create_scheduler(
        optimizer,
        'reduce_on_plateau',
        patience=5,
        factor=0.5
    )
    current_lr = LearningRateManager.get_current_lr(optimizer)
    print(f"Current learning rate: {current_lr}")


# ============================================================================
# Example 4: Inference with Micro-Modules
# ============================================================================

def example_inference_components():
    """Example of granular inference components"""
    print("\n=== Example 4: Inference Components ===")
    
    # Create model
    model = nn.Linear(10, 3)
    model.eval()
    
    # Batch processor
    processor = BatchProcessor(batch_size=4)
    inputs = [torch.randn(10) for _ in range(10)]
    
    def process_batch(batch):
        batch_tensor = torch.stack(batch)
        return model(batch_tensor)
    
    results = processor.process(inputs, process_batch)
    print(f"Processed {len(inputs)} inputs in batches, got {len(results)} results")
    
    # Cache manager
    cache = CacheManager(max_size=100)
    cache_key = "input_1"
    
    cached_result = cache.get(cache_key)
    if cached_result is None:
        result = model(inputs[0].unsqueeze(0))
        cache.set(cache_key, result)
        print("Result computed and cached")
    else:
        print("Result retrieved from cache")
    
    cache_stats = cache.get_stats()
    print(f"Cache stats: {cache_stats}")
    
    # Output formatter
    output = model(inputs[0].unsqueeze(0))
    numpy_output = OutputFormatter.to_numpy(output)
    list_output = OutputFormatter.to_list(output)
    print(f"Numpy output shape: {numpy_output.shape}")
    print(f"List output length: {len(list_output)}")
    
    # Post-processor
    post_processor = PostProcessor()
    post_processor.add_processor(PostProcessor.threshold(0.5))
    post_processor.add_processor(PostProcessor.argmax())
    
    processed = post_processor.process(output)
    print(f"Post-processed output: {processed}")


# ============================================================================
# Example 5: Complete Workflow with Micro-Modules
# ============================================================================

def example_complete_workflow():
    """Complete workflow using micro-modules"""
    print("\n=== Example 5: Complete Workflow ===")
    
    # 1. Data preparation
    normalizer = StandardNormalizer()
    padder = ZeroPadder()
    validator = TensorValidator()
    
    data = torch.randn(10)
    data = normalizer.normalize(data)
    data = padder.pad(data, 20)
    assert validator.validate(data)
    print("✓ Data prepared")
    
    # 2. Model setup
    model = nn.Linear(20, 1)
    ModelInitializer.initialize(model, 'kaiming')
    compiled_model = ModelCompiler.compile(model)
    print("✓ Model setup complete")
    
    # 3. Training setup
    optimizer = torch.optim.Adam(model.parameters())
    criterion = LossCalculator.create('mse')
    scheduler = LearningRateManager.create_scheduler(optimizer, 'step', step_size=10)
    checkpoint_mgr = CheckpointManager("checkpoints")
    print("✓ Training setup complete")
    
    # 4. Inference setup
    inference_processor = BatchProcessor(batch_size=8)
    cache = CacheManager()
    formatter = OutputFormatter()
    print("✓ Inference setup complete")
    
    print("\nAll micro-modules integrated successfully!")


# ============================================================================
# Example 6: Composing Micro-Modules
# ============================================================================

def example_composition():
    """Example of composing micro-modules"""
    print("\n=== Example 6: Composing Micro-Modules ===")
    
    # Create a composed data processor
    class ComposedProcessor:
        def __init__(self):
            self.normalizer = StandardNormalizer()
            self.padder = ZeroPadder()
            self.validator = TensorValidator()
        
        def process(self, data, target_length=20):
            # Normalize
            normalized = self.normalizer.normalize(data)
            
            # Pad
            padded = self.padder.pad(normalized, target_length)
            
            # Validate
            if not self.validator.validate(padded):
                raise ValueError("Invalid data after processing")
            
            return padded
    
    # Use composed processor
    processor = ComposedProcessor()
    data = torch.randn(10)
    processed = processor.process(data)
    print(f"Composed processing: {data.shape} -> {processed.shape}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Ultra-Granular Micro-Modules Examples")
    print("=" * 60)
    
    try:
        example_data_processing()
        example_model_management()
        example_training_components()
        example_inference_components()
        example_complete_workflow()
        example_composition()
        
        print("\n" + "=" * 60)
        print("All micro-module examples completed!")
        print("=" * 60)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()



