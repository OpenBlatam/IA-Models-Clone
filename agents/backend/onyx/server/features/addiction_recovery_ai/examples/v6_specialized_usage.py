"""
Ultra Modular V6 - Specialized Components Usage Examples
Demonstrates the maximum specialization with individual component files
"""

import torch
import torch.nn as nn
from addiction_recovery_ai.core.layers.micro_modules import (
    # Initializers
    XavierInitializer,
    KaimingInitializer,
    InitializerFactory,
    # Compilers
    TorchCompileCompiler,
    CompilerFactory,
    # Optimizers
    MixedPrecisionOptimizer,
    PruningOptimizer,
    OptimizerFactory,
    # Quantizers
    DynamicQuantizer,
    QuantizerFactory,
    # Losses
    MSELoss,
    FocalLoss,
    LossFactory,
)


# Example 1: Model Initialization Pipeline
def example_initialization():
    """Demonstrate specialized initializers"""
    print("=" * 60)
    print("Example 1: Model Initialization")
    print("=" * 60)
    
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 10)
    )
    
    # Method 1: Direct use
    print("\n1. Direct Initializer Usage:")
    initializer = XavierInitializer(gain=1.0)
    initializer.initialize(model)
    print(f"   ✓ Model initialized with {initializer.name}")
    
    # Method 2: Factory pattern
    print("\n2. Factory Pattern:")
    initializer = InitializerFactory.create('kaiming', nonlinearity='relu')
    initializer.initialize(model)
    print(f"   ✓ Model initialized with {initializer.name}")
    
    # Method 3: Different strategies
    strategies = ['xavier', 'kaiming', 'orthogonal', 'normal']
    for strategy in strategies:
        model = nn.Sequential(nn.Linear(10, 5))
        initializer = InitializerFactory.create(strategy)
        initializer.initialize(model)
        print(f"   ✓ {strategy.capitalize()} initialization applied")


# Example 2: Model Compilation Pipeline
def example_compilation():
    """Demonstrate specialized compilers"""
    print("\n" + "=" * 60)
    print("Example 2: Model Compilation")
    print("=" * 60)
    
    model = nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 10)
    )
    
    # Method 1: Direct use
    print("\n1. Direct Compiler Usage:")
    compiler = TorchCompileCompiler(mode="reduce-overhead")
    compiled = compiler.compile(model)
    print(f"   ✓ Model compiled with {compiler.name}")
    
    # Method 2: Factory pattern
    print("\n2. Factory Pattern:")
    compiler = CompilerFactory.create('torch_compile', mode="max-autotune")
    compiled = compiler.compile(model)
    print(f"   ✓ Model compiled with factory")
    
    # Method 3: Optimize for inference
    print("\n3. Inference Optimization:")
    compiler = CompilerFactory.create('optimize')
    optimized = compiler.compile(model)
    print(f"   ✓ Model optimized for inference")


# Example 3: Model Optimization Pipeline
def example_optimization():
    """Demonstrate specialized optimizers"""
    print("\n" + "=" * 60)
    print("Example 3: Model Optimization")
    print("=" * 60)
    
    model = nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 10)
    )
    
    # Method 1: Mixed precision
    print("\n1. Mixed Precision:")
    if torch.cuda.is_available():
        model = model.cuda()
        optimizer = MixedPrecisionOptimizer()
        optimized = optimizer.optimize(model)
        print(f"   ✓ Model optimized with {optimizer.name}")
    else:
        print("   ⚠ CUDA not available for mixed precision")
    
    # Method 2: Pruning
    print("\n2. Weight Pruning:")
    optimizer = PruningOptimizer(pruning_ratio=0.1)
    pruned = optimizer.optimize(model)
    print(f"   ✓ Model pruned with {optimizer.name}")
    
    # Method 3: Factory pattern
    print("\n3. Factory Pattern:")
    optimizer = OptimizerFactory.create('pruning', pruning_ratio=0.2)
    optimized = optimizer.optimize(model)
    print(f"   ✓ Model optimized with factory")


# Example 4: Model Quantization Pipeline
def example_quantization():
    """Demonstrate specialized quantizers"""
    print("\n" + "=" * 60)
    print("Example 4: Model Quantization")
    print("=" * 60)
    
    model = nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 10)
    )
    model.eval()
    
    # Method 1: Dynamic quantization
    print("\n1. Dynamic Quantization:")
    quantizer = DynamicQuantizer(dtype=torch.qint8)
    quantized = quantizer.quantize(model)
    print(f"   ✓ Model quantized with {quantizer.name}")
    
    # Method 2: Factory pattern
    print("\n2. Factory Pattern:")
    quantizer = QuantizerFactory.create('dynamic', dtype=torch.qint8)
    quantized = quantizer.quantize(model)
    print(f"   ✓ Model quantized with factory")


# Example 5: Loss Functions Pipeline
def example_losses():
    """Demonstrate specialized loss functions"""
    print("\n" + "=" * 60)
    print("Example 5: Loss Functions")
    print("=" * 60)
    
    # Create dummy predictions and targets
    predictions = torch.randn(32, 10)
    targets = torch.randint(0, 10, (32,))
    
    # Method 1: Direct use
    print("\n1. Direct Loss Usage:")
    loss_fn = MSELoss()
    loss = loss_fn.compute(predictions, targets)
    print(f"   ✓ {loss_fn.name}: {loss.item():.4f}")
    
    # Method 2: Focal loss
    print("\n2. Focal Loss (for imbalanced datasets):")
    focal_loss = FocalLoss(alpha=1.0, gamma=2.0)
    # For classification, we need logits
    logits = torch.randn(32, 10)
    loss = focal_loss.compute(logits, targets)
    print(f"   ✓ {focal_loss.name}: {loss.item():.4f}")
    
    # Method 3: Factory pattern
    print("\n3. Factory Pattern:")
    loss_types = ['mse', 'mae', 'bce', 'ce', 'smooth_l1']
    for loss_type in loss_types:
        try:
            loss_fn = LossFactory.create(loss_type)
            if loss_type in ['bce', 'ce']:
                # For classification losses
                logits = torch.randn(32, 10)
                loss = loss_fn.compute(logits, targets)
            else:
                # For regression losses
                targets_reg = torch.randn(32, 10)
                loss = loss_fn.compute(predictions, targets_reg)
            print(f"   ✓ {loss_type.upper()}: {loss.item():.4f}")
        except Exception as e:
            print(f"   ⚠ {loss_type}: {e}")


# Example 6: Complete Pipeline
def example_complete_pipeline():
    """Demonstrate complete model preparation pipeline"""
    print("\n" + "=" * 60)
    print("Example 6: Complete Pipeline")
    print("=" * 60)
    
    # Step 1: Create model
    model = nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 10)
    )
    print("\n1. Model Created")
    
    # Step 2: Initialize
    initializer = InitializerFactory.create('xavier')
    initializer.initialize(model)
    print("2. Model Initialized")
    
    # Step 3: Compile
    compiler = CompilerFactory.create('torch_compile', mode="reduce-overhead")
    compiled = compiler.compile(model)
    print("3. Model Compiled")
    
    # Step 4: Optimize (if CUDA available)
    if torch.cuda.is_available():
        compiled = compiled.cuda()
        optimizer = OptimizerFactory.create('mixed_precision')
        optimized = optimizer.optimize(compiled)
        print("4. Model Optimized (Mixed Precision)")
    
    # Step 5: Quantize (optional)
    optimized.eval()
    quantizer = QuantizerFactory.create('dynamic')
    quantized = quantizer.quantize(optimized)
    print("5. Model Quantized")
    
    # Step 6: Loss function
    loss_fn = LossFactory.create('mse')
    print("6. Loss Function Ready")
    
    print("\n✓ Complete pipeline executed successfully!")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Ultra Modular V6 - Specialized Components Examples")
    print("=" * 60)
    
    # Run examples
    example_initialization()
    example_compilation()
    example_optimization()
    example_quantization()
    example_losses()
    example_complete_pipeline()
    
    print("\n" + "=" * 60)
    print("All Examples Completed!")
    print("=" * 60)



