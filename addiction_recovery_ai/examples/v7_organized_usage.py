"""
Ultra Modular V7 - Organized Subdirectories Usage Examples
Demonstrates the organized structure with subdirectories
"""

import torch
import torch.nn as nn
import torch.optim as optim

# Import from organized subdirectories
from addiction_recovery_ai.core.layers.micro_modules.data import (
    NormalizerFactory,
    TokenizerFactory,
    PadderFactory,
    AugmenterFactory
)

from addiction_recovery_ai.core.layers.micro_modules.model import (
    InitializerFactory,
    CompilerFactory,
    OptimizerFactory,
    QuantizerFactory
)

from addiction_recovery_ai.core.layers.micro_modules.training import (
    LossFactory,
    GradientManagerFactory,
    LRManagerFactory,
    CheckpointManagerFactory
)

from addiction_recovery_ai.core.layers.micro_modules.inference import (
    BatchProcessor,
    CacheManager
)


# Example 1: Data Processing Pipeline
def example_data_processing():
    """Demonstrate data processing from data subdirectory"""
    print("=" * 60)
    print("Example 1: Data Processing")
    print("=" * 60)
    
    # Normalization
    normalizer = NormalizerFactory.create('standard')
    data = torch.randn(100, 10)
    normalized = normalizer.normalize(data)
    print(f"✓ Normalized data shape: {normalized.shape}")
    
    # Tokenization
    tokenizer = TokenizerFactory.create('simple')
    text = "Hello world"
    tokens = tokenizer.tokenize(text)
    print(f"✓ Tokenized: {tokens}")
    
    # Padding
    padder = PadderFactory.create('zero', target_length=20)
    sequence = torch.tensor([1, 2, 3])
    padded = padder.pad(sequence)
    print(f"✓ Padded sequence: {padded.shape}")


# Example 2: Model Optimization Pipeline
def example_model_optimization():
    """Demonstrate model optimization from model subdirectory"""
    print("\n" + "=" * 60)
    print("Example 2: Model Optimization")
    print("=" * 60)
    
    model = nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 10)
    )
    
    # Initialize
    initializer = InitializerFactory.create('xavier')
    initializer.initialize(model)
    print("✓ Model initialized")
    
    # Compile
    compiler = CompilerFactory.create('torch_compile', mode="reduce-overhead")
    compiled = compiler.compile(model)
    print("✓ Model compiled")
    
    # Optimize
    if torch.cuda.is_available():
        model = model.cuda()
        optimizer = OptimizerFactory.create('mixed_precision')
        optimized = optimizer.optimize(model)
        print("✓ Model optimized (mixed precision)")


# Example 3: Training Pipeline
def example_training_pipeline():
    """Demonstrate training components from training subdirectory"""
    print("\n" + "=" * 60)
    print("Example 3: Training Pipeline")
    print("=" * 60)
    
    model = nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 10)
    )
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Loss function
    loss_fn = LossFactory.create('mse')
    predictions = torch.randn(32, 10)
    targets = torch.randn(32, 10)
    loss = loss_fn.compute(predictions, targets)
    print(f"✓ Loss computed: {loss.item():.4f}")
    
    # Gradient management
    loss.backward()
    grad_manager = GradientManagerFactory.create('clip', max_norm=1.0)
    result = grad_manager.manage(model, loss)
    print(f"✓ Gradients clipped: {result['clipped']}")
    
    # Learning rate scheduler
    lr_manager = LRManagerFactory.create('cosine')
    scheduler = lr_manager.get_scheduler(optimizer, T_max=10)
    print(f"✓ LR scheduler created: {scheduler}")
    
    # Checkpoint management
    checkpoint_manager = CheckpointManagerFactory.create('full')
    checkpoint_manager.save(model, 'checkpoint.pt', optimizer=optimizer, epoch=1)
    print("✓ Checkpoint saved")


# Example 4: Complete Workflow
def example_complete_workflow():
    """Demonstrate complete workflow using all subdirectories"""
    print("\n" + "=" * 60)
    print("Example 4: Complete Workflow")
    print("=" * 60)
    
    # 1. Data Processing
    print("\n1. Data Processing:")
    normalizer = NormalizerFactory.create('standard')
    data = torch.randn(100, 10)
    normalized = normalizer.normalize(data)
    print("   ✓ Data normalized")
    
    # 2. Model Setup
    print("\n2. Model Setup:")
    model = nn.Sequential(
        nn.Linear(10, 50),
        nn.ReLU(),
        nn.Linear(50, 1)
    )
    
    initializer = InitializerFactory.create('xavier')
    initializer.initialize(model)
    print("   ✓ Model initialized")
    
    compiler = CompilerFactory.create('torch_compile')
    compiled = compiler.compile(model)
    print("   ✓ Model compiled")
    
    # 3. Training Setup
    print("\n3. Training Setup:")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    loss_fn = LossFactory.create('mse')
    print("   ✓ Loss function created")
    
    lr_manager = LRManagerFactory.create('step')
    scheduler = lr_manager.get_scheduler(optimizer, step_size=30, gamma=0.1)
    print("   ✓ LR scheduler created")
    
    checkpoint_manager = CheckpointManagerFactory.create('best', metric_name='loss', mode='min')
    print("   ✓ Checkpoint manager created")
    
    # 4. Training Step
    print("\n4. Training Step:")
    predictions = model(normalized[:32])
    targets = torch.randn(32, 1)
    loss = loss_fn.compute(predictions, targets)
    print(f"   ✓ Loss: {loss.item():.4f}")
    
    loss.backward()
    grad_manager = GradientManagerFactory.create('clip', max_norm=1.0)
    grad_manager.manage(model, loss)
    print("   ✓ Gradients clipped")
    
    optimizer.step()
    optimizer.zero_grad()
    print("   ✓ Optimizer step")
    
    # 5. Checkpoint
    print("\n5. Checkpoint:")
    checkpoint_manager.save(model, 'best_model.pt', metric_value=loss.item())
    print("   ✓ Best model saved")
    
    print("\n✓ Complete workflow executed successfully!")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Ultra Modular V7 - Organized Subdirectories Examples")
    print("=" * 60)
    
    # Run examples
    example_data_processing()
    example_model_optimization()
    example_training_pipeline()
    example_complete_workflow()
    
    print("\n" + "=" * 60)
    print("All Examples Completed!")
    print("=" * 60)



