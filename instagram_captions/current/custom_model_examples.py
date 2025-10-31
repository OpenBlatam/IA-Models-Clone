"""
Custom Model Architectures - Usage Examples
Comprehensive examples demonstrating the usage of custom nn.Module architectures
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import yaml
import logging
from pathlib import Path
import time
from tqdm import tqdm
import numpy as np

# Import our custom model architectures
from custom_model_architectures import (
    BaseModel,
    CustomTransformerModel,
    CustomCNNModel,
    CustomRNNModel,
    CNNTransformerHybrid,
    create_model_from_config,
    get_model_summary
)

# Import PyTorch framework system
from pytorch_primary_framework_system import (
    PyTorchPrimaryFrameworkSystem,
    PyTorchFrameworkConfig
)

# =============================================================================
# EXAMPLE 1: BASIC MODEL CREATION AND USAGE
# =============================================================================

def example_basic_model_creation():
    """Basic example of creating and using custom models"""
    
    print("=== Example 1: Basic Model Creation and Usage ===\n")
    
    # 1. Create Transformer model directly
    print("1. Creating Transformer Model Directly...")
    transformer_model = CustomTransformerModel(
        vocab_size=10000,
        d_model=256,
        n_heads=4,
        n_layers=4,
        d_ff=1024,
        max_seq_len=128,
        dropout=0.1,
        activation='gelu',
        norm_first=True,
        use_relative_position=False,
        tie_weights=True
    )
    
    print(get_model_summary(transformer_model))
    
    # 2. Create CNN model directly
    print("\n2. Creating CNN Model Directly...")
    cnn_model = CustomCNNModel(
        input_channels=3,
        num_classes=1000,
        base_channels=64,
        num_layers=5,
        use_batch_norm=True,
        use_dropout=True,
        activation='relu',
        architecture='resnet'
    )
    
    print(get_model_summary(cnn_model))
    
    # 3. Test forward pass
    print("\n3. Testing Forward Pass...")
    
    # Test transformer
    transformer_input = torch.randint(0, 10000, (2, 64))
    transformer_output = transformer_model(transformer_input)
    print(f"Transformer input shape: {transformer_input.shape}")
    print(f"Transformer output shape: {transformer_output.shape}")
    
    # Test CNN
    cnn_input = torch.randn(2, 3, 224, 224)
    cnn_output = cnn_model(cnn_input)
    print(f"CNN input shape: {cnn_input.shape}")
    print(f"CNN output shape: {cnn_output.shape}")
    
    print("\nBasic model creation example completed successfully!\n")

# =============================================================================
# EXAMPLE 2: MODEL CREATION FROM CONFIGURATION
# =============================================================================

def example_model_from_config():
    """Example of creating models from configuration files"""
    
    print("=== Example 2: Model Creation from Configuration ===\n")
    
    # 1. Load configuration file
    config_path = Path("custom_model_config.yaml")
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print("Configuration loaded successfully!")
        
        # 2. Create models from different configurations
        print("\n2. Creating Models from Configurations...")
        
        # Small transformer
        small_transformer = create_model_from_config(config['transformer_models']['small'])
        print(f"Small Transformer: {small_transformer.count_parameters():,} parameters")
        
        # Standard CNN
        standard_cnn = create_model_from_config(config['cnn_models']['standard'])
        print(f"Standard CNN: {standard_cnn.count_parameters():,} parameters")
        
        # Standard RNN
        standard_rnn = create_model_from_config(config['rnn_models']['standard'])
        print(f"Standard RNN: {standard_rnn.count_parameters():,} parameters")
        
        # Image captioning hybrid
        hybrid_model = create_model_from_config(config['hybrid_models']['image_captioning'])
        print(f"Hybrid Model: {hybrid_model.count_parameters():,} parameters")
        
        # 3. Test all models
        print("\n3. Testing All Models...")
        
        # Test small transformer
        transformer_input = torch.randint(0, 10000, (2, 64))
        transformer_output = small_transformer(transformer_input)
        print(f"Small Transformer output shape: {transformer_output.shape}")
        
        # Test standard CNN
        cnn_input = torch.randn(2, 3, 224, 224)
        cnn_output = standard_cnn(cnn_input)
        print(f"Standard CNN output shape: {cnn_output.shape}")
        
        # Test standard RNN
        rnn_input = torch.randn(2, 50, 512)
        rnn_output = standard_rnn(rnn_input)
        print(f"Standard RNN output shape: {rnn_output.shape}")
        
        # Test hybrid model
        hybrid_input = torch.randn(2, 3, 224, 224)
        hybrid_output = hybrid_model(hybrid_input)
        print(f"Hybrid model output shape: {hybrid_output.shape}")
        
    else:
        print("Configuration file not found. Please ensure 'custom_model_config.yaml' exists.")
    
    print("\nModel from configuration example completed successfully!\n")

# =============================================================================
# EXAMPLE 3: TASK-SPECIFIC MODEL CONFIGURATIONS
# =============================================================================

def example_task_specific_models():
    """Example of using task-specific model configurations"""
    
    print("=== Example 3: Task-Specific Model Configurations ===\n")
    
    # 1. Load configuration
    config_path = Path("custom_model_config.yaml")
    
    if not config_path.exists():
        print("Configuration file not found. Skipping this example.")
        return
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 2. Create Instagram caption generation model
    print("1. Creating Instagram Caption Generation Model...")
    instagram_config = config['task_specific']['instagram_caption']
    
    # Get base transformer config
    base_transformer_config = config['transformer_models'][instagram_config['config']].copy()
    
    # Apply customizations
    for key, value in instagram_config['customizations'].items():
        base_transformer_config[key] = value
    
    instagram_model = create_model_from_config(base_transformer_config)
    print(get_model_summary(instagram_model))
    
    # 3. Create image classification model
    print("\n2. Creating Image Classification Model...")
    image_class_config = config['task_specific']['image_classification']
    
    # Get base CNN config
    base_cnn_config = config['cnn_models'][image_class_config['config']].copy()
    
    # Apply customizations
    for key, value in image_class_config['customizations'].items():
        base_cnn_config[key] = value
    
    image_class_model = create_model_from_config(base_cnn_config)
    print(get_model_summary(image_class_model))
    
    # 4. Test task-specific models
    print("\n3. Testing Task-Specific Models...")
    
    # Test Instagram caption model
    caption_input = torch.randint(0, 25000, (2, 64))
    caption_output = instagram_model(caption_input)
    print(f"Instagram Caption Model output shape: {caption_output.shape}")
    
    # Test image classification model
    class_input = torch.randn(2, 3, 224, 224)
    class_output = image_class_model(class_input)
    print(f"Image Classification Model output shape: {class_output.shape}")
    
    print("\nTask-specific model example completed successfully!\n")

# =============================================================================
# EXAMPLE 4: MODEL TRAINING WITH PYTORCH FRAMEWORK
# =============================================================================

def example_model_training():
    """Example of training custom models with PyTorch framework system"""
    
    print("=== Example 4: Model Training with PyTorch Framework ===\n")
    
    # 1. Create PyTorch framework system
    print("1. Setting up PyTorch Framework System...")
    pytorch_config = PyTorchFrameworkConfig(
        use_cuda=True,
        use_mixed_precision=True,
        use_gradient_checkpointing=True,
        compile_model=True,
        use_tensorboard=True,
        batch_size=16,
        num_workers=2
    )
    
    pytorch_system = PyTorchPrimaryFrameworkSystem(pytorch_config)
    
    # 2. Create custom transformer model
    print("\n2. Creating Custom Transformer Model...")
    model = CustomTransformerModel(
        vocab_size=10000,
        d_model=256,
        n_heads=4,
        n_layers=4,
        d_ff=1024,
        max_seq_len=128,
        dropout=0.1
    )
    
    # 3. Optimize model for PyTorch
    print("\n3. Optimizing Model for PyTorch...")
    model = pytorch_system.optimize_model_for_pytorch(model)
    
    # 4. Create optimizer and scheduler
    print("\n4. Creating Optimizer and Scheduler...")
    optimizer = pytorch_system.create_pytorch_optimizer(model, "adamw", lr=1e-4)
    scheduler = pytorch_system.create_pytorch_scheduler(optimizer, "cosine", T_max=50)
    
    # 5. Create sample data
    print("\n5. Creating Sample Training Data...")
    batch_size = 16
    seq_len = 64
    vocab_size = 10000
    
    # Create random training data
    train_data = torch.randint(0, vocab_size, (1000, seq_len))
    train_targets = torch.randint(0, vocab_size, (1000, seq_len))
    
    # Create validation data
    val_data = torch.randint(0, vocab_size, (200, seq_len))
    val_targets = torch.randint(0, vocab_size, (200, seq_len))
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(train_data, train_targets)
    val_dataset = TensorDataset(val_data, val_targets)
    
    train_dataloader = pytorch_system.create_optimized_dataloader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_dataloader = pytorch_system.create_optimized_dataloader(
        val_dataset, batch_size=batch_size, shuffle=False
    )
    
    # 6. Loss function
    criterion = nn.CrossEntropyLoss()
    
    # 7. Training loop
    print("\n6. Starting Training Loop...")
    num_epochs = 5
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_idx, (data, targets) in enumerate(tqdm(train_dataloader, desc=f"Train Epoch {epoch + 1}")):
            step_result = pytorch_system.train_step_pytorch(
                model=model,
                data=data,
                targets=targets,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                grad_clip_norm=1.0,
                global_step=epoch * len(train_dataloader) + batch_idx
            )
            
            train_loss += step_result['loss']
        
        avg_train_loss = train_loss / len(train_dataloader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for data, targets in tqdm(val_dataloader, desc=f"Val Epoch {epoch + 1}"):
                val_result = pytorch_system.validate_step_pytorch(
                    model, data, targets, criterion
                )
                val_loss += val_result['loss']
        
        avg_val_loss = val_loss / len(val_dataloader)
        
        # Print metrics
        print(f"Epoch {epoch + 1}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Save checkpoint
        if (epoch + 1) % 2 == 0:
            checkpoint_path = pytorch_system.save_pytorch_checkpoint(
                model, optimizer, scheduler, epoch + 1, avg_val_loss,
                f"custom_model_epoch_{epoch + 1}.pth"
            )
            print(f"  Checkpoint saved: {checkpoint_path}")
    
    # 8. Export model
    print("\n7. Exporting Trained Model...")
    export_path = pytorch_system.export_pytorch_model(
        model, (1, 64), "torchscript", "custom_transformer_trained"
    )
    print(f"Model exported to: {export_path}")
    
    # 9. Cleanup
    pytorch_system.cleanup()
    
    print("\nModel training example completed successfully!\n")

# =============================================================================
# EXAMPLE 5: MODEL COMPARISON AND BENCHMARKING
# =============================================================================

def example_model_comparison():
    """Example of comparing different model architectures"""
    
    print("=== Example 5: Model Comparison and Benchmarking ===\n")
    
    # 1. Create different model types
    print("1. Creating Different Model Types...")
    
    # Small transformer
    transformer_model = CustomTransformerModel(
        vocab_size=10000,
        d_model=256,
        n_heads=4,
        n_layers=4,
        d_ff=1024,
        max_seq_len=128,
        dropout=0.1
    )
    
    # Standard CNN
    cnn_model = CustomCNNModel(
        input_channels=3,
        num_classes=1000,
        base_channels=64,
        num_layers=5,
        use_batch_norm=True,
        use_dropout=True
    )
    
    # Standard RNN
    rnn_model = CustomRNNModel(
        input_size=256,
        hidden_size=128,
        num_layers=3,
        num_classes=10,
        dropout=0.1,
        bidirectional=True
    )
    
    # 2. Benchmark models
    print("\n2. Benchmarking Models...")
    
    models = {
        'Transformer': (transformer_model, (1, 64)),
        'CNN': (cnn_model, (1, 3, 224, 224)),
        'RNN': (rnn_model, (1, 100, 256))
    }
    
    results = {}
    
    for model_name, (model, input_shape) in models.items():
        print(f"\nBenchmarking {model_name}...")
        
        # Move to device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Create input
        if model_name == 'Transformer':
            input_tensor = torch.randint(0, 10000, input_shape, device=device)
        else:
            input_tensor = torch.randn(input_shape, device=device)
        
        # Warmup
        model.eval()
        with torch.no_grad():
            for _ in range(10):
                _ = model(input_tensor)
        
        # Benchmark
        num_runs = 100
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(input_tensor)
        
        end_time = time.time()
        
        # Calculate metrics
        total_time = end_time - start_time
        avg_time = total_time / num_runs
        throughput = num_runs / total_time
        
        # Memory usage
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / (1024**3)
            memory_reserved = torch.cuda.memory_reserved() / (1024**3)
        else:
            memory_allocated = 0
            memory_reserved = 0
        
        results[model_name] = {
            'avg_inference_time_ms': avg_time * 1000,
            'throughput_inferences_per_sec': throughput,
            'memory_allocated_gb': memory_allocated,
            'memory_reserved_gb': memory_reserved,
            'total_parameters': model.count_parameters()
        }
        
        print(f"  Average inference time: {avg_time * 1000:.2f} ms")
        print(f"  Throughput: {throughput:.2f} inferences/sec")
        print(f"  Memory allocated: {memory_allocated:.2f} GB")
        print(f"  Total parameters: {model.count_parameters():,}")
    
    # 3. Print comparison summary
    print("\n3. Model Comparison Summary:")
    print("=" * 80)
    print(f"{'Model':<15} {'Time (ms)':<12} {'Throughput':<15} {'Memory (GB)':<15} {'Params':<12}")
    print("=" * 80)
    
    for model_name, result in results.items():
        print(f"{model_name:<15} {result['avg_inference_time_ms']:<12.2f} "
              f"{result['throughput_inferences_per_sec']:<15.2f} "
              f"{result['memory_allocated_gb']:<15.2f} "
              f"{result['total_parameters']:<12,}")
    
    print("=" * 80)
    
    # 4. Find best model for different criteria
    print("\n4. Best Models by Criteria:")
    
    fastest_model = min(results.items(), key=lambda x: x[1]['avg_inference_time_ms'])
    most_efficient_model = min(results.items(), key=lambda x: x[1]['memory_allocated_gb'])
    highest_throughput_model = max(results.items(), key=lambda x: x[1]['throughput_inferences_per_sec'])
    
    print(f"  Fastest inference: {fastest_model[0]} ({fastest_model[1]['avg_inference_time_ms']:.2f} ms)")
    print(f"  Most memory efficient: {most_efficient_model[0]} ({most_efficient_model[1]['memory_allocated_gb']:.2f} GB)")
    print(f"  Highest throughput: {highest_throughput_model[0]} ({highest_throughput_model[1]['throughput_inferences_per_sec']:.2f} inf/sec)")
    
    print("\nModel comparison example completed successfully!\n")

# =============================================================================
# EXAMPLE 6: ADVANCED MODEL FEATURES
# =============================================================================

def example_advanced_features():
    """Example of using advanced model features"""
    
    print("=== Example 6: Advanced Model Features ===\n")
    
    # 1. Create a large transformer model
    print("1. Creating Large Transformer Model...")
    large_transformer = CustomTransformerModel(
        vocab_size=50000,
        d_model=768,
        n_heads=12,
        n_layers=12,
        d_ff=3072,
        max_seq_len=512,
        dropout=0.1,
        use_relative_position=True
    )
    
    print(f"Large Transformer: {large_transformer.count_parameters():,} parameters")
    
    # 2. Test layer freezing/unfreezing
    print("\n2. Testing Layer Freezing/Unfreezing...")
    
    # Freeze embedding layers
    large_transformer.freeze_layers(['token_embedding', 'position_embedding'])
    
    # Check which parameters are frozen
    frozen_params = sum(1 for p in large_transformer.parameters() if not p.requires_grad)
    total_params = large_transformer.count_parameters()
    print(f"Frozen parameters: {frozen_params:,} / {total_params:,}")
    
    # Unfreeze all layers
    large_transformer.unfreeze_layers(['token_embedding', 'position_embedding'])
    
    # 3. Test with different input sizes
    print("\n3. Testing with Different Input Sizes...")
    
    input_sizes = [64, 128, 256, 512]
    
    for seq_len in input_sizes:
        input_tensor = torch.randint(0, 50000, (2, seq_len))
        try:
            with torch.no_grad():
                output = large_transformer(input_tensor)
            print(f"  Input size {seq_len}: Output shape {output.shape}")
        except Exception as e:
            print(f"  Input size {seq_len}: Error - {e}")
    
    # 4. Test model info and summary
    print("\n4. Model Information and Summary...")
    model_info = large_transformer.get_model_info()
    
    print(f"Model name: {model_info['model_name']}")
    print(f"Total parameters: {model_info['total_parameters']:,}")
    print(f"Model size: {model_info['model_size_mb']:.2f} MB")
    print("Architecture details:")
    for key, value in model_info['architecture'].items():
        print(f"  {key}: {value}")
    
    print("\nAdvanced features example completed successfully!\n")

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Run all examples"""
    
    print("🚀 Custom Model Architectures - Examples")
    print("=" * 60)
    
    try:
        # Run all examples
        example_basic_model_creation()
        example_model_from_config()
        example_task_specific_models()
        example_model_training()
        example_model_comparison()
        example_advanced_features()
        
        print("✅ All examples completed successfully!")
        print("\n🎯 Key Takeaways:")
        print("  - Custom nn.Module architectures provide flexible model building")
        print("  - Configuration-driven model creation for easy experimentation")
        print("  - Task-specific optimizations for different use cases")
        print("  - Seamless integration with PyTorch framework system")
        print("  - Comprehensive benchmarking and comparison capabilities")
        print("  - Advanced features like layer freezing and model analysis")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()


