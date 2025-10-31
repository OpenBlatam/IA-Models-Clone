"""
PyTorch Primary Framework System - Usage Examples
Demonstrates various use cases and configurations for the PyTorch framework system
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import yaml
import logging
from pathlib import Path
import time
from tqdm import tqdm

# Import our PyTorch framework system
from pytorch_primary_framework_system import (
    PyTorchPrimaryFrameworkSystem,
    PyTorchFrameworkConfig,
    PyTorchTransformerModel,
    PyTorchCNNModel,
    PyTorchTrainingUtilities
)

# =============================================================================
# EXAMPLE 1: BASIC PYTORCH FRAMEWORK USAGE
# =============================================================================

def example_basic_pytorch_framework():
    """Basic example of using PyTorch Primary Framework System"""
    
    print("=== Example 1: Basic PyTorch Framework Usage ===")
    
    # 1. Create basic configuration
    config = PyTorchFrameworkConfig(
        use_cuda=True,
        use_mixed_precision=True,
        use_gradient_checkpointing=True,
        compile_model=True,
        use_tensorboard=True
    )
    
    # 2. Initialize system
    pytorch_system = PyTorchPrimaryFrameworkSystem(config)
    
    # 3. Create a simple transformer model
    model = PyTorchTransformerModel(
        vocab_size=10000,
        d_model=256,
        n_heads=4,
        n_layers=4,
        d_ff=1024,
        max_seq_len=128,
        dropout=0.1
    )
    
    # 4. Optimize model for PyTorch
    model = pytorch_system.optimize_model_for_pytorch(model)
    
    # 5. Create optimizer and scheduler
    optimizer = pytorch_system.create_pytorch_optimizer(
        model, "adamw", lr=1e-4, weight_decay=1e-5
    )
    scheduler = pytorch_system.create_pytorch_scheduler(
        optimizer, "cosine", T_max=50
    )
    
    # 6. Create sample data
    batch_size = 8
    seq_len = 64
    vocab_size = 10000
    
    sample_data = torch.randint(0, vocab_size, (batch_size, seq_len))
    sample_targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # 7. Create dataset and dataloader
    dataset = TensorDataset(sample_data, sample_targets)
    dataloader = pytorch_system.create_optimized_dataloader(
        dataset, batch_size=batch_size, shuffle=True
    )
    
    # 8. Loss function
    criterion = nn.CrossEntropyLoss()
    
    # 9. Training loop
    print("Starting basic training...")
    model.train()
    
    for epoch in range(3):
        total_loss = 0.0
        
        for batch_idx, (data, targets) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch + 1}")):
            # Training step
            step_result = pytorch_system.train_step_pytorch(
                model=model,
                data=data,
                targets=targets,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                grad_clip_norm=1.0,
                global_step=epoch * len(dataloader) + batch_idx
            )
            
            total_loss += step_result['loss']
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}")
    
    # 10. Save checkpoint
    checkpoint_path = pytorch_system.save_pytorch_checkpoint(
        model, optimizer, scheduler, 3, avg_loss, "basic_example_checkpoint.pth"
    )
    print(f"Checkpoint saved: {checkpoint_path}")
    
    # 11. Cleanup
    pytorch_system.cleanup()
    
    print("Basic example completed successfully!\n")

# =============================================================================
# EXAMPLE 2: ADVANCED CONFIGURATION FROM YAML
# =============================================================================

def example_advanced_configuration():
    """Example using advanced configuration from YAML file"""
    
    print("=== Example 2: Advanced Configuration from YAML ===")
    
    # 1. Load configuration from YAML
    config_path = Path("pytorch_framework_config.yaml")
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # 2. Create configuration object with advanced settings
        config = PyTorchFrameworkConfig(
            use_cuda=config_dict['framework']['use_cuda'],
            use_mixed_precision=config_dict['framework']['use_mixed_precision'],
            use_gradient_checkpointing=config_dict['framework']['use_gradient_checkpointing'],
            compile_model=config_dict['performance']['compile_model'],
            use_tensorboard=config_dict['monitoring']['use_tensorboard'],
            use_profiler=config_dict['monitoring']['use_profiler'],
            batch_size=config_dict['training']['batch_size'],
            num_workers=config_dict['training']['num_workers']
        )
        
        print(f"Configuration loaded from YAML:")
        print(f"  - CUDA: {config.use_cuda}")
        print(f"  - Mixed Precision: {config.use_mixed_precision}")
        print(f"  - Model Compilation: {config.compile_model}")
        print(f"  - Batch Size: {config.batch_size}")
        print(f"  - Workers: {config.num_workers}")
        
    else:
        print("YAML config file not found, using default configuration")
        config = PyTorchFrameworkConfig()
    
    # 3. Initialize system with advanced config
    pytorch_system = PyTorchPrimaryFrameworkSystem(config)
    
    # 4. Create a larger transformer model
    model = PyTorchTransformerModel(
        vocab_size=50000,
        d_model=512,
        n_heads=8,
        n_layers=6,
        d_ff=2048,
        max_seq_len=256,
        dropout=0.1
    )
    
    # 5. Optimize model
    model = pytorch_system.optimize_model_for_pytorch(model)
    
    # 6. Create advanced optimizer and scheduler
    optimizer = pytorch_system.create_pytorch_optimizer(
        model, "adamw", lr=1e-4, weight_decay=1e-5, betas=(0.9, 0.999)
    )
    scheduler = pytorch_system.create_pytorch_scheduler(
        optimizer, "onecycle", max_lr=1e-3, epochs=10, steps_per_epoch=100
    )
    
    # 7. Profile model performance
    if config.use_profiler:
        print("Profiling model performance...")
        profile_result = pytorch_system.profile_pytorch_model(
            model, (1, 256), num_runs=50
        )
        print(f"Profile result: {profile_result}")
    
    # 8. Get memory information
    memory_info = pytorch_system.get_pytorch_memory_info()
    print(f"Memory Info: {memory_info}")
    
    # 9. Cleanup
    pytorch_system.cleanup()
    
    print("Advanced configuration example completed successfully!\n")

# =============================================================================
# EXAMPLE 3: CNN MODEL TRAINING
# =============================================================================

def example_cnn_training():
    """Example of training a CNN model with PyTorch framework"""
    
    print("=== Example 3: CNN Model Training ===")
    
    # 1. Configuration for CNN training
    config = PyTorchFrameworkConfig(
        use_cuda=True,
        use_mixed_precision=True,
        use_channels_last=True,  # Optimized for CNN
        compile_model=True,
        use_tensorboard=True
    )
    
    # 2. Initialize system
    pytorch_system = PyTorchPrimaryFrameworkSystem(config)
    
    # 3. Create CNN model
    model = PyTorchCNNModel(
        input_channels=3,
        num_classes=10,
        base_channels=32,
        num_layers=4
    )
    
    # 4. Optimize model
    model = pytorch_system.optimize_model_for_pytorch(model)
    
    # 5. Create optimizer and scheduler
    optimizer = pytorch_system.create_pytorch_optimizer(
        model, "sgd", lr=1e-3, momentum=0.9, weight_decay=1e-4
    )
    scheduler = pytorch_system.create_pytorch_scheduler(
        optimizer, "step", step_size=30, gamma=0.1
    )
    
    # 6. Create sample image data
    batch_size = 16
    channels = 3
    height = 64
    width = 64
    
    sample_images = torch.randn(batch_size, channels, height, width)
    sample_labels = torch.randint(0, 10, (batch_size,))
    
    # 7. Create dataset and dataloader
    dataset = TensorDataset(sample_images, sample_labels)
    dataloader = pytorch_system.create_optimized_dataloader(
        dataset, batch_size=batch_size, shuffle=True
    )
    
    # 8. Loss function
    criterion = nn.CrossEntropyLoss()
    
    # 9. Training loop
    print("Starting CNN training...")
    model.train()
    
    for epoch in range(5):
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch + 1}")):
            # Training step
            step_result = pytorch_system.train_step_pytorch(
                model=model,
                data=images,
                targets=labels,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                global_step=epoch * len(dataloader) + batch_idx
            )
            
            total_loss += step_result['loss']
            
            # Calculate accuracy
            with torch.no_grad():
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100 * correct / total
        
        print(f"Epoch {epoch + 1}: Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    # 10. Export model
    print("Exporting CNN model...")
    export_path = pytorch_system.export_pytorch_model(
        model, (1, 3, 64, 64), "torchscript", "cnn_model"
    )
    print(f"Model exported to: {export_path}")
    
    # 11. Cleanup
    pytorch_system.cleanup()
    
    print("CNN training example completed successfully!\n")

# =============================================================================
# EXAMPLE 4: FRAMEWORK COMPARISON
# =============================================================================

def example_framework_comparison():
    """Example comparing PyTorch with other frameworks"""
    
    print("=== Example 4: Framework Comparison ===")
    
    # 1. PyTorch configuration
    pytorch_config = PyTorchFrameworkConfig(
        use_cuda=True,
        use_mixed_precision=True,
        compile_model=True,
        use_tensorboard=True
    )
    
    # 2. Initialize PyTorch system
    pytorch_system = PyTorchPrimaryFrameworkSystem(pytorch_config)
    
    # 3. Create model for comparison
    model = PyTorchTransformerModel(
        vocab_size=10000,
        d_model=256,
        n_heads=4,
        n_layers=4,
        d_ff=1024,
        max_seq_len=128
    )
    
    # 4. Optimize model
    model = pytorch_system.optimize_model_for_pytorch(model)
    
    # 5. Create sample data
    batch_size = 16
    seq_len = 64
    vocab_size = 10000
    
    sample_data = torch.randint(0, vocab_size, (batch_size, seq_len))
    sample_targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # 6. Benchmark PyTorch performance
    print("Benchmarking PyTorch performance...")
    
    # Warmup
    model.eval()
    with torch.no_grad():
        for _ in range(10):
            _ = model(sample_data)
    
    # Benchmark
    start_time = time.time()
    model.eval()
    with torch.no_grad():
        for _ in range(100):
            _ = model(sample_data)
    
    pytorch_time = time.time() - start_time
    pytorch_memory = pytorch_system.get_pytorch_memory_info()
    
    print(f"PyTorch Performance:")
    print(f"  - Time for 100 forward passes: {pytorch_time:.4f} seconds")
    print(f"  - Memory allocated: {pytorch_memory.get('memory_allocated_gb', 0):.2f} GB")
    
    # 7. Export model for framework comparison
    print("Exporting model for framework comparison...")
    
    # Export to TorchScript
    torchscript_path = pytorch_system.export_pytorch_model(
        model, (1, 64), "torchscript", "comparison_model"
    )
    
    # Export to ONNX
    onnx_path = pytorch_system.export_pytorch_model(
        model, (1, 64), "onnx", "comparison_model"
    )
    
    print(f"Models exported:")
    print(f"  - TorchScript: {torchscript_path}")
    print(f"  - ONNX: {onnx_path}")
    
    # 8. Load and test TorchScript model
    if torchscript_path:
        print("Testing TorchScript model...")
        try:
            torchscript_model = torch.jit.load(torchscript_path)
            
            # Benchmark TorchScript
            start_time = time.time()
            torchscript_model.eval()
            with torch.no_grad():
                for _ in range(100):
                    _ = torchscript_model(sample_data)
            
            torchscript_time = time.time() - start_time
            
            print(f"TorchScript Performance:")
            print(f"  - Time for 100 forward passes: {torchscript_time:.4f} seconds")
            print(f"  - Speedup: {pytorch_time / torchscript_time:.2f}x")
            
        except Exception as e:
            print(f"TorchScript test failed: {e}")
    
    # 9. Cleanup
    pytorch_system.cleanup()
    
    print("Framework comparison example completed successfully!\n")

# =============================================================================
# EXAMPLE 5: PRODUCTION TRAINING PIPELINE
# =============================================================================

def example_production_training():
    """Example of a production-ready training pipeline"""
    
    print("=== Example 5: Production Training Pipeline ===")
    
    # 1. Production configuration
    config = PyTorchFrameworkConfig(
        use_cuda=True,
        use_mixed_precision=True,
        use_gradient_checkpointing=True,
        compile_model=True,
        use_tensorboard=True,
        use_profiler=False,  # Disable in production
        batch_size=32,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )
    
    # 2. Initialize system
    pytorch_system = PyTorchPrimaryFrameworkSystem(config)
    
    # 3. Create production model
    model = PyTorchTransformerModel(
        vocab_size=50000,
        d_model=768,
        n_heads=12,
        n_layers=12,
        d_ff=3072,
        max_seq_len=512,
        dropout=0.1
    )
    
    # 4. Optimize model
    model = pytorch_system.optimize_model_for_pytorch(model)
    
    # 5. Create production optimizer and scheduler
    optimizer = pytorch_system.create_pytorch_optimizer(
        model, "adamw", lr=1e-4, weight_decay=1e-5
    )
    scheduler = pytorch_system.create_pytorch_scheduler(
        optimizer, "warmup_cosine", T_max=1000, warmup_epochs=100
    )
    
    # 6. Create production dataset
    batch_size = 32
    seq_len = 256
    vocab_size = 50000
    
    # Simulate large dataset
    num_samples = 10000
    sample_data = torch.randint(0, vocab_size, (num_samples, seq_len))
    sample_targets = torch.randint(0, vocab_size, (num_samples, seq_len))
    
    # Split into train/val
    train_size = int(0.9 * num_samples)
    train_data = sample_data[:train_size]
    train_targets = sample_targets[:train_size]
    val_data = sample_data[train_size:]
    val_targets = sample_targets[train_size:]
    
    # Create dataloaders
    train_dataset = TensorDataset(train_data, train_targets)
    val_dataset = TensorDataset(val_data, val_targets)
    
    train_dataloader = pytorch_system.create_optimized_dataloader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_dataloader = pytorch_system.create_optimized_dataloader(
        val_dataset, batch_size=batch_size, shuffle=False
    )
    
    # 7. Loss function
    criterion = nn.CrossEntropyLoss()
    
    # 8. Production training loop with validation
    print("Starting production training...")
    
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 10
    
    for epoch in range(100):
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
        
        # Learning rate scheduling
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # Save best model
            best_model_path = pytorch_system.save_pytorch_checkpoint(
                model, optimizer, scheduler, epoch + 1, best_val_loss,
                "best_model_checkpoint.pth"
            )
            print(f"New best model saved: {best_model_path}")
        else:
            patience_counter += 1
        
        # Log metrics
        print(f"Epoch {epoch + 1}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
        print(f"  Patience: {patience_counter}/{max_patience}")
        
        # Regular checkpointing
        if (epoch + 1) % 10 == 0:
            checkpoint_path = pytorch_system.save_pytorch_checkpoint(
                model, optimizer, scheduler, epoch + 1, avg_val_loss,
                f"epoch_{epoch + 1}_checkpoint.pth"
            )
            print(f"Checkpoint saved: {checkpoint_path}")
        
        # Early stopping check
        if patience_counter >= max_patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break
    
    # 9. Final model export
    print("Exporting final production model...")
    
    # Export to multiple formats
    export_formats = ["torchscript", "onnx", "pytorch"]
    
    for export_format in export_formats:
        try:
            export_path = pytorch_system.export_pytorch_model(
                model, (1, 256), export_format, f"production_model_{export_format}"
            )
            print(f"Exported to {export_format}: {export_path}")
        except Exception as e:
            print(f"Export to {export_format} failed: {e}")
    
    # 10. Final memory report
    final_memory = pytorch_system.get_pytorch_memory_info()
    print(f"Final Memory Usage: {final_memory}")
    
    # 11. Cleanup
    pytorch_system.cleanup()
    
    print("Production training example completed successfully!\n")

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Run all examples"""
    
    print("🚀 PyTorch Primary Framework System - Examples")
    print("=" * 60)
    
    try:
        # Run all examples
        example_basic_pytorch_framework()
        example_advanced_configuration()
        example_cnn_training()
        example_framework_comparison()
        example_production_training()
        
        print("✅ All examples completed successfully!")
        print("\n🎯 Key Takeaways:")
        print("  - PyTorch framework system provides comprehensive deep learning capabilities")
        print("  - Easy configuration management with YAML files")
        print("  - Built-in performance optimizations and monitoring")
        print("  - Production-ready training pipelines")
        print("  - Framework comparison and model export capabilities")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()


