#!/usr/bin/env python3
"""
Quick Start PyTorch Debugging for Video-OpusClip

This script demonstrates how to quickly enable and use PyTorch debugging tools
in the Video-OpusClip system, with focus on autograd.detect_anomaly().
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def quick_start_autograd_debugging():
    """Quick start example for autograd anomaly detection."""
    
    print("🚀 Quick Start: Autograd Anomaly Detection")
    print("=" * 50)
    
    try:
        # Import PyTorch debugging tools
        from pytorch_debug_tools import PyTorchDebugManager, PyTorchDebugConfig
        
        # Create a simple model
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 1)
        )
        
        # Create data
        inputs = torch.randn(32, 10)
        targets = torch.randn(32, 1)
        
        # Setup training components
        optimizer = optim.Adam(model.parameters())
        loss_fn = nn.MSELoss()
        
        # Initialize debug manager with basic configuration
        config = PyTorchDebugConfig(
            enable_autograd_anomaly=True,
            enable_gradient_debugging=True,
            enable_memory_debugging=True
        )
        
        debug_manager = PyTorchDebugManager(config)
        
        print("✅ PyTorch debugging tools initialized")
        print("🔍 Starting training with anomaly detection...")
        
        # Training loop with debugging
        for step in range(5):
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            
            # Backward pass with anomaly detection
            with debug_manager.anomaly_detector.detect_anomaly():
                loss.backward()
            
            # Check gradients
            gradient_info = debug_manager.gradient_debugger.check_gradients(model, step)
            
            optimizer.step()
            
            print(f"Step {step}: Loss = {loss.item():.6f}")
            if gradient_info.get('anomalies'):
                print(f"  ⚠️  Gradient anomalies: {gradient_info['anomalies']}")
        
        # Generate debug report
        report = debug_manager.generate_comprehensive_report()
        print(f"✅ Training completed. Debug report generated.")
        print(f"📊 Total anomalies detected: {report['anomaly_detection']['total_anomalies']}")
        
        return True
        
    except ImportError as e:
        print(f"❌ PyTorch debugging tools not available: {e}")
        print("Please ensure pytorch_debug_tools.py is in your Python path")
        return False
    except Exception as e:
        print(f"❌ Error during debugging: {e}")
        return False

def quick_start_gradient_debugging():
    """Quick start example for gradient debugging."""
    
    print("\n🚀 Quick Start: Gradient Debugging")
    print("=" * 50)
    
    try:
        from pytorch_debug_tools import GradientDebugger, PyTorchDebugConfig
        
        # Create model with potential gradient issues
        model = nn.Sequential(
            nn.Linear(10, 100),
            nn.Tanh(),
            nn.Linear(100, 50),
            nn.Tanh(),
            nn.Linear(50, 1)
        )
        
        # Initialize with large weights to potentially cause exploding gradients
        for module in model.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=5.0)
        
        # Create data
        inputs = torch.randn(64, 10)
        targets = torch.randn(64, 1)
        
        # Setup training
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        loss_fn = nn.MSELoss()
        
        # Initialize gradient debugger
        config = PyTorchDebugConfig(
            enable_gradient_debugging=True,
            gradient_norm_threshold=10.0,
            gradient_clip_threshold=1.0
        )
        gradient_debugger = GradientDebugger(config)
        
        print("✅ Gradient debugger initialized")
        print("🔍 Monitoring gradient flow...")
        
        # Training with gradient monitoring
        for step in range(3):
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            
            # Check gradients for anomalies
            gradient_info = gradient_debugger.check_gradients(model, step)
            
            print(f"Step {step}:")
            print(f"  Loss: {loss.item():.6f}")
            print(f"  Gradient norm: {gradient_info['statistics']['total_norm']:.6f}")
            print(f"  Anomalies: {gradient_info['anomalies']}")
            
            # Clip gradients if needed
            if gradient_info['anomalies']:
                print("  🔧 Applying gradient clipping...")
                gradient_debugger.clip_gradients(model, max_norm=1.0)
            
            optimizer.step()
        
        # Analyze gradient flow
        flow_analysis = gradient_debugger.analyze_gradient_flow(model)
        print(f"\n📊 Gradient flow analysis:")
        print(f"  Vanishing gradients: {flow_analysis['vanishing_gradients']}")
        print(f"  Exploding gradients: {flow_analysis['exploding_gradients']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during gradient debugging: {e}")
        return False

def quick_start_memory_debugging():
    """Quick start example for memory debugging."""
    
    print("\n🚀 Quick Start: Memory Debugging")
    print("=" * 50)
    
    try:
        from pytorch_debug_tools import PyTorchMemoryDebugger, PyTorchDebugConfig
        
        # Initialize memory debugger
        config = PyTorchDebugConfig(
            enable_memory_debugging=True,
            track_tensor_memory=True,
            memory_snapshot_frequency=2
        )
        memory_debugger = PyTorchMemoryDebugger(config)
        
        print("✅ Memory debugger initialized")
        print("🔍 Tracking tensor memory usage...")
        
        # Take initial snapshot
        initial_snapshot = memory_debugger.take_memory_snapshot("initial")
        print(f"Initial GPU memory: {initial_snapshot['gpu_memory']}")
        
        # Simulate video processing operations
        for operation_idx in range(3):
            # Create large tensors
            tensor_size = (1000, 1000)
            tensor = torch.randn(tensor_size, device='cuda' if torch.cuda.is_available() else 'cpu')
            memory_debugger.track_tensor(tensor, f"operation_{operation_idx}")
            
            # Process tensor
            processed_tensor = torch.nn.functional.interpolate(
                tensor.unsqueeze(0), scale_factor=0.5
            ).squeeze(0)
            memory_debugger.track_tensor(processed_tensor, f"processed_{operation_idx}")
            
            # Take snapshot
            snapshot = memory_debugger.take_memory_snapshot(f"operation_{operation_idx}")
            print(f"Operation {operation_idx}: GPU memory = {snapshot['gpu_memory']}")
            
            # Clean up
            del tensor, processed_tensor
        
        # Take final snapshot
        final_snapshot = memory_debugger.take_memory_snapshot("final")
        print(f"Final GPU memory: {final_snapshot['gpu_memory']}")
        
        # Get memory report
        report = memory_debugger.get_memory_report()
        print(f"📊 Memory tracking completed. Snapshots: {len(report['snapshots'])}")
        
        # Clear memory
        memory_debugger.clear_memory()
        print("🧹 Memory cleared")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during memory debugging: {e}")
        return False

def quick_start_comprehensive_debugging():
    """Quick start example for comprehensive debugging."""
    
    print("\n🚀 Quick Start: Comprehensive Debugging")
    print("=" * 50)
    
    try:
        from pytorch_debug_tools import PyTorchDebugManager, PyTorchDebugConfig
        
        # Initialize comprehensive debug manager
        config = PyTorchDebugConfig(
            enable_autograd_anomaly=True,
            enable_gradient_debugging=True,
            enable_memory_debugging=True,
            enable_model_debugging=True,
            enable_training_debugging=True,
            save_debug_reports=True,
            debug_output_dir="quick_start_debug_reports"
        )
        
        debug_manager = PyTorchDebugManager(config)
        
        # Create model and data
        model = nn.Sequential(
            nn.Linear(10, 50),
            nn.ReLU(),
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.Linear(20, 1)
        )
        
        # Create dataset
        X = torch.randn(200, 10)
        y = torch.randn(200, 1)
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Setup training
        optimizer = optim.Adam(model.parameters())
        loss_fn = nn.MSELoss()
        
        print("✅ Comprehensive debugging initialized")
        print("🔍 Running training with full debugging...")
        
        # Debug complete training loop
        debug_manager.debug_training_loop(
            model, dataloader, optimizer, loss_fn, num_epochs=1
        )
        
        # Generate comprehensive report
        report = debug_manager.generate_comprehensive_report()
        
        print("✅ Comprehensive debugging completed!")
        print(f"📊 Report generated with {len(report)} sections")
        print(f"📁 Debug reports saved to: {config.debug_output_dir}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during comprehensive debugging: {e}")
        return False

def quick_start_integration_with_training():
    """Quick start example integrating debugging with existing training."""
    
    print("\n🚀 Quick Start: Integration with Training")
    print("=" * 50)
    
    try:
        # Import existing training components
        from optimized_training import OptimizedTrainer, TrainingConfig
        from pytorch_debug_tools import PyTorchDebugConfig
        
        # Create model
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 1)
        )
        
        # Create dataset
        X = torch.randn(500, 10)
        y = torch.randn(500, 1)
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Create training config with debugging enabled
        config = TrainingConfig(
            epochs=2,
            batch_size=32,
            learning_rate=0.001,
            enable_pytorch_debugging=True,
            debug_config=PyTorchDebugConfig(
                enable_autograd_anomaly=True,
                enable_gradient_debugging=True,
                enable_memory_debugging=True,
                enable_training_debugging=True
            )
        )
        
        # Create trainer
        trainer = OptimizedTrainer(
            model=model,
            train_loader=dataloader,
            config=config
        )
        
        print("✅ Trainer with debugging initialized")
        print("🔍 Starting training with integrated debugging...")
        
        # Train with debugging
        results = trainer.train()
        
        print("✅ Training with debugging completed!")
        print(f"📊 Best metric: {results['best_metric']:.6f}")
        print(f"📊 Best epoch: {results['best_epoch']}")
        
        if results.get('debug_report'):
            print("📁 Debug report generated")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during training integration: {e}")
        return False

def main():
    """Run all quick start examples."""
    
    print("🚀 PyTorch Debugging Quick Start for Video-OpusClip")
    print("=" * 60)
    
    examples = [
        ("Autograd Anomaly Detection", quick_start_autograd_debugging),
        ("Gradient Debugging", quick_start_gradient_debugging),
        ("Memory Debugging", quick_start_memory_debugging),
        ("Comprehensive Debugging", quick_start_comprehensive_debugging),
        ("Training Integration", quick_start_integration_with_training)
    ]
    
    results = {}
    
    for name, example_func in examples:
        try:
            print(f"\n{'='*20} {name} {'='*20}")
            result = example_func()
            results[name] = result
            if result:
                print(f"✅ {name} completed successfully")
            else:
                print(f"❌ {name} failed")
        except Exception as e:
            print(f"❌ {name} failed with exception: {e}")
            results[name] = False
    
    print(f"\n{'='*60}")
    print("🎉 Quick Start Examples Summary")
    print(f"Successful examples: {sum(1 for r in results.values() if r)}/{len(examples)}")
    
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {name}: {status}")
    
    print(f"\n📚 For detailed usage, see PYTORCH_DEBUGGING_GUIDE.md")
    print(f"🔧 For examples, see pytorch_debugging_examples.py")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Run specific example
        example_name = sys.argv[1].lower()
        
        if example_name == "autograd":
            quick_start_autograd_debugging()
        elif example_name == "gradient":
            quick_start_gradient_debugging()
        elif example_name == "memory":
            quick_start_memory_debugging()
        elif example_name == "comprehensive":
            quick_start_comprehensive_debugging()
        elif example_name == "training":
            quick_start_integration_with_training()
        else:
            print(f"❌ Unknown example: {example_name}")
            print("Available examples: autograd, gradient, memory, comprehensive, training")
    else:
        # Run all examples
        main() 