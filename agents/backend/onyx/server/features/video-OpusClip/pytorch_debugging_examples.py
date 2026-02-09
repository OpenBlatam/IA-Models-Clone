"""
PyTorch Debugging Examples for Video-OpusClip

Practical examples demonstrating how to integrate PyTorch debugging tools
into the Video-OpusClip system, with focus on autograd.detect_anomaly()
and other debugging features.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

# Import PyTorch debugging tools
from pytorch_debug_tools import (
    PyTorchDebugManager, PyTorchDebugConfig, AutogradAnomalyDetector,
    GradientDebugger, PyTorchMemoryDebugger, ModelDebugger,
    TrainingDebugger, CUDADebugger, PyTorchProfiler
)

# Import existing Video-OpusClip components
from optimized_training import OptimizedTrainer, TrainingConfig
from optimized_video_processor import VideoProcessor
from optimized_config import get_config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# EXAMPLE 1: BASIC AUTOGRAD ANOMALY DETECTION
# =============================================================================

def example_basic_autograd_debugging():
    """Basic example of using autograd.detect_anomaly() in training."""
    
    print("🔍 Example 1: Basic Autograd Anomaly Detection")
    print("=" * 50)
    
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 1)
    )
    
    # Create dummy data
    inputs = torch.randn(32, 10)
    targets = torch.randn(32, 1)
    
    # Setup optimizer and loss
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.MSELoss()
    
    # Initialize anomaly detector
    config = PyTorchDebugConfig(enable_autograd_anomaly=True)
    anomaly_detector = AutogradAnomalyDetector(config)
    
    print("Training with anomaly detection...")
    
    # Training loop with anomaly detection
    for step in range(10):
        optimizer.zero_grad()
        
        # Wrap backward pass with anomaly detection
        with anomaly_detector.detect_anomaly():
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
        
        optimizer.step()
        
        if step % 2 == 0:
            print(f"Step {step}: Loss = {loss.item():.6f}")
    
    # Get anomaly report
    report = anomaly_detector.get_anomaly_report()
    print(f"Anomaly detection completed. Total anomalies: {report['total_anomalies']}")
    
    return anomaly_detector

# =============================================================================
# EXAMPLE 2: GRADIENT DEBUGGING IN TRAINING
# =============================================================================

def example_gradient_debugging():
    """Example of gradient debugging during training."""
    
    print("\n🔍 Example 2: Gradient Debugging in Training")
    print("=" * 50)
    
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
    
    # Setup training components
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.MSELoss()
    
    # Initialize gradient debugger
    config = PyTorchDebugConfig(
        enable_gradient_debugging=True,
        gradient_norm_threshold=10.0,
        gradient_clip_threshold=1.0
    )
    gradient_debugger = GradientDebugger(config)
    
    print("Training with gradient debugging...")
    
    # Training loop with gradient monitoring
    for step in range(5):
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
            print("  Applying gradient clipping...")
            gradient_debugger.clip_gradients(model, max_norm=1.0)
        
        optimizer.step()
    
    # Get gradient analysis
    flow_analysis = gradient_debugger.analyze_gradient_flow(model)
    print(f"\nGradient flow analysis:")
    print(f"  Vanishing gradients: {flow_analysis['vanishing_gradients']}")
    print(f"  Exploding gradients: {flow_analysis['exploding_gradients']}")
    
    return gradient_debugger

# =============================================================================
# EXAMPLE 3: MEMORY DEBUGGING FOR VIDEO PROCESSING
# =============================================================================

def example_memory_debugging():
    """Example of memory debugging for video processing operations."""
    
    print("\n🔍 Example 3: Memory Debugging for Video Processing")
    print("=" * 50)
    
    # Initialize memory debugger
    config = PyTorchDebugConfig(
        enable_memory_debugging=True,
        track_tensor_memory=True,
        memory_snapshot_frequency=2
    )
    memory_debugger = PyTorchMemoryDebugger(config)
    
    # Simulate video processing operations
    print("Simulating video processing with memory tracking...")
    
    # Take initial snapshot
    initial_snapshot = memory_debugger.take_memory_snapshot("initial")
    print(f"Initial GPU memory: {initial_snapshot['gpu_memory']}")
    
    # Simulate video frames processing
    for frame_idx in range(5):
        # Create large tensors simulating video frames
        frame_tensor = torch.randn(3, 1080, 1920, device='cuda' if torch.cuda.is_available() else 'cpu')
        memory_debugger.track_tensor(frame_tensor, f"frame_{frame_idx}")
        
        # Process frame (simulate some operations)
        processed_frame = torch.nn.functional.interpolate(
            frame_tensor.unsqueeze(0), scale_factor=0.5
        ).squeeze(0)
        memory_debugger.track_tensor(processed_frame, f"processed_frame_{frame_idx}")
        
        # Take snapshot periodically
        if frame_idx % 2 == 0:
            snapshot = memory_debugger.take_memory_snapshot(f"frame_{frame_idx}")
            print(f"Frame {frame_idx}: GPU memory = {snapshot['gpu_memory']}")
        
        # Clean up old tensors
        if frame_idx > 0:
            memory_debugger.untrack_tensor(frame_tensor)
    
    # Take final snapshot
    final_snapshot = memory_debugger.take_memory_snapshot("final")
    print(f"Final GPU memory: {final_snapshot['gpu_memory']}")
    
    # Get memory report
    report = memory_debugger.get_memory_report()
    print(f"Memory tracking completed. Snapshots: {len(report['snapshots'])}")
    
    # Clear memory
    memory_debugger.clear_memory()
    print("Memory cleared")
    
    return memory_debugger

# =============================================================================
# EXAMPLE 4: MODEL DEBUGGING FOR VIDEO MODELS
# =============================================================================

def example_model_debugging():
    """Example of model debugging for video processing models."""
    
    print("\n🔍 Example 4: Model Debugging for Video Models")
    print("=" * 50)
    
    # Create a video processing model
    class VideoModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
            self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
            self.conv3 = nn.Conv2d(128, 3, 3, padding=1)
            self.relu = nn.ReLU()
            
        def forward(self, x):
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = self.conv3(x)
            return x
    
    model = VideoModel()
    
    # Initialize model debugger
    config = PyTorchDebugConfig(
        enable_model_debugging=True,
        check_model_parameters=True,
        validate_model_inputs=True
    )
    model_debugger = ModelDebugger(config)
    
    # Inspect model
    print("Inspecting video model...")
    inspection = model_debugger.inspect_model(model, input_shape=(1, 3, 224, 224))
    
    print(f"Model info:")
    print(f"  Total parameters: {inspection['model_info']['total_parameters']}")
    print(f"  Trainable parameters: {inspection['model_info']['trainable_parameters']}")
    print(f"  Modules count: {inspection['model_info']['modules_count']}")
    
    print(f"\nParameter statistics:")
    stats = inspection['parameter_info']['statistics']
    print(f"  Zero parameters: {stats['zero_params']}")
    print(f"  NaN parameters: {stats['nan_params']}")
    print(f"  Inf parameters: {stats['inf_params']}")
    
    print(f"\nLayer information:")
    for name, info in inspection['layer_info'].items():
        print(f"  {name}: {info['type']} - {info['parameters']} params")
    
    # Validate model inputs
    print(f"\nInput validation:")
    validation = inspection['input_validation']
    if validation['forward_pass_successful']:
        print(f"  ✅ Model inputs are valid")
        print(f"  Output shape: {validation['output_shape']}")
    else:
        print(f"  ❌ Input validation failed: {validation['error']}")
    
    return model_debugger

# =============================================================================
# EXAMPLE 5: TRAINING DEBUGGING INTEGRATION
# =============================================================================

def example_training_debugging():
    """Example of integrating debugging into training loops."""
    
    print("\n🔍 Example 5: Training Debugging Integration")
    print("=" * 50)
    
    # Create model and data
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 1)
    )
    
    # Create dataset
    X = torch.randn(1000, 10)
    y = torch.randn(1000, 1)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Setup training components
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.MSELoss()
    
    # Initialize training debugger
    config = PyTorchDebugConfig(
        enable_training_debugging=True,
        debug_loss_computation=True,
        debug_optimizer_steps=True
    )
    training_debugger = TrainingDebugger(config)
    
    print("Training with comprehensive debugging...")
    
    # Training loop with debugging
    for epoch in range(2):
        epoch_loss = 0.0
        batch_count = 0
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            # Debug training step
            debug_info = training_debugger.debug_training_step(
                model, optimizer, loss_fn, inputs, targets, batch_idx
            )
            
            # Check for issues
            if not debug_info['loss_computation']['computation_successful']:
                print(f"❌ Training step failed: {debug_info['loss_computation']['error']}")
                break
            
            # Normal training
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}: Loss = {loss.item():.6f}")
        
        avg_loss = epoch_loss / batch_count
        print(f"Epoch {epoch} average loss: {avg_loss:.6f}")
    
    # Get training debug report
    report = training_debugger.get_training_report()
    print(f"Training debugging completed. Debug info entries: {len(report['training_debug_info'])}")
    
    return training_debugger

# =============================================================================
# EXAMPLE 6: CUDA DEBUGGING
# =============================================================================

def example_cuda_debugging():
    """Example of CUDA debugging for GPU operations."""
    
    print("\n🔍 Example 6: CUDA Debugging")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping CUDA debugging example")
        return None
    
    # Initialize CUDA debugger
    config = PyTorchDebugConfig(
        enable_cuda_debugging=True,
        cuda_memory_fraction=0.8,
        cuda_synchronize=True
    )
    cuda_debugger = CUDADebugger(config)
    
    # Create model on GPU
    model = nn.Sequential(
        nn.Linear(1000, 2000),
        nn.ReLU(),
        nn.Linear(2000, 1000)
    ).cuda()
    
    print("Running CUDA operations with debugging...")
    
    # Debug CUDA operations
    for operation_idx in range(3):
        # Create large tensors on GPU
        inputs = torch.randn(100, 1000, device='cuda')
        
        # Debug the operation
        cuda_info = cuda_debugger.debug_cuda_operations(f"operation_{operation_idx}")
        
        print(f"Operation {operation_idx}:")
        print(f"  GPU memory allocated: {cuda_info['memory_info']['allocated']:.2f} GB")
        print(f"  GPU memory cached: {cuda_info['memory_info']['cached']:.2f} GB")
        print(f"  Memory fraction: {cuda_info['memory_info']['memory_fraction']:.2%}")
        
        # Perform operation
        outputs = model(inputs)
        
        # Clear some memory
        del inputs, outputs
        torch.cuda.empty_cache()
    
    # Get CUDA report
    report = cuda_debugger.get_cuda_report()
    print(f"CUDA debugging completed. Debug info entries: {len(report['cuda_debug_info'])}")
    
    return cuda_debugger

# =============================================================================
# EXAMPLE 7: PROFILING INTEGRATION
# =============================================================================

def example_profiling():
    """Example of PyTorch profiling for performance analysis."""
    
    print("\n🔍 Example 7: PyTorch Profiling")
    print("=" * 50)
    
    # Initialize profiler
    config = PyTorchDebugConfig(
        enable_profiling=True,
        profile_memory=True,
        profile_cpu=True,
        profile_cuda=True
    )
    profiler = PyTorchProfiler(config)
    
    # Create model
    model = nn.Sequential(
        nn.Linear(1000, 2000),
        nn.ReLU(),
        nn.Linear(2000, 1000),
        nn.ReLU(),
        nn.Linear(1000, 100)
    )
    
    # Create data
    inputs = torch.randn(64, 1000)
    targets = torch.randn(64, 100)
    
    # Setup training components
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.MSELoss()
    
    print("Profiling training operations...")
    
    # Profile training operations
    with profiler.profile_operation("model_training") as prof:
        for step in range(10):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            
            if step % 2 == 0:
                print(f"Step {step}: Loss = {loss.item():.6f}")
    
    # Get profiling results
    results = profiler.get_profiler_report()
    print(f"Profiling completed. Results saved.")
    
    return profiler

# =============================================================================
# EXAMPLE 8: COMPREHENSIVE DEBUGGING INTEGRATION
# =============================================================================

def example_comprehensive_debugging():
    """Example of comprehensive debugging integration for Video-OpusClip."""
    
    print("\n🔍 Example 8: Comprehensive Debugging Integration")
    print("=" * 50)
    
    # Initialize comprehensive debug manager
    config = PyTorchDebugConfig(
        enable_autograd_anomaly=True,
        enable_gradient_debugging=True,
        enable_memory_debugging=True,
        enable_model_debugging=True,
        enable_training_debugging=True,
        enable_cuda_debugging=True,
        enable_profiling=True,
        save_debug_reports=True
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
    X = torch.randn(500, 10)
    y = torch.randn(500, 1)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Setup training components
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.MSELoss()
    
    print("Running comprehensive debugging...")
    
    # Debug complete training loop
    debug_manager.debug_training_loop(
        model, dataloader, optimizer, loss_fn, num_epochs=1
    )
    
    # Generate comprehensive report
    report = debug_manager.generate_comprehensive_report()
    
    print("Comprehensive debugging completed!")
    print(f"Report generated with {len(report)} sections")
    
    return debug_manager, report

# =============================================================================
# EXAMPLE 9: INTEGRATION WITH EXISTING VIDEO-OPUSCLIP COMPONENTS
# =============================================================================

def example_video_opusclip_integration():
    """Example of integrating PyTorch debugging with existing Video-OpusClip components."""
    
    print("\n🔍 Example 9: Video-OpusClip Integration")
    print("=" * 50)
    
    # Initialize debug manager with Video-OpusClip specific config
    config = PyTorchDebugConfig(
        enable_autograd_anomaly=True,
        enable_gradient_debugging=True,
        enable_memory_debugging=True,
        enable_model_debugging=True,
        enable_training_debugging=True,
        enable_cuda_debugging=True,
        enable_profiling=False,  # Disable profiling for production-like setup
        save_debug_reports=True,
        debug_output_dir="video_opusclip_debug_reports"
    )
    
    debug_manager = PyTorchDebugManager(config)
    
    # Simulate Video-OpusClip training setup
    class VideoOpusClipModel(nn.Module):
        def __init__(self):
            super().__init__()
            # Simulate video processing layers
            self.feature_extractor = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1))
            )
            self.classifier = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 10)
            )
            
        def forward(self, x):
            features = self.feature_extractor(x)
            features = features.view(features.size(0), -1)
            output = self.classifier(features)
            return output
    
    # Create model and inspect it
    model = VideoOpusClipModel()
    model_inspection = debug_manager.model_debugger.inspect_model(
        model, input_shape=(1, 3, 224, 224)
    )
    
    print(f"Video-OpusClip model inspection:")
    print(f"  Total parameters: {model_inspection['model_info']['total_parameters']}")
    print(f"  Model type: {model_inspection['model_info']['model_type']}")
    
    # Create simulated video data
    video_frames = torch.randn(100, 3, 224, 224)
    labels = torch.randint(0, 10, (100,))
    dataset = TensorDataset(video_frames, labels)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Setup training
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    
    print("Training Video-OpusClip model with debugging...")
    
    # Training loop with comprehensive debugging
    for epoch in range(2):
        epoch_loss = 0.0
        batch_count = 0
        
        for batch_idx, (frames, labels) in enumerate(dataloader):
            # Debug training step
            with debug_manager.anomaly_detector.detect_anomaly():
                optimizer.zero_grad()
                outputs = model(frames)
                loss = loss_fn(outputs, labels)
                loss.backward()
                
                # Check gradients
                gradient_info = debug_manager.gradient_debugger.check_gradients(model, batch_idx)
                
                optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
            
            # Memory snapshot every 10 batches
            if batch_idx % 10 == 0:
                debug_manager.memory_debugger.take_memory_snapshot(f"epoch_{epoch}_batch_{batch_idx}")
            
            if batch_idx % 5 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}: Loss = {loss.item():.6f}")
        
        avg_loss = epoch_loss / batch_count
        print(f"Epoch {epoch} average loss: {avg_loss:.6f}")
    
    # Generate final report
    report = debug_manager.generate_comprehensive_report()
    
    print("Video-OpusClip debugging completed!")
    print(f"Debug report saved to: {config.debug_output_dir}")
    
    return debug_manager, report

# =============================================================================
# EXAMPLE 10: ERROR RECOVERY AND GRACEFUL HANDLING
# =============================================================================

def example_error_recovery():
    """Example of error recovery and graceful handling with PyTorch debugging."""
    
    print("\n🔍 Example 10: Error Recovery and Graceful Handling")
    print("=" * 50)
    
    # Initialize debug manager
    config = PyTorchDebugConfig(
        enable_autograd_anomaly=True,
        enable_gradient_debugging=True,
        enable_memory_debugging=True
    )
    debug_manager = PyTorchDebugManager(config)
    
    # Create model with potential issues
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 1)
    )
    
    # Create problematic data (with NaN values)
    inputs = torch.randn(32, 10)
    inputs[0, 0] = float('nan')  # Introduce NaN
    targets = torch.randn(32, 1)
    
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.MSELoss()
    
    print("Training with error recovery...")
    
    # Training loop with error recovery
    for step in range(10):
        try:
            optimizer.zero_grad()
            
            # Use anomaly detection
            with debug_manager.anomaly_detector.detect_anomaly():
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                loss.backward()
            
            # Check gradients
            gradient_info = debug_manager.gradient_debugger.check_gradients(model, step)
            
            if gradient_info['anomalies']:
                print(f"Step {step}: Gradient anomalies detected - {gradient_info['anomalies']}")
                # Apply gradient clipping
                debug_manager.gradient_debugger.clip_gradients(model, max_norm=1.0)
            
            optimizer.step()
            
            print(f"Step {step}: Loss = {loss.item():.6f}")
            
        except Exception as e:
            print(f"Step {step}: Error occurred - {e}")
            
            # Recovery strategies
            print("  Applying recovery strategies...")
            
            # Clear gradients
            optimizer.zero_grad()
            
            # Clear memory
            debug_manager.memory_debugger.clear_memory()
            
            # Skip this batch and continue
            continue
    
    print("Error recovery example completed")
    
    return debug_manager

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_all_examples():
    """Run all PyTorch debugging examples."""
    
    print("🚀 Running PyTorch Debugging Examples for Video-OpusClip")
    print("=" * 60)
    
    examples = [
        ("Basic Autograd Anomaly Detection", example_basic_autograd_debugging),
        ("Gradient Debugging in Training", example_gradient_debugging),
        ("Memory Debugging for Video Processing", example_memory_debugging),
        ("Model Debugging for Video Models", example_model_debugging),
        ("Training Debugging Integration", example_training_debugging),
        ("CUDA Debugging", example_cuda_debugging),
        ("PyTorch Profiling", example_profiling),
        ("Comprehensive Debugging Integration", example_comprehensive_debugging),
        ("Video-OpusClip Integration", example_video_opusclip_integration),
        ("Error Recovery and Graceful Handling", example_error_recovery)
    ]
    
    results = {}
    
    for name, example_func in examples:
        try:
            print(f"\n{'='*20} {name} {'='*20}")
            result = example_func()
            results[name] = result
            print(f"✅ {name} completed successfully")
        except Exception as e:
            print(f"❌ {name} failed: {e}")
            results[name] = None
    
    print(f"\n{'='*60}")
    print("🎉 All examples completed!")
    print(f"Successful examples: {sum(1 for r in results.values() if r is not None)}/{len(examples)}")
    
    return results

def run_specific_example(example_name: str):
    """Run a specific example by name."""
    
    example_map = {
        "autograd": example_basic_autograd_debugging,
        "gradient": example_gradient_debugging,
        "memory": example_memory_debugging,
        "model": example_model_debugging,
        "training": example_training_debugging,
        "cuda": example_cuda_debugging,
        "profiling": example_profiling,
        "comprehensive": example_comprehensive_debugging,
        "integration": example_video_opusclip_integration,
        "recovery": example_error_recovery
    }
    
    if example_name in example_map:
        print(f"🚀 Running {example_name} example...")
        return example_map[example_name]()
    else:
        print(f"❌ Example '{example_name}' not found. Available examples: {list(example_map.keys())}")
        return None

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Run specific example
        example_name = sys.argv[1]
        run_specific_example(example_name)
    else:
        # Run all examples
        run_all_examples() 