from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import time
import json
from typing import Dict, List, Any, Tuple
import traceback
from gradio_app import (
        from gradio_app import performance_optimizer
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
🚀 Multi-GPU Training Example
=============================

This example demonstrates comprehensive multi-GPU training using
DataParallel and DistributedDataParallel in the Gradio app.
"""


# Import the multi-GPU trainer from gradio_app
    multi_gpu_trainer, train_with_multi_gpu, evaluate_with_multi_gpu,
    get_multi_gpu_status, log_training_start, log_training_end
)

class SimpleNeuralNetwork(nn.Module):
    """Simple neural network for multi-GPU training demonstration."""
    
    def __init__(self, input_size=784, hidden_size=512, output_size=10) -> Any:
        super(SimpleNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x) -> Any:
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class DummyDataset(data.Dataset):
    """Dummy dataset for demonstration purposes."""
    
    def __init__(self, num_samples=1000, input_size=784, num_classes=10) -> Any:
        self.num_samples = num_samples
        self.input_size = input_size
        self.num_classes = num_classes
        
        # Generate dummy data
        self.data = torch.randn(num_samples, input_size)
        self.targets = torch.randint(0, num_classes, (num_samples,))
        
    def __len__(self) -> Any:
        return self.num_samples
    
    def __getitem__(self, idx) -> Optional[Dict[str, Any]]:
        return self.data[idx], self.targets[idx]

def demonstrate_gpu_info():
    """Demonstrate comprehensive GPU information gathering."""
    print("🔍 Demonstrating GPU Information")
    print("=" * 60)
    
    try:
        # Get GPU information
        gpu_info = multi_gpu_trainer.get_gpu_info()
        
        print(f"CUDA Available: {gpu_info['cuda_available']}")
        print(f"GPU Count: {gpu_info['gpu_count']}")
        print(f"Total Memory: {gpu_info['total_memory_gb']} GB")
        print(f"Compute Capabilities: {gpu_info['compute_capability']}")
        
        print("\nGPU Details:")
        for gpu in gpu_info['gpu_details']:
            print(f"  GPU {gpu['id']}: {gpu['name']}")
            print(f"    Memory: {gpu['memory_gb']} GB")
            print(f"    Compute Capability: {gpu['compute_capability']}")
            print(f"    Multi-Processors: {gpu['multi_processor_count']}")
            print(f"    Status: {gpu['status']}")
        
        # Check multi-GPU availability
        if gpu_info['cuda_available'] and gpu_info['gpu_count'] >= 2:
            print(f"\n✅ Multi-GPU training available with {gpu_info['gpu_count']} GPUs")
        else:
            print(f"\n⚠️ Multi-GPU training not available (GPUs: {gpu_info['gpu_count']})")
        
    except Exception as e:
        print(f"❌ Error in GPU info demo: {e}")
        traceback.print_exc()

def demonstrate_data_parallel():
    """Demonstrate DataParallel setup and usage."""
    print("\n📦 Demonstrating DataParallel")
    print("=" * 60)
    
    try:
        # Create model
        model = SimpleNeuralNetwork()
        
        # Setup DataParallel
        model, success = multi_gpu_trainer.setup_data_parallel(model)
        
        if success:
            print("✅ DataParallel setup successful")
            print(f"Strategy: {multi_gpu_trainer.current_strategy}")
            print(f"GPU Config: {multi_gpu_trainer.gpu_config}")
            
            # Test forward pass
            x = torch.randn(32, 784)
            if torch.cuda.is_available():
                x = x.cuda()
            
            with torch.no_grad():
                output = model(x)
                print(f"Output shape: {output.shape}")
                print(f"Model device: {next(model.parameters()).device}")
            
        else:
            print("❌ DataParallel setup failed")
        
    except Exception as e:
        print(f"❌ Error in DataParallel demo: {e}")
        traceback.print_exc()

def demonstrate_distributed_data_parallel():
    """Demonstrate DistributedDataParallel setup and usage."""
    print("\n🌐 Demonstrating DistributedDataParallel")
    print("=" * 60)
    
    try:
        # Create model
        model = SimpleNeuralNetwork()
        
        # Setup DistributedDataParallel
        model, success = multi_gpu_trainer.setup_distributed_data_parallel(
            model, backend='nccl', world_size=2, rank=0
        )
        
        if success:
            print("✅ DistributedDataParallel setup successful")
            print(f"Strategy: {multi_gpu_trainer.current_strategy}")
            print(f"GPU Config: {multi_gpu_trainer.gpu_config}")
            
            # Test forward pass
            x = torch.randn(32, 784)
            if torch.cuda.is_available():
                x = x.cuda()
            
            with torch.no_grad():
                output = model(x)
                print(f"Output shape: {output.shape}")
                print(f"Model device: {next(model.parameters()).device}")
            
        else:
            print("❌ DistributedDataParallel setup failed")
        
    except Exception as e:
        print(f"❌ Error in DistributedDataParallel demo: {e}")
        traceback.print_exc()

def demonstrate_auto_strategy_selection():
    """Demonstrate automatic strategy selection."""
    print("\n🤖 Demonstrating Auto Strategy Selection")
    print("=" * 60)
    
    try:
        # Create model
        model = SimpleNeuralNetwork()
        
        # Auto-select strategy
        model, success, gpu_info = multi_gpu_trainer.setup_multi_gpu_training(
            model, strategy='auto'
        )
        
        if success:
            print("✅ Auto strategy selection successful")
            print(f"Selected strategy: {multi_gpu_trainer.current_strategy}")
            print(f"GPU info: {gpu_info}")
            print(f"GPU config: {multi_gpu_trainer.gpu_config}")
            
            # Test forward pass
            x = torch.randn(32, 784)
            if torch.cuda.is_available():
                x = x.cuda()
            
            with torch.no_grad():
                output = model(x)
                print(f"Output shape: {output.shape}")
            
        else:
            print("❌ Auto strategy selection failed")
        
    except Exception as e:
        print(f"❌ Error in auto strategy selection demo: {e}")
        traceback.print_exc()

def demonstrate_batch_size_optimization():
    """Demonstrate batch size optimization for multi-GPU training."""
    print("\n📊 Demonstrating Batch Size Optimization")
    print("=" * 60)
    
    try:
        # Test different scenarios
        scenarios = [
            {'base_batch_size': 32, 'gpu_count': 2, 'strategy': 'DataParallel'},
            {'base_batch_size': 64, 'gpu_count': 4, 'strategy': 'DataParallel'},
            {'base_batch_size': 16, 'gpu_count': 2, 'strategy': 'DistributedDataParallel'},
            {'base_batch_size': 32, 'gpu_count': 4, 'strategy': 'DistributedDataParallel'}
        ]
        
        for scenario in scenarios:
            optimization = multi_gpu_trainer.optimize_batch_size_for_multi_gpu(
                scenario['base_batch_size'],
                scenario['gpu_count'],
                scenario['strategy']
            )
            
            print(f"\n{scenario['strategy']} with {scenario['gpu_count']} GPUs:")
            print(f"  Base batch size: {optimization['base_batch_size']}")
            print(f"  Effective batch size: {optimization['effective_batch_size']}")
            print(f"  Batch per GPU: {optimization['batch_per_gpu']}")
            print(f"  Scaling factor: {optimization['scaling_factor']}")
        
    except Exception as e:
        print(f"❌ Error in batch size optimization demo: {e}")
        traceback.print_exc()

def demonstrate_multi_gpu_training():
    """Demonstrate complete multi-GPU training workflow."""
    print("\n🏋️ Demonstrating Multi-GPU Training")
    print("=" * 60)
    
    try:
        # Create model, dataset, and data loader
        model = SimpleNeuralNetwork()
        dataset = DummyDataset(num_samples=500, input_size=784, num_classes=10)
        train_loader = data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
        
        # Setup optimizer and criterion
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Log training start
        log_training_start(
            model_name="SimpleNeuralNetwork",
            total_epochs=3,
            total_steps=len(train_loader) * 3,
            batch_size=32,
            learning_rate=0.001,
            optimizer="Adam"
        )
        
        # Train with multi-GPU
        training_results = train_with_multi_gpu(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            num_epochs=3,
            strategy='auto',
            use_mixed_precision=True,
            gradient_accumulation_steps=1
        )
        
        print("Training Results:")
        print(f"  Success: {training_results['success']}")
        print(f"  Epochs completed: {training_results['epochs_completed']}")
        print(f"  Final loss: {training_results['final_loss']:.4f}")
        print(f"  Training time: {training_results['training_time']:.2f}s")
        
        if 'gpu_utilization' in training_results:
            print(f"  GPU utilization: {training_results['gpu_utilization']}")
        
        if training_results['error']:
            print(f"  Error: {training_results['error']}")
        
    except Exception as e:
        print(f"❌ Error in multi-GPU training demo: {e}")
        traceback.print_exc()

def demonstrate_multi_gpu_evaluation():
    """Demonstrate multi-GPU evaluation."""
    print("\n📈 Demonstrating Multi-GPU Evaluation")
    print("=" * 60)
    
    try:
        # Create model and test dataset
        model = SimpleNeuralNetwork()
        test_dataset = DummyDataset(num_samples=200, input_size=784, num_classes=10)
        test_loader = data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
        
        # Setup criterion
        criterion = nn.CrossEntropyLoss()
        
        # Evaluate with multi-GPU
        evaluation_results = evaluate_with_multi_gpu(
            model=model,
            test_loader=test_loader,
            criterion=criterion,
            strategy='auto'
        )
        
        print("Evaluation Results:")
        print(f"  Success: {evaluation_results['success']}")
        print(f"  Test loss: {evaluation_results['test_loss']:.4f}")
        print(f"  Accuracy: {evaluation_results['accuracy']:.2f}%")
        
        if 'gpu_utilization' in evaluation_results:
            print(f"  GPU utilization: {evaluation_results['gpu_utilization']}")
        
        if evaluation_results['error']:
            print(f"  Error: {evaluation_results['error']}")
        
    except Exception as e:
        print(f"❌ Error in multi-GPU evaluation demo: {e}")
        traceback.print_exc()

def demonstrate_training_metrics():
    """Demonstrate training metrics logging and retrieval."""
    print("\n📊 Demonstrating Training Metrics")
    print("=" * 60)
    
    try:
        # Log some training metrics
        for epoch in range(2):
            for step in range(5):
                loss = 1.0 - (epoch * 0.3 + step * 0.1)  # Simulate decreasing loss
                learning_rate = 0.001 * (0.9 ** epoch)  # Simulate learning rate decay
                
                # Get GPU utilization
                gpu_utilization = {
                    'gpu_0_memory': 0.8,
                    'gpu_1_memory': 0.7,
                    'gpu_0_utilization': 0.9,
                    'gpu_1_utilization': 0.85
                }
                
                multi_gpu_trainer.log_training_metrics(
                    epoch, step, loss, learning_rate, gpu_utilization
                )
        
        # Get training metrics
        metrics = multi_gpu_trainer.get_multi_gpu_metrics()
        
        print("Training Metrics:")
        print(f"  Strategy: {metrics['strategy']}")
        print(f"  GPU config: {metrics['gpu_config']}")
        print(f"  DDP initialized: {metrics['ddp_initialized']}")
        print(f"  DP initialized: {metrics['dp_initialized']}")
        
        print("\nTraining history:")
        for epoch_key, epoch_metrics in metrics['training_metrics'].items():
            print(f"  {epoch_key}: {len(epoch_metrics)} steps logged")
            if epoch_metrics:
                latest = epoch_metrics[-1]
                print(f"    Latest - Loss: {latest['loss']:.4f}, LR: {latest['learning_rate']:.6f}")
        
    except Exception as e:
        print(f"❌ Error in training metrics demo: {e}")
        traceback.print_exc()

def demonstrate_multi_gpu_status():
    """Demonstrate comprehensive multi-GPU status monitoring."""
    print("\n📋 Demonstrating Multi-GPU Status")
    print("=" * 60)
    
    try:
        # Get multi-GPU status
        status = get_multi_gpu_status()
        
        print("Multi-GPU Status:")
        print(f"  Multi-GPU available: {status['multi_gpu_available']}")
        print(f"  Current strategy: {status['current_strategy']}")
        print(f"  Status: {status['status']}")
        
        if 'gpu_info' in status:
            gpu_info = status['gpu_info']
            print(f"  GPU count: {gpu_info['gpu_count']}")
            print(f"  Total memory: {gpu_info['total_memory_gb']} GB")
        
        if 'training_metrics' in status:
            training_metrics = status['training_metrics']
            print(f"  Training metrics available: {len(training_metrics) > 0}")
        
        if 'performance_summary' in status:
            perf_summary = status['performance_summary']
            print(f"  Performance operations tracked: {perf_summary.get('total_operations', 0)}")
        
        # Export status to JSON
        with open('logs/multi_gpu_status.json', 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(status, f, indent=2)
        print("\n✅ Multi-GPU status exported to logs/multi_gpu_status.json")
        
    except Exception as e:
        print(f"❌ Error in multi-GPU status demo: {e}")
        traceback.print_exc()

def demonstrate_distributed_cleanup():
    """Demonstrate distributed training cleanup."""
    print("\n🧹 Demonstrating Distributed Cleanup")
    print("=" * 60)
    
    try:
        # Setup distributed training first
        model = SimpleNeuralNetwork()
        model, success = multi_gpu_trainer.setup_distributed_data_parallel(model)
        
        if success:
            print("✅ Distributed training setup successful")
            print(f"Strategy: {multi_gpu_trainer.current_strategy}")
            print(f"DDP initialized: {multi_gpu_trainer.ddp_initialized}")
            
            # Cleanup distributed resources
            multi_gpu_trainer.cleanup_distributed()
            
            print("✅ Distributed cleanup completed")
            print(f"Strategy after cleanup: {multi_gpu_trainer.current_strategy}")
            print(f"DDP initialized after cleanup: {multi_gpu_trainer.ddp_initialized}")
        else:
            print("❌ Distributed training setup failed")
        
    except Exception as e:
        print(f"❌ Error in distributed cleanup demo: {e}")
        traceback.print_exc()

def demonstrate_performance_comparison():
    """Demonstrate performance comparison between single and multi-GPU."""
    print("\n⚡ Demonstrating Performance Comparison")
    print("=" * 60)
    
    try:
        # Create model and dataset
        model = SimpleNeuralNetwork()
        dataset = DummyDataset(num_samples=100, input_size=784, num_classes=10)
        train_loader = data.DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Test single GPU performance
        print("Testing single GPU performance...")
        model_single = SimpleNeuralNetwork()
        if torch.cuda.is_available():
            model_single = model_single.cuda()
        
        start_time = time.time()
        model_single.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            
            optimizer.zero_grad()
            output = model_single(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx >= 5:  # Test first 5 batches
                break
        
        single_gpu_time = time.time() - start_time
        print(f"Single GPU time: {single_gpu_time:.3f}s")
        
        # Test multi-GPU performance
        print("Testing multi-GPU performance...")
        model_multi = SimpleNeuralNetwork()
        model_multi, success = multi_gpu_trainer.setup_data_parallel(model_multi)
        
        if success:
            start_time = time.time()
            model_multi.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                if torch.cuda.is_available():
                    data, target = data.cuda(), target.cuda()
                
                optimizer.zero_grad()
                output = model_multi(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                if batch_idx >= 5:  # Test first 5 batches
                    break
            
            multi_gpu_time = time.time() - start_time
            print(f"Multi-GPU time: {multi_gpu_time:.3f}s")
            
            # Calculate speedup
            if multi_gpu_time > 0:
                speedup = single_gpu_time / multi_gpu_time
                print(f"Speedup: {speedup:.2f}x")
            
            # Cleanup
            multi_gpu_trainer.cleanup_distributed()
        else:
            print("❌ Multi-GPU setup failed for performance comparison")
        
    except Exception as e:
        print(f"❌ Error in performance comparison demo: {e}")
        traceback.print_exc()

def demonstrate_error_handling():
    """Demonstrate error handling in multi-GPU training."""
    print("\n🚨 Demonstrating Error Handling")
    print("=" * 60)
    
    try:
        # Test with invalid device IDs
        model = SimpleNeuralNetwork()
        model, success = multi_gpu_trainer.setup_data_parallel(
            model, device_ids=[999, 1000]  # Invalid device IDs
        )
        
        if not success:
            print("✅ Correctly handled invalid device IDs")
        
        # Test with single GPU when multi-GPU is requested
        if torch.cuda.device_count() < 2:
            model = SimpleNeuralNetwork()
            model, success = multi_gpu_trainer.setup_data_parallel(model)
            
            if not success:
                print("✅ Correctly handled single GPU scenario")
        
        # Test with CUDA not available
        if not torch.cuda.is_available():
            model = SimpleNeuralNetwork()
            model, success = multi_gpu_trainer.setup_data_parallel(model)
            
            if not success:
                print("✅ Correctly handled CUDA not available")
        
        print("✅ Error handling tests completed")
        
    except Exception as e:
        print(f"❌ Error in error handling demo: {e}")
        traceback.print_exc()

def demonstrate_integration_with_performance_optimizer():
    """Demonstrate integration with performance optimizer."""
    print("\n🔗 Demonstrating Integration with Performance Optimizer")
    print("=" * 60)
    
    try:
        
        # Create model
        model = SimpleNeuralNetwork()
        
        # Setup multi-GPU
        model, success = multi_gpu_trainer.setup_data_parallel(model)
        
        if success:
            # Apply performance optimizations
            optimizations = performance_optimizer.optimize_pipeline_performance(model)
            print(f"✅ Performance optimizations applied: {optimizations}")
            
            # Get performance metrics
            result, metrics = performance_optimizer.measure_performance(
                "multi_gpu_forward",
                lambda: model(torch.randn(32, 784).cuda() if torch.cuda.is_available() else torch.randn(32, 784))
            )
            
            print(f"✅ Performance measurement: {metrics}")
            
            # Get combined metrics
            multi_gpu_metrics = multi_gpu_trainer.get_multi_gpu_metrics()
            performance_summary = performance_optimizer.get_performance_summary()
            
            print(f"✅ Multi-GPU metrics: {len(multi_gpu_metrics)} entries")
            print(f"✅ Performance summary: {len(performance_summary.get('performance_by_operation', {}))} operations")
            
            # Cleanup
            multi_gpu_trainer.cleanup_distributed()
        else:
            print("❌ Multi-GPU setup failed for integration test")
        
    except Exception as e:
        print(f"❌ Error in integration demo: {e}")
        traceback.print_exc()

def main():
    """Run all multi-GPU training demonstrations."""
    print("🚀 Multi-GPU Training Demonstration")
    print("=" * 80)
    
    demonstrations = [
        demonstrate_gpu_info,
        demonstrate_data_parallel,
        demonstrate_distributed_data_parallel,
        demonstrate_auto_strategy_selection,
        demonstrate_batch_size_optimization,
        demonstrate_training_metrics,
        demonstrate_multi_gpu_training,
        demonstrate_multi_gpu_evaluation,
        demonstrate_multi_gpu_status,
        demonstrate_distributed_cleanup,
        demonstrate_performance_comparison,
        demonstrate_error_handling,
        demonstrate_integration_with_performance_optimizer
    ]
    
    for i, demo in enumerate(demonstrations, 1):
        try:
            print(f"\n[{i}/{len(demonstrations)}] Running {demo.__name__}...")
            demo()
        except Exception as e:
            print(f"❌ Failed to run {demo.__name__}: {e}")
            traceback.print_exc()
    
    print("\n🎉 Multi-GPU training demonstration completed!")
    print("Check the 'logs' directory for exported metrics and status files.")

match __name__:
    case "__main__":
    main() 