#!/usr/bin/env python3
"""
Example Bulk Usage - Comprehensive example of bulk optimization system
Demonstrates how to use the bulk optimization system for various scenarios
"""

import torch
import torch.nn as nn
import time
import json
import tempfile
import os
from typing import List, Tuple
import numpy as np

# Import bulk components
from bulk_optimizer import (
    create_bulk_optimizer, optimize_models_bulk_simple,
    BulkOptimizerConfig, OperationType
)
from bulk_optimization_core import (
    create_bulk_optimization_core, BulkOptimizationConfig
)
from bulk_data_processor import (
    create_bulk_data_processor, BulkDataConfig, BulkDataset
)
from bulk_operation_manager import (
    create_bulk_operation_manager, BulkOperationConfig
)

def create_example_models() -> List[Tuple[str, nn.Module]]:
    """Create example models for bulk optimization."""
    models = []
    
    # Simple Linear Model
    class SimpleLinear(nn.Module):
        def __init__(self, input_size=10, output_size=5):
            super().__init__()
            self.linear = nn.Linear(input_size, output_size)
        
        def forward(self, x):
            return self.linear(x)
    
    # MLP Model
    class MLPModel(nn.Module):
        def __init__(self, input_size=10, hidden_size=20, output_size=5):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_size)
            )
        
        def forward(self, x):
            return self.layers(x)
    
    # CNN Model
    class CNNModel(nn.Module):
        def __init__(self, num_classes=5):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(4),
                nn.Conv2d(16, 32, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(2)
            )
            self.classifier = nn.Linear(32 * 4, num_classes)
        
        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            return self.classifier(x)
    
    # LSTM Model
    class LSTMModel(nn.Module):
        def __init__(self, input_size=10, hidden_size=20, output_size=5):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
            self.classifier = nn.Linear(hidden_size, output_size)
        
        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            return self.classifier(lstm_out[:, -1, :])
    
    # Create models
    models.extend([
        ("SimpleLinear_1", SimpleLinear()),
        ("SimpleLinear_2", SimpleLinear()),
        ("MLP_1", MLPModel()),
        ("MLP_2", MLPModel()),
        ("CNN_1", CNNModel()),
        ("LSTM_1", LSTMModel())
    ])
    
    return models

def create_example_datasets() -> List[BulkDataset]:
    """Create example datasets for bulk processing."""
    datasets = []
    
    # Create temporary data files
    temp_files = []
    
    try:
        # JSON dataset
        json_data = [
            {"features": [1, 2, 3, 4, 5], "label": 0},
            {"features": [6, 7, 8, 9, 10], "label": 1},
            {"features": [11, 12, 13, 14, 15], "label": 0},
            {"features": [16, 17, 18, 19, 20], "label": 1}
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(json_data, f)
            json_file = f.name
            temp_files.append(json_file)
        
        # Create dataset configurations
        config1 = BulkDataConfig(batch_size=2, num_workers=0, enable_parallel_processing=False)
        config2 = BulkDataConfig(batch_size=4, num_workers=0, enable_parallel_processing=False)
        
        # Create datasets
        dataset1 = BulkDataset(json_file, config1)
        dataset2 = BulkDataset(json_file, config2)
        
        datasets.extend([dataset1, dataset2])
        
    finally:
        # Clean up temporary files
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    return datasets

def example_basic_bulk_optimization():
    """Example of basic bulk optimization."""
    print("🚀 Example: Basic Bulk Optimization")
    print("-" * 40)
    
    # Create models
    models = create_example_models()
    print(f"Created {len(models)} models for optimization")
    
    # Configure bulk optimizer
    config = BulkOptimizerConfig(
        max_models_per_batch=3,
        enable_parallel_optimization=True,
        optimization_strategies=['memory', 'computational'],
        enable_optimization_core=True
    )
    
    # Create bulk optimizer
    optimizer = create_bulk_optimizer(config)
    print("✅ Bulk optimizer created")
    
    # Optimize models
    print("🔄 Starting bulk optimization...")
    start_time = time.time()
    
    results = optimizer.optimize_models_bulk(models)
    
    end_time = time.time()
    print(f"✅ Bulk optimization completed in {end_time - start_time:.2f} seconds")
    
    # Print results
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    
    print(f"\n📊 Results Summary:")
    print(f"  - Total models: {len(results)}")
    print(f"  - Successful: {len(successful)}")
    print(f"  - Failed: {len(failed)}")
    print(f"  - Success rate: {len(successful)/len(results)*100:.1f}%")
    
    if successful:
        avg_time = np.mean([r.optimization_time for r in successful])
        avg_reduction = np.mean([r.parameter_reduction for r in successful])
        print(f"  - Average optimization time: {avg_time:.2f}s")
        print(f"  - Average parameter reduction: {avg_reduction:.2%}")
    
    return results

def example_advanced_bulk_optimization():
    """Example of advanced bulk optimization with custom configuration."""
    print("\n🚀 Example: Advanced Bulk Optimization")
    print("-" * 40)
    
    # Create models
    models = create_example_models()
    
    # Advanced configuration
    config = BulkOptimizerConfig(
        max_models_per_batch=2,
        enable_parallel_optimization=True,
        optimization_strategies=['memory', 'computational', 'mcts', 'hybrid'],
        enable_optimization_core=True,
        enable_data_processor=True,
        enable_operation_manager=True,
        max_concurrent_operations=2,
        enable_performance_monitoring=True,
        enable_result_persistence=True
    )
    
    # Create bulk optimizer
    optimizer = create_bulk_optimizer(config)
    print("✅ Advanced bulk optimizer created")
    
    # Submit optimization operation
    operation_id = optimizer.submit_bulk_operation(
        OperationType.OPTIMIZATION,
        models,
        config={'optimization_strategy': 'auto'}
    )
    print(f"📝 Submitted operation: {operation_id}")
    
    # Monitor operation
    print("🔄 Monitoring operation...")
    for i in range(10):  # Check for 10 seconds
        status = optimizer.get_operation_status(operation_id)
        print(f"  Status: {status}")
        
        if status in [OperationStatus.COMPLETED, OperationStatus.FAILED]:
            break
        
        time.sleep(1)
    
    # Get results
    results = optimizer.get_operation_results(operation_id)
    if results:
        print("✅ Operation completed successfully")
        print(f"  - Results: {results}")
    else:
        print("❌ Operation failed or not completed")
    
    return results

def example_bulk_dataset_processing():
    """Example of bulk dataset processing."""
    print("\n🚀 Example: Bulk Dataset Processing")
    print("-" * 40)
    
    # Create datasets
    datasets = create_example_datasets()
    print(f"Created {len(datasets)} datasets for processing")
    
    # Configure data processor
    config = BulkDataConfig(
        batch_size=2,
        num_workers=0,
        enable_parallel_processing=False,
        enable_data_augmentation=True,
        enable_performance_monitoring=True
    )
    
    # Create data processor
    processor = create_bulk_data_processor(config)
    print("✅ Bulk data processor created")
    
    # Process datasets
    print("🔄 Processing datasets...")
    start_time = time.time()
    
    results = []
    for i, dataset in enumerate(datasets):
        print(f"  Processing dataset {i+1}/{len(datasets)}")
        result = processor.process_dataset(dataset)
        results.append(result)
    
    end_time = time.time()
    print(f"✅ Dataset processing completed in {end_time - start_time:.2f} seconds")
    
    # Print results
    for i, result in enumerate(results):
        print(f"\n📊 Dataset {i+1} Results:")
        if 'summary' in result:
            summary = result['summary']
            print(f"  - Total batches: {summary.get('total_batches', 0)}")
            print(f"  - Successful batches: {summary.get('successful_batches', 0)}")
            print(f"  - Success rate: {summary.get('success_rate', 0)*100:.1f}%")
    
    return results

def example_operation_management():
    """Example of operation management."""
    print("\n🚀 Example: Operation Management")
    print("-" * 40)
    
    # Create operation manager
    config = BulkOperationConfig(
        max_concurrent_operations=2,
        operation_timeout=300.0,
        enable_operation_queue=True,
        enable_operation_monitoring=True
    )
    
    manager = create_bulk_operation_manager(config)
    print("✅ Operation manager created")
    
    # Create models for operations
    models = create_example_models()[:3]
    
    # Submit multiple operations
    operation_ids = []
    for i in range(3):
        operation_id = submit_bulk_operation(
            OperationType.OPTIMIZATION,
            models,
            config={'optimization_strategy': 'memory'}
        )
        operation_ids.append(operation_id)
        print(f"📝 Submitted operation {i+1}: {operation_id}")
    
    # Monitor operations
    print("\n🔄 Monitoring operations...")
    for i in range(15):  # Monitor for 15 seconds
        print(f"\n--- Status Check {i+1} ---")
        
        for j, op_id in enumerate(operation_ids):
            status = manager.get_operation_status(op_id)
            print(f"  Operation {j+1} ({op_id}): {status}")
        
        # Check if all operations are completed
        all_completed = all(
            manager.get_operation_status(op_id) in [OperationStatus.COMPLETED, OperationStatus.FAILED]
            for op_id in operation_ids
        )
        
        if all_completed:
            print("✅ All operations completed")
            break
        
        time.sleep(1)
    
    # Get operation statistics
    stats = manager.get_operation_statistics()
    print(f"\n📊 Operation Statistics:")
    print(f"  - Total operations: {stats.get('total_operations', 0)}")
    print(f"  - Completed operations: {stats.get('completed_operations', 0)}")
    print(f"  - Failed operations: {stats.get('failed_operations', 0)}")
    print(f"  - Success rate: {stats.get('success_rate', 0)*100:.1f}%")
    
    return operation_ids

def example_simple_bulk_optimization():
    """Example using simple bulk optimization function."""
    print("\n🚀 Example: Simple Bulk Optimization")
    print("-" * 40)
    
    # Create models
    models = create_example_models()[:3]
    
    # Simple bulk optimization
    print("🔄 Running simple bulk optimization...")
    start_time = time.time()
    
    results = optimize_models_bulk_simple(
        models,
        config={
            'max_workers': 1,
            'enable_parallel_processing': False,
            'optimization_strategies': ['memory']
        }
    )
    
    end_time = time.time()
    print(f"✅ Simple bulk optimization completed in {end_time - start_time:.2f} seconds")
    
    # Print results
    successful = [r for r in results if r.success]
    print(f"\n📊 Results:")
    print(f"  - Models processed: {len(results)}")
    print(f"  - Successful optimizations: {len(successful)}")
    
    for result in results:
        status = "✅" if result.success else "❌"
        print(f"  {status} {result.model_name}: {result.optimization_time:.2f}s")
    
    return results

def main():
    """Main function to run all examples."""
    print("🚀 Bulk Optimization System Examples")
    print("=" * 50)
    
    try:
        # Run examples
        example_basic_bulk_optimization()
        example_advanced_bulk_optimization()
        example_bulk_dataset_processing()
        example_operation_management()
        example_simple_bulk_optimization()
        
        print("\n🎉 All examples completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

