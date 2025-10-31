#!/usr/bin/env python3
"""
Run Bulk Optimization - Main script for running bulk optimization operations
Demonstrates the complete bulk optimization system
"""

import torch
import torch.nn as nn
import time
import json
import sys
import os
from typing import List, Tuple
import argparse

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

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

def create_demo_models(num_models: int = 5) -> List[Tuple[str, nn.Module]]:
    """Create demo models for bulk optimization."""
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
    
    # Create models
    for i in range(num_models):
        if i % 3 == 0:
            models.append((f"SimpleLinear_{i}", SimpleLinear()))
        elif i % 3 == 1:
            models.append((f"MLP_{i}", MLPModel()))
        else:
            models.append((f"CNN_{i}", CNNModel()))
    
    return models

def run_simple_bulk_optimization(num_models: int = 5):
    """Run simple bulk optimization."""
    print("🚀 Running Simple Bulk Optimization")
    print("=" * 50)
    
    # Create models
    models = create_demo_models(num_models)
    print(f"Created {len(models)} demo models")
    
    # Run simple bulk optimization
    print("🔄 Starting bulk optimization...")
    start_time = time.time()
    
    results = optimize_models_bulk_simple(
        models,
        config={
            'max_workers': 2,
            'enable_parallel_processing': True,
            'optimization_strategies': ['memory', 'computational']
        }
    )
    
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
        import numpy as np
        avg_time = np.mean([r.optimization_time for r in successful])
        avg_reduction = np.mean([r.parameter_reduction for r in successful])
        print(f"  - Average optimization time: {avg_time:.2f}s")
        print(f"  - Average parameter reduction: {avg_reduction:.2%}")
    
    # Print detailed results
    print(f"\n📋 Detailed Results:")
    for result in results:
        status = "✅" if result.success else "❌"
        print(f"  {status} {result.model_name}:")
        print(f"    - Optimization time: {result.optimization_time:.2f}s")
        print(f"    - Memory usage: {result.memory_usage:.2f}MB")
        print(f"    - Parameter reduction: {result.parameter_reduction:.2%}")
        print(f"    - Accuracy score: {result.accuracy_score:.3f}")
        if result.optimizations_applied:
            print(f"    - Optimizations: {', '.join(result.optimizations_applied)}")
        if result.error_message:
            print(f"    - Error: {result.error_message}")
    
    return results

def run_advanced_bulk_optimization(num_models: int = 5):
    """Run advanced bulk optimization with full system."""
    print("\n🚀 Running Advanced Bulk Optimization")
    print("=" * 50)
    
    # Create models
    models = create_demo_models(num_models)
    print(f"Created {len(models)} demo models")
    
    # Advanced configuration
    config = BulkOptimizerConfig(
        max_models_per_batch=3,
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
    
    # Optimize models
    print("🔄 Starting advanced bulk optimization...")
    start_time = time.time()
    
    results = optimizer.optimize_models_bulk(models)
    
    end_time = time.time()
    print(f"✅ Advanced bulk optimization completed in {end_time - start_time:.2f} seconds")
    
    # Print results
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    
    print(f"\n📊 Results Summary:")
    print(f"  - Total models: {len(results)}")
    print(f"  - Successful: {len(successful)}")
    print(f"  - Failed: {len(failed)}")
    print(f"  - Success rate: {len(successful)/len(results)*100:.1f}%")
    
    # Get optimization statistics
    stats = optimizer.get_optimization_statistics()
    if stats:
        print(f"\n📈 Optimization Statistics:")
        print(f"  - Total operations: {stats.get('total_operations', 0)}")
        print(f"  - Completed operations: {stats.get('completed_operations', 0)}")
        print(f"  - Success rate: {stats.get('success_rate', 0)*100:.1f}%")
    
    # Save results
    if config.enable_result_persistence:
        timestamp = int(time.time())
        results_file = f"bulk_optimization_results_{timestamp}.json"
        optimizer.save_optimization_results(results, results_file)
        print(f"📊 Results saved to {results_file}")
    
    return results

def run_operation_management_demo():
    """Run operation management demo."""
    print("\n🚀 Running Operation Management Demo")
    print("=" * 50)
    
    # Create models
    models = create_demo_models(3)
    
    # Create operation manager
    from bulk_operation_manager import create_bulk_operation_manager, BulkOperationConfig
    
    config = BulkOperationConfig(
        max_concurrent_operations=2,
        operation_timeout=300.0,
        enable_operation_queue=True,
        enable_operation_monitoring=True
    )
    
    manager = create_bulk_operation_manager(config)
    print("✅ Operation manager created")
    
    # Submit operations
    from bulk_operation_manager import submit_bulk_operation, OperationType
    
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
    for i in range(10):  # Monitor for 10 seconds
        print(f"\n--- Status Check {i+1} ---")
        
        for j, op_id in enumerate(operation_ids):
            status = manager.get_operation_status(op_id)
            print(f"  Operation {j+1} ({op_id}): {status}")
        
        # Check if all operations are completed
        from bulk_operation_manager import OperationStatus
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

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Run bulk optimization operations')
    parser.add_argument('--mode', choices=['simple', 'advanced', 'operations', 'all'], 
                       default='all', help='Mode to run')
    parser.add_argument('--models', type=int, default=5, 
                       help='Number of models to create')
    parser.add_argument('--verbose', action='store_true', 
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    print("🚀 Bulk Optimization System")
    print("=" * 50)
    print(f"Mode: {args.mode}")
    print(f"Models: {args.models}")
    print(f"Verbose: {args.verbose}")
    print()
    
    try:
        if args.mode in ['simple', 'all']:
            run_simple_bulk_optimization(args.models)
        
        if args.mode in ['advanced', 'all']:
            run_advanced_bulk_optimization(args.models)
        
        if args.mode in ['operations', 'all']:
            run_operation_management_demo()
        
        print("\n🎉 All operations completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error running bulk optimization: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

