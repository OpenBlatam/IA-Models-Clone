#!/usr/bin/env python3
"""
Bulk Optimizer - Main bulk optimization system
Integrates bulk optimization core, data processor, and operation manager
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
import time
import json
import logging
import threading
import queue
import concurrent.futures
from pathlib import Path
import psutil
import gc
from collections import defaultdict, deque
import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# Import bulk components
from .bulk_optimization_core import (
    BulkOptimizationCore, BulkOptimizationConfig, BulkOptimizationResult,
    create_bulk_optimization_core, optimize_models_bulk
)
from .bulk_data_processor import (
    BulkDataProcessor, BulkDataConfig, BulkDataset,
    create_bulk_data_processor, process_dataset_bulk
)
from .bulk_operation_manager import (
    BulkOperationManager, BulkOperationConfig, BulkOperation, OperationType, OperationStatus,
    create_bulk_operation_manager, submit_bulk_operation
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BulkOptimizerConfig:
    """Configuration for the main bulk optimizer."""
    # Core components
    enable_optimization_core: bool = True
    enable_data_processor: bool = True
    enable_operation_manager: bool = True
    
    # Optimization settings
    optimization_strategies: List[str] = field(default_factory=lambda: [
        'memory', 'computational', 'mcts', 'hybrid', 'ultra'
    ])
    max_models_per_batch: int = 10
    enable_parallel_optimization: bool = True
    
    # Data processing settings
    batch_size: int = 32
    num_workers: int = 4
    enable_data_augmentation: bool = True
    
    # Operation management
    max_concurrent_operations: int = 3
    operation_timeout: float = 3600.0
    enable_operation_queue: bool = True
    
    # Performance settings
    enable_memory_optimization: bool = True
    max_memory_gb: float = 16.0
    enable_gpu_acceleration: bool = True
    enable_mixed_precision: bool = True
    
    # Monitoring and logging
    enable_performance_monitoring: bool = True
    enable_detailed_logging: bool = True
    enable_progress_tracking: bool = True
    
    # Persistence
    enable_result_persistence: bool = True
    persistence_directory: str = "./bulk_results"
    enable_operation_history: bool = True

class BulkOptimizer:
    """Main bulk optimization system that coordinates all bulk operations."""
    
    def __init__(self, config: BulkOptimizerConfig):
        self.config = config
        self.optimization_history = []
        self.performance_metrics = defaultdict(list)
        
        # Initialize components
        self.optimization_core = None
        self.data_processor = None
        self.operation_manager = None
        
        if self.config.enable_optimization_core:
            opt_config = BulkOptimizationConfig(
                optimization_strategies=self.config.optimization_strategies,
                max_workers=self.config.num_workers,
                enable_parallel_processing=self.config.enable_parallel_optimization
            )
            self.optimization_core = BulkOptimizationCore(opt_config)
        
        if self.config.enable_data_processor:
            data_config = BulkDataConfig(
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers,
                enable_data_augmentation=self.config.enable_data_augmentation
            )
            self.data_processor = BulkDataProcessor(data_config)
        
        if self.config.enable_operation_manager:
            op_config = BulkOperationConfig(
                max_concurrent_operations=self.config.max_concurrent_operations,
                operation_timeout=self.config.operation_timeout,
                enable_operation_queue=self.config.enable_operation_queue
            )
            self.operation_manager = BulkOperationManager(op_config)
        
        # Setup persistence
        if self.config.enable_result_persistence:
            self._setup_persistence()
        
        logger.info("🚀 Bulk Optimizer initialized")
    
    def optimize_models_bulk(self, models: List[Tuple[str, nn.Module]], 
                           strategy: str = 'auto',
                           enable_parallel: bool = True) -> List[BulkOptimizationResult]:
        """Optimize multiple models in bulk."""
        logger.info(f"🚀 Starting bulk optimization of {len(models)} models")
        
        if not self.optimization_core:
            raise RuntimeError("Optimization core not enabled")
        
        # Split models into batches if needed
        if len(models) > self.config.max_models_per_batch:
            model_batches = self._split_models_into_batches(models)
            all_results = []
            
            for batch_idx, batch in enumerate(model_batches):
                logger.info(f"Processing batch {batch_idx + 1}/{len(model_batches)}")
                batch_results = self.optimization_core.optimize_models_bulk(batch, strategy)
                all_results.extend(batch_results)
                
                # Memory cleanup between batches
                if self.config.enable_memory_optimization:
                    gc.collect()
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            return all_results
        else:
            return self.optimization_core.optimize_models_bulk(models, strategy)
    
    def process_datasets_bulk(self, datasets: List[BulkDataset],
                             model: Optional[nn.Module] = None,
                             processing_function: Optional[Callable] = None) -> List[Dict[str, Any]]:
        """Process multiple datasets in bulk."""
        logger.info(f"🚀 Starting bulk processing of {len(datasets)} datasets")
        
        if not self.data_processor:
            raise RuntimeError("Data processor not enabled")
        
        results = []
        for dataset_idx, dataset in enumerate(datasets):
            logger.info(f"Processing dataset {dataset_idx + 1}/{len(datasets)}")
            result = self.data_processor.process_dataset(dataset, model, processing_function)
            results.append(result)
        
        return results
    
    def submit_bulk_operation(self, operation_type: OperationType,
                            models: List[Tuple[str, nn.Module]],
                            datasets: Optional[List[BulkDataset]] = None,
                            config: Optional[Dict[str, Any]] = None) -> str:
        """Submit a bulk operation."""
        if not self.operation_manager:
            raise RuntimeError("Operation manager not enabled")
        
        return self.operation_manager.submit_operation(
            BulkOperation(
                operation_id=f"{operation_type.value}_{int(time.time())}",
                operation_type=operation_type,
                models=models,
                datasets=datasets,
                config=config or {}
            )
        )
    
    def get_operation_status(self, operation_id: str) -> Optional[OperationStatus]:
        """Get status of an operation."""
        if not self.operation_manager:
            return None
        
        return self.operation_manager.get_operation_status(operation_id)
    
    def get_operation_results(self, operation_id: str) -> Optional[Dict[str, Any]]:
        """Get results of a completed operation."""
        if not self.operation_manager:
            return None
        
        return self.operation_manager.get_operation_results(operation_id)
    
    def list_operations(self, status_filter: Optional[OperationStatus] = None) -> List[BulkOperation]:
        """List operations with optional status filter."""
        if not self.operation_manager:
            return []
        
        return self.operation_manager.list_operations(status_filter)
    
    def _split_models_into_batches(self, models: List[Tuple[str, nn.Module]]) -> List[List[Tuple[str, nn.Module]]]:
        """Split models into batches for processing."""
        batches = []
        batch_size = self.config.max_models_per_batch
        
        for i in range(0, len(models), batch_size):
            batch = models[i:i + batch_size]
            batches.append(batch)
        
        return batches
    
    def _setup_persistence(self):
        """Setup result persistence."""
        persistence_dir = Path(self.config.persistence_directory)
        persistence_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"📁 Persistence directory: {persistence_dir}")
    
    def save_optimization_results(self, results: List[BulkOptimizationResult], 
                                 filepath: Optional[str] = None):
        """Save optimization results to file."""
        if not filepath:
            timestamp = int(time.time())
            filepath = f"bulk_optimization_results_{timestamp}.json"
        
        # Convert results to serializable format
        serializable_results = []
        for result in results:
            serializable_results.append({
                'model_name': result.model_name,
                'success': result.success,
                'optimization_time': result.optimization_time,
                'memory_usage': result.memory_usage,
                'parameter_reduction': result.parameter_reduction,
                'accuracy_score': result.accuracy_score,
                'optimizations_applied': result.optimizations_applied,
                'error_message': result.error_message,
                'performance_metrics': result.performance_metrics
            })
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump({
                'timestamp': time.time(),
                'config': self.config.__dict__,
                'results': serializable_results,
                'summary': {
                    'total_models': len(results),
                    'successful_optimizations': len([r for r in results if r.success]),
                    'failed_optimizations': len([r for r in results if not r.success]),
                    'average_optimization_time': np.mean([r.optimization_time for r in results if r.success]) if any(r.success for r in results) else 0,
                    'average_parameter_reduction': np.mean([r.parameter_reduction for r in results if r.success]) if any(r.success for r in results) else 0
                }
            }, f, indent=2)
        
        logger.info(f"📊 Optimization results saved to {filepath}")
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        if not self.operation_manager:
            return {}
        
        return self.operation_manager.get_operation_statistics()
    
    def cleanup_old_results(self, max_age_hours: float = 24.0):
        """Clean up old results and operations."""
        if self.operation_manager:
            self.operation_manager.cleanup_operations(max_age_hours)
        
        logger.info(f"🧹 Cleaned up results older than {max_age_hours} hours")
    
    def shutdown(self):
        """Shutdown the bulk optimizer."""
        logger.info("🛑 Shutting down bulk optimizer")
        
        if self.operation_manager:
            self.operation_manager.shutdown()
        
        # Save final results
        if self.config.enable_result_persistence:
            self._save_final_results()
    
    def _save_final_results(self):
        """Save final results before shutdown."""
        try:
            timestamp = int(time.time())
            results_file = f"bulk_optimizer_final_results_{timestamp}.json"
            
            final_results = {
                'timestamp': timestamp,
                'config': self.config.__dict__,
                'optimization_history': self.optimization_history,
                'performance_metrics': dict(self.performance_metrics),
                'statistics': self.get_optimization_statistics()
            }
            
            with open(results_file, 'w') as f:
                json.dump(final_results, f, indent=2)
            
            logger.info(f"📊 Final results saved to {results_file}")
        except Exception as e:
            logger.error(f"Failed to save final results: {e}")

def create_bulk_optimizer(config: Optional[Dict[str, Any]] = None) -> BulkOptimizer:
    """Create a bulk optimizer instance."""
    if config is None:
        config = {}
    
    optimizer_config = BulkOptimizerConfig(**config)
    return BulkOptimizer(optimizer_config)

def optimize_models_bulk_simple(models: List[Tuple[str, nn.Module]], 
                               config: Optional[Dict[str, Any]] = None) -> List[BulkOptimizationResult]:
    """Simple bulk optimization function."""
    optimizer = create_bulk_optimizer(config)
    return optimizer.optimize_models_bulk(models)

def process_datasets_bulk_simple(datasets: List[BulkDataset],
                                model: Optional[nn.Module] = None,
                                config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Simple bulk dataset processing function."""
    optimizer = create_bulk_optimizer(config)
    return optimizer.process_datasets_bulk(datasets, model)

if __name__ == "__main__":
    print("🚀 Bulk Optimizer")
    print("=" * 40)
    
    # Example usage
    config = {
        'max_models_per_batch': 5,
        'enable_parallel_optimization': True,
        'optimization_strategies': ['memory', 'computational', 'hybrid'],
        'enable_operation_manager': True
    }
    
    optimizer = create_bulk_optimizer(config)
    print(f"✅ Bulk optimizer created with {len(optimizer.config.optimization_strategies)} optimization strategies")
    
    # Example of creating a simple model for testing
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 5)
        
        def forward(self, x):
            return self.linear(x)
    
    # Create test models
    test_models = [
        ("model_1", SimpleModel()),
        ("model_2", SimpleModel()),
        ("model_3", SimpleModel())
    ]
    
    print(f"📝 Created {len(test_models)} test models for optimization")
    print("🚀 Ready for bulk optimization operations!")

