#!/usr/bin/env python3
"""
Bulk Operation Manager - Manages bulk operations and coordinates optimization
Coordinates between bulk optimization core and data processor for efficient bulk operations
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
from enum import Enum
import asyncio
import aiofiles
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# Import bulk components
from .bulk_optimization_core import BulkOptimizationCore, BulkOptimizationConfig, BulkOptimizationResult
from .bulk_data_processor import BulkDataProcessor, BulkDataConfig, BulkDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OperationType(Enum):
    """Types of bulk operations."""
    OPTIMIZATION = "optimization"
    TRAINING = "training"
    INFERENCE = "inference"
    EVALUATION = "evaluation"
    PREPROCESSING = "preprocessing"
    POSTPROCESSING = "postprocessing"

class OperationStatus(Enum):
    """Status of bulk operations."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class BulkOperation:
    """Represents a bulk operation."""
    operation_id: str
    operation_type: OperationType
    models: List[Tuple[str, nn.Module]]
    datasets: Optional[List[BulkDataset]] = None
    config: Dict[str, Any] = field(default_factory=dict)
    status: OperationStatus = OperationStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    results: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    progress: float = 0.0

@dataclass
class BulkOperationConfig:
    """Configuration for bulk operation manager."""
    # Core settings
    max_concurrent_operations: int = 3
    operation_timeout: float = 3600.0  # 1 hour
    enable_operation_queue: bool = True
    queue_size: int = 100
    
    # Resource management
    max_memory_gb: float = 16.0
    max_cpu_usage: float = 80.0
    enable_resource_monitoring: bool = True
    
    # Operation settings
    enable_operation_retry: bool = True
    max_retry_attempts: int = 3
    retry_delay: float = 5.0
    
    # Persistence
    enable_operation_persistence: bool = True
    persistence_directory: str = "./bulk_operations"
    enable_result_caching: bool = True
    cache_ttl: float = 3600.0  # 1 hour
    
    # Monitoring
    enable_operation_monitoring: bool = True
    monitoring_interval: float = 1.0
    enable_detailed_logging: bool = True
    
    # Performance
    enable_async_operations: bool = True
    enable_parallel_execution: bool = True
    enable_operation_pipelining: bool = True

class BulkOperationManager:
    """Manages bulk operations and coordinates optimization processes."""
    
    def __init__(self, config: BulkOperationConfig):
        self.config = config
        self.operations = {}
        self.operation_queue = queue.Queue(maxsize=self.config.queue_size)
        self.active_operations = {}
        self.completed_operations = {}
        self.failed_operations = {}
        
        # Initialize components
        self.optimization_core = BulkOptimizationCore(BulkOptimizationConfig())
        self.data_processor = BulkDataProcessor(BulkDataConfig())
        
        # Monitoring
        self.monitoring_thread = None
        self.stop_monitoring = threading.Event()
        self.operation_stats = defaultdict(list)
        
        # Persistence
        if self.config.enable_operation_persistence:
            self._setup_persistence()
        
        # Start monitoring
        if self.config.enable_operation_monitoring:
            self._start_monitoring()
    
    def submit_operation(self, operation: BulkOperation) -> str:
        """Submit a bulk operation for processing."""
        logger.info(f"📝 Submitting operation {operation.operation_id} of type {operation.operation_type.value}")
        
        # Validate operation
        if not self._validate_operation(operation):
            raise ValueError(f"Invalid operation {operation.operation_id}")
        
        # Add to operations registry
        self.operations[operation.operation_id] = operation
        
        # Add to queue if enabled
        if self.config.enable_operation_queue:
            try:
                self.operation_queue.put(operation, timeout=1.0)
            except queue.Full:
                raise RuntimeError("Operation queue is full")
        
        # Start processing if not queued
        if not self.config.enable_operation_queue:
            self._process_operation(operation)
        
        return operation.operation_id
    
    def get_operation_status(self, operation_id: str) -> Optional[OperationStatus]:
        """Get status of an operation."""
        if operation_id in self.operations:
            return self.operations[operation_id].status
        return None
    
    def get_operation_results(self, operation_id: str) -> Optional[Dict[str, Any]]:
        """Get results of a completed operation."""
        if operation_id in self.completed_operations:
            return self.completed_operations[operation_id].results
        return None
    
    def cancel_operation(self, operation_id: str) -> bool:
        """Cancel a running operation."""
        if operation_id in self.active_operations:
            operation = self.active_operations[operation_id]
            operation.status = OperationStatus.CANCELLED
            operation.completed_at = time.time()
            
            # Move to completed operations
            self.completed_operations[operation_id] = operation
            del self.active_operations[operation_id]
            
            logger.info(f"🚫 Cancelled operation {operation_id}")
            return True
        return False
    
    def list_operations(self, status_filter: Optional[OperationStatus] = None) -> List[BulkOperation]:
        """List operations with optional status filter."""
        operations = list(self.operations.values())
        
        if status_filter:
            operations = [op for op in operations if op.status == status_filter]
        
        return operations
    
    def _validate_operation(self, operation: BulkOperation) -> bool:
        """Validate an operation before processing."""
        if not operation.operation_id:
            return False
        
        if not operation.models:
            return False
        
        if operation.operation_type not in OperationType:
            return False
        
        return True
    
    def _process_operation(self, operation: BulkOperation):
        """Process a bulk operation."""
        logger.info(f"🔄 Processing operation {operation.operation_id}")
        
        operation.status = OperationStatus.RUNNING
        operation.started_at = time.time()
        self.active_operations[operation.operation_id] = operation
        
        try:
            # Route to appropriate processor
            if operation.operation_type == OperationType.OPTIMIZATION:
                results = self._process_optimization_operation(operation)
            elif operation.operation_type == OperationType.TRAINING:
                results = self._process_training_operation(operation)
            elif operation.operation_type == OperationType.INFERENCE:
                results = self._process_inference_operation(operation)
            elif operation.operation_type == OperationType.EVALUATION:
                results = self._process_evaluation_operation(operation)
            elif operation.operation_type == OperationType.PREPROCESSING:
                results = self._process_preprocessing_operation(operation)
            elif operation.operation_type == OperationType.POSTPROCESSING:
                results = self._process_postprocessing_operation(operation)
            else:
                raise ValueError(f"Unknown operation type: {operation.operation_type}")
            
            # Update operation
            operation.status = OperationStatus.COMPLETED
            operation.completed_at = time.time()
            operation.results = results
            operation.progress = 100.0
            
            # Move to completed operations
            self.completed_operations[operation.operation_id] = operation
            del self.active_operations[operation.operation_id]
            
            logger.info(f"✅ Completed operation {operation.operation_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed operation {operation.operation_id}: {e}")
            
            operation.status = OperationStatus.FAILED
            operation.completed_at = time.time()
            operation.error_message = str(e)
            operation.progress = 0.0
            
            # Move to failed operations
            self.failed_operations[operation.operation_id] = operation
            del self.active_operations[operation.operation_id]
    
    def _process_optimization_operation(self, operation: BulkOperation) -> Dict[str, Any]:
        """Process optimization operation."""
        results = self.optimization_core.optimize_models_bulk(
            operation.models,
            operation.config.get('optimization_strategy', 'auto')
        )
        
        return {
            'operation_type': 'optimization',
            'results': [r.__dict__ for r in results],
            'success_count': len([r for r in results if r.success]),
            'total_count': len(results)
        }
    
    def _process_training_operation(self, operation: BulkOperation) -> Dict[str, Any]:
        """Process training operation."""
        # This would implement bulk training
        return {
            'operation_type': 'training',
            'message': 'Training operation not yet implemented',
            'models_count': len(operation.models)
        }
    
    def _process_inference_operation(self, operation: BulkOperation) -> Dict[str, Any]:
        """Process inference operation."""
        # This would implement bulk inference
        return {
            'operation_type': 'inference',
            'message': 'Inference operation not yet implemented',
            'models_count': len(operation.models)
        }
    
    def _process_evaluation_operation(self, operation: BulkOperation) -> Dict[str, Any]:
        """Process evaluation operation."""
        # This would implement bulk evaluation
        return {
            'operation_type': 'evaluation',
            'message': 'Evaluation operation not yet implemented',
            'models_count': len(operation.models)
        }
    
    def _process_preprocessing_operation(self, operation: BulkOperation) -> Dict[str, Any]:
        """Process preprocessing operation."""
        if not operation.datasets:
            raise ValueError("Preprocessing operation requires datasets")
        
        results = []
        for dataset in operation.datasets:
            result = self.data_processor.process_dataset(dataset)
            results.append(result)
        
        return {
            'operation_type': 'preprocessing',
            'results': results,
            'datasets_count': len(operation.datasets)
        }
    
    def _process_postprocessing_operation(self, operation: BulkOperation) -> Dict[str, Any]:
        """Process postprocessing operation."""
        # This would implement bulk postprocessing
        return {
            'operation_type': 'postprocessing',
            'message': 'Postprocessing operation not yet implemented',
            'models_count': len(operation.models)
        }
    
    def _start_monitoring(self):
        """Start operation monitoring."""
        if self.monitoring_thread is None or not self.monitoring_thread.is_alive():
            self.stop_monitoring.clear()
            self.monitoring_thread = threading.Thread(target=self._monitor_operations)
            self.monitoring_thread.start()
    
    def _stop_monitoring(self):
        """Stop operation monitoring."""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.stop_monitoring.set()
            self.monitoring_thread.join()
    
    def _monitor_operations(self):
        """Monitor operations and system resources."""
        while not self.stop_monitoring.is_set():
            try:
                # Monitor system resources
                memory_usage = psutil.virtual_memory().percent
                cpu_usage = psutil.cpu_percent()
                
                self.operation_stats['memory'].append(memory_usage)
                self.operation_stats['cpu'].append(cpu_usage)
                
                # Check for timeout operations
                current_time = time.time()
                for operation_id, operation in list(self.active_operations.items()):
                    if current_time - operation.started_at > self.config.operation_timeout:
                        logger.warning(f"⏰ Operation {operation_id} timed out")
                        self.cancel_operation(operation_id)
                
                # Process queued operations
                if self.config.enable_operation_queue:
                    self._process_queued_operations()
                
                time.sleep(self.config.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring: {e}")
                break
    
    def _process_queued_operations(self):
        """Process queued operations."""
        while not self.operation_queue.empty() and len(self.active_operations) < self.config.max_concurrent_operations:
            try:
                operation = self.operation_queue.get_nowait()
                self._process_operation(operation)
            except queue.Empty:
                break
            except Exception as e:
                logger.error(f"Error processing queued operation: {e}")
    
    def _setup_persistence(self):
        """Setup operation persistence."""
        persistence_dir = Path(self.config.persistence_directory)
        persistence_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing operations
        self._load_persisted_operations()
    
    def _load_persisted_operations(self):
        """Load persisted operations from disk."""
        persistence_dir = Path(self.config.persistence_directory)
        operations_file = persistence_dir / "operations.json"
        
        if operations_file.exists():
            try:
                with open(operations_file, 'r') as f:
                    operations_data = json.load(f)
                
                for op_data in operations_data:
                    operation = BulkOperation(**op_data)
                    self.operations[operation.operation_id] = operation
                
                logger.info(f"📂 Loaded {len(operations_data)} persisted operations")
            except Exception as e:
                logger.error(f"Failed to load persisted operations: {e}")
    
    def _save_operations(self):
        """Save operations to disk."""
        if not self.config.enable_operation_persistence:
            return
        
        persistence_dir = Path(self.config.persistence_directory)
        operations_file = persistence_dir / "operations.json"
        
        try:
            operations_data = [op.__dict__ for op in self.operations.values()]
            with open(operations_file, 'w') as f:
                json.dump(operations_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save operations: {e}")
    
    def get_operation_statistics(self) -> Dict[str, Any]:
        """Get operation statistics."""
        total_operations = len(self.operations)
        completed_operations = len(self.completed_operations)
        failed_operations = len(self.failed_operations)
        active_operations = len(self.active_operations)
        
        return {
            'total_operations': total_operations,
            'completed_operations': completed_operations,
            'failed_operations': failed_operations,
            'active_operations': active_operations,
            'success_rate': completed_operations / total_operations if total_operations > 0 else 0,
            'performance_metrics': dict(self.operation_stats)
        }
    
    def cleanup_operations(self, max_age_hours: float = 24.0):
        """Clean up old operations."""
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        # Clean up completed operations
        to_remove = []
        for operation_id, operation in self.completed_operations.items():
            if current_time - operation.completed_at > max_age_seconds:
                to_remove.append(operation_id)
        
        for operation_id in to_remove:
            del self.completed_operations[operation_id]
            if operation_id in self.operations:
                del self.operations[operation_id]
        
        logger.info(f"🧹 Cleaned up {len(to_remove)} old operations")
    
    def shutdown(self):
        """Shutdown the operation manager."""
        logger.info("🛑 Shutting down bulk operation manager")
        
        # Stop monitoring
        if self.config.enable_operation_monitoring:
            self._stop_monitoring()
        
        # Save operations
        self._save_operations()
        
        # Cancel active operations
        for operation_id in list(self.active_operations.keys()):
            self.cancel_operation(operation_id)

def create_bulk_operation_manager(config: Optional[Dict[str, Any]] = None) -> BulkOperationManager:
    """Create a bulk operation manager instance."""
    if config is None:
        config = {}
    
    operation_config = BulkOperationConfig(**config)
    return BulkOperationManager(operation_config)

def submit_bulk_operation(operation_type: OperationType,
                          models: List[Tuple[str, nn.Module]],
                          config: Optional[Dict[str, Any]] = None,
                          datasets: Optional[List[BulkDataset]] = None) -> str:
    """Convenience function for submitting bulk operations."""
    manager = create_bulk_operation_manager(config)
    
    operation = BulkOperation(
        operation_id=f"{operation_type.value}_{int(time.time())}",
        operation_type=operation_type,
        models=models,
        datasets=datasets,
        config=config or {}
    )
    
    return manager.submit_operation(operation)

if __name__ == "__main__":
    print("🚀 Bulk Operation Manager")
    print("=" * 40)
    
    # Example usage
    config = {
        'max_concurrent_operations': 2,
        'operation_timeout': 1800.0,
        'enable_operation_queue': True
    }
    
    manager = create_bulk_operation_manager(config)
    print(f"✅ Bulk operation manager created with {manager.config.max_concurrent_operations} max concurrent operations")

