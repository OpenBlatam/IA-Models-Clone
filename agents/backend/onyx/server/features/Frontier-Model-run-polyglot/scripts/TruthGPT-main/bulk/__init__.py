"""
Bulk Optimization Module - Adapted optimization core for bulk processing
Provides comprehensive bulk optimization capabilities for multiple models and datasets
"""

from .bulk_optimization_core import (
    BulkOptimizationCore, BulkOptimizationConfig, BulkOptimizationResult,
    create_bulk_optimization_core, optimize_models_bulk
)

from .bulk_data_processor import (
    BulkDataProcessor, BulkDataConfig, BulkDataset, BulkDataAugmentation,
    create_bulk_data_processor, process_dataset_bulk
)

from .bulk_operation_manager import (
    BulkOperationManager, BulkOperationConfig, BulkOperation, 
    OperationType, OperationStatus,
    create_bulk_operation_manager, submit_bulk_operation
)

from .bulk_optimizer import (
    BulkOptimizer, BulkOptimizerConfig,
    create_bulk_optimizer, optimize_models_bulk_simple, process_datasets_bulk_simple
)

__all__ = [
    # Bulk Optimization Core
    'BulkOptimizationCore',
    'BulkOptimizationConfig', 
    'BulkOptimizationResult',
    'create_bulk_optimization_core',
    'optimize_models_bulk',
    
    # Bulk Data Processor
    'BulkDataProcessor',
    'BulkDataConfig',
    'BulkDataset',
    'BulkDataAugmentation',
    'create_bulk_data_processor',
    'process_dataset_bulk',
    
    # Bulk Operation Manager
    'BulkOperationManager',
    'BulkOperationConfig',
    'BulkOperation',
    'OperationType',
    'OperationStatus',
    'create_bulk_operation_manager',
    'submit_bulk_operation',
    
    # Main Bulk Optimizer
    'BulkOptimizer',
    'BulkOptimizerConfig',
    'create_bulk_optimizer',
    'optimize_models_bulk_simple',
    'process_datasets_bulk_simple'
]

__version__ = "1.0.0"
__author__ = "TruthGPT Bulk Optimization Team"
__description__ = "Bulk optimization system adapted from optimization_core for efficient bulk processing"

