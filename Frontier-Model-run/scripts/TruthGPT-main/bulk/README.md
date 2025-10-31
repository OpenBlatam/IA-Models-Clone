# Bulk Optimization System

A comprehensive bulk optimization system adapted from the optimization_core for efficient processing of multiple models and datasets.

## Overview

The bulk optimization system provides advanced capabilities for:
- **Bulk Model Optimization**: Optimize multiple models simultaneously with parallel processing
- **Bulk Data Processing**: Process large datasets efficiently with memory optimization
- **Operation Management**: Coordinate and manage bulk operations with queuing and monitoring
- **Performance Monitoring**: Track system performance and optimization metrics

## Components

### 1. Bulk Optimization Core (`bulk_optimization_core.py`)
- Core optimization engine for bulk model processing
- Parallel and sequential optimization strategies
- Memory and computational optimizations
- Performance monitoring and reporting

### 2. Bulk Data Processor (`bulk_data_processor.py`)
- Efficient bulk dataset processing
- Memory mapping and compression support
- Data augmentation capabilities
- Quality control and validation

### 3. Bulk Operation Manager (`bulk_operation_manager.py`)
- Operation coordination and management
- Queue-based operation processing
- Resource monitoring and management
- Operation persistence and history

### 4. Main Bulk Optimizer (`bulk_optimizer.py`)
- Integrated bulk optimization system
- Coordinates all components
- High-level API for bulk operations
- Result persistence and reporting

## Features

### 🚀 **Parallel Processing**
- Multi-threaded and multi-process optimization
- Configurable worker pools
- Load balancing and resource management

### 🧠 **Memory Optimization**
- Memory pooling and caching
- Gradient accumulation
- Dynamic memory management
- Memory usage monitoring

### 📊 **Performance Monitoring**
- Real-time performance tracking
- System resource monitoring
- Optimization metrics collection
- Detailed performance reports

### 🔄 **Operation Management**
- Operation queuing and scheduling
- Status tracking and monitoring
- Error handling and retry mechanisms
- Operation persistence

### 🎯 **Optimization Strategies**
- Memory optimizations (FP16, quantization, pruning)
- Computational optimizations (kernel fusion, attention fusion)
- MCTS optimization
- Hybrid optimization strategies
- Ultra optimization techniques

## Quick Start

### Basic Usage

```python
from bulk_optimizer import create_bulk_optimizer, optimize_models_bulk_simple

# Create test models
models = [
    ("model_1", your_model_1),
    ("model_2", your_model_2),
    ("model_3", your_model_3)
]

# Simple bulk optimization
results = optimize_models_bulk_simple(models)

# Print results
for result in results:
    print(f"{result.model_name}: {result.success} - {result.optimization_time:.2f}s")
```

### Advanced Usage

```python
from bulk_optimizer import create_bulk_optimizer, OperationType

# Create bulk optimizer with custom configuration
config = {
    'max_models_per_batch': 5,
    'enable_parallel_optimization': True,
    'optimization_strategies': ['memory', 'computational', 'hybrid'],
    'enable_operation_manager': True
}

optimizer = create_bulk_optimizer(config)

# Optimize models
results = optimizer.optimize_models_bulk(models)

# Submit operation
operation_id = optimizer.submit_bulk_operation(
    OperationType.OPTIMIZATION,
    models
)
```

### Dataset Processing

```python
from bulk_data_processor import create_bulk_data_processor, BulkDataset

# Create dataset
dataset = BulkDataset("data.json", BulkDataConfig())

# Process dataset
processor = create_bulk_data_processor()
result = processor.process_dataset(dataset)
```

## Configuration

### Bulk Optimizer Configuration

```python
config = BulkOptimizerConfig(
    # Core settings
    max_models_per_batch=10,
    enable_parallel_optimization=True,
    optimization_strategies=['memory', 'computational', 'mcts'],
    
    # Performance settings
    enable_memory_optimization=True,
    max_memory_gb=16.0,
    enable_gpu_acceleration=True,
    
    # Monitoring
    enable_performance_monitoring=True,
    enable_detailed_logging=True
)
```

### Optimization Core Configuration

```python
config = BulkOptimizationConfig(
    # Processing
    max_workers=4,
    batch_size=8,
    enable_parallel_processing=True,
    
    # Optimization strategies
    optimization_strategies=['memory', 'computational', 'hybrid'],
    
    # Performance
    memory_limit_gb=8.0,
    target_memory_reduction=0.3,
    target_speed_improvement=2.0
)
```

### Data Processor Configuration

```python
config = BulkDataConfig(
    # Processing
    batch_size=32,
    num_workers=4,
    enable_parallel_processing=True,
    
    # Memory management
    max_memory_gb=8.0,
    enable_memory_mapping=True,
    enable_compression=True,
    
    # Data augmentation
    enable_data_augmentation=True,
    augmentation_probability=0.3
)
```

## Examples

### Example 1: Basic Bulk Optimization

```python
import torch
import torch.nn as nn
from bulk_optimizer import optimize_models_bulk_simple

# Create simple models
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)
    
    def forward(self, x):
        return self.linear(x)

# Create models
models = [
    ("model_1", SimpleModel()),
    ("model_2", SimpleModel()),
    ("model_3", SimpleModel())
]

# Optimize
results = optimize_models_bulk_simple(models)
```

### Example 2: Advanced Bulk Optimization

```python
from bulk_optimizer import create_bulk_optimizer, OperationType

# Create optimizer
optimizer = create_bulk_optimizer({
    'max_models_per_batch': 5,
    'enable_parallel_optimization': True,
    'optimization_strategies': ['memory', 'computational', 'hybrid']
})

# Optimize models
results = optimizer.optimize_models_bulk(models)

# Submit operation
operation_id = optimizer.submit_bulk_operation(
    OperationType.OPTIMIZATION,
    models
)
```

### Example 3: Dataset Processing

```python
from bulk_data_processor import create_bulk_data_processor, BulkDataset, BulkDataConfig

# Create dataset
config = BulkDataConfig(batch_size=16, num_workers=2)
dataset = BulkDataset("data.json", config)

# Process dataset
processor = create_bulk_data_processor()
result = processor.process_dataset(dataset)
```

## Testing

Run the comprehensive test suite:

```bash
python test_bulk_optimization.py
```

Run the example usage:

```bash
python example_bulk_usage.py
```

## Performance

### Optimization Performance
- **Parallel Processing**: Up to 4x speedup with parallel optimization
- **Memory Efficiency**: 30-50% memory reduction with memory optimizations
- **Parameter Reduction**: 10-30% parameter reduction with pruning
- **Speed Improvement**: 2-3x speedup with computational optimizations

### Scalability
- **Model Batching**: Process up to 10 models per batch
- **Memory Management**: Automatic memory management up to 16GB
- **Worker Scaling**: Configurable worker pools for parallel processing
- **Queue Management**: Efficient operation queuing and scheduling

## Monitoring

### Performance Metrics
- Optimization time per model
- Memory usage and efficiency
- Parameter reduction rates
- Accuracy scores
- System resource usage

### Operation Tracking
- Operation status and progress
- Success/failure rates
- Performance statistics
- Error reporting and logging

## Integration

The bulk optimization system integrates with:
- **Optimization Core**: Advanced optimization techniques
- **TruthGPT Models**: All TruthGPT model variants
- **Data Processing**: Efficient dataset processing
- **Monitoring Systems**: Performance and resource monitoring

## Requirements

- Python 3.8+
- PyTorch 1.9+
- NumPy
- Psutil
- H5py (optional)
- Zarr (optional)

## License

Part of the TruthGPT optimization system.

## Support

For issues and questions, please refer to the main TruthGPT documentation.

