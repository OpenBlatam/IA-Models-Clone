# 🚀 Efficient Data Loading Guide
## Production-Ready PyTorch DataLoader System

### Table of Contents
1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Core Components](#core-components)
4. [Advanced Features](#advanced-features)
5. [Performance Optimization](#performance-optimization)
6. [Production Deployment](#production-deployment)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

---

## Overview

This guide covers the production-ready efficient data loading system for Blatam Academy's AI infrastructure. The system provides:

- **Optimized DataLoading**: PyTorch DataLoader with advanced optimizations
- **Intelligent Caching**: Memory, disk, and hybrid caching strategies
- **Multi-Format Support**: CSV, JSON, HDF5, LMDB, Parquet, and more
- **Streaming Datasets**: For large datasets that don't fit in memory
- **Performance Profiling**: Automated optimization and benchmarking
- **Production Features**: GPU optimization, distributed loading, monitoring

---

## Quick Start

### 1. Installation

```bash
pip install -r requirements_training_evaluation.txt
```

### 2. Basic Data Loading

```python
import asyncio
from efficient_data_loader import quick_dataloader, DataFormat

# Quick data loading
dataset, dataloader = await quick_dataloader(
    data_path="data/sentiment_dataset.csv",
    data_format=DataFormat.CSV,
    batch_size=32,
    num_workers=4
)

print(f"Dataset size: {len(dataset)}")
print(f"DataLoader batches: {len(dataloader)}")

# Test batch loading
for batch in dataloader:
    print(f"Batch keys: {batch.keys()}")
    break
```

### 3. Train/Val/Test Split

```python
from efficient_data_loader import quick_dataloader_split

# Create split DataLoaders
train_loader, val_loader, test_loader = await quick_dataloader_split(
    data_path="data/sentiment_dataset.csv",
    data_format=DataFormat.CSV,
    batch_size=32,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1
)

print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")
print(f"Test batches: {len(test_loader)}")
```

---

## Core Components

### DataLoaderConfig

```python
from efficient_data_loader import DataLoaderConfig, CacheStrategy

config = DataLoaderConfig(
    batch_size=32,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
    shuffle=True,
    
    # Caching
    cache_strategy=CacheStrategy.MEMORY,
    cache_dir="cache",
    cache_size_gb=10.0,
    
    # Performance
    pin_memory_device="cuda",
    non_blocking=True
)
```

### OptimizedTextDataset

```python
from efficient_data_loader import OptimizedTextDataset
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Create optimized dataset
dataset = OptimizedTextDataset(
    texts=["Text 1", "Text 2", "Text 3"],
    labels=[0, 1, 0],
    tokenizer=tokenizer,
    max_length=512,
    cache_encodings=True  # Pre-tokenize for speed
)

# Access items
item = dataset[0]
print(f"Input IDs shape: {item['input_ids'].shape}")
print(f"Attention mask shape: {item['attention_mask'].shape}")
print(f"Label: {item['labels']}")
```

### CachedDataset

```python
from efficient_data_loader import CachedDataset, CacheStrategy

# Create base dataset
base_dataset = OptimizedTextDataset(texts, labels)

# Add caching
cached_dataset = CachedDataset(
    base_dataset,
    cache_strategy=CacheStrategy.HYBRID,  # Memory + disk
    cache_dir="cache",
    cache_size_gb=5.0
)

# First access: loads from disk, caches in memory
item1 = cached_dataset[0]

# Second access: loads from memory (fast)
item2 = cached_dataset[0]
```

### DataLoaderManager

```python
from efficient_data_loader import DataLoaderManager, DeviceManager

# Initialize manager
device_manager = DeviceManager()
manager = await create_data_loader_manager(device_manager)

# Load dataset
dataset, dataloader = await manager.load_dataset(
    data_path="data/sentiment.csv",
    data_format=DataFormat.CSV,
    config=DataLoaderConfig(batch_size=32)
)

# Split dataset
train_dataset, val_dataset, test_dataset = manager.split_dataset(
    dataset,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1
)

# Create DataLoaders
train_loader, val_loader, test_loader = manager.create_dataloaders(
    train_dataset, val_dataset, test_dataset,
    DataLoaderConfig(batch_size=32, shuffle=True)
)
```

---

## Advanced Features

### 1. Streaming Datasets

For large datasets that don't fit in memory:

```python
from efficient_data_loader import StreamingDataset, DataFormat

# Create streaming dataset
streaming_dataset = StreamingDataset(
    data_path="large_dataset.h5",
    data_format=DataFormat.HDF5,
    chunk_size=1000,
    shuffle=True
)

# Create DataLoader
dataloader = DataLoader(
    streaming_dataset,
    batch_size=32,
    num_workers=4
)

# Iterate over chunks
for batch in dataloader:
    process_batch(batch)
```

### 2. Multi-Format Support

```python
# CSV
dataset, loader = await manager.load_dataset(
    "data.csv", DataFormat.CSV, config
)

# JSON
dataset, loader = await manager.load_dataset(
    "data.json", DataFormat.JSON, config
)

# HDF5
dataset, loader = await manager.load_dataset(
    "data.h5", DataFormat.HDF5, config
)

# LMDB
dataset, loader = await manager.load_dataset(
    "data.lmdb", DataFormat.LMDB, config
)

# Parquet
dataset, loader = await manager.load_dataset(
    "data.parquet", DataFormat.PARQUET, config
)
```

### 3. Distributed Data Loading

```python
from efficient_data_loader import DataLoaderFactory

factory = DataLoaderFactory(device_manager)

# Create distributed DataLoader
distributed_loader = factory.create_distributed_dataloader(
    dataset,
    config,
    world_size=4,  # Number of processes
    rank=0  # Current process rank
)
```

### 4. Weighted Sampling

```python
# Create weights for imbalanced dataset
weights = [1.0 if label == 0 else 5.0 for label in labels]

# Create weighted DataLoader
weighted_loader = factory.create_weighted_dataloader(
    dataset,
    config,
    weights=weights
)
```

### 5. Performance Profiling

```python
from efficient_data_loader import DataLoaderProfiler

# Create profiler
profiler = DataLoaderProfiler()

# Profile DataLoader
metrics = profiler.profile_dataloader(dataloader, num_batches=10)

print(f"Throughput: {metrics['throughput_samples_per_sec']:.2f} samples/sec")
print(f"Avg batch time: {metrics['avg_batch_time']:.4f} seconds")

# Optimize configuration
optimized_config = profiler.optimize_dataloader_config(
    current_config,
    target_throughput=1000
)
```

---

## Performance Optimization

### 1. Auto-Optimization

```python
# Let the system auto-optimize
config = DataLoaderConfig(
    num_workers=-1,  # Auto-detect
    batch_size=32,
    pin_memory=True
)

# The system will:
# - Auto-detect optimal number of workers
# - Optimize batch size for GPU memory
# - Enable persistent workers
# - Set optimal prefetch factor
```

### 2. Caching Strategies

```python
# Memory caching (fastest, limited by RAM)
config.cache_strategy = CacheStrategy.MEMORY

# Disk caching (slower, unlimited size)
config.cache_strategy = CacheStrategy.DISK

# Hybrid caching (best of both)
config.cache_strategy = CacheStrategy.HYBRID
```

### 3. GPU Optimization

```python
config = DataLoaderConfig(
    pin_memory=True,
    pin_memory_device="cuda",
    non_blocking=True,
    pin_memory_batch_size=64  # Optimize for GPU
)
```

### 4. Worker Optimization

```python
# Optimal worker count based on system
import multiprocessing as mp

cpu_count = mp.cpu_count()
memory_gb = psutil.virtual_memory().total / (1024**3)

# Heuristic: 1 worker per 2GB RAM, max 8 workers
optimal_workers = min(cpu_count, int(memory_gb / 2), 8)

config = DataLoaderConfig(num_workers=optimal_workers)
```

### 5. Batch Size Optimization

```python
# Start with small batch size
config.batch_size = 16

# Profile and increase
profiler = DataLoaderProfiler()
metrics = profiler.profile_dataloader(dataloader)

if metrics['throughput_samples_per_sec'] < target_throughput:
    config.batch_size *= 2
    # Recreate DataLoader with new batch size
```

---

## Production Deployment

### 1. Docker Configuration

```dockerfile
FROM pytorch/pytorch:2.0.0-cuda11.8-cudnn8-runtime

# Install dependencies
COPY requirements_training_evaluation.txt .
RUN pip install -r requirements_training_evaluation.txt

# Set environment variables
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

# Copy application
COPY . /app
WORKDIR /app

# Run with optimized settings
CMD ["python", "serve_dataloader.py"]
```

### 2. Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: dataloader-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: dataloader
  template:
    metadata:
      labels:
        app: dataloader
    spec:
      containers:
      - name: dataloader-service
        image: blatam/dataloader:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
        env:
        - name: OMP_NUM_THREADS
          value: "1"
        - name: MKL_NUM_THREADS
          value: "1"
```

### 3. Monitoring & Observability

```python
import structlog
import prometheus_client
from efficient_data_loader import DataLoaderProfiler

# Structured logging
logger = structlog.get_logger()

# Metrics
dataloader_requests = prometheus_client.Counter(
    'dataloader_requests_total', 'Total DataLoader requests'
)
dataloader_duration = prometheus_client.Histogram(
    'dataloader_duration_seconds', 'DataLoader duration'
)

# Profiling
profiler = DataLoaderProfiler()

@dataloader_duration.time()
async def load_data_with_monitoring(data_path, config):
    dataloader_requests.inc()
    
    # Load data
    dataset, dataloader = await manager.load_dataset(data_path, config)
    
    # Profile performance
    metrics = profiler.profile_dataloader(dataloader, num_batches=5)
    
    logger.info("Data loaded successfully", 
                dataset_size=len(dataset),
                throughput=metrics['throughput_samples_per_sec'])
    
    return dataset, dataloader
```

### 4. Caching Strategy

```python
# Production caching configuration
config = DataLoaderConfig(
    cache_strategy=CacheStrategy.HYBRID,
    cache_dir="/data/cache",  # Persistent storage
    cache_size_gb=50.0,  # Large cache for production
    
    # Performance
    batch_size=64,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=3
)
```

---

## Best Practices

### 1. Data Preparation

```python
def prepare_data_for_loading(data_path):
    """Prepare data for efficient loading."""
    # Validate data
    df = pd.read_csv(data_path)
    
    # Check for issues
    assert df.isnull().sum().sum() == 0, "Data contains missing values"
    assert len(df) > 0, "Dataset is empty"
    
    # Optimize data types
    df['label'] = df['label'].astype('int32')
    df['text'] = df['text'].astype('string')
    
    # Save optimized data
    optimized_path = data_path.replace('.csv', '_optimized.parquet')
    df.to_parquet(optimized_path, index=False)
    
    return optimized_path
```

### 2. Memory Management

```python
# Monitor memory usage
import psutil
import GPUtil

def monitor_resources():
    """Monitor system resources."""
    cpu_percent = psutil.cpu_percent()
    memory_percent = psutil.virtual_memory().percent
    
    gpu_memory = None
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu_memory = gpus[0].memoryUtil * 100
    except:
        pass
    
    return {
        'cpu_percent': cpu_percent,
        'memory_percent': memory_percent,
        'gpu_memory_percent': gpu_memory
    }

# Use in DataLoader configuration
resources = monitor_resources()
if resources['memory_percent'] > 80:
    config.cache_strategy = CacheStrategy.DISK
else:
    config.cache_strategy = CacheStrategy.MEMORY
```

### 3. Error Handling

```python
async def robust_data_loading(data_path, config, max_retries=3):
    """Robust data loading with retries."""
    for attempt in range(max_retries):
        try:
            dataset, dataloader = await manager.load_dataset(
                data_path, DataFormat.CSV, config
            )
            return dataset, dataloader
            
        except FileNotFoundError:
            logger.error(f"Data file not found: {data_path}")
            raise
            
        except Exception as e:
            logger.warning(f"Data loading attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(2 ** attempt)  # Exponential backoff
```

### 4. Performance Tuning

```python
def optimize_for_system():
    """Optimize DataLoader for current system."""
    # Get system info
    cpu_count = mp.cpu_count()
    memory_gb = psutil.virtual_memory().total / (1024**3)
    
    # Optimize configuration
    config = DataLoaderConfig()
    
    # Workers: 1 per 2GB RAM, max 8
    config.num_workers = min(cpu_count, int(memory_gb / 2), 8)
    
    # Batch size: based on memory
    if memory_gb > 16:
        config.batch_size = 64
    elif memory_gb > 8:
        config.batch_size = 32
    else:
        config.batch_size = 16
    
    # Caching: based on available memory
    if memory_gb > 32:
        config.cache_strategy = CacheStrategy.MEMORY
        config.cache_size_gb = memory_gb * 0.3
    else:
        config.cache_strategy = CacheStrategy.HYBRID
        config.cache_size_gb = 5.0
    
    return config
```

### 5. Data Validation

```python
def validate_dataset(dataset, expected_columns=None):
    """Validate dataset integrity."""
    # Check dataset size
    assert len(dataset) > 0, "Dataset is empty"
    
    # Check sample item
    sample = dataset[0]
    assert isinstance(sample, dict), "Dataset items should be dictionaries"
    
    # Check required columns
    if expected_columns:
        for col in expected_columns:
            assert col in sample, f"Missing column: {col}"
    
    # Check data types
    if 'labels' in sample:
        assert isinstance(sample['labels'], torch.Tensor), "Labels should be tensors"
    
    if 'text' in sample:
        assert isinstance(sample['text'], str), "Text should be strings"
    
    return True
```

---

## Troubleshooting

### Common Issues

#### 1. Out of Memory (OOM)
```python
# Solutions:
config.batch_size = config.batch_size // 2  # Reduce batch size
config.cache_strategy = CacheStrategy.DISK  # Use disk caching
config.num_workers = 1  # Reduce workers
```

#### 2. Slow Data Loading
```python
# Solutions:
config.num_workers = min(mp.cpu_count(), 8)  # Increase workers
config.prefetch_factor = 3  # Increase prefetch
config.persistent_workers = True  # Enable persistent workers
config.pin_memory = True  # Enable pin memory
```

#### 3. High CPU Usage
```python
# Solutions:
config.num_workers = max(1, mp.cpu_count() // 2)  # Reduce workers
config.prefetch_factor = 1  # Reduce prefetch
config.persistent_workers = False  # Disable persistent workers
```

#### 4. Cache Issues
```python
# Solutions:
# Clear cache
import shutil
shutil.rmtree("cache", ignore_errors=True)

# Use different cache strategy
config.cache_strategy = CacheStrategy.NONE  # Disable caching
```

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Profile DataLoader
profiler = DataLoaderProfiler()
metrics = profiler.profile_dataloader(dataloader, num_batches=5)

# Check system resources
resources = monitor_resources()
print(f"System resources: {resources}")

# Validate dataset
validate_dataset(dataset, expected_columns=['text', 'labels'])
```

### Performance Profiling

```python
# Detailed profiling
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Run data loading
dataset, dataloader = await manager.load_dataset(data_path, config)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)  # Top 10 functions
```

---

## API Reference

### DataLoaderConfig
```python
@dataclass
class DataLoaderConfig:
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    drop_last: bool = False
    shuffle: bool = True
    cache_strategy: CacheStrategy = CacheStrategy.MEMORY
    cache_dir: str = "cache"
    cache_size_gb: float = 10.0
```

### DataLoaderManager
```python
class DataLoaderManager:
    async def load_dataset(self, data_path, data_format, config) -> Tuple[Dataset, DataLoader]
    def split_dataset(self, dataset, train_ratio, val_ratio, test_ratio) -> Tuple[Dataset, Dataset, Dataset]
    def create_dataloaders(self, train_dataset, val_dataset, test_dataset, config) -> Tuple[DataLoader, DataLoader, DataLoader]
    def get_dataloader_stats(self, dataloader) -> Dict[str, Any]
```

### DataLoaderFactory
```python
class DataLoaderFactory:
    def create_dataloader(self, dataset, config, sampler=None) -> DataLoader
    def create_distributed_dataloader(self, dataset, config, world_size, rank) -> DataLoader
    def create_weighted_dataloader(self, dataset, config, weights) -> DataLoader
```

### DataLoaderProfiler
```python
class DataLoaderProfiler:
    def profile_dataloader(self, dataloader, num_batches) -> Dict[str, float]
    def optimize_dataloader_config(self, current_config, target_throughput) -> DataLoaderConfig
```

---

## Examples

### Complete Data Loading Pipeline

```python
import asyncio
from efficient_data_loader import (
    DataLoaderManager, DataLoaderConfig, DataFormat, CacheStrategy,
    create_data_loader_manager
)

async def complete_data_pipeline():
    # 1. Initialize
    device_manager = DeviceManager()
    manager = await create_data_loader_manager(device_manager)
    
    # 2. Configuration
    config = DataLoaderConfig(
        batch_size=32,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        cache_strategy=CacheStrategy.HYBRID,
        cache_dir="cache",
        cache_size_gb=10.0
    )
    
    # 3. Load dataset
    dataset, dataloader = await manager.load_dataset(
        "data/sentiment.csv",
        DataFormat.CSV,
        config
    )
    
    # 4. Split dataset
    train_dataset, val_dataset, test_dataset = manager.split_dataset(
        dataset,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1
    )
    
    # 5. Create DataLoaders
    train_loader, val_loader, test_loader = manager.create_dataloaders(
        train_dataset, val_dataset, test_dataset, config
    )
    
    # 6. Get statistics
    stats = manager.get_dataloader_stats(train_loader)
    
    return {
        'dataset': dataset,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'stats': stats
    }

# Run pipeline
result = await complete_data_pipeline()
print(f"Pipeline completed: {result['stats']}")
```

### Performance Optimization

```python
async def optimize_data_loading(data_path, target_throughput=1000):
    """Optimize data loading for target throughput."""
    device_manager = DeviceManager()
    manager = await create_data_loader_manager(device_manager)
    profiler = DataLoaderProfiler()
    
    # Start with default config
    config = DataLoaderConfig()
    
    # Iterative optimization
    for iteration in range(5):
        # Load data
        dataset, dataloader = await manager.load_dataset(
            data_path, DataFormat.CSV, config
        )
        
        # Profile performance
        metrics = profiler.profile_dataloader(dataloader, num_batches=10)
        
        print(f"Iteration {iteration + 1}: {metrics['throughput_samples_per_sec']:.2f} samples/sec")
        
        # Check if target reached
        if metrics['throughput_samples_per_sec'] >= target_throughput:
            print(f"Target throughput reached!")
            break
        
        # Optimize configuration
        config = profiler.optimize_dataloader_config(config, target_throughput)
    
    return dataset, dataloader, config

# Run optimization
dataset, dataloader, optimized_config = await optimize_data_loading(
    "data/large_dataset.csv",
    target_throughput=2000
)
```

---

## Support

For issues and questions:

1. **Documentation**: Check this guide and inline code comments
2. **Tests**: Run `python test_efficient_data_loader.py`
3. **Profiling**: Use `DataLoaderProfiler` for performance analysis
4. **Logs**: Check structured logs for debugging information
5. **Community**: Check Blatam Academy documentation

---

*Last updated: 2024* 