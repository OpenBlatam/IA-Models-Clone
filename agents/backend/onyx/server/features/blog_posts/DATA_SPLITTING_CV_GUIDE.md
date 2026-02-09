# 🚀 Data Splitting & Cross-Validation Guide

## Overview

This guide covers the production-ready data splitting and cross-validation system for Blatam Academy's AI training pipeline. The system provides enterprise-grade functionality for proper train/validation/test splits and advanced cross-validation strategies.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Core Concepts](#core-concepts)
3. [Split Strategies](#split-strategies)
4. [Cross-Validation Strategies](#cross-validation-strategies)
5. [Usage Examples](#usage-examples)
6. [Advanced Features](#advanced-features)
7. [Best Practices](#best-practices)
8. [Performance Optimization](#performance-optimization)
9. [Production Deployment](#production-deployment)
10. [Troubleshooting](#troubleshooting)
11. [API Reference](#api-reference)

## Quick Start

### Basic Data Splitting

```python
import asyncio
from agents.backend.onyx.server.features.blog_posts.data_splitting_cv import (
    quick_split_dataset, SplitStrategy
)
from agents.backend.onyx.server.features.blog_posts.efficient_data_loader import OptimizedTextDataset

async def basic_split_example():
    # Create sample dataset
    texts = [f"Sample text {i}" for i in range(1000)]
    labels = [i % 3 for i in range(1000)]  # 3 classes
    dataset = OptimizedTextDataset(texts, labels)
    
    # Quick stratified split
    split_result = await quick_split_dataset(
        dataset,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        strategy=SplitStrategy.STRATIFIED
    )
    
    print(f"Split sizes: {split_result.get_split_sizes()}")
    print(f"Split ratios: {split_result.get_split_ratios()}")

asyncio.run(basic_split_example())
```

### Basic Cross-Validation

```python
import asyncio
from agents.backend.onyx.server.features.blog_posts.data_splitting_cv import (
    quick_cross_validate, CrossValidationStrategy
)

async def basic_cv_example():
    # Create sample dataset
    texts = [f"Sample text {i}" for i in range(500)]
    labels = [i % 3 for i in range(500)]  # 3 classes
    dataset = OptimizedTextDataset(texts, labels)
    
    # Quick cross-validation
    cv_result = await quick_cross_validate(
        dataset,
        n_splits=5,
        strategy=CrossValidationStrategy.STRATIFIED_K_FOLD
    )
    
    print(f"CV summary: {cv_result.get_summary()}")

asyncio.run(basic_cv_example())
```

## Core Concepts

### Split Configuration

The `SplitConfig` class controls how datasets are split:

```python
from agents.backend.onyx.server.features.blog_posts.data_splitting_cv import SplitConfig, SplitStrategy

config = SplitConfig(
    strategy=SplitStrategy.STRATIFIED,  # Split strategy
    train_ratio=0.7,                   # Training set ratio
    val_ratio=0.15,                    # Validation set ratio
    test_ratio=0.15,                   # Test set ratio
    random_state=42,                   # Random seed
    shuffle=True,                      # Shuffle data
    stratify_by='labels'               # Column for stratification
)
```

### Cross-Validation Configuration

The `CrossValidationConfig` class controls cross-validation:

```python
from agents.backend.onyx.server.features.blog_posts.data_splitting_cv import (
    CrossValidationConfig, CrossValidationStrategy
)

cv_config = CrossValidationConfig(
    strategy=CrossValidationStrategy.STRATIFIED_K_FOLD,
    n_splits=5,                        # Number of folds
    n_repeats=3,                       # Number of repetitions
    random_state=42,                   # Random seed
    shuffle=True                       # Shuffle data
)
```

## Split Strategies

### 1. Random Split

Simple random splitting without considering data characteristics:

```python
config = SplitConfig(
    strategy=SplitStrategy.RANDOM,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15
)
```

**Use when:**
- Data is already well-mixed
- No specific ordering requirements
- Quick prototyping

### 2. Stratified Split

Maintains class distribution across splits:

```python
config = SplitConfig(
    strategy=SplitStrategy.STRATIFIED,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    stratify_by='labels'  # Column name for stratification
)
```

**Use when:**
- Imbalanced datasets
- Classification tasks
- Need consistent class distribution

### 3. Time Series Split

Respects temporal ordering:

```python
config = SplitConfig(
    strategy=SplitStrategy.TIME_SERIES,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    time_column='timestamp'  # Column name for time ordering
)
```

**Use when:**
- Time series data
- Sequential dependencies
- Future prediction tasks

### 4. Group Split

Ensures groups don't leak across splits:

```python
config = SplitConfig(
    strategy=SplitStrategy.GROUP,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    group_column='user_id'  # Column name for grouping
)
```

**Use when:**
- Multiple samples per entity
- Need to avoid data leakage
- User-based recommendations

### 5. Custom Split

Custom splitting logic:

```python
def custom_split_function(dataset, indices, config):
    # Custom splitting logic
    train_indices = indices[:700]
    val_indices = indices[700:850]
    test_indices = indices[850:]
    return train_indices, val_indices, test_indices

config = SplitConfig(
    strategy=SplitStrategy.CUSTOM,
    custom_split_function=custom_split_function
)
```

## Cross-Validation Strategies

### 1. K-Fold Cross-Validation

Standard k-fold splitting:

```python
cv_config = CrossValidationConfig(
    strategy=CrossValidationStrategy.K_FOLD,
    n_splits=5
)
```

### 2. Stratified K-Fold

Maintains class distribution in each fold:

```python
cv_config = CrossValidationConfig(
    strategy=CrossValidationStrategy.STRATIFIED_K_FOLD,
    n_splits=5
)
```

### 3. Time Series Cross-Validation

Respects temporal ordering:

```python
cv_config = CrossValidationConfig(
    strategy=CrossValidationStrategy.TIME_SERIES_CV,
    n_splits=5,
    time_column='timestamp'
)
```

### 4. Group K-Fold

Ensures groups don't leak across folds:

```python
cv_config = CrossValidationConfig(
    strategy=CrossValidationStrategy.GROUP_K_FOLD,
    n_splits=5,
    group_column='user_id'
)
```

### 5. Leave-One-Out

Uses all but one sample for training:

```python
cv_config = CrossValidationConfig(
    strategy=CrossValidationStrategy.LEAVE_ONE_OUT
)
```

### 6. Leave-P-Out

Uses all but P samples for training:

```python
cv_config = CrossValidationConfig(
    strategy=CrossValidationStrategy.LEAVE_P_OUT,
    p=2  # Leave 2 samples out
)
```

### 7. Repeated Stratified K-Fold

Repeats stratified k-fold multiple times:

```python
cv_config = CrossValidationConfig(
    strategy=CrossValidationStrategy.REPEATED_STRATIFIED_K_FOLD,
    n_splits=5,
    n_repeats=3
)
```

## Usage Examples

### Example 1: Text Classification with Stratified Splitting

```python
import asyncio
from agents.backend.onyx.server.features.blog_posts.data_splitting_cv import (
    DataSplittingManager, SplitConfig, SplitStrategy
)
from agents.backend.onyx.server.features.blog_posts.efficient_data_loader import (
    DataLoaderManager, OptimizedTextDataset
)
from agents.backend.onyx.server.features.blog_posts.production_transformers import DeviceManager

async def text_classification_example():
    # Initialize managers
    device_manager = DeviceManager()
    splitting_manager = DataSplittingManager(device_manager)
    data_loader_manager = DataLoaderManager(device_manager)
    
    # Create dataset
    texts = [
        "This is a positive review",
        "This is a negative review", 
        "This is a neutral review",
        # ... more texts
    ]
    labels = [1, 0, 2, 1, 0, 2, 1, 0, 2, 1]  # 0=negative, 1=positive, 2=neutral
    dataset = OptimizedTextDataset(texts, labels)
    
    # Configure splitting
    split_config = SplitConfig(
        strategy=SplitStrategy.STRATIFIED,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42
    )
    
    # Split dataset
    split_result = splitting_manager.splitter.split_dataset(dataset, split_config)
    
    # Analyze split quality
    quality = splitting_manager.analyze_split_quality(split_result)
    print(f"Split quality: {quality['distribution_similarity']:.4f}")
    
    # Create DataLoaders
    from agents.backend.onyx.server.features.blog_posts.efficient_data_loader import DataLoaderConfig
    
    dataloader_config = DataLoaderConfig(
        batch_size=16,
        num_workers=4,
        pin_memory=True
    )
    
    train_loader, val_loader, test_loader = splitting_manager.create_dataloaders_from_split(
        split_result, dataloader_config
    )
    
    return train_loader, val_loader, test_loader

asyncio.run(text_classification_example())
```

### Example 2: Time Series Forecasting

```python
async def time_series_example():
    # Create time series dataset
    timestamps = list(range(1000))
    values = [np.sin(t * 0.1) + np.random.normal(0, 0.1) for t in timestamps]
    
    # Create dataset with time information
    dataset_data = []
    for i, (ts, val) in enumerate(zip(timestamps, values)):
        dataset_data.append({
            'text': f"Time series value at {ts}",
            'timestamp': ts,
            'value': val
        })
    
    dataset = OptimizedTextDataset(dataset_data, labels=values)
    
    # Configure time series splitting
    split_config = SplitConfig(
        strategy=SplitStrategy.TIME_SERIES,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        time_column='timestamp'
    )
    
    # Split dataset
    device_manager = DeviceManager()
    splitting_manager = DataSplittingManager(device_manager)
    split_result = splitting_manager.splitter.split_dataset(dataset, split_config)
    
    return split_result
```

### Example 3: User-Based Recommendation System

```python
async def user_based_example():
    # Create user-based dataset
    user_ids = [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]
    texts = [f"User {uid} interaction {i}" for uid, i in zip(user_ids, range(len(user_ids)))]
    labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    
    # Create dataset with user information
    dataset_data = []
    for i, (uid, text, label) in enumerate(zip(user_ids, texts, labels)):
        dataset_data.append({
            'text': text,
            'user_id': uid,
            'labels': label
        })
    
    dataset = OptimizedTextDataset(dataset_data, labels)
    
    # Configure group splitting
    split_config = SplitConfig(
        strategy=SplitStrategy.GROUP,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        group_column='user_id'
    )
    
    # Split dataset
    device_manager = DeviceManager()
    splitting_manager = DataSplittingManager(device_manager)
    split_result = splitting_manager.splitter.split_dataset(dataset, split_config)
    
    return split_result
```

### Example 4: Cross-Validation with Model Training

```python
async def cv_with_training_example():
    # Create dataset
    texts = [f"Sample {i}" for i in range(1000)]
    labels = [i % 3 for i in range(1000)]
    dataset = OptimizedTextDataset(texts, labels)
    
    # Configure cross-validation
    cv_config = CrossValidationConfig(
        strategy=CrossValidationStrategy.STRATIFIED_K_FOLD,
        n_splits=5,
        random_state=42
    )
    
    # Perform cross-validation
    device_manager = DeviceManager()
    splitting_manager = DataSplittingManager(device_manager)
    data_loader_manager = DataLoaderManager(device_manager)
    
    cv_result = await splitting_manager.cross_validator.cross_validate(
        dataset, cv_config, None, None, data_loader_manager
    )
    
    # Analyze results
    summary = cv_result.get_summary()
    print(f"Mean F1 Score: {summary['mean_scores']['val_f1_score']:.4f}")
    print(f"Std F1 Score: {summary['std_scores']['val_f1_score']:.4f}")
    print(f"Best Fold: {summary['best_fold']}")
    print(f"Worst Fold: {summary['worst_fold']}")
    
    return cv_result
```

## Advanced Features

### Split Quality Analysis

```python
# Analyze split quality
quality = splitting_manager.analyze_split_quality(split_result)

print("Class Distributions:")
for split_name, dist in quality['class_distributions'].items():
    print(f"  {split_name}: {dist}")

print(f"Distribution Similarity: {quality['distribution_similarity']:.4f}")
print(f"Class Coverage: {quality['class_coverage']}")
```

### Custom Split Functions

```python
def domain_aware_split(dataset, indices, config):
    """Split based on domain knowledge."""
    # Example: Split by text length
    short_texts = []
    long_texts = []
    
    for idx in indices:
        item = dataset[idx]
        if isinstance(item, dict):
            text = item.get('text', '')
        else:
            text = str(item)
        
        if len(text) < 100:
            short_texts.append(idx)
        else:
            long_texts.append(idx)
    
    # Ensure each split has both short and long texts
    train_size = int(config.train_ratio * len(indices))
    val_size = int(config.val_ratio * len(indices))
    
    # Distribute short and long texts across splits
    train_indices = short_texts[:train_size//2] + long_texts[:train_size//2]
    val_indices = short_texts[train_size//2:train_size//2+val_size//2] + long_texts[train_size//2:train_size//2+val_size//2]
    test_indices = short_texts[train_size//2+val_size//2:] + long_texts[train_size//2+val_size//2:]
    
    return train_indices, val_indices, test_indices

config = SplitConfig(
    strategy=SplitStrategy.CUSTOM,
    custom_split_function=domain_aware_split
)
```

### Integration with Model Training

```python
from agents.backend.onyx.server.features.blog_posts.model_training import ModelTrainer, TrainingConfig

async def integrated_training_example():
    # Create trainer
    device_manager = DeviceManager()
    trainer = ModelTrainer(device_manager)
    
    # Configure training
    config = TrainingConfig(
        model_type=ModelType.TRANSFORMER,
        training_mode=TrainingMode.FINE_TUNE,
        model_name="my_model",
        dataset_path="path/to/dataset",
        cross_validation_folds=5,  # Enable cross-validation
        eval_split=0.15,
        test_split=0.15
    )
    
    # Train with automatic splitting and cross-validation
    results = await trainer.train(config)
    
    # Access results
    print(f"Training completed in {results['total_training_time']:.2f} seconds")
    print(f"Best model saved at: {results['best_model_path']}")
    
    if results['cross_validation_result']:
        cv_summary = results['cross_validation_result'].get_summary()
        print(f"CV Mean F1: {cv_summary['mean_scores']['val_f1_score']:.4f}")
```

## Best Practices

### 1. Choose the Right Split Strategy

- **Stratified**: Use for classification tasks with imbalanced classes
- **Time Series**: Use for sequential data or forecasting
- **Group**: Use when you have multiple samples per entity
- **Random**: Use for well-mixed, independent data

### 2. Validate Split Quality

```python
# Always check split quality
quality = splitting_manager.analyze_split_quality(split_result)

if quality['distribution_similarity'] < 0.8:
    print("Warning: Poor split quality detected")
    # Consider adjusting split strategy or data preprocessing
```

### 3. Use Appropriate Cross-Validation

- **Stratified K-Fold**: Default choice for classification
- **Time Series CV**: For sequential data
- **Group K-Fold**: For grouped data
- **Leave-One-Out**: For small datasets

### 4. Set Proper Random Seeds

```python
# Use consistent random seeds for reproducibility
config = SplitConfig(random_state=42)
cv_config = CrossValidationConfig(random_state=42)
```

### 5. Handle Imbalanced Data

```python
# For highly imbalanced data, consider custom splitting
def balanced_split(dataset, indices, config):
    # Implement custom logic to ensure each split has all classes
    pass

config = SplitConfig(
    strategy=SplitStrategy.CUSTOM,
    custom_split_function=balanced_split
)
```

### 6. Monitor Cross-Validation Performance

```python
# Track CV performance across folds
cv_result = await splitting_manager.cross_validator.cross_validate(...)

# Check for high variance
std_f1 = cv_result.std_scores['val_f1_score']
if std_f1 > 0.1:
    print("Warning: High variance in CV results")
    # Consider more folds or different strategy
```

## Performance Optimization

### 1. Efficient Data Loading

```python
# Use efficient data loading with caching
dataloader_config = DataLoaderConfig(
    batch_size=32,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    cache_strategy=CacheStrategy.HYBRID,
    cache_dir="./cache"
)
```

### 2. Parallel Processing

```python
# Enable parallel CV when possible
cv_config = CrossValidationConfig(
    strategy=CrossValidationStrategy.STRATIFIED_K_FOLD,
    n_splits=5,
    n_jobs=-1  # Use all CPU cores
)
```

### 3. Memory Management

```python
# Use memory-efficient strategies for large datasets
config = SplitConfig(
    strategy=SplitStrategy.STRATIFIED,
    # Consider streaming for very large datasets
)
```

## Production Deployment

### 1. Configuration Management

```python
# Use environment variables for configuration
import os

config = SplitConfig(
    strategy=SplitStrategy.STRATIFIED,
    train_ratio=float(os.getenv('TRAIN_RATIO', '0.7')),
    val_ratio=float(os.getenv('VAL_RATIO', '0.15')),
    test_ratio=float(os.getenv('TEST_RATIO', '0.15')),
    random_state=int(os.getenv('RANDOM_STATE', '42'))
)
```

### 2. Logging and Monitoring

```python
import logging

# Set up comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Log split information
logger.info(f"Dataset split completed: {split_result.get_split_sizes()}")
logger.info(f"Split quality: {quality['distribution_similarity']:.4f}")
```

### 3. Error Handling

```python
try:
    split_result = splitting_manager.splitter.split_dataset(dataset, config)
except ValueError as e:
    logger.error(f"Split configuration error: {e}")
    # Fall back to default configuration
    config = SplitConfig()
    split_result = splitting_manager.splitter.split_dataset(dataset, config)
except Exception as e:
    logger.error(f"Unexpected error during splitting: {e}")
    raise
```

### 4. Testing

```python
# Run comprehensive tests
from agents.backend.onyx.server.features.blog_posts.test_data_splitting_cv import run_all_tests

success = run_all_tests()
if not success:
    raise RuntimeError("Data splitting tests failed")
```

## Troubleshooting

### Common Issues

#### 1. Poor Split Quality

**Problem**: Low distribution similarity between splits

**Solution**:
```python
# Check class distribution
quality = splitting_manager.analyze_split_quality(split_result)
print(quality['class_distributions'])

# Use stratified splitting
config = SplitConfig(strategy=SplitStrategy.STRATIFIED)
```

#### 2. Memory Issues

**Problem**: Out of memory during splitting

**Solution**:
```python
# Use smaller batch sizes
dataloader_config = DataLoaderConfig(batch_size=8)

# Use memory-efficient caching
dataloader_config.cache_strategy = CacheStrategy.DISK
```

#### 3. Slow Cross-Validation

**Problem**: CV taking too long

**Solution**:
```python
# Reduce number of folds
cv_config = CrossValidationConfig(n_splits=3)

# Use parallel processing
cv_config.n_jobs = -1
```

#### 4. Inconsistent Results

**Problem**: Different results on each run

**Solution**:
```python
# Set random seeds
config = SplitConfig(random_state=42)
cv_config = CrossValidationConfig(random_state=42)
```

### Debug Mode

```python
# Enable debug logging
logging.getLogger('agents.backend.onyx.server.features.blog_posts.data_splitting_cv').setLevel(logging.DEBUG)

# Use smaller datasets for testing
test_dataset = dataset[:100]  # Use first 100 samples
```

## API Reference

### SplitConfig

```python
@dataclass
class SplitConfig:
    strategy: SplitStrategy = SplitStrategy.STRATIFIED
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_state: int = 42
    shuffle: bool = True
    stratify_by: Optional[str] = None
    time_column: Optional[str] = None
    group_column: Optional[str] = None
    custom_split_function: Optional[Callable] = None
```

### CrossValidationConfig

```python
@dataclass
class CrossValidationConfig:
    strategy: CrossValidationStrategy = CrossValidationStrategy.STRATIFIED_K_FOLD
    n_splits: int = 5
    n_repeats: int = 3
    test_size: float = 0.2
    random_state: int = 42
    shuffle: bool = True
    stratify_by: Optional[str] = None
    time_column: Optional[str] = None
    group_column: Optional[str] = None
    p: int = 1
    custom_cv_function: Optional[Callable] = None
```

### DataSplittingManager

```python
class DataSplittingManager:
    def __init__(self, device_manager: DeviceManager)
    
    async def split_and_validate(
        self, 
        dataset: Dataset, 
        split_config: SplitConfig,
        cv_config: Optional[CrossValidationConfig] = None,
        data_loader_manager: Optional[DataLoaderManager] = None
    ) -> Dict[str, Any]
    
    def create_dataloaders_from_split(
        self, 
        split_result: SplitResult,
        data_loader_config: DataLoaderConfig
    ) -> Tuple[DataLoader, DataLoader, DataLoader]
    
    def analyze_split_quality(self, split_result: SplitResult) -> Dict[str, Any]
```

### Quick Functions

```python
async def quick_split_dataset(
    dataset: Dataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    strategy: SplitStrategy = SplitStrategy.STRATIFIED
) -> SplitResult

async def quick_cross_validate(
    dataset: Dataset,
    n_splits: int = 5,
    strategy: CrossValidationStrategy = CrossValidationStrategy.STRATIFIED_K_FOLD
) -> CrossValidationResult
```

## Conclusion

This data splitting and cross-validation system provides enterprise-grade functionality for proper train/validation/test splits and advanced cross-validation strategies. By following the best practices outlined in this guide, you can ensure robust, reproducible, and high-quality model training pipelines.

For additional support or questions, refer to the test suite and examples provided in the codebase. 