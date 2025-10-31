# Data Splitting and Cross-Validation Framework

A comprehensive framework for proper train/validation/test splits and cross-validation strategies, specifically designed for SEO deep learning tasks with support for time series data, imbalanced datasets, and domain-specific splitting.

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Architecture](#architecture)
4. [Installation](#installation)
5. [Quick Start](#quick-start)
6. [Data Splitting Strategies](#data-splitting-strategies)
7. [Cross-Validation Strategies](#cross-validation-strategies)
8. [SEO-Specific Splitting](#seo-specific-splitting)
9. [Integration with Training Framework](#integration-with-training-framework)
10. [Best Practices](#best-practices)
11. [Advanced Features](#advanced-features)
12. [Troubleshooting](#troubleshooting)
13. [API Reference](#api-reference)

## Overview

The Data Splitting and Cross-Validation Framework provides robust, flexible, and SEO-aware data splitting capabilities. It addresses common challenges in deep learning:

- **Proper Data Leakage Prevention**: Domain-aware splitting for SEO data
- **Imbalanced Data Handling**: Stratified splitting for skewed class distributions
- **Time Series Awareness**: Temporal splitting for time-dependent data
- **Cross-Validation Support**: Multiple CV strategies for robust evaluation
- **SEO-Specific Strategies**: Domain, keyword, and content type splitting

## Key Features

### 🎯 **Data Splitting Strategies**
- **Standard Splitting**: Train/validation/test with configurable ratios
- **Stratified Splitting**: Maintains class distribution across splits
- **Group-Based Splitting**: Prevents data leakage within groups
- **Time Series Splitting**: Temporal ordering for time-dependent data

### 🔄 **Cross-Validation Strategies**
- **Stratified K-Fold**: Maintains class distribution in each fold
- **K-Fold**: Standard k-fold cross-validation
- **Time Series Split**: Forward chaining for time series data
- **Group K-Fold**: Group-aware cross-validation
- **Repeated CV**: Multiple runs for robust evaluation

### 🎯 **SEO-Specific Features**
- **Domain Splitting**: Prevents cross-domain data leakage
- **Keyword Splitting**: Splits by keyword groups or categories
- **Content Type Splitting**: Splits by content type (blog, product, etc.)
- **Metadata Preservation**: Maintains SEO-specific metadata

### 📊 **Analysis and Monitoring**
- **Split Quality Analysis**: Evaluates stratification quality
- **Distribution Analysis**: Analyzes label distribution across splits
- **Cross-Validation Statistics**: Aggregates results across folds
- **Visualization Support**: Ready for plotting and analysis

## Architecture

```
data_splitting_cross_validation.py
├── DataSplitConfig          # Configuration management
├── DataSplit               # Single split container
├── CrossValidationSplit    # CV splits container
├── DataSplitter           # Core splitting logic
├── CrossValidator         # Cross-validation logic
├── SEOSpecificSplitter    # SEO-specific strategies
└── DataSplitManager       # High-level manager
```

### Core Components

1. **DataSplitConfig**: Centralized configuration for all splitting parameters
2. **DataSplitter**: Handles standard data splitting with multiple strategies
3. **CrossValidator**: Manages cross-validation with various strategies
4. **SEOSpecificSplitter**: Implements SEO-aware splitting strategies
5. **DataSplitManager**: High-level interface for all splitting operations

## Installation

```bash
# Install required dependencies
pip install torch pandas numpy scikit-learn

# The framework is part of the SEO deep learning system
# No additional installation required
```

## Quick Start

### Basic Data Splitting

```python
from data_splitting_cross_validation import DataSplitConfig, DataSplitManager

# Create configuration
config = DataSplitConfig(
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    stratify_by='labels',  # Stratify by labels
    random_state=42
)

# Create split manager
split_manager = DataSplitManager(config)

# Split your data
splits = split_manager.create_splits(dataset, labels=labels)

# Create PyTorch datasets
train_dataset, val_dataset, test_dataset = split_manager.create_datasets(dataset, splits)

# Analyze splits
analysis = split_manager.analyze_splits(splits, labels=labels)
print(f"Stratification quality: {analysis['stratification_quality']['overall_quality']:.4f}")
```

### Cross-Validation

```python
# Configuration for cross-validation
config = DataSplitConfig(
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    use_cross_validation=True,
    cv_folds=5,
    cv_strategy="stratified",
    stratify_by='labels',
    random_state=42
)

# Create cross-validation splits
split_manager = DataSplitManager(config)
cv_splits = split_manager.create_splits(dataset, labels=labels)

# Train on each fold
fold_datasets = split_manager.create_datasets(dataset, cv_splits)
for fold_idx, (train_dataset, val_dataset, test_dataset) in enumerate(fold_datasets):
    # Train your model
    pass
```

### SEO-Specific Splitting

```python
# Split by domain to prevent cross-domain leakage
splits = split_manager.create_seo_splits(dataset, split_strategy="domain")

# Split by keyword groups
splits = split_manager.create_seo_splits(dataset, split_strategy="keyword")

# Split by content type
splits = split_manager.create_seo_splits(dataset, split_strategy="content_type")
```

## Data Splitting Strategies

### 1. Standard Splitting

```python
config = DataSplitConfig(
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    shuffle=True,
    random_state=42
)
```

### 2. Stratified Splitting

```python
config = DataSplitConfig(
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    stratify_by='labels',  # Column name for stratification
    random_state=42
)
```

### 3. Group-Based Splitting

```python
config = DataSplitConfig(
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    group_by='domain',  # Column name for grouping
    random_state=42
)
```

### 4. Time Series Splitting

```python
config = DataSplitConfig(
    train_ratio=0.6,
    val_ratio=0.2,
    test_ratio=0.2,
    time_series_split=True,
    preserve_order=True,
    random_state=42
)
```

## Cross-Validation Strategies

### 1. Stratified K-Fold

```python
config = DataSplitConfig(
    use_cross_validation=True,
    cv_folds=5,
    cv_strategy="stratified",
    stratify_by='labels',
    random_state=42
)
```

### 2. K-Fold

```python
config = DataSplitConfig(
    use_cross_validation=True,
    cv_folds=5,
    cv_strategy="kfold",
    random_state=42
)
```

### 3. Time Series Split

```python
config = DataSplitConfig(
    use_cross_validation=True,
    cv_folds=5,
    cv_strategy="timeseries",
    random_state=42
)
```

### 4. Group K-Fold

```python
config = DataSplitConfig(
    use_cross_validation=True,
    cv_folds=5,
    cv_strategy="group",
    group_by='domain',
    random_state=42
)
```

### 5. Repeated Cross-Validation

```python
config = DataSplitConfig(
    use_cross_validation=True,
    cv_folds=5,
    cv_repeats=3,
    cv_strategy="stratified",
    stratify_by='labels',
    random_state=42
)
```

## SEO-Specific Splitting

### Domain Splitting

Prevents cross-domain data leakage, ensuring that content from the same domain doesn't appear in both training and test sets.

```python
# Split by domain
splits = split_manager.create_seo_splits(dataset, split_strategy="domain")

# This ensures that if domain.com appears in training,
# it won't appear in validation or test sets
```

### Keyword Splitting

Splits data by keyword groups or categories to prevent keyword-specific leakage.

```python
# Split by keyword groups
splits = split_manager.create_seo_splits(dataset, split_strategy="keyword")

# Keywords are grouped by similarity or category
# to prevent related keywords from leaking across splits
```

### Content Type Splitting

Splits data by content type (blog, product, landing page, etc.) to ensure diverse content representation.

```python
# Split by content type
splits = split_manager.create_seo_splits(dataset, split_strategy="content_type")

# Ensures each split contains a mix of content types
```

## Integration with Training Framework

### Basic Integration

```python
from data_splitting_cross_validation import DataSplitConfig, DataSplitManager
from model_training_evaluation import TrainingConfig, ModelTrainer

# Create data splits
split_config = DataSplitConfig(
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    stratify_by='labels',
    random_state=42
)

split_manager = DataSplitManager(split_config)
splits = split_manager.create_splits(dataset, labels=labels)
train_dataset, val_dataset, test_dataset = split_manager.create_datasets(dataset, splits)

# Create training configuration
training_config = TrainingConfig(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    test_dataset=test_dataset,
    epochs=100,
    batch_size=32,
    learning_rate=1e-4
)

# Train model
trainer = ModelTrainer(training_config)
metrics = trainer.train()
```

### Cross-Validation Training

```python
# Create cross-validation splits
cv_config = DataSplitConfig(
    use_cross_validation=True,
    cv_folds=5,
    cv_strategy="stratified",
    stratify_by='labels',
    random_state=42
)

cv_split_manager = DataSplitManager(cv_config)
cv_splits = cv_split_manager.create_splits(dataset, labels=labels)

# Train on each fold
fold_results = []
fold_datasets = cv_split_manager.create_datasets(dataset, cv_splits)

for fold_idx, (train_dataset, val_dataset, test_dataset) in enumerate(fold_datasets):
    # Create model for this fold
    model = YourModel()
    
    # Create training configuration
    training_config = TrainingConfig(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        epochs=50,
        batch_size=32
    )
    
    # Train
    trainer = ModelTrainer(training_config)
    metrics = trainer.train()
    
    # Evaluate
    evaluator = ModelEvaluator(model)
    test_metrics = evaluator.evaluate(test_loader)
    
    fold_results.append({
        'fold': fold_idx + 1,
        'train_metrics': metrics,
        'test_metrics': test_metrics
    })

# Aggregate results
test_accuracies = [result['test_metrics']['accuracy'] for result in fold_results]
mean_accuracy = np.mean(test_accuracies)
std_accuracy = np.std(test_accuracies)

print(f"CV Results: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
```

## Best Practices

### 1. Data Leakage Prevention

```python
# ❌ Bad: Random splitting for SEO data
config = DataSplitConfig(
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    shuffle=True  # May cause domain leakage
)

# ✅ Good: Domain-aware splitting
config = DataSplitConfig(
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    group_by='domain'  # Prevents cross-domain leakage
)
```

### 2. Imbalanced Data Handling

```python
# ❌ Bad: Random splitting for imbalanced data
config = DataSplitConfig(
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15
)

# ✅ Good: Stratified splitting
config = DataSplitConfig(
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    stratify_by='labels'  # Maintains class distribution
)
```

### 3. Time Series Data

```python
# ❌ Bad: Random splitting for time series
config = DataSplitConfig(
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    shuffle=True
)

# ✅ Good: Time series splitting
config = DataSplitConfig(
    train_ratio=0.6,
    val_ratio=0.2,
    test_ratio=0.2,
    time_series_split=True,
    preserve_order=True
)
```

### 4. Cross-Validation for Small Datasets

```python
# For small datasets, use more folds
config = DataSplitConfig(
    use_cross_validation=True,
    cv_folds=10,  # More folds for smaller datasets
    cv_strategy="stratified",
    stratify_by='labels'
)
```

### 5. Repeated Cross-Validation

```python
# For robust evaluation, use repeated CV
config = DataSplitConfig(
    use_cross_validation=True,
    cv_folds=5,
    cv_repeats=3,  # Multiple runs
    cv_strategy="stratified",
    stratify_by='labels'
)
```

## Advanced Features

### 1. Custom Splitting Strategies

```python
class CustomSplitter:
    def __init__(self, config):
        self.config = config
    
    def split(self, data, labels=None, groups=None):
        # Implement custom splitting logic
        pass

# Use custom splitter
splitter = CustomSplitter(config)
splits = splitter.split(dataset, labels=labels)
```

### 2. Split Quality Analysis

```python
# Analyze split quality
analysis = split_manager.analyze_splits(splits, labels=labels)

print(f"Stratification quality: {analysis['stratification_quality']['overall_quality']:.4f}")
print(f"Label distribution:")
for split_name, dist in analysis['label_distribution'].items():
    print(f"  {split_name}: {dist['counts']}")
```

### 3. Multiple Split Strategies

```python
# Compare different splitting strategies
strategies = ['domain', 'keyword', 'content_type']
results = {}

for strategy in strategies:
    splits = split_manager.create_seo_splits(dataset, split_strategy=strategy)
    analysis = split_manager.analyze_splits(splits)
    results[strategy] = analysis

# Compare results
for strategy, analysis in results.items():
    print(f"{strategy}: {analysis['sample_counts']}")
```

## Troubleshooting

### Common Issues

1. **Data Leakage**
   ```python
   # Problem: Same domain in train and test
   # Solution: Use domain-based splitting
   config = DataSplitConfig(group_by='domain')
   ```

2. **Imbalanced Splits**
   ```python
   # Problem: Uneven class distribution
   # Solution: Use stratified splitting
   config = DataSplitConfig(stratify_by='labels')
   ```

3. **Time Series Issues**
   ```python
   # Problem: Future data in training
   # Solution: Use time series splitting
   config = DataSplitConfig(time_series_split=True)
   ```

4. **Small Dataset CV**
   ```python
   # Problem: Too few samples per fold
   # Solution: Use more folds or repeated CV
   config = DataSplitConfig(cv_folds=10, cv_repeats=3)
   ```

### Performance Optimization

```python
# For large datasets, use efficient splitting
config = DataSplitConfig(
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    shuffle=False,  # Faster for large datasets
    random_state=42
)
```

## API Reference

### DataSplitConfig

```python
@dataclass
class DataSplitConfig:
    # Split ratios
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Cross-validation configuration
    use_cross_validation: bool = False
    cv_folds: int = 5
    cv_strategy: str = "stratified"
    cv_repeats: int = 1
    
    # Stratification configuration
    stratify_by: Optional[str] = None
    group_by: Optional[str] = None
    
    # Time series configuration
    time_series_split: bool = False
    preserve_order: bool = False
    
    # SEO-specific configuration
    seo_domain_split: bool = False
    seo_keyword_split: bool = False
    seo_content_type_split: bool = False
    
    # Random state
    random_state: int = 42
    shuffle: bool = True
```

### DataSplitManager

```python
class DataSplitManager:
    def __init__(self, config: DataSplitConfig)
    
    def create_splits(self, data, labels=None, groups=None) -> Union[DataSplit, CrossValidationSplit]
    def create_seo_splits(self, data, split_strategy="domain") -> DataSplit
    def create_datasets(self, dataset, split) -> Union[Tuple[Dataset, Dataset, Dataset], List[Tuple[Dataset, Dataset, Dataset]]]
    def analyze_splits(self, split, labels=None) -> Dict[str, Any]
```

### DataSplit

```python
@dataclass
class DataSplit:
    train_indices: List[int]
    val_indices: List[int]
    test_indices: List[int]
    split_info: Dict[str, Any]
```

### CrossValidationSplit

```python
@dataclass
class CrossValidationSplit:
    fold_splits: List[DataSplit]
    cv_info: Dict[str, Any]
```

## Examples

See `example_data_splitting_cv.py` for comprehensive examples including:

- Standard data splitting with stratification
- Cross-validation with multiple strategies
- SEO-specific splitting (domain, keyword, content type)
- Time series aware splitting
- Training integration with proper splits
- Cross-validation training
- Imbalanced data handling

## Contributing

When contributing to the data splitting framework:

1. Follow the existing code style
2. Add comprehensive docstrings
3. Include unit tests for new features
4. Update documentation for API changes
5. Test with different data types and sizes

## License

This framework is part of the SEO Deep Learning System and follows the same licensing terms. 