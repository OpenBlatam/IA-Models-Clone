# Data Splitting and Validation System

## Overview

This document provides a comprehensive overview of the data splitting and validation system that implements proper train/validation/test splits and cross-validation strategies for deep learning applications.

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Split Types](#split-types)
3. [Cross-Validation Strategies](#cross-validation-strategies)
4. [Configuration Options](#configuration-options)
5. [Data Splitter Implementation](#data-splitter-implementation)
6. [Cross-Validation Evaluator](#cross-validation-evaluator)
7. [Usage Examples](#usage-examples)
8. [Best Practices](#best-practices)
9. [Validation and Monitoring](#validation-and-monitoring)

## System Architecture

### Core Components

The data splitting and validation system consists of several key components:

```python
class DataSplitter:
    """Comprehensive data splitting and validation system."""
    
    def __init__(self, config: SplitConfig):
        self.config = config
        self.logger = self._setup_logging()
        
        # Set random seeds
        random.seed(config.random_state)
        np.random.seed(config.random_state)
        torch.manual_seed(config.random_state)
        
        # Store splits
        self.splits = {}
        self.cv_splits = {}
        self.split_metadata = {}
```

### Cross-Validation Evaluator

```python
class CrossValidationEvaluator:
    """Cross-validation evaluation system."""
    
    def __init__(self, model_class: type, model_params: Dict[str, Any],
                 splitter: DataSplitter):
        self.model_class = model_class
        self.model_params = model_params
        self.splitter = splitter
        self.logger = self._setup_logging()
        self.results = {}
```

## Split Types

### Supported Split Types

```python
class SplitType(Enum):
    """Types of data splits."""
    TRAIN_VAL_TEST = "train_val_test"
    TRAIN_TEST = "train_test"
    CROSS_VALIDATION = "cross_validation"
    TIME_SERIES = "time_series"
    GROUP_SPLIT = "group_split"
```

### Split Characteristics

#### **Train/Validation/Test Split**
- **Purpose**: Standard split for model development and evaluation
- **Ratios**: Typically 70/15/15 or 80/10/10
- **Use Case**: Most common for supervised learning tasks
- **Validation**: Ensures proper evaluation on unseen data

#### **Train/Test Split**
- **Purpose**: Simple split for quick experiments
- **Ratios**: Typically 80/20 or 70/30
- **Use Case**: Initial model development and prototyping
- **Limitation**: No separate validation set for hyperparameter tuning

#### **Cross-Validation Split**
- **Purpose**: Robust model evaluation with multiple folds
- **Folds**: Typically 5 or 10 folds
- **Use Case**: Small datasets, hyperparameter tuning
- **Advantage**: Better estimate of model performance

#### **Time Series Split**
- **Purpose**: Maintains temporal order for time series data
- **Order**: Train → Validation → Test (chronological)
- **Use Case**: Time series forecasting, sequential data
- **Constraint**: Cannot shuffle data to maintain temporal order

#### **Group Split**
- **Purpose**: Ensures groups don't leak across splits
- **Groups**: Based on subject ID, patient ID, etc.
- **Use Case**: Medical data, user-based data, grouped observations
- **Advantage**: Prevents data leakage between splits

## Cross-Validation Strategies

### Supported Cross-Validation Types

```python
class CrossValidationType(Enum):
    """Types of cross-validation."""
    K_FOLD = "k_fold"
    STRATIFIED_K_FOLD = "stratified_k_fold"
    SHUFFLE_SPLIT = "shuffle_split"
    STRATIFIED_SHUFFLE_SPLIT = "stratified_shuffle_split"
    LEAVE_ONE_OUT = "leave_one_out"
    LEAVE_P_OUT = "leave_p_out"
    GROUP_K_FOLD = "group_k_fold"
    STRATIFIED_GROUP_K_FOLD = "stratified_group_k_fold"
    TIME_SERIES_SPLIT = "time_series_split"
    REPEATED_K_FOLD = "repeated_k_fold"
    REPEATED_STRATIFIED_K_FOLD = "repeated_stratified_k_fold"
```

### Cross-Validation Characteristics

#### **K-Fold Cross-Validation**
```python
# Standard k-fold cross-validation
cv = KFold(n_splits=5, shuffle=True, random_state=42)
```
- **Folds**: Divides data into k equal parts
- **Use Case**: General purpose, balanced datasets
- **Advantage**: Simple and widely understood

#### **Stratified K-Fold Cross-Validation**
```python
# Stratified k-fold maintains class distribution
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```
- **Stratification**: Maintains class distribution in each fold
- **Use Case**: Imbalanced datasets, classification tasks
- **Advantage**: Better representation of minority classes

#### **Shuffle Split Cross-Validation**
```python
# Random sampling with replacement
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
```
- **Sampling**: Random sampling with replacement
- **Use Case**: Large datasets, quick evaluation
- **Advantage**: Fast computation, flexible test size

#### **Leave-One-Out Cross-Validation**
```python
# Leave one sample out for validation
cv = LeaveOneOut()
```
- **Folds**: N folds (one sample per fold)
- **Use Case**: Very small datasets
- **Disadvantage**: Computationally expensive for large datasets

#### **Group K-Fold Cross-Validation**
```python
# Ensures groups don't split across folds
cv = GroupKFold(n_splits=5)
```
- **Groups**: Keeps groups together in same fold
- **Use Case**: Grouped data (patients, subjects, etc.)
- **Advantage**: Prevents data leakage between groups

#### **Time Series Split**
```python
# Maintains temporal order
cv = TimeSeriesSplit(n_splits=5)
```
- **Order**: Maintains chronological order
- **Use Case**: Time series forecasting
- **Constraint**: Cannot shuffle data

#### **Repeated Cross-Validation**
```python
# Multiple runs of cross-validation
cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
```
- **Repeats**: Multiple runs with different random seeds
- **Use Case**: Robust evaluation, reduce variance
- **Advantage**: More stable performance estimates

## Configuration Options

### Comprehensive Configuration

```python
@dataclass
class SplitConfig:
    """Configuration for data splitting."""
    # Split ratios
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Cross-validation settings
    cv_folds: int = 5
    cv_repeats: int = 3
    cv_strategy: CrossValidationType = CrossValidationType.STRATIFIED_K_FOLD
    
    # Stratification
    stratify: bool = True
    stratify_column: Optional[str] = None
    
    # Group splitting
    group_column: Optional[str] = None
    
    # Time series
    time_column: Optional[str] = None
    
    # Random state
    random_state: int = 42
    
    # Validation
    validate_splits: bool = True
    min_samples_per_split: int = 10
    
    # Output
    save_splits: bool = True
    splits_file: str = "data_splits.json"
```

### Configuration Options

#### **Split Ratios**
- `train_ratio`: Proportion of data for training (default: 0.7)
- `val_ratio`: Proportion of data for validation (default: 0.15)
- `test_ratio`: Proportion of data for testing (default: 0.15)

#### **Cross-Validation Settings**
- `cv_folds`: Number of cross-validation folds (default: 5)
- `cv_repeats`: Number of repeated cross-validation runs (default: 3)
- `cv_strategy`: Type of cross-validation strategy

#### **Stratification**
- `stratify`: Enable stratification (default: True)
- `stratify_column`: Column to use for stratification

#### **Group Splitting**
- `group_column`: Column containing group identifiers

#### **Time Series**
- `time_column`: Column containing time information

#### **Validation**
- `validate_splits`: Validate split quality (default: True)
- `min_samples_per_split`: Minimum samples per split (default: 10)

## Data Splitter Implementation

### Train/Validation/Test Split

```python
def _train_val_test_split(self, data: Union[Dataset, np.ndarray, pd.DataFrame],
                          targets: Optional[Union[np.ndarray, pd.Series]] = None) -> Dict[str, Any]:
    """Perform train/validation/test split."""
    # Convert data to indices if it's a Dataset
    if isinstance(data, Dataset):
        indices = list(range(len(data)))
        data_array = indices
    else:
        data_array = data
    
    # Calculate split sizes
    total_size = len(data_array)
    train_size = int(total_size * self.config.train_ratio)
    val_size = int(total_size * self.config.val_ratio)
    test_size = total_size - train_size - val_size
    
    # Prepare stratification target
    stratify_target = None
    if self.config.stratify and targets is not None:
        stratify_target = targets
    elif self.config.stratify and self.config.stratify_column and isinstance(data, pd.DataFrame):
        stratify_target = data[self.config.stratify_column]
    
    # Perform splits
    if stratify_target is not None:
        # Stratified split
        train_indices, temp_indices = train_test_split(
            range(total_size),
            train_size=train_size,
            stratify=stratify_target,
            random_state=self.config.random_state
        )
        
        # Split remaining data into validation and test
        remaining_targets = [stratify_target[i] for i in temp_indices]
        val_indices, test_indices = train_test_split(
            temp_indices,
            train_size=val_size,
            stratify=remaining_targets,
            random_state=self.config.random_state
        )
    else:
        # Random split
        train_indices, temp_indices = train_test_split(
            range(total_size),
            train_size=train_size,
            random_state=self.config.random_state
        )
        
        val_indices, test_indices = train_test_split(
            temp_indices,
            train_size=val_size,
            random_state=self.config.random_state
        )
    
    # Create splits
    splits = {
        'train': train_indices,
        'val': val_indices,
        'test': test_indices
    }
    
    return splits
```

### Cross-Validation Split

```python
def _cross_validation_split(self, data: Union[Dataset, np.ndarray, pd.DataFrame],
                           targets: Optional[Union[np.ndarray, pd.Series]] = None) -> Dict[str, Any]:
    """Perform cross-validation split."""
    # Convert data to indices if it's a Dataset
    if isinstance(data, Dataset):
        indices = list(range(len(data)))
        data_array = indices
    else:
        data_array = data
    
    total_size = len(data_array)
    self.logger.info(f"Performing {self.config.cv_folds}-fold cross-validation on {total_size} samples")
    
    # Prepare stratification target
    stratify_target = None
    if self.config.stratify and targets is not None:
        stratify_target = targets
    elif self.config.stratify and self.config.stratify_column and isinstance(data, pd.DataFrame):
        stratify_target = data[self.config.stratify_column]
    
    # Create cross-validation splits based on strategy
    if self.config.cv_strategy == CrossValidationType.K_FOLD:
        cv = KFold(n_splits=self.config.cv_folds, shuffle=True, random_state=self.config.random_state)
    elif self.config.cv_strategy == CrossValidationType.STRATIFIED_K_FOLD:
        cv = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, random_state=self.config.random_state)
    elif self.config.cv_strategy == CrossValidationType.REPEATED_K_FOLD:
        cv = RepeatedKFold(n_splits=self.config.cv_folds, n_repeats=self.config.cv_repeats, random_state=self.config.random_state)
    # ... other strategies
    
    # Generate splits
    cv_splits = []
    split_generator = cv.split(data_array, stratify_target) if stratify_target is not None else cv.split(data_array)
    
    for fold, (train_indices, val_indices) in enumerate(split_generator):
        cv_splits.append({
            'fold': fold,
            'train': train_indices,
            'val': val_indices
        })
        
        self.logger.info(f"Fold {fold + 1}: Train={len(train_indices)}, Val={len(val_indices)}")
    
    return {'cv_splits': cv_splits}
```

### Time Series Split

```python
def _time_series_split(self, data: Union[Dataset, np.ndarray, pd.DataFrame],
                       targets: Optional[Union[np.ndarray, pd.Series]] = None) -> Dict[str, Any]:
    """Perform time series split."""
    if self.config.time_column is None:
        raise ValueError("Time column required for time series split")
    
    # Convert data to DataFrame if needed
    if isinstance(data, Dataset):
        raise ValueError("Time series split requires DataFrame or array with time column")
    
    if isinstance(data, np.ndarray):
        # Assume first column is time
        df = pd.DataFrame(data)
        time_col = 0
    else:
        df = data
        time_col = self.config.time_column
    
    # Sort by time
    df_sorted = df.sort_values(time_col)
    total_size = len(df_sorted)
    
    # Calculate split sizes
    train_size = int(total_size * self.config.train_ratio)
    val_size = int(total_size * self.config.val_ratio)
    test_size = total_size - train_size - val_size
    
    # Create splits (maintaining temporal order)
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, train_size + val_size))
    test_indices = list(range(train_size + val_size, total_size))
    
    splits = {
        'train': train_indices,
        'val': val_indices,
        'test': test_indices
    }
    
    return splits
```

### Group Split

```python
def _group_split(self, data: Union[Dataset, np.ndarray, pd.DataFrame],
                 targets: Optional[Union[np.ndarray, pd.Series]] = None) -> Dict[str, Any]:
    """Perform group-based split."""
    if self.config.group_column is None:
        raise ValueError("Group column required for group split")
    
    # Convert data to DataFrame if needed
    if isinstance(data, Dataset):
        raise ValueError("Group split requires DataFrame or array with group column")
    
    if isinstance(data, np.ndarray):
        # Assume first column is group
        df = pd.DataFrame(data)
        group_col = 0
    else:
        df = data
        group_col = self.config.group_column
    
    # Get unique groups
    groups = df[group_col].unique()
    total_groups = len(groups)
    
    # Calculate split sizes
    train_groups = int(total_groups * self.config.train_ratio)
    val_groups = int(total_groups * self.config.val_ratio)
    test_groups = total_groups - train_groups - val_groups
    
    # Shuffle groups
    np.random.shuffle(groups)
    
    # Split groups
    train_group_ids = groups[:train_groups]
    val_group_ids = groups[train_groups:train_groups + val_groups]
    test_group_ids = groups[train_groups + val_groups:]
    
    # Get indices for each split
    train_indices = df[df[group_col].isin(train_group_ids)].index.tolist()
    val_indices = df[df[group_col].isin(val_group_ids)].index.tolist()
    test_indices = df[df[group_col].isin(test_group_ids)].index.tolist()
    
    splits = {
        'train': train_indices,
        'val': val_indices,
        'test': test_indices
    }
    
    return splits
```

## Cross-Validation Evaluator

### Evaluation Process

```python
def evaluate_cv(self, dataset: Dataset, targets: np.ndarray,
               batch_size: int = 32, num_epochs: int = 10,
               device: str = 'cpu') -> Dict[str, Any]:
    """Evaluate model using cross-validation."""
    self.logger.info("Starting cross-validation evaluation")
    
    cv_results = []
    
    for fold in range(len(self.splitter.cv_splits)):
        self.logger.info(f"Evaluating fold {fold + 1}/{len(self.splitter.cv_splits)}")
        
        # Create dataloaders for this fold
        dataloaders = self.splitter.create_cv_dataloaders(
            dataset, fold, batch_size=batch_size
        )
        
        # Create and train model
        model = self.model_class(**self.model_params).to(device)
        fold_result = self._train_and_evaluate_fold(
            model, dataloaders, targets, num_epochs, device
        )
        
        cv_results.append(fold_result)
    
    # Aggregate results
    aggregated_results = self._aggregate_cv_results(cv_results)
    
    self.results = {
        'cv_results': cv_results,
        'aggregated_results': aggregated_results
    }
    
    return self.results
```

### Training and Evaluation

```python
def _train_and_evaluate_fold(self, model: nn.Module, dataloaders: Dict[str, DataLoader],
                             targets: np.ndarray, num_epochs: int, device: str) -> Dict[str, Any]:
    """Train and evaluate model for a single fold."""
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    # Training loop
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_idx, (data, target) in enumerate(dataloaders['train']):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(dataloaders['train'])
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in dataloaders['val']:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
                
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        val_loss /= len(dataloaders['val'])
        val_accuracy = correct / total
        
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'final_val_accuracy': val_accuracies[-1],
        'final_val_loss': val_losses[-1]
    }
```

### Results Aggregation

```python
def _aggregate_cv_results(self, cv_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate cross-validation results."""
    final_accuracies = [result['final_val_accuracy'] for result in cv_results]
    final_losses = [result['final_val_loss'] for result in cv_results]
    
    aggregated = {
        'mean_accuracy': np.mean(final_accuracies),
        'std_accuracy': np.std(final_accuracies),
        'mean_loss': np.mean(final_losses),
        'std_loss': np.std(final_losses),
        'min_accuracy': np.min(final_accuracies),
        'max_accuracy': np.max(final_accuracies),
        'min_loss': np.min(final_losses),
        'max_loss': np.max(final_losses)
    }
    
    self.logger.info(f"Cross-validation results:")
    self.logger.info(f"  Mean Accuracy: {aggregated['mean_accuracy']:.4f} ± {aggregated['std_accuracy']:.4f}")
    self.logger.info(f"  Mean Loss: {aggregated['mean_loss']:.4f} ± {aggregated['std_loss']:.4f}")
    self.logger.info(f"  Accuracy Range: {aggregated['min_accuracy']:.4f} - {aggregated['max_accuracy']:.4f}")
    
    return aggregated
```

## Usage Examples

### Basic Train/Validation/Test Split

```python
# Create configuration
config = SplitConfig(
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    stratify=True,
    random_state=42
)

# Create splitter
splitter = DataSplitter(config)

# Split data
splits = splitter.split_data(
    dataset, targets=y,
    split_type=SplitType.TRAIN_VAL_TEST
)

# Create dataloaders
dataloaders = splitter.create_dataloaders(dataset, batch_size=32)

# Use dataloaders
train_loader = dataloaders['train']
val_loader = dataloaders['val']
test_loader = dataloaders['test']
```

### Cross-Validation

```python
# Create configuration for cross-validation
config = SplitConfig(
    cv_folds=5,
    cv_strategy=CrossValidationType.STRATIFIED_K_FOLD,
    stratify=True,
    random_state=42
)

# Create splitter
splitter = DataSplitter(config)

# Perform cross-validation split
cv_splits = splitter.split_data(
    dataset, targets=y,
    split_type=SplitType.CROSS_VALIDATION
)

# Create cross-validation evaluator
evaluator = CrossValidationEvaluator(
    model_class=MyModel,
    model_params={'input_size': 10, 'hidden_size': 64, 'num_classes': 3},
    splitter=splitter
)

# Evaluate model using cross-validation
results = evaluator.evaluate_cv(
    dataset, targets=y,
    batch_size=32, num_epochs=10, device='cpu'
)

# Plot results
evaluator.plot_cv_results(save_path='cv_results.png')
```

### Time Series Split

```python
# Create configuration for time series
config = SplitConfig(
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    time_column='timestamp',
    random_state=42
)

# Create splitter
splitter = DataSplitter(config)

# Split time series data
splits = splitter.split_data(
    df, targets=y,
    split_type=SplitType.TIME_SERIES
)
```

### Group Split

```python
# Create configuration for group split
config = SplitConfig(
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    group_column='patient_id',
    random_state=42
)

# Create splitter
splitter = DataSplitter(config)

# Split grouped data
splits = splitter.split_data(
    df, targets=y,
    split_type=SplitType.GROUP_SPLIT
)
```

## Best Practices

### 1. Split Ratios

- **Standard Split**: 70/15/15 for train/val/test
- **Large Datasets**: 80/10/10 for train/val/test
- **Small Datasets**: Use cross-validation instead
- **Time Series**: Maintain temporal order

### 2. Stratification

- **Always stratify**: For classification tasks
- **Check distribution**: Ensure balanced representation
- **Handle imbalance**: Use stratified sampling for imbalanced classes

### 3. Cross-Validation

- **Small datasets**: Use k-fold cross-validation
- **Imbalanced data**: Use stratified k-fold
- **Grouped data**: Use group k-fold
- **Time series**: Use time series split

### 4. Validation

- **Check splits**: Validate split quality
- **Monitor distribution**: Ensure target distribution consistency
- **Test leakage**: Verify no data leakage between splits

### 5. Reproducibility

- **Set random seed**: Ensure reproducible splits
- **Save splits**: Save split indices for later use
- **Document process**: Record split configuration

## Validation and Monitoring

### Split Validation

```python
def _validate_splits(self, splits: Dict[str, List[int]], data: Any, targets: Optional[Any] = None):
    """Validate data splits."""
    self.logger.info("Validating data splits...")
    
    # Check minimum samples per split
    for split_name, indices in splits.items():
        if len(indices) < self.config.min_samples_per_split:
            self.logger.warning(f"{split_name} split has only {len(indices)} samples (minimum: {self.config.min_samples_per_split})")
    
    # Check for overlap
    all_indices = []
    for indices in splits.values():
        all_indices.extend(indices)
    
    if len(all_indices) != len(set(all_indices)):
        self.logger.warning("Overlapping indices detected in splits")
    
    # Check stratification if targets provided
    if targets is not None:
        for split_name, indices in splits.items():
            split_targets = [targets[i] for i in indices]
            unique_targets, counts = np.unique(split_targets, return_counts=True)
            self.logger.info(f"{split_name} split target distribution: {dict(zip(unique_targets, counts))}")
    
    self.logger.info("Split validation completed")
```

### Performance Monitoring

```python
def plot_cv_results(self, save_path: Optional[str] = None):
    """Plot cross-validation results."""
    if not self.results:
        self.logger.warning("No results to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot accuracy per fold
    accuracies = [result['final_val_accuracy'] for result in self.results['cv_results']]
    axes[0, 0].bar(range(1, len(accuracies) + 1), accuracies)
    axes[0, 0].set_title('Validation Accuracy per Fold')
    axes[0, 0].set_xlabel('Fold')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].axhline(y=self.results['aggregated_results']['mean_accuracy'], 
                       color='r', linestyle='--', label='Mean')
    axes[0, 0].legend()
    
    # Plot loss per fold
    losses = [result['final_val_loss'] for result in self.results['cv_results']]
    axes[0, 1].bar(range(1, len(losses) + 1), losses)
    axes[0, 1].set_title('Validation Loss per Fold')
    axes[0, 1].set_xlabel('Fold')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].axhline(y=self.results['aggregated_results']['mean_loss'], 
                       color='r', linestyle='--', label='Mean')
    axes[0, 1].legend()
    
    # Plot training curves for first fold
    if self.results['cv_results']:
        first_fold = self.results['cv_results'][0]
        axes[1, 0].plot(first_fold['train_losses'], label='Train Loss')
        axes[1, 0].plot(first_fold['val_losses'], label='Val Loss')
        axes[1, 0].set_title('Training Curves (Fold 1)')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        
        axes[1, 1].plot(first_fold['val_accuracies'], label='Val Accuracy')
        axes[1, 1].set_title('Validation Accuracy (Fold 1)')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"Plot saved to {save_path}")
    
    plt.show()
```

## Conclusion

The data splitting and validation system provides:

1. **Comprehensive Split Types**: Train/val/test, cross-validation, time series, group splits
2. **Multiple Cross-Validation Strategies**: K-fold, stratified, repeated, group-based
3. **Proper Validation**: Stratification, overlap detection, distribution monitoring
4. **Flexible Configuration**: Customizable ratios, strategies, and validation options
5. **Cross-Validation Evaluation**: Complete training and evaluation pipeline
6. **Results Analysis**: Aggregation, visualization, and performance monitoring
7. **Reproducibility**: Random seed control, split saving/loading
8. **Best Practices**: Guidelines for different data types and scenarios

This system ensures proper data splitting and validation for robust model development and evaluation in deep learning applications. 