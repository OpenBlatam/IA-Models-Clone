# 🎯 Evaluation Metrics Guide

## Overview

This guide covers the production-ready evaluation metrics system for Blatam Academy's AI training pipeline. The system provides task-specific metrics for classification, regression, generation, and other AI tasks.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Task Types](#task-types)
3. [Metric Types](#metric-types)
4. [Usage Examples](#usage-examples)
5. [Integration](#integration)
6. [Best Practices](#best-practices)
7. [API Reference](#api-reference)

## Quick Start

### Basic Classification Evaluation

```python
import asyncio
from agents.backend.onyx.server.features.blog_posts.evaluation_metrics import (
    create_evaluation_metrics, create_metric_config, TaskType, MetricType
)
from agents.backend.onyx.server.features.blog_posts.production_transformers import DeviceManager

async def basic_classification_evaluation():
    # Create evaluator
    device_manager = DeviceManager()
    evaluator = await create_evaluation_metrics(device_manager)
    
    # Create metric config
    config = create_metric_config(
        task_type=TaskType.CLASSIFICATION,
        metric_types=[
            MetricType.ACCURACY, MetricType.PRECISION, MetricType.RECALL,
            MetricType.F1, MetricType.ROC_AUC
        ]
    )
    
    # Simulate data
    import numpy as np
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 0, 1, 0, 1, 0, 1])
    y_prob = np.random.rand(10, 2)
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
    
    # Evaluate
    result = evaluator.evaluate(config, y_true, y_pred, y_prob)
    
    print(f"Accuracy: {result.metrics['accuracy']:.4f}")
    print(f"F1 Score: {result.metrics['f1']:.4f}")

asyncio.run(basic_classification_evaluation())
```

### Basic Regression Evaluation

```python
async def basic_regression_evaluation():
    # Create evaluator
    device_manager = DeviceManager()
    evaluator = await create_evaluation_metrics(device_manager)
    
    # Create metric config
    config = create_metric_config(
        task_type=TaskType.REGRESSION,
        metric_types=[
            MetricType.MSE, MetricType.MAE, MetricType.R2,
            MetricType.EXPLAINED_VARIANCE
        ]
    )
    
    # Simulate data
    import numpy as np
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
    
    # Evaluate
    result = evaluator.evaluate(config, y_true, y_pred)
    
    print(f"MSE: {result.metrics['mse']:.4f}")
    print(f"R²: {result.metrics['r2']:.4f}")

asyncio.run(basic_regression_evaluation())
```

## Task Types

### Classification Tasks

```python
# Binary Classification
config = create_metric_config(
    task_type=TaskType.CLASSIFICATION,
    metric_types=[
        MetricType.ACCURACY, MetricType.PRECISION, MetricType.RECALL,
        MetricType.F1, MetricType.ROC_AUC, MetricType.PR_AUC
    ]
)

# Multi-class Classification
config = create_metric_config(
    task_type=TaskType.CLASSIFICATION,
    metric_types=[
        MetricType.ACCURACY, MetricType.PRECISION, MetricType.RECALL,
        MetricType.F1, MetricType.ROC_AUC, MetricType.TOP_K_ACCURACY
    ],
    average='weighted',
    k=3
)

# Multi-label Classification
config = create_metric_config(
    task_type=TaskType.MULTI_LABEL,
    metric_types=[
        MetricType.HAMMING_LOSS, MetricType.JACCARD,
        MetricType.PRECISION, MetricType.RECALL, MetricType.F1
    ]
)
```

### Regression Tasks

```python
# Standard Regression
config = create_metric_config(
    task_type=TaskType.REGRESSION,
    metric_types=[
        MetricType.MSE, MetricType.MAE, MetricType.R2,
        MetricType.EXPLAINED_VARIANCE, MetricType.MAX_ERROR
    ]
)

# Advanced Regression
config = create_metric_config(
    task_type=TaskType.REGRESSION,
    metric_types=[
        MetricType.MSE, MetricType.MAE, MetricType.R2,
        MetricType.MAPE, MetricType.MSLE, MetricType.MEDIAN_AE,
        MetricType.MEAN_PINBALL_LOSS
    ]
)
```

### Generation Tasks

```python
# Text Generation
config = create_metric_config(
    task_type=TaskType.GENERATION,
    metric_types=[
        MetricType.BLEU, MetricType.ROUGE, MetricType.METEOR,
        MetricType.PERPLEXITY
    ]
)

# Translation
config = create_metric_config(
    task_type=TaskType.TRANSLATION,
    metric_types=[
        MetricType.BLEU, MetricType.ROUGE, MetricType.METEOR,
        MetricType.BLEURT, MetricType.COMET
    ]
)
```

### Specialized Tasks

```python
# Question Answering
config = create_metric_config(
    task_type=TaskType.QUESTION_ANSWERING,
    metric_types=[
        MetricType.ACCURACY, MetricType.F1, MetricType.EM
    ]
)

# Named Entity Recognition
config = create_metric_config(
    task_type=TaskType.NAMED_ENTITY_RECOGNITION,
    metric_types=[
        MetricType.PRECISION, MetricType.RECALL, MetricType.F1,
        MetricType.ACCURACY
    ]
)

# Sentiment Analysis
config = create_metric_config(
    task_type=TaskType.SENTIMENT_ANALYSIS,
    metric_types=[
        MetricType.ACCURACY, MetricType.PRECISION, MetricType.RECALL,
        MetricType.F1, MetricType.ROC_AUC
    ]
)
```

## Metric Types

### Classification Metrics

- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1**: Harmonic mean of precision and recall
- **ROC AUC**: Area under ROC curve
- **PR AUC**: Area under Precision-Recall curve
- **Confusion Matrix**: Detailed classification results
- **Jaccard**: Intersection over union
- **Hamming Loss**: Fraction of incorrectly predicted labels
- **Log Loss**: Logarithmic loss
- **Matthews Correlation**: Correlation coefficient
- **Cohen's Kappa**: Agreement between predictions and true labels
- **Balanced Accuracy**: Average of sensitivity and specificity
- **Top-K Accuracy**: Accuracy considering top K predictions

### Regression Metrics

- **MSE**: Mean Squared Error
- **MAE**: Mean Absolute Error
- **R²**: Coefficient of determination
- **Explained Variance**: Explained variance score
- **Max Error**: Maximum absolute error
- **MAPE**: Mean Absolute Percentage Error
- **MSLE**: Mean Squared Logarithmic Error
- **Median AE**: Median Absolute Error
- **Mean Pinball Loss**: Quantile regression loss

### Generation Metrics

- **BLEU**: Bilingual Evaluation Understudy
- **ROUGE**: Recall-Oriented Understudy for Gisting Evaluation
- **METEOR**: Metric for Evaluation of Translation with Explicit ORdering
- **CIDEr**: Consensus-based Image Description Evaluation
- **BERT Score**: BERT-based evaluation metric
- **Perplexity**: Language model perplexity
- **BLEURT**: BLEU, Recall, and Understudy for Re-ranking Translation
- **COMET**: Crosslingual Optimized Metric for Evaluation of Translation

## Usage Examples

### Example 1: Comprehensive Classification Evaluation

```python
async def comprehensive_classification_evaluation():
    device_manager = DeviceManager()
    evaluator = await create_evaluation_metrics(device_manager)
    
    # Create comprehensive metric config
    config = create_metric_config(
        task_type=TaskType.CLASSIFICATION,
        metric_types=[
            MetricType.ACCURACY, MetricType.PRECISION, MetricType.RECALL,
            MetricType.F1, MetricType.ROC_AUC, MetricType.PR_AUC,
            MetricType.CONFUSION_MATRIX, MetricType.CLASSIFICATION_REPORT,
            MetricType.JACCARD, MetricType.HAMMING_LOSS, MetricType.LOG_LOSS,
            MetricType.MATTHEWS_CORR, MetricType.COHEN_KAPPA,
            MetricType.BALANCED_ACCURACY, MetricType.TOP_K_ACCURACY
        ],
        average='weighted',
        k=3
    )
    
    # Simulate data
    import numpy as np
    from sklearn.datasets import make_classification
    
    X, y_true = make_classification(n_samples=1000, n_features=20, n_classes=5, 
                                   n_informative=10, random_state=42)
    
    # Simulate predictions
    np.random.seed(42)
    y_pred = np.random.randint(0, 5, 1000)
    y_prob = np.random.rand(1000, 5)
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
    
    # Evaluate
    result = evaluator.evaluate(config, y_true, y_pred, y_prob)
    
    # Print results
    print("Classification Evaluation Results:")
    print(f"Accuracy: {result.metrics['accuracy']:.4f}")
    print(f"Precision: {result.metrics['precision']:.4f}")
    print(f"Recall: {result.metrics['recall']:.4f}")
    print(f"F1 Score: {result.metrics['f1']:.4f}")
    print(f"ROC AUC: {result.metrics['roc_auc']:.4f}")
    print(f"Top-3 Accuracy: {result.metrics['top_k_accuracy']:.4f}")
    
    # Plot confusion matrix
    evaluator.plot_confusion_matrix(result.confusion_matrix)
    
    return result

asyncio.run(comprehensive_classification_evaluation())
```

### Example 2: Advanced Regression Evaluation

```python
async def advanced_regression_evaluation():
    device_manager = DeviceManager()
    evaluator = await create_evaluation_metrics(device_manager)
    
    # Create comprehensive metric config
    config = create_metric_config(
        task_type=TaskType.REGRESSION,
        metric_types=[
            MetricType.MSE, MetricType.MAE, MetricType.R2,
            MetricType.EXPLAINED_VARIANCE, MetricType.MAX_ERROR,
            MetricType.MAPE, MetricType.MSLE, MetricType.MEDIAN_AE,
            MetricType.MEAN_PINBALL_LOSS
        ]
    )
    
    # Simulate data
    import numpy as np
    from sklearn.datasets import make_regression
    
    X, y_true = make_regression(n_samples=1000, n_features=20, random_state=42)
    
    # Simulate predictions with some noise
    np.random.seed(42)
    y_pred = y_true + np.random.randn(1000) * 0.1
    
    # Evaluate
    result = evaluator.evaluate(config, y_true, y_pred)
    
    # Print results
    print("Regression Evaluation Results:")
    print(f"MSE: {result.metrics['mse']:.4f}")
    print(f"MAE: {result.metrics['mae']:.4f}")
    print(f"R²: {result.metrics['r2']:.4f}")
    print(f"Explained Variance: {result.metrics['explained_variance']:.4f}")
    print(f"MAPE: {result.metrics['mape']:.4f}")
    
    return result

asyncio.run(advanced_regression_evaluation())
```

### Example 3: Custom Metrics

```python
async def custom_metrics_evaluation():
    device_manager = DeviceManager()
    evaluator = await create_evaluation_metrics(device_manager)
    
    # Define custom metrics
    def custom_accuracy(y_true, y_pred, y_prob):
        return np.mean(y_true == y_pred) * 100
    
    def custom_f1_weighted(y_true, y_pred, y_prob):
        from sklearn.metrics import f1_score
        return f1_score(y_true, y_pred, average='weighted')
    
    def custom_rmse(y_true, y_pred, y_prob):
        return np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    # Create metric config with custom metrics
    config = create_metric_config(
        task_type=TaskType.CLASSIFICATION,
        metric_types=[MetricType.CUSTOM],
        custom_metrics={
            'custom_accuracy': custom_accuracy,
            'custom_f1_weighted': custom_f1_weighted,
            'custom_rmse': custom_rmse
        }
    )
    
    # Simulate data
    import numpy as np
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 0, 1, 0, 1, 0, 1])
    
    # Evaluate
    result = evaluator.evaluate(config, y_true, y_pred)
    
    # Print custom results
    print("Custom Metrics Results:")
    print(f"Custom Accuracy: {result.metrics['custom_accuracy']:.2f}%")
    print(f"Custom F1 Weighted: {result.metrics['custom_f1_weighted']:.4f}")
    print(f"Custom RMSE: {result.metrics['custom_rmse']:.4f}")
    
    return result

asyncio.run(custom_metrics_evaluation())
```

### Example 4: Task-Specific Evaluation

```python
async def task_specific_evaluation():
    device_manager = DeviceManager()
    evaluator = await create_evaluation_metrics(device_manager)
    
    # Test different task types
    task_configs = [
        (TaskType.SENTIMENT_ANALYSIS, "Sentiment Analysis"),
        (TaskType.TEXT_CLASSIFICATION, "Text Classification"),
        (TaskType.NAMED_ENTITY_RECOGNITION, "NER"),
        (TaskType.QUESTION_ANSWERING, "Question Answering"),
        (TaskType.ANOMALY_DETECTION, "Anomaly Detection")
    ]
    
    results = {}
    
    for task_type, task_name in task_configs:
        print(f"\nEvaluating {task_name}...")
        
        # Create appropriate data for each task
        if task_type == TaskType.ANOMALY_DETECTION:
            # Anomaly detection: mostly normal, few anomalies
            y_true = np.random.choice([0, 1], 1000, p=[0.95, 0.05])
            y_pred = np.random.choice([0, 1], 1000, p=[0.90, 0.10])
            y_prob = np.random.rand(1000, 2)
            y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
        else:
            # Standard classification
            y_true = np.random.randint(0, 3, 1000)
            y_pred = np.random.randint(0, 3, 1000)
            y_prob = np.random.rand(1000, 3)
            y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
        
        # Create metric config
        config = create_metric_config(task_type=task_type)
        
        # Evaluate
        result = evaluator.evaluate(config, y_true, y_pred, y_prob)
        
        # Store results
        results[task_name] = result
        
        # Print key metrics
        if 'accuracy' in result.metrics:
            print(f"  Accuracy: {result.metrics['accuracy']:.4f}")
        if 'f1' in result.metrics:
            print(f"  F1 Score: {result.metrics['f1']:.4f}")
        if 'roc_auc' in result.metrics:
            print(f"  ROC AUC: {result.metrics['roc_auc']:.4f}")
    
    return results

asyncio.run(task_specific_evaluation())
```

## Integration

### Integration with Model Training

```python
from agents.backend.onyx.server.features.blog_posts.model_training import ModelTrainer, TrainingConfig

async def training_with_evaluation():
    # Create trainer
    device_manager = DeviceManager()
    trainer = ModelTrainer(device_manager)
    
    # Configure training with evaluation
    config = TrainingConfig(
        model_type=ModelType.TRANSFORMER,
        training_mode=TrainingMode.FINE_TUNE,
        model_name="my_model",
        dataset_path="path/to/dataset",
        task_type="classification",  # Specify task type
        learning_rate=1e-3,
        num_epochs=10
    )
    
    # Train with automatic evaluation
    results = await trainer.train(config)
    
    # Access evaluation results
    evaluation_result = results['evaluation_result']
    print(f"Test Accuracy: {evaluation_result.test_accuracy:.4f}")
    print(f"Test F1: {evaluation_result.test_f1:.4f}")
    
    # Access detailed metrics from training monitor
    training_summary = results['training_summary']
    if 'evaluation_metrics' in training_summary:
        detailed_metrics = training_summary['evaluation_metrics']
        print(f"ROC AUC: {detailed_metrics.get('roc_auc', 'N/A')}")
        print(f"Precision: {detailed_metrics.get('precision', 'N/A')}")
    
    return results

asyncio.run(training_with_evaluation())
```

### Integration with Cross-Validation

```python
async def cross_validation_with_evaluation():
    device_manager = DeviceManager()
    trainer = ModelTrainer(device_manager)
    
    # Configure cross-validation with evaluation
    config = TrainingConfig(
        model_type=ModelType.TRANSFORMER,
        training_mode=TrainingMode.FINE_TUNE,
        model_name="my_model",
        dataset_path="path/to/dataset",
        task_type="classification",
        cross_validation_folds=5
    )
    
    # Perform cross-validation with evaluation
    cv_result = await trainer.perform_cross_validation(train_dataset, config)
    
    # Access cross-validation results
    print(f"Mean CV Accuracy: {cv_result.mean_scores.get('val_accuracy', 0):.4f}")
    print(f"Mean CV F1: {cv_result.mean_scores.get('val_f1_score', 0):.4f}")
    print(f"CV Standard Deviation: {cv_result.std_scores.get('val_accuracy', 0):.4f}")
    
    return cv_result

asyncio.run(cross_validation_with_evaluation())
```

## Best Practices

### 1. Choose Appropriate Metrics

```python
# For imbalanced classification
config = create_metric_config(
    task_type=TaskType.CLASSIFICATION,
    metric_types=[
        MetricType.BALANCED_ACCURACY, MetricType.F1, MetricType.ROC_AUC,
        MetricType.PR_AUC, MetricType.COHEN_KAPPA
    ],
    average='weighted'
)

# For multi-class classification
config = create_metric_config(
    task_type=TaskType.CLASSIFICATION,
    metric_types=[
        MetricType.ACCURACY, MetricType.F1, MetricType.ROC_AUC,
        MetricType.TOP_K_ACCURACY
    ],
    average='weighted',
    k=3
)

# For regression with outliers
config = create_metric_config(
    task_type=TaskType.REGRESSION,
    metric_types=[
        MetricType.MAE, MetricType.MEDIAN_AE, MetricType.R2,
        MetricType.EXPLAINED_VARIANCE
    ]
)
```

### 2. Handle Edge Cases

```python
# Handle single class
if len(np.unique(y_true)) == 1:
    config = create_metric_config(
        task_type=TaskType.CLASSIFICATION,
        metric_types=[MetricType.ACCURACY]  # Only accuracy makes sense
    )

# Handle very small datasets
if len(y_true) < 10:
    config = create_metric_config(
        task_type=TaskType.CLASSIFICATION,
        metric_types=[MetricType.ACCURACY, MetricType.CONFUSION_MATRIX]
    )
```

### 3. Use Appropriate Averaging

```python
# For balanced datasets
config = create_metric_config(
    task_type=TaskType.CLASSIFICATION,
    average='macro'  # Equal weight to all classes
)

# For imbalanced datasets
config = create_metric_config(
    task_type=TaskType.CLASSIFICATION,
    average='weighted'  # Weight by class frequency
)

# For micro-averaging
config = create_metric_config(
    task_type=TaskType.CLASSIFICATION,
    average='micro'  # Aggregate all classes
)
```

### 4. Save and Load Results

```python
# Save evaluation results
result.save("evaluation_results.json")

# Load evaluation results
loaded_result = EvaluationResult.load("evaluation_results.json")

# Compare results
print(f"Original Accuracy: {result.metrics['accuracy']:.4f}")
print(f"Loaded Accuracy: {loaded_result.metrics['accuracy']:.4f}")
```

### 5. Visualize Results

```python
# Plot confusion matrix
evaluator.plot_confusion_matrix(
    result.confusion_matrix,
    class_names=['Class 0', 'Class 1', 'Class 2'],
    save_path="confusion_matrix.png"
)

# Plot metrics comparison
results = [result1, result2, result3]
evaluator.plot_metrics_comparison(results, save_path="metrics_comparison.png")
```

## API Reference

### MetricConfig

```python
@dataclass
class MetricConfig:
    task_type: TaskType
    metric_types: List[MetricType] = field(default_factory=list)
    average: str = "weighted"
    beta: float = 1.0
    k: int = 5
    threshold: float = 0.5
    custom_metrics: Dict[str, Callable] = field(default_factory=dict)
    bleu_weights: Tuple[float, ...] = (0.25, 0.25, 0.25, 0.25)
    rouge_metrics: List[str] = field(default_factory=lambda: ['rouge1', 'rouge2', 'rougeL'])
    meteor_alpha: float = 0.9
    meteor_beta: float = 3.0
    multioutput: str = "uniform_average"
    sample_weight: Optional[np.ndarray] = None
```

### EvaluationResult

```python
@dataclass
class EvaluationResult:
    task_type: TaskType
    metrics: Dict[str, float] = field(default_factory=dict)
    confusion_matrix: Optional[np.ndarray] = None
    classification_report: Optional[str] = None
    detailed_metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]
    def save(self, filepath: str)
    @classmethod
    def load(cls, filepath: str) -> 'EvaluationResult'
```

### EvaluationMetrics

```python
class EvaluationMetrics:
    def __init__(self, device_manager: DeviceManager)
    
    def evaluate(self, config: MetricConfig, y_true: np.ndarray, 
                y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None,
                sample_weight: Optional[np.ndarray] = None) -> EvaluationResult
    
    def plot_confusion_matrix(self, confusion_matrix: np.ndarray,
                             class_names: Optional[List[str]] = None,
                             save_path: Optional[str] = None)
    
    def plot_metrics_comparison(self, results: List[EvaluationResult],
                               save_path: Optional[str] = None)
```

### Quick Functions

```python
async def create_evaluation_metrics(device_manager: DeviceManager) -> EvaluationMetrics

def create_metric_config(task_type: TaskType, 
                        metric_types: Optional[List[MetricType]] = None,
                        **kwargs) -> MetricConfig
```

This evaluation metrics system provides comprehensive, task-specific evaluation capabilities for all your AI training needs, ensuring accurate and meaningful model assessment across different domains and use cases. 