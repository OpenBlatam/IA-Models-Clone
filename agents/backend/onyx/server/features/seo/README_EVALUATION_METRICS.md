# Evaluation Metrics Framework

A comprehensive evaluation metrics framework for deep learning models, specifically designed for SEO tasks and various machine learning scenarios. This framework provides task-specific metrics, statistical analysis, visualization capabilities, and seamless integration with existing training pipelines.

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Task-Specific Metrics](#task-specific-metrics)
6. [SEO-Specific Metrics](#seo-specific-metrics)
7. [Statistical Analysis](#statistical-analysis)
8. [Visualization](#visualization)
9. [Advanced Usage](#advanced-usage)
10. [Integration](#integration)
11. [Best Practices](#best-practices)
12. [Examples](#examples)
13. [API Reference](#api-reference)
14. [Troubleshooting](#troubleshooting)

## Overview

This evaluation metrics framework provides comprehensive evaluation capabilities for deep learning models across different task types. It includes specialized metrics for SEO tasks, statistical analysis, confidence intervals, and advanced visualization capabilities.

### Key Benefits

- **Task-Specific Metrics**: Classification, regression, ranking, and multi-task evaluation
- **SEO-Optimized**: Specialized metrics for SEO and content analysis
- **Statistical Analysis**: Confidence intervals, hypothesis testing, and bootstrap analysis
- **Comprehensive Visualization**: ROC curves, confusion matrices, regression plots, and ranking metrics
- **Easy Integration**: Works with any PyTorch model and training framework
- **Extensible**: Support for custom metrics and evaluation strategies

## Features

### Core Evaluation Features

- ✅ **Classification Metrics**: Accuracy, precision, recall, F1, AUC, AP, confusion matrix
- ✅ **Regression Metrics**: MSE, RMSE, MAE, R², MAPE, SMAPE, correlation coefficients
- ✅ **Ranking Metrics**: NDCG, MRR, MAP, Precision@k, Recall@k
- ✅ **Multi-Task Metrics**: Task-specific and overall performance measures
- ✅ **SEO Metrics**: Ranking accuracy, CTR, bounce rate, content quality scores
- ✅ **Statistical Analysis**: Confidence intervals, hypothesis testing, bootstrap analysis
- ✅ **Visualization**: Comprehensive plotting capabilities for all metric types
- ✅ **Custom Metrics**: Support for user-defined evaluation functions

### Advanced Features

- ✅ **Confidence Intervals**: Bootstrap-based confidence intervals for all metrics
- ✅ **Statistical Testing**: Correlation analysis, distribution analysis, error analysis
- ✅ **Model Comparison**: Easy comparison of multiple models
- ✅ **Result Export**: JSON, CSV, and Excel output formats
- ✅ **Performance Optimization**: Efficient computation for large datasets
- ✅ **Integration**: Seamless integration with training frameworks

## Installation

```bash
# Install required dependencies
pip install torch torchvision torchaudio
pip install numpy pandas matplotlib seaborn
pip install scikit-learn scipy

# For additional visualization capabilities
pip install plotly bokeh

# For Excel export
pip install openpyxl
```

## Quick Start

```python
import numpy as np
from evaluation_metrics import EvaluationConfig, ModelEvaluator

# Create sample data
y_true = np.random.randint(0, 3, 1000)
y_pred = np.random.randint(0, 3, 1000)
y_prob = np.random.rand(1000, 3)
y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)

# Configure evaluation
config = EvaluationConfig(
    task_type="classification",
    classification_metrics=["accuracy", "precision", "recall", "f1", "auc"],
    average_method="weighted",
    create_plots=True,
    save_plots=True
)

# Create evaluator and evaluate
evaluator = ModelEvaluator(config)
result = evaluator.evaluate(y_true, y_pred, y_prob)

# Print results
for metric, value in result.metrics.items():
    print(f"{metric}: {value:.4f}")

# Save results
evaluator.save_results(result, "evaluation_results.json")
```

## Task-Specific Metrics

### Classification Metrics

```python
from evaluation_metrics import ClassificationMetrics

# Calculate all classification metrics
metrics = ClassificationMetrics.calculate_all_metrics(y_true, y_pred, y_prob)

# Individual metrics
accuracy = ClassificationMetrics.calculate_accuracy(y_true, y_pred)
precision = ClassificationMetrics.calculate_precision(y_true, y_pred, average="weighted")
recall = ClassificationMetrics.calculate_recall(y_true, y_pred, average="weighted")
f1 = ClassificationMetrics.calculate_f1(y_true, y_pred, average="weighted")
auc = ClassificationMetrics.calculate_auc(y_true, y_prob, average="weighted")
ap = ClassificationMetrics.calculate_average_precision(y_true, y_prob, average="weighted")
confusion = ClassificationMetrics.calculate_confusion_matrix(y_true, y_pred)
kappa = ClassificationMetrics.calculate_cohen_kappa(y_true, y_pred)
mcc = ClassificationMetrics.calculate_matthews_correlation(y_true, y_pred)
log_loss = ClassificationMetrics.calculate_log_loss(y_true, y_prob)
```

### Regression Metrics

```python
from evaluation_metrics import RegressionMetrics

# Calculate all regression metrics
metrics = RegressionMetrics.calculate_all_metrics(y_true, y_pred)

# Individual metrics
mse = RegressionMetrics.calculate_mse(y_true, y_pred)
rmse = RegressionMetrics.calculate_rmse(y_true, y_pred)
mae = RegressionMetrics.calculate_mae(y_true, y_pred)
r2 = RegressionMetrics.calculate_r2(y_true, y_pred)
mape = RegressionMetrics.calculate_mape(y_true, y_pred)
smape = RegressionMetrics.calculate_smape(y_true, y_pred)
pearson = RegressionMetrics.calculate_pearson_correlation(y_true, y_pred)
spearman = RegressionMetrics.calculate_spearman_correlation(y_true, y_pred)
kendall = RegressionMetrics.calculate_kendall_correlation(y_true, y_pred)
```

### Ranking Metrics

```python
from evaluation_metrics import RankingMetrics

# Calculate all ranking metrics
metrics = RankingMetrics.calculate_all_metrics(y_true, y_pred, k_values=[1, 3, 5, 10])

# Individual metrics
ndcg = RankingMetrics.calculate_ndcg(y_true, y_pred, k=10)
mrr = RankingMetrics.calculate_mrr(y_true, y_pred)
map_score = RankingMetrics.calculate_map(y_true, y_pred)
precision_at_k = RankingMetrics.calculate_precision_at_k(y_true, y_pred, k=5)
recall_at_k = RankingMetrics.calculate_recall_at_k(y_true, y_pred, k=5)
```

### Multi-Task Metrics

```python
from evaluation_metrics import MultiTaskMetrics

# Multi-task data
y_true = {
    'task1': np.random.randint(0, 3, 1000),
    'task2': np.random.randint(0, 2, 1000),
    'task3': np.random.normal(0, 1, 1000)
}

y_pred = {
    'task1': np.random.randint(0, 3, 1000),
    'task2': np.random.randint(0, 2, 1000),
    'task3': np.random.normal(0, 1, 1000)
}

# Calculate all multi-task metrics
metrics = MultiTaskMetrics.calculate_all_metrics(y_true, y_pred)
```

## SEO-Specific Metrics

### SEO Metrics Overview

The framework includes specialized metrics for SEO tasks:

- **Ranking Accuracy**: Measure how well predictions match actual search rankings
- **Click-Through Rate (CTR)**: Percentage of impressions that result in clicks
- **Bounce Rate**: Percentage of single-page sessions
- **Time on Page**: Average time users spend on content
- **Conversion Rate**: Percentage of sessions that result in conversions
- **Organic Traffic**: Percentage of traffic from organic search
- **Keyword Density**: Frequency of target keywords in content
- **Content Quality Score**: Composite score based on multiple content factors

### SEO Metrics Usage

```python
from evaluation_metrics import SEOMetrics

# Ranking data
ranking_data = {
    'y_true': np.random.randint(1, 11, 1000),  # True rankings 1-10
    'y_pred': np.random.randint(1, 11, 1000)   # Predicted rankings 1-10
}

# Traffic data
traffic_data = {
    'clicks': np.random.poisson(50, 1000),
    'impressions': np.random.poisson(1000, 1000),
    'bounces': np.random.poisson(20, 1000),
    'sessions': np.random.poisson(100, 1000),
    'time_spent': np.random.exponential(120, 1000),
    'page_views': np.random.poisson(2, 1000),
    'conversions': np.random.poisson(5, 1000),
    'organic_sessions': np.random.poisson(80, 1000),
    'total_sessions': np.random.poisson(100, 1000)
}

# Content data
content_data = {
    'text': "Sample SEO content with relevant keywords for search engine optimization.",
    'keyword': "SEO",
    'readability_score': 75.5,
    'word_count': 1500,
    'keyword_density': 0.02,
    'internal_links': 8
}

# Calculate all SEO metrics
metrics = SEOMetrics.calculate_all_metrics(ranking_data, traffic_data, content_data)
```

### Individual SEO Metrics

```python
# Ranking accuracy with tolerance
ranking_acc = SEOMetrics.calculate_ranking_accuracy(y_true, y_pred, tolerance=0.1)

# Click-through rate
ctr = SEOMetrics.calculate_click_through_rate(clicks, impressions)

# Bounce rate
bounce_rate = SEOMetrics.calculate_bounce_rate(bounces, sessions)

# Time on page
time_on_page = SEOMetrics.calculate_time_on_page(time_spent, page_views)

# Conversion rate
conversion_rate = SEOMetrics.calculate_conversion_rate(conversions, sessions)

# Organic traffic percentage
organic_traffic = SEOMetrics.calculate_organic_traffic(organic_sessions, total_sessions)

# Keyword density
keyword_density = SEOMetrics.calculate_keyword_density(text, keyword)

# Content quality score
quality_score = SEOMetrics.calculate_content_quality_score(
    readability_score, word_count, keyword_density, internal_links
)
```

## Statistical Analysis

### Confidence Intervals

```python
from evaluation_metrics import StatisticalAnalysis

# Bootstrap confidence intervals
confidence_intervals = StatisticalAnalysis.bootstrap_confidence_intervals(
    y_true, y_pred, 
    metric_func=lambda y_t, y_p: f1_score(y_t, y_p, average='weighted'),
    n_bootstrap=1000,
    confidence_level=0.95
)

# Simple confidence intervals
ci = StatisticalAnalysis.calculate_confidence_intervals(metric_values, confidence_level=0.95)
```

### Statistical Tests

```python
# Perform comprehensive statistical tests
statistical_tests = StatisticalAnalysis.perform_statistical_tests(
    y_true, y_pred, baseline_pred
)

# Results include:
# - Correlation with baseline
# - Prediction distribution analysis
# - Error analysis
```

## Visualization

### Classification Visualizations

```python
from evaluation_metrics import EvaluationVisualizer

# Confusion matrix
EvaluationVisualizer.plot_confusion_matrix(y_true, y_pred, class_names=['A', 'B', 'C'])

# ROC curve
EvaluationVisualizer.plot_roc_curve(y_true, y_prob, class_names=['A', 'B', 'C'])

# Precision-Recall curve
EvaluationVisualizer.plot_precision_recall_curve(y_true, y_prob)
```

### Regression Visualizations

```python
# Comprehensive regression plots
EvaluationVisualizer.plot_regression_results(y_true, y_pred)
# Includes: scatter plot, residuals plot, residuals histogram, Q-Q plot
```

### Ranking Visualizations

```python
# Ranking metrics plots
EvaluationVisualizer.plot_ranking_metrics(metrics, k_values=[1, 3, 5, 10])
# Includes: NDCG@k, Precision@k, Recall@k, overall metrics
```

## Advanced Usage

### Custom Metrics

```python
def custom_balanced_accuracy(y_true, y_pred):
    """Custom balanced accuracy metric"""
    from sklearn.metrics import balanced_accuracy_score
    return balanced_accuracy_score(y_true, y_pred)

def custom_hamming_loss(y_true, y_pred):
    """Custom Hamming loss metric"""
    return np.mean(y_true != y_pred)

# Use custom metrics in evaluation
config = EvaluationConfig(
    task_type="classification",
    classification_metrics=["accuracy", "precision", "recall", "f1"],
    create_plots=True
)

evaluator = ModelEvaluator(config)
result = evaluator.evaluate(y_true, y_pred, y_prob)

# Add custom metrics
result.metrics['balanced_accuracy'] = custom_balanced_accuracy(y_true, y_pred)
result.metrics['hamming_loss'] = custom_hamming_loss(y_true, y_pred)
```

### Model Comparison

```python
# Compare multiple models
models = {
    'Model_A': y_pred_a,
    'Model_B': y_pred_b,
    'Model_C': y_pred_c
}

results = {}
for model_name, y_pred in models.items():
    result = evaluator.evaluate(y_true, y_pred, y_prob)
    results[model_name] = result.metrics

# Create comparison DataFrame
comparison_df = pd.DataFrame(results).T

# Find best model for each metric
for metric in ['accuracy', 'f1', 'auc']:
    best_model = comparison_df[metric].idxmax()
    best_score = comparison_df[metric].max()
    print(f"Best {metric}: {best_model} ({best_score:.4f})")
```

### Comprehensive Evaluation

```python
# Comprehensive evaluation with all features
config = EvaluationConfig(
    task_type="classification",
    classification_metrics=["accuracy", "precision", "recall", "f1", "auc", "ap"],
    average_method="weighted",
    statistical_analysis=True,
    confidence_intervals=True,
    bootstrap_samples=1000,
    confidence_level=0.95,
    create_plots=True,
    save_plots=True,
    save_results=True,
    output_format="json",
    verbose=True
)

evaluator = ModelEvaluator(config)
result = evaluator.evaluate(y_true, y_pred, y_prob)

# Save comprehensive results
evaluator.save_results(result, "comprehensive_evaluation.json")
```

## Integration

### Integration with Training Framework

```python
from model_training_evaluation import ModelEvaluator as TrainingEvaluator
from evaluation_metrics import ModelEvaluator as MetricsEvaluator

# Use with training framework
training_evaluator = TrainingEvaluator(model, device)
training_metrics = training_evaluator.evaluate(test_loader)

# Use with metrics framework
metrics_config = EvaluationConfig(task_type="classification")
metrics_evaluator = MetricsEvaluator(metrics_config)
metrics_result = metrics_evaluator.evaluate(y_true, y_pred, y_prob)
```

### Integration with Deep Learning Framework

```python
from deep_learning_framework import DeepLearningFramework

# Create framework
framework = DeepLearningFramework(config)

# Evaluate with comprehensive metrics
evaluation_config = EvaluationConfig(
    task_type="classification",
    create_plots=True,
    save_results=True
)

result = framework.evaluate_with_comprehensive_metrics(
    model, test_dataset, evaluation_config
)
```

### Integration with Early Stopping Framework

```python
from early_stopping_lr_scheduling import TrainingOptimizer

# Create training optimizer with evaluation
trainer = TrainingOptimizer(
    model=model,
    optimizer=optimizer,
    early_stopping_config=early_stopping_config,
    lr_scheduler_config=lr_scheduler_config
)

# Train with evaluation
summary = trainer.train(train_loader, val_loader, criterion, device)

# Get evaluation results
evaluation_result = trainer.monitor.get_training_summary()
```

## Best Practices

### Metric Selection

1. **Classification Tasks**
   ```python
   # For balanced datasets
   metrics = ["accuracy", "precision", "recall", "f1"]
   
   # For imbalanced datasets
   metrics = ["precision", "recall", "f1", "auc", "ap"]
   
   # For multi-class
   metrics = ["accuracy", "precision", "recall", "f1", "cohen_kappa"]
   ```

2. **Regression Tasks**
   ```python
   # For general regression
   metrics = ["mse", "rmse", "mae", "r2"]
   
   # For percentage errors
   metrics = ["mape", "smape"]
   
   # For correlation analysis
   metrics = ["pearson_correlation", "spearman_correlation"]
   ```

3. **Ranking Tasks**
   ```python
   # For general ranking
   metrics = ["ndcg", "mrr", "map"]
   
   # For specific positions
   metrics = ["precision@1", "precision@5", "precision@10"]
   ```

4. **SEO Tasks**
   ```python
   # For ranking prediction
   metrics = ["ranking_accuracy", "ndcg", "mrr"]
   
   # For traffic analysis
   metrics = ["click_through_rate", "bounce_rate", "conversion_rate"]
   
   # For content analysis
   metrics = ["keyword_density", "content_quality_score"]
   ```

### Configuration Best Practices

```python
# Basic configuration
config = EvaluationConfig(
    task_type="classification",
    classification_metrics=["accuracy", "precision", "recall", "f1"],
    create_plots=True,
    save_results=True
)

# Advanced configuration
config = EvaluationConfig(
    task_type="classification",
    classification_metrics=["accuracy", "precision", "recall", "f1", "auc", "ap"],
    average_method="weighted",
    statistical_analysis=True,
    confidence_intervals=True,
    bootstrap_samples=1000,
    confidence_level=0.95,
    create_plots=True,
    save_plots=True,
    save_results=True,
    output_format="json",
    verbose=True
)
```

### Performance Optimization

```python
# For large datasets
config = EvaluationConfig(
    bootstrap_samples=100,  # Reduce for speed
    create_plots=False,     # Disable if not needed
    verbose=False           # Reduce logging
)

# For production use
config = EvaluationConfig(
    statistical_analysis=False,  # Disable for speed
    confidence_intervals=False,
    create_plots=False,
    save_plots=False,
    save_results=True,
    output_format="json"
)
```

## Examples

### Example 1: Basic Classification Evaluation

```python
import numpy as np
from evaluation_metrics import EvaluationConfig, ModelEvaluator

# Create sample data
y_true = np.random.randint(0, 3, 1000)
y_pred = np.random.randint(0, 3, 1000)
y_prob = np.random.rand(1000, 3)
y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)

# Configure evaluation
config = EvaluationConfig(
    task_type="classification",
    classification_metrics=["accuracy", "precision", "recall", "f1", "auc"],
    average_method="weighted",
    create_plots=True
)

# Evaluate
evaluator = ModelEvaluator(config)
result = evaluator.evaluate(y_true, y_pred, y_prob)

# Print results
for metric, value in result.metrics.items():
    print(f"{metric}: {value:.4f}")
```

### Example 2: SEO-Specific Evaluation

```python
# SEO ranking evaluation
ranking_data = {
    'y_true': np.random.randint(1, 11, 1000),
    'y_pred': np.random.randint(1, 11, 1000)
}

traffic_data = {
    'clicks': np.random.poisson(50, 1000),
    'impressions': np.random.poisson(1000, 1000),
    'bounces': np.random.poisson(20, 1000),
    'sessions': np.random.poisson(100, 1000)
}

content_data = {
    'text': "Sample SEO content with keywords for optimization.",
    'keyword': "SEO",
    'readability_score': 75.5,
    'word_count': 1500,
    'keyword_density': 0.02,
    'internal_links': 8
}

config = EvaluationConfig(task_type="seo")
evaluator = ModelEvaluator(config)
result = evaluator.evaluate(None, None, ranking_data=ranking_data, 
                          traffic_data=traffic_data, content_data=content_data)

for metric, value in result.metrics.items():
    print(f"{metric}: {value:.4f}")
```

### Example 3: Model Comparison

```python
# Compare multiple models
models = {
    'BERT': y_pred_bert,
    'RoBERTa': y_pred_roberta,
    'DistilBERT': y_pred_distilbert
}

results = {}
config = EvaluationConfig(task_type="classification")
evaluator = ModelEvaluator(config)

for model_name, y_pred in models.items():
    result = evaluator.evaluate(y_true, y_pred, y_prob)
    results[model_name] = result.metrics

# Create comparison
comparison_df = pd.DataFrame(results).T
print(comparison_df.round(4))

# Find best model
best_model = comparison_df['f1'].idxmax()
print(f"Best model by F1: {best_model}")
```

### Example 4: Advanced Statistical Analysis

```python
# Comprehensive evaluation with statistical analysis
config = EvaluationConfig(
    task_type="classification",
    statistical_analysis=True,
    confidence_intervals=True,
    bootstrap_samples=1000,
    confidence_level=0.95
)

evaluator = ModelEvaluator(config)
result = evaluator.evaluate(y_true, y_pred, y_prob)

# Print confidence intervals
if result.confidence_intervals:
    for metric, (lower, upper) in result.confidence_intervals.items():
        print(f"{metric}: ({lower:.4f}, {upper:.4f})")

# Print statistical tests
if result.statistical_tests:
    for test_name, test_result in result.statistical_tests.items():
        print(f"{test_name}: {test_result}")
```

## API Reference

### EvaluationConfig

```python
@dataclass
class EvaluationConfig:
    task_type: str = "classification"
    classification_metrics: List[str] = field(default_factory=lambda: [
        "accuracy", "precision", "recall", "f1", "auc", "ap", "confusion_matrix"
    ])
    average_method: str = "weighted"
    regression_metrics: List[str] = field(default_factory=lambda: [
        "mse", "rmse", "mae", "r2", "mape", "smape"
    ])
    ranking_metrics: List[str] = field(default_factory=lambda: [
        "ndcg", "mrr", "map", "precision_at_k", "recall_at_k"
    ])
    k_values: List[int] = field(default_factory=lambda: [1, 3, 5, 10])
    seo_metrics: List[str] = field(default_factory=lambda: [
        "ranking_accuracy", "click_through_rate", "bounce_rate", "time_on_page",
        "conversion_rate", "organic_traffic", "keyword_density", "content_quality"
    ])
    multitask_metrics: List[str] = field(default_factory=lambda: [
        "task_accuracy", "overall_accuracy", "task_f1", "overall_f1"
    ])
    statistical_analysis: bool = True
    confidence_intervals: bool = True
    bootstrap_samples: int = 1000
    confidence_level: float = 0.95
    create_plots: bool = True
    save_plots: bool = True
    plot_format: str = "png"
    dpi: int = 300
    save_results: bool = True
    output_format: str = "json"
    verbose: bool = True
```

### ModelEvaluator

```python
class ModelEvaluator:
    def __init__(self, config: EvaluationConfig)
    
    def evaluate_classification(self, y_true: np.ndarray, y_pred: np.ndarray, 
                              y_prob: Optional[np.ndarray] = None) -> EvaluationResult
    
    def evaluate_regression(self, y_true: np.ndarray, y_pred: np.ndarray) -> EvaluationResult
    
    def evaluate_ranking(self, y_true: np.ndarray, y_pred: np.ndarray) -> EvaluationResult
    
    def evaluate_seo(self, ranking_data: Dict[str, np.ndarray], 
                    traffic_data: Dict[str, np.ndarray],
                    content_data: Dict[str, Any]) -> EvaluationResult
    
    def evaluate_multitask(self, y_true: Dict[str, np.ndarray], 
                          y_pred: Dict[str, np.ndarray]) -> EvaluationResult
    
    def evaluate(self, y_true: Union[np.ndarray, Dict[str, np.ndarray]], 
                y_pred: Union[np.ndarray, Dict[str, np.ndarray]],
                y_prob: Optional[np.ndarray] = None,
                **kwargs) -> EvaluationResult
    
    def save_results(self, result: EvaluationResult, save_path: str)
    
    def get_summary(self) -> Dict[str, Any]
```

### EvaluationResult

```python
@dataclass
class EvaluationResult:
    task_type: str
    metrics: Dict[str, float]
    predictions: Optional[np.ndarray] = None
    targets: Optional[np.ndarray] = None
    probabilities: Optional[np.ndarray] = None
    confidence_intervals: Optional[Dict[str, Tuple[float, float]]] = None
    statistical_tests: Optional[Dict[str, Any]] = None
    timestamp: float = field(default_factory=time.time)
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```python
   # Ensure all dependencies are installed
   pip install scikit-learn scipy matplotlib seaborn
   ```

2. **Memory Issues with Large Datasets**
   ```python
   # Reduce bootstrap samples
   config = EvaluationConfig(bootstrap_samples=100)
   
   # Disable statistical analysis
   config = EvaluationConfig(statistical_analysis=False)
   ```

3. **Plotting Issues**
   ```python
   # Disable plots if matplotlib is not available
   config = EvaluationConfig(create_plots=False)
   
   # Use different backend
   import matplotlib
   matplotlib.use('Agg')
   ```

4. **Metric Calculation Errors**
   ```python
   # Check data types
   y_true = y_true.astype(int)
   y_pred = y_pred.astype(int)
   
   # Handle edge cases
   if len(np.unique(y_true)) == 1:
       print("Warning: Only one class in true labels")
   ```

### Performance Optimization

1. **Large Datasets**
   ```python
   # Use sampling for statistical analysis
   config = EvaluationConfig(
       bootstrap_samples=100,  # Reduce from 1000
       statistical_analysis=False  # Disable if not needed
   )
   ```

2. **Production Use**
   ```python
   # Minimal configuration for production
   config = EvaluationConfig(
       create_plots=False,
       save_plots=False,
       statistical_analysis=False,
       confidence_intervals=False,
       verbose=False
   )
   ```

3. **Memory Management**
   ```python
   # Clear large arrays after use
   del y_true, y_pred, y_prob
   import gc
   gc.collect()
   ```

## Conclusion

The evaluation metrics framework provides comprehensive evaluation capabilities for deep learning models, with special emphasis on SEO tasks. It includes task-specific metrics, statistical analysis, visualization capabilities, and seamless integration with existing frameworks.

The framework is designed to be flexible, efficient, and easy to use while providing advanced capabilities for complex evaluation scenarios. It successfully addresses the need for robust evaluation in SEO deep learning applications and general machine learning tasks.

For more advanced usage and integration examples, refer to the example scripts and the main training framework documentation. 