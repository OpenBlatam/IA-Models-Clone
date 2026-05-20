# Comprehensive Evaluation Metrics System

## Overview

This document provides a comprehensive overview of the evaluation metrics system that implements advanced evaluation techniques for different deep learning tasks including classification, regression, generation, and clustering.

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Task Types](#task-types)
3. [Classification Metrics](#classification-metrics)
4. [Regression Metrics](#regression-metrics)
5. [Generation Metrics](#generation-metrics)
6. [Clustering Metrics](#clustering-metrics)
7. [Evaluation Manager](#evaluation-manager)
8. [Configuration Options](#configuration-options)
9. [Usage Examples](#usage-examples)
10. [Best Practices](#best-practices)
11. [Visualization and Analysis](#visualization-and-analysis)

## System Architecture

### Core Components

The evaluation metrics system consists of several key components:

```python
class TaskType(Enum):
    """Types of deep learning tasks."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    GENERATION = "generation"
    SEGMENTATION = "segmentation"
    DETECTION = "detection"
    CLUSTERING = "clustering"
    RECOMMENDATION = "recommendation"
    ANOMALY_DETECTION = "anomaly_detection"
    MULTI_LABEL = "multi_label"
    MULTI_TASK = "multi_task"
```

### Metric Types

```python
class MetricType(Enum):
    """Types of evaluation metrics."""
    # Classification metrics
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    ROC_AUC = "roc_auc"
    PR_AUC = "pr_auc"
    CONFUSION_MATRIX = "confusion_matrix"
    CLASSIFICATION_REPORT = "classification_report"
    LOG_LOSS = "log_loss"
    COHEN_KAPPA = "cohen_kappa"
    MATTHEWS_CORRELATION = "matthews_correlation"
    
    # Regression metrics
    MSE = "mse"
    RMSE = "rmse"
    MAE = "mae"
    R2_SCORE = "r2_score"
    EXPLAINED_VARIANCE = "explained_variance"
    MAX_ERROR = "max_error"
    MEAN_ABSOLUTE_PERCENTAGE_ERROR = "mape"
    
    # Generation metrics
    BLEU = "bleu"
    METEOR = "meteor"
    ROUGE = "rouge"
    NIST = "nist"
    PERPLEXITY = "perplexity"
    DIVERSITY = "diversity"
    COHERENCE = "coherence"
    
    # Clustering metrics
    SILHOUETTE_SCORE = "silhouette_score"
    CALINSKI_HARABASZ = "calinski_harabasz"
    DAVIES_BOULDIN = "davies_bouldin"
    ADJUSTED_RAND = "adjusted_rand"
    NORMALIZED_MUTUAL_INFO = "normalized_mutual_info"
    HOMOGENEITY = "homogeneity"
    COMPLETENESS = "completeness"
    V_MEASURE = "v_measure"
    
    # Custom metrics
    CUSTOM = "custom"
```

### Configuration

```python
@dataclass
class MetricConfig:
    """Configuration for evaluation metrics."""
    # Task type
    task_type: TaskType = TaskType.CLASSIFICATION
    
    # Metric selection
    metrics: List[MetricType] = field(default_factory=lambda: [MetricType.ACCURACY, MetricType.F1_SCORE])
    
    # Classification settings
    average: str = "weighted"  # micro, macro, weighted, samples
    zero_division: int = 0
    
    # Regression settings
    multioutput: str = "uniform_average"  # raw_values, uniform_average, variance_weighted
    
    # Generation settings
    n_gram: int = 4  # for BLEU and ROUGE
    smoothing: bool = True
    smoothing_method: str = "add-k"
    smoothing_value: float = 0.1
    
    # Clustering settings
    metric: str = "euclidean"  # for silhouette score
    
    # Custom settings
    custom_metrics: Dict[str, Callable] = field(default_factory=dict)
    
    # Output settings
    save_results: bool = True
    results_file: str = "evaluation_results.json"
    plot_results: bool = True
    plot_file: str = "evaluation_plots.png"
```

## Task Types

### Supported Task Types

1. **Classification**: Binary and multi-class classification tasks
2. **Regression**: Continuous value prediction tasks
3. **Generation**: Text generation and language modeling tasks
4. **Segmentation**: Image and text segmentation tasks
5. **Detection**: Object detection and anomaly detection tasks
6. **Clustering**: Unsupervised clustering tasks
7. **Recommendation**: Recommendation system evaluation
8. **Anomaly Detection**: Outlier detection tasks
9. **Multi-Label**: Multi-label classification tasks
10. **Multi-Task**: Multi-task learning evaluation

## Classification Metrics

### Core Classification Metrics

#### **1. Accuracy**
```python
def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, 
             y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Evaluate classification metrics."""
    if metric_type == MetricType.ACCURACY:
        results['accuracy'] = accuracy_score(y_true, y_pred)
```

#### **2. Precision**
```python
elif metric_type == MetricType.PRECISION:
    results['precision'] = precision_score(
        y_true, y_pred, average=self.config.average, 
        zero_division=self.config.zero_division
    )
```

#### **3. Recall**
```python
elif metric_type == MetricType.RECALL:
    results['recall'] = recall_score(
        y_true, y_pred, average=self.config.average,
        zero_division=self.config.zero_division
    )
```

#### **4. F1 Score**
```python
elif metric_type == MetricType.F1_SCORE:
    results['f1_score'] = f1_score(
        y_true, y_pred, average=self.config.average,
        zero_division=self.config.zero_division
    )
```

#### **5. ROC AUC**
```python
elif metric_type == MetricType.ROC_AUC:
    if y_prob is not None:
        if len(np.unique(y_true)) == 2:
            # Binary classification
            results['roc_auc'] = roc_auc_score(y_true, y_prob[:, 1])
        else:
            # Multi-class classification
            results['roc_auc'] = roc_auc_score(
                y_true, y_prob, multi_class='ovr', average=self.config.average
            )
    else:
        self.logger.warning("ROC AUC requires probability predictions")
```

#### **6. PR AUC**
```python
elif metric_type == MetricType.PR_AUC:
    if y_prob is not None:
        if len(np.unique(y_true)) == 2:
            # Binary classification
            results['pr_auc'] = average_precision_score(y_true, y_prob[:, 1])
        else:
            # Multi-class classification
            results['pr_auc'] = average_precision_score(
                y_true, y_prob, average=self.config.average
            )
    else:
        self.logger.warning("PR AUC requires probability predictions")
```

#### **7. Confusion Matrix**
```python
elif metric_type == MetricType.CONFUSION_MATRIX:
    results['confusion_matrix'] = confusion_matrix(y_true, y_pred)
```

#### **8. Classification Report**
```python
elif metric_type == MetricType.CLASSIFICATION_REPORT:
    results['classification_report'] = classification_report(
        y_true, y_pred, output_dict=True
    )
```

#### **9. Log Loss**
```python
elif metric_type == MetricType.LOG_LOSS:
    if y_prob is not None:
        results['log_loss'] = log_loss(y_true, y_prob)
    else:
        self.logger.warning("Log loss requires probability predictions")
```

#### **10. Cohen's Kappa**
```python
elif metric_type == MetricType.COHEN_KAPPA:
    results['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
```

#### **11. Matthews Correlation**
```python
elif metric_type == MetricType.MATTHEWS_CORRELATION:
    if len(np.unique(y_true)) == 2:
        results['matthews_correlation'] = matthews_corrcoef(y_true, y_pred)
    else:
        self.logger.warning("Matthews correlation only for binary classification")
```

### Classification Visualization

#### **1. Confusion Matrix Plot**
```python
def plot_confusion_matrix(self, save_path: Optional[str] = None):
    """Plot confusion matrix."""
    if 'confusion_matrix' not in self.results:
        self.logger.warning("No confusion matrix available")
        return
    
    cm = self.results['confusion_matrix']
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"Confusion matrix plot saved to {save_path}")
    
    plt.show()
```

#### **2. ROC Curve Plot**
```python
def plot_roc_curve(self, y_true: np.ndarray, y_prob: np.ndarray, 
                   save_path: Optional[str] = None):
    """Plot ROC curve."""
    if len(np.unique(y_true)) != 2:
        self.logger.warning("ROC curve only for binary classification")
        return
    
    fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
    auc = roc_auc_score(y_true, y_prob[:, 1])
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"ROC curve plot saved to {save_path}")
    
    plt.show()
```

#### **3. Precision-Recall Curve Plot**
```python
def plot_precision_recall_curve(self, y_true: np.ndarray, y_prob: np.ndarray,
                               save_path: Optional[str] = None):
    """Plot Precision-Recall curve."""
    if len(np.unique(y_true)) != 2:
        self.logger.warning("PR curve only for binary classification")
        return
    
    precision, recall, _ = precision_recall_curve(y_true, y_prob[:, 1])
    ap = average_precision_score(y_true, y_prob[:, 1])
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'PR Curve (AP = {ap:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"PR curve plot saved to {save_path}")
    
    plt.show()
```

## Regression Metrics

### Core Regression Metrics

#### **1. Mean Squared Error (MSE)**
```python
def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Evaluate regression metrics."""
    if metric_type == MetricType.MSE:
        results['mse'] = mean_squared_error(y_true, y_pred, multioutput=self.config.multioutput)
```

#### **2. Root Mean Squared Error (RMSE)**
```python
elif metric_type == MetricType.RMSE:
    results['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred, multioutput=self.config.multioutput))
```

#### **3. Mean Absolute Error (MAE)**
```python
elif metric_type == MetricType.MAE:
    results['mae'] = mean_absolute_error(y_true, y_pred, multioutput=self.config.multioutput)
```

#### **4. R² Score**
```python
elif metric_type == MetricType.R2_SCORE:
    results['r2_score'] = r2_score(y_true, y_pred, multioutput=self.config.multioutput)
```

#### **5. Explained Variance**
```python
elif metric_type == MetricType.EXPLAINED_VARIANCE:
    from sklearn.metrics import explained_variance_score
    results['explained_variance'] = explained_variance_score(y_true, y_pred, multioutput=self.config.multioutput)
```

#### **6. Max Error**
```python
elif metric_type == MetricType.MAX_ERROR:
    from sklearn.metrics import max_error
    results['max_error'] = max_error(y_true, y_pred)
```

#### **7. Mean Absolute Percentage Error (MAPE)**
```python
elif metric_type == MetricType.MEAN_ABSOLUTE_PERCENTAGE_ERROR:
    results['mape'] = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
```

### Regression Visualization

#### **1. Predictions vs Actual Plot**
```python
def plot_predictions(self, y_true: np.ndarray, y_pred: np.ndarray,
                    save_path: Optional[str] = None):
    """Plot predictions vs actual values."""
    plt.figure(figsize=(10, 8))
    
    # Scatter plot
    plt.subplot(2, 2, 1)
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Predictions vs Actual')
    plt.grid(True)
    
    # Residuals plot
    plt.subplot(2, 2, 2)
    residuals = y_true - y_pred
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals Plot')
    plt.grid(True)
    
    # Residuals histogram
    plt.subplot(2, 2, 3)
    plt.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Residuals Distribution')
    plt.grid(True)
    
    # Q-Q plot
    plt.subplot(2, 2, 4)
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q Plot')
    plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"Regression plots saved to {save_path}")
    
    plt.show()
```

## Generation Metrics

### Core Generation Metrics

#### **1. BLEU Score**
```python
def _calculate_bleu(self, references: List[List[List[str]]], 
                   predictions: List[List[str]], n: int) -> float:
    """Calculate BLEU score."""
    if self.config.smoothing:
        from nltk.translate.bleu_score import SmoothingFunction
        smoothing = SmoothingFunction().method1
    else:
        smoothing = None
    
    scores = []
    for refs, pred in zip(references, predictions):
        try:
            score = sentence_bleu(refs, pred, weights=tuple([1.0/n] * n), smoothing_function=smoothing)
            scores.append(score)
        except:
            scores.append(0.0)
    
    return np.mean(scores)
```

#### **2. METEOR Score**
```python
elif metric_type == MetricType.METEOR:
    meteor_scores = []
    for refs, pred in zip(references, predictions):
        try:
            score = meteor_score(refs, pred)
            meteor_scores.append(score)
        except:
            meteor_scores.append(0.0)
    results['meteor'] = np.mean(meteor_scores)
```

#### **3. ROUGE Score**
```python
elif metric_type == MetricType.ROUGE:
    rouge_1_scores = []
    rouge_2_scores = []
    rouge_l_scores = []
    
    for refs, pred in zip(references, predictions):
        # ROUGE-1
        try:
            score = rouge_n([pred], [refs[0]], 1)
            rouge_1_scores.append(score['rouge-1']['f'])
        except:
            rouge_1_scores.append(0.0)
        
        # ROUGE-2
        try:
            score = rouge_n([pred], [refs[0]], 2)
            rouge_2_scores.append(score['rouge-2']['f'])
        except:
            rouge_2_scores.append(0.0)
        
        # ROUGE-L
        try:
            score = rouge_l([pred], [refs[0]])
            rouge_l_scores.append(score['rouge-l']['f'])
        except:
            rouge_l_scores.append(0.0)
    
    results['rouge_1'] = np.mean(rouge_1_scores)
    results['rouge_2'] = np.mean(rouge_2_scores)
    results['rouge_l'] = np.mean(rouge_l_scores)
```

#### **4. NIST Score**
```python
def _calculate_nist(self, references: List[List[List[str]]], 
                   predictions: List[List[str]], n: int) -> float:
    """Calculate NIST score."""
    scores = []
    for refs, pred in zip(references, predictions):
        try:
            score = sentence_nist(refs, pred, n)
            scores.append(score)
        except:
            scores.append(0.0)
    
    return np.mean(scores)
```

#### **5. Perplexity**
```python
def _calculate_perplexity(self, predictions: List[str]) -> float:
    """Calculate perplexity (simplified)."""
    # This is a simplified perplexity calculation
    # In practice, you'd use a language model
    total_words = sum(len(word_tokenize(pred)) for pred in predictions)
    return total_words / len(predictions) if predictions else 0.0
```

#### **6. Diversity**
```python
def _calculate_diversity(self, predictions: List[str]) -> float:
    """Calculate diversity (type-token ratio)."""
    all_words = []
    for pred in predictions:
        all_words.extend(word_tokenize(pred.lower()))
    
    if not all_words:
        return 0.0
    
    unique_words = set(all_words)
    return len(unique_words) / len(all_words)
```

#### **7. Coherence**
```python
def _calculate_coherence(self, predictions: List[str]) -> float:
    """Calculate coherence (simplified)."""
    # This is a simplified coherence calculation
    # In practice, you'd use more sophisticated methods
    coherence_scores = []
    for pred in predictions:
        words = word_tokenize(pred.lower())
        if len(words) < 2:
            coherence_scores.append(0.0)
        else:
            # Simple coherence based on word co-occurrence
            coherence = len(set(words)) / len(words)
            coherence_scores.append(coherence)
    
    return np.mean(coherence_scores)
```

## Clustering Metrics

### Core Clustering Metrics

#### **1. Silhouette Score**
```python
def evaluate(self, X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Evaluate clustering metrics."""
    if metric_type == MetricType.SILHOUETTE_SCORE:
        results['silhouette_score'] = silhouette_score(X, labels, metric=self.config.metric)
```

#### **2. Calinski-Harabasz Score**
```python
elif metric_type == MetricType.CALINSKI_HARABASZ:
    results['calinski_harabasz'] = calinski_harabasz_score(X, labels)
```

#### **3. Davies-Bouldin Score**
```python
elif metric_type == MetricType.DAVIES_BOULDIN:
    results['davies_bouldin'] = davies_bouldin_score(X, labels)
```

#### **4. Adjusted Rand Index**
```python
def evaluate_with_true_labels(self, X: np.ndarray, labels: np.ndarray, 
                             true_labels: np.ndarray) -> Dict[str, float]:
    """Evaluate clustering metrics with true labels."""
    if metric_type == MetricType.ADJUSTED_RAND:
        results['adjusted_rand'] = adjusted_rand_score(true_labels, labels)
```

#### **5. Normalized Mutual Information**
```python
elif metric_type == MetricType.NORMALIZED_MUTUAL_INFO:
    results['normalized_mutual_info'] = normalized_mutual_info_score(true_labels, labels)
```

#### **6. Homogeneity**
```python
elif metric_type == MetricType.HOMOGENEITY:
    results['homogeneity'] = homogeneity_score(true_labels, labels)
```

#### **7. Completeness**
```python
elif metric_type == MetricType.COMPLETENESS:
    results['completeness'] = completeness_score(true_labels, labels)
```

#### **8. V-Measure**
```python
elif metric_type == MetricType.V_MEASURE:
    results['v_measure'] = v_measure_score(true_labels, labels)
```

## Evaluation Manager

### Core Evaluation Manager

```python
class EvaluationManager:
    """Comprehensive evaluation manager for different tasks."""
    
    def __init__(self, config: MetricConfig):
        self.config = config
        self.logger = self._setup_logging()
        
        # Initialize metric evaluators
        self.classification_metrics = ClassificationMetrics(config)
        self.regression_metrics = RegressionMetrics(config)
        self.generation_metrics = GenerationMetrics(config)
        self.clustering_metrics = ClusteringMetrics(config)
        
        # Results storage
        self.results = {}
```

### Task-Specific Evaluation

#### **1. Classification Evaluation**
```python
def evaluate_classification(self, y_true: np.ndarray, y_pred: np.ndarray,
                          y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Evaluate classification task."""
    self.logger.info("Evaluating classification task")
    
    results = self.classification_metrics.evaluate(y_true, y_pred, y_prob)
    self.results['classification'] = results
    
    return results
```

#### **2. Regression Evaluation**
```python
def evaluate_regression(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Evaluate regression task."""
    self.logger.info("Evaluating regression task")
    
    results = self.regression_metrics.evaluate(y_true, y_pred)
    self.results['regression'] = results
    
    return results
```

#### **3. Generation Evaluation**
```python
def evaluate_generation(self, references: List[List[str]], 
                       predictions: List[str]) -> Dict[str, float]:
    """Evaluate generation task."""
    self.logger.info("Evaluating generation task")
    
    results = self.generation_metrics.evaluate(references, predictions)
    self.results['generation'] = results
    
    return results
```

#### **4. Clustering Evaluation**
```python
def evaluate_clustering(self, X: np.ndarray, labels: np.ndarray,
                       true_labels: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Evaluate clustering task."""
    self.logger.info("Evaluating clustering task")
    
    if true_labels is not None:
        results = self.clustering_metrics.evaluate_with_true_labels(X, labels, true_labels)
    else:
        results = self.clustering_metrics.evaluate(X, labels)
    
    self.results['clustering'] = results
    
    return results
```

### Unified Evaluation Interface

```python
def evaluate(self, task_type: TaskType, **kwargs) -> Dict[str, float]:
    """Evaluate based on task type."""
    if task_type == TaskType.CLASSIFICATION:
        return self.evaluate_classification(
            kwargs['y_true'], kwargs['y_pred'], kwargs.get('y_prob')
        )
    
    elif task_type == TaskType.REGRESSION:
        return self.evaluate_regression(kwargs['y_true'], kwargs['y_pred'])
    
    elif task_type == TaskType.GENERATION:
        return self.evaluate_generation(
            kwargs['references'], kwargs['predictions']
        )
    
    elif task_type == TaskType.CLUSTERING:
        return self.evaluate_clustering(
            kwargs['X'], kwargs['labels'], kwargs.get('true_labels')
        )
    
    else:
        raise ValueError(f"Unsupported task type: {task_type}")
```

## Configuration Options

### Metric Configuration

#### **1. Task-Specific Settings**
- `task_type`: Type of deep learning task
- `metrics`: List of metrics to compute
- `average`: Averaging method for multi-class metrics
- `zero_division`: Handling of zero division cases

#### **2. Classification Settings**
- `average`: micro, macro, weighted, samples
- `zero_division`: 0, 1, or 'warn'

#### **3. Regression Settings**
- `multioutput`: raw_values, uniform_average, variance_weighted

#### **4. Generation Settings**
- `n_gram`: N-gram size for BLEU and ROUGE
- `smoothing`: Enable smoothing for BLEU
- `smoothing_method`: Smoothing method
- `smoothing_value`: Smoothing value

#### **5. Clustering Settings**
- `metric`: Distance metric for silhouette score

#### **6. Output Settings**
- `save_results`: Save results to file
- `results_file`: Results file path
- `plot_results`: Generate plots
- `plot_file`: Plot file path

## Usage Examples

### Basic Classification Evaluation

```python
# Create configuration
config = MetricConfig(
    task_type=TaskType.CLASSIFICATION,
    metrics=[MetricType.ACCURACY, MetricType.PRECISION, MetricType.RECALL, 
             MetricType.F1_SCORE, MetricType.ROC_AUC],
    average='weighted'
)

# Create evaluation manager
evaluator = EvaluationManager(config)

# Evaluate classification
results = evaluator.evaluate_classification(
    y_true=y_true,
    y_pred=y_pred,
    y_prob=y_prob
)

print("Classification Results:")
for metric_name, value in results.items():
    if isinstance(value, (int, float)):
        print(f"  {metric_name}: {value:.4f}")
```

### Basic Regression Evaluation

```python
# Create configuration
config = MetricConfig(
    task_type=TaskType.REGRESSION,
    metrics=[MetricType.MSE, MetricType.RMSE, MetricType.MAE, MetricType.R2_SCORE]
)

# Create evaluation manager
evaluator = EvaluationManager(config)

# Evaluate regression
results = evaluator.evaluate_regression(y_true=y_true, y_pred=y_pred)

print("Regression Results:")
for metric_name, value in results.items():
    if isinstance(value, (int, float)):
        print(f"  {metric_name}: {value:.4f}")
```

### Basic Generation Evaluation

```python
# Create configuration
config = MetricConfig(
    task_type=TaskType.GENERATION,
    metrics=[MetricType.BLEU, MetricType.METEOR, MetricType.ROUGE],
    n_gram=4,
    smoothing=True
)

# Create evaluation manager
evaluator = EvaluationManager(config)

# Evaluate generation
results = evaluator.evaluate_generation(
    references=references,
    predictions=predictions
)

print("Generation Results:")
for metric_name, value in results.items():
    if isinstance(value, (int, float)):
        print(f"  {metric_name}: {value:.4f}")
```

### Basic Clustering Evaluation

```python
# Create configuration
config = MetricConfig(
    task_type=TaskType.CLUSTERING,
    metrics=[MetricType.SILHOUETTE_SCORE, MetricType.CALINSKI_HARABASZ, 
             MetricType.DAVIES_BOULDIN]
)

# Create evaluation manager
evaluator = EvaluationManager(config)

# Evaluate clustering
results = evaluator.evaluate_clustering(X=X, labels=labels)

print("Clustering Results:")
for metric_name, value in results.items():
    if isinstance(value, (int, float)):
        print(f"  {metric_name}: {value:.4f}")
```

### Unified Evaluation Interface

```python
# Create configuration
config = MetricConfig(
    task_type=TaskType.CLASSIFICATION,
    metrics=[MetricType.ACCURACY, MetricType.F1_SCORE, MetricType.ROC_AUC]
)

# Create evaluation manager
evaluator = EvaluationManager(config)

# Evaluate using unified interface
results = evaluator.evaluate(
    task_type=TaskType.CLASSIFICATION,
    y_true=y_true,
    y_pred=y_pred,
    y_prob=y_prob
)

print("Evaluation Results:")
for metric_name, value in results.items():
    if isinstance(value, (int, float)):
        print(f"  {metric_name}: {value:.4f}")
```

### Advanced Usage with Visualization

```python
# Create comprehensive configuration
config = MetricConfig(
    task_type=TaskType.CLASSIFICATION,
    metrics=[MetricType.ACCURACY, MetricType.PRECISION, MetricType.RECALL, 
             MetricType.F1_SCORE, MetricType.ROC_AUC, MetricType.CONFUSION_MATRIX],
    average='weighted',
    save_results=True,
    plot_results=True
)

# Create evaluation manager
evaluator = EvaluationManager(config)

# Evaluate classification
results = evaluator.evaluate_classification(y_true, y_pred, y_prob)

# Plot confusion matrix
evaluator.classification_metrics.plot_confusion_matrix("confusion_matrix.png")

# Plot ROC curve
evaluator.classification_metrics.plot_roc_curve(y_true, y_prob, "roc_curve.png")

# Plot PR curve
evaluator.classification_metrics.plot_precision_recall_curve(y_true, y_prob, "pr_curve.png")

# Save results
evaluator.save_results("classification_results.json")

# Plot all results
evaluator.plot_results("evaluation_plots.png")
```

## Best Practices

### 1. Metric Selection

#### **Classification Tasks**
- **Binary Classification**: Accuracy, Precision, Recall, F1-Score, ROC AUC, PR AUC
- **Multi-Class Classification**: Accuracy, Macro/Micro F1-Score, ROC AUC (one-vs-rest)
- **Imbalanced Classes**: F1-Score, ROC AUC, PR AUC, Matthews Correlation
- **Probability Output**: Log Loss, ROC AUC, PR AUC

#### **Regression Tasks**
- **General Purpose**: MSE, RMSE, MAE, R² Score
- **Outlier Sensitive**: MAE, Median Absolute Error
- **Percentage Errors**: MAPE, Symmetric MAPE
- **Scale Independent**: R² Score, Explained Variance

#### **Generation Tasks**
- **Machine Translation**: BLEU, METEOR, NIST
- **Text Summarization**: ROUGE, BLEU
- **Text Generation**: BLEU, Perplexity, Diversity, Coherence
- **Dialogue Systems**: BLEU, METEOR, ROUGE

#### **Clustering Tasks**
- **Unsupervised**: Silhouette Score, Calinski-Harabasz, Davies-Bouldin
- **Supervised**: Adjusted Rand, Normalized Mutual Info, Homogeneity, Completeness

### 2. Configuration Guidelines

#### **Classification Configuration**
```python
# Binary classification
config = MetricConfig(
    task_type=TaskType.CLASSIFICATION,
    metrics=[MetricType.ACCURACY, MetricType.PRECISION, MetricType.RECALL, 
             MetricType.F1_SCORE, MetricType.ROC_AUC, MetricType.PR_AUC],
    average='binary'
)

# Multi-class classification
config = MetricConfig(
    task_type=TaskType.CLASSIFICATION,
    metrics=[MetricType.ACCURACY, MetricType.PRECISION, MetricType.RECALL, 
             MetricType.F1_SCORE, MetricType.CONFUSION_MATRIX],
    average='weighted'
)
```

#### **Regression Configuration**
```python
config = MetricConfig(
    task_type=TaskType.REGRESSION,
    metrics=[MetricType.MSE, MetricType.RMSE, MetricType.MAE, 
             MetricType.R2_SCORE, MetricType.EXPLAINED_VARIANCE],
    multioutput='uniform_average'
)
```

#### **Generation Configuration**
```python
config = MetricConfig(
    task_type=TaskType.GENERATION,
    metrics=[MetricType.BLEU, MetricType.METEOR, MetricType.ROUGE, 
             MetricType.NIST, MetricType.PERPLEXITY],
    n_gram=4,
    smoothing=True,
    smoothing_method="add-k",
    smoothing_value=0.1
)
```

#### **Clustering Configuration**
```python
config = MetricConfig(
    task_type=TaskType.CLUSTERING,
    metrics=[MetricType.SILHOUETTE_SCORE, MetricType.CALINSKI_HARABASZ, 
             MetricType.DAVIES_BOULDIN],
    metric='euclidean'
)
```

### 3. Error Handling

#### **Missing Probability Predictions**
```python
# Handle missing probability predictions
if y_prob is not None:
    results['roc_auc'] = roc_auc_score(y_true, y_prob[:, 1])
else:
    self.logger.warning("ROC AUC requires probability predictions")
```

#### **Binary Classification Only Metrics**
```python
# Handle binary classification only metrics
if len(np.unique(y_true)) == 2:
    results['matthews_correlation'] = matthews_corrcoef(y_true, y_pred)
else:
    self.logger.warning("Matthews correlation only for binary classification")
```

#### **True Labels Required**
```python
# Handle metrics requiring true labels
if true_labels is not None:
    results['adjusted_rand'] = adjusted_rand_score(true_labels, labels)
else:
    self.logger.warning("Adjusted Rand Index requires true labels")
```

### 4. Performance Considerations

#### **Large Datasets**
- Use sampling for expensive metrics
- Enable parallel processing where possible
- Cache intermediate results

#### **Memory Management**
- Process data in batches
- Use efficient data structures
- Clean up temporary variables

#### **Computation Optimization**
- Use vectorized operations
- Avoid redundant computations
- Profile performance bottlenecks

## Visualization and Analysis

### 1. Results Visualization

#### **Comprehensive Results Plot**
```python
def plot_results(self, save_path: Optional[str] = None):
    """Plot evaluation results."""
    if not self.results:
        self.logger.warning("No results to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot classification results
    if 'classification' in self.results:
        class_results = self.results['classification']
        metrics = [k for k, v in class_results.items() if isinstance(v, (int, float))]
        values = [class_results[k] for k in metrics]
        
        axes[0, 0].bar(metrics, values)
        axes[0, 0].set_title('Classification Metrics')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Plot regression results
    if 'regression' in self.results:
        reg_results = self.results['regression']
        metrics = [k for k, v in reg_results.items() if isinstance(v, (int, float))]
        values = [reg_results[k] for k in metrics]
        
        axes[0, 1].bar(metrics, values)
        axes[0, 1].set_title('Regression Metrics')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Plot generation results
    if 'generation' in self.results:
        gen_results = self.results['generation']
        metrics = [k for k, v in gen_results.items() if isinstance(v, (int, float))]
        values = [gen_results[k] for k in metrics]
        
        axes[1, 0].bar(metrics, values)
        axes[1, 0].set_title('Generation Metrics')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Plot clustering results
    if 'clustering' in self.results:
        clust_results = self.results['clustering']
        metrics = [k for k, v in clust_results.items() if isinstance(v, (int, float))]
        values = [clust_results[k] for k in metrics]
        
        axes[1, 1].bar(metrics, values)
        axes[1, 1].set_title('Clustering Metrics')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"Results plot saved to {save_path}")
    
    plt.show()
```

### 2. Results Management

#### **Save Results**
```python
def save_results(self, file_path: Optional[str] = None):
    """Save evaluation results."""
    if file_path is None:
        file_path = self.config.results_file
    
    with open(file_path, 'w') as f:
        json.dump(self.results, f, indent=2, default=str)
    
    self.logger.info(f"Results saved to {file_path}")
```

#### **Load Results**
```python
def load_results(self, file_path: str):
    """Load evaluation results."""
    with open(file_path, 'r') as f:
        self.results = json.load(f)
    
    self.logger.info(f"Results loaded from {file_path}")
```

### 3. Custom Metrics

#### **Custom Metric Definition**
```python
def custom_metric(y_true, y_pred, **kwargs):
    """Custom evaluation metric."""
    # Implement custom metric logic
    return custom_score

# Add to configuration
config = MetricConfig(
    task_type=TaskType.CLASSIFICATION,
    metrics=[MetricType.ACCURACY, MetricType.CUSTOM],
    custom_metrics={'custom_metric': custom_metric}
)
```

## Conclusion

The comprehensive evaluation metrics system provides:

1. **Task-Specific Metrics**: Specialized metrics for classification, regression, generation, and clustering
2. **Multiple Metric Types**: Accuracy, precision, recall, F1-score, ROC AUC, MSE, RMSE, MAE, R², BLEU, METEOR, ROUGE, silhouette score, and more
3. **Flexible Configuration**: Extensive configuration options for different use cases
4. **Visualization Tools**: Comprehensive plotting and analysis capabilities
5. **Error Handling**: Robust error handling and logging
6. **Results Management**: Save, load, and analyze evaluation results
7. **Custom Metrics**: Support for custom evaluation metrics
8. **Best Practices**: Guidelines for metric selection and configuration
9. **Performance Optimization**: Efficient computation and memory management
10. **Production Ready**: Comprehensive logging, error handling, and documentation

This system ensures appropriate evaluation metrics for specific tasks, providing comprehensive analysis capabilities for deep learning model evaluation across different domains and applications. 