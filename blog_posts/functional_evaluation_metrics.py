from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from functools import partial, reduce
from operator import itemgetter
import logging
import time
from pathlib import Path
import json
import warnings
from sklearn.metrics import (
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from typing import Any, List, Dict, Optional
import asyncio
"""
🚀 Functional Evaluation Metrics System
======================================

Pure functional, declarative approach to evaluation metrics.
Uses data transformations, pure functions, and functional patterns instead of classes.

Key Principles:
- Pure functions with no side effects
- Data transformations over mutable state
- Composition over inheritance
- Immutable data structures
- Declarative configuration
"""


    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, classification_report, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score,
    log_loss, cohen_kappa_score, matthews_corrcoef,
    hamming_loss, jaccard_score, f1_score,
    precision_score, recall_score, roc_curve, precision_recall_curve
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Pure Data Structures
# ============================================================================

class TaskType(Enum):
    """Available task types."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    GENERATION = "generation"
    MULTILABEL = "multilabel"
    MULTICLASS = "multiclass"
    BINARY = "binary"

class MetricType(Enum):
    """Available metric types."""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1 = "f1"
    ROC_AUC = "roc_auc"
    MSE = "mse"
    MAE = "mae"
    R2 = "r2"
    CUSTOM = "custom"

@dataclass(frozen=True)
class MetricConfig:
    """Immutable metric configuration."""
    task_type: TaskType
    metric_type: MetricType
    average: str = "weighted"
    zero_division: int = 0
    sample_weight: Optional[np.ndarray] = None
    custom_metric: Optional[Callable] = None

@dataclass(frozen=True)
class EvaluationResult:
    """Immutable evaluation result."""
    task_type: TaskType
    metrics: Dict[str, float]
    confusion_matrix: Optional[np.ndarray] = None
    classification_report: Optional[str] = None
    roc_curve_data: Optional[Dict[str, np.ndarray]] = None
    precision_recall_curve_data: Optional[Dict[str, np.ndarray]] = None
    inference_time_ms: float = 0.0
    timestamp: str = ""

@dataclass(frozen=True)
class ModelComparison:
    """Immutable model comparison result."""
    model_names: List[str]
    comparison_metrics: Dict[str, List[float]]
    best_model: str
    improvement_scores: Dict[str, float]

# ============================================================================
# Pure Functions - Metric Calculations
# ============================================================================

def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    config: MetricConfig = None
) -> Dict[str, float]:
    """Calculate classification metrics in a pure functional way."""
    if config is None:
        config = MetricConfig(task_type=TaskType.CLASSIFICATION, metric_type=MetricType.ACCURACY)
    
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average=config.average, zero_division=config.zero_division)
    metrics['recall'] = recall_score(y_true, y_pred, average=config.average, zero_division=config.zero_division)
    metrics['f1'] = f1_score(y_true, y_pred, average=config.average, zero_division=config.zero_division)
    
    # Additional metrics
    metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
    metrics['matthews_corrcoef'] = matthews_corrcoef(y_true, y_pred)
    
    # ROC AUC if probabilities available
    if y_prob is not None:
        try:
            if len(np.unique(y_true)) == 2:
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob[:, 1])
            else:
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average=config.average)
        except Exception as e:
            logger.warning(f"Could not calculate ROC AUC: {e}")
    
    # Log loss if probabilities available
    if y_prob is not None:
        try:
            metrics['log_loss'] = log_loss(y_true, y_prob)
        except Exception as e:
            logger.warning(f"Could not calculate log loss: {e}")
    
    return metrics

def calculate_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    config: MetricConfig = None
) -> Dict[str, float]:
    """Calculate regression metrics in a pure functional way."""
    if config is None:
        config = MetricConfig(task_type=TaskType.REGRESSION, metric_type=MetricType.MSE)
    
    metrics = {}
    
    # Basic regression metrics
    metrics['mse'] = mean_squared_error(y_true, y_pred)
    metrics['rmse'] = np.sqrt(metrics['mse'])
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    metrics['r2'] = r2_score(y_true, y_pred)
    
    # Additional metrics
    metrics['mape'] = np.mean(np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, 1))) * 100
    metrics['smape'] = 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
    
    return metrics

def calculate_multilabel_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    config: MetricConfig = None
) -> Dict[str, float]:
    """Calculate multilabel metrics in a pure functional way."""
    if config is None:
        config = MetricConfig(task_type=TaskType.MULTILABEL, metric_type=MetricType.F1)
    
    metrics = {}
    
    # Multilabel specific metrics
    metrics['hamming_loss'] = hamming_loss(y_true, y_pred)
    metrics['exact_match'] = accuracy_score(y_true, y_pred)
    
    # Per-label metrics
    metrics['micro_precision'] = precision_score(y_true, y_pred, average='micro', zero_division=config.zero_division)
    metrics['micro_recall'] = recall_score(y_true, y_pred, average='micro', zero_division=config.zero_division)
    metrics['micro_f1'] = f1_score(y_true, y_pred, average='micro', zero_division=config.zero_division)
    
    metrics['macro_precision'] = precision_score(y_true, y_pred, average='macro', zero_division=config.zero_division)
    metrics['macro_recall'] = recall_score(y_true, y_pred, average='macro', zero_division=config.zero_division)
    metrics['macro_f1'] = f1_score(y_true, y_pred, average='macro', zero_division=config.zero_division)
    
    # Jaccard score
    metrics['jaccard_micro'] = jaccard_score(y_true, y_pred, average='micro')
    metrics['jaccard_macro'] = jaccard_score(y_true, y_pred, average='macro')
    
    return metrics

def calculate_generation_metrics(
    predictions: List[str],
    references: List[str],
    config: MetricConfig = None
) -> Dict[str, float]:
    """Calculate generation metrics in a pure functional way."""
    if config is None:
        config = MetricConfig(task_type=TaskType.GENERATION, metric_type=MetricType.CUSTOM)
    
    metrics = {}
    
    # BLEU score (simplified)
    try:
        smoothie = SmoothingFunction().method1
        
        bleu_scores = []
        for pred, ref in zip(predictions, references):
            pred_tokens = pred.split()
            ref_tokens = ref.split()
            bleu_scores.append(sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothie))
        
        metrics['bleu'] = np.mean(bleu_scores)
    except ImportError:
        logger.warning("NLTK not available, skipping BLEU calculation")
    
    # Exact match
    exact_matches = sum(1 for pred, ref in zip(predictions, references) if pred.strip() == ref.strip())
    metrics['exact_match'] = exact_matches / len(predictions) if predictions else 0
    
    # Length ratio
    pred_lengths = [len(pred.split()) for pred in predictions]
    ref_lengths = [len(ref.split()) for ref in references]
    metrics['length_ratio'] = np.mean(pred_lengths) / np.mean(ref_lengths) if ref_lengths else 0
    
    return metrics

# ============================================================================
# Pure Functions - Evaluation Pipeline
# ============================================================================

def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    task_type: TaskType = TaskType.CLASSIFICATION,
    config: MetricConfig = None
) -> EvaluationResult:
    """Evaluate model in a pure functional way."""
    start_time = time.time()
    
    # Calculate metrics based on task type
    if task_type == TaskType.CLASSIFICATION:
        metrics = calculate_classification_metrics(y_true, y_pred, y_prob, config)
        confusion_mat = confusion_matrix(y_true, y_pred)
        class_report = classification_report(y_true, y_pred, output_dict=False)
        
        # ROC curve data
        roc_data = None
        if y_prob is not None and len(np.unique(y_true)) == 2:
            try:
                fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
                roc_data = {'fpr': fpr, 'tpr': tpr}
            except Exception as e:
                logger.warning(f"Could not calculate ROC curve: {e}")
        
        # Precision-Recall curve data
        pr_data = None
        if y_prob is not None and len(np.unique(y_true)) == 2:
            try:
                precision, recall, _ = precision_recall_curve(y_true, y_prob[:, 1])
                pr_data = {'precision': precision, 'recall': recall}
            except Exception as e:
                logger.warning(f"Could not calculate PR curve: {e}")
        
    elif task_type == TaskType.REGRESSION:
        metrics = calculate_regression_metrics(y_true, y_pred, config)
        confusion_mat = None
        class_report = None
        roc_data = None
        pr_data = None
        
    elif task_type == TaskType.MULTILABEL:
        metrics = calculate_multilabel_metrics(y_true, y_pred, y_prob, config)
        confusion_mat = None
        class_report = None
        roc_data = None
        pr_data = None
        
    else:
        raise ValueError(f"Unsupported task type: {task_type}")
    
    inference_time = (time.time() - start_time) * 1000  # Convert to milliseconds
    
    return EvaluationResult(
        task_type=task_type,
        metrics=metrics,
        confusion_matrix=confusion_mat,
        classification_report=class_report,
        roc_curve_data=roc_data,
        precision_recall_curve_data=pr_data,
        inference_time_ms=inference_time,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
    )

def evaluate_generation_model(
    predictions: List[str],
    references: List[str],
    config: MetricConfig = None
) -> EvaluationResult:
    """Evaluate generation model in a pure functional way."""
    start_time = time.time()
    
    metrics = calculate_generation_metrics(predictions, references, config)
    
    inference_time = (time.time() - start_time) * 1000
    
    return EvaluationResult(
        task_type=TaskType.GENERATION,
        metrics=metrics,
        confusion_matrix=None,
        classification_report=None,
        roc_curve_data=None,
        precision_recall_curve_data=None,
        inference_time_ms=inference_time,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
    )

# ============================================================================
# Pure Functions - Model Comparison
# ============================================================================

def compare_models(
    model_results: Dict[str, EvaluationResult],
    metric_name: str = "f1"
) -> ModelComparison:
    """Compare multiple models in a pure functional way."""
    if not model_results:
        raise ValueError("No model results provided")
    
    model_names = list(model_results.keys())
    comparison_metrics = {}
    
    # Extract metrics for comparison
    for metric in model_results[model_names[0]].metrics.keys():
        comparison_metrics[metric] = [
            model_results[name].metrics.get(metric, 0.0) 
            for name in model_names
        ]
    
    # Find best model based on specified metric
    if metric_name in comparison_metrics:
        best_idx = np.argmax(comparison_metrics[metric_name])
        best_model = model_names[best_idx]
    else:
        best_model = model_names[0]
    
    # Calculate improvement scores
    improvement_scores = {}
    if metric_name in comparison_metrics:
        best_score = max(comparison_metrics[metric_name])
        for i, name in enumerate(model_names):
            current_score = comparison_metrics[metric_name][i]
            improvement = ((current_score - best_score) / best_score) * 100 if best_score != 0 else 0
            improvement_scores[name] = improvement
    
    return ModelComparison(
        model_names=model_names,
        comparison_metrics=comparison_metrics,
        best_model=best_model,
        improvement_scores=improvement_scores
    )

def rank_models(
    model_results: Dict[str, EvaluationResult],
    metric_name: str = "f1"
) -> List[Tuple[str, float]]:
    """Rank models by metric in a pure functional way."""
    rankings = []
    
    for model_name, result in model_results.items():
        score = result.metrics.get(metric_name, 0.0)
        rankings.append((model_name, score))
    
    # Sort by score (descending)
    rankings.sort(key=lambda x: x[1], reverse=True)
    
    return rankings

# ============================================================================
# Pure Functions - Visualization
# ============================================================================

def create_confusion_matrix_plot(
    confusion_matrix: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """Create confusion matrix plot in a pure functional way."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return {
        'figure_size': (10, 8),
        'save_path': save_path,
        'class_names': class_names
    }

def create_roc_curve_plot(
    roc_data: Dict[str, np.ndarray],
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """Create ROC curve plot in a pure functional way."""
    plt.figure(figsize=(8, 6))
    plt.plot(roc_data['fpr'], roc_data['tpr'], linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return {
        'figure_size': (8, 6),
        'save_path': save_path,
        'auc': np.trapz(roc_data['tpr'], roc_data['fpr'])
    }

def create_metrics_comparison_plot(
    comparison: ModelComparison,
    metric_name: str = "f1",
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """Create metrics comparison plot in a pure functional way."""
    plt.figure(figsize=(12, 6))
    
    metric_values = comparison.comparison_metrics.get(metric_name, [])
    plt.bar(comparison.model_names, metric_values)
    plt.title(f'{metric_name.upper()} Comparison')
    plt.ylabel(metric_name.upper())
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return {
        'figure_size': (12, 6),
        'save_path': save_path,
        'metric_name': metric_name,
        'best_model': comparison.best_model
    }

# ============================================================================
# Pure Functions - Results Export
# ============================================================================

def export_evaluation_results(
    results: EvaluationResult,
    output_path: str,
    include_plots: bool = True
) -> Dict[str, str]:
    """Export evaluation results in a pure functional way."""
    output_dir = Path(output_path)
    output_dir.mkdir(exist_ok=True)
    
    exported_files = {}
    
    # Export metrics to JSON
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        json.dump({
            'task_type': results.task_type.value,
            'metrics': results.metrics,
            'inference_time_ms': results.inference_time_ms,
            'timestamp': results.timestamp
        }, f, indent=2)
    exported_files['metrics'] = str(metrics_path)
    
    # Export confusion matrix
    if results.confusion_matrix is not None:
        cm_path = output_dir / "confusion_matrix.npy"
        np.save(cm_path, results.confusion_matrix)
        exported_files['confusion_matrix'] = str(cm_path)
        
        if include_plots:
            cm_plot_path = output_dir / "confusion_matrix.png"
            create_confusion_matrix_plot(results.confusion_matrix, save_path=str(cm_plot_path))
            exported_files['confusion_matrix_plot'] = str(cm_plot_path)
    
    # Export classification report
    if results.classification_report is not None:
        report_path = output_dir / "classification_report.txt"
        with open(report_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write(results.classification_report)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        exported_files['classification_report'] = str(report_path)
    
    # Export ROC curve data
    if results.roc_curve_data is not None:
        roc_path = output_dir / "roc_curve_data.npz"
        np.savez(roc_path, **results.roc_curve_data)
        exported_files['roc_curve_data'] = str(roc_path)
        
        if include_plots:
            roc_plot_path = output_dir / "roc_curve.png"
            create_roc_curve_plot(results.roc_curve_data, save_path=str(roc_plot_path))
            exported_files['roc_curve_plot'] = str(roc_plot_path)
    
    return exported_files

def export_model_comparison(
    comparison: ModelComparison,
    output_path: str,
    include_plots: bool = True
) -> Dict[str, str]:
    """Export model comparison in a pure functional way."""
    output_dir = Path(output_path)
    output_dir.mkdir(exist_ok=True)
    
    exported_files = {}
    
    # Export comparison data to JSON
    comparison_path = output_dir / "model_comparison.json"
    with open(comparison_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        json.dump({
            'model_names': comparison.model_names,
            'comparison_metrics': comparison.comparison_metrics,
            'best_model': comparison.best_model,
            'improvement_scores': comparison.improvement_scores
        }, f, indent=2)
    exported_files['comparison_data'] = str(comparison_path)
    
    # Export comparison plots
    if include_plots:
        for metric_name in comparison.comparison_metrics.keys():
            plot_path = output_dir / f"{metric_name}_comparison.png"
            create_metrics_comparison_plot(comparison, metric_name, save_path=str(plot_path))
            exported_files[f'{metric_name}_plot'] = str(plot_path)
    
    return exported_files

# ============================================================================
# Pure Functions - Quick Evaluation
# ============================================================================

def quick_evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Quick classification evaluation in a pure functional way."""
    config = MetricConfig(task_type=TaskType.CLASSIFICATION, metric_type=MetricType.F1)
    result = evaluate_model(y_true, y_pred, y_prob, TaskType.CLASSIFICATION, config)
    return result.metrics

def quick_evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """Quick regression evaluation in a pure functional way."""
    config = MetricConfig(task_type=TaskType.REGRESSION, metric_type=MetricType.MSE)
    result = evaluate_model(y_true, y_pred, None, TaskType.REGRESSION, config)
    return result.metrics

def quick_evaluate_generation(
    predictions: List[str],
    references: List[str]
) -> Dict[str, float]:
    """Quick generation evaluation in a pure functional way."""
    result = evaluate_generation_model(predictions, references)
    return result.metrics

# ============================================================================
# Demo Functions
# ============================================================================

def demo_evaluation_metrics():
    """Demo the functional evaluation metrics system."""
    print("🚀 Functional Evaluation Metrics Demo")
    print("=" * 50)
    
    # Generate sample data
    np.random.seed(42)
    y_true = np.random.randint(0, 3, 1000)
    y_pred = np.random.randint(0, 3, 1000)
    y_prob = np.random.rand(1000, 3)
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
    
    # Classification evaluation
    print("Classification Evaluation:")
    class_result = evaluate_model(y_true, y_pred, y_prob, TaskType.CLASSIFICATION)
    for metric, value in class_result.metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Regression evaluation
    print("\nRegression Evaluation:")
    y_true_reg = np.random.randn(1000)
    y_pred_reg = y_true_reg + np.random.randn(1000) * 0.1
    reg_result = evaluate_model(y_true_reg, y_pred_reg, None, TaskType.REGRESSION)
    for metric, value in reg_result.metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Model comparison
    print("\nModel Comparison:")
    model_results = {
        'Model A': class_result,
        'Model B': EvaluationResult(
            task_type=TaskType.CLASSIFICATION,
            metrics={'accuracy': 0.85, 'f1': 0.83, 'precision': 0.84, 'recall': 0.82},
            confusion_matrix=None,
            classification_report=None,
            roc_curve_data=None,
            precision_recall_curve_data=None,
            inference_time_ms=50.0,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
    }
    
    comparison = compare_models(model_results, 'f1')
    print(f"Best model: {comparison.best_model}")
    print(f"Improvement scores: {comparison.improvement_scores}")
    
    # Quick evaluation
    print("\nQuick Evaluation:")
    quick_metrics = quick_evaluate_classification(y_true, y_pred, y_prob)
    print(f"Quick F1 score: {quick_metrics['f1']:.4f}")

match __name__:
    case "__main__":
    demo_evaluation_metrics() 