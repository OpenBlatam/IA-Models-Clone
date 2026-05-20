from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import time
import logging
from typing import Dict, Any, Optional, List, Union, Tuple, Callable, TypeVar
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
import numpy as np
import pandas as pd
from functional_utils import (
        from sklearn.metrics import accuracy_score
        from sklearn.metrics import precision_score
        from sklearn.metrics import recall_score
        from sklearn.metrics import f1_score
        from sklearn.metrics import cohen_kappa_score
        from sklearn.metrics import matthews_corrcoef
        from sklearn.metrics import roc_auc_score
        from sklearn.metrics import log_loss
        from sklearn.metrics import mean_squared_error
        from sklearn.metrics import mean_squared_error
        from sklearn.metrics import mean_absolute_error
        from sklearn.metrics import r2_score
        from sklearn.metrics import hamming_loss
        from sklearn.metrics import accuracy_score
        from sklearn.metrics import precision_score
        from sklearn.metrics import recall_score
        from sklearn.metrics import f1_score
        from sklearn.metrics import precision_score
        from sklearn.metrics import recall_score
        from sklearn.metrics import f1_score
        from sklearn.metrics import jaccard_score
        from sklearn.metrics import jaccard_score
        from sklearn.metrics import confusion_matrix, classification_report, roc_curve, precision_recall_curve
from typing import Any, List, Dict, Optional
import asyncio
"""
🔧 Modular Evaluation Framework
===============================

Modular evaluation framework using iteration and modularization.
Eliminates code duplication through reusable, composable evaluation functions.

Key Principles:
- Iteration over duplication
- Modularization over repetition
- Composition over inheritance
- Pure functions with no side effects
- Immutable data transformations
"""


    Result, ValidationResult, safe_execute, timer_context,
    transform_list, filter_list, group_by, sort_by,
    pipe, compose, log_function_call, log_data_info
)

# Set up logging
logger = logging.getLogger(__name__)

# ============================================================================
# Generic Types
# ============================================================================

T = TypeVar('T')
U = TypeVar('U')

# ============================================================================
# Core Data Structures
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
# Metric Calculation Modules
# ============================================================================

class MetricCalculator:
    """Modular metric calculator using iteration and composition."""
    
    @staticmethod
    def calculate_classification_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        config: MetricConfig = None
    ) -> Dict[str, float]:
        """Calculate classification metrics using modular approach."""
        if config is None:
            config = MetricConfig(task_type=TaskType.CLASSIFICATION, metric_type=MetricType.ACCURACY)
        
        # Define metric calculations as functions
        metric_functions = {
            'accuracy': lambda: MetricCalculator._calculate_accuracy(y_true, y_pred),
            'precision': lambda: MetricCalculator._calculate_precision(y_true, y_pred, config),
            'recall': lambda: MetricCalculator._calculate_recall(y_true, y_pred, config),
            'f1': lambda: MetricCalculator._calculate_f1(y_true, y_pred, config),
            'cohen_kappa': lambda: MetricCalculator._calculate_cohen_kappa(y_true, y_pred),
            'matthews_corrcoef': lambda: MetricCalculator._calculate_matthews_corrcoef(y_true, y_pred)
        }
        
        # Add probability-based metrics if available
        if y_prob is not None:
            metric_functions.update({
                'roc_auc': lambda: MetricCalculator._calculate_roc_auc(y_true, y_prob, config),
                'log_loss': lambda: MetricCalculator._calculate_log_loss(y_true, y_prob)
            })
        
        # Calculate all metrics using iteration
        calculated_metrics = {}
        for metric_name, calc_func in metric_functions.items():
            result = safe_execute(calc_func)
            if result.is_successful:
                calculated_metrics[metric_name] = result.value
            else:
                logger.warning(f"Could not calculate {metric_name}: {result.error}")
        
        return metrics
    
    @staticmethod
    def calculate_regression_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        config: MetricConfig = None
    ) -> Dict[str, float]:
        """Calculate regression metrics using modular approach."""
        if config is None:
            config = MetricConfig(task_type=TaskType.REGRESSION, metric_type=MetricType.MSE)
        
        # Define metric calculations as functions
        metric_functions = {
            'mse': lambda: MetricCalculator._calculate_mse(y_true, y_pred),
            'rmse': lambda: MetricCalculator._calculate_rmse(y_true, y_pred),
            'mae': lambda: MetricCalculator._calculate_mae(y_true, y_pred),
            'r2': lambda: MetricCalculator._calculate_r2(y_true, y_pred),
            'mape': lambda: MetricCalculator._calculate_mape(y_true, y_pred),
            'smape': lambda: MetricCalculator._calculate_smape(y_true, y_pred)
        }
        
        # Calculate all metrics using iteration
        metrics = {}
        for metric_name, calc_func in metric_functions.items():
            result = safe_execute(calc_func)
            if result.success:
                metrics[metric_name] = result.value
            else:
                logger.warning(f"Could not calculate {metric_name}: {result.error}")
        
        return metrics
    
    @staticmethod
    def calculate_multilabel_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        config: MetricConfig = None
    ) -> Dict[str, float]:
        """Calculate multilabel metrics using modular approach."""
        if config is None:
            config = MetricConfig(task_type=TaskType.MULTILABEL, metric_type=MetricType.F1)
        
        # Define metric calculations as functions
        metric_functions = {
            'hamming_loss': lambda: MetricCalculator._calculate_hamming_loss(y_true, y_pred),
            'exact_match': lambda: MetricCalculator._calculate_exact_match(y_true, y_pred),
            'micro_precision': lambda: MetricCalculator._calculate_micro_precision(y_true, y_pred, config),
            'micro_recall': lambda: MetricCalculator._calculate_micro_recall(y_true, y_pred, config),
            'micro_f1': lambda: MetricCalculator._calculate_micro_f1(y_true, y_pred, config),
            'macro_precision': lambda: MetricCalculator._calculate_macro_precision(y_true, y_pred, config),
            'macro_recall': lambda: MetricCalculator._calculate_macro_recall(y_true, y_pred, config),
            'macro_f1': lambda: MetricCalculator._calculate_macro_f1(y_true, y_pred, config),
            'jaccard_micro': lambda: MetricCalculator._calculate_jaccard_micro(y_true, y_pred),
            'jaccard_macro': lambda: MetricCalculator._calculate_jaccard_macro(y_true, y_pred)
        }
        
        # Calculate all metrics using iteration
        metrics = {}
        for metric_name, calc_func in metric_functions.items():
            result = safe_execute(calc_func)
            if result.success:
                metrics[metric_name] = result.value
            else:
                logger.warning(f"Could not calculate {metric_name}: {result.error}")
        
        return metrics
    
    # Individual metric calculation methods
    @staticmethod
    def _calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return accuracy_score(y_true, y_pred)
    
    @staticmethod
    def _calculate_precision(y_true: np.ndarray, y_pred: np.ndarray, config: MetricConfig) -> float:
        return precision_score(y_true, y_pred, average=config.average, zero_division=config.zero_division)
    
    @staticmethod
    def _calculate_recall(y_true: np.ndarray, y_pred: np.ndarray, config: MetricConfig) -> float:
        return recall_score(y_true, y_pred, average=config.average, zero_division=config.zero_division)
    
    @staticmethod
    def _calculate_f1(y_true: np.ndarray, y_pred: np.ndarray, config: MetricConfig) -> float:
        return f1_score(y_true, y_pred, average=config.average, zero_division=config.zero_division)
    
    @staticmethod
    def _calculate_cohen_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return cohen_kappa_score(y_true, y_pred)
    
    @staticmethod
    def _calculate_matthews_corrcoef(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return matthews_corrcoef(y_true, y_pred)
    
    @staticmethod
    def _calculate_roc_auc(y_true: np.ndarray, y_prob: np.ndarray, config: MetricConfig) -> float:
        if len(np.unique(y_true)) == 2:
            return roc_auc_score(y_true, y_prob[:, 1])
        else:
            return roc_auc_score(y_true, y_prob, multi_class='ovr', average=config.average)
    
    @staticmethod
    def _calculate_log_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        return log_loss(y_true, y_prob)
    
    @staticmethod
    def _calculate_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return mean_squared_error(y_true, y_pred)
    
    @staticmethod
    def _calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    @staticmethod
    def _calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return mean_absolute_error(y_true, y_pred)
    
    @staticmethod
    def _calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return r2_score(y_true, y_pred)
    
    @staticmethod
    def _calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, 1))) * 100
    
    @staticmethod
    def _calculate_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
    
    @staticmethod
    def _calculate_hamming_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return hamming_loss(y_true, y_pred)
    
    @staticmethod
    def _calculate_exact_match(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return accuracy_score(y_true, y_pred)
    
    @staticmethod
    def _calculate_micro_precision(y_true: np.ndarray, y_pred: np.ndarray, config: MetricConfig) -> float:
        return precision_score(y_true, y_pred, average='micro', zero_division=config.zero_division)
    
    @staticmethod
    def _calculate_micro_recall(y_true: np.ndarray, y_pred: np.ndarray, config: MetricConfig) -> float:
        return recall_score(y_true, y_pred, average='micro', zero_division=config.zero_division)
    
    @staticmethod
    def _calculate_micro_f1(y_true: np.ndarray, y_pred: np.ndarray, config: MetricConfig) -> float:
        return f1_score(y_true, y_pred, average='micro', zero_division=config.zero_division)
    
    @staticmethod
    def _calculate_macro_precision(y_true: np.ndarray, y_pred: np.ndarray, config: MetricConfig) -> float:
        return precision_score(y_true, y_pred, average='macro', zero_division=config.zero_division)
    
    @staticmethod
    def _calculate_macro_recall(y_true: np.ndarray, y_pred: np.ndarray, config: MetricConfig) -> float:
        return recall_score(y_true, y_pred, average='macro', zero_division=config.zero_division)
    
    @staticmethod
    def _calculate_macro_f1(y_true: np.ndarray, y_pred: np.ndarray, config: MetricConfig) -> float:
        return f1_score(y_true, y_pred, average='macro', zero_division=config.zero_division)
    
    @staticmethod
    def _calculate_jaccard_micro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return jaccard_score(y_true, y_pred, average='micro')
    
    @staticmethod
    def _calculate_jaccard_macro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return jaccard_score(y_true, y_pred, average='macro')

# ============================================================================
# Evaluation Pipeline Modules
# ============================================================================

class EvaluationPipeline:
    """Modular evaluation pipeline using composition and iteration."""
    
    @staticmethod
    @log_function_call
    def evaluate_model(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        task_type: TaskType = TaskType.CLASSIFICATION,
        config: MetricConfig = None
    ) -> EvaluationResult:
        """Evaluate model using modular pipeline."""
        with timer_context(f"Model evaluation for {task_type.value}"):
            # Calculate metrics based on task type
            metrics = EvaluationPipeline._calculate_metrics_by_task(
                y_true, y_pred, y_prob, task_type, config
            )
            
            # Calculate additional data based on task type
            additional_data = EvaluationPipeline._calculate_additional_data(
                y_true, y_pred, y_prob, task_type
            )
            
            # Create evaluation result
            return EvaluationResult(
                task_type=task_type,
                metrics=metrics,
                **additional_data,
                inference_time_ms=0.0,  # Will be set by caller
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
            )
    
    @staticmethod
    def _calculate_metrics_by_task(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray],
        task_type: TaskType,
        config: MetricConfig
    ) -> Dict[str, float]:
        """Calculate metrics based on task type using modular approach."""
        task_metric_calculators = {
            TaskType.CLASSIFICATION: MetricCalculator.calculate_classification_metrics,
            TaskType.REGRESSION: MetricCalculator.calculate_regression_metrics,
            TaskType.MULTILABEL: MetricCalculator.calculate_multilabel_metrics,
        }
        
        if task_type not in task_metric_calculators:
            raise ValueError(f"Unsupported task type: {task_type}")
        
        return task_metric_calculators[task_type](y_true, y_pred, y_prob, config)
    
    @staticmethod
    def _calculate_additional_data(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray],
        task_type: TaskType
    ) -> Dict[str, Any]:
        """Calculate additional data based on task type."""
        additional_data = {
            'confusion_matrix': None,
            'classification_report': None,
            'roc_curve_data': None,
            'precision_recall_curve_data': None
        }
        
        if task_type == TaskType.CLASSIFICATION:
            additional_data.update(EvaluationPipeline._calculate_classification_data(y_true, y_pred, y_prob))
        
        return additional_data
    
    @staticmethod
    def _calculate_classification_data(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray]
    ) -> Dict[str, Any]:
        """Calculate classification-specific data."""
        
        data = {
            'confusion_matrix': confusion_matrix(y_true, y_pred),
            'classification_report': classification_report(y_true, y_pred, output_dict=False)
        }
        
        # Calculate ROC curve data
        if y_prob is not None and len(np.unique(y_true)) == 2:
            try:
                fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
                data['roc_curve_data'] = {'fpr': fpr, 'tpr': tpr}
            except Exception as e:
                logger.warning(f"Could not calculate ROC curve: {e}")
        
        # Calculate Precision-Recall curve data
        if y_prob is not None and len(np.unique(y_true)) == 2:
            try:
                precision, recall, _ = precision_recall_curve(y_true, y_prob[:, 1])
                data['precision_recall_curve_data'] = {'precision': precision, 'recall': recall}
            except Exception as e:
                logger.warning(f"Could not calculate PR curve: {e}")
        
        return data

# ============================================================================
# Model Comparison Modules
# ============================================================================

class ModelComparisonPipeline:
    """Modular model comparison pipeline using iteration and composition."""
    
    @staticmethod
    def compare_models(
        model_results: Dict[str, EvaluationResult],
        metric_name: str = "f1"
    ) -> ModelComparison:
        """Compare multiple models using modular approach."""
        if not model_results:
            raise ValueError("No model results provided")
        
        # Extract model names and metrics
        model_names = list(model_results.keys())
        comparison_metrics = ModelComparisonPipeline._extract_comparison_metrics(model_results)
        
        # Find best model
        best_model = ModelComparisonPipeline._find_best_model(comparison_metrics, metric_name, model_names)
        
        # Calculate improvement scores
        improvement_scores = ModelComparisonPipeline._calculate_improvement_scores(
            comparison_metrics, metric_name, model_names
        )
        
        return ModelComparison(
            model_names=model_names,
            comparison_metrics=comparison_metrics,
            best_model=best_model,
            improvement_scores=improvement_scores
        )
    
    @staticmethod
    def _extract_comparison_metrics(model_results: Dict[str, EvaluationResult]) -> Dict[str, List[float]]:
        """Extract metrics for comparison using iteration."""
        if not model_results:
            return {}
        
        # Get all available metrics from the first model
        first_model = next(iter(model_results.values()))
        available_metrics = first_model.metrics.keys()
        
        # Extract metrics for each model
        comparison_metrics = {}
        for metric in available_metrics:
            comparison_metrics[metric] = [
                model_results[name].metrics.get(metric, 0.0) 
                for name in model_results.keys()
            ]
        
        return comparison_metrics
    
    @staticmethod
    def _find_best_model(
        comparison_metrics: Dict[str, List[float]],
        metric_name: str,
        model_names: List[str]
    ) -> str:
        """Find best model based on specified metric."""
        if metric_name in comparison_metrics:
            best_idx = np.argmax(comparison_metrics[metric_name])
            return model_names[best_idx]
        else:
            return model_names[0]
    
    @staticmethod
    def _calculate_improvement_scores(
        comparison_metrics: Dict[str, List[float]],
        metric_name: str,
        model_names: List[str]
    ) -> Dict[str, float]:
        """Calculate improvement scores using iteration."""
        improvement_scores = {}
        
        if metric_name in comparison_metrics:
            scores = comparison_metrics[metric_name]
            best_score = max(scores)
            
            for i, name in enumerate(model_names):
                current_score = scores[i]
                improvement = ((current_score - best_score) / best_score) * 100 if best_score != 0 else 0
                improvement_scores[name] = improvement
        
        return improvement_scores
    
    @staticmethod
    def rank_models(
        model_results: Dict[str, EvaluationResult],
        metric_name: str = "f1"
    ) -> List[Tuple[str, float]]:
        """Rank models by metric using modular approach."""
        # Extract scores
        scores = [
            (name, result.metrics.get(metric_name, 0.0))
            for name, result in model_results.items()
        ]
        
        # Sort by score (descending)
        return sort_by(scores, key_fn=lambda x: x[1], reverse=True)

# ============================================================================
# Quick Evaluation Functions (Composed from modules)
# ============================================================================

def quick_evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Quick classification evaluation using modular approach."""
    config = MetricConfig(task_type=TaskType.CLASSIFICATION, metric_type=MetricType.F1)
    return MetricCalculator.calculate_classification_metrics(y_true, y_pred, y_prob, config)

def quick_evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """Quick regression evaluation using modular approach."""
    config = MetricConfig(task_type=TaskType.REGRESSION, metric_type=MetricType.MSE)
    return MetricCalculator.calculate_regression_metrics(y_true, y_pred, config)

def quick_evaluate_multilabel(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Quick multilabel evaluation using modular approach."""
    config = MetricConfig(task_type=TaskType.MULTILABEL, metric_type=MetricType.F1)
    return MetricCalculator.calculate_multilabel_metrics(y_true, y_pred, y_prob, config)

# ============================================================================
# Batch Evaluation Functions
# ============================================================================

def batch_evaluate_models(
    model_predictions: Dict[str, Dict[str, np.ndarray]],
    task_type: TaskType = TaskType.CLASSIFICATION,
    config: MetricConfig = None
) -> Dict[str, EvaluationResult]:
    """Evaluate multiple models in batch using modular approach."""
    results = {}
    
    for model_name, predictions in model_predictions.items():
        y_true = predictions.get('y_true')
        y_pred = predictions.get('y_pred')
        y_prob = predictions.get('y_prob')
        
        if y_true is not None and y_pred is not None:
            result = EvaluationPipeline.evaluate_model(
                y_true, y_pred, y_prob, task_type, config
            )
            results[model_name] = result
    
    return results

def evaluate_model_ensemble(
    model_predictions: List[Dict[str, np.ndarray]],
    ensemble_method: str = "voting",
    task_type: TaskType = TaskType.CLASSIFICATION,
    config: MetricConfig = None
) -> EvaluationResult:
    """Evaluate model ensemble using modular approach."""
    if not model_predictions:
        raise ValueError("No model predictions provided")
    
    # Combine predictions based on ensemble method
    if ensemble_method == "voting":
        y_pred = _combine_voting_predictions(model_predictions)
    elif ensemble_method == "averaging":
        y_pred = _combine_averaging_predictions(model_predictions)
    else:
        raise ValueError(f"Unknown ensemble method: {ensemble_method}")
    
    # Use predictions from first model for ground truth
    y_true = model_predictions[0].get('y_true')
    y_prob = model_predictions[0].get('y_prob')
    
    return EvaluationPipeline.evaluate_model(y_true, y_pred, y_prob, task_type, config)

def _combine_voting_predictions(model_predictions: List[Dict[str, np.ndarray]]) -> np.ndarray:
    """Combine predictions using voting."""
    predictions = [pred['y_pred'] for pred in model_predictions]
    return np.mean(predictions, axis=0).round().astype(int)

def _combine_averaging_predictions(model_predictions: List[Dict[str, np.ndarray]]) -> np.ndarray:
    """Combine predictions using averaging."""
    predictions = [pred['y_pred'] for pred in model_predictions]
    return np.mean(predictions, axis=0)

# ============================================================================
# Demo Functions
# ============================================================================

def demo_modular_evaluation():
    """Demo the modular evaluation framework."""
    print("🔧 Modular Evaluation Demo")
    print("=" * 50)
    
    # Generate test data
    np.random.seed(42)
    y_true = np.random.randint(0, 3, 100)
    y_pred = np.random.randint(0, 3, 100)
    y_prob = np.random.rand(100, 3)
    
    # Test classification evaluation
    print("Classification Evaluation:")
    class_metrics = quick_evaluate_classification(y_true, y_pred, y_prob)
    print(f"  Metrics: {list(class_metrics.keys())}")
    
    # Test regression evaluation
    print("\nRegression Evaluation:")
    y_true_reg = np.random.randn(100)
    y_pred_reg = y_true_reg + np.random.randn(100) * 0.1
    reg_metrics = quick_evaluate_regression(y_true_reg, y_pred_reg)
    print(f"  Metrics: {list(reg_metrics.keys())}")
    
    # Test model comparison
    print("\nModel Comparison:")
    model_results = {
        'Model A': EvaluationPipeline.evaluate_model(y_true, y_pred, y_prob, TaskType.CLASSIFICATION),
        'Model B': EvaluationPipeline.evaluate_model(y_true, y_pred + 1, y_prob, TaskType.CLASSIFICATION)
    }
    
    comparison = ModelComparisonPipeline.compare_models(model_results, 'f1')
    print(f"  Best model: {comparison.best_model}")
    print(f"  Improvement scores: {comparison.improvement_scores}")
    
    # Test batch evaluation
    print("\nBatch Evaluation:")
    model_predictions = {
        'Model A': {'y_true': y_true, 'y_pred': y_pred, 'y_prob': y_prob},
        'Model B': {'y_true': y_true, 'y_pred': y_pred + 1, 'y_prob': y_prob}
    }
    
    batch_results = batch_evaluate_models(model_predictions, TaskType.CLASSIFICATION)
    print(f"  Evaluated {len(batch_results)} models")
    
    print("\n🎉 All modular evaluation functions working correctly!")

match __name__:
    case "__main__":
    demo_modular_evaluation() 