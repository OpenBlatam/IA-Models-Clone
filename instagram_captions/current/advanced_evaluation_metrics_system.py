import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass
import logging
import time
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    classification_report, roc_auc_score, average_precision_score,
    mean_squared_error, mean_absolute_error, r2_score,
    cohen_kappa_score, matthews_corrcoef, hamming_loss,
    jaccard_score, f1_score, precision_score, recall_score
)
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')


@dataclass
class ClassificationMetricsConfig:
    """Configuration for classification metrics."""
    
    # Basic metrics
    compute_accuracy: bool = True
    compute_precision: bool = True
    compute_recall: bool = True
    compute_f1: bool = True
    
    # Advanced metrics
    compute_roc_auc: bool = True
    compute_pr_auc: bool = True
    compute_confusion_matrix: bool = True
    compute_classification_report: bool = True
    
    # Additional metrics
    compute_cohen_kappa: bool = True
    compute_matthews_corrcoef: bool = True
    compute_hamming_loss: bool = True
    compute_jaccard: bool = True
    
    # Multi-class settings
    average_method: str = "weighted"  # "micro", "macro", "weighted", "samples"
    zero_division: int = 0
    
    # Thresholds
    classification_threshold: float = 0.5
    confidence_threshold: float = 0.8


@dataclass
class RegressionMetricsConfig:
    """Configuration for regression metrics."""
    
    # Basic metrics
    compute_mse: bool = True
    compute_mae: bool = True
    compute_rmse: bool = True
    compute_r2: bool = True
    
    # Additional metrics
    compute_mape: bool = True
    compute_smape: bool = True
    compute_huber_loss: bool = True
    compute_log_cosh_loss: bool = True
    
    # Statistical metrics
    compute_correlation: bool = True
    compute_explained_variance: bool = True
    compute_max_error: bool = True
    
    # Thresholds
    mape_epsilon: float = 1e-8
    huber_delta: float = 1.0


@dataclass
class TextGenerationMetricsConfig:
    """Configuration for text generation metrics."""
    
    # Perplexity
    compute_perplexity: bool = True
    perplexity_model_name: str = "gpt2"
    
    # BLEU score
    compute_bleu: bool = True
    bleu_weights: Tuple[float, ...] = (0.25, 0.25, 0.25, 0.25)
    bleu_smoothing: bool = True
    
    # ROUGE score
    compute_rouge: bool = True
    rouge_metrics: List[str] = None
    
    # METEOR score
    compute_meteor: bool = True
    
    # Custom metrics
    compute_bert_score: bool = True
    compute_semantic_similarity: bool = True
    
    def __post_init__(self):
        if self.rouge_metrics is None:
            self.rouge_metrics = ["rouge1", "rouge2", "rougeL"]


@dataclass
class CustomMetricsConfig:
    """Configuration for custom metrics."""
    
    # Custom metric functions
    custom_metrics: Dict[str, Callable] = None
    
    # Metric parameters
    metric_parameters: Dict[str, Dict[str, Any]] = None
    
    # Weighting
    metric_weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.custom_metrics is None:
            self.custom_metrics = {}
        if self.metric_parameters is None:
            self.metric_parameters = {}
        if self.metric_weights is None:
            self.metric_weights = {}


class AdvancedClassificationMetrics:
    """Advanced classification metrics implementation."""
    
    def __init__(self, config: ClassificationMetricsConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def compute_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Compute comprehensive classification metrics."""
        metrics = {}
        
        # Basic metrics
        if self.config.compute_accuracy:
            metrics["accuracy"] = accuracy_score(y_true, y_pred)
        
        if any([self.config.compute_precision, self.config.compute_recall, self.config.compute_f1]):
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, 
                average=self.config.average_method, 
                zero_division=self.config.zero_division
            )
            
            if self.config.compute_precision:
                metrics["precision"] = precision
            if self.config.compute_recall:
                metrics["recall"] = recall
            if self.config.compute_f1:
                metrics["f1"] = f1
        
        # Advanced metrics
        if self.config.compute_roc_auc and y_proba is not None:
            try:
                if y_proba.ndim == 1:
                    metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
                else:
                    metrics["roc_auc"] = roc_auc_score(y_true, y_proba, multi_class='ovr')
            except Exception as e:
                self.logger.warning(f"Could not compute ROC AUC: {e}")
        
        if self.config.compute_pr_auc and y_proba is not None:
            try:
                if y_proba.ndim == 1:
                    metrics["pr_auc"] = average_precision_score(y_true, y_proba)
                else:
                    metrics["pr_auc"] = average_precision_score(y_true, y_proba, average=self.config.average_method)
            except Exception as e:
                self.logger.warning(f"Could not compute PR AUC: {e}")
        
        # Additional metrics
        if self.config.compute_cohen_kappa:
            try:
                metrics["cohen_kappa"] = cohen_kappa_score(y_true, y_pred)
            except Exception as e:
                self.logger.warning(f"Could not compute Cohen's Kappa: {e}")
        
        if self.config.compute_matthews_corrcoef:
            try:
                metrics["matthews_corrcoef"] = matthews_corrcoef(y_true, y_pred)
            except Exception as e:
                self.logger.warning(f"Could not compute Matthews Correlation: {e}")
        
        if self.config.compute_hamming_loss:
            try:
                metrics["hamming_loss"] = hamming_loss(y_true, y_pred)
            except Exception as e:
                self.logger.warning(f"Could not compute Hamming Loss: {e}")
        
        if self.config.compute_jaccard:
            try:
                metrics["jaccard"] = jaccard_score(y_true, y_pred, average=self.config.average_method)
            except Exception as e:
                self.logger.warning(f"Could not compute Jaccard Score: {e}")
        
        # Detailed reports
        if self.config.compute_confusion_matrix:
            try:
                cm = confusion_matrix(y_true, y_pred)
                metrics["confusion_matrix"] = cm.tolist()
            except Exception as e:
                self.logger.warning(f"Could not compute confusion matrix: {e}")
        
        if self.config.compute_classification_report:
            try:
                report = classification_report(y_true, y_pred, output_dict=True)
                metrics["classification_report"] = report
            except Exception as e:
                self.logger.warning(f"Could not compute classification report: {e}")
        
        return metrics
    
    def compute_per_class_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_proba: Optional[np.ndarray] = None
    ) -> Dict[str, Dict[str, float]]:
        """Compute per-class metrics."""
        per_class_metrics = {}
        
        unique_classes = np.unique(y_true)
        
        for class_idx in unique_classes:
            class_name = f"class_{class_idx}"
            per_class_metrics[class_name] = {}
            
            # Binary classification for this class
            y_true_binary = (y_true == class_idx).astype(int)
            y_pred_binary = (y_pred == class_idx).astype(int)
            
            if y_proba is not None:
                if y_proba.ndim == 1:
                    y_proba_binary = y_proba
                else:
                    y_proba_binary = y_proba[:, class_idx]
            else:
                y_proba_binary = None
            
            # Compute metrics for this class
            class_metrics = self.compute_metrics(y_true_binary, y_pred_binary, y_proba_binary)
            per_class_metrics[class_name] = class_metrics
        
        return per_class_metrics


class AdvancedRegressionMetrics:
    """Advanced regression metrics implementation."""
    
    def __init__(self, config: RegressionMetricsConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute comprehensive regression metrics."""
        metrics = {}
        
        # Basic metrics
        if self.config.compute_mse:
            metrics["mse"] = mean_squared_error(y_true, y_pred)
        
        if self.config.compute_mae:
            metrics["mae"] = mean_absolute_error(y_true, y_pred)
        
        if self.config.compute_rmse:
            metrics["rmse"] = np.sqrt(mean_squared_error(y_true, y_pred))
        
        if self.config.compute_r2:
            metrics["r2"] = r2_score(y_true, y_pred)
        
        # Additional metrics
        if self.config.compute_mape:
            mape = self._compute_mape(y_true, y_pred)
            metrics["mape"] = mape
        
        if self.config.compute_smape:
            smape = self._compute_smape(y_true, y_pred)
            metrics["smape"] = smape
        
        if self.config.compute_huber_loss:
            huber_loss = self._compute_huber_loss(y_true, y_pred)
            metrics["huber_loss"] = huber_loss
        
        if self.config.compute_log_cosh_loss:
            log_cosh_loss = self._compute_log_cosh_loss(y_true, y_pred)
            metrics["log_cosh_loss"] = log_cosh_loss
        
        # Statistical metrics
        if self.config.compute_correlation:
            correlation = np.corrcoef(y_true, y_pred)[0, 1]
            metrics["correlation"] = correlation if not np.isnan(correlation) else 0.0
        
        if self.config.compute_explained_variance:
            explained_variance = 1 - np.var(y_true - y_pred) / np.var(y_true)
            metrics["explained_variance"] = explained_variance
        
        if self.config.compute_max_error:
            max_error = np.max(np.abs(y_true - y_pred))
            metrics["max_error"] = max_error
        
        return metrics
    
    def _compute_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute Mean Absolute Percentage Error."""
        epsilon = self.config.mape_epsilon
        mape = np.mean(np.abs((y_true - y_pred) / np.where(np.abs(y_true) > epsilon, y_true, epsilon))) * 100
        return mape
    
    def _compute_smape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute Symmetric Mean Absolute Percentage Error."""
        epsilon = self.config.mape_epsilon
        smape = 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + epsilon))
        return smape
    
    def _compute_huber_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute Huber Loss."""
        delta = self.config.huber_delta
        error = y_true - y_pred
        abs_error = np.abs(error)
        quadratic = np.minimum(abs_error, delta)
        linear = abs_error - quadratic
        huber_loss = np.mean(0.5 * quadratic**2 + delta * linear)
        return huber_loss
    
    def _compute_log_cosh_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute Log-Cosh Loss."""
        error = y_true - y_pred
        log_cosh_loss = np.mean(np.log(np.cosh(error)))
        return log_cosh_loss


class AdvancedTextGenerationMetrics:
    """Advanced text generation metrics implementation."""
    
    def __init__(self, config: TextGenerationMetricsConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def compute_metrics(
        self, 
        references: List[List[str]], 
        predictions: List[str]
    ) -> Dict[str, float]:
        """Compute comprehensive text generation metrics."""
        metrics = {}
        
        # Perplexity
        if self.config.compute_perplexity:
            perplexity = self._compute_perplexity(predictions)
            metrics["perplexity"] = perplexity
        
        # BLEU score
        if self.config.compute_bleu:
            bleu_score = self._compute_bleu(references, predictions)
            metrics["bleu"] = bleu_score
        
        # ROUGE score
        if self.config.compute_rouge:
            rouge_scores = self._compute_rouge(references, predictions)
            metrics.update(rouge_scores)
        
        # METEOR score
        if self.config.compute_meteor:
            meteor_score = self._compute_meteor(references, predictions)
            metrics["meteor"] = meteor_score
        
        # BERT Score
        if self.config.compute_bert_score:
            bert_score = self._compute_bert_score(references, predictions)
            metrics["bert_score"] = bert_score
        
        # Semantic similarity
        if self.config.compute_semantic_similarity:
            semantic_sim = self._compute_semantic_similarity(references, predictions)
            metrics["semantic_similarity"] = semantic_sim
        
        return metrics
    
    def _compute_perplexity(self, texts: List[str]) -> float:
        """Compute perplexity using a language model."""
        try:
            # This is a placeholder implementation
            # In practice, you would use a language model like GPT-2
            return 0.0
        except Exception as e:
            self.logger.warning(f"Could not compute perplexity: {e}")
            return 0.0
    
    def _compute_bleu(self, references: List[List[str]], predictions: List[str]) -> float:
        """Compute BLEU score."""
        try:
            # This is a placeholder implementation
            # In practice, you would use nltk.translate.bleu_score
            return 0.0
        except Exception as e:
            self.logger.warning(f"Could not compute BLEU score: {e}")
            return 0.0
    
    def _compute_rouge(self, references: List[List[str]], predictions: List[str]) -> Dict[str, float]:
        """Compute ROUGE scores."""
        try:
            # This is a placeholder implementation
            # In practice, you would use rouge-score library
            rouge_scores = {}
            for metric in self.config.rouge_metrics:
                rouge_scores[metric] = 0.0
            return rouge_scores
        except Exception as e:
            self.logger.warning(f"Could not compute ROUGE scores: {e}")
            return {metric: 0.0 for metric in self.config.rouge_metrics}
    
    def _compute_meteor(self, references: List[List[str]], predictions: List[str]) -> float:
        """Compute METEOR score."""
        try:
            # This is a placeholder implementation
            # In practice, you would use nltk.translate.meteor_score
            return 0.0
        except Exception as e:
            self.logger.warning(f"Could not compute METEOR score: {e}")
            return 0.0
    
    def _compute_bert_score(self, references: List[List[str]], predictions: List[str]) -> float:
        """Compute BERT Score."""
        try:
            # This is a placeholder implementation
            # In practice, you would use bert-score library
            return 0.0
        except Exception as e:
            self.logger.warning(f"Could not compute BERT Score: {e}")
            return 0.0
    
    def _compute_semantic_similarity(self, references: List[List[str]], predictions: List[str]) -> float:
        """Compute semantic similarity."""
        try:
            # This is a placeholder implementation
            # In practice, you would use sentence-transformers
            return 0.0
        except Exception as e:
            self.logger.warning(f"Could not compute semantic similarity: {e}")
            return 0.0


class CustomMetricsEvaluator:
    """Custom metrics evaluator."""
    
    def __init__(self, config: CustomMetricsConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def compute_custom_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        **kwargs
    ) -> Dict[str, float]:
        """Compute custom metrics."""
        metrics = {}
        
        for metric_name, metric_func in self.config.custom_metrics.items():
            try:
                # Get parameters for this metric
                params = self.config.metric_parameters.get(metric_name, {})
                
                # Compute metric
                metric_value = metric_func(y_true, y_pred, **params, **kwargs)
                metrics[metric_name] = metric_value
                
            except Exception as e:
                self.logger.warning(f"Could not compute custom metric {metric_name}: {e}")
                metrics[metric_name] = 0.0
        
        return metrics
    
    def compute_weighted_score(self, metrics: Dict[str, float]) -> float:
        """Compute weighted score from multiple metrics."""
        if not self.config.metric_weights:
            return np.mean(list(metrics.values()))
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for metric_name, metric_value in metrics.items():
            weight = self.config.metric_weights.get(metric_name, 1.0)
            weighted_sum += metric_value * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0


class ComprehensiveEvaluationSystem:
    """Comprehensive evaluation system for all task types."""
    
    def __init__(
        self,
        classification_config: Optional[ClassificationMetricsConfig] = None,
        regression_config: Optional[RegressionMetricsConfig] = None,
        text_generation_config: Optional[TextGenerationMetricsConfig] = None,
        custom_config: Optional[CustomMetricsConfig] = None
    ):
        self.classification_config = classification_config or ClassificationMetricsConfig()
        self.regression_config = regression_config or RegressionMetricsConfig()
        self.text_generation_config = text_generation_config or TextGenerationMetricsConfig()
        self.custom_config = custom_config or CustomMetricsConfig()
        
        self.classification_evaluator = AdvancedClassificationMetrics(self.classification_config)
        self.regression_evaluator = AdvancedRegressionMetrics(self.regression_config)
        self.text_generation_evaluator = AdvancedTextGenerationMetrics(self.text_generation_config)
        self.custom_evaluator = CustomMetricsEvaluator(self.custom_config)
        
        self.logger = logging.getLogger(__name__)
    
    def evaluate_classification(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_proba: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Evaluate classification task."""
        # Standard metrics
        metrics = self.classification_evaluator.compute_metrics(y_true, y_pred, y_proba)
        
        # Per-class metrics
        per_class_metrics = self.classification_evaluator.compute_per_class_metrics(y_true, y_pred, y_proba)
        metrics["per_class"] = per_class_metrics
        
        # Custom metrics
        custom_metrics = self.custom_evaluator.compute_custom_metrics(y_true, y_pred)
        metrics["custom"] = custom_metrics
        
        # Weighted score
        all_metrics = {**metrics, **custom_metrics}
        metrics["weighted_score"] = self.custom_evaluator.compute_weighted_score(all_metrics)
        
        return metrics
    
    def evaluate_regression(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """Evaluate regression task."""
        # Standard metrics
        metrics = self.regression_evaluator.compute_metrics(y_true, y_pred)
        
        # Custom metrics
        custom_metrics = self.custom_evaluator.compute_custom_metrics(y_true, y_pred)
        metrics["custom"] = custom_metrics
        
        # Weighted score
        all_metrics = {**metrics, **custom_metrics}
        metrics["weighted_score"] = self.custom_evaluator.compute_weighted_score(all_metrics)
        
        return metrics
    
    def evaluate_text_generation(
        self, 
        references: List[List[str]], 
        predictions: List[str]
    ) -> Dict[str, Any]:
        """Evaluate text generation task."""
        # Standard metrics
        metrics = self.text_generation_evaluator.compute_metrics(references, predictions)
        
        # Custom metrics (if applicable)
        # For text generation, custom metrics might need different handling
        metrics["custom"] = {}
        
        # Weighted score
        all_metrics = {**metrics, **metrics["custom"]}
        metrics["weighted_score"] = self.custom_evaluator.compute_weighted_score(all_metrics)
        
        return metrics
    
    def evaluate_multi_task(
        self, 
        task_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Evaluate multiple tasks and compute overall performance."""
        overall_metrics = {
            "task_metrics": task_results,
            "overall_score": 0.0,
            "task_weights": {}
        }
        
        # Compute overall score (simple average for now)
        total_score = 0.0
        task_count = 0
        
        for task_name, task_metrics in task_results.items():
            if "weighted_score" in task_metrics:
                total_score += task_metrics["weighted_score"]
                task_count += 1
        
        if task_count > 0:
            overall_metrics["overall_score"] = total_score / task_count
        
        return overall_metrics


# Example custom metrics
def custom_f1_score(y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
    """Custom F1 score implementation."""
    return f1_score(y_true, y_pred, average='weighted', zero_division=0)


def custom_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
    """Custom balanced accuracy implementation."""
    from sklearn.metrics import balanced_accuracy_score
    return balanced_accuracy_score(y_true, y_pred)


def custom_top_k_accuracy(y_true: np.ndarray, y_pred_proba: np.ndarray, k: int = 3, **kwargs) -> float:
    """Custom top-k accuracy implementation."""
    if y_pred_proba.ndim == 1:
        return 0.0
    
    top_k_indices = np.argsort(y_pred_proba, axis=1)[:, -k:]
    correct = 0
    total = len(y_true)
    
    for i, true_label in enumerate(y_true):
        if true_label in top_k_indices[i]:
            correct += 1
    
    return correct / total


# Example usage
def create_comprehensive_evaluation_example():
    """Example of using the comprehensive evaluation system."""
    
    # Custom metrics configuration
    custom_config = CustomMetricsConfig(
        custom_metrics={
            "custom_f1": custom_f1_score,
            "balanced_accuracy": custom_balanced_accuracy,
            "top_3_accuracy": lambda y_true, y_pred_proba: custom_top_k_accuracy(y_true, y_pred_proba, k=3)
        },
        metric_weights={
            "accuracy": 0.3,
            "f1": 0.3,
            "roc_auc": 0.2,
            "custom_f1": 0.2
        }
    )
    
    # Create evaluation system
    evaluation_system = ComprehensiveEvaluationSystem(
        custom_config=custom_config
    )
    
    return evaluation_system


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create example
    evaluation_system = create_comprehensive_evaluation_example()
    print("Comprehensive evaluation system created successfully!")
    
    # Example usage
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 0, 1, 1, 1])
    y_proba = np.array([0.1, 0.9, 0.2, 0.4, 0.1, 0.8, 0.7, 0.9])
    
    metrics = evaluation_system.evaluate_classification(y_true, y_pred, y_proba)
    print(f"Classification metrics: {metrics}")




