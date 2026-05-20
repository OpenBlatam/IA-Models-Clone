from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import asyncio
import time
import logging
import json
import math
import warnings
from typing import Dict, Any, Optional, List, Union, Callable, Tuple, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
from collections import defaultdict, Counter
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import (
from sklearn.preprocessing import label_binarize
from sklearn.utils.multiclass import type_of_target
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from .production_transformers import DeviceManager
from typing import Any, List, Dict, Optional
"""
🎯 Evaluation Metrics System - Production Ready
==============================================

Enterprise-grade evaluation metrics system with task-specific metrics for
classification, regression, generation, and other AI tasks.
"""


    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, mean_squared_error, mean_absolute_error,
    r2_score, explained_variance_score, max_error,
    mean_poisson_deviance, mean_gamma_deviance, mean_tweedie_deviance,
    d2_tweedie_score, mean_absolute_percentage_error,
    mean_squared_log_error, median_absolute_error, mean_pinball_loss,
    jaccard_score, hamming_loss, zero_one_loss, hinge_loss,
    log_loss, matthews_corrcoef, cohen_kappa_score,
    balanced_accuracy_score, top_k_accuracy_score
)

# Import our production engines

logger = logging.getLogger(__name__)

class TaskType(Enum):
    """Supported task types."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    GENERATION = "generation"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"
    QUESTION_ANSWERING = "question_answering"
    NAMED_ENTITY_RECOGNITION = "ner"
    SENTIMENT_ANALYSIS = "sentiment"
    TEXT_CLASSIFICATION = "text_classification"
    IMAGE_CLASSIFICATION = "image_classification"
    OBJECT_DETECTION = "object_detection"
    SEGMENTATION = "segmentation"
    MULTI_LABEL = "multi_label"
    RANKING = "ranking"
    RECOMMENDATION = "recommendation"
    ANOMALY_DETECTION = "anomaly_detection"
    CLUSTERING = "clustering"
    CUSTOM = "custom"

class MetricType(Enum):
    """Metric types."""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1 = "f1"
    ROC_AUC = "roc_auc"
    PR_AUC = "pr_auc"
    MSE = "mse"
    MAE = "mae"
    R2 = "r2"
    EXPLAINED_VARIANCE = "explained_variance"
    MAX_ERROR = "max_error"
    MEAN_POISSON_DEVANCE = "mean_poisson_deviance"
    MEAN_GAMMA_DEVANCE = "mean_gamma_deviance"
    MEAN_TWEEDIE_DEVANCE = "mean_tweedie_deviance"
    D2_TWEEDIE = "d2_tweedie"
    MAPE = "mape"
    MSLE = "msle"
    MEDIAN_AE = "median_ae"
    MEAN_PINBALL_LOSS = "mean_pinball_loss"
    JACCARD = "jaccard"
    HAMMING_LOSS = "hamming_loss"
    ZERO_ONE_LOSS = "zero_one_loss"
    HINGE_LOSS = "hinge_loss"
    LOG_LOSS = "log_loss"
    MATTHEWS_CORR = "matthews_corr"
    COHEN_KAPPA = "cohen_kappa"
    BALANCED_ACCURACY = "balanced_accuracy"
    TOP_K_ACCURACY = "top_k_accuracy"
    BLEU = "bleu"
    ROUGE = "rouge"
    METEOR = "meteor"
    CIDEr = "cider"
    BERT_SCORE = "bert_score"
    PERPLEXITY = "perplexity"
    BLEURT = "bleurt"
    COMET = "comet"
    CUSTOM = "custom"

@dataclass
class MetricConfig:
    """Configuration for evaluation metrics."""
    task_type: TaskType
    metric_types: List[MetricType] = field(default_factory=list)
    average: str = "weighted"  # For multi-class: 'micro', 'macro', 'weighted', 'samples'
    beta: float = 1.0  # For F-beta score
    k: int = 5  # For top-k accuracy
    threshold: float = 0.5  # For binary classification
    custom_metrics: Dict[str, Callable] = field(default_factory=dict)
    
    # Generation metrics
    bleu_weights: Tuple[float, ...] = (0.25, 0.25, 0.25, 0.25)
    rouge_metrics: List[str] = field(default_factory=lambda: ['rouge1', 'rouge2', 'rougeL'])
    meteor_alpha: float = 0.9
    meteor_beta: float = 3.0
    
    # Regression metrics
    multioutput: str = "uniform_average"
    sample_weight: Optional[np.ndarray] = None
    
    def __post_init__(self) -> Any:
        """Set default metrics based on task type."""
        if not self.metric_types:
            self.metric_types = self._get_default_metrics()
    
    def _get_default_metrics(self) -> List[MetricType]:
        """Get default metrics for task type."""
        defaults = {
            TaskType.CLASSIFICATION: [
                MetricType.ACCURACY, MetricType.PRECISION, MetricType.RECALL,
                MetricType.F1, MetricType.ROC_AUC, MetricType.CONFUSION_MATRIX
            ],
            TaskType.REGRESSION: [
                MetricType.MSE, MetricType.MAE, MetricType.R2,
                MetricType.EXPLAINED_VARIANCE, MetricType.MAX_ERROR
            ],
            TaskType.GENERATION: [
                MetricType.BLEU, MetricType.ROUGE, MetricType.METEOR,
                MetricType.PERPLEXITY
            ],
            TaskType.TRANSLATION: [
                MetricType.BLEU, MetricType.ROUGE, MetricType.METEOR,
                MetricType.BLEURT, MetricType.COMET
            ],
            TaskType.SUMMARIZATION: [
                MetricType.ROUGE, MetricType.BLEU, MetricType.METEOR,
                MetricType.CIDEr
            ],
            TaskType.QUESTION_ANSWERING: [
                MetricType.ACCURACY, MetricType.F1, MetricType.EM
            ],
            TaskType.NAMED_ENTITY_RECOGNITION: [
                MetricType.PRECISION, MetricType.RECALL, MetricType.F1,
                MetricType.ACCURACY
            ],
            TaskType.SENTIMENT_ANALYSIS: [
                MetricType.ACCURACY, MetricType.PRECISION, MetricType.RECALL,
                MetricType.F1, MetricType.ROC_AUC
            ],
            TaskType.TEXT_CLASSIFICATION: [
                MetricType.ACCURACY, MetricType.PRECISION, MetricType.RECALL,
                MetricType.F1, MetricType.ROC_AUC
            ],
            TaskType.IMAGE_CLASSIFICATION: [
                MetricType.ACCURACY, MetricType.TOP_K_ACCURACY,
                MetricType.PRECISION, MetricType.RECALL, MetricType.F1
            ],
            TaskType.OBJECT_DETECTION: [
                MetricType.MAP, MetricType.PRECISION, MetricType.RECALL,
                MetricType.IOU
            ],
            TaskType.SEGMENTATION: [
                MetricType.IOU, MetricType.DICE, MetricType.PRECISION,
                MetricType.RECALL
            ],
            TaskType.MULTI_LABEL: [
                MetricType.HAMMING_LOSS, MetricType.JACCARD,
                MetricType.PRECISION, MetricType.RECALL, MetricType.F1
            ],
            TaskType.RANKING: [
                MetricType.NDCG, MetricType.MAP, MetricType.MRR,
                MetricType.PRECISION_AT_K
            ],
            TaskType.RECOMMENDATION: [
                MetricType.PRECISION_AT_K, MetricType.RECALL_AT_K,
                MetricType.NDCG, MetricType.MAP
            ],
            TaskType.ANOMALY_DETECTION: [
                MetricType.ROC_AUC, MetricType.PR_AUC, MetricType.F1,
                MetricType.PRECISION, MetricType.RECALL
            ],
            TaskType.CLUSTERING: [
                MetricType.SILHOUETTE_SCORE, MetricType.CALINSKI_HARABASZ,
                MetricType.DAVIES_BOULDIN
            ]
        }
        return defaults.get(self.task_type, [MetricType.ACCURACY])

@dataclass
class EvaluationResult:
    """Evaluation results."""
    task_type: TaskType
    metrics: Dict[str, float] = field(default_factory=dict)
    confusion_matrix: Optional[np.ndarray] = None
    classification_report: Optional[str] = None
    detailed_metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'task_type': self.task_type.value,
            'metrics': self.metrics,
            'confusion_matrix': self.confusion_matrix.tolist() if self.confusion_matrix is not None else None,
            'classification_report': self.classification_report,
            'detailed_metrics': self.detailed_metrics,
            'metadata': self.metadata
        }
    
    def save(self, filepath: str):
        """Save results to file."""
        with open(filepath, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(self.to_dict(), f, indent=2, default=str)
    
    @classmethod
    def load(cls, filepath: str) -> 'EvaluationResult':
        """Load results from file."""
        with open(filepath, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            data = json.load(f)
        
        return cls(
            task_type=TaskType(data['task_type']),
            metrics=data['metrics'],
            confusion_matrix=np.array(data['confusion_matrix']) if data['confusion_matrix'] else None,
            classification_report=data['classification_report'],
            detailed_metrics=data['detailed_metrics'],
            metadata=data['metadata']
        )

class EvaluationMetrics:
    """Production-ready evaluation metrics system."""
    
    def __init__(self, device_manager: DeviceManager):
        
    """__init__ function."""
self.device_manager = device_manager
        self.logger = logging.getLogger(f"{__name__}.EvaluationMetrics")
        
        # Initialize metric calculators
        self._init_metric_calculators()
    
    def _init_metric_calculators(self) -> Any:
        """Initialize metric calculators."""
        self.metric_calculators = {
            # Classification metrics
            MetricType.ACCURACY: self._calculate_accuracy,
            MetricType.PRECISION: self._calculate_precision,
            MetricType.RECALL: self._calculate_recall,
            MetricType.F1: self._calculate_f1,
            MetricType.ROC_AUC: self._calculate_roc_auc,
            MetricType.PR_AUC: self._calculate_pr_auc,
            MetricType.CONFUSION_MATRIX: self._calculate_confusion_matrix,
            MetricType.CLASSIFICATION_REPORT: self._calculate_classification_report,
            MetricType.JACCARD: self._calculate_jaccard,
            MetricType.HAMMING_LOSS: self._calculate_hamming_loss,
            MetricType.ZERO_ONE_LOSS: self._calculate_zero_one_loss,
            MetricType.HINGE_LOSS: self._calculate_hinge_loss,
            MetricType.LOG_LOSS: self._calculate_log_loss,
            MetricType.MATTHEWS_CORR: self._calculate_matthews_corr,
            MetricType.COHEN_KAPPA: self._calculate_cohen_kappa,
            MetricType.BALANCED_ACCURACY: self._calculate_balanced_accuracy,
            MetricType.TOP_K_ACCURACY: self._calculate_top_k_accuracy,
            
            # Regression metrics
            MetricType.MSE: self._calculate_mse,
            MetricType.MAE: self._calculate_mae,
            MetricType.R2: self._calculate_r2,
            MetricType.EXPLAINED_VARIANCE: self._calculate_explained_variance,
            MetricType.MAX_ERROR: self._calculate_max_error,
            MetricType.MEAN_POISSON_DEVANCE: self._calculate_mean_poisson_deviance,
            MetricType.MEAN_GAMMA_DEVANCE: self._calculate_mean_gamma_deviance,
            MetricType.MEAN_TWEEDIE_DEVANCE: self._calculate_mean_tweedie_deviance,
            MetricType.D2_TWEEDIE: self._calculate_d2_tweedie,
            MetricType.MAPE: self._calculate_mape,
            MetricType.MSLE: self._calculate_msle,
            MetricType.MEDIAN_AE: self._calculate_median_ae,
            MetricType.MEAN_PINBALL_LOSS: self._calculate_mean_pinball_loss,
            
            # Generation metrics
            MetricType.BLEU: self._calculate_bleu,
            MetricType.ROUGE: self._calculate_rouge,
            MetricType.METEOR: self._calculate_meteor,
            MetricType.CIDEr: self._calculate_cider,
            MetricType.BERT_SCORE: self._calculate_bert_score,
            MetricType.PERPLEXITY: self._calculate_perplexity,
            MetricType.BLEURT: self._calculate_bleurt,
            MetricType.COMET: self._calculate_comet,
        }
    
    def evaluate(self, config: MetricConfig, y_true: np.ndarray, y_pred: np.ndarray,
                 y_prob: Optional[np.ndarray] = None, sample_weight: Optional[np.ndarray] = None) -> EvaluationResult:
        """Evaluate predictions using specified metrics."""
        self.logger.info(f"Evaluating {config.task_type.value} task with {len(config.metric_types)} metrics")
        
        result = EvaluationResult(task_type=config.task_type)
        
        # Calculate metrics
        for metric_type in config.metric_types:
            try:
                if metric_type in self.metric_calculators:
                    metric_value = self.metric_calculators[metric_type](
                        y_true, y_pred, y_prob, config, sample_weight
                    )
                    result.metrics[metric_type.value] = metric_value
                elif metric_type == MetricType.CUSTOM:
                    # Handle custom metrics
                    for name, func in config.custom_metrics.items():
                        metric_value = func(y_true, y_pred, y_prob)
                        result.metrics[name] = metric_value
                else:
                    self.logger.warning(f"Metric {metric_type.value} not implemented")
            except Exception as e:
                self.logger.error(f"Error calculating {metric_type.value}: {e}")
                result.metrics[metric_type.value] = None
        
        # Add metadata
        result.metadata = {
            'num_samples': len(y_true),
            'num_classes': len(np.unique(y_true)) if config.task_type == TaskType.CLASSIFICATION else None,
            'evaluation_time': time.time(),
            'config': {
                'average': config.average,
                'beta': config.beta,
                'k': config.k,
                'threshold': config.threshold
            }
        }
        
        return result
    
    # Classification metrics
    def _calculate_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray, 
                           y_prob: Optional[np.ndarray], config: MetricConfig,
                           sample_weight: Optional[np.ndarray]) -> float:
        """Calculate accuracy."""
        return accuracy_score(y_true, y_pred, sample_weight=sample_weight)
    
    def _calculate_precision(self, y_true: np.ndarray, y_pred: np.ndarray,
                            y_prob: Optional[np.ndarray], config: MetricConfig,
                            sample_weight: Optional[np.ndarray]) -> float:
        """Calculate precision."""
        return precision_score(y_true, y_pred, average=config.average, 
                              sample_weight=sample_weight, zero_division=0)
    
    def _calculate_recall(self, y_true: np.ndarray, y_pred: np.ndarray,
                         y_prob: Optional[np.ndarray], config: MetricConfig,
                         sample_weight: Optional[np.ndarray]) -> float:
        """Calculate recall."""
        return recall_score(y_true, y_pred, average=config.average,
                           sample_weight=sample_weight, zero_division=0)
    
    def _calculate_f1(self, y_true: np.ndarray, y_pred: np.ndarray,
                     y_prob: Optional[np.ndarray], config: MetricConfig,
                     sample_weight: Optional[np.ndarray]) -> float:
        """Calculate F1 score."""
        return f1_score(y_true, y_pred, average=config.average,
                       sample_weight=sample_weight, zero_division=0)
    
    def _calculate_roc_auc(self, y_true: np.ndarray, y_pred: np.ndarray,
                          y_prob: Optional[np.ndarray], config: MetricConfig,
                          sample_weight: Optional[np.ndarray]) -> float:
        """Calculate ROC AUC."""
        if y_prob is None:
            return None
        
        if len(np.unique(y_true)) == 2:
            return roc_auc_score(y_true, y_prob[:, 1] if y_prob.ndim > 1 else y_prob,
                                sample_weight=sample_weight)
        else:
            return roc_auc_score(y_true, y_prob, multi_class='ovr',
                                average=config.average, sample_weight=sample_weight)
    
    def _calculate_pr_auc(self, y_true: np.ndarray, y_pred: np.ndarray,
                         y_prob: Optional[np.ndarray], config: MetricConfig,
                         sample_weight: Optional[np.ndarray]) -> float:
        """Calculate PR AUC."""
        if y_prob is None:
            return None
        
        if len(np.unique(y_true)) == 2:
            return average_precision_score(y_true, y_prob[:, 1] if y_prob.ndim > 1 else y_prob,
                                         sample_weight=sample_weight)
        else:
            # Multi-class PR AUC
            y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
            return average_precision_score(y_true_bin, y_prob, average=config.average,
                                         sample_weight=sample_weight)
    
    def _calculate_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                                   y_prob: Optional[np.ndarray], config: MetricConfig,
                                   sample_weight: Optional[np.ndarray]) -> np.ndarray:
        """Calculate confusion matrix."""
        return confusion_matrix(y_true, y_pred, sample_weight=sample_weight)
    
    def _calculate_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                                        y_prob: Optional[np.ndarray], config: MetricConfig,
                                        sample_weight: Optional[np.ndarray]) -> str:
        """Calculate classification report."""
        return classification_report(y_true, y_pred, sample_weight=sample_weight)
    
    def _calculate_jaccard(self, y_true: np.ndarray, y_pred: np.ndarray,
                          y_prob: Optional[np.ndarray], config: MetricConfig,
                          sample_weight: Optional[np.ndarray]) -> float:
        """Calculate Jaccard score."""
        return jaccard_score(y_true, y_pred, average=config.average,
                            sample_weight=sample_weight, zero_division=0)
    
    def _calculate_hamming_loss(self, y_true: np.ndarray, y_pred: np.ndarray,
                               y_prob: Optional[np.ndarray], config: MetricConfig,
                               sample_weight: Optional[np.ndarray]) -> float:
        """Calculate Hamming loss."""
        return hamming_loss(y_true, y_pred, sample_weight=sample_weight)
    
    def _calculate_zero_one_loss(self, y_true: np.ndarray, y_pred: np.ndarray,
                                y_prob: Optional[np.ndarray], config: MetricConfig,
                                sample_weight: Optional[np.ndarray]) -> float:
        """Calculate zero-one loss."""
        return zero_one_loss(y_true, y_pred, sample_weight=sample_weight)
    
    def _calculate_hinge_loss(self, y_true: np.ndarray, y_pred: np.ndarray,
                             y_prob: Optional[np.ndarray], config: MetricConfig,
                             sample_weight: Optional[np.ndarray]) -> float:
        """Calculate hinge loss."""
        return hinge_loss(y_true, y_pred, sample_weight=sample_weight)
    
    def _calculate_log_loss(self, y_true: np.ndarray, y_pred: np.ndarray,
                           y_prob: Optional[np.ndarray], config: MetricConfig,
                           sample_weight: Optional[np.ndarray]) -> float:
        """Calculate log loss."""
        if y_prob is None:
            return None
        return log_loss(y_true, y_prob, sample_weight=sample_weight)
    
    def _calculate_matthews_corr(self, y_true: np.ndarray, y_pred: np.ndarray,
                                y_prob: Optional[np.ndarray], config: MetricConfig,
                                sample_weight: Optional[np.ndarray]) -> float:
        """Calculate Matthews correlation coefficient."""
        return matthews_corrcoef(y_true, y_pred, sample_weight=sample_weight)
    
    def _calculate_cohen_kappa(self, y_true: np.ndarray, y_pred: np.ndarray,
                              y_prob: Optional[np.ndarray], config: MetricConfig,
                              sample_weight: Optional[np.ndarray]) -> float:
        """Calculate Cohen's kappa."""
        return cohen_kappa_score(y_true, y_pred, sample_weight=sample_weight)
    
    def _calculate_balanced_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray,
                                    y_prob: Optional[np.ndarray], config: MetricConfig,
                                    sample_weight: Optional[np.ndarray]) -> float:
        """Calculate balanced accuracy."""
        return balanced_accuracy_score(y_true, y_pred, sample_weight=sample_weight)
    
    def _calculate_top_k_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray,
                                 y_prob: Optional[np.ndarray], config: MetricConfig,
                                 sample_weight: Optional[np.ndarray]) -> float:
        """Calculate top-k accuracy."""
        if y_prob is None:
            return None
        return top_k_accuracy_score(y_true, y_prob, k=config.k, sample_weight=sample_weight)
    
    # Regression metrics
    def _calculate_mse(self, y_true: np.ndarray, y_pred: np.ndarray,
                      y_prob: Optional[np.ndarray], config: MetricConfig,
                      sample_weight: Optional[np.ndarray]) -> float:
        """Calculate mean squared error."""
        return mean_squared_error(y_true, y_pred, sample_weight=sample_weight,
                                 multioutput=config.multioutput)
    
    def _calculate_mae(self, y_true: np.ndarray, y_pred: np.ndarray,
                      y_prob: Optional[np.ndarray], config: MetricConfig,
                      sample_weight: Optional[np.ndarray]) -> float:
        """Calculate mean absolute error."""
        return mean_absolute_error(y_true, y_pred, sample_weight=sample_weight,
                                  multioutput=config.multioutput)
    
    def _calculate_r2(self, y_true: np.ndarray, y_pred: np.ndarray,
                     y_prob: Optional[np.ndarray], config: MetricConfig,
                     sample_weight: Optional[np.ndarray]) -> float:
        """Calculate R² score."""
        return r2_score(y_true, y_pred, sample_weight=sample_weight,
                       multioutput=config.multioutput)
    
    def _calculate_explained_variance(self, y_true: np.ndarray, y_pred: np.ndarray,
                                     y_prob: Optional[np.ndarray], config: MetricConfig,
                                     sample_weight: Optional[np.ndarray]) -> float:
        """Calculate explained variance score."""
        return explained_variance_score(y_true, y_pred, sample_weight=sample_weight,
                                       multioutput=config.multioutput)
    
    def _calculate_max_error(self, y_true: np.ndarray, y_pred: np.ndarray,
                            y_prob: Optional[np.ndarray], config: MetricConfig,
                            sample_weight: Optional[np.ndarray]) -> float:
        """Calculate max error."""
        return max_error(y_true, y_pred)
    
    def _calculate_mean_poisson_deviance(self, y_true: np.ndarray, y_pred: np.ndarray,
                                        y_prob: Optional[np.ndarray], config: MetricConfig,
                                        sample_weight: Optional[np.ndarray]) -> float:
        """Calculate mean Poisson deviance."""
        return mean_poisson_deviance(y_true, y_pred, sample_weight=sample_weight)
    
    def _calculate_mean_gamma_deviance(self, y_true: np.ndarray, y_pred: np.ndarray,
                                      y_prob: Optional[np.ndarray], config: MetricConfig,
                                      sample_weight: Optional[np.ndarray]) -> float:
        """Calculate mean gamma deviance."""
        return mean_gamma_deviance(y_true, y_pred, sample_weight=sample_weight)
    
    def _calculate_mean_tweedie_deviance(self, y_true: np.ndarray, y_pred: np.ndarray,
                                        y_prob: Optional[np.ndarray], config: MetricConfig,
                                        sample_weight: Optional[np.ndarray]) -> float:
        """Calculate mean Tweedie deviance."""
        return mean_tweedie_deviance(y_true, y_pred, sample_weight=sample_weight)
    
    def _calculate_d2_tweedie(self, y_true: np.ndarray, y_pred: np.ndarray,
                             y_prob: Optional[np.ndarray], config: MetricConfig,
                             sample_weight: Optional[np.ndarray]) -> float:
        """Calculate D² Tweedie score."""
        return d2_tweedie_score(y_true, y_pred, sample_weight=sample_weight)
    
    def _calculate_mape(self, y_true: np.ndarray, y_pred: np.ndarray,
                       y_prob: Optional[np.ndarray], config: MetricConfig,
                       sample_weight: Optional[np.ndarray]) -> float:
        """Calculate mean absolute percentage error."""
        return mean_absolute_percentage_error(y_true, y_pred, sample_weight=sample_weight)
    
    def _calculate_msle(self, y_true: np.ndarray, y_pred: np.ndarray,
                       y_prob: Optional[np.ndarray], config: MetricConfig,
                       sample_weight: Optional[np.ndarray]) -> float:
        """Calculate mean squared logarithmic error."""
        return mean_squared_log_error(y_true, y_pred, sample_weight=sample_weight)
    
    def _calculate_median_ae(self, y_true: np.ndarray, y_pred: np.ndarray,
                            y_prob: Optional[np.ndarray], config: MetricConfig,
                            sample_weight: Optional[np.ndarray]) -> float:
        """Calculate median absolute error."""
        return median_absolute_error(y_true, y_pred, sample_weight=sample_weight)
    
    def _calculate_mean_pinball_loss(self, y_true: np.ndarray, y_pred: np.ndarray,
                                    y_prob: Optional[np.ndarray], config: MetricConfig,
                                    sample_weight: Optional[np.ndarray]) -> float:
        """Calculate mean pinball loss."""
        return mean_pinball_loss(y_true, y_pred, alpha=0.5, sample_weight=sample_weight)
    
    # Generation metrics (placeholder implementations)
    def _calculate_bleu(self, y_true: np.ndarray, y_pred: np.ndarray,
                       y_prob: Optional[np.ndarray], config: MetricConfig,
                       sample_weight: Optional[np.ndarray]) -> float:
        """Calculate BLEU score."""
        # Placeholder - would need nltk or sacrebleu
        return 0.0
    
    def _calculate_rouge(self, y_true: np.ndarray, y_pred: np.ndarray,
                        y_prob: Optional[np.ndarray], config: MetricConfig,
                        sample_weight: Optional[np.ndarray]) -> float:
        """Calculate ROUGE score."""
        # Placeholder - would need rouge-score
        return 0.0
    
    def _calculate_meteor(self, y_true: np.ndarray, y_pred: np.ndarray,
                         y_prob: Optional[np.ndarray], config: MetricConfig,
                         sample_weight: Optional[np.ndarray]) -> float:
        """Calculate METEOR score."""
        # Placeholder - would need nltk
        return 0.0
    
    def _calculate_cider(self, y_true: np.ndarray, y_pred: np.ndarray,
                        y_prob: Optional[np.ndarray], config: MetricConfig,
                        sample_weight: Optional[np.ndarray]) -> float:
        """Calculate CIDEr score."""
        # Placeholder - would need pycocoevalcap
        return 0.0
    
    def _calculate_bert_score(self, y_true: np.ndarray, y_pred: np.ndarray,
                             y_prob: Optional[np.ndarray], config: MetricConfig,
                             sample_weight: Optional[np.ndarray]) -> float:
        """Calculate BERT score."""
        # Placeholder - would need bert-score
        return 0.0
    
    def _calculate_perplexity(self, y_true: np.ndarray, y_pred: np.ndarray,
                             y_prob: Optional[np.ndarray], config: MetricConfig,
                             sample_weight: Optional[np.ndarray]) -> float:
        """Calculate perplexity."""
        if y_prob is None:
            return None
        return np.exp(-np.mean(np.log(y_prob + 1e-10)))
    
    def _calculate_bleurt(self, y_true: np.ndarray, y_pred: np.ndarray,
                         y_prob: Optional[np.ndarray], config: MetricConfig,
                         sample_weight: Optional[np.ndarray]) -> float:
        """Calculate BLEURT score."""
        # Placeholder - would need bleurt
        return 0.0
    
    def _calculate_comet(self, y_true: np.ndarray, y_pred: np.ndarray,
                        y_prob: Optional[np.ndarray], config: MetricConfig,
                        sample_weight: Optional[np.ndarray]) -> float:
        """Calculate COMET score."""
        # Placeholder - would need unbabel-comet
        return 0.0
    
    def plot_confusion_matrix(self, confusion_matrix: np.ndarray, 
                             class_names: Optional[List[str]] = None,
                             save_path: Optional[str] = None):
        """Plot confusion matrix."""
        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def plot_metrics_comparison(self, results: List[EvaluationResult],
                               save_path: Optional[str] = None):
        """Plot metrics comparison across different models/configurations."""
        if not results:
            return
        
        # Extract common metrics
        common_metrics = set.intersection(*[set(r.metrics.keys()) for r in results])
        
        if not common_metrics:
            return
        
        fig, axes = plt.subplots(1, len(common_metrics), figsize=(5*len(common_metrics), 6))
        if len(common_metrics) == 1:
            axes = [axes]
        
        for i, metric in enumerate(common_metrics):
            values = [r.metrics[metric] for r in results if r.metrics[metric] is not None]
            labels = [f"Model {j+1}" for j, r in enumerate(results) if r.metrics[metric] is not None]
            
            axes[i].bar(labels, values)
            axes[i].set_title(f'{metric.upper()} Comparison')
            axes[i].set_ylabel(metric.upper())
            axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Metrics comparison saved to {save_path}")
        
        plt.show()

# Factory functions
async def create_evaluation_metrics(device_manager: DeviceManager) -> EvaluationMetrics:
    """Create evaluation metrics instance."""
    return EvaluationMetrics(device_manager)

def create_metric_config(task_type: TaskType, metric_types: Optional[List[MetricType]] = None,
                        **kwargs) -> MetricConfig:
    """Create metric configuration."""
    return MetricConfig(task_type=task_type, metric_types=metric_types or [], **kwargs)

# Quick usage functions
async def quick_classification_evaluation():
    """Quick example of classification evaluation."""
    device_manager = DeviceManager()
    evaluator = await create_evaluation_metrics(device_manager)
    
    # Create metric config
    config = create_metric_config(
        task_type=TaskType.CLASSIFICATION,
        metric_types=[
            MetricType.ACCURACY, MetricType.PRECISION, MetricType.RECALL,
            MetricType.F1, MetricType.ROC_AUC
        ],
        average='weighted'
    )
    
    # Simulate data
    np.random.seed(42)
    y_true = np.random.randint(0, 3, 1000)
    y_pred = np.random.randint(0, 3, 1000)
    y_prob = np.random.rand(1000, 3)
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
    
    # Evaluate
    result = evaluator.evaluate(config, y_true, y_pred, y_prob)
    
    print("✅ Classification evaluation completed successfully")
    print(f"Accuracy: {result.metrics['accuracy']:.4f}")
    print(f"F1 Score: {result.metrics['f1']:.4f}")
    
    return result

async def quick_regression_evaluation():
    """Quick example of regression evaluation."""
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
    np.random.seed(42)
    y_true = np.random.randn(1000)
    y_pred = y_true + np.random.randn(1000) * 0.1
    
    # Evaluate
    result = evaluator.evaluate(config, y_true, y_pred)
    
    print("✅ Regression evaluation completed successfully")
    print(f"MSE: {result.metrics['mse']:.4f}")
    print(f"R²: {result.metrics['r2']:.4f}")
    
    return result

# Example usage
if __name__ == "__main__":
    async def demo():
        
    """demo function."""
# Quick examples
        print("🧪 Classification Evaluation:")
        await quick_classification_evaluation()
        
        print("\n🧪 Regression Evaluation:")
        await quick_regression_evaluation()
    
    asyncio.run(demo()) 