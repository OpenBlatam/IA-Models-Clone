from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union, Callable, Iterator
from dataclasses import dataclass, field
import logging
from pathlib import Path
import time
import json
import pickle
from enum import Enum
import os
import math
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
from sklearn.metrics import (
from sklearn.metrics import (
import nltk
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk.translate.meteor_score import meteor_score
from nltk.translate.nist_score import sentence_nist, corpus_nist
from nltk.translate.rouge_score import rouge_n, rouge_l
from nltk.tokenize import word_tokenize
import re
import cv2
from PIL import Image
import io
import base64
                    from sklearn.metrics import explained_variance_score
                    from sklearn.metrics import max_error
        from scipy import stats
            from nltk.translate.bleu_score import SmoothingFunction
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
Comprehensive Evaluation Metrics System
Advanced evaluation metrics for different deep learning tasks.
"""

    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score,
    log_loss, cohen_kappa_score, matthews_corrcoef,
    precision_recall_curve, roc_curve, average_precision_score
)
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    adjusted_rand_score, normalized_mutual_info_score,
    homogeneity_score, completeness_score, v_measure_score
)
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


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


class ClassificationMetrics:
    """Comprehensive classification metrics."""
    
    def __init__(self, config: MetricConfig):
        
    """__init__ function."""
self.config = config
        self.logger = self._setup_logging()
        self.results = {}
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, 
                 y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Evaluate classification metrics."""
        self.logger.info("Evaluating classification metrics")
        
        # Convert to numpy arrays if needed
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        if y_prob is not None:
            y_prob = np.array(y_prob)
        
        results = {}
        
        for metric_type in self.config.metrics:
            try:
                if metric_type == MetricType.ACCURACY:
                    results['accuracy'] = accuracy_score(y_true, y_pred)
                
                elif metric_type == MetricType.PRECISION:
                    results['precision'] = precision_score(
                        y_true, y_pred, average=self.config.average, 
                        zero_division=self.config.zero_division
                    )
                
                elif metric_type == MetricType.RECALL:
                    results['recall'] = recall_score(
                        y_true, y_pred, average=self.config.average,
                        zero_division=self.config.zero_division
                    )
                
                elif metric_type == MetricType.F1_SCORE:
                    results['f1_score'] = f1_score(
                        y_true, y_pred, average=self.config.average,
                        zero_division=self.config.zero_division
                    )
                
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
                
                elif metric_type == MetricType.CONFUSION_MATRIX:
                    results['confusion_matrix'] = confusion_matrix(y_true, y_pred)
                
                elif metric_type == MetricType.CLASSIFICATION_REPORT:
                    results['classification_report'] = classification_report(
                        y_true, y_pred, output_dict=True
                    )
                
                elif metric_type == MetricType.LOG_LOSS:
                    if y_prob is not None:
                        results['log_loss'] = log_loss(y_true, y_prob)
                    else:
                        self.logger.warning("Log loss requires probability predictions")
                
                elif metric_type == MetricType.COHEN_KAPPA:
                    results['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
                
                elif metric_type == MetricType.MATTHEWS_CORRELATION:
                    if len(np.unique(y_true)) == 2:
                        results['matthews_correlation'] = matthews_corrcoef(y_true, y_pred)
                    else:
                        self.logger.warning("Matthews correlation only for binary classification")
                
            except Exception as e:
                self.logger.error(f"Error computing {metric_type.value}: {e}")
        
        # Store results
        self.results = results
        
        # Log results
        for metric_name, value in results.items():
            if isinstance(value, (int, float)):
                self.logger.info(f"{metric_name}: {value:.4f}")
        
        return results
    
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


class RegressionMetrics:
    """Comprehensive regression metrics."""
    
    def __init__(self, config: MetricConfig):
        
    """__init__ function."""
self.config = config
        self.logger = self._setup_logging()
        self.results = {}
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Evaluate regression metrics."""
        self.logger.info("Evaluating regression metrics")
        
        # Convert to numpy arrays if needed
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        results = {}
        
        for metric_type in self.config.metrics:
            try:
                if metric_type == MetricType.MSE:
                    results['mse'] = mean_squared_error(y_true, y_pred, multioutput=self.config.multioutput)
                
                elif metric_type == MetricType.RMSE:
                    results['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred, multioutput=self.config.multioutput))
                
                elif metric_type == MetricType.MAE:
                    results['mae'] = mean_absolute_error(y_true, y_pred, multioutput=self.config.multioutput)
                
                elif metric_type == MetricType.R2_SCORE:
                    results['r2_score'] = r2_score(y_true, y_pred, multioutput=self.config.multioutput)
                
                elif metric_type == MetricType.EXPLAINED_VARIANCE:
                    results['explained_variance'] = explained_variance_score(y_true, y_pred, multioutput=self.config.multioutput)
                
                elif metric_type == MetricType.MAX_ERROR:
                    results['max_error'] = max_error(y_true, y_pred)
                
                elif metric_type == MetricType.MEAN_ABSOLUTE_PERCENTAGE_ERROR:
                    results['mape'] = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
                
            except Exception as e:
                self.logger.error(f"Error computing {metric_type.value}: {e}")
        
        # Store results
        self.results = results
        
        # Log results
        for metric_name, value in results.items():
            if isinstance(value, (int, float)):
                self.logger.info(f"{metric_name}: {value:.4f}")
        
        return results
    
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
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title('Q-Q Plot')
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Regression plots saved to {save_path}")
        
        plt.show()


class GenerationMetrics:
    """Comprehensive generation metrics for text generation."""
    
    def __init__(self, config: MetricConfig):
        
    """__init__ function."""
self.config = config
        self.logger = self._setup_logging()
        self.results = {}
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def evaluate(self, references: List[List[str]], predictions: List[str]) -> Dict[str, float]:
        """Evaluate generation metrics."""
        self.logger.info("Evaluating generation metrics")
        
        results = {}
        
        for metric_type in self.config.metrics:
            try:
                if metric_type == MetricType.BLEU:
                    # Tokenize references and predictions
                    tokenized_refs = [[word_tokenize(ref) for ref in refs] for refs in references]
                    tokenized_preds = [word_tokenize(pred) for pred in predictions]
                    
                    # Calculate BLEU scores
                    bleu_1 = self._calculate_bleu(tokenized_refs, tokenized_preds, 1)
                    bleu_2 = self._calculate_bleu(tokenized_refs, tokenized_preds, 2)
                    bleu_3 = self._calculate_bleu(tokenized_refs, tokenized_preds, 3)
                    bleu_4 = self._calculate_bleu(tokenized_refs, tokenized_preds, 4)
                    
                    results['bleu_1'] = bleu_1
                    results['bleu_2'] = bleu_2
                    results['bleu_3'] = bleu_3
                    results['bleu_4'] = bleu_4
                
                elif metric_type == MetricType.METEOR:
                    meteor_scores = []
                    for refs, pred in zip(references, predictions):
                        try:
                            score = meteor_score(refs, pred)
                            meteor_scores.append(score)
                        except:
                            meteor_scores.append(0.0)
                    results['meteor'] = np.mean(meteor_scores)
                
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
                
                elif metric_type == MetricType.NIST:
                    # Tokenize for NIST
                    tokenized_refs = [[word_tokenize(ref) for ref in refs] for refs in references]
                    tokenized_preds = [word_tokenize(pred) for pred in predictions]
                    
                    nist_1 = self._calculate_nist(tokenized_refs, tokenized_preds, 1)
                    nist_2 = self._calculate_nist(tokenized_refs, tokenized_preds, 2)
                    nist_3 = self._calculate_nist(tokenized_refs, tokenized_preds, 3)
                    nist_4 = self._calculate_nist(tokenized_refs, tokenized_preds, 4)
                    
                    results['nist_1'] = nist_1
                    results['nist_2'] = nist_2
                    results['nist_3'] = nist_3
                    results['nist_4'] = nist_4
                
                elif metric_type == MetricType.PERPLEXITY:
                    # Simple perplexity calculation
                    # In practice, you'd use a language model
                    results['perplexity'] = self._calculate_perplexity(predictions)
                
                elif metric_type == MetricType.DIVERSITY:
                    results['diversity'] = self._calculate_diversity(predictions)
                
                elif metric_type == MetricType.COHERENCE:
                    results['coherence'] = self._calculate_coherence(predictions)
                
            except Exception as e:
                self.logger.error(f"Error computing {metric_type.value}: {e}")
        
        # Store results
        self.results = results
        
        # Log results
        for metric_name, value in results.items():
            if isinstance(value, (int, float)):
                self.logger.info(f"{metric_name}: {value:.4f}")
        
        return results
    
    def _calculate_bleu(self, references: List[List[List[str]]], 
                       predictions: List[List[str]], n: int) -> float:
        """Calculate BLEU score."""
        if self.config.smoothing:
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
    
    def _calculate_perplexity(self, predictions: List[str]) -> float:
        """Calculate perplexity (simplified)."""
        # This is a simplified perplexity calculation
        # In practice, you'd use a language model
        total_words = sum(len(word_tokenize(pred)) for pred in predictions)
        return total_words / len(predictions) if predictions else 0.0
    
    def _calculate_diversity(self, predictions: List[str]) -> float:
        """Calculate diversity (type-token ratio)."""
        all_words = []
        for pred in predictions:
            all_words.extend(word_tokenize(pred.lower()))
        
        if not all_words:
            return 0.0
        
        unique_words = set(all_words)
        return len(unique_words) / len(all_words)
    
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


class ClusteringMetrics:
    """Comprehensive clustering metrics."""
    
    def __init__(self, config: MetricConfig):
        
    """__init__ function."""
self.config = config
        self.logger = self._setup_logging()
        self.results = {}
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def evaluate(self, X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Evaluate clustering metrics."""
        self.logger.info("Evaluating clustering metrics")
        
        # Convert to numpy arrays if needed
        X = np.array(X)
        labels = np.array(labels)
        
        results = {}
        
        for metric_type in self.config.metrics:
            try:
                if metric_type == MetricType.SILHOUETTE_SCORE:
                    results['silhouette_score'] = silhouette_score(X, labels, metric=self.config.metric)
                
                elif metric_type == MetricType.CALINSKI_HARABASZ:
                    results['calinski_harabasz'] = calinski_harabasz_score(X, labels)
                
                elif metric_type == MetricType.DAVIES_BOULDIN:
                    results['davies_bouldin'] = davies_bouldin_score(X, labels)
                
                elif metric_type == MetricType.ADJUSTED_RAND:
                    # This requires true labels, so we'll skip if not available
                    self.logger.warning("Adjusted Rand Index requires true labels")
                
                elif metric_type == MetricType.NORMALIZED_MUTUAL_INFO:
                    # This requires true labels, so we'll skip if not available
                    self.logger.warning("Normalized Mutual Info requires true labels")
                
                elif metric_type == MetricType.HOMOGENEITY:
                    # This requires true labels, so we'll skip if not available
                    self.logger.warning("Homogeneity requires true labels")
                
                elif metric_type == MetricType.COMPLETENESS:
                    # This requires true labels, so we'll skip if not available
                    self.logger.warning("Completeness requires true labels")
                
                elif metric_type == MetricType.V_MEASURE:
                    # This requires true labels, so we'll skip if not available
                    self.logger.warning("V-measure requires true labels")
                
            except Exception as e:
                self.logger.error(f"Error computing {metric_type.value}: {e}")
        
        # Store results
        self.results = results
        
        # Log results
        for metric_name, value in results.items():
            if isinstance(value, (int, float)):
                self.logger.info(f"{metric_name}: {value:.4f}")
        
        return results
    
    def evaluate_with_true_labels(self, X: np.ndarray, labels: np.ndarray, 
                                 true_labels: np.ndarray) -> Dict[str, float]:
        """Evaluate clustering metrics with true labels."""
        self.logger.info("Evaluating clustering metrics with true labels")
        
        # Convert to numpy arrays if needed
        X = np.array(X)
        labels = np.array(labels)
        true_labels = np.array(true_labels)
        
        results = {}
        
        for metric_type in self.config.metrics:
            try:
                if metric_type == MetricType.ADJUSTED_RAND:
                    results['adjusted_rand'] = adjusted_rand_score(true_labels, labels)
                
                elif metric_type == MetricType.NORMALIZED_MUTUAL_INFO:
                    results['normalized_mutual_info'] = normalized_mutual_info_score(true_labels, labels)
                
                elif metric_type == MetricType.HOMOGENEITY:
                    results['homogeneity'] = homogeneity_score(true_labels, labels)
                
                elif metric_type == MetricType.COMPLETENESS:
                    results['completeness'] = completeness_score(true_labels, labels)
                
                elif metric_type == MetricType.V_MEASURE:
                    results['v_measure'] = v_measure_score(true_labels, labels)
                
            except Exception as e:
                self.logger.error(f"Error computing {metric_type.value}: {e}")
        
        # Add unsupervised metrics
        unsupervised_results = self.evaluate(X, labels)
        results.update(unsupervised_results)
        
        # Store results
        self.results = results
        
        # Log results
        for metric_name, value in results.items():
            if isinstance(value, (int, float)):
                self.logger.info(f"{metric_name}: {value:.4f}")
        
        return results


class EvaluationManager:
    """Comprehensive evaluation manager for different tasks."""
    
    def __init__(self, config: MetricConfig):
        
    """__init__ function."""
self.config = config
        self.logger = self._setup_logging()
        
        # Initialize metric evaluators
        self.classification_metrics = ClassificationMetrics(config)
        self.regression_metrics = RegressionMetrics(config)
        self.generation_metrics = GenerationMetrics(config)
        self.clustering_metrics = ClusteringMetrics(config)
        
        # Results storage
        self.results = {}
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def evaluate_classification(self, y_true: np.ndarray, y_pred: np.ndarray,
                              y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Evaluate classification task."""
        self.logger.info("Evaluating classification task")
        
        results = self.classification_metrics.evaluate(y_true, y_pred, y_prob)
        self.results['classification'] = results
        
        return results
    
    def evaluate_regression(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Evaluate regression task."""
        self.logger.info("Evaluating regression task")
        
        results = self.regression_metrics.evaluate(y_true, y_pred)
        self.results['regression'] = results
        
        return results
    
    def evaluate_generation(self, references: List[List[str]], 
                          predictions: List[str]) -> Dict[str, float]:
        """Evaluate generation task."""
        self.logger.info("Evaluating generation task")
        
        results = self.generation_metrics.evaluate(references, predictions)
        self.results['generation'] = results
        
        return results
    
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
    
    def save_results(self, file_path: Optional[str] = None):
        """Save evaluation results."""
        if file_path is None:
            file_path = self.config.results_file
        
        with open(file_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(self.results, f, indent=2, default=str)
        
        self.logger.info(f"Results saved to {file_path}")
    
    def load_results(self, file_path: str):
        """Load evaluation results."""
        with open(file_path, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            self.results = json.load(f)
        
        self.logger.info(f"Results loaded from {file_path}")
    
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


def demonstrate_evaluation_metrics():
    """Demonstrate evaluation metrics capabilities."""
    print("Evaluation Metrics Demonstration")
    print("=" * 40)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    n_classes = 3
    
    # Classification data
    X_class = np.random.randn(n_samples, n_features)
    y_true_class = np.random.randint(0, n_classes, n_samples)
    y_pred_class = np.random.randint(0, n_classes, n_samples)
    y_prob_class = np.random.rand(n_samples, n_classes)
    y_prob_class = y_prob_class / y_prob_class.sum(axis=1, keepdims=True)
    
    # Regression data
    X_reg = np.random.randn(n_samples, n_features)
    y_true_reg = np.random.randn(n_samples)
    y_pred_reg = y_true_reg + np.random.normal(0, 0.1, n_samples)
    
    # Generation data
    references = [
        ["The cat is on the mat", "A cat sits on the mat"],
        ["The dog runs fast", "A dog is running quickly"],
        ["The bird sings beautifully", "A bird is singing nicely"]
    ]
    predictions = [
        "The cat is on the mat",
        "The dog runs fast",
        "The bird sings beautifully"
    ]
    
    # Clustering data
    X_clust = np.random.randn(n_samples, 2)
    labels_clust = np.random.randint(0, 3, n_samples)
    
    # Test different configurations
    configs = [
        {
            'name': 'Classification Metrics',
            'config': MetricConfig(
                task_type=TaskType.CLASSIFICATION,
                metrics=[MetricType.ACCURACY, MetricType.PRECISION, MetricType.RECALL, 
                        MetricType.F1_SCORE, MetricType.ROC_AUC, MetricType.CONFUSION_MATRIX]
            ),
            'task_type': TaskType.CLASSIFICATION,
            'kwargs': {
                'y_true': y_true_class,
                'y_pred': y_pred_class,
                'y_prob': y_prob_class
            }
        },
        {
            'name': 'Regression Metrics',
            'config': MetricConfig(
                task_type=TaskType.REGRESSION,
                metrics=[MetricType.MSE, MetricType.RMSE, MetricType.MAE, MetricType.R2_SCORE]
            ),
            'task_type': TaskType.REGRESSION,
            'kwargs': {
                'y_true': y_true_reg,
                'y_pred': y_pred_reg
            }
        },
        {
            'name': 'Generation Metrics',
            'config': MetricConfig(
                task_type=TaskType.GENERATION,
                metrics=[MetricType.BLEU, MetricType.METEOR, MetricType.ROUGE]
            ),
            'task_type': TaskType.GENERATION,
            'kwargs': {
                'references': references,
                'predictions': predictions
            }
        },
        {
            'name': 'Clustering Metrics',
            'config': MetricConfig(
                task_type=TaskType.CLUSTERING,
                metrics=[MetricType.SILHOUETTE_SCORE, MetricType.CALINSKI_HARABASZ, 
                        MetricType.DAVIES_BOULDIN]
            ),
            'task_type': TaskType.CLUSTERING,
            'kwargs': {
                'X': X_clust,
                'labels': labels_clust
            }
        }
    ]
    
    results = {}
    
    for i, config in enumerate(configs):
        print(f"\nTesting configuration {i+1}: {config['name']}")
        print(f"  Task type: {config['task_type'].value}")
        print(f"  Metrics: {[m.value for m in config['config'].metrics]}")
        
        try:
            # Create evaluation manager
            evaluator = EvaluationManager(config['config'])
            
            # Evaluate
            task_results = evaluator.evaluate(config['task_type'], **config['kwargs'])
            
            print(f"  Results:")
            for metric_name, value in task_results.items():
                if isinstance(value, (int, float)):
                    print(f"    {metric_name}: {value:.4f}")
            
            results[f"config_{i}"] = {
                'config': config,
                'results': task_results,
                'success': True
            }
            
        except Exception as e:
            print(f"  Error: {e}")
            results[f"config_{i}"] = {
                'config': config,
                'error': str(e),
                'success': False
            }
    
    return results


if __name__ == "__main__":
    # Demonstrate evaluation metrics
    results = demonstrate_evaluation_metrics()
    print("\nEvaluation metrics demonstration completed!") 