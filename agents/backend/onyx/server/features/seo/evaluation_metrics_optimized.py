#!/usr/bin/env python3
"""
OPTIMIZED SEO-Specific Evaluation Metrics System
High-performance evaluation framework for SEO deep learning models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
import logging
import time
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score, 
    confusion_matrix, classification_report, mean_squared_error, 
    mean_absolute_error, r2_score, jaccard_score, f1_score,
    precision_score, recall_score, cohen_kappa_score, matthews_corrcoef,
    silhouette_score, calinski_harabasz_score, davies_bouldin_score
)
from sklearn.preprocessing import label_binarize
from scipy import stats
from scipy.stats import pearsonr, spearmanr, kendalltau
import warnings
import asyncio
from typing_extensions import Literal, TypedDict
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# Constants
MAX_CONNECTIONS = 1000
MAX_RETRIES = 100
BATCH_SIZE = 10000  # Optimized batch size for memory efficiency
CACHE_SIZE = 128  # LRU cache size for expensive computations

# =============================================================================
# OPTIMIZED SEO-SPECIFIC EVALUATION METRICS IMPLEMENTATION
# =============================================================================

@dataclass
class SEOMetricsConfig:
    """Configuration for SEO-specific evaluation metrics."""
    # Core SEO metrics
    ranking_metrics: bool = True
    content_quality_metrics: bool = True
    user_engagement_metrics: bool = True
    technical_seo_metrics: bool = True
    
    # Ranking evaluation
    ndcg_k_values: List[int] = field(default_factory=lambda: [1, 3, 5, 10])
    map_k_values: List[int] = field(default_factory=lambda: [1, 3, 5, 10])
    
    # Content quality thresholds
    min_content_length: int = 300
    max_keyword_density: float = 0.03
    min_readability_score: float = 60.0
    
    # User engagement thresholds
    min_time_on_page: float = 30.0
    min_click_through_rate: float = 0.01
    max_bounce_rate: float = 0.7
    
    # Performance optimization
    use_vectorization: bool = True
    use_caching: bool = True
    batch_size: int = BATCH_SIZE
    max_workers: int = mp.cpu_count()

@dataclass
class ClassificationMetricsConfig:
    """Configuration for classification metrics."""
    average: str = "weighted"  # micro, macro, weighted, binary
    zero_division: int = 0
    sample_weight: Optional[np.ndarray] = None
    use_vectorization: bool = True

@dataclass
class RegressionMetricsConfig:
    """Configuration for regression metrics."""
    multioutput: str = "uniform_average"  # raw_values, uniform_average
    sample_weight: Optional[np.ndarray] = None
    use_vectorization: bool = True

@dataclass
class RankingMetricsConfig:
    """Configuration for ranking metrics."""
    k_values: List[int] = field(default_factory=lambda: [1, 3, 5, 10])
    relevance_threshold: float = 0.5
    use_vectorization: bool = True

class OptimizedSEOSpecificMetrics:
    """Optimized SEO-specific evaluation metrics for deep learning models."""
    
    def __init__(self, config: SEOMetricsConfig):
        self.config = config
        self._cache = {}
        self._vectorized_ops = config.use_vectorization
        
    def calculate_ranking_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                relevance_scores: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate SEO ranking metrics with vectorized operations."""
        if relevance_scores is None:
            relevance_scores = y_true
        
        # Pre-compute sorted indices for all k values
        max_k = max(self.config.ndcg_k_values + self.config.map_k_values)
        sorted_indices = np.argsort(y_pred)[::-1][:max_k]
        
        metrics = {}
        
        # Vectorized NDCG calculation
        if self._vectorized_ops:
            metrics.update(self._calculate_ndcg_vectorized(y_true, y_pred, relevance_scores, sorted_indices))
        else:
            for k in self.config.ndcg_k_values:
                ndcg_score = self._calculate_ndcg_optimized(y_true, y_pred, k, relevance_scores, sorted_indices)
                metrics[f'ndcg_at_{k}'] = ndcg_score
        
        # Vectorized MAP calculation
        if self._vectorized_ops:
            metrics.update(self._calculate_map_vectorized(y_true, y_pred, sorted_indices))
        else:
            for k in self.config.map_k_values:
                map_score = self._calculate_map_optimized(y_true, y_pred, k, sorted_indices)
                metrics[f'map_at_{k}'] = map_score
        
        # Optimized MRR calculation
        metrics['mrr'] = self._calculate_mrr_optimized(y_true, y_pred, sorted_indices)
        
        # Vectorized Precision and Recall at K
        if self._vectorized_ops:
            metrics.update(self._calculate_precision_recall_vectorized(y_true, y_pred, sorted_indices))
        else:
            for k in self.config.ndcg_k_values:
                precision_k = self._calculate_precision_at_k_optimized(y_true, y_pred, k, sorted_indices)
                recall_k = self._calculate_recall_at_k_optimized(y_true, y_pred, k, sorted_indices)
                metrics[f'precision_at_{k}'] = precision_k
                metrics[f'recall_at_{k}'] = recall_k
        
        return metrics
    
    def _calculate_ndcg_vectorized(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                  relevance_scores: np.ndarray, sorted_indices: np.ndarray) -> Dict[str, float]:
        """Vectorized NDCG calculation for all k values."""
        metrics = {}
        
        # Pre-compute log denominators
        log_denoms = np.log2(np.arange(2, len(sorted_indices) + 2))
        
        for k in self.config.ndcg_k_values:
            if k <= len(sorted_indices):
                k_indices = sorted_indices[:k]
                k_log_denoms = log_denoms[:k]
                
                # Calculate DCG
                dcg = np.sum(relevance_scores[k_indices] / k_log_denoms)
                
                # Calculate IDCG
                ideal_indices = np.argsort(relevance_scores)[::-1][:k]
                idcg = np.sum(relevance_scores[ideal_indices] / k_log_denoms)
                
                metrics[f'ndcg_at_{k}'] = dcg / idcg if idcg > 0 else 0.0
        
        return metrics
    
    def _calculate_map_vectorized(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                sorted_indices: np.ndarray) -> Dict[str, float]:
        """Vectorized MAP calculation for all k values."""
        metrics = {}
        
        for k in self.config.map_k_values:
            if k <= len(sorted_indices):
                k_indices = sorted_indices[:k]
                
                # Vectorized precision calculation
                relevant_mask = y_true[k_indices] > 0
                if np.any(relevant_mask):
                    cumulative_relevant = np.cumsum(relevant_mask)
                    precision_values = cumulative_relevant / np.arange(1, k + 1)
                    ap = np.sum(precision_values * relevant_mask) / np.sum(relevant_mask)
                else:
                    ap = 0.0
                
                metrics[f'map_at_{k}'] = ap
        
        return metrics
    
    def _calculate_precision_recall_vectorized(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                            sorted_indices: np.ndarray) -> Dict[str, float]:
        """Vectorized precision and recall calculation for all k values."""
        metrics = {}
        total_relevant = np.sum(y_true > 0)
        
        for k in self.config.ndcg_k_values:
            if k <= len(sorted_indices):
                k_indices = sorted_indices[:k]
                relevant_count = np.sum(y_true[k_indices] > 0)
                
                metrics[f'precision_at_{k}'] = relevant_count / k
                metrics[f'recall_at_{k}'] = relevant_count / total_relevant if total_relevant > 0 else 0.0
        
        return metrics
    
    @lru_cache(maxsize=CACHE_SIZE)
    def _calculate_ndcg_optimized(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                 k: int, relevance_scores: np.ndarray, 
                                 sorted_indices: np.ndarray) -> float:
        """Optimized NDCG calculation with caching."""
        if k > len(sorted_indices):
            return 0.0
        
        k_indices = sorted_indices[:k]
        log_denoms = np.log2(np.arange(2, k + 2))
        
        # Calculate DCG
        dcg = np.sum(relevance_scores[k_indices] / log_denoms)
        
        # Calculate IDCG
        ideal_indices = np.argsort(relevance_scores)[::-1][:k]
        idcg = np.sum(relevance_scores[ideal_indices] / log_denoms)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def _calculate_map_optimized(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                k: int, sorted_indices: np.ndarray) -> float:
        """Optimized MAP calculation."""
        if k > len(sorted_indices):
            return 0.0
        
        k_indices = sorted_indices[:k]
        relevant_mask = y_true[k_indices] > 0
        
        if not np.any(relevant_mask):
            return 0.0
        
        cumulative_relevant = np.cumsum(relevant_mask)
        precision_values = cumulative_relevant / np.arange(1, k + 1)
        ap = np.sum(precision_values * relevant_mask) / np.sum(relevant_mask)
        
        return ap
    
    def _calculate_mrr_optimized(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                sorted_indices: np.ndarray) -> float:
        """Optimized MRR calculation."""
        relevant_positions = np.where(y_true[sorted_indices] > 0)[0]
        if len(relevant_positions) > 0:
            return 1.0 / (relevant_positions[0] + 1)
        return 0.0
    
    def _calculate_precision_at_k_optimized(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                          k: int, sorted_indices: np.ndarray) -> float:
        """Optimized precision at k calculation."""
        if k > len(sorted_indices):
            return 0.0
        
        k_indices = sorted_indices[:k]
        relevant_count = np.sum(y_true[k_indices] > 0)
        return relevant_count / k
    
    def _calculate_recall_at_k_optimized(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                       k: int, sorted_indices: np.ndarray) -> float:
        """Optimized recall at k calculation."""
        if k > len(sorted_indices):
            return 0.0
        
        k_indices = sorted_indices[:k]
        relevant_count = np.sum(y_true[k_indices] > 0)
        total_relevant = np.sum(y_true > 0)
        return relevant_count / total_relevant if total_relevant > 0 else 0.0
    
    def calculate_content_quality_metrics(self, content_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate content quality metrics with vectorization."""
        metrics = {}
        
        # Vectorized content length calculation
        if isinstance(content_data.get('content_length'), (list, np.ndarray)):
            content_lengths = np.array(content_data['content_length'])
            metrics['content_length'] = np.mean(content_lengths)
            metrics['content_length_score'] = np.mean(np.minimum(1.0, content_lengths / self.config.min_content_length))
        else:
            content_length = content_data.get('content_length', 0)
            metrics['content_length'] = content_length
            metrics['content_length_score'] = min(1.0, content_length / self.config.min_content_length)
        
        # Vectorized keyword density calculation
        if isinstance(content_data.get('keyword_density'), (list, np.ndarray)):
            keyword_densities = np.array(content_data['keyword_density'])
            metrics['keyword_density'] = np.mean(keyword_densities)
            metrics['keyword_density_score'] = np.mean(1.0 - np.minimum(1.0, keyword_densities / self.config.max_keyword_density))
        else:
            keyword_density = content_data.get('keyword_density', 0.0)
            metrics['keyword_density'] = keyword_density
            metrics['keyword_density_score'] = 1.0 - min(1.0, keyword_density / self.config.max_keyword_density)
        
        # Vectorized readability calculation
        if isinstance(content_data.get('readability_score'), (list, np.ndarray)):
            readability_scores = np.array(content_data['readability_score'])
            metrics['readability_score'] = np.mean(readability_scores)
            metrics['readability_score_normalized'] = np.mean(np.maximum(0.0, readability_scores / 100.0))
        else:
            readability_score = content_data.get('readability_score', 0.0)
            metrics['readability_score'] = readability_score
            metrics['readability_score_normalized'] = max(0.0, readability_score / 100.0)
        
        return metrics
    
    def calculate_user_engagement_metrics(self, engagement_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate user engagement metrics with vectorization."""
        metrics = {}
        
        # Vectorized time on page calculation
        if isinstance(engagement_data.get('time_on_page'), (list, np.ndarray)):
            time_on_pages = np.array(engagement_data['time_on_page'])
            metrics['time_on_page'] = np.mean(time_on_pages)
            metrics['time_on_page_score'] = np.mean(np.minimum(1.0, time_on_pages / self.config.min_time_on_page))
        else:
            time_on_page = engagement_data.get('time_on_page', 0.0)
            metrics['time_on_page'] = time_on_page
            metrics['time_on_page_score'] = min(1.0, time_on_page / self.config.min_time_on_page)
        
        # Vectorized CTR calculation
        if isinstance(engagement_data.get('click_through_rate'), (list, np.ndarray)):
            ctrs = np.array(engagement_data['click_through_rate'])
            metrics['click_through_rate'] = np.mean(ctrs)
            metrics['ctr_score'] = np.mean(np.minimum(1.0, ctrs / self.config.min_click_through_rate))
        else:
            ctr = engagement_data.get('click_through_rate', 0.0)
            metrics['click_through_rate'] = ctr
            metrics['ctr_score'] = min(1.0, ctr / self.config.min_click_through_rate)
        
        # Vectorized bounce rate calculation
        if isinstance(engagement_data.get('bounce_rate'), (list, np.ndarray)):
            bounce_rates = np.array(engagement_data['bounce_rate'])
            metrics['bounce_rate'] = np.mean(bounce_rates)
            metrics['bounce_rate_score'] = np.mean(np.maximum(0.0, 1.0 - bounce_rates / self.config.max_bounce_rate))
        else:
            bounce_rate = engagement_data.get('bounce_rate', 1.0)
            metrics['bounce_rate'] = bounce_rate
            metrics['bounce_rate_score'] = max(0.0, 1.0 - bounce_rate / self.config.max_bounce_rate)
        
        return metrics
    
    def calculate_technical_seo_metrics(self, technical_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate technical SEO metrics with vectorization."""
        metrics = {}
        
        # Core Web Vitals
        if 'core_web_vitals' in technical_data:
            vitals = technical_data['core_web_vitals']
            
            # LCP (Largest Contentful Paint)
            if isinstance(vitals.get('lcp'), (list, np.ndarray)):
                lcp_values = np.array(vitals['lcp'])
                metrics['lcp'] = np.mean(lcp_values)
                metrics['lcp_score'] = np.mean(np.maximum(0.0, 1.0 - lcp_values / 2500))  # 2.5s threshold
            else:
                lcp = vitals.get('lcp', 0)
                metrics['lcp'] = lcp
                metrics['lcp_score'] = max(0.0, 1.0 - lcp / 2500)
            
            # FID (First Input Delay)
            if isinstance(vitals.get('fid'), (list, np.ndarray)):
                fid_values = np.array(vitals['fid'])
                metrics['fid'] = np.mean(fid_values)
                metrics['fid_score'] = np.mean(np.maximum(0.0, 1.0 - fid_values / 100))  # 100ms threshold
            else:
                fid = vitals.get('fid', 0)
                metrics['fid'] = fid
                metrics['fid_score'] = max(0.0, 1.0 - fid / 100)
            
            # CLS (Cumulative Layout Shift)
            if isinstance(vitals.get('cls'), (list, np.ndarray)):
                cls_values = np.array(vitals['cls'])
                metrics['cls'] = np.mean(cls_values)
                metrics['cls_score'] = np.mean(np.maximum(0.0, 1.0 - cls_values / 0.1))  # 0.1 threshold
            else:
                cls = vitals.get('cls', 0)
                metrics['cls'] = cls
                metrics['cls_score'] = max(0.0, 1.0 - cls / 0.1)
        
        # Mobile friendliness
        if 'mobile_friendly' in technical_data:
            mobile_scores = technical_data['mobile_friendly']
            if isinstance(mobile_scores, (list, np.ndarray)):
                metrics['mobile_friendliness'] = np.mean(mobile_scores)
            else:
                metrics['mobile_friendliness'] = mobile_scores
        
        # Page load speed
        if 'page_load_speed' in technical_data:
            load_speeds = technical_data['page_load_speed']
            if isinstance(load_speeds, (list, np.ndarray)):
                metrics['page_load_speed'] = np.mean(load_speeds)
                metrics['load_speed_score'] = np.mean(np.maximum(0.0, 1.0 - load_speeds / 3000))  # 3s threshold
            else:
                load_speed = load_speeds
                metrics['page_load_speed'] = load_speed
                metrics['load_speed_score'] = max(0.0, 1.0 - load_speed / 3000)
        
        return metrics

class OptimizedClassificationMetrics:
    """Optimized classification metrics calculator."""
    
    def __init__(self, config: ClassificationMetricsConfig):
        self.config = config
        self._vectorized = config.use_vectorization
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate classification metrics with optimization."""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # Precision, Recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average=self.config.average, 
            zero_division=self.config.zero_division
        )
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1_score'] = f1
        
        # Additional metrics
        metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
        metrics['matthews_corrcoef'] = matthews_corrcoef(y_true, y_pred)
        
        # ROC AUC if probabilities available
        if y_prob is not None:
            try:
                if len(np.unique(y_true)) == 2:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
                else:
                    # Multi-class ROC AUC
                    y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
                    metrics['roc_auc'] = roc_auc_score(y_true_bin, y_prob, multi_class='ovr')
            except Exception as e:
                logger.warning(f"Could not calculate ROC AUC: {e}")
                metrics['roc_auc'] = 0.0
        
        return metrics

class OptimizedRegressionMetrics:
    """Optimized regression metrics calculator."""
    
    def __init__(self, config: RegressionMetricsConfig):
        self.config = config
        self._vectorized = config.use_vectorization
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate regression metrics with optimization."""
        metrics = {}
        
        # Basic regression metrics
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['r2_score'] = r2_score(y_true, y_pred)
        
        # Additional metrics
        metrics['mape'] = self._calculate_mape(y_true, y_pred)
        metrics['smape'] = self._calculate_smape(y_true, y_pred)
        metrics['huber_loss'] = self._calculate_huber_loss(y_true, y_pred)
        
        # Correlation coefficients
        try:
            pearson_corr, _ = pearsonr(y_true, y_pred)
            spearman_corr, _ = spearmanr(y_true, y_pred)
            kendall_corr, _ = kendalltau(y_true, y_pred)
            
            metrics['pearson_correlation'] = pearson_corr
            metrics['spearman_correlation'] = spearman_corr
            metrics['kendall_correlation'] = kendall_corr
        except Exception as e:
            logger.warning(f"Could not calculate correlation coefficients: {e}")
            metrics['pearson_correlation'] = 0.0
            metrics['spearman_correlation'] = 0.0
            metrics['kendall_correlation'] = 0.0
        
        return metrics
    
    def _calculate_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error."""
        mask = y_true != 0
        if np.any(mask):
            return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        return 0.0
    
    def _calculate_smape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Symmetric Mean Absolute Percentage Error."""
        mask = (y_true != 0) | (y_pred != 0)
        if np.any(mask):
            numerator = np.abs(y_true[mask] - y_pred[mask])
            denominator = (np.abs(y_true[mask]) + np.abs(y_pred[mask])) / 2
            return np.mean(numerator / denominator) * 100
        return 0.0
    
    def _calculate_huber_loss(self, y_true: np.ndarray, y_pred: np.ndarray, delta: float = 1.0) -> float:
        """Calculate Huber Loss."""
        error = y_true - y_pred
        abs_error = np.abs(error)
        quadratic = np.minimum(abs_error, delta)
        linear = abs_error - quadratic
        return np.mean(0.5 * quadratic**2 + delta * linear)

class OptimizedSEOModelEvaluator:
    """Optimized comprehensive SEO model evaluator."""
    
    def __init__(self, seo_config: SEOMetricsConfig,
                 classification_config: ClassificationMetricsConfig,
                 regression_config: RegressionMetricsConfig):
        self.seo_config = seo_config
        self.classification_config = classification_config
        self.regression_config = regression_config
        
        # Initialize optimized metric calculators
        self.seo_metrics = OptimizedSEOSpecificMetrics(seo_config)
        self.classification_metrics = OptimizedClassificationMetrics(classification_config)
        self.regression_metrics = OptimizedRegressionMetrics(regression_config)
        
        # Results storage with memory optimization
        self.evaluation_results = {}
        self._executor = ThreadPoolExecutor(max_workers=seo_config.max_workers)
        
        # Performance monitoring
        self.evaluation_times = {}
        self.memory_usage = {}
    
    async def evaluate_seo_model(self, model: nn.Module, 
                                test_data: Dict[str, Any],
                                task_type: str = "ranking") -> Dict[str, Any]:
        """Evaluate SEO model comprehensively with optimization."""
        start_time = time.time()
        print(f"🚀 Evaluating SEO model for task: {task_type}")
        
        try:
            # Extract and preprocess data
            y_true, y_pred, y_prob = self._extract_data(test_data)
            
            # Validate data
            self._validate_data(y_true, y_pred, task_type)
            
            # Calculate task-specific metrics
            if task_type == "classification":
                results = self.classification_metrics.calculate_metrics(y_true, y_pred, y_prob)
            elif task_type == "regression":
                results = self.regression_metrics.calculate_metrics(y_true, y_pred)
            elif task_type == "ranking":
                results = self.seo_metrics.calculate_ranking_metrics(y_true, y_pred, y_prob)
            else:
                raise ValueError(f"Unknown task type: {task_type}")
            
            # Add SEO-specific metrics asynchronously if available
            if any(key in test_data for key in ['content_data', 'engagement_data', 'technical_data']):
                seo_metrics = await self._calculate_seo_metrics_async(test_data)
                results.update(seo_metrics)
            
            # Store results
            self.evaluation_results[task_type] = results
            
            # Record performance metrics
            evaluation_time = time.time() - start_time
            self.evaluation_times[task_type] = evaluation_time
            
            print(f"✅ {task_type.capitalize()} evaluation completed in {evaluation_time:.4f}s")
            return results
            
        except Exception as e:
            print(f"❌ Error in {task_type} evaluation: {e}")
            raise
    
    def _extract_data(self, test_data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Extract and convert data efficiently."""
        y_true = test_data.get('y_true')
        y_pred = test_data.get('y_pred')
        y_prob = test_data.get('y_prob')
        
        # Convert to numpy if needed
        if torch.is_tensor(y_true):
            y_true = y_true.cpu().numpy()
        if torch.is_tensor(y_pred):
            y_pred = y_pred.cpu().numpy()
        if y_prob is not None and torch.is_tensor(y_prob):
            y_prob = y_prob.cpu().numpy()
        
        return y_true, y_pred, y_prob
    
    def _validate_data(self, y_true: np.ndarray, y_pred: np.ndarray, task_type: str):
        """Validate input data."""
        if y_true is None or y_pred is None:
            raise ValueError("Missing required data: y_true and y_pred")
        
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
        
        if len(y_true) == 0:
            raise ValueError("Empty data provided")
    
    async def _calculate_seo_metrics_async(self, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate SEO metrics asynchronously."""
        seo_metrics = {}
        
        # Create tasks for different metric types
        tasks = []
        
        if 'content_data' in test_data:
            tasks.append(self._executor.submit(
                self.seo_metrics.calculate_content_quality_metrics, 
                test_data['content_data']
            ))
        
        if 'engagement_data' in test_data:
            tasks.append(self._executor.submit(
                self.seo_metrics.calculate_user_engagement_metrics, 
                test_data['engagement_data']
            ))
        
        if 'technical_data' in test_data:
            tasks.append(self._executor.submit(
                self.seo_metrics.calculate_technical_seo_metrics, 
                test_data['technical_data']
            ))
        
        # Wait for all tasks to complete
        for future in tasks:
            try:
                result = future.result()
                seo_metrics.update(result)
            except Exception as e:
                logger.warning(f"Error calculating SEO metrics: {e}")
        
        return seo_metrics
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get comprehensive evaluation summary with performance metrics."""
        return {
            'task_results': self.evaluation_results,
            'overall_score': self._calculate_overall_score(),
            'recommendations': self._generate_recommendations(),
            'performance_metrics': {
                'evaluation_times': self.evaluation_times,
                'total_evaluation_time': sum(self.evaluation_times.values()),
                'average_evaluation_time': np.mean(list(self.evaluation_times.values())) if self.evaluation_times else 0.0
            }
        }
    
    def _calculate_overall_score(self) -> float:
        """Calculate overall model performance score."""
        if not self.evaluation_results:
            return 0.0
        
        scores = []
        for task, results in self.evaluation_results.items():
            if task == "classification":
                score = results.get('f1_score', 0.0)
            elif task == "regression":
                score = results.get('r2_score', 0.0)
            elif task == "ranking":
                score = results.get('ndcg_at_5', 0.0)
            else:
                score = 0.0
            
            scores.append(score)
        
        return np.mean(scores) if scores else 0.0
    
    def _generate_recommendations(self) -> List[str]:
        """Generate improvement recommendations based on metrics."""
        recommendations = []
        
        for task, results in self.evaluation_results.items():
            if task == "classification":
                if results.get('f1_score', 0.0) < 0.7:
                    recommendations.append("Classification F1 score is low. Consider data augmentation or model tuning.")
                if results.get('precision', 0.0) < 0.6:
                    recommendations.append("Low precision indicates high false positives. Review feature engineering.")
            
            elif task == "regression":
                if results.get('r2_score', 0.0) < 0.5:
                    recommendations.append("Low R² score suggests poor fit. Consider feature selection or model complexity.")
                if results.get('mae', float('inf')) > np.std(results.get('y_true', [0])) * 0.5:
                    recommendations.append("High MAE relative to data variance. Check for outliers or scaling issues.")
            
            elif task == "ranking":
                if results.get('ndcg_at_5', 0.0) < 0.6:
                    recommendations.append("Low NDCG@5 indicates poor ranking quality. Review relevance scoring.")
                if results.get('map_at_5', 0.0) < 0.5:
                    recommendations.append("Low MAP@5 suggests poor precision in top results. Optimize ranking algorithm.")
        
        return recommendations
    
    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=True)
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()

# =============================================================================
# UTILITY FUNCTIONS FOR OPTIMIZATION
# =============================================================================

def batch_process_data(data: np.ndarray, batch_size: int = BATCH_SIZE) -> List[np.ndarray]:
    """Process data in batches for memory efficiency."""
    return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

def optimize_memory_usage(data: np.ndarray, target_dtype: np.dtype = np.float32) -> np.ndarray:
    """Optimize memory usage by converting to appropriate data type."""
    if data.dtype != target_dtype:
        return data.astype(target_dtype)
    return data

def calculate_memory_usage(data: np.ndarray) -> float:
    """Calculate memory usage in MB."""
    return data.nbytes / (1024 * 1024)

# =============================================================================
# MAIN USAGE EXAMPLE
# =============================================================================

async def main():
    """Main function demonstrating optimized SEO evaluation."""
    
    # Create optimized configurations
    seo_config = SEOMetricsConfig(
        ranking_metrics=True,
        content_quality_metrics=True,
        user_engagement_metrics=True,
        technical_seo_metrics=True,
        use_vectorization=True,
        use_caching=True,
        batch_size=BATCH_SIZE,
        max_workers=mp.cpu_count()
    )
    
    classification_config = ClassificationMetricsConfig(
        average="weighted",
        use_vectorization=True
    )
    
    regression_config = RegressionMetricsConfig(
        multioutput="uniform_average",
        use_vectorization=True
    )
    
    # Create optimized evaluator
    evaluator = OptimizedSEOModelEvaluator(
        seo_config=seo_config,
        classification_config=classification_config,
        regression_config=regression_config
    )
    
    try:
        # Example data (replace with your actual data)
        test_data = {
            'y_true': np.random.randint(0, 2, 1000),
            'y_pred': np.random.randint(0, 2, 1000),
            'y_prob': np.random.random(1000),
            'content_data': {
                'content_length': np.random.randint(200, 2000, 1000),
                'keyword_density': np.random.random(1000) * 0.05,
                'readability_score': np.random.random(1000) * 100
            }
        }
        
        # Evaluate model
        results = await evaluator.evaluate_seo_model(
            model=None,  # Replace with your actual model
            test_data=test_data,
            task_type="classification"
        )
        
        # Get comprehensive summary
        summary = evaluator.get_evaluation_summary()
        
        print("🎯 Evaluation Results:")
        print(f"Overall Score: {summary['overall_score']:.4f}")
        print(f"Performance: {summary['performance_metrics']}")
        print(f"Recommendations: {summary['recommendations']}")
        
    finally:
        # Clean up resources
        evaluator.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
