#!/usr/bin/env python3
"""
Specialized SEO Evaluation Metrics
Appropriate evaluation metrics for SEO-specific tasks and analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    confusion_matrix, classification_report, mean_squared_error,
    mean_absolute_error, r2_score, jaccard_score, f1_score,
    recall_score, precision_score, cohen_kappa_score
)
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import warnings
import re
from collections import defaultdict, Counter
import math

warnings.filterwarnings('ignore')

@dataclass
class SEOMetricsConfig:
    """Configuration for SEO evaluation metrics."""
    # Task-specific settings
    task_type: str = "classification"  # classification, regression, clustering, ranking
    num_classes: int = 2
    average: str = "weighted"  # micro, macro, weighted, binary
    
    # SEO-specific thresholds
    seo_score_threshold: float = 0.7
    content_quality_threshold: float = 0.6
    keyword_density_threshold: float = 0.02
    readability_threshold: float = 0.5
    
    # Evaluation settings
    use_custom_metrics: bool = True
    use_seo_specific: bool = True
    normalize_scores: bool = True

class SEOSpecificMetrics:
    """SEO-specific evaluation metrics for content analysis."""
    
    def __init__(self, config: SEOMetricsConfig):
        self.config = config
        self.seo_keywords = [
            'seo', 'search', 'optimization', 'keywords', 'meta', 'title', 'description',
            'content', 'ranking', 'google', 'backlinks', 'analytics', 'traffic',
            'organic', 'sem', 'ppc', 'conversion', 'ctr', 'bounce_rate'
        ]
        
        # SEO scoring weights
        self.seo_weights = {
            'content_quality': 0.25,
            'keyword_optimization': 0.20,
            'technical_seo': 0.20,
            'readability': 0.15,
            'user_experience': 0.20
        }
    
    def calculate_content_quality_score(self, text: str) -> float:
        """Calculate content quality score based on SEO best practices."""
        if not text or len(text.strip()) == 0:
            return 0.0
        
        # Text length score (optimal: 300-2000 words)
        word_count = len(text.split())
        length_score = min(1.0, word_count / 1000) if word_count > 100 else word_count / 100
        
        # Keyword density score
        keyword_density = self._calculate_keyword_density(text)
        keyword_score = min(1.0, keyword_density / self.config.keyword_density_threshold)
        
        # Readability score
        readability_score = self._calculate_readability_score(text)
        
        # Content structure score
        structure_score = self._calculate_structure_score(text)
        
        # Overall content quality
        content_score = (
            length_score * 0.3 +
            keyword_score * 0.3 +
            readability_score * 0.2 +
            structure_score * 0.2
        )
        
        return min(1.0, max(0.0, content_score))
    
    def _calculate_keyword_density(self, text: str) -> float:
        """Calculate keyword density in text."""
        if not text:
            return 0.0
        
        text_lower = text.lower()
        total_words = len(text.split())
        
        if total_words == 0:
            return 0.0
        
        keyword_count = sum(text_lower.count(keyword) for keyword in self.seo_keywords)
        return keyword_count / total_words
    
    def _calculate_readability_score(self, text: str) -> float:
        """Calculate readability score using Flesch Reading Ease."""
        if not text:
            return 0.0
        
        sentences = len(re.split(r'[.!?]+', text))
        words = len(text.split())
        syllables = self._count_syllables(text)
        
        if sentences == 0 or words == 0:
            return 0.0
        
        # Flesch Reading Ease formula
        flesch_score = 206.835 - (1.015 * (words / sentences)) - (84.6 * (syllables / words))
        
        # Normalize to 0-1 scale
        if flesch_score >= 90:
            return 1.0
        elif flesch_score >= 80:
            return 0.9
        elif flesch_score >= 70:
            return 0.8
        elif flesch_score >= 60:
            return 0.7
        elif flesch_score >= 50:
            return 0.6
        elif flesch_score >= 30:
            return 0.5
        else:
            return 0.3
    
    def _count_syllables(self, text: str) -> int:
        """Count syllables in text (simplified approach)."""
        text = text.lower()
        count = 0
        vowels = "aeiouy"
        on_vowel = False
        
        for char in text:
            is_vowel = char in vowels
            if is_vowel and not on_vowel:
                count += 1
            on_vowel = is_vowel
        
        return max(1, count)
    
    def _calculate_structure_score(self, text: str) -> float:
        """Calculate content structure score."""
        if not text:
            return 0.0
        
        # Check for headings
        heading_patterns = [r'<h[1-6][^>]*>', r'^#+\s+', r'^\*\*[^*]+\*\*']
        heading_score = 0.0
        
        for pattern in heading_patterns:
            if re.search(pattern, text, re.MULTILINE):
                heading_score += 0.3
        
        # Check for paragraphs
        paragraph_count = len(re.split(r'\n\s*\n', text))
        paragraph_score = min(0.4, paragraph_count * 0.1)
        
        # Check for lists
        list_patterns = [r'^\s*[-*+]\s+', r'^\s*\d+\.\s+']
        list_score = 0.0
        
        for pattern in list_patterns:
            if re.search(pattern, text, re.MULTILINE):
                list_score += 0.2
        
        return min(1.0, heading_score + paragraph_score + list_score)
    
    def calculate_technical_seo_score(self, html_content: str) -> float:
        """Calculate technical SEO score from HTML content."""
        if not html_content:
            return 0.0
        
        scores = {}
        
        # Meta title
        meta_title = re.search(r'<title[^>]*>([^<]+)</title>', html_content, re.IGNORECASE)
        if meta_title:
            title_length = len(meta_title.group(1))
            scores['meta_title'] = 1.0 if 30 <= title_length <= 60 else 0.5
        else:
            scores['meta_title'] = 0.0
        
        # Meta description
        meta_desc = re.search(r'<meta[^>]*name=["\']description["\'][^>]*content=["\']([^"\']+)["\']', html_content, re.IGNORECASE)
        if meta_desc:
            desc_length = len(meta_desc.group(1))
            scores['meta_description'] = 1.0 if 120 <= desc_length <= 160 else 0.5
        else:
            scores['meta_description'] = 0.0
        
        # Headings structure
        headings = re.findall(r'<h([1-6])[^>]*>', html_content, re.IGNORECASE)
        if headings:
            heading_levels = [int(h) for h in headings]
            if heading_levels[0] == 1 and all(h <= prev + 1 for prev, h in zip(heading_levels[:-1], heading_levels[1:])):
                scores['headings'] = 1.0
            else:
                scores['headings'] = 0.5
        else:
            scores['headings'] = 0.0
        
        # Images with alt text
        images = re.findall(r'<img[^>]*>', html_content, re.IGNORECASE)
        alt_images = re.findall(r'<img[^>]*alt=["\'][^"\']+["\'][^>]*>', html_content, re.IGNORECASE)
        
        if images:
            scores['image_alt'] = len(alt_images) / len(images)
        else:
            scores['image_alt'] = 1.0
        
        # Overall technical score
        technical_score = sum(scores.values()) / len(scores)
        return technical_score
    
    def calculate_user_experience_score(self, text: str, html_content: str = "") -> float:
        """Calculate user experience score."""
        if not text:
            return 0.0
        
        scores = {}
        
        # Content engagement (word count, readability)
        scores['engagement'] = self._calculate_readability_score(text)
        
        # Content structure
        scores['structure'] = self._calculate_structure_score(text)
        
        # Mobile-friendly content (simplified check)
        if html_content:
            viewport = re.search(r'<meta[^>]*name=["\']viewport["\']', html_content, re.IGNORECASE)
            scores['mobile'] = 1.0 if viewport else 0.5
        else:
            scores['mobile'] = 0.5
        
        # Overall UX score
        ux_score = sum(scores.values()) / len(scores)
        return ux_score
    
    def calculate_overall_seo_score(self, text: str, html_content: str = "") -> Dict[str, float]:
        """Calculate overall SEO score with component breakdown."""
        scores = {}
        
        # Content quality
        scores['content_quality'] = self.calculate_content_quality_score(text)
        
        # Keyword optimization
        scores['keyword_optimization'] = min(1.0, self._calculate_keyword_density(text) / self.config.keyword_density_threshold)
        
        # Technical SEO
        scores['technical_seo'] = self.calculate_technical_seo_score(html_content)
        
        # Readability
        scores['readability'] = self._calculate_readability_score(text)
        
        # User experience
        scores['user_experience'] = self.calculate_user_experience_score(text, html_content)
        
        # Weighted overall score
        overall_score = sum(
            scores[component] * self.seo_weights[component]
            for component in self.seo_weights.keys()
        )
        
        scores['overall_seo_score'] = overall_score
        
        return scores

class SEOModelEvaluator:
    """Comprehensive SEO model evaluator with appropriate metrics."""
    
    def __init__(self, config: SEOMetricsConfig):
        self.config = config
        self.seo_metrics = SEOSpecificMetrics(config)
        self.metrics_history = []
    
    def evaluate_classification(self, y_true: Union[np.ndarray, torch.Tensor], 
                              y_pred: Union[np.ndarray, torch.Tensor],
                              y_prob: Optional[Union[np.ndarray, torch.Tensor]] = None) -> Dict[str, float]:
        """Evaluate classification model with SEO-appropriate metrics."""
        # Convert to numpy if needed
        if torch.is_tensor(y_true):
            y_true = y_true.cpu().numpy()
        if torch.is_tensor(y_pred):
            y_pred = y_pred.cpu().numpy()
        if y_prob is not None and torch.is_tensor(y_prob):
            y_prob = y_prob.cpu().numpy()
        
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average=self.config.average, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average=self.config.average, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, average=self.config.average, zero_division=0)
        
        # Advanced classification metrics
        if self.config.num_classes == 2 and y_prob is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
            except ValueError:
                metrics['roc_auc'] = 0.0
        
        # Cohen's Kappa for imbalanced datasets
        metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
        
        # Per-class metrics for multi-class
        if self.config.num_classes > 2:
            precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
                y_true, y_pred, average=None, zero_division=0
            )
            
            for i in range(self.config.num_classes):
                metrics[f'precision_class_{i}'] = precision_per_class[i]
                metrics[f'recall_class_{i}'] = recall_per_class[i]
                metrics[f'f1_class_{i}'] = f1_per_class[i]
        
        # SEO-specific classification metrics
        if self.config.use_seo_specific:
            metrics.update(self._calculate_seo_classification_metrics(y_true, y_pred))
        
        return metrics
    
    def evaluate_regression(self, y_true: Union[np.ndarray, torch.Tensor],
                          y_pred: Union[np.ndarray, torch.Tensor]) -> Dict[str, float]:
        """Evaluate regression model with SEO-appropriate metrics."""
        # Convert to numpy if needed
        if torch.is_tensor(y_true):
            y_true = y_true.cpu().numpy()
        if torch.is_tensor(y_pred):
            y_pred = y_pred.cpu().numpy()
        
        metrics = {}
        
        # Basic regression metrics
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['r2'] = r2_score(y_true, y_pred)
        
        # Additional regression metrics
        metrics['mape'] = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-8))) * 100
        metrics['smape'] = 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))) * 100
        
        # SEO-specific regression metrics
        if self.config.use_seo_specific:
            metrics.update(self._calculate_seo_regression_metrics(y_true, y_pred))
        
        return metrics
    
    def evaluate_ranking(self, y_true: Union[np.ndarray, torch.Tensor],
                        y_pred: Union[np.ndarray, torch.Tensor],
                        query_ids: Optional[List[str]] = None) -> Dict[str, float]:
        """Evaluate ranking model with SEO-appropriate metrics."""
        # Convert to numpy if needed
        if torch.is_tensor(y_true):
            y_true = y_true.cpu().numpy()
        if torch.is_tensor(y_pred):
            y_pred = y_pred.cpu().numpy()
        
        metrics = {}
        
        # Ranking metrics
        metrics['ndcg'] = self._calculate_ndcg(y_true, y_pred)
        metrics['map'] = self._calculate_map(y_true, y_pred)
        metrics['mrr'] = self._calculate_mrr(y_true, y_pred)
        
        # SEO-specific ranking metrics
        if self.config.use_seo_specific:
            metrics.update(self._calculate_seo_ranking_metrics(y_true, y_pred, query_ids))
        
        return metrics
    
    def evaluate_clustering(self, features: Union[np.ndarray, torch.Tensor],
                          labels: Union[np.ndarray, torch.Tensor]) -> Dict[str, float]:
        """Evaluate clustering model with SEO-appropriate metrics."""
        # Convert to numpy if needed
        if torch.is_tensor(features):
            features = features.cpu().numpy()
        if torch.is_tensor(labels):
            labels = labels.cpu().numpy()
        
        metrics = {}
        
        # Clustering metrics
        try:
            metrics['silhouette'] = silhouette_score(features, labels)
        except ValueError:
            metrics['silhouette'] = 0.0
        
        try:
            metrics['calinski_harabasz'] = calinski_harabasz_score(features, labels)
        except ValueError:
            metrics['calinski_harabasz'] = 0.0
        
        # SEO-specific clustering metrics
        if self.config.use_seo_specific:
            metrics.update(self._calculate_seo_clustering_metrics(features, labels))
        
        return metrics
    
    def evaluate_seo_content(self, text: str, html_content: str = "") -> Dict[str, float]:
        """Evaluate SEO content quality."""
        return self.seo_metrics.calculate_overall_seo_score(text, html_content)
    
    def _calculate_seo_classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate SEO-specific classification metrics."""
        metrics = {}
        
        # SEO relevance metrics
        true_positives = np.sum((y_true == 1) & (y_pred == 1))
        false_positives = np.sum((y_true == 0) & (y_pred == 1))
        false_negatives = np.sum((y_true == 1) & (y_pred == 0))
        
        # SEO precision (how many predicted SEO-optimized are actually good)
        if true_positives + false_positives > 0:
            metrics['seo_precision'] = true_positives / (true_positives + false_positives)
        else:
            metrics['seo_precision'] = 0.0
        
        # SEO recall (how many actual SEO-optimized were detected)
        if true_positives + false_negatives > 0:
            metrics['seo_recall'] = true_positives / (true_positives + false_negatives)
        else:
            metrics['seo_recall'] = 0.0
        
        # SEO F1 score
        if metrics['seo_precision'] + metrics['seo_recall'] > 0:
            metrics['seo_f1'] = 2 * (metrics['seo_precision'] * metrics['seo_recall']) / (metrics['seo_precision'] + metrics['seo_recall'])
        else:
            metrics['seo_f1'] = 0.0
        
        return metrics
    
    def _calculate_seo_regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate SEO-specific regression metrics."""
        metrics = {}
        
        # SEO score accuracy within threshold
        seo_threshold = self.config.seo_score_threshold
        within_threshold = np.abs(y_true - y_pred) <= seo_threshold
        
        metrics['seo_accuracy_within_threshold'] = np.mean(within_threshold)
        
        # High-quality content detection accuracy
        high_quality_true = y_true >= seo_threshold
        high_quality_pred = y_pred >= seo_threshold
        
        metrics['high_quality_detection_accuracy'] = np.mean(high_quality_true == high_quality_pred)
        
        return metrics
    
    def _calculate_seo_ranking_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                     query_ids: Optional[List[str]] = None) -> Dict[str, float]:
        """Calculate SEO-specific ranking metrics."""
        metrics = {}
        
        # Top-k SEO relevance
        k_values = [1, 3, 5, 10]
        
        for k in k_values:
            if len(y_pred) >= k:
                top_k_indices = np.argsort(y_pred)[-k:]
                top_k_relevance = y_true[top_k_indices]
                metrics[f'top_{k}_seo_relevance'] = np.mean(top_k_relevance)
        
        return metrics
    
    def _calculate_seo_clustering_metrics(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Calculate SEO-specific clustering metrics."""
        metrics = {}
        
        # Content diversity within clusters
        unique_labels = np.unique(labels)
        diversity_scores = []
        
        for label in unique_labels:
            cluster_features = features[labels == label]
            if len(cluster_features) > 1:
                # Calculate average distance within cluster
                distances = []
                for i in range(len(cluster_features)):
                    for j in range(i + 1, len(cluster_features)):
                        dist = np.linalg.norm(cluster_features[i] - cluster_features[j])
                        distances.append(dist)
                
                if distances:
                    diversity_scores.append(np.mean(distances))
        
        if diversity_scores:
            metrics['content_diversity'] = np.mean(diversity_scores)
        else:
            metrics['content_diversity'] = 0.0
        
        return metrics
    
    def _calculate_ndcg(self, y_true: np.ndarray, y_pred: np.ndarray, k: int = 10) -> float:
        """Calculate Normalized Discounted Cumulative Gain."""
        if len(y_pred) < k:
            k = len(y_pred)
        
        # Sort by predicted scores
        sorted_indices = np.argsort(y_pred)[::-1][:k]
        sorted_relevance = y_true[sorted_indices]
        
        # Calculate DCG
        dcg = np.sum(sorted_relevance / np.log2(np.arange(2, k + 2)))
        
        # Calculate IDCG (ideal DCG)
        ideal_relevance = np.sort(y_true)[::-1][:k]
        idcg = np.sum(ideal_relevance / np.log2(np.arange(2, k + 2)))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def _calculate_map(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Average Precision."""
        # Sort by predicted scores
        sorted_indices = np.argsort(y_pred)[::-1]
        sorted_relevance = y_true[sorted_indices]
        
        # Calculate AP
        relevant_count = 0
        precision_sum = 0.0
        
        for i, relevance in enumerate(sorted_relevance):
            if relevance > 0:
                relevant_count += 1
                precision = relevant_count / (i + 1)
                precision_sum += precision
        
        return precision_sum / np.sum(y_true > 0) if np.sum(y_true > 0) > 0 else 0.0
    
    def _calculate_mrr(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Reciprocal Rank."""
        # Sort by predicted scores
        sorted_indices = np.argsort(y_pred)[::-1]
        sorted_relevance = y_true[sorted_indices]
        
        # Find first relevant item
        for i, relevance in enumerate(sorted_relevance):
            if relevance > 0:
                return 1.0 / (i + 1)
        
        return 0.0
    
    def generate_evaluation_report(self, metrics: Dict[str, float], 
                                 task_name: str = "SEO Evaluation") -> str:
        """Generate comprehensive evaluation report."""
        report = f"📊 {task_name} - Evaluation Report\n"
        report += "=" * 50 + "\n\n"
        
        # Group metrics by category
        basic_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        seo_metrics = [k for k in metrics.keys() if k.startswith('seo_')]
        ranking_metrics = ['ndcg', 'map', 'mrr']
        clustering_metrics = ['silhouette', 'calinski_harabasz']
        
        # Basic metrics
        if any(metric in metrics for metric in basic_metrics):
            report += "🎯 Basic Metrics:\n"
            for metric in basic_metrics:
                if metric in metrics:
                    report += f"  {metric.replace('_', ' ').title()}: {metrics[metric]:.4f}\n"
            report += "\n"
        
        # SEO-specific metrics
        if seo_metrics:
            report += "🔍 SEO-Specific Metrics:\n"
            for metric in seo_metrics:
                report += f"  {metric.replace('_', ' ').title()}: {metrics[metric]:.4f}\n"
            report += "\n"
        
        # Ranking metrics
        if any(metric in metrics for metric in ranking_metrics):
            report += "📈 Ranking Metrics:\n"
            for metric in ranking_metrics:
                if metric in metrics:
                    report += f"  {metric.upper()}: {metrics[metric]:.4f}\n"
            report += "\n"
        
        # Clustering metrics
        if any(metric in metrics for metric in clustering_metrics):
            report += "🔗 Clustering Metrics:\n"
            for metric in clustering_metrics:
                if metric in metrics:
                    report += f"  {metric.replace('_', ' ').title()}: {metrics[metric]:.4f}\n"
            report += "\n"
        
        # Summary
        if 'overall_seo_score' in metrics:
            report += f"🏆 Overall SEO Score: {metrics['overall_seo_score']:.4f}\n"
        
        return report
    
    def save_metrics(self, metrics: Dict[str, float], filename: str = "seo_metrics.json"):
        """Save metrics to file."""
        import json
        with open(filename, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def load_metrics(self, filename: str = "seo_metrics.json") -> Dict[str, float]:
        """Load metrics from file."""
        import json
        with open(filename, 'r') as f:
            return json.load(f)

# Usage example
def main():
    # Configuration
    config = SEOMetricsConfig(
        task_type="classification",
        num_classes=2,
        average="weighted",
        use_seo_specific=True
    )
    
    # Initialize evaluator
    evaluator = SEOModelEvaluator(config)
    
    # Example data
    y_true = np.array([1, 0, 1, 1, 0, 1, 0, 1])
    y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 1])
    y_prob = np.array([0.9, 0.1, 0.8, 0.4, 0.2, 0.7, 0.6, 0.9])
    
    # Evaluate classification
    classification_metrics = evaluator.evaluate_classification(y_true, y_pred, y_prob)
    
    # Evaluate SEO content
    sample_text = """
    <h1>SEO Optimization Guide</h1>
    <p>Learn about search engine optimization techniques to improve your website's ranking.</p>
    <h2>Key Strategies</h2>
    <ul>
        <li>Keyword research and optimization</li>
        <li>Content quality and relevance</li>
        <li>Technical SEO implementation</li>
    </ul>
    """
    
    seo_metrics = evaluator.evaluate_seo_content(sample_text, sample_text)
    
    # Generate report
    report = evaluator.generate_evaluation_report(classification_metrics, "SEO Classification")
    print(report)
    
    # Print SEO metrics
    print("🔍 SEO Content Metrics:")
    for key, value in seo_metrics.items():
        print(f"  {key.replace('_', ' ').title()}: {value:.4f}")

if __name__ == "__main__":
    main()
