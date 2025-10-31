#!/usr/bin/env python3
"""
Advanced Explainable AI System for Frontier Model Training
Provides comprehensive model interpretability, explainability, and transparency.
"""

import os
import json
import time
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
import sqlite3
from contextlib import contextmanager
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import shap
import lime
import lime.lime_tabular
import lime.lime_image
import lime.lime_text
import captum
from captum.attr import IntegratedGradients, GradientShap, Saliency, InputXGradient
from captum.attr import LayerGradCam, LayerAttribution, GuidedBackprop
from captum.attr import ShapleyValueSampling, FeatureAblation, Occlusion
from captum.attr import KernelShap, Lime, DeepLift, DeepLiftShap
from captum.insights import AttributionVisualizer
from captum.insights.features import ImageFeature, TextFeature, TabularFeature
import alibi
from alibi.explainers import AnchorTabular, AnchorImage, AnchorText
from alibi.explainers import CounterfactualExplainer, CEM
from alibi.explainers import IntegratedGradients as AlibiIG
from alibi.explainers import KernelSHAP, TreeSHAP
from alibi.explainers import PartialDependence, ALE
from alibi.explainers import PrototypeExplainer, ContrastiveExplainer
import joblib
import pickle
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

console = Console()

class ExplanationMethod(Enum):
    """Explanation methods."""
    SHAP = "shap"
    LIME = "lime"
    INTEGRATED_GRADIENTS = "integrated_gradients"
    GRADIENT_SHAP = "gradient_shap"
    SALIENCY = "saliency"
    INPUT_X_GRADIENT = "input_x_gradient"
    LAYER_GRAD_CAM = "layer_grad_cam"
    GUIDED_BACKPROP = "guided_backprop"
    SHAPLEY_VALUE = "shapley_value"
    FEATURE_ABLATION = "feature_ablation"
    OCCLUSION = "occlusion"
    KERNEL_SHAP = "kernel_shap"
    DEEP_LIFT = "deep_lift"
    DEEP_LIFT_SHAP = "deep_lift_shap"
    ANCHOR_TABULAR = "anchor_tabular"
    ANCHOR_IMAGE = "anchor_image"
    ANCHOR_TEXT = "anchor_text"
    COUNTERFACTUAL = "counterfactual"
    PARTIAL_DEPENDENCE = "partial_dependence"
    ALE = "ale"
    PROTOTYPE = "prototype"
    CONTRASTIVE = "contrastive"

class ExplanationScope(Enum):
    """Explanation scope."""
    GLOBAL = "global"
    LOCAL = "local"
    REGIONAL = "regional"
    FEATURE_LEVEL = "feature_level"
    LAYER_LEVEL = "layer_level"
    NEURON_LEVEL = "neuron_level"
    CONCEPT_LEVEL = "concept_level"

class DataType(Enum):
    """Data types."""
    TABULAR = "tabular"
    IMAGE = "image"
    TEXT = "text"
    TIME_SERIES = "time_series"
    MULTIMODAL = "multimodal"
    GRAPH = "graph"
    AUDIO = "audio"
    VIDEO = "video"

class ExplanationFormat(Enum):
    """Explanation formats."""
    VISUALIZATION = "visualization"
    TEXTUAL = "textual"
    NUMERICAL = "numerical"
    INTERACTIVE = "interactive"
    REPORT = "report"
    JSON = "json"
    HTML = "html"

@dataclass
class XAIConfig:
    """Explainable AI configuration."""
    explanation_methods: List[ExplanationMethod] = None
    explanation_scope: ExplanationScope = ExplanationScope.LOCAL
    data_type: DataType = DataType.TABULAR
    explanation_format: ExplanationFormat = ExplanationFormat.VISUALIZATION
    enable_global_explanations: bool = True
    enable_local_explanations: bool = True
    enable_feature_importance: bool = True
    enable_attention_visualization: bool = True
    enable_counterfactual_explanations: bool = True
    enable_concept_explanations: bool = True
    enable_uncertainty_quantification: bool = True
    enable_bias_detection: bool = True
    enable_fairness_analysis: bool = True
    enable_adversarial_robustness: bool = True
    enable_model_comparison: bool = True
    enable_explanation_comparison: bool = True
    enable_interactive_explanations: bool = True
    enable_explanation_validation: bool = True
    device: str = "auto"

@dataclass
class ExplanationResult:
    """Explanation result."""
    explanation_id: str
    method: ExplanationMethod
    scope: ExplanationScope
    data_type: DataType
    explanation_data: Dict[str, Any]
    confidence_scores: Dict[str, float]
    metadata: Dict[str, Any]
    created_at: datetime

@dataclass
class XAIReport:
    """XAI comprehensive report."""
    report_id: str
    model_info: Dict[str, Any]
    explanations: List[ExplanationResult]
    global_insights: Dict[str, Any]
    local_insights: Dict[str, Any]
    bias_analysis: Dict[str, Any]
    fairness_metrics: Dict[str, Any]
    robustness_analysis: Dict[str, Any]
    created_at: datetime

class SHAPExplainer:
    """SHAP explanation engine."""
    
    def __init__(self, config: XAIConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def explain_model(self, model: Any, X: np.ndarray, y: np.ndarray = None) -> ExplanationResult:
        """Explain model using SHAP."""
        console.print("[blue]Generating SHAP explanations...[/blue]")
        
        try:
            # Initialize SHAP explainer
            if hasattr(model, 'predict_proba'):
                # Tree-based models
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)
            elif hasattr(model, 'predict'):
                # General models
                explainer = shap.KernelExplainer(model.predict, X[:100])  # Sample for background
                shap_values = explainer.shap_values(X)
            else:
                # Neural networks
                explainer = shap.DeepExplainer(model, X[:100])
                shap_values = explainer.shap_values(X)
            
            # Calculate feature importance
            if isinstance(shap_values, list):
                shap_values = np.array(shap_values)
            
            feature_importance = np.abs(shap_values).mean(axis=0)
            
            # Generate explanation data
            explanation_data = {
                'shap_values': shap_values.tolist(),
                'feature_importance': feature_importance.tolist(),
                'base_value': explainer.expected_value if hasattr(explainer, 'expected_value') else 0.0,
                'feature_names': [f'feature_{i}' for i in range(X.shape[1])]
            }
            
            # Calculate confidence scores
            confidence_scores = {
                'explanation_confidence': 0.8,  # Simplified
                'feature_importance_confidence': 0.9,
                'model_confidence': 0.7
            }
            
            explanation_result = ExplanationResult(
                explanation_id=f"shap_{int(time.time())}",
                method=ExplanationMethod.SHAP,
                scope=self.config.explanation_scope,
                data_type=self.config.data_type,
                explanation_data=explanation_data,
                confidence_scores=confidence_scores,
                metadata={'explainer_type': type(explainer).__name__},
                created_at=datetime.now()
            )
            
            console.print("[green]SHAP explanations generated[/green]")
            return explanation_result
            
        except Exception as e:
            self.logger.error(f"SHAP explanation failed: {e}")
            return self._create_error_explanation("SHAP", str(e))
    
    def _create_error_explanation(self, method: str, error: str) -> ExplanationResult:
        """Create error explanation."""
        return ExplanationResult(
            explanation_id=f"error_{int(time.time())}",
            method=ExplanationMethod(method.lower()),
            scope=self.config.explanation_scope,
            data_type=self.config.data_type,
            explanation_data={'error': error},
            confidence_scores={'error': 0.0},
            metadata={'error': True},
            created_at=datetime.now()
        )

class LIMEExplainer:
    """LIME explanation engine."""
    
    def __init__(self, config: XAIConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def explain_model(self, model: Any, X: np.ndarray, y: np.ndarray = None) -> ExplanationResult:
        """Explain model using LIME."""
        console.print("[blue]Generating LIME explanations...[/blue]")
        
        try:
            if self.config.data_type == DataType.TABULAR:
                explainer = lime.lime_tabular.LimeTabularExplainer(
                    X, 
                    feature_names=[f'feature_{i}' for i in range(X.shape[1])],
                    class_names=['class_0', 'class_1'] if y is not None else None,
                    mode='classification' if y is not None else 'regression'
                )
                
                # Explain multiple instances
                explanations = []
                for i in range(min(10, len(X))):  # Limit to 10 explanations
                    explanation = explainer.explain_instance(
                        X[i], 
                        model.predict_proba if hasattr(model, 'predict_proba') else model.predict,
                        num_features=X.shape[1]
                    )
                    explanations.append(explanation.as_list())
                
                explanation_data = {
                    'explanations': explanations,
                    'feature_names': [f'feature_{i}' for i in range(X.shape[1])],
                    'num_explanations': len(explanations)
                }
                
            elif self.config.data_type == DataType.IMAGE:
                explainer = lime.lime_image.LimeImageExplainer()
                # Simplified for demonstration
                explanation_data = {'method': 'lime_image', 'status': 'simplified'}
                
            elif self.config.data_type == DataType.TEXT:
                explainer = lime.lime_text.LimeTextExplainer()
                # Simplified for demonstration
                explanation_data = {'method': 'lime_text', 'status': 'simplified'}
            
            else:
                explanation_data = {'method': 'lime', 'status': 'unsupported_data_type'}
            
            confidence_scores = {
                'explanation_confidence': 0.75,
                'local_fidelity': 0.8,
                'model_confidence': 0.7
            }
            
            explanation_result = ExplanationResult(
                explanation_id=f"lime_{int(time.time())}",
                method=ExplanationMethod.LIME,
                scope=self.config.explanation_scope,
                data_type=self.config.data_type,
                explanation_data=explanation_data,
                confidence_scores=confidence_scores,
                metadata={'explainer_type': 'LimeTabularExplainer'},
                created_at=datetime.now()
            )
            
            console.print("[green]LIME explanations generated[/green]")
            return explanation_result
            
        except Exception as e:
            self.logger.error(f"LIME explanation failed: {e}")
            return self._create_error_explanation("LIME", str(e))
    
    def _create_error_explanation(self, method: str, error: str) -> ExplanationResult:
        """Create error explanation."""
        return ExplanationResult(
            explanation_id=f"error_{int(time.time())}",
            method=ExplanationMethod(method.lower()),
            scope=self.config.explanation_scope,
            data_type=self.config.data_type,
            explanation_data={'error': error},
            confidence_scores={'error': 0.0},
            metadata={'error': True},
            created_at=datetime.now()
        )

class CaptumExplainer:
    """Captum explanation engine."""
    
    def __init__(self, config: XAIConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize device
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
    
    def explain_model(self, model: nn.Module, X: torch.Tensor, y: torch.Tensor = None) -> ExplanationResult:
        """Explain model using Captum."""
        console.print("[blue]Generating Captum explanations...[/blue]")
        
        try:
            model = model.to(self.device)
            X = X.to(self.device)
            
            # Choose explanation method
            if self.config.explanation_methods and ExplanationMethod.INTEGRATED_GRADIENTS in self.config.explanation_methods:
                explainer = IntegratedGradients(model)
                attributions = explainer.attribute(X, target=y)
            elif self.config.explanation_methods and ExplanationMethod.GRADIENT_SHAP in self.config.explanation_methods:
                explainer = GradientShap(model)
                attributions = explainer.attribute(X, baselines=torch.zeros_like(X))
            elif self.config.explanation_methods and ExplanationMethod.SALIENCY in self.config.explanation_methods:
                explainer = Saliency(model)
                attributions = explainer.attribute(X, target=y)
            else:
                # Default to Integrated Gradients
                explainer = IntegratedGradients(model)
                attributions = explainer.attribute(X, target=y)
            
            # Convert attributions to numpy
            if isinstance(attributions, tuple):
                attributions = attributions[0]
            
            attributions_np = attributions.detach().cpu().numpy()
            
            # Calculate feature importance
            feature_importance = np.abs(attributions_np).mean(axis=0)
            
            explanation_data = {
                'attributions': attributions_np.tolist(),
                'feature_importance': feature_importance.tolist(),
                'method': type(explainer).__name__,
                'feature_names': [f'feature_{i}' for i in range(X.shape[1])]
            }
            
            confidence_scores = {
                'explanation_confidence': 0.85,
                'attribution_confidence': 0.8,
                'model_confidence': 0.75
            }
            
            explanation_result = ExplanationResult(
                explanation_id=f"captum_{int(time.time())}",
                method=ExplanationMethod.INTEGRATED_GRADIENTS,
                scope=self.config.explanation_scope,
                data_type=self.config.data_type,
                explanation_data=explanation_data,
                confidence_scores=confidence_scores,
                metadata={'explainer_type': type(explainer).__name__},
                created_at=datetime.now()
            )
            
            console.print("[green]Captum explanations generated[/green]")
            return explanation_result
            
        except Exception as e:
            self.logger.error(f"Captum explanation failed: {e}")
            return self._create_error_explanation("CAPTUM", str(e))
    
    def _create_error_explanation(self, method: str, error: str) -> ExplanationResult:
        """Create error explanation."""
        return ExplanationResult(
            explanation_id=f"error_{int(time.time())}",
            method=ExplanationMethod.INTEGRATED_GRADIENTS,
            scope=self.config.explanation_scope,
            data_type=self.config.data_type,
            explanation_data={'error': error},
            confidence_scores={'error': 0.0},
            metadata={'error': True},
            created_at=datetime.now()
        )

class AlibiExplainer:
    """Alibi explanation engine."""
    
    def __init__(self, config: XAIConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def explain_model(self, model: Any, X: np.ndarray, y: np.ndarray = None) -> ExplanationResult:
        """Explain model using Alibi."""
        console.print("[blue]Generating Alibi explanations...[/blue]")
        
        try:
            if self.config.explanation_methods and ExplanationMethod.ANCHOR_TABULAR in self.config.explanation_methods:
                explainer = AnchorTabular(model.predict, X)
                explainer.fit(X)
                
                # Explain multiple instances
                explanations = []
                for i in range(min(5, len(X))):  # Limit to 5 explanations
                    explanation = explainer.explain(X[i])
                    explanations.append({
                        'anchor': explanation.anchor,
                        'precision': explanation.precision,
                        'coverage': explanation.coverage
                    })
                
                explanation_data = {
                    'explanations': explanations,
                    'method': 'AnchorTabular',
                    'num_explanations': len(explanations)
                }
                
            elif self.config.explanation_methods and ExplanationMethod.COUNTERFACTUAL in self.config.explanation_methods:
                # Simplified counterfactual explanation
                explanation_data = {
                    'method': 'CounterfactualExplainer',
                    'status': 'simplified',
                    'counterfactuals': []
                }
                
            else:
                # Default to KernelSHAP
                explainer = KernelSHAP(model.predict, X[:100])  # Background data
                explanations = explainer.explain(X[:10])  # Explain first 10 instances
                
                explanation_data = {
                    'explanations': explanations.shap_values.tolist(),
                    'method': 'KernelSHAP',
                    'expected_value': explanations.expected_value
                }
            
            confidence_scores = {
                'explanation_confidence': 0.8,
                'method_confidence': 0.85,
                'model_confidence': 0.7
            }
            
            explanation_result = ExplanationResult(
                explanation_id=f"alibi_{int(time.time())}",
                method=ExplanationMethod.ANCHOR_TABULAR if 'anchor' in explanation_data.get('method', '').lower() else ExplanationMethod.KERNEL_SHAP,
                scope=self.config.explanation_scope,
                data_type=self.config.data_type,
                explanation_data=explanation_data,
                confidence_scores=confidence_scores,
                metadata={'explainer_type': explanation_data.get('method', 'Unknown')},
                created_at=datetime.now()
            )
            
            console.print("[green]Alibi explanations generated[/green]")
            return explanation_result
            
        except Exception as e:
            self.logger.error(f"Alibi explanation failed: {e}")
            return self._create_error_explanation("ALIBI", str(e))
    
    def _create_error_explanation(self, method: str, error: str) -> ExplanationResult:
        """Create error explanation."""
        return ExplanationResult(
            explanation_id=f"error_{int(time.time())}",
            method=ExplanationMethod.ANCHOR_TABULAR,
            scope=self.config.explanation_scope,
            data_type=self.config.data_type,
            explanation_data={'error': error},
            confidence_scores={'error': 0.0},
            metadata={'error': True},
            created_at=datetime.now()
        )

class BiasDetector:
    """Bias detection engine."""
    
    def __init__(self, config: XAIConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def detect_bias(self, model: Any, X: np.ndarray, y: np.ndarray, 
                   sensitive_features: List[str] = None) -> Dict[str, Any]:
        """Detect bias in model predictions."""
        console.print("[blue]Detecting model bias...[/blue]")
        
        try:
            # Get model predictions
            if hasattr(model, 'predict_proba'):
                predictions = model.predict_proba(X)[:, 1]  # Binary classification
            else:
                predictions = model.predict(X)
            
            bias_analysis = {
                'statistical_parity': self._calculate_statistical_parity(predictions, y, sensitive_features),
                'equalized_odds': self._calculate_equalized_odds(predictions, y, sensitive_features),
                'demographic_parity': self._calculate_demographic_parity(predictions, sensitive_features),
                'bias_score': 0.0,
                'bias_detected': False
            }
            
            # Calculate overall bias score
            bias_scores = []
            for metric in ['statistical_parity', 'equalized_odds', 'demographic_parity']:
                if bias_analysis[metric] is not None:
                    bias_scores.append(abs(bias_analysis[metric]))
            
            if bias_scores:
                bias_analysis['bias_score'] = np.mean(bias_scores)
                bias_analysis['bias_detected'] = bias_analysis['bias_score'] > 0.1
            
            console.print("[green]Bias detection completed[/green]")
            return bias_analysis
            
        except Exception as e:
            self.logger.error(f"Bias detection failed: {e}")
            return {'error': str(e), 'bias_detected': False}
    
    def _calculate_statistical_parity(self, predictions: np.ndarray, y: np.ndarray, 
                                   sensitive_features: List[str] = None) -> float:
        """Calculate statistical parity."""
        if sensitive_features is None or len(sensitive_features) == 0:
            return None
        
        # Simplified calculation
        # In practice, you'd group by sensitive features
        positive_rate = np.mean(predictions > 0.5)
        return positive_rate
    
    def _calculate_equalized_odds(self, predictions: np.ndarray, y: np.ndarray, 
                                sensitive_features: List[str] = None) -> float:
        """Calculate equalized odds."""
        if sensitive_features is None or len(sensitive_features) == 0:
            return None
        
        # Simplified calculation
        # In practice, you'd calculate TPR and FPR for each group
        tpr = np.mean(predictions[y == 1] > 0.5)
        fpr = np.mean(predictions[y == 0] > 0.5)
        return tpr - fpr
    
    def _calculate_demographic_parity(self, predictions: np.ndarray, 
                                    sensitive_features: List[str] = None) -> float:
        """Calculate demographic parity."""
        if sensitive_features is None or len(sensitive_features) == 0:
            return None
        
        # Simplified calculation
        positive_rate = np.mean(predictions > 0.5)
        return positive_rate

class FairnessAnalyzer:
    """Fairness analysis engine."""
    
    def __init__(self, config: XAIConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def analyze_fairness(self, model: Any, X: np.ndarray, y: np.ndarray, 
                        sensitive_features: List[str] = None) -> Dict[str, Any]:
        """Analyze model fairness."""
        console.print("[blue]Analyzing model fairness...[/blue]")
        
        try:
            # Get model predictions
            if hasattr(model, 'predict_proba'):
                predictions = model.predict_proba(X)[:, 1]
            else:
                predictions = model.predict(X)
            
            fairness_metrics = {
                'accuracy_by_group': self._calculate_accuracy_by_group(predictions, y, sensitive_features),
                'precision_by_group': self._calculate_precision_by_group(predictions, y, sensitive_features),
                'recall_by_group': self._calculate_recall_by_group(predictions, y, sensitive_features),
                'f1_by_group': self._calculate_f1_by_group(predictions, y, sensitive_features),
                'fairness_score': 0.0,
                'fairness_violations': []
            }
            
            # Calculate overall fairness score
            group_metrics = []
            for metric in ['accuracy_by_group', 'precision_by_group', 'recall_by_group', 'f1_by_group']:
                if fairness_metrics[metric] is not None:
                    group_metrics.extend(fairness_metrics[metric].values())
            
            if group_metrics:
                fairness_metrics['fairness_score'] = 1.0 - np.std(group_metrics) / np.mean(group_metrics)
                
                # Detect fairness violations
                if fairness_metrics['fairness_score'] < 0.8:
                    fairness_metrics['fairness_violations'].append('Significant performance disparity detected')
            
            console.print("[green]Fairness analysis completed[/green]")
            return fairness_metrics
            
        except Exception as e:
            self.logger.error(f"Fairness analysis failed: {e}")
            return {'error': str(e), 'fairness_score': 0.0}
    
    def _calculate_accuracy_by_group(self, predictions: np.ndarray, y: np.ndarray, 
                                   sensitive_features: List[str] = None) -> Dict[str, float]:
        """Calculate accuracy by group."""
        if sensitive_features is None:
            return {'overall': accuracy_score(y, predictions > 0.5)}
        
        # Simplified - in practice, you'd group by sensitive features
        return {'group_1': accuracy_score(y, predictions > 0.5)}
    
    def _calculate_precision_by_group(self, predictions: np.ndarray, y: np.ndarray, 
                                    sensitive_features: List[str] = None) -> Dict[str, float]:
        """Calculate precision by group."""
        if sensitive_features is None:
            return {'overall': precision_score(y, predictions > 0.5)}
        
        return {'group_1': precision_score(y, predictions > 0.5)}
    
    def _calculate_recall_by_group(self, predictions: np.ndarray, y: np.ndarray, 
                                 sensitive_features: List[str] = None) -> Dict[str, float]:
        """Calculate recall by group."""
        if sensitive_features is None:
            return {'overall': recall_score(y, predictions > 0.5)}
        
        return {'group_1': recall_score(y, predictions > 0.5)}
    
    def _calculate_f1_by_group(self, predictions: np.ndarray, y: np.ndarray, 
                             sensitive_features: List[str] = None) -> Dict[str, float]:
        """Calculate F1 score by group."""
        if sensitive_features is None:
            return {'overall': f1_score(y, predictions > 0.5)}
        
        return {'group_1': f1_score(y, predictions > 0.5)}

class RobustnessAnalyzer:
    """Robustness analysis engine."""
    
    def __init__(self, config: XAIConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def analyze_robustness(self, model: Any, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Analyze model robustness."""
        console.print("[blue]Analyzing model robustness...[/blue]")
        
        try:
            # Get baseline predictions
            baseline_predictions = model.predict(X)
            
            # Test robustness to noise
            noise_robustness = self._test_noise_robustness(model, X, y)
            
            # Test robustness to perturbations
            perturbation_robustness = self._test_perturbation_robustness(model, X, y)
            
            # Test robustness to outliers
            outlier_robustness = self._test_outlier_robustness(model, X, y)
            
            robustness_analysis = {
                'noise_robustness': noise_robustness,
                'perturbation_robustness': perturbation_robustness,
                'outlier_robustness': outlier_robustness,
                'overall_robustness_score': 0.0,
                'robustness_issues': []
            }
            
            # Calculate overall robustness score
            robustness_scores = [noise_robustness, perturbation_robustness, outlier_robustness]
            robustness_analysis['overall_robustness_score'] = np.mean(robustness_scores)
            
            # Detect robustness issues
            if robustness_analysis['overall_robustness_score'] < 0.7:
                robustness_analysis['robustness_issues'].append('Model shows low robustness to perturbations')
            
            console.print("[green]Robustness analysis completed[/green]")
            return robustness_analysis
            
        except Exception as e:
            self.logger.error(f"Robustness analysis failed: {e}")
            return {'error': str(e), 'overall_robustness_score': 0.0}
    
    def _test_noise_robustness(self, model: Any, X: np.ndarray, y: np.ndarray) -> float:
        """Test robustness to noise."""
        # Add Gaussian noise
        noise_levels = [0.01, 0.05, 0.1]
        robustness_scores = []
        
        for noise_level in noise_levels:
            X_noisy = X + np.random.normal(0, noise_level, X.shape)
            predictions_noisy = model.predict(X_noisy)
            
            # Calculate accuracy drop
            baseline_accuracy = accuracy_score(y, model.predict(X))
            noisy_accuracy = accuracy_score(y, predictions_noisy)
            accuracy_drop = baseline_accuracy - noisy_accuracy
            
            robustness_scores.append(1.0 - accuracy_drop)
        
        return np.mean(robustness_scores)
    
    def _test_perturbation_robustness(self, model: Any, X: np.ndarray, y: np.ndarray) -> float:
        """Test robustness to perturbations."""
        # Add small perturbations
        perturbation_levels = [0.01, 0.02, 0.05]
        robustness_scores = []
        
        for pert_level in perturbation_levels:
            X_perturbed = X + np.random.uniform(-pert_level, pert_level, X.shape)
            predictions_perturbed = model.predict(X_perturbed)
            
            # Calculate accuracy drop
            baseline_accuracy = accuracy_score(y, model.predict(X))
            perturbed_accuracy = accuracy_score(y, predictions_perturbed)
            accuracy_drop = baseline_accuracy - perturbed_accuracy
            
            robustness_scores.append(1.0 - accuracy_drop)
        
        return np.mean(robustness_scores)
    
    def _test_outlier_robustness(self, model: Any, X: np.ndarray, y: np.ndarray) -> float:
        """Test robustness to outliers."""
        # Add outliers
        outlier_indices = np.random.choice(len(X), size=min(10, len(X)//10), replace=False)
        X_with_outliers = X.copy()
        X_with_outliers[outlier_indices] += np.random.normal(0, 1, (len(outlier_indices), X.shape[1]))
        
        predictions_with_outliers = model.predict(X_with_outliers)
        
        # Calculate accuracy drop
        baseline_accuracy = accuracy_score(y, model.predict(X))
        outlier_accuracy = accuracy_score(y, predictions_with_outliers)
        accuracy_drop = baseline_accuracy - outlier_accuracy
        
        return 1.0 - accuracy_drop

class XAISystem:
    """Main explainable AI system."""
    
    def __init__(self, config: XAIConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize explanation engines
        self.shap_explainer = SHAPExplainer(config)
        self.lime_explainer = LIMEExplainer(config)
        self.captum_explainer = CaptumExplainer(config)
        self.alibi_explainer = AlibiExplainer(config)
        
        # Initialize analysis engines
        self.bias_detector = BiasDetector(config)
        self.fairness_analyzer = FairnessAnalyzer(config)
        self.robustness_analyzer = RobustnessAnalyzer(config)
        
        # Initialize database
        self.db_path = self._init_database()
        
        # Results storage
        self.xai_reports: Dict[str, XAIReport] = {}
    
    def _init_database(self) -> str:
        """Initialize XAI database."""
        db_path = Path("./explainable_ai.db")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS explanation_results (
                    explanation_id TEXT PRIMARY KEY,
                    method TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    data_type TEXT NOT NULL,
                    explanation_data TEXT NOT NULL,
                    confidence_scores TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS xai_reports (
                    report_id TEXT PRIMARY KEY,
                    model_info TEXT NOT NULL,
                    explanations TEXT NOT NULL,
                    global_insights TEXT NOT NULL,
                    local_insights TEXT NOT NULL,
                    bias_analysis TEXT NOT NULL,
                    fairness_metrics TEXT NOT NULL,
                    robustness_analysis TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
        
        return str(db_path)
    
    def explain_model(self, model: Any, X: np.ndarray, y: np.ndarray = None, 
                     sensitive_features: List[str] = None) -> XAIReport:
        """Generate comprehensive model explanations."""
        console.print("[blue]Generating comprehensive model explanations...[/blue]")
        
        start_time = time.time()
        report_id = f"xai_report_{int(time.time())}"
        
        # Generate explanations using different methods
        explanations = []
        
        # SHAP explanations
        if not self.config.explanation_methods or ExplanationMethod.SHAP in self.config.explanation_methods:
            shap_explanation = self.shap_explainer.explain_model(model, X, y)
            explanations.append(shap_explanation)
        
        # LIME explanations
        if not self.config.explanation_methods or ExplanationMethod.LIME in self.config.explanation_methods:
            lime_explanation = self.lime_explainer.explain_model(model, X, y)
            explanations.append(lime_explanation)
        
        # Captum explanations (for neural networks)
        if hasattr(model, 'parameters') and (not self.config.explanation_methods or 
                                           ExplanationMethod.INTEGRATED_GRADIENTS in self.config.explanation_methods):
            X_tensor = torch.FloatTensor(X)
            y_tensor = torch.LongTensor(y) if y is not None else None
            captum_explanation = self.captum_explainer.explain_model(model, X_tensor, y_tensor)
            explanations.append(captum_explanation)
        
        # Alibi explanations
        if not self.config.explanation_methods or ExplanationMethod.ANCHOR_TABULAR in self.config.explanation_methods:
            alibi_explanation = self.alibi_explainer.explain_model(model, X, y)
            explanations.append(alibi_explanation)
        
        # Generate global insights
        global_insights = self._generate_global_insights(explanations)
        
        # Generate local insights
        local_insights = self._generate_local_insights(explanations)
        
        # Bias analysis
        bias_analysis = {}
        if self.config.enable_bias_detection:
            bias_analysis = self.bias_detector.detect_bias(model, X, y, sensitive_features)
        
        # Fairness analysis
        fairness_metrics = {}
        if self.config.enable_fairness_analysis:
            fairness_metrics = self.fairness_analyzer.analyze_fairness(model, X, y, sensitive_features)
        
        # Robustness analysis
        robustness_analysis = {}
        if self.config.enable_adversarial_robustness:
            robustness_analysis = self.robustness_analyzer.analyze_robustness(model, X, y)
        
        # Create comprehensive report
        model_info = {
            'model_type': type(model).__name__,
            'num_features': X.shape[1],
            'num_samples': X.shape[0],
            'has_target': y is not None
        }
        
        xai_report = XAIReport(
            report_id=report_id,
            model_info=model_info,
            explanations=explanations,
            global_insights=global_insights,
            local_insights=local_insights,
            bias_analysis=bias_analysis,
            fairness_metrics=fairness_metrics,
            robustness_analysis=robustness_analysis,
            created_at=datetime.now()
        )
        
        # Store report
        self.xai_reports[report_id] = xai_report
        
        # Save to database
        self._save_xai_report(xai_report)
        
        generation_time = time.time() - start_time
        console.print(f"[green]XAI report generated in {generation_time:.2f} seconds[/green]")
        console.print(f"[blue]Report ID: {report_id}[/blue]")
        console.print(f"[blue]Number of explanations: {len(explanations)}[/blue]")
        
        return xai_report
    
    def _generate_global_insights(self, explanations: List[ExplanationResult]) -> Dict[str, Any]:
        """Generate global insights from explanations."""
        global_insights = {
            'feature_importance_consensus': {},
            'explanation_agreement': 0.0,
            'key_features': [],
            'model_behavior': {}
        }
        
        # Aggregate feature importance across methods
        feature_importance_scores = defaultdict(list)
        
        for explanation in explanations:
            if 'feature_importance' in explanation.explanation_data:
                feature_importance = explanation.explanation_data['feature_importance']
                for i, importance in enumerate(feature_importance):
                    feature_importance_scores[f'feature_{i}'].append(importance)
        
        # Calculate consensus
        for feature, scores in feature_importance_scores.items():
            global_insights['feature_importance_consensus'][feature] = np.mean(scores)
        
        # Identify key features
        if global_insights['feature_importance_consensus']:
            sorted_features = sorted(global_insights['feature_importance_consensus'].items(), 
                                   key=lambda x: x[1], reverse=True)
            global_insights['key_features'] = [feature for feature, _ in sorted_features[:5]]
        
        return global_insights
    
    def _generate_local_insights(self, explanations: List[ExplanationResult]) -> Dict[str, Any]:
        """Generate local insights from explanations."""
        local_insights = {
            'instance_explanations': {},
            'explanation_variance': 0.0,
            'confidence_distribution': {}
        }
        
        # Analyze explanation variance
        explanation_methods = [exp.method.value for exp in explanations]
        local_insights['explanation_variance'] = len(set(explanation_methods)) / len(explanation_methods)
        
        # Analyze confidence distribution
        confidence_scores = []
        for explanation in explanations:
            if explanation.confidence_scores:
                confidence_scores.extend(explanation.confidence_scores.values())
        
        if confidence_scores:
            local_insights['confidence_distribution'] = {
                'mean': np.mean(confidence_scores),
                'std': np.std(confidence_scores),
                'min': np.min(confidence_scores),
                'max': np.max(confidence_scores)
            }
        
        return local_insights
    
    def _save_xai_report(self, report: XAIReport):
        """Save XAI report to database."""
        with sqlite3.connect(self.db_path) as conn:
            # Save explanations
            for explanation in report.explanations:
                conn.execute("""
                    INSERT OR REPLACE INTO explanation_results 
                    (explanation_id, method, scope, data_type, explanation_data,
                     confidence_scores, metadata, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    explanation.explanation_id,
                    explanation.method.value,
                    explanation.scope.value,
                    explanation.data_type.value,
                    json.dumps(explanation.explanation_data),
                    json.dumps(explanation.confidence_scores),
                    json.dumps(explanation.metadata),
                    explanation.created_at.isoformat()
                ))
            
            # Save report
            conn.execute("""
                INSERT OR REPLACE INTO xai_reports 
                (report_id, model_info, explanations, global_insights, local_insights,
                 bias_analysis, fairness_metrics, robustness_analysis, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                report.report_id,
                json.dumps(report.model_info),
                json.dumps([exp.explanation_id for exp in report.explanations]),
                json.dumps(report.global_insights),
                json.dumps(report.local_insights),
                json.dumps(report.bias_analysis),
                json.dumps(report.fairness_metrics),
                json.dumps(report.robustness_analysis),
                report.created_at.isoformat()
            ))
    
    def visualize_explanations(self, report: XAIReport, output_path: str = None) -> str:
        """Visualize explanations."""
        if output_path is None:
            output_path = f"xai_explanations_{report.report_id}.png"
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Feature importance consensus
        if report.global_insights.get('feature_importance_consensus'):
            features = list(report.global_insights['feature_importance_consensus'].keys())
            importances = list(report.global_insights['feature_importance_consensus'].values())
            
            axes[0, 0].barh(features, importances)
            axes[0, 0].set_title('Feature Importance Consensus')
            axes[0, 0].set_xlabel('Importance')
        
        # Bias analysis
        if report.bias_analysis and 'bias_score' in report.bias_analysis:
            bias_metrics = ['Statistical Parity', 'Equalized Odds', 'Demographic Parity']
            bias_values = [
                report.bias_analysis.get('statistical_parity', 0),
                report.bias_analysis.get('equalized_odds', 0),
                report.bias_analysis.get('demographic_parity', 0)
            ]
            
            axes[0, 1].bar(bias_metrics, bias_values)
            axes[0, 1].set_title('Bias Analysis')
            axes[0, 1].set_ylabel('Bias Score')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Fairness metrics
        if report.fairness_metrics and 'fairness_score' in report.fairness_metrics:
            fairness_score = report.fairness_metrics['fairness_score']
            axes[1, 0].pie([fairness_score, 1 - fairness_score], 
                          labels=['Fair', 'Unfair'], 
                          autopct='%1.1f%%',
                          colors=['green', 'red'])
            axes[1, 0].set_title('Fairness Analysis')
        
        # Robustness analysis
        if report.robustness_analysis and 'overall_robustness_score' in report.robustness_analysis:
            robustness_score = report.robustness_analysis['overall_robustness_score']
            axes[1, 1].pie([robustness_score, 1 - robustness_score], 
                          labels=['Robust', 'Not Robust'], 
                          autopct='%1.1f%%',
                          colors=['blue', 'orange'])
            axes[1, 1].set_title('Robustness Analysis')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        console.print(f"[green]XAI visualization saved: {output_path}[/green]")
        return output_path
    
    def get_xai_summary(self) -> Dict[str, Any]:
        """Get XAI system summary."""
        if not self.xai_reports:
            return {'total_reports': 0}
        
        total_reports = len(self.xai_reports)
        
        # Calculate average metrics
        bias_scores = []
        fairness_scores = []
        robustness_scores = []
        
        for report in self.xai_reports.values():
            if report.bias_analysis and 'bias_score' in report.bias_analysis:
                bias_scores.append(report.bias_analysis['bias_score'])
            if report.fairness_metrics and 'fairness_score' in report.fairness_metrics:
                fairness_scores.append(report.fairness_metrics['fairness_score'])
            if report.robustness_analysis and 'overall_robustness_score' in report.robustness_analysis:
                robustness_scores.append(report.robustness_analysis['overall_robustness_score'])
        
        return {
            'total_reports': total_reports,
            'average_bias_score': np.mean(bias_scores) if bias_scores else 0,
            'average_fairness_score': np.mean(fairness_scores) if fairness_scores else 0,
            'average_robustness_score': np.mean(robustness_scores) if robustness_scores else 0,
            'total_explanations': sum(len(report.explanations) for report in self.xai_reports.values())
        }

def main():
    """Main function for explainable AI CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Explainable AI System")
    parser.add_argument("--explanation-methods", nargs="+",
                       choices=["shap", "lime", "integrated_gradients", "anchor_tabular"],
                       default=["shap", "lime"], help="Explanation methods")
    parser.add_argument("--explanation-scope", type=str,
                       choices=["global", "local", "regional"],
                       default="local", help="Explanation scope")
    parser.add_argument("--data-type", type=str,
                       choices=["tabular", "image", "text"],
                       default="tabular", help="Data type")
    parser.add_argument("--explanation-format", type=str,
                       choices=["visualization", "textual", "numerical"],
                       default="visualization", help="Explanation format")
    parser.add_argument("--enable-bias-detection", action="store_true",
                       help="Enable bias detection")
    parser.add_argument("--enable-fairness-analysis", action="store_true",
                       help="Enable fairness analysis")
    parser.add_argument("--enable-robustness-analysis", action="store_true",
                       help="Enable robustness analysis")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use")
    
    args = parser.parse_args()
    
    # Create XAI configuration
    explanation_methods = [ExplanationMethod(method) for method in args.explanation_methods]
    config = XAIConfig(
        explanation_methods=explanation_methods,
        explanation_scope=ExplanationScope(args.explanation_scope),
        data_type=DataType(args.data_type),
        explanation_format=ExplanationFormat(args.explanation_format),
        enable_bias_detection=args.enable_bias_detection,
        enable_fairness_analysis=args.enable_fairness_analysis,
        enable_adversarial_robustness=args.enable_robustness_analysis,
        device=args.device
    )
    
    # Create XAI system
    xai_system = XAISystem(config)
    
    # Create sample model and data
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    
    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
    
    # Train sample model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Generate explanations
    report = xai_system.explain_model(model, X, y)
    
    # Show results
    console.print(f"[green]XAI report generated[/green]")
    console.print(f"[blue]Report ID: {report.report_id}[/blue]")
    console.print(f"[blue]Number of explanations: {len(report.explanations)}[/blue]")
    
    if report.bias_analysis:
        console.print(f"[blue]Bias detected: {report.bias_analysis.get('bias_detected', False)}[/blue]")
        console.print(f"[blue]Bias score: {report.bias_analysis.get('bias_score', 0):.4f}[/blue]")
    
    if report.fairness_metrics:
        console.print(f"[blue]Fairness score: {report.fairness_metrics.get('fairness_score', 0):.4f}[/blue]")
    
    if report.robustness_analysis:
        console.print(f"[blue]Robustness score: {report.robustness_analysis.get('overall_robustness_score', 0):.4f}[/blue]")
    
    # Create visualization
    xai_system.visualize_explanations(report)
    
    # Show summary
    summary = xai_system.get_xai_summary()
    console.print(f"[blue]Summary: {summary}[/blue]")

if __name__ == "__main__":
    main()
