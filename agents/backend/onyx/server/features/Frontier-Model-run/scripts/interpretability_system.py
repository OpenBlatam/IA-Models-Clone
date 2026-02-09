#!/usr/bin/env python3
"""
Advanced Model Interpretability System for Frontier Model Training
Provides comprehensive model explanation, visualization, and analysis capabilities.
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import permutation_importance
import shap
import lime
import lime.lime_tabular
import lime.lime_image
from lime import submodular_pick
import captum
from captum.attr import IntegratedGradients, GradientShap, Saliency, InputXGradient
from captum.attr import LayerGradCam, LayerAttribution, GuidedBackprop
from captum.attr import DeepLift, DeepLiftShap, FeatureAblation, Occlusion
from captum.attr import ShapleyValueSampling, KernelShap
from captum.insights import AttributionVisualizer
from captum.insights.features import ImageFeature
import cv2
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
import pandas as pd
import joblib
import pickle
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

console = Console()

class InterpretabilityMethod(Enum):
    """Interpretability methods."""
    SHAP = "shap"
    LIME = "lime"
    GRAD_CAM = "grad_cam"
    INTEGRATED_GRADIENTS = "integrated_gradients"
    GRADIENT_SHAP = "gradient_shap"
    SALIENCY = "saliency"
    INPUT_X_GRADIENT = "input_x_gradient"
    DEEP_LIFT = "deep_lift"
    DEEP_LIFT_SHAP = "deep_lift_shap"
    FEATURE_ABLATION = "feature_ablation"
    OCCLUSION = "occlusion"
    SHAPLEY_VALUE = "shapley_value"
    KERNEL_SHAP = "kernel_shap"
    PERMUTATION_IMPORTANCE = "permutation_importance"
    PARTIAL_DEPENDENCE = "partial_dependence"
    ICE_PLOTS = "ice_plots"
    FEATURE_INTERACTION = "feature_interaction"
    ATTENTION_VISUALIZATION = "attention_visualization"
    ACTIVATION_MAXIMIZATION = "activation_maximization"

class ModelType(Enum):
    """Model types."""
    NEURAL_NETWORK = "neural_network"
    RANDOM_FOREST = "random_forest"
    LINEAR_MODEL = "linear_model"
    DECISION_TREE = "decision_tree"
    TRANSFORMER = "transformer"
    CNN = "cnn"
    RNN = "rnn"
    GRAPH_NEURAL_NETWORK = "graph_neural_network"
    ENSEMBLE = "ensemble"

class DataType(Enum):
    """Data types."""
    TABULAR = "tabular"
    IMAGE = "image"
    TEXT = "text"
    TIME_SERIES = "time_series"
    GRAPH = "graph"
    MULTIMODAL = "multimodal"

@dataclass
class InterpretabilityConfig:
    """Interpretability configuration."""
    methods: List[InterpretabilityMethod] = None
    model_type: ModelType = ModelType.NEURAL_NETWORK
    data_type: DataType = DataType.TABULAR
    num_samples: int = 100
    num_features: int = 10
    background_samples: int = 50
    max_evals: int = 1000
    batch_size: int = 32
    device: str = "auto"
    enable_visualization: bool = True
    enable_statistical_tests: bool = True
    enable_feature_importance: bool = True
    enable_global_explanations: bool = True
    enable_local_explanations: bool = True
    save_explanations: bool = True
    explanation_format: str = "json"

@dataclass
class Explanation:
    """Model explanation."""
    explanation_id: str
    method: InterpretabilityMethod
    model_id: str
    sample_id: str
    feature_importance: Dict[str, float]
    prediction: Any
    confidence: float
    created_at: datetime
    metadata: Dict[str, Any] = None

@dataclass
class GlobalExplanation:
    """Global model explanation."""
    explanation_id: str
    method: InterpretabilityMethod
    model_id: str
    feature_importance: Dict[str, float]
    feature_interactions: Dict[str, float]
    model_complexity: Dict[str, float]
    created_at: datetime
    metadata: Dict[str, Any] = None

class SHAPExplainer:
    """SHAP-based explainer."""
    
    def __init__(self, config: InterpretabilityConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize device
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
    
    def explain_model(self, model: Any, X: np.ndarray, y: np.ndarray = None) -> Dict[str, Any]:
        """Explain model using SHAP."""
        explanations = {}
        
        try:
            if self.config.data_type == DataType.TABULAR:
                explanations = self._explain_tabular_model(model, X, y)
            elif self.config.data_type == DataType.IMAGE:
                explanations = self._explain_image_model(model, X, y)
            elif self.config.data_type == DataType.TEXT:
                explanations = self._explain_text_model(model, X, y)
            
        except Exception as e:
            self.logger.error(f"SHAP explanation failed: {e}")
            explanations = {"error": str(e)}
        
        return explanations
    
    def _explain_tabular_model(self, model: Any, X: np.ndarray, y: np.ndarray = None) -> Dict[str, Any]:
        """Explain tabular model."""
        # Create SHAP explainer
        if hasattr(model, 'predict_proba'):
            explainer = shap.Explainer(model.predict_proba, X[:self.config.background_samples])
        else:
            explainer = shap.Explainer(model.predict, X[:self.config.background_samples])
        
        # Calculate SHAP values
        shap_values = explainer(X[:self.config.num_samples])
        
        # Global explanations
        global_explanation = {
            'feature_importance': self._calculate_feature_importance(shap_values),
            'feature_interactions': self._calculate_feature_interactions(shap_values),
            'summary_statistics': self._calculate_summary_statistics(shap_values)
        }
        
        # Local explanations
        local_explanations = []
        for i in range(min(self.config.num_samples, len(shap_values.values))):
            local_explanation = {
                'sample_id': f"sample_{i}",
                'feature_values': X[i].tolist(),
                'shap_values': shap_values.values[i].tolist(),
                'base_value': shap_values.base_values[i] if hasattr(shap_values, 'base_values') else 0,
                'prediction': model.predict(X[i:i+1])[0] if hasattr(model, 'predict') else None
            }
            local_explanations.append(local_explanation)
        
        return {
            'global_explanation': global_explanation,
            'local_explanations': local_explanations,
            'shap_values': shap_values
        }
    
    def _explain_image_model(self, model: Any, X: np.ndarray, y: np.ndarray = None) -> Dict[str, Any]:
        """Explain image model."""
        # Create SHAP explainer for images
        explainer = shap.Explainer(model, X[:self.config.background_samples])
        
        # Calculate SHAP values
        shap_values = explainer(X[:self.config.num_samples])
        
        # Process SHAP values for images
        explanations = {
            'shap_values': shap_values.values,
            'base_values': shap_values.base_values if hasattr(shap_values, 'base_values') else None,
            'predictions': [model.predict(X[i:i+1])[0] for i in range(min(self.config.num_samples, len(X)))]
        }
        
        return explanations
    
    def _explain_text_model(self, model: Any, X: np.ndarray, y: np.ndarray = None) -> Dict[str, Any]:
        """Explain text model."""
        # For text models, we might need tokenization
        # This is a simplified version
        explainer = shap.Explainer(model, X[:self.config.background_samples])
        shap_values = explainer(X[:self.config.num_samples])
        
        return {
            'shap_values': shap_values.values,
            'base_values': shap_values.base_values if hasattr(shap_values, 'base_values') else None
        }
    
    def _calculate_feature_importance(self, shap_values) -> Dict[str, float]:
        """Calculate feature importance from SHAP values."""
        if hasattr(shap_values, 'values'):
            values = shap_values.values
        else:
            values = shap_values
        
        # Calculate mean absolute SHAP values
        mean_abs_values = np.mean(np.abs(values), axis=0)
        
        # Normalize to sum to 1
        total_importance = np.sum(mean_abs_values)
        if total_importance > 0:
            normalized_importance = mean_abs_values / total_importance
        else:
            normalized_importance = mean_abs_values
        
        return {f"feature_{i}": float(importance) for i, importance in enumerate(normalized_importance)}
    
    def _calculate_feature_interactions(self, shap_values) -> Dict[str, float]:
        """Calculate feature interactions."""
        # Simplified feature interaction calculation
        if hasattr(shap_values, 'values'):
            values = shap_values.values
        else:
            values = shap_values
        
        interactions = {}
        n_features = values.shape[1]
        
        for i in range(n_features):
            for j in range(i + 1, n_features):
                # Calculate correlation between SHAP values
                corr = np.corrcoef(values[:, i], values[:, j])[0, 1]
                interactions[f"feature_{i}_feature_{j}"] = float(corr)
        
        return interactions
    
    def _calculate_summary_statistics(self, shap_values) -> Dict[str, float]:
        """Calculate summary statistics."""
        if hasattr(shap_values, 'values'):
            values = shap_values.values
        else:
            values = shap_values
        
        return {
            'mean_shap_value': float(np.mean(values)),
            'std_shap_value': float(np.std(values)),
            'min_shap_value': float(np.min(values)),
            'max_shap_value': float(np.max(values)),
            'num_features': int(values.shape[1]),
            'num_samples': int(values.shape[0])
        }

class LIMEExplainer:
    """LIME-based explainer."""
    
    def __init__(self, config: InterpretabilityConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def explain_model(self, model: Any, X: np.ndarray, y: np.ndarray = None) -> Dict[str, Any]:
        """Explain model using LIME."""
        explanations = {}
        
        try:
            if self.config.data_type == DataType.TABULAR:
                explanations = self._explain_tabular_model(model, X, y)
            elif self.config.data_type == DataType.IMAGE:
                explanations = self._explain_image_model(model, X, y)
            
        except Exception as e:
            self.logger.error(f"LIME explanation failed: {e}")
            explanations = {"error": str(e)}
        
        return explanations
    
    def _explain_tabular_model(self, model: Any, X: np.ndarray, y: np.ndarray = None) -> Dict[str, Any]:
        """Explain tabular model using LIME."""
        # Create LIME explainer
        explainer = lime.lime_tabular.LimeTabularExplainer(
            X,
            mode='classification' if hasattr(model, 'predict_proba') else 'regression',
            feature_names=[f'feature_{i}' for i in range(X.shape[1])],
            discretize_continuous=True
        )
        
        # Generate explanations for samples
        local_explanations = []
        for i in range(min(self.config.num_samples, len(X))):
            try:
                explanation = explainer.explain_instance(
                    X[i],
                    model.predict_proba if hasattr(model, 'predict_proba') else model.predict,
                    num_features=self.config.num_features
                )
                
                local_explanation = {
                    'sample_id': f"sample_{i}",
                    'feature_importance': dict(explanation.as_list()),
                    'prediction': model.predict(X[i:i+1])[0] if hasattr(model, 'predict') else None,
                    'confidence': explanation.score if hasattr(explanation, 'score') else None
                }
                local_explanations.append(local_explanation)
                
            except Exception as e:
                self.logger.warning(f"LIME explanation failed for sample {i}: {e}")
        
        return {
            'local_explanations': local_explanations,
            'method': 'lime_tabular'
        }
    
    def _explain_image_model(self, model: Any, X: np.ndarray, y: np.ndarray = None) -> Dict[str, Any]:
        """Explain image model using LIME."""
        # Create LIME explainer for images
        explainer = lime.lime_image.LimeImageExplainer()
        
        # Generate explanations for samples
        local_explanations = []
        for i in range(min(self.config.num_samples, len(X))):
            try:
                explanation = explainer.explain_instance(
                    X[i],
                    model.predict,
                    top_labels=5,
                    hide_color=0,
                    num_samples=1000
                )
                
                # Get explanation for top prediction
                top_prediction = explanation.top_labels[0]
                explanation_image, explanation_mask = explanation.get_image_and_mask(
                    top_prediction,
                    positive_only=True,
                    num_features=5,
                    hide_rest=True
                )
                
                local_explanation = {
                    'sample_id': f"sample_{i}",
                    'top_prediction': top_prediction,
                    'explanation_mask': explanation_mask.tolist(),
                    'prediction': model.predict(X[i:i+1])[0] if hasattr(model, 'predict') else None
                }
                local_explanations.append(local_explanation)
                
            except Exception as e:
                self.logger.warning(f"LIME explanation failed for image {i}: {e}")
        
        return {
            'local_explanations': local_explanations,
            'method': 'lime_image'
        }

class CaptumExplainer:
    """Captum-based explainer for PyTorch models."""
    
    def __init__(self, config: InterpretabilityConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize device
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
    
    def explain_model(self, model: nn.Module, X: torch.Tensor, y: torch.Tensor = None) -> Dict[str, Any]:
        """Explain PyTorch model using Captum."""
        explanations = {}
        
        try:
            model = model.to(self.device)
            X = X.to(self.device)
            
            if self.config.data_type == DataType.TABULAR:
                explanations = self._explain_tabular_model(model, X, y)
            elif self.config.data_type == DataType.IMAGE:
                explanations = self._explain_image_model(model, X, y)
            
        except Exception as e:
            self.logger.error(f"Captum explanation failed: {e}")
            explanations = {"error": str(e)}
        
        return explanations
    
    def _explain_tabular_model(self, model: nn.Module, X: torch.Tensor, y: torch.Tensor = None) -> Dict[str, Any]:
        """Explain tabular PyTorch model."""
        explanations = {}
        
        # Integrated Gradients
        if InterpretabilityMethod.INTEGRATED_GRADIENTS in self.config.methods:
            ig = IntegratedGradients(model)
            ig_attr = ig.attribute(X[:self.config.num_samples], target=y[:self.config.num_samples] if y is not None else None)
            explanations['integrated_gradients'] = ig_attr.cpu().numpy().tolist()
        
        # Gradient SHAP
        if InterpretabilityMethod.GRADIENT_SHAP in self.config.methods:
            gs = GradientShap(model)
            baseline = torch.zeros_like(X[:self.config.num_samples])
            gs_attr = gs.attribute(X[:self.config.num_samples], baselines=baseline, target=y[:self.config.num_samples] if y is not None else None)
            explanations['gradient_shap'] = gs_attr.cpu().numpy().tolist()
        
        # Saliency
        if InterpretabilityMethod.SALIENCY in self.config.methods:
            saliency = Saliency(model)
            saliency_attr = saliency.attribute(X[:self.config.num_samples], target=y[:self.config.num_samples] if y is not None else None)
            explanations['saliency'] = saliency_attr.cpu().numpy().tolist()
        
        # Input X Gradient
        if InterpretabilityMethod.INPUT_X_GRADIENT in self.config.methods:
            ixg = InputXGradient(model)
            ixg_attr = ixg.attribute(X[:self.config.num_samples], target=y[:self.config.num_samples] if y is not None else None)
            explanations['input_x_gradient'] = ixg_attr.cpu().numpy().tolist()
        
        # DeepLift
        if InterpretabilityMethod.DEEP_LIFT in self.config.methods:
            dl = DeepLift(model)
            dl_attr = dl.attribute(X[:self.config.num_samples], target=y[:self.config.num_samples] if y is not None else None)
            explanations['deep_lift'] = dl_attr.cpu().numpy().tolist()
        
        # Feature Ablation
        if InterpretabilityMethod.FEATURE_ABLATION in self.config.methods:
            fa = FeatureAblation(model)
            fa_attr = fa.attribute(X[:self.config.num_samples], target=y[:self.config.num_samples] if y is not None else None)
            explanations['feature_ablation'] = fa_attr.cpu().numpy().tolist()
        
        return explanations
    
    def _explain_image_model(self, model: nn.Module, X: torch.Tensor, y: torch.Tensor = None) -> Dict[str, Any]:
        """Explain image PyTorch model."""
        explanations = {}
        
        # GradCAM
        if InterpretabilityMethod.GRAD_CAM in self.config.methods:
            # Get the last convolutional layer
            conv_layer = None
            for module in model.modules():
                if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                    conv_layer = module
            
            if conv_layer is not None:
                gc = LayerGradCam(model, conv_layer)
                gc_attr = gc.attribute(X[:self.config.num_samples], target=y[:self.config.num_samples] if y is not None else None)
                explanations['grad_cam'] = gc_attr.cpu().numpy().tolist()
        
        # Integrated Gradients for images
        if InterpretabilityMethod.INTEGRATED_GRADIENTS in self.config.methods:
            ig = IntegratedGradients(model)
            ig_attr = ig.attribute(X[:self.config.num_samples], target=y[:self.config.num_samples] if y is not None else None)
            explanations['integrated_gradients'] = ig_attr.cpu().numpy().tolist()
        
        # Saliency for images
        if InterpretabilityMethod.SALIENCY in self.config.methods:
            saliency = Saliency(model)
            saliency_attr = saliency.attribute(X[:self.config.num_samples], target=y[:self.config.num_samples] if y is not None else None)
            explanations['saliency'] = saliency_attr.cpu().numpy().tolist()
        
        return explanations

class PermutationImportanceExplainer:
    """Permutation importance explainer."""
    
    def __init__(self, config: InterpretabilityConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def explain_model(self, model: Any, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Explain model using permutation importance."""
        try:
            # Calculate permutation importance
            perm_importance = permutation_importance(
                model, X, y, 
                n_repeats=10, 
                random_state=42,
                scoring='accuracy' if hasattr(model, 'predict_proba') else 'neg_mean_squared_error'
            )
            
            # Create feature importance dictionary
            feature_importance = {
                f"feature_{i}": {
                    'importance': float(perm_importance.importances_mean[i]),
                    'std': float(perm_importance.importances_std[i])
                }
                for i in range(len(perm_importance.importances_mean))
            }
            
            return {
                'feature_importance': feature_importance,
                'method': 'permutation_importance',
                'n_repeats': 10
            }
            
        except Exception as e:
            self.logger.error(f"Permutation importance explanation failed: {e}")
            return {"error": str(e)}

class ModelInterpretabilitySystem:
    """Main model interpretability system."""
    
    def __init__(self, config: InterpretabilityConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize explainers
        self.explainers = {}
        self._init_explainers()
        
        # Initialize database
        self.db_path = self._init_database()
        
        # Results storage
        self.explanations: Dict[str, Explanation] = {}
        self.global_explanations: Dict[str, GlobalExplanation] = {}
    
    def _init_explainers(self):
        """Initialize explainers based on configuration."""
        if InterpretabilityMethod.SHAP in self.config.methods:
            self.explainers['shap'] = SHAPExplainer(self.config)
        
        if InterpretabilityMethod.LIME in self.config.methods:
            self.explainers['lime'] = LIMEExplainer(self.config)
        
        if any(method in self.config.methods for method in [
            InterpretabilityMethod.INTEGRATED_GRADIENTS,
            InterpretabilityMethod.GRADIENT_SHAP,
            InterpretabilityMethod.SALIENCY,
            InterpretabilityMethod.GRAD_CAM
        ]):
            self.explainers['captum'] = CaptumExplainer(self.config)
        
        if InterpretabilityMethod.PERMUTATION_IMPORTANCE in self.config.methods:
            self.explainers['permutation'] = PermutationImportanceExplainer(self.config)
    
    def _init_database(self) -> str:
        """Initialize interpretability database."""
        db_path = Path("./interpretability.db")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS explanations (
                    explanation_id TEXT PRIMARY KEY,
                    method TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    sample_id TEXT NOT NULL,
                    feature_importance TEXT NOT NULL,
                    prediction TEXT,
                    confidence REAL,
                    created_at TEXT NOT NULL,
                    metadata TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS global_explanations (
                    explanation_id TEXT PRIMARY KEY,
                    method TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    feature_importance TEXT NOT NULL,
                    feature_interactions TEXT,
                    model_complexity TEXT,
                    created_at TEXT NOT NULL,
                    metadata TEXT
                )
            """)
        
        return str(db_path)
    
    def explain_model(self, model: Any, X: np.ndarray, y: np.ndarray = None, 
                     model_id: str = "default_model") -> Dict[str, Any]:
        """Explain model using multiple methods."""
        console.print(f"[blue]Explaining model {model_id} using {len(self.explainers)} methods[/blue]")
        
        all_explanations = {}
        
        for explainer_name, explainer in self.explainers.items():
            console.print(f"[blue]Running {explainer_name} explainer...[/blue]")
            
            try:
                # Convert to appropriate format
                if explainer_name == 'captum' and isinstance(model, nn.Module):
                    X_tensor = torch.FloatTensor(X)
                    y_tensor = torch.LongTensor(y) if y is not None else None
                    explanations = explainer.explain_model(model, X_tensor, y_tensor)
                else:
                    explanations = explainer.explain_model(model, X, y)
                
                all_explanations[explainer_name] = explanations
                
                # Save explanations
                if self.config.save_explanations:
                    self._save_explanations(explainer_name, explanations, model_id)
                
            except Exception as e:
                self.logger.error(f"Explainer {explainer_name} failed: {e}")
                all_explanations[explainer_name] = {"error": str(e)}
        
        # Generate visualizations
        if self.config.enable_visualization:
            self._create_visualizations(all_explanations, model_id)
        
        # Generate summary report
        summary = self._generate_summary_report(all_explanations, model_id)
        
        console.print(f"[green]Model explanation completed for {model_id}[/green]")
        
        return {
            'explanations': all_explanations,
            'summary': summary,
            'model_id': model_id,
            'timestamp': datetime.now().isoformat()
        }
    
    def _save_explanations(self, method: str, explanations: Dict[str, Any], model_id: str):
        """Save explanations to database."""
        explanation_id = f"{method}_{model_id}_{int(time.time())}"
        
        with sqlite3.connect(self.db_path) as conn:
            if 'local_explanations' in explanations:
                for local_exp in explanations['local_explanations']:
                    conn.execute("""
                        INSERT INTO explanations 
                        (explanation_id, method, model_id, sample_id, feature_importance, 
                         prediction, confidence, created_at, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        f"{explanation_id}_{local_exp['sample_id']}",
                        method,
                        model_id,
                        local_exp['sample_id'],
                        json.dumps(local_exp.get('feature_importance', {})),
                        json.dumps(local_exp.get('prediction')),
                        local_exp.get('confidence'),
                        datetime.now().isoformat(),
                        json.dumps(local_exp.get('metadata', {}))
                    ))
            
            if 'global_explanation' in explanations:
                global_exp = explanations['global_explanation']
                conn.execute("""
                    INSERT INTO global_explanations 
                    (explanation_id, method, model_id, feature_importance, 
                     feature_interactions, model_complexity, created_at, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    explanation_id,
                    method,
                    model_id,
                    json.dumps(global_exp.get('feature_importance', {})),
                    json.dumps(global_exp.get('feature_interactions', {})),
                    json.dumps(global_exp.get('model_complexity', {})),
                    datetime.now().isoformat(),
                    json.dumps(global_exp.get('metadata', {}))
                ))
    
    def _create_visualizations(self, explanations: Dict[str, Any], model_id: str):
        """Create visualizations for explanations."""
        try:
            # Feature importance plot
            self._plot_feature_importance(explanations, model_id)
            
            # SHAP summary plot
            if 'shap' in explanations and 'shap_values' in explanations['shap']:
                self._plot_shap_summary(explanations['shap'], model_id)
            
            # LIME explanations plot
            if 'lime' in explanations and 'local_explanations' in explanations['lime']:
                self._plot_lime_explanations(explanations['lime'], model_id)
            
            # Captum attributions plot
            if 'captum' in explanations:
                self._plot_captum_attributions(explanations['captum'], model_id)
            
        except Exception as e:
            self.logger.error(f"Visualization creation failed: {e}")
    
    def _plot_feature_importance(self, explanations: Dict[str, Any], model_id: str):
        """Plot feature importance across methods."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        method_idx = 0
        for method, exp in explanations.items():
            if method_idx >= 4:
                break
            
            if 'global_explanation' in exp and 'feature_importance' in exp['global_explanation']:
                feature_importance = exp['global_explanation']['feature_importance']
                
                # Sort features by importance
                sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                features, importances = zip(*sorted_features[:10])  # Top 10 features
                
                axes[method_idx].bar(range(len(features)), importances)
                axes[method_idx].set_title(f'{method.upper()} Feature Importance')
                axes[method_idx].set_xlabel('Features')
                axes[method_idx].set_ylabel('Importance')
                axes[method_idx].set_xticks(range(len(features)))
                axes[method_idx].set_xticklabels(features, rotation=45)
                
                method_idx += 1
        
        plt.tight_layout()
        plt.savefig(f'feature_importance_{model_id}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        console.print(f"[green]Feature importance plot saved: feature_importance_{model_id}.png[/green]")
    
    def _plot_shap_summary(self, shap_explanations: Dict[str, Any], model_id: str):
        """Plot SHAP summary."""
        try:
            if 'shap_values' in shap_explanations:
                shap_values = shap_explanations['shap_values']
                
                # Create SHAP summary plot
                plt.figure(figsize=(10, 8))
                shap.summary_plot(shap_values, show=False)
                plt.title(f'SHAP Summary Plot - {model_id}')
                plt.tight_layout()
                plt.savefig(f'shap_summary_{model_id}.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                console.print(f"[green]SHAP summary plot saved: shap_summary_{model_id}.png[/green]")
                
        except Exception as e:
            self.logger.error(f"SHAP summary plot failed: {e}")
    
    def _plot_lime_explanations(self, lime_explanations: Dict[str, Any], model_id: str):
        """Plot LIME explanations."""
        try:
            if 'local_explanations' in lime_explanations:
                local_explanations = lime_explanations['local_explanations']
                
                # Plot first few explanations
                num_plots = min(4, len(local_explanations))
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                axes = axes.flatten()
                
                for i in range(num_plots):
                    exp = local_explanations[i]
                    if 'feature_importance' in exp:
                        features = list(exp['feature_importance'].keys())
                        importances = list(exp['feature_importance'].values())
                        
                        axes[i].barh(features, importances)
                        axes[i].set_title(f'LIME Explanation - Sample {exp["sample_id"]}')
                        axes[i].set_xlabel('Importance')
                
                plt.tight_layout()
                plt.savefig(f'lime_explanations_{model_id}.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                console.print(f"[green]LIME explanations plot saved: lime_explanations_{model_id}.png[/green]")
                
        except Exception as e:
            self.logger.error(f"LIME explanations plot failed: {e}")
    
    def _plot_captum_attributions(self, captum_explanations: Dict[str, Any], model_id: str):
        """Plot Captum attributions."""
        try:
            # Plot different attribution methods
            num_methods = len(captum_explanations)
            if num_methods == 0:
                return
            
            fig, axes = plt.subplots(1, num_methods, figsize=(5 * num_methods, 5))
            if num_methods == 1:
                axes = [axes]
            
            method_idx = 0
            for method, attributions in captum_explanations.items():
                if isinstance(attributions, list) and len(attributions) > 0:
                    # Plot mean attribution across samples
                    mean_attribution = np.mean(attributions, axis=0)
                    
                    if len(mean_attribution.shape) == 1:  # Tabular data
                        axes[method_idx].bar(range(len(mean_attribution)), mean_attribution)
                        axes[method_idx].set_title(f'{method.upper()} Attribution')
                        axes[method_idx].set_xlabel('Features')
                        axes[method_idx].set_ylabel('Attribution')
                    else:  # Image data
                        axes[method_idx].imshow(mean_attribution[0], cmap='hot')
                        axes[method_idx].set_title(f'{method.upper()} Attribution')
                        axes[method_idx].axis('off')
                    
                    method_idx += 1
            
            plt.tight_layout()
            plt.savefig(f'captum_attributions_{model_id}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            console.print(f"[green]Captum attributions plot saved: captum_attributions_{model_id}.png[/green]")
            
        except Exception as e:
            self.logger.error(f"Captum attributions plot failed: {e}")
    
    def _generate_summary_report(self, explanations: Dict[str, Any], model_id: str) -> Dict[str, Any]:
        """Generate summary report."""
        summary = {
            'model_id': model_id,
            'timestamp': datetime.now().isoformat(),
            'methods_used': list(explanations.keys()),
            'successful_methods': [],
            'failed_methods': [],
            'feature_importance_consensus': {},
            'key_insights': []
        }
        
        # Analyze each method
        feature_importance_scores = defaultdict(list)
        
        for method, exp in explanations.items():
            if 'error' in exp:
                summary['failed_methods'].append(method)
            else:
                summary['successful_methods'].append(method)
                
                # Extract feature importance if available
                if 'global_explanation' in exp and 'feature_importance' in exp['global_explanation']:
                    feature_importance = exp['global_explanation']['feature_importance']
                    for feature, importance in feature_importance.items():
                        feature_importance_scores[feature].append(importance)
        
        # Calculate consensus feature importance
        for feature, scores in feature_importance_scores.items():
            if scores:
                summary['feature_importance_consensus'][feature] = {
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'count': len(scores)
                }
        
        # Generate key insights
        if summary['feature_importance_consensus']:
            # Find most important features
            sorted_features = sorted(
                summary['feature_importance_consensus'].items(),
                key=lambda x: x[1]['mean'],
                reverse=True
            )
            
            top_features = sorted_features[:5]
            summary['key_insights'].append(f"Top 5 most important features: {[f[0] for f in top_features]}")
            
            # Check for consensus
            high_consensus_features = [
                f for f, stats in summary['feature_importance_consensus'].items()
                if stats['std'] < 0.1 and stats['count'] > 1
            ]
            if high_consensus_features:
                summary['key_insights'].append(f"High consensus features: {high_consensus_features}")
        
        return summary
    
    def compare_models(self, models: Dict[str, Any], X: np.ndarray, y: np.ndarray = None) -> Dict[str, Any]:
        """Compare interpretability across multiple models."""
        console.print("[blue]Comparing interpretability across models...[/blue]")
        
        comparison_results = {}
        
        for model_id, model in models.items():
            console.print(f"[blue]Explaining model: {model_id}[/blue]")
            
            explanations = self.explain_model(model, X, y, model_id)
            comparison_results[model_id] = explanations
        
        # Generate comparison report
        comparison_report = self._generate_comparison_report(comparison_results)
        
        console.print("[green]Model comparison completed[/green]")
        
        return {
            'model_explanations': comparison_results,
            'comparison_report': comparison_report
        }
    
    def _generate_comparison_report(self, comparison_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comparison report across models."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'models_compared': list(comparison_results.keys()),
            'feature_importance_comparison': {},
            'consensus_features': {},
            'model_rankings': {}
        }
        
        # Compare feature importance across models
        all_feature_importance = defaultdict(list)
        
        for model_id, results in comparison_results.items():
            if 'summary' in results and 'feature_importance_consensus' in results['summary']:
                feature_importance = results['summary']['feature_importance_consensus']
                for feature, stats in feature_importance.items():
                    all_feature_importance[feature].append({
                        'model': model_id,
                        'importance': stats['mean']
                    })
        
        # Find consensus features
        for feature, model_importances in all_feature_importance.items():
            if len(model_importances) > 1:
                importances = [mi['importance'] for mi in model_importances]
                if np.std(importances) < 0.1:  # Low standard deviation indicates consensus
                    report['consensus_features'][feature] = {
                        'mean_importance': np.mean(importances),
                        'std_importance': np.std(importances),
                        'models': [mi['model'] for mi in model_importances]
                    }
        
        return report

def main():
    """Main function for interpretability CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Model Interpretability System")
    parser.add_argument("--methods", nargs="+",
                       choices=["shap", "lime", "grad_cam", "integrated_gradients", "saliency"],
                       default=["shap", "lime"], help="Interpretability methods")
    parser.add_argument("--model-type", type=str,
                       choices=["neural_network", "random_forest", "linear_model"],
                       default="neural_network", help="Model type")
    parser.add_argument("--data-type", type=str,
                       choices=["tabular", "image", "text"],
                       default="tabular", help="Data type")
    parser.add_argument("--num-samples", type=int, default=100,
                       help="Number of samples to explain")
    parser.add_argument("--num-features", type=int, default=10,
                       help="Number of features to show")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use")
    parser.add_argument("--enable-visualization", action="store_true",
                       help="Enable visualization")
    
    args = parser.parse_args()
    
    # Create interpretability configuration
    methods = [InterpretabilityMethod(method) for method in args.methods]
    config = InterpretabilityConfig(
        methods=methods,
        model_type=ModelType(args.model_type),
        data_type=DataType(args.data_type),
        num_samples=args.num_samples,
        num_features=args.num_features,
        device=args.device,
        enable_visualization=args.enable_visualization
    )
    
    # Create interpretability system
    interpretability_system = ModelInterpretabilitySystem(config)
    
    # Generate sample data and model
    if args.data_type == "tabular":
        from sklearn.datasets import make_classification
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        
        X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Explain model
        explanations = interpretability_system.explain_model(model, X_test, y_test, "sample_model")
        
    elif args.data_type == "image":
        # For image data, we would need a CNN model
        console.print("[red]Image model explanation requires a trained CNN model[/red]")
        return
    
    else:
        console.print(f"[red]Unsupported data type: {args.data_type}[/red]")
        return
    
    # Show results
    console.print(f"[green]Model explanation completed[/green]")
    console.print(f"[blue]Methods used: {explanations['summary']['methods_used']}[/blue]")
    console.print(f"[blue]Successful methods: {explanations['summary']['successful_methods']}[/blue]")
    
    if explanations['summary']['key_insights']:
        console.print("[blue]Key insights:[/blue]")
        for insight in explanations['summary']['key_insights']:
            console.print(f"[blue]  - {insight}[/blue]")

if __name__ == "__main__":
    main()
