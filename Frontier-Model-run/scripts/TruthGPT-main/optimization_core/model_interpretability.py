"""
Advanced Neural Network Model Interpretability System for TruthGPT Optimization Core
Complete model interpretability with gradient-based explanations, attention analysis, and perturbation methods
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
import logging
from dataclasses import dataclass, field
from enum import Enum
import time
import json
from pathlib import Path
import pickle
from abc import ABC, abstractmethod
import math
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class InterpretabilityMethod(Enum):
    """Interpretability methods"""
    GRADIENT_BASED = "gradient_based"
    ATTENTION_BASED = "attention_based"
    PERTURBATION_BASED = "perturbation_based"
    LAYER_WISE_RELEVANCE = "layer_wise_relevance"
    INTEGRATED_GRADIENTS = "integrated_gradients"
    GRAD_CAM = "grad_cam"
    LIME = "lime"
    SHAP = "shap"

class ExplanationType(Enum):
    """Explanation types"""
    FEATURE_IMPORTANCE = "feature_importance"
    ATTENTION_MAPS = "attention_maps"
    SALIENCY_MAPS = "saliency_maps"
    ACTIVATION_MAPS = "activation_maps"
    DECISION_BOUNDARIES = "decision_boundaries"
    CONCEPT_ACTIVATION = "concept_activation"

class VisualizationType(Enum):
    """Visualization types"""
    HEATMAP = "heatmap"
    BAR_CHART = "bar_chart"
    SCATTER_PLOT = "scatter_plot"
    LINE_PLOT = "line_plot"
    CONTOUR_PLOT = "contour_plot"
    NETWORK_DIAGRAM = "network_diagram"

class InterpretabilityConfig:
    """Configuration for model interpretability system"""
    # Basic settings
    interpretability_method: InterpretabilityMethod = InterpretabilityMethod.GRADIENT_BASED
    explanation_type: ExplanationType = ExplanationType.FEATURE_IMPORTANCE
    visualization_type: VisualizationType = VisualizationType.HEATMAP
    
    # Gradient-based settings
    gradient_method: str = "integrated_gradients"
    gradient_steps: int = 50
    gradient_baseline: str = "zero"
    
    # Attention-based settings
    attention_layers: List[int] = field(default_factory=lambda: [0, 1, 2])
    attention_heads: List[int] = field(default_factory=lambda: [0, 1, 2, 3])
    attention_aggregation: str = "mean"
    
    # Perturbation-based settings
    perturbation_method: str = "occlusion"
    perturbation_size: int = 1
    perturbation_stride: int = 1
    perturbation_value: float = 0.0
    
    # Layer-wise relevance settings
    relevance_method: str = "lrp"
    relevance_rule: str = "alpha_beta"
    relevance_alpha: float = 1.0
    relevance_beta: float = 0.0
    
    # Visualization settings
    colormap: str = "viridis"
    figure_size: Tuple[int, int] = (10, 8)
    dpi: int = 300
    
    # Advanced features
    enable_feature_importance: bool = True
    enable_attention_analysis: bool = True
    enable_activation_analysis: bool = True
    enable_concept_analysis: bool = False
    enable_uncertainty_analysis: bool = True
    
    def __post_init__(self):
        """Validate interpretability configuration"""
        if self.gradient_steps <= 0:
            raise ValueError("Gradient steps must be positive")
        if self.perturbation_size <= 0:
            raise ValueError("Perturbation size must be positive")
        if self.perturbation_stride <= 0:
            raise ValueError("Perturbation stride must be positive")
        if not (0 <= self.relevance_alpha <= 1):
            raise ValueError("Relevance alpha must be between 0 and 1")
        if not (0 <= self.relevance_beta <= 1):
            raise ValueError("Relevance beta must be between 0 and 1")
        if self.figure_size[0] <= 0 or self.figure_size[1] <= 0:
            raise ValueError("Figure size must be positive")
        if self.dpi <= 0:
            raise ValueError("DPI must be positive")

class GradientExplainer:
    """Gradient-based explanations"""
    
    def __init__(self, config: InterpretabilityConfig):
        self.config = config
        self.explanation_history = []
        logger.info("✅ Gradient Explainer initialized")
    
    def explain(self, model: nn.Module, input_data: torch.Tensor, 
                target_class: int = None) -> Dict[str, Any]:
        """Generate gradient-based explanations"""
        logger.info(f"🔍 Generating gradient-based explanations using {self.config.gradient_method}")
        
        if self.config.gradient_method == "integrated_gradients":
            explanations = self._integrated_gradients(model, input_data, target_class)
        elif self.config.gradient_method == "saliency":
            explanations = self._saliency_maps(model, input_data, target_class)
        elif self.config.gradient_method == "grad_cam":
            explanations = self._grad_cam(model, input_data, target_class)
        else:
            explanations = self._integrated_gradients(model, input_data, target_class)
        
        # Store explanation history
        self.explanation_history.append({
            'gradient_method': self.config.gradient_method,
            'target_class': target_class,
            'explanations': explanations
        })
        
        return explanations
    
    def _integrated_gradients(self, model: nn.Module, input_data: torch.Tensor, 
                             target_class: int = None) -> Dict[str, Any]:
        """Integrated Gradients explanation"""
        logger.info("🔍 Computing Integrated Gradients")
        
        model.eval()
        input_data.requires_grad_(True)
        
        # Create baseline
        if self.config.gradient_baseline == "zero":
            baseline = torch.zeros_like(input_data)
        elif self.config.gradient_baseline == "mean":
            baseline = torch.mean(input_data, dim=0, keepdim=True)
        else:
            baseline = torch.zeros_like(input_data)
        
        # Generate interpolated inputs
        interpolated_inputs = []
        for i in range(self.config.gradient_steps + 1):
            alpha = i / self.config.gradient_steps
            interpolated = baseline + alpha * (input_data - baseline)
            interpolated_inputs.append(interpolated)
        
        interpolated_inputs = torch.cat(interpolated_inputs, dim=0)
        
        # Compute gradients
        gradients = []
        for interpolated_input in interpolated_inputs:
            interpolated_input.requires_grad_(True)
            
            # Forward pass
            output = model(interpolated_input)
            
            if target_class is None:
                target_class = torch.argmax(output, dim=1)
            
            # Backward pass
            model.zero_grad()
            output[0, target_class].backward()
            
            gradients.append(interpolated_input.grad.clone())
        
        gradients = torch.stack(gradients)
        
        # Compute integrated gradients
        integrated_gradients = torch.mean(gradients, dim=0) * (input_data - baseline)
        
        explanations = {
            'integrated_gradients': integrated_gradients,
            'saliency_map': torch.abs(integrated_gradients),
            'feature_importance': torch.sum(torch.abs(integrated_gradients), dim=0)
        }
        
        return explanations
    
    def _saliency_maps(self, model: nn.Module, input_data: torch.Tensor, 
                       target_class: int = None) -> Dict[str, Any]:
        """Saliency maps explanation"""
        logger.info("🔍 Computing Saliency Maps")
        
        model.eval()
        input_data.requires_grad_(True)
        
        # Forward pass
        output = model(input_data)
        
        if target_class is None:
            target_class = torch.argmax(output, dim=1)
        
        # Backward pass
        model.zero_grad()
        output[0, target_class].backward()
        
        # Get gradients
        gradients = input_data.grad
        
        explanations = {
            'gradients': gradients,
            'saliency_map': torch.abs(gradients),
            'feature_importance': torch.sum(torch.abs(gradients), dim=0)
        }
        
        return explanations
    
    def _grad_cam(self, model: nn.Module, input_data: torch.Tensor, 
                  target_class: int = None) -> Dict[str, Any]:
        """Grad-CAM explanation"""
        logger.info("🔍 Computing Grad-CAM")
        
        model.eval()
        
        # Register hooks for feature maps
        feature_maps = []
        gradients = []
        
        def forward_hook(module, input, output):
            feature_maps.append(output)
        
        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0])
        
        # Register hooks on convolutional layers
        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                hook = module.register_forward_hook(forward_hook)
                hooks.append(hook)
                hook = module.register_backward_hook(backward_hook)
                hooks.append(hook)
        
        # Forward pass
        input_data.requires_grad_(True)
        output = model(input_data)
        
        if target_class is None:
            target_class = torch.argmax(output, dim=1)
        
        # Backward pass
        model.zero_grad()
        output[0, target_class].backward()
        
        # Compute Grad-CAM
        if feature_maps and gradients:
            feature_map = feature_maps[-1]
            gradient = gradients[-1]
            
            # Global average pooling of gradients
            weights = torch.mean(gradient, dim=(2, 3), keepdim=True)
            
            # Weighted combination of feature maps
            grad_cam = torch.sum(weights * feature_map, dim=1, keepdim=True)
            grad_cam = F.relu(grad_cam)
            
            # Resize to input size
            grad_cam = F.interpolate(grad_cam, size=input_data.shape[2:], mode='bilinear', align_corners=False)
        else:
            grad_cam = torch.zeros_like(input_data)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        explanations = {
            'grad_cam': grad_cam,
            'feature_maps': feature_maps,
            'gradients': gradients
        }
        
        return explanations

class AttentionExplainer:
    """Attention-based explanations"""
    
    def __init__(self, config: InterpretabilityConfig):
        self.config = config
        self.explanation_history = []
        logger.info("✅ Attention Explainer initialized")
    
    def explain(self, model: nn.Module, input_data: torch.Tensor) -> Dict[str, Any]:
        """Generate attention-based explanations"""
        logger.info("🔍 Generating attention-based explanations")
        
        model.eval()
        
        # Extract attention weights
        attention_weights = self._extract_attention_weights(model, input_data)
        
        # Aggregate attention weights
        aggregated_attention = self._aggregate_attention(attention_weights)
        
        # Generate attention maps
        attention_maps = self._generate_attention_maps(aggregated_attention, input_data)
        
        explanations = {
            'attention_weights': attention_weights,
            'aggregated_attention': aggregated_attention,
            'attention_maps': attention_maps,
            'attention_importance': torch.sum(aggregated_attention, dim=0)
        }
        
        # Store explanation history
        self.explanation_history.append({
            'attention_layers': self.config.attention_layers,
            'attention_heads': self.config.attention_heads,
            'explanations': explanations
        })
        
        return explanations
    
    def _extract_attention_weights(self, model: nn.Module, input_data: torch.Tensor) -> List[torch.Tensor]:
        """Extract attention weights from model"""
        attention_weights = []
        
        def attention_hook(module, input, output):
            if hasattr(module, 'attention_weights'):
                attention_weights.append(module.attention_weights)
            elif isinstance(output, tuple) and len(output) > 1:
                attention_weights.append(output[1])  # Assume attention weights are second output
        
        # Register hooks on attention layers
        hooks = []
        for name, module in model.named_modules():
            if 'attention' in name.lower() or isinstance(module, nn.MultiheadAttention):
                hook = module.register_forward_hook(attention_hook)
                hooks.append(hook)
        
        # Forward pass
        with torch.no_grad():
            _ = model(input_data)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return attention_weights
    
    def _aggregate_attention(self, attention_weights: List[torch.Tensor]) -> torch.Tensor:
        """Aggregate attention weights"""
        if not attention_weights:
            return torch.zeros(1, 1)
        
        # Select layers and heads
        selected_weights = []
        for i, weights in enumerate(attention_weights):
            if i in self.config.attention_layers:
                if len(weights.shape) > 2:  # Multi-head attention
                    selected_heads = weights[:, self.config.attention_heads, :, :]
                    selected_weights.append(selected_heads)
                else:
                    selected_weights.append(weights)
        
        if not selected_weights:
            return torch.zeros(1, 1)
        
        # Aggregate weights
        if self.config.attention_aggregation == "mean":
            aggregated = torch.mean(torch.stack(selected_weights), dim=0)
        elif self.config.attention_aggregation == "max":
            aggregated = torch.max(torch.stack(selected_weights), dim=0)[0]
        elif self.config.attention_aggregation == "sum":
            aggregated = torch.sum(torch.stack(selected_weights), dim=0)
        else:
            aggregated = torch.mean(torch.stack(selected_weights), dim=0)
        
        return aggregated
    
    def _generate_attention_maps(self, attention_weights: torch.Tensor, 
                               input_data: torch.Tensor) -> torch.Tensor:
        """Generate attention maps"""
        if attention_weights.dim() > 2:
            attention_weights = torch.mean(attention_weights, dim=1)  # Average over heads
        
        # Resize attention weights to input size
        if len(attention_weights.shape) == 2:
            attention_maps = F.interpolate(
                attention_weights.unsqueeze(0).unsqueeze(0),
                size=input_data.shape[2:],
                mode='bilinear',
                align_corners=False
            )
        else:
            attention_maps = attention_weights
        
        return attention_maps

class PerturbationExplainer:
    """Perturbation-based explanations"""
    
    def __init__(self, config: InterpretabilityConfig):
        self.config = config
        self.explanation_history = []
        logger.info("✅ Perturbation Explainer initialized")
    
    def explain(self, model: nn.Module, input_data: torch.Tensor, 
                target_class: int = None) -> Dict[str, Any]:
        """Generate perturbation-based explanations"""
        logger.info(f"🔍 Generating perturbation-based explanations using {self.config.perturbation_method}")
        
        if self.config.perturbation_method == "occlusion":
            explanations = self._occlusion_analysis(model, input_data, target_class)
        elif self.config.perturbation_method == "sensitivity":
            explanations = self._sensitivity_analysis(model, input_data, target_class)
        elif self.config.perturbation_method == "shapley":
            explanations = self._shapley_values(model, input_data, target_class)
        else:
            explanations = self._occlusion_analysis(model, input_data, target_class)
        
        # Store explanation history
        self.explanation_history.append({
            'perturbation_method': self.config.perturbation_method,
            'target_class': target_class,
            'explanations': explanations
        })
        
        return explanations
    
    def _occlusion_analysis(self, model: nn.Module, input_data: torch.Tensor, 
                           target_class: int = None) -> Dict[str, Any]:
        """Occlusion analysis"""
        logger.info("🔍 Performing occlusion analysis")
        
        model.eval()
        
        # Get original prediction
        with torch.no_grad():
            original_output = model(input_data)
            if target_class is None:
                target_class = torch.argmax(original_output, dim=1)
            original_score = original_output[0, target_class]
        
        # Perform occlusion
        occlusion_scores = []
        input_shape = input_data.shape
        
        for i in range(0, input_shape[2], self.config.perturbation_stride):
            for j in range(0, input_shape[3], self.config.perturbation_stride):
                # Create occluded input
                occluded_input = input_data.clone()
                occluded_input[0, :, i:i+self.config.perturbation_size, j:j+self.config.perturbation_size] = self.config.perturbation_value
                
                # Get prediction
                with torch.no_grad():
                    occluded_output = model(occluded_input)
                    occluded_score = occluded_output[0, target_class]
                
                # Calculate importance
                importance = original_score - occluded_score
                occlusion_scores.append(importance.item())
        
        # Reshape scores to input dimensions
        occlusion_scores = np.array(occlusion_scores)
        occlusion_map = torch.tensor(occlusion_scores).reshape(input_shape[2], input_shape[3])
        
        explanations = {
            'occlusion_map': occlusion_map,
            'occlusion_scores': occlusion_scores,
            'feature_importance': torch.sum(occlusion_map, dim=0)
        }
        
        return explanations
    
    def _sensitivity_analysis(self, model: nn.Module, input_data: torch.Tensor, 
                             target_class: int = None) -> Dict[str, Any]:
        """Sensitivity analysis"""
        logger.info("🔍 Performing sensitivity analysis")
        
        model.eval()
        
        # Get original prediction
        with torch.no_grad():
            original_output = model(input_data)
            if target_class is None:
                target_class = torch.argmax(original_output, dim=1)
            original_score = original_output[0, target_class]
        
        # Calculate sensitivity
        sensitivity_map = torch.zeros_like(input_data[0])
        
        for i in range(input_data.shape[1]):
            for j in range(input_data.shape[2]):
                for k in range(input_data.shape[3]):
                    # Create perturbed input
                    perturbed_input = input_data.clone()
                    perturbed_input[0, i, j, k] += 0.01  # Small perturbation
                    
                    # Get prediction
                    with torch.no_grad():
                        perturbed_output = model(perturbed_input)
                        perturbed_score = perturbed_output[0, target_class]
                    
                    # Calculate sensitivity
                    sensitivity = abs(perturbed_score - original_score) / 0.01
                    sensitivity_map[i, j, k] = sensitivity
        
        explanations = {
            'sensitivity_map': sensitivity_map,
            'feature_importance': torch.sum(sensitivity_map, dim=0)
        }
        
        return explanations
    
    def _shapley_values(self, model: nn.Module, input_data: torch.Tensor, 
                        target_class: int = None) -> Dict[str, Any]:
        """Shapley values (simplified)"""
        logger.info("🔍 Computing Shapley values")
        
        model.eval()
        
        # Get original prediction
        with torch.no_grad():
            original_output = model(input_data)
            if target_class is None:
                target_class = torch.argmax(original_output, dim=1)
            original_score = original_output[0, target_class]
        
        # Simplified Shapley values calculation
        shapley_values = torch.zeros_like(input_data[0])
        
        # For each feature, calculate marginal contribution
        for i in range(input_data.shape[1]):
            for j in range(input_data.shape[2]):
                for k in range(input_data.shape[3]):
                    # Create input without this feature
                    masked_input = input_data.clone()
                    masked_input[0, i, j, k] = 0
                    
                    # Get prediction without feature
                    with torch.no_grad():
                        masked_output = model(masked_input)
                        masked_score = masked_output[0, target_class]
                    
                    # Calculate marginal contribution
                    marginal_contribution = original_score - masked_score
                    shapley_values[i, j, k] = marginal_contribution
        
        explanations = {
            'shapley_values': shapley_values,
            'feature_importance': torch.sum(shapley_values, dim=0)
        }
        
        return explanations

class FeatureImportanceAnalyzer:
    """Feature importance analysis"""
    
    def __init__(self, config: InterpretabilityConfig):
        self.config = config
        self.analysis_history = []
        logger.info("✅ Feature Importance Analyzer initialized")
    
    def analyze_feature_importance(self, model: nn.Module, input_data: torch.Tensor,
                                 target_class: int = None) -> Dict[str, Any]:
        """Analyze feature importance"""
        logger.info("🔍 Analyzing feature importance")
        
        # Get feature importance from different methods
        gradient_explainer = GradientExplainer(self.config)
        gradient_explanations = gradient_explainer.explain(model, input_data, target_class)
        
        perturbation_explainer = PerturbationExplainer(self.config)
        perturbation_explanations = perturbation_explainer.explain(model, input_data, target_class)
        
        # Combine feature importance
        combined_importance = self._combine_feature_importance(
            gradient_explanations, perturbation_explanations
        )
        
        # Rank features
        feature_ranking = self._rank_features(combined_importance)
        
        analysis_results = {
            'gradient_importance': gradient_explanations.get('feature_importance', torch.zeros(1)),
            'perturbation_importance': perturbation_explanations.get('feature_importance', torch.zeros(1)),
            'combined_importance': combined_importance,
            'feature_ranking': feature_ranking,
            'top_features': feature_ranking[:10]  # Top 10 features
        }
        
        # Store analysis history
        self.analysis_history.append({
            'target_class': target_class,
            'analysis_results': analysis_results
        })
        
        return analysis_results
    
    def _combine_feature_importance(self, gradient_explanations: Dict[str, Any],
                                   perturbation_explanations: Dict[str, Any]) -> torch.Tensor:
        """Combine feature importance from different methods"""
        gradient_importance = gradient_explanations.get('feature_importance', torch.zeros(1))
        perturbation_importance = perturbation_explanations.get('feature_importance', torch.zeros(1))
        
        # Normalize importance scores
        gradient_importance = gradient_importance / (torch.sum(gradient_importance) + 1e-8)
        perturbation_importance = perturbation_importance / (torch.sum(perturbation_importance) + 1e-8)
        
        # Combine with equal weights
        combined_importance = 0.5 * gradient_importance + 0.5 * perturbation_importance
        
        return combined_importance
    
    def _rank_features(self, feature_importance: torch.Tensor) -> List[int]:
        """Rank features by importance"""
        importance_scores = feature_importance.flatten()
        ranked_indices = torch.argsort(importance_scores, descending=True)
        
        return ranked_indices.tolist()

class XAIReportGenerator:
    """XAI report generator"""
    
    def __init__(self, config: InterpretabilityConfig):
        self.config = config
        self.report_history = []
        logger.info("✅ XAI Report Generator initialized")
    
    def generate_report(self, explanations: Dict[str, Any], 
                       model_info: Dict[str, Any] = None) -> str:
        """Generate comprehensive XAI report"""
        logger.info("📋 Generating XAI report")
        
        report = []
        report.append("=" * 60)
        report.append("EXPLAINABLE AI (XAI) ANALYSIS REPORT")
        report.append("=" * 60)
        
        # Model information
        if model_info:
            report.append("\nMODEL INFORMATION:")
            report.append("-" * 18)
            for key, value in model_info.items():
                report.append(f"{key}: {value}")
        
        # Configuration
        report.append("\nINTERPRETABILITY CONFIGURATION:")
        report.append("-" * 32)
        report.append(f"Interpretability Method: {self.config.interpretability_method.value}")
        report.append(f"Explanation Type: {self.config.explanation_type.value}")
        report.append(f"Visualization Type: {self.config.visualization_type.value}")
        report.append(f"Gradient Method: {self.config.gradient_method}")
        report.append(f"Gradient Steps: {self.config.gradient_steps}")
        report.append(f"Gradient Baseline: {self.config.gradient_baseline}")
        report.append(f"Attention Layers: {self.config.attention_layers}")
        report.append(f"Attention Heads: {self.config.attention_heads}")
        report.append(f"Attention Aggregation: {self.config.attention_aggregation}")
        report.append(f"Perturbation Method: {self.config.perturbation_method}")
        report.append(f"Perturbation Size: {self.config.perturbation_size}")
        report.append(f"Perturbation Stride: {self.config.perturbation_stride}")
        report.append(f"Perturbation Value: {self.config.perturbation_value}")
        report.append(f"Relevance Method: {self.config.relevance_method}")
        report.append(f"Relevance Rule: {self.config.relevance_rule}")
        report.append(f"Relevance Alpha: {self.config.relevance_alpha}")
        report.append(f"Relevance Beta: {self.config.relevance_beta}")
        report.append(f"Colormap: {self.config.colormap}")
        report.append(f"Figure Size: {self.config.figure_size}")
        report.append(f"DPI: {self.config.dpi}")
        report.append(f"Feature Importance: {'Enabled' if self.config.enable_feature_importance else 'Disabled'}")
        report.append(f"Attention Analysis: {'Enabled' if self.config.enable_attention_analysis else 'Disabled'}")
        report.append(f"Activation Analysis: {'Enabled' if self.config.enable_activation_analysis else 'Disabled'}")
        report.append(f"Concept Analysis: {'Enabled' if self.config.enable_concept_analysis else 'Disabled'}")
        report.append(f"Uncertainty Analysis: {'Enabled' if self.config.enable_uncertainty_analysis else 'Disabled'}")
        
        # Explanation results
        report.append("\nEXPLANATION RESULTS:")
        report.append("-" * 20)
        
        for method, results in explanations.items():
            report.append(f"\n{method.upper()}:")
            report.append("-" * len(method))
            
            if isinstance(results, dict):
                for key, value in results.items():
                    if isinstance(value, torch.Tensor):
                        report.append(f"  {key}: Tensor shape {value.shape}")
                    else:
                        report.append(f"  {key}: {value}")
            else:
                report.append(f"  Results: {results}")
        
        # Store report history
        self.report_history.append({
            'timestamp': time.time(),
            'explanations': explanations,
            'model_info': model_info
        })
        
        return "\n".join(report)
    
    def visualize_explanations(self, explanations: Dict[str, Any], 
                             save_path: str = None):
        """Visualize explanations"""
        logger.info("📊 Visualizing explanations")
        
        n_methods = len(explanations)
        if n_methods == 0:
            logger.warning("No explanations to visualize")
            return
        
        fig, axes = plt.subplots(2, (n_methods + 1) // 2, figsize=self.config.figure_size)
        if n_methods == 1:
            axes = [axes]
        elif n_methods == 2:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for i, (method, results) in enumerate(explanations.items()):
            if i >= len(axes):
                break
            
            ax = axes[i]
            
            if 'saliency_map' in results:
                # Plot saliency map
                saliency_map = results['saliency_map'].squeeze().detach().numpy()
                im = ax.imshow(saliency_map, cmap=self.config.colormap)
                ax.set_title(f'{method} - Saliency Map')
                plt.colorbar(im, ax=ax)
            
            elif 'attention_map' in results:
                # Plot attention map
                attention_map = results['attention_map'].squeeze().detach().numpy()
                im = ax.imshow(attention_map, cmap=self.config.colormap)
                ax.set_title(f'{method} - Attention Map')
                plt.colorbar(im, ax=ax)
            
            elif 'feature_importance' in results:
                # Plot feature importance
                importance = results['feature_importance'].squeeze().detach().numpy()
                ax.bar(range(len(importance)), importance)
                ax.set_title(f'{method} - Feature Importance')
                ax.set_xlabel('Feature Index')
                ax.set_ylabel('Importance')
            
            else:
                # Default plot
                ax.text(0.5, 0.5, f'{method}\nNo visualization available', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{method}')
        
        # Hide unused subplots
        for i in range(n_methods, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()

class ExplainableAISystem:
    """Main explainable AI system"""
    
    def __init__(self, config: InterpretabilityConfig):
        self.config = config
        
        # Components
        self.gradient_explainer = GradientExplainer(config)
        self.attention_explainer = AttentionExplainer(config)
        self.perturbation_explainer = PerturbationExplainer(config)
        self.feature_analyzer = FeatureImportanceAnalyzer(config)
        self.report_generator = XAIReportGenerator(config)
        
        # XAI state
        self.xai_history = []
        
        logger.info("✅ Explainable AI System initialized")
    
    def explain_model(self, model: nn.Module, input_data: torch.Tensor,
                     target_class: int = None) -> Dict[str, Any]:
        """Explain model predictions"""
        logger.info(f"🔍 Explaining model using {self.config.interpretability_method.value}")
        
        xai_results = {
            'start_time': time.time(),
            'config': self.config,
            'explanations': {}
        }
        
        # Generate explanations based on method
        if self.config.interpretability_method == InterpretabilityMethod.GRADIENT_BASED:
            logger.info("🔍 Stage 1: Gradient-based explanations")
            
            gradient_explanations = self.gradient_explainer.explain(model, input_data, target_class)
            xai_results['explanations']['gradient'] = gradient_explanations
        
        elif self.config.interpretability_method == InterpretabilityMethod.ATTENTION_BASED:
            logger.info("🔍 Stage 1: Attention-based explanations")
            
            attention_explanations = self.attention_explainer.explain(model, input_data)
            xai_results['explanations']['attention'] = attention_explanations
        
        elif self.config.interpretability_method == InterpretabilityMethod.PERTURBATION_BASED:
            logger.info("🔍 Stage 1: Perturbation-based explanations")
            
            perturbation_explanations = self.perturbation_explainer.explain(model, input_data, target_class)
            xai_results['explanations']['perturbation'] = perturbation_explanations
        
        else:
            # Generate all types of explanations
            logger.info("🔍 Stage 1: Comprehensive explanations")
            
            gradient_explanations = self.gradient_explainer.explain(model, input_data, target_class)
            attention_explanations = self.attention_explainer.explain(model, input_data)
            perturbation_explanations = self.perturbation_explainer.explain(model, input_data, target_class)
            
            xai_results['explanations']['gradient'] = gradient_explanations
            xai_results['explanations']['attention'] = attention_explanations
            xai_results['explanations']['perturbation'] = perturbation_explanations
        
        # Feature importance analysis
        if self.config.enable_feature_importance:
            logger.info("🔍 Stage 2: Feature importance analysis")
            
            feature_analysis = self.feature_analyzer.analyze_feature_importance(
                model, input_data, target_class
            )
            xai_results['explanations']['feature_analysis'] = feature_analysis
        
        # Generate report
        logger.info("📋 Stage 3: Generating XAI report")
        
        model_info = {
            'model_type': type(model).__name__,
            'num_parameters': sum(p.numel() for p in model.parameters()),
            'input_shape': input_data.shape,
            'target_class': target_class
        }
        
        xai_report = self.report_generator.generate_report(
            xai_results['explanations'], model_info
        )
        xai_results['report'] = xai_report
        
        # Final evaluation
        xai_results['end_time'] = time.time()
        xai_results['total_duration'] = xai_results['end_time'] - xai_results['start_time']
        
        # Store results
        self.xai_history.append(xai_results)
        
        logger.info("✅ Model explanation completed")
        return xai_results
    
    def visualize_explanations(self, xai_results: Dict[str, Any], save_path: str = None):
        """Visualize explanations"""
        logger.info("📊 Visualizing explanations")
        
        self.report_generator.visualize_explanations(
            xai_results['explanations'], save_path
        )
    
    def generate_xai_report(self, xai_results: Dict[str, Any]) -> str:
        """Generate XAI report"""
        return xai_results.get('report', 'No report available')

# Factory functions
def create_interpretability_config(**kwargs) -> InterpretabilityConfig:
    """Create interpretability configuration"""
    return InterpretabilityConfig(**kwargs)

def create_gradient_explainer(config: InterpretabilityConfig) -> GradientExplainer:
    """Create gradient explainer"""
    return GradientExplainer(config)

def create_attention_explainer(config: InterpretabilityConfig) -> AttentionExplainer:
    """Create attention explainer"""
    return AttentionExplainer(config)

def create_perturbation_explainer(config: InterpretabilityConfig) -> PerturbationExplainer:
    """Create perturbation explainer"""
    return PerturbationExplainer(config)

def create_feature_importance_analyzer(config: InterpretabilityConfig) -> FeatureImportanceAnalyzer:
    """Create feature importance analyzer"""
    return FeatureImportanceAnalyzer(config)

def create_xai_report_generator(config: InterpretabilityConfig) -> XAIReportGenerator:
    """Create XAI report generator"""
    return XAIReportGenerator(config)

def create_explainable_ai_system(config: InterpretabilityConfig) -> ExplainableAISystem:
    """Create explainable AI system"""
    return ExplainableAISystem(config)

# Example usage
def example_explainable_ai():
    """Example of explainable AI system"""
    # Create configuration
    config = create_interpretability_config(
        interpretability_method=InterpretabilityMethod.GRADIENT_BASED,
        explanation_type=ExplanationType.FEATURE_IMPORTANCE,
        visualization_type=VisualizationType.HEATMAP,
        gradient_method="integrated_gradients",
        gradient_steps=50,
        gradient_baseline="zero",
        attention_layers=[0, 1, 2],
        attention_heads=[0, 1, 2, 3],
        attention_aggregation="mean",
        perturbation_method="occlusion",
        perturbation_size=1,
        perturbation_stride=1,
        perturbation_value=0.0,
        relevance_method="lrp",
        relevance_rule="alpha_beta",
        relevance_alpha=1.0,
        relevance_beta=0.0,
        colormap="viridis",
        figure_size=(10, 8),
        dpi=300,
        enable_feature_importance=True,
        enable_attention_analysis=True,
        enable_activation_analysis=True,
        enable_concept_analysis=False,
        enable_uncertainty_analysis=True
    )
    
    # Create explainable AI system
    xai_system = create_explainable_ai_system(config)
    
    # Create dummy model
    model = nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(64, 10)
    )
    
    # Generate dummy data
    input_data = torch.randn(1, 3, 32, 32)
    target_class = 5
    
    # Explain model
    xai_results = xai_system.explain_model(model, input_data, target_class)
    
    # Generate report
    xai_report = xai_system.generate_xai_report(xai_results)
    
    print(f"✅ Explainable AI Example Complete!")
    print(f"🚀 Explainable AI Statistics:")
    print(f"   Interpretability Method: {config.interpretability_method.value}")
    print(f"   Explanation Type: {config.explanation_type.value}")
    print(f"   Visualization Type: {config.visualization_type.value}")
    print(f"   Gradient Method: {config.gradient_method}")
    print(f"   Gradient Steps: {config.gradient_steps}")
    print(f"   Gradient Baseline: {config.gradient_baseline}")
    print(f"   Attention Layers: {config.attention_layers}")
    print(f"   Attention Heads: {config.attention_heads}")
    print(f"   Attention Aggregation: {config.attention_aggregation}")
    print(f"   Perturbation Method: {config.perturbation_method}")
    print(f"   Perturbation Size: {config.perturbation_size}")
    print(f"   Perturbation Stride: {config.perturbation_stride}")
    print(f"   Perturbation Value: {config.perturbation_value}")
    print(f"   Relevance Method: {config.relevance_method}")
    print(f"   Relevance Rule: {config.relevance_rule}")
    print(f"   Relevance Alpha: {config.relevance_alpha}")
    print(f"   Relevance Beta: {config.relevance_beta}")
    print(f"   Colormap: {config.colormap}")
    print(f"   Figure Size: {config.figure_size}")
    print(f"   DPI: {config.dpi}")
    print(f"   Feature Importance: {'Enabled' if config.enable_feature_importance else 'Disabled'}")
    print(f"   Attention Analysis: {'Enabled' if config.enable_attention_analysis else 'Disabled'}")
    print(f"   Activation Analysis: {'Enabled' if config.enable_activation_analysis else 'Disabled'}")
    print(f"   Concept Analysis: {'Enabled' if config.enable_concept_analysis else 'Disabled'}")
    print(f"   Uncertainty Analysis: {'Enabled' if config.enable_uncertainty_analysis else 'Disabled'}")
    
    print(f"\n📊 Explainable AI Results:")
    print(f"   XAI History Length: {len(xai_system.xai_history)}")
    print(f"   Total Duration: {xai_results.get('total_duration', 0):.2f} seconds")
    
    # Show explanation results summary
    if 'explanations' in xai_results:
        print(f"   Number of Explanation Methods: {len(xai_results['explanations'])}")
    
    print(f"\n📋 Explainable AI Report:")
    print(xai_report)
    
    return xai_system

# Export utilities
__all__ = [
    'InterpretabilityMethod',
    'ExplanationType',
    'VisualizationType',
    'InterpretabilityConfig',
    'GradientExplainer',
    'AttentionExplainer',
    'PerturbationExplainer',
    'FeatureImportanceAnalyzer',
    'XAIReportGenerator',
    'ExplainableAISystem',
    'create_interpretability_config',
    'create_gradient_explainer',
    'create_attention_explainer',
    'create_perturbation_explainer',
    'create_feature_importance_analyzer',
    'create_xai_report_generator',
    'create_explainable_ai_system',
    'example_explainable_ai'
]

if __name__ == "__main__":
    example_explainable_ai()
    print("✅ Explainable AI example completed successfully!")