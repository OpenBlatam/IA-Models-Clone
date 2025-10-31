"""
Advanced Neural Network Explainable AI System for TruthGPT Optimization Core
Complete explainable AI with gradient-based, attention-based, and perturbation-based explanations
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

class ExplanationMethod(Enum):
    """Explanation methods"""
    GRADIENT_BASED = "gradient_based"
    ATTENTION_BASED = "attention_based"
    PERTURBATION_BASED = "perturbation_based"
    LAYER_WISE_RELEVANCE = "layer_wise_relevance"
    INTEGRATED_GRADIENTS = "integrated_gradients"
    GRAD_CAM = "grad_cam"
    LIME = "lime"
    SHAP = "shap"
    COUNTERFACTUAL = "counterfactual"

class ExplanationType(Enum):
    """Explanation types"""
    LOCAL_EXPLANATION = "local_explanation"
    GLOBAL_EXPLANATION = "global_explanation"
    FEATURE_IMPORTANCE = "feature_importance"
    CONCEPT_EXPLANATION = "concept_explanation"
    CAUSAL_EXPLANATION = "causal_explanation"
    CONTRASTIVE_EXPLANATION = "contrastive_explanation"

class VisualizationType(Enum):
    """Visualization types"""
    HEATMAP = "heatmap"
    SALIENCY_MAP = "saliency_map"
    ATTENTION_MAP = "attention_map"
    FEATURE_MAP = "feature_map"
    CONCEPT_MAP = "concept_map"
    CAUSAL_GRAPH = "causal_graph"

class XAIConfig:
    """Configuration for explainable AI system"""
    # Basic settings
    explanation_method: ExplanationMethod = ExplanationMethod.GRADIENT_BASED
    explanation_type: ExplanationType = ExplanationType.LOCAL_EXPLANATION
    visualization_type: VisualizationType = VisualizationType.HEATMAP
    
    # Gradient-based settings
    gradient_method: str = "saliency"
    integrated_gradients_steps: int = 50
    smooth_grad_noise: float = 0.1
    smooth_grad_samples: int = 10
    
    # Attention-based settings
    attention_layers: List[str] = field(default_factory=lambda: ["attention"])
    attention_aggregation: str = "mean"
    
    # Perturbation-based settings
    perturbation_method: str = "occlusion"
    perturbation_size: int = 8
    perturbation_stride: int = 4
    
    # Layer-wise relevance settings
    lrp_rule: str = "alpha_beta"
    lrp_alpha: float = 1.0
    lrp_beta: float = 0.0
    
    # Advanced features
    enable_uncertainty_estimation: bool = True
    enable_concept_analysis: bool = True
    enable_causal_analysis: bool = False
    
    def __post_init__(self):
        """Validate XAI configuration"""
        if self.integrated_gradients_steps <= 0:
            raise ValueError("Integrated gradients steps must be positive")
        if not (0 <= self.smooth_grad_noise <= 1):
            raise ValueError("Smooth grad noise must be between 0 and 1")
        if self.smooth_grad_samples <= 0:
            raise ValueError("Smooth grad samples must be positive")
        if self.perturbation_size <= 0:
            raise ValueError("Perturbation size must be positive")
        if self.perturbation_stride <= 0:
            raise ValueError("Perturbation stride must be positive")
        if not (0 <= self.lrp_alpha <= 1):
            raise ValueError("LRP alpha must be between 0 and 1")
        if not (0 <= self.lrp_beta <= 1):
            raise ValueError("LRP beta must be between 0 and 1")

class GradientExplainer:
    """Gradient-based explanation implementation"""
    
    def __init__(self, config: XAIConfig):
        self.config = config
        self.explanation_history = []
        logger.info("✅ Gradient Explainer initialized")
    
    def compute_saliency(self, model: nn.Module, input_tensor: torch.Tensor, 
                        target_class: int = None) -> torch.Tensor:
        """Compute saliency map"""
        logger.info("🔍 Computing saliency map")
        
        model.eval()
        input_tensor.requires_grad_(True)
        
        # Forward pass
        output = model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1)
        
        # Backward pass
        model.zero_grad()
        output[0, target_class].backward()
        
        # Get gradients
        saliency = input_tensor.grad.abs()
        
        return saliency
    
    def compute_integrated_gradients(self, model: nn.Module, input_tensor: torch.Tensor,
                                   target_class: int = None, baseline: torch.Tensor = None) -> torch.Tensor:
        """Compute integrated gradients"""
        logger.info("🔗 Computing integrated gradients")
        
        if baseline is None:
            baseline = torch.zeros_like(input_tensor)
        
        model.eval()
        
        # Generate interpolated inputs
        interpolated_inputs = []
        for i in range(self.config.integrated_gradients_steps + 1):
            alpha = i / self.config.integrated_gradients_steps
            interpolated = baseline + alpha * (input_tensor - baseline)
            interpolated_inputs.append(interpolated)
        
        interpolated_inputs = torch.stack(interpolated_inputs)
        interpolated_inputs.requires_grad_(True)
        
        # Forward pass
        output = model(interpolated_inputs)
        
        if target_class is None:
            target_class = output[-1].argmax(dim=0)
        
        # Backward pass
        model.zero_grad()
        output[:, target_class].sum().backward()
        
        # Compute integrated gradients
        gradients = interpolated_inputs.grad
        integrated_gradients = gradients.mean(dim=0) * (input_tensor - baseline)
        
        return integrated_gradients
    
    def compute_smooth_grad(self, model: nn.Module, input_tensor: torch.Tensor,
                          target_class: int = None) -> torch.Tensor:
        """Compute smooth grad"""
        logger.info("🌊 Computing smooth grad")
        
        smooth_gradients = []
        
        for _ in range(self.config.smooth_grad_samples):
            # Add noise
            noisy_input = input_tensor + torch.randn_like(input_tensor) * self.config.smooth_grad_noise
            noisy_input.requires_grad_(True)
            
            # Compute gradients
            output = model(noisy_input)
            
            if target_class is None:
                target_class = output.argmax(dim=1)
            
            model.zero_grad()
            output[0, target_class].backward()
            
            smooth_gradients.append(noisy_input.grad)
        
        # Average gradients
        smooth_grad = torch.stack(smooth_gradients).mean(dim=0)
        
        return smooth_grad
    
    def explain(self, model: nn.Module, input_tensor: torch.Tensor, 
                target_class: int = None) -> Dict[str, Any]:
        """Generate gradient-based explanation"""
        logger.info("🎯 Generating gradient-based explanation")
        
        explanation_result = {
            'method': ExplanationMethod.GRADIENT_BASED.value,
            'gradient_method': self.config.gradient_method,
            'explanations': {}
        }
        
        if self.config.gradient_method == "saliency":
            saliency = self.compute_saliency(model, input_tensor, target_class)
            explanation_result['explanations']['saliency'] = saliency
        
        elif self.config.gradient_method == "integrated_gradients":
            integrated_grad = self.compute_integrated_gradients(model, input_tensor, target_class)
            explanation_result['explanations']['integrated_gradients'] = integrated_grad
        
        elif self.config.gradient_method == "smooth_grad":
            smooth_grad = self.compute_smooth_grad(model, input_tensor, target_class)
            explanation_result['explanations']['smooth_grad'] = smooth_grad
        
        # Store explanation
        self.explanation_history.append(explanation_result)
        
        return explanation_result

class AttentionExplainer:
    """Attention-based explanation implementation"""
    
    def __init__(self, config: XAIConfig):
        self.config = config
        self.explanation_history = []
        logger.info("✅ Attention Explainer initialized")
    
    def extract_attention_weights(self, model: nn.Module, input_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract attention weights from model"""
        logger.info("👁️ Extracting attention weights")
        
        attention_weights = {}
        hooks = []
        
        def attention_hook(name):
            def hook(module, input, output):
                if hasattr(output, 'attentions'):
                    attention_weights[name] = output.attentions
                elif isinstance(output, tuple) and len(output) > 1:
                    attention_weights[name] = output[1]  # Assume second output is attention
            return hook
        
        # Register hooks for attention layers
        for name, module in model.named_modules():
            if any(attn_layer in name.lower() for attn_layer in self.config.attention_layers):
                hook = module.register_forward_hook(attention_hook(name))
                hooks.append(hook)
        
        # Forward pass
        with torch.no_grad():
            model(input_tensor)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return attention_weights
    
    def aggregate_attention_weights(self, attention_weights: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Aggregate attention weights"""
        if not attention_weights:
            return torch.tensor([])
        
        if self.config.attention_aggregation == "mean":
            aggregated = torch.stack(list(attention_weights.values())).mean(dim=0)
        elif self.config.attention_aggregation == "max":
            aggregated = torch.stack(list(attention_weights.values())).max(dim=0)[0]
        elif self.config.attention_aggregation == "sum":
            aggregated = torch.stack(list(attention_weights.values())).sum(dim=0)
        else:
            aggregated = torch.stack(list(attention_weights.values())).mean(dim=0)
        
        return aggregated
    
    def explain(self, model: nn.Module, input_tensor: torch.Tensor) -> Dict[str, Any]:
        """Generate attention-based explanation"""
        logger.info("🎯 Generating attention-based explanation")
        
        # Extract attention weights
        attention_weights = self.extract_attention_weights(model, input_tensor)
        
        # Aggregate attention weights
        aggregated_attention = self.aggregate_attention_weights(attention_weights)
        
        explanation_result = {
            'method': ExplanationMethod.ATTENTION_BASED.value,
            'attention_layers': self.config.attention_layers,
            'attention_aggregation': self.config.attention_aggregation,
            'attention_weights': attention_weights,
            'aggregated_attention': aggregated_attention,
            'explanations': {
                'attention_map': aggregated_attention
            }
        }
        
        # Store explanation
        self.explanation_history.append(explanation_result)
        
        return explanation_result

class PerturbationExplainer:
    """Perturbation-based explanation implementation"""
    
    def __init__(self, config: XAIConfig):
        self.config = config
        self.explanation_history = []
        logger.info("✅ Perturbation Explainer initialized")
    
    def compute_occlusion(self, model: nn.Module, input_tensor: torch.Tensor,
                         target_class: int = None) -> torch.Tensor:
        """Compute occlusion-based explanation"""
        logger.info("🔍 Computing occlusion-based explanation")
        
        model.eval()
        
        # Get original prediction
        with torch.no_grad():
            original_output = model(input_tensor)
            if target_class is None:
                target_class = original_output.argmax(dim=1)
            original_score = original_output[0, target_class].item()
        
        # Compute occlusion map
        occlusion_map = torch.zeros_like(input_tensor)
        
        for i in range(0, input_tensor.shape[-2] - self.config.perturbation_size + 1, self.config.perturbation_stride):
            for j in range(0, input_tensor.shape[-1] - self.config.perturbation_size + 1, self.config.perturbation_stride):
                # Create occluded input
                occluded_input = input_tensor.clone()
                occluded_input[0, :, i:i+self.config.perturbation_size, j:j+self.config.perturbation_size] = 0
                
                # Get prediction
                with torch.no_grad():
                    occluded_output = model(occluded_input)
                    occluded_score = occluded_output[0, target_class].item()
                
                # Compute importance
                importance = original_score - occluded_score
                occlusion_map[0, :, i:i+self.config.perturbation_size, j:j+self.config.perturbation_size] = importance
        
        return occlusion_map
    
    def compute_sensitivity_analysis(self, model: nn.Module, input_tensor: torch.Tensor,
                                   target_class: int = None) -> torch.Tensor:
        """Compute sensitivity analysis"""
        logger.info("📊 Computing sensitivity analysis")
        
        model.eval()
        
        # Get original prediction
        with torch.no_grad():
            original_output = model(input_tensor)
            if target_class is None:
                target_class = original_output.argmax(dim=1)
            original_score = original_output[0, target_class].item()
        
        # Compute sensitivity map
        sensitivity_map = torch.zeros_like(input_tensor)
        
        for i in range(input_tensor.shape[-2]):
            for j in range(input_tensor.shape[-1]):
                # Create perturbed input
                perturbed_input = input_tensor.clone()
                perturbed_input[0, :, i, j] += 0.1  # Small perturbation
                
                # Get prediction
                with torch.no_grad():
                    perturbed_output = model(perturbed_input)
                    perturbed_score = perturbed_output[0, target_class].item()
                
                # Compute sensitivity
                sensitivity = abs(perturbed_score - original_score)
                sensitivity_map[0, :, i, j] = sensitivity
        
        return sensitivity_map
    
    def explain(self, model: nn.Module, input_tensor: torch.Tensor,
                target_class: int = None) -> Dict[str, Any]:
        """Generate perturbation-based explanation"""
        logger.info("🎯 Generating perturbation-based explanation")
        
        explanation_result = {
            'method': ExplanationMethod.PERTURBATION_BASED.value,
            'perturbation_method': self.config.perturbation_method,
            'perturbation_size': self.config.perturbation_size,
            'perturbation_stride': self.config.perturbation_stride,
            'explanations': {}
        }
        
        if self.config.perturbation_method == "occlusion":
            occlusion_map = self.compute_occlusion(model, input_tensor, target_class)
            explanation_result['explanations']['occlusion'] = occlusion_map
        
        elif self.config.perturbation_method == "sensitivity":
            sensitivity_map = self.compute_sensitivity_analysis(model, input_tensor, target_class)
            explanation_result['explanations']['sensitivity'] = sensitivity_map
        
        # Store explanation
        self.explanation_history.append(explanation_result)
        
        return explanation_result

class LayerWiseRelevanceExplainer:
    """Layer-wise relevance propagation implementation"""
    
    def __init__(self, config: XAIConfig):
        self.config = config
        self.explanation_history = []
        logger.info("✅ Layer-wise Relevance Explainer initialized")
    
    def compute_lrp(self, model: nn.Module, input_tensor: torch.Tensor,
                   target_class: int = None) -> torch.Tensor:
        """Compute layer-wise relevance propagation"""
        logger.info("🔗 Computing layer-wise relevance propagation")
        
        model.eval()
        
        # Forward pass to get activations
        activations = []
        hooks = []
        
        def activation_hook(name):
            def hook(module, input, output):
                activations.append(output)
            return hook
        
        # Register hooks
        for name, module in model.named_modules():
            hook = module.register_forward_hook(activation_hook(name))
            hooks.append(hook)
        
        # Forward pass
        output = model(input_tensor)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        if target_class is None:
            target_class = output.argmax(dim=1)
        
        # Initialize relevance
        relevance = torch.zeros_like(output)
        relevance[0, target_class] = output[0, target_class]
        
        # Backward pass for relevance propagation
        for i in range(len(activations) - 1, -1, -1):
            if i == 0:
                # Input layer
                relevance = self._propagate_relevance_input(activations[i], relevance)
            else:
                # Hidden layers
                relevance = self._propagate_relevance_layer(activations[i], relevance)
        
        return relevance
    
    def _propagate_relevance_input(self, activation: torch.Tensor, relevance: torch.Tensor) -> torch.Tensor:
        """Propagate relevance to input layer"""
        # Simplified relevance propagation
        return relevance
    
    def _propagate_relevance_layer(self, activation: torch.Tensor, relevance: torch.Tensor) -> torch.Tensor:
        """Propagate relevance through layer"""
        # Simplified relevance propagation
        return relevance
    
    def explain(self, model: nn.Module, input_tensor: torch.Tensor,
                target_class: int = None) -> Dict[str, Any]:
        """Generate layer-wise relevance explanation"""
        logger.info("🎯 Generating layer-wise relevance explanation")
        
        # Compute LRP
        relevance = self.compute_lrp(model, input_tensor, target_class)
        
        explanation_result = {
            'method': ExplanationMethod.LAYER_WISE_RELEVANCE.value,
            'lrp_rule': self.config.lrp_rule,
            'lrp_alpha': self.config.lrp_alpha,
            'lrp_beta': self.config.lrp_beta,
            'explanations': {
                'relevance': relevance
            }
        }
        
        # Store explanation
        self.explanation_history.append(explanation_result)
        
        return explanation_result

class ConceptExplainer:
    """Concept-based explanation implementation"""
    
    def __init__(self, config: XAIConfig):
        self.config = config
        self.explanation_history = []
        logger.info("✅ Concept Explainer initialized")
    
    def extract_concepts(self, model: nn.Module, input_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract concept activations"""
        logger.info("🧠 Extracting concept activations")
        
        concepts = {}
        
        # Extract features from different layers
        with torch.no_grad():
            # Get intermediate features
            for name, module in model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    # Extract features
                    features = module(input_tensor)
                    concepts[name] = features
        
        return concepts
    
    def analyze_concept_importance(self, concepts: Dict[str, torch.Tensor], 
                                 target_class: int) -> Dict[str, float]:
        """Analyze concept importance"""
        logger.info("📊 Analyzing concept importance")
        
        concept_importance = {}
        
        for concept_name, concept_features in concepts.items():
            # Compute concept importance (simplified)
            importance = concept_features.abs().mean().item()
            concept_importance[concept_name] = importance
        
        return concept_importance
    
    def explain(self, model: nn.Module, input_tensor: torch.Tensor,
                target_class: int = None) -> Dict[str, Any]:
        """Generate concept-based explanation"""
        logger.info("🎯 Generating concept-based explanation")
        
        # Extract concepts
        concepts = self.extract_concepts(model, input_tensor)
        
        # Analyze concept importance
        concept_importance = self.analyze_concept_importance(concepts, target_class)
        
        explanation_result = {
            'method': 'concept_explanation',
            'concepts': concepts,
            'concept_importance': concept_importance,
            'explanations': {
                'concept_map': concepts,
                'importance_scores': concept_importance
            }
        }
        
        # Store explanation
        self.explanation_history.append(explanation_result)
        
        return explanation_result

class XAIReportGenerator:
    """XAI report generator"""
    
    def __init__(self, config: XAIConfig):
        self.config = config
        logger.info("✅ XAI Report Generator initialized")
    
    def generate_explanation_report(self, explanations: List[Dict[str, Any]]) -> str:
        """Generate explanation report"""
        report = []
        report.append("=" * 50)
        report.append("EXPLAINABLE AI REPORT")
        report.append("=" * 50)
        
        # Configuration
        report.append("\nXAI CONFIGURATION:")
        report.append("-" * 18)
        report.append(f"Explanation Method: {self.config.explanation_method.value}")
        report.append(f"Explanation Type: {self.config.explanation_type.value}")
        report.append(f"Visualization Type: {self.config.visualization_type.value}")
        report.append(f"Gradient Method: {self.config.gradient_method}")
        report.append(f"Integrated Gradients Steps: {self.config.integrated_gradients_steps}")
        report.append(f"Smooth Grad Noise: {self.config.smooth_grad_noise}")
        report.append(f"Smooth Grad Samples: {self.config.smooth_grad_samples}")
        report.append(f"Attention Layers: {self.config.attention_layers}")
        report.append(f"Attention Aggregation: {self.config.attention_aggregation}")
        report.append(f"Perturbation Method: {self.config.perturbation_method}")
        report.append(f"Perturbation Size: {self.config.perturbation_size}")
        report.append(f"Perturbation Stride: {self.config.perturbation_stride}")
        report.append(f"LRP Rule: {self.config.lrp_rule}")
        report.append(f"LRP Alpha: {self.config.lrp_alpha}")
        report.append(f"LRP Beta: {self.config.lrp_beta}")
        report.append(f"Uncertainty Estimation: {'Enabled' if self.config.enable_uncertainty_estimation else 'Disabled'}")
        report.append(f"Concept Analysis: {'Enabled' if self.config.enable_concept_analysis else 'Disabled'}")
        report.append(f"Causal Analysis: {'Enabled' if self.config.enable_causal_analysis else 'Disabled'}")
        
        # Explanations
        report.append("\nEXPLANATIONS:")
        report.append("-" * 13)
        report.append(f"Number of Explanations: {len(explanations)}")
        
        for i, explanation in enumerate(explanations):
            report.append(f"\nExplanation {i + 1}:")
            report.append("-" * 15)
            report.append(f"  Method: {explanation.get('method', 'Unknown')}")
            
            if 'explanations' in explanation:
                for exp_type, exp_data in explanation['explanations'].items():
                    if isinstance(exp_data, torch.Tensor):
                        report.append(f"  {exp_type}: Shape {exp_data.shape}")
                    else:
                        report.append(f"  {exp_type}: {type(exp_data).__name__}")
        
        return "\n".join(report)
    
    def visualize_explanations(self, explanations: List[Dict[str, Any]], save_path: str = None):
        """Visualize explanations"""
        if not explanations:
            logger.warning("No explanations to visualize")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Explanation methods distribution
        methods = [exp.get('method', 'Unknown') for exp in explanations]
        method_counts = {}
        for method in methods:
            method_counts[method] = method_counts.get(method, 0) + 1
        
        axes[0, 0].pie(method_counts.values(), labels=method_counts.keys(), autopct='%1.1f%%')
        axes[0, 0].set_title('Explanation Methods Distribution')
        
        # Plot 2: Explanation types distribution
        types = [exp.get('explanation_type', 'Unknown') for exp in explanations]
        type_counts = {}
        for exp_type in types:
            type_counts[exp_type] = type_counts.get(exp_type, 0) + 1
        
        axes[0, 1].pie(type_counts.values(), labels=type_counts.keys(), autopct='%1.1f%%')
        axes[0, 1].set_title('Explanation Types Distribution')
        
        # Plot 3: XAI configuration
        config_values = [
            self.config.integrated_gradients_steps,
            self.config.smooth_grad_samples,
            self.config.perturbation_size,
            self.config.perturbation_stride
        ]
        config_labels = ['IG Steps', 'Smooth Grad Samples', 'Perturbation Size', 'Perturbation Stride']
        
        axes[1, 0].bar(config_labels, config_values, color=['blue', 'green', 'orange', 'red'])
        axes[1, 0].set_ylabel('Value')
        axes[1, 0].set_title('XAI Configuration')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot 4: Explanation statistics
        stats_values = [
            len(explanations),
            len(method_counts),
            len(type_counts),
            sum(method_counts.values())
        ]
        stats_labels = ['Total Explanations', 'Unique Methods', 'Unique Types', 'Total Count']
        
        axes[1, 1].bar(stats_labels, stats_values, color=['blue', 'green', 'orange', 'red'])
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Explanation Statistics')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()

class ExplainableAISystem:
    """Main explainable AI system"""
    
    def __init__(self, config: XAIConfig):
        self.config = config
        
        # Components
        self.gradient_explainer = GradientExplainer(config)
        self.attention_explainer = AttentionExplainer(config)
        self.perturbation_explainer = PerturbationExplainer(config)
        self.lrp_explainer = LayerWiseRelevanceExplainer(config)
        self.concept_explainer = ConceptExplainer(config)
        self.report_generator = XAIReportGenerator(config)
        
        # XAI state
        self.xai_history = []
        
        logger.info("✅ Explainable AI System initialized")
    
    def explain_model(self, model: nn.Module, input_tensor: torch.Tensor,
                     target_class: int = None) -> Dict[str, Any]:
        """Explain model predictions"""
        logger.info(f"🎯 Explaining model predictions using method: {self.config.explanation_method.value}")
        
        xai_results = {
            'start_time': time.time(),
            'config': self.config,
            'explanations': []
        }
        
        # Stage 1: Gradient-based Explanation
        if self.config.explanation_method == ExplanationMethod.GRADIENT_BASED:
            logger.info("🔍 Stage 1: Gradient-based Explanation")
            
            gradient_explanation = self.gradient_explainer.explain(model, input_tensor, target_class)
            
            xai_results['explanations'].append(gradient_explanation)
        
        # Stage 2: Attention-based Explanation
        elif self.config.explanation_method == ExplanationMethod.ATTENTION_BASED:
            logger.info("👁️ Stage 2: Attention-based Explanation")
            
            attention_explanation = self.attention_explainer.explain(model, input_tensor)
            
            xai_results['explanations'].append(attention_explanation)
        
        # Stage 3: Perturbation-based Explanation
        elif self.config.explanation_method == ExplanationMethod.PERTURBATION_BASED:
            logger.info("🔍 Stage 3: Perturbation-based Explanation")
            
            perturbation_explanation = self.perturbation_explainer.explain(model, input_tensor, target_class)
            
            xai_results['explanations'].append(perturbation_explanation)
        
        # Stage 4: Layer-wise Relevance Explanation
        elif self.config.explanation_method == ExplanationMethod.LAYER_WISE_RELEVANCE:
            logger.info("🔗 Stage 4: Layer-wise Relevance Explanation")
            
            lrp_explanation = self.lrp_explainer.explain(model, input_tensor, target_class)
            
            xai_results['explanations'].append(lrp_explanation)
        
        # Stage 5: Concept-based Explanation
        elif self.config.explanation_method == ExplanationMethod.CONCEPT_EXPLANATION:
            logger.info("🧠 Stage 5: Concept-based Explanation")
            
            concept_explanation = self.concept_explainer.explain(model, input_tensor, target_class)
            
            xai_results['explanations'].append(concept_explanation)
        
        # Final evaluation
        xai_results['end_time'] = time.time()
        xai_results['total_duration'] = xai_results['end_time'] - xai_results['start_time']
        
        # Store results
        self.xai_history.append(xai_results)
        
        logger.info("✅ Model explanation completed")
        return xai_results
    
    def generate_xai_report(self, results: Dict[str, Any]) -> str:
        """Generate XAI report"""
        return self.report_generator.generate_explanation_report(results['explanations'])
    
    def visualize_xai_results(self, results: Dict[str, Any], save_path: str = None):
        """Visualize XAI results"""
        self.report_generator.visualize_explanations(results['explanations'], save_path)

# Factory functions
def create_xai_config(**kwargs) -> XAIConfig:
    """Create XAI configuration"""
    return XAIConfig(**kwargs)

def create_gradient_explainer(config: XAIConfig) -> GradientExplainer:
    """Create gradient explainer"""
    return GradientExplainer(config)

def create_attention_explainer(config: XAIConfig) -> AttentionExplainer:
    """Create attention explainer"""
    return AttentionExplainer(config)

def create_perturbation_explainer(config: XAIConfig) -> PerturbationExplainer:
    """Create perturbation explainer"""
    return PerturbationExplainer(config)

def create_lrp_explainer(config: XAIConfig) -> LayerWiseRelevanceExplainer:
    """Create LRP explainer"""
    return LayerWiseRelevanceExplainer(config)

def create_concept_explainer(config: XAIConfig) -> ConceptExplainer:
    """Create concept explainer"""
    return ConceptExplainer(config)

def create_xai_report_generator(config: XAIConfig) -> XAIReportGenerator:
    """Create XAI report generator"""
    return XAIReportGenerator(config)

def create_explainable_ai_system(config: XAIConfig) -> ExplainableAISystem:
    """Create explainable AI system"""
    return ExplainableAISystem(config)

# Example usage
def example_explainable_ai():
    """Example of explainable AI system"""
    # Create configuration
    config = create_xai_config(
        explanation_method=ExplanationMethod.GRADIENT_BASED,
        explanation_type=ExplanationType.LOCAL_EXPLANATION,
        visualization_type=VisualizationType.HEATMAP,
        gradient_method="saliency",
        integrated_gradients_steps=50,
        smooth_grad_noise=0.1,
        smooth_grad_samples=10,
        attention_layers=["attention"],
        attention_aggregation="mean",
        perturbation_method="occlusion",
        perturbation_size=8,
        perturbation_stride=4,
        lrp_rule="alpha_beta",
        lrp_alpha=1.0,
        lrp_beta=0.0,
        enable_uncertainty_estimation=True,
        enable_concept_analysis=True,
        enable_causal_analysis=False
    )
    
    # Create explainable AI system
    xai_system = create_explainable_ai_system(config)
    
    # Create dummy model and input
    model = nn.Sequential(
        nn.Conv2d(3, 64, 3, 1, 1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(64, 10)
    )
    
    input_tensor = torch.randn(1, 3, 32, 32)
    target_class = 0
    
    # Explain model
    xai_results = xai_system.explain_model(model, input_tensor, target_class)
    
    # Generate report
    xai_report = xai_system.generate_xai_report(xai_results)
    
    print(f"✅ Explainable AI Example Complete!")
    print(f"🚀 XAI Statistics:")
    print(f"   Explanation Method: {config.explanation_method.value}")
    print(f"   Explanation Type: {config.explanation_type.value}")
    print(f"   Visualization Type: {config.visualization_type.value}")
    print(f"   Gradient Method: {config.gradient_method}")
    print(f"   Integrated Gradients Steps: {config.integrated_gradients_steps}")
    print(f"   Smooth Grad Noise: {config.smooth_grad_noise}")
    print(f"   Smooth Grad Samples: {config.smooth_grad_samples}")
    print(f"   Attention Layers: {config.attention_layers}")
    print(f"   Attention Aggregation: {config.attention_aggregation}")
    print(f"   Perturbation Method: {config.perturbation_method}")
    print(f"   Perturbation Size: {config.perturbation_size}")
    print(f"   Perturbation Stride: {config.perturbation_stride}")
    print(f"   LRP Rule: {config.lrp_rule}")
    print(f"   LRP Alpha: {config.lrp_alpha}")
    print(f"   LRP Beta: {config.lrp_beta}")
    print(f"   Uncertainty Estimation: {'Enabled' if config.enable_uncertainty_estimation else 'Disabled'}")
    print(f"   Concept Analysis: {'Enabled' if config.enable_concept_analysis else 'Disabled'}")
    print(f"   Causal Analysis: {'Enabled' if config.enable_causal_analysis else 'Disabled'}")
    
    print(f"\n📊 XAI Results:")
    print(f"   XAI History Length: {len(xai_system.xai_history)}")
    print(f"   Total Duration: {xai_results.get('total_duration', 0):.2f} seconds")
    print(f"   Number of Explanations: {len(xai_results['explanations'])}")
    
    print(f"\n📋 XAI Report:")
    print(xai_report)
    
    return xai_system

# Export utilities
__all__ = [
    'ExplanationMethod',
    'ExplanationType',
    'VisualizationType',
    'XAIConfig',
    'GradientExplainer',
    'AttentionExplainer',
    'PerturbationExplainer',
    'LayerWiseRelevanceExplainer',
    'ConceptExplainer',
    'XAIReportGenerator',
    'ExplainableAISystem',
    'create_xai_config',
    'create_gradient_explainer',
    'create_attention_explainer',
    'create_perturbation_explainer',
    'create_lrp_explainer',
    'create_concept_explainer',
    'create_xai_report_generator',
    'create_explainable_ai_system',
    'example_explainable_ai'
]

if __name__ == "__main__":
    example_explainable_ai()
    print("✅ Explainable AI example completed successfully!")