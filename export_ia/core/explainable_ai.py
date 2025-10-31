"""
Explainable AI Engine for Export IA
Advanced explainability with SHAP, LIME, Grad-CAM, and attention visualization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
import time
import json
import random
from pathlib import Path
from collections import defaultdict, deque
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image
import shap
import lime
import lime.lime_tabular
import lime.lime_image
import lime.lime_text
from captum.attr import IntegratedGradients, GradientShap, Saliency, InputXGradient
from captum.attr import LayerGradCam, LayerAttribution, GuidedBackprop
from captum.attr import ShapleyValueSampling, FeatureAblation
from captum.attr import Lime, KernelShap
from captum.attr import AttentionRollout, AttentionAblation
from captum.attr import TokenReferenceGenerator
from captum.attr import visualization as viz
import transformers
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)

@dataclass
class ExplainabilityConfig:
    """Configuration for explainable AI"""
    # Explanation methods
    explanation_methods: List[str] = None  # shap, lime, grad_cam, attention, integrated_gradients
    
    # SHAP parameters
    shap_explainer: str = "kernel"  # kernel, tree, deep, linear
    shap_n_samples: int = 100
    shap_background_samples: int = 50
    
    # LIME parameters
    lime_n_samples: int = 1000
    lime_n_features: int = 10
    lime_kernel_width: float = 0.75
    
    # Grad-CAM parameters
    grad_cam_layers: List[str] = None
    grad_cam_upsample: bool = True
    grad_cam_alpha: float = 0.4
    
    # Attention parameters
    attention_layers: List[str] = None
    attention_heads: List[int] = None
    attention_rollout: bool = True
    
    # Integrated Gradients parameters
    ig_steps: int = 50
    ig_baseline: str = "zeros"  # zeros, mean, random
    
    # Visualization parameters
    visualization_methods: List[str] = None  # heatmap, bar, waterfall, force
    save_visualizations: bool = True
    visualization_format: str = "png"  # png, svg, pdf
    colormap: str = "RdBu_r"
    
    # Text explanation parameters
    text_tokenizer: str = "bert"  # bert, roberta, custom
    text_max_length: int = 512
    text_highlight_threshold: float = 0.1
    
    # Image explanation parameters
    image_resize: Tuple[int, int] = (224, 224)
    image_normalize: bool = True
    image_overlay_alpha: float = 0.6
    
    # Tabular explanation parameters
    tabular_feature_names: List[str] = None
    tabular_categorical_features: List[int] = None
    
    # Evaluation parameters
    evaluate_explanations: bool = True
    explanation_metrics: List[str] = None  # faithfulness, stability, complexity
    
    # Logging parameters
    log_explanations: bool = True
    explanation_log_dir: str = "./explanations"

class SHAPExplainer:
    """SHAP-based explainer"""
    
    def __init__(self, config: ExplainabilityConfig):
        self.config = config
        self.explainer = None
        self.background_data = None
        
    def fit(self, model: nn.Module, background_data: torch.Tensor, 
            feature_names: List[str] = None):
        """Fit SHAP explainer"""
        
        self.background_data = background_data
        
        if self.config.shap_explainer == "kernel":
            self.explainer = shap.KernelExplainer(
                self._model_predict, 
                background_data.numpy(),
                feature_names=feature_names
            )
        elif self.config.shap_explainer == "deep":
            self.explainer = shap.DeepExplainer(model, background_data)
        elif self.config.shap_explainer == "linear":
            self.explainer = shap.LinearExplainer(model, background_data)
        else:
            self.explainer = shap.TreeExplainer(model)
            
    def _model_predict(self, X):
        """Model prediction function for SHAP"""
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            predictions = self.model(X_tensor)
            return predictions.numpy()
            
    def explain(self, inputs: torch.Tensor, model: nn.Module) -> Dict[str, Any]:
        """Generate SHAP explanations"""
        
        if self.explainer is None:
            raise ValueError("Explainer not fitted. Call fit() first.")
            
        # Generate SHAP values
        if self.config.shap_explainer == "kernel":
            shap_values = self.explainer.shap_values(
                inputs.numpy(), 
                nsamples=self.config.shap_n_samples
            )
        else:
            shap_values = self.explainer.shap_values(inputs)
            
        # Compute feature importance
        feature_importance = np.abs(shap_values).mean(axis=0)
        
        return {
            'shap_values': shap_values,
            'feature_importance': feature_importance,
            'explanation_type': 'shap'
        }
        
    def visualize(self, shap_values: np.ndarray, inputs: torch.Tensor, 
                  feature_names: List[str] = None, save_path: str = None):
        """Visualize SHAP explanations"""
        
        if self.config.visualization_methods is None:
            self.config.visualization_methods = ['bar', 'waterfall']
            
        for method in self.config.visualization_methods:
            if method == 'bar':
                shap.summary_plot(shap_values, inputs.numpy(), 
                                feature_names=feature_names, show=False)
            elif method == 'waterfall':
                shap.waterfall_plot(shap_values[0], show=False)
            elif method == 'force':
                shap.force_plot(shap_values[0], inputs.numpy()[0], 
                              feature_names=feature_names, show=False)
                              
            if save_path:
                plt.savefig(f"{save_path}_shap_{method}.{self.config.visualization_format}")
                plt.close()

class LIMEExplainer:
    """LIME-based explainer"""
    
    def __init__(self, config: ExplainabilityConfig):
        self.config = config
        self.explainer = None
        
    def fit(self, model: nn.Module, training_data: torch.Tensor, 
            feature_names: List[str] = None, data_type: str = "tabular"):
        """Fit LIME explainer"""
        
        if data_type == "tabular":
            self.explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data.numpy(),
                feature_names=feature_names,
                mode='regression',
                kernel_width=self.config.lime_kernel_width
            )
        elif data_type == "image":
            self.explainer = lime.lime_image.LimeImageExplainer()
        elif data_type == "text":
            self.explainer = lime.lime_text.LimeTextExplainer()
        else:
            raise ValueError(f"Unsupported data type: {data_type}")
            
    def explain(self, inputs: torch.Tensor, model: nn.Module, 
                data_type: str = "tabular") -> Dict[str, Any]:
        """Generate LIME explanations"""
        
        if self.explainer is None:
            raise ValueError("Explainer not fitted. Call fit() first.")
            
        explanations = []
        
        for i in range(inputs.size(0)):
            if data_type == "tabular":
                explanation = self.explainer.explain_instance(
                    inputs[i].numpy(),
                    self._model_predict,
                    num_features=self.config.lime_n_features,
                    num_samples=self.config.lime_n_samples
                )
            elif data_type == "image":
                explanation = self.explainer.explain_instance(
                    inputs[i].numpy().transpose(1, 2, 0),
                    self._model_predict,
                    top_labels=5,
                    hide_color=0,
                    num_samples=self.config.lime_n_samples
                )
            elif data_type == "text":
                explanation = self.explainer.explain_instance(
                    inputs[i],
                    self._model_predict,
                    num_features=self.config.lime_n_features
                )
                
            explanations.append(explanation)
            
        return {
            'explanations': explanations,
            'explanation_type': 'lime'
        }
        
    def _model_predict(self, X):
        """Model prediction function for LIME"""
        
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X_tensor = torch.FloatTensor(X)
            else:
                X_tensor = X
                
            predictions = self.model(X_tensor)
            return predictions.numpy()

class GradCAMExplainer:
    """Grad-CAM-based explainer"""
    
    def __init__(self, config: ExplainabilityConfig):
        self.config = config
        
    def explain(self, model: nn.Module, inputs: torch.Tensor, 
                target_class: int = None) -> Dict[str, Any]:
        """Generate Grad-CAM explanations"""
        
        model.eval()
        inputs.requires_grad_(True)
        
        # Forward pass
        outputs = model(inputs)
        
        if target_class is None:
            target_class = outputs.argmax(dim=1)
            
        # Backward pass
        model.zero_grad()
        outputs[0, target_class].backward()
        
        # Get gradients
        gradients = inputs.grad
        
        # Compute Grad-CAM
        grad_cam = self._compute_grad_cam(gradients, inputs)
        
        return {
            'grad_cam': grad_cam,
            'gradients': gradients,
            'explanation_type': 'grad_cam'
        }
        
    def _compute_grad_cam(self, gradients: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        """Compute Grad-CAM"""
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        
        # Weighted combination of feature maps
        grad_cam = (weights * inputs).sum(dim=1, keepdim=True)
        
        # Apply ReLU
        grad_cam = F.relu(grad_cam)
        
        # Normalize
        grad_cam = (grad_cam - grad_cam.min()) / (grad_cam.max() - grad_cam.min())
        
        return grad_cam
        
    def visualize(self, grad_cam: torch.Tensor, original_image: torch.Tensor, 
                  save_path: str = None):
        """Visualize Grad-CAM"""
        
        # Convert to numpy
        grad_cam_np = grad_cam.squeeze().cpu().numpy()
        original_np = original_image.squeeze().cpu().numpy().transpose(1, 2, 0)
        
        # Resize Grad-CAM to match original image
        grad_cam_resized = cv2.resize(grad_cam_np, (original_np.shape[1], original_np.shape[0]))
        
        # Create heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * grad_cam_resized), cv2.COLORMAP_JET)
        
        # Overlay on original image
        overlay = cv2.addWeighted(
            np.uint8(255 * original_np), 
            self.config.image_overlay_alpha,
            heatmap, 
            1 - self.config.image_overlay_alpha, 
            0
        )
        
        # Display
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(original_np)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(grad_cam_resized, cmap='jet')
        axes[1].set_title('Grad-CAM')
        axes[1].axis('off')
        
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        if save_path:
            plt.savefig(f"{save_path}_gradcam.{self.config.visualization_format}")
            plt.close()

class AttentionExplainer:
    """Attention-based explainer"""
    
    def __init__(self, config: ExplainabilityConfig):
        self.config = config
        
    def explain(self, model: nn.Module, inputs: torch.Tensor, 
                attention_layers: List[str] = None) -> Dict[str, Any]:
        """Generate attention explanations"""
        
        model.eval()
        
        # Forward pass with attention weights
        outputs = model(inputs, output_attentions=True)
        
        if hasattr(outputs, 'attentions'):
            attentions = outputs.attentions
        else:
            # Extract attention weights from model
            attentions = self._extract_attention_weights(model, inputs)
            
        # Compute attention rollout
        if self.config.attention_rollout:
            attention_rollout = self._compute_attention_rollout(attentions)
        else:
            attention_rollout = None
            
        # Compute attention importance
        attention_importance = self._compute_attention_importance(attentions)
        
        return {
            'attentions': attentions,
            'attention_rollout': attention_rollout,
            'attention_importance': attention_importance,
            'explanation_type': 'attention'
        }
        
    def _extract_attention_weights(self, model: nn.Module, inputs: torch.Tensor) -> List[torch.Tensor]:
        """Extract attention weights from model"""
        
        attentions = []
        
        # Hook function to capture attention weights
        def attention_hook(module, input, output):
            if hasattr(output, 'attentions'):
                attentions.append(output.attentions)
            elif isinstance(output, tuple) and len(output) > 1:
                attentions.append(output[1])
                
        # Register hooks
        hooks = []
        for name, module in model.named_modules():
            if 'attention' in name.lower() or 'attn' in name.lower():
                hook = module.register_forward_hook(attention_hook)
                hooks.append(hook)
                
        # Forward pass
        with torch.no_grad():
            _ = model(inputs)
            
        # Remove hooks
        for hook in hooks:
            hook.remove()
            
        return attentions
        
    def _compute_attention_rollout(self, attentions: List[torch.Tensor]) -> torch.Tensor:
        """Compute attention rollout"""
        
        if not attentions:
            return None
            
        # Start with identity matrix
        rollout = torch.eye(attentions[0].size(-1))
        
        # Rollout through layers
        for attention in attentions:
            # Average over heads
            attention_avg = attention.mean(dim=1)
            
            # Apply rollout
            rollout = torch.matmul(attention_avg, rollout)
            
        return rollout
        
    def _compute_attention_importance(self, attentions: List[torch.Tensor]) -> torch.Tensor:
        """Compute attention importance"""
        
        if not attentions:
            return None
            
        # Average attention across layers and heads
        attention_avg = torch.stack(attentions).mean(dim=(0, 1))
        
        # Compute importance as attention magnitude
        importance = attention_avg.sum(dim=0)
        
        return importance
        
    def visualize(self, attention_weights: torch.Tensor, tokens: List[str] = None, 
                  save_path: str = None):
        """Visualize attention weights"""
        
        # Convert to numpy
        attention_np = attention_weights.cpu().numpy()
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(attention_np, 
                   xticklabels=tokens, 
                   yticklabels=tokens,
                   cmap=self.config.colormap,
                   cbar=True)
        
        plt.title('Attention Weights')
        plt.xlabel('Key Tokens')
        plt.ylabel('Query Tokens')
        
        if save_path:
            plt.savefig(f"{save_path}_attention.{self.config.visualization_format}")
            plt.close()

class IntegratedGradientsExplainer:
    """Integrated Gradients explainer"""
    
    def __init__(self, config: ExplainabilityConfig):
        self.config = config
        self.ig = IntegratedGradients(self._model_forward)
        
    def _model_forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Model forward function for Integrated Gradients"""
        
        return self.model(inputs)
        
    def explain(self, model: nn.Module, inputs: torch.Tensor, 
                target_class: int = None) -> Dict[str, Any]:
        """Generate Integrated Gradients explanations"""
        
        self.model = model
        
        # Set target class
        if target_class is None:
            with torch.no_grad():
                outputs = model(inputs)
                target_class = outputs.argmax(dim=1)
                
        # Generate baseline
        baseline = self._generate_baseline(inputs)
        
        # Compute integrated gradients
        attributions = self.ig.attribute(
            inputs,
            baselines=baseline,
            target=target_class,
            n_steps=self.config.ig_steps
        )
        
        return {
            'attributions': attributions,
            'baseline': baseline,
            'explanation_type': 'integrated_gradients'
        }
        
    def _generate_baseline(self, inputs: torch.Tensor) -> torch.Tensor:
        """Generate baseline for Integrated Gradients"""
        
        if self.config.ig_baseline == "zeros":
            return torch.zeros_like(inputs)
        elif self.config.ig_baseline == "mean":
            return torch.mean(inputs, dim=0, keepdim=True)
        elif self.config.ig_baseline == "random":
            return torch.randn_like(inputs)
        else:
            return torch.zeros_like(inputs)

class ExplainableAIEngine:
    """Main Explainable AI Engine"""
    
    def __init__(self, config: ExplainabilityConfig):
        self.config = config
        
        # Initialize explainers
        self.explainers = {}
        
        if "shap" in (self.config.explanation_methods or []):
            self.explainers["shap"] = SHAPExplainer(config)
        if "lime" in (self.config.explanation_methods or []):
            self.explainers["lime"] = LIMEExplainer(config)
        if "grad_cam" in (self.config.explanation_methods or []):
            self.explainers["grad_cam"] = GradCAMExplainer(config)
        if "attention" in (self.config.explanation_methods or []):
            self.explainers["attention"] = AttentionExplainer(config)
        if "integrated_gradients" in (self.config.explanation_methods or []):
            self.explainers["integrated_gradients"] = IntegratedGradientsExplainer(config)
            
        # Explanation storage
        self.explanations = defaultdict(list)
        self.visualizations = defaultdict(list)
        
    def explain(self, model: nn.Module, inputs: torch.Tensor, 
                method: str = None, **kwargs) -> Dict[str, Any]:
        """Generate explanations using specified method"""
        
        if method is None:
            method = list(self.explainers.keys())[0]
            
        if method not in self.explainers:
            raise ValueError(f"Explainer {method} not available")
            
        explainer = self.explainers[method]
        
        # Generate explanation
        explanation = explainer.explain(model, inputs, **kwargs)
        
        # Store explanation
        self.explanations[method].append(explanation)
        
        return explanation
        
    def explain_all_methods(self, model: nn.Module, inputs: torch.Tensor, 
                           **kwargs) -> Dict[str, Any]:
        """Generate explanations using all available methods"""
        
        all_explanations = {}
        
        for method, explainer in self.explainers.items():
            try:
                explanation = explainer.explain(model, inputs, **kwargs)
                all_explanations[method] = explanation
            except Exception as e:
                logger.error(f"Explanation failed for {method}: {e}")
                continue
                
        return all_explanations
        
    def visualize_explanations(self, explanations: Dict[str, Any], 
                              inputs: torch.Tensor, save_path: str = None):
        """Visualize explanations"""
        
        for method, explanation in explanations.items():
            if method in self.explainers:
                explainer = self.explainers[method]
                
                if hasattr(explainer, 'visualize'):
                    try:
                        explainer.visualize(explanation, inputs, save_path)
                    except Exception as e:
                        logger.error(f"Visualization failed for {method}: {e}")
                        
    def evaluate_explanations(self, explanations: Dict[str, Any], 
                             model: nn.Module, inputs: torch.Tensor) -> Dict[str, float]:
        """Evaluate explanation quality"""
        
        if not self.config.evaluate_explanations:
            return {}
            
        evaluation_metrics = {}
        
        for method, explanation in explanations.items():
            method_metrics = {}
            
            # Faithfulness (how well explanation reflects model behavior)
            if 'attributions' in explanation:
                faithfulness = self._compute_faithfulness(explanation['attributions'], model, inputs)
                method_metrics['faithfulness'] = faithfulness
                
            # Stability (consistency across similar inputs)
            stability = self._compute_stability(explanation, model, inputs)
            method_metrics['stability'] = stability
            
            # Complexity (simplicity of explanation)
            complexity = self._compute_complexity(explanation)
            method_metrics['complexity'] = complexity
            
            evaluation_metrics[method] = method_metrics
            
        return evaluation_metrics
        
    def _compute_faithfulness(self, attributions: torch.Tensor, 
                            model: nn.Module, inputs: torch.Tensor) -> float:
        """Compute faithfulness metric"""
        
        # Simplified faithfulness computation
        # In practice, you'd use more sophisticated methods
        
        with torch.no_grad():
            original_output = model(inputs)
            
            # Remove top features and measure performance drop
            top_features = torch.topk(attributions.abs(), k=5, dim=-1).indices
            
            masked_inputs = inputs.clone()
            for i in range(inputs.size(0)):
                masked_inputs[i, top_features[i]] = 0
                
            masked_output = model(masked_inputs)
            
            # Compute performance drop
            performance_drop = torch.mean(torch.abs(original_output - masked_output))
            
        return performance_drop.item()
        
    def _compute_stability(self, explanation: Dict[str, Any], 
                          model: nn.Module, inputs: torch.Tensor) -> float:
        """Compute stability metric"""
        
        # Simplified stability computation
        # In practice, you'd use more sophisticated methods
        
        # Add small noise and measure explanation change
        noise = torch.randn_like(inputs) * 0.01
        noisy_inputs = inputs + noise
        
        # Generate explanation for noisy inputs
        noisy_explanation = self.explain(model, noisy_inputs)
        
        # Compute explanation similarity
        if 'attributions' in explanation and 'attributions' in noisy_explanation:
            similarity = torch.cosine_similarity(
                explanation['attributions'].flatten(),
                noisy_explanation['attributions'].flatten(),
                dim=0
            )
            return similarity.item()
        else:
            return 0.0
            
    def _compute_complexity(self, explanation: Dict[str, Any]) -> float:
        """Compute complexity metric"""
        
        # Simplified complexity computation
        # In practice, you'd use more sophisticated methods
        
        if 'attributions' in explanation:
            attributions = explanation['attributions']
            # Compute sparsity (fraction of non-zero attributions)
            sparsity = (attributions.abs() > 0.01).float().mean()
            return sparsity.item()
        else:
            return 0.0
            
    def save_explanations(self, filepath: str):
        """Save explanations to file"""
        
        explanations_data = {
            'explanations': dict(self.explanations),
            'config': self.config,
            'timestamp': time.time()
        }
        
        with open(filepath, 'w') as f:
            json.dump(explanations_data, f, default=str)
            
    def load_explanations(self, filepath: str):
        """Load explanations from file"""
        
        with open(filepath, 'r') as f:
            explanations_data = json.load(f)
            
        self.explanations = defaultdict(list, explanations_data['explanations'])

# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Test explainable AI
    print("Testing Explainable AI Engine...")
    
    # Create test model
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
            self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(128 * 8 * 8, 256)
            self.fc2 = nn.Linear(256, 10)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.2)
            
        def forward(self, x):
            x = self.pool(self.relu(self.conv1(x)))
            x = self.pool(self.relu(self.conv2(x)))
            x = x.view(x.size(0), -1)
            x = self.dropout(self.relu(self.fc1(x)))
            x = self.fc2(x)
            return x
    
    model = TestModel()
    model.eval()
    
    # Create explainability config
    config = ExplainabilityConfig(
        explanation_methods=["grad_cam", "integrated_gradients"],
        visualization_methods=["heatmap"],
        save_visualizations=True,
        evaluate_explanations=True
    )
    
    # Create explainable AI engine
    xai_engine = ExplainableAIEngine(config)
    
    # Create dummy inputs
    dummy_inputs = torch.randn(1, 3, 32, 32)
    
    # Test explanations
    print("Testing Grad-CAM explanation...")
    grad_cam_explanation = xai_engine.explain(model, dummy_inputs, method="grad_cam")
    print(f"Grad-CAM explanation: {grad_cam_explanation['explanation_type']}")
    
    print("Testing Integrated Gradients explanation...")
    ig_explanation = xai_engine.explain(model, dummy_inputs, method="integrated_gradients")
    print(f"Integrated Gradients explanation: {ig_explanation['explanation_type']}")
    
    # Test all methods
    print("Testing all explanation methods...")
    all_explanations = xai_engine.explain_all_methods(model, dummy_inputs)
    print(f"Generated explanations: {list(all_explanations.keys())}")
    
    # Test evaluation
    print("Testing explanation evaluation...")
    evaluation_metrics = xai_engine.evaluate_explanations(all_explanations, model, dummy_inputs)
    print(f"Evaluation metrics: {evaluation_metrics}")
    
    print("\nExplainable AI engine initialized successfully!")
























