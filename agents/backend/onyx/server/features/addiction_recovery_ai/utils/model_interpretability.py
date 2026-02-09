"""
Model Interpretability using SHAP and LIME
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not available. Install: pip install shap")

try:
    from lime import lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    logger.warning("LIME not available. Install: pip install lime")


class ModelInterpreter:
    """Model interpretability using SHAP and LIME"""
    
    def __init__(
        self,
        model: torch.nn.Module,
        background_data: torch.Tensor,
        feature_names: Optional[List[str]] = None
    ):
        """
        Initialize model interpreter
        
        Args:
            model: Model to interpret
            background_data: Background data for SHAP
            feature_names: Optional feature names
        """
        self.model = model
        self.background_data = background_data
        self.feature_names = feature_names or [f"feature_{i}" for i in range(background_data.shape[1])]
        
        self.shap_explainer = None
        self.lime_explainer = None
        
        logger.info("ModelInterpreter initialized")
    
    def create_shap_explainer(self, method: str = "deep"):
        """
        Create SHAP explainer
        
        Args:
            method: SHAP method (deep, gradient, kernel)
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is required for interpretability")
        
        self.model.eval()
        
        if method == "deep":
            self.shap_explainer = shap.DeepExplainer(
                self.model,
                self.background_data
            )
        elif method == "gradient":
            self.shap_explainer = shap.GradientExplainer(
                self.model,
                self.background_data
            )
        else:
            # Kernel SHAP
            def model_fn(x):
                self.model.eval()
                with torch.no_grad():
                    return self.model(torch.tensor(x, dtype=torch.float32)).numpy()
            
            self.shap_explainer = shap.KernelExplainer(
                model_fn,
                self.background_data.numpy()
            )
        
        logger.info(f"SHAP explainer created: {method}")
    
    def explain_shap(
        self,
        instances: torch.Tensor,
        max_evals: int = 100
    ) -> Dict[str, Any]:
        """
        Explain predictions using SHAP
        
        Args:
            instances: Instances to explain
            max_evals: Maximum evaluations
        
        Returns:
            Explanation dictionary
        """
        if self.shap_explainer is None:
            self.create_shap_explainer()
        
        shap_values = self.shap_explainer.shap_values(instances)
        
        # Convert to dictionary
        explanations = []
        for i, instance in enumerate(instances):
            explanation = {
                "instance": instance.tolist(),
                "shap_values": shap_values[i].tolist() if isinstance(shap_values, list) else shap_values[i].tolist(),
                "feature_names": self.feature_names,
                "feature_importance": dict(zip(
                    self.feature_names,
                    shap_values[i] if isinstance(shap_values, list) else shap_values[i]
                ))
            }
            explanations.append(explanation)
        
        return {
            "explanations": explanations,
            "method": "SHAP"
        }
    
    def create_lime_explainer(self, training_data: np.ndarray):
        """
        Create LIME explainer
        
        Args:
            training_data: Training data for LIME
        """
        if not LIME_AVAILABLE:
            raise ImportError("LIME is required for interpretability")
        
        def model_fn(x):
            self.model.eval()
            with torch.no_grad():
                return self.model(torch.tensor(x, dtype=torch.float32)).numpy()
        
        self.lime_explainer = lime_tabular.LimeTabularExplainer(
            training_data,
            feature_names=self.feature_names,
            mode="regression"
        )
        
        logger.info("LIME explainer created")
    
    def explain_lime(
        self,
        instance: torch.Tensor,
        num_features: int = 10
    ) -> Dict[str, Any]:
        """
        Explain prediction using LIME
        
        Args:
            instance: Instance to explain
            num_features: Number of features to show
        
        Returns:
            Explanation dictionary
        """
        if self.lime_explainer is None:
            raise ValueError("LIME explainer not created. Call create_lime_explainer first.")
        
        def model_fn(x):
            self.model.eval()
            with torch.no_grad():
                return self.model(torch.tensor(x, dtype=torch.float32)).numpy()
        
        explanation = self.lime_explainer.explain_instance(
            instance.numpy().flatten(),
            model_fn,
            num_features=num_features
        )
        
        # Extract explanation
        exp_list = explanation.as_list()
        
        return {
            "instance": instance.tolist(),
            "explanation": exp_list,
            "feature_importance": dict(exp_list),
            "method": "LIME"
        }
    
    def get_feature_importance(
        self,
        instances: torch.Tensor,
        method: str = "shap"
    ) -> Dict[str, float]:
        """
        Get feature importance
        
        Args:
            instances: Instances to analyze
            method: Method (shap or lime)
        
        Returns:
            Feature importance dictionary
        """
        if method == "shap":
            explanations = self.explain_shap(instances)
            # Aggregate importance
            all_importances = []
            for exp in explanations["explanations"]:
                all_importances.append(exp["feature_importance"])
            
            # Average importance
            avg_importance = {}
            for feature in self.feature_names:
                avg_importance[feature] = np.mean([
                    imp.get(feature, 0) for imp in all_importances
                ])
            
            return avg_importance
        else:
            # LIME
            importances = {}
            for instance in instances:
                explanation = self.explain_lime(instance)
                for feature, importance in explanation["feature_importance"].items():
                    if feature not in importances:
                        importances[feature] = []
                    importances[feature].append(importance)
            
            # Average
            return {k: np.mean(v) for k, v in importances.items()}


class AttentionVisualizer:
    """Visualize attention weights for transformer models"""
    
    def __init__(self, model: torch.nn.Module):
        """
        Initialize attention visualizer
        
        Args:
            model: Transformer model
        """
        self.model = model
        self.attention_weights = []
    
    def get_attention(
        self,
        inputs: torch.Tensor,
        return_attentions: bool = True
    ) -> List[torch.Tensor]:
        """
        Get attention weights
        
        Args:
            inputs: Input tensor
            return_attentions: Whether to return attentions
        
        Returns:
            List of attention weights
        """
        self.model.eval()
        with torch.no_grad():
            # Forward pass with attention
            outputs = self.model(inputs, output_attentions=return_attentions)
            
            if hasattr(outputs, 'attentions'):
                return outputs.attentions
            elif isinstance(outputs, tuple) and len(outputs) > 1:
                return outputs[1]  # Assume attentions are second output
            else:
                return []

