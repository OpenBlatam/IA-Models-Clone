"""
Ensemble Models
===============
Model ensemble for improved predictions
"""

from typing import Dict, Any, List, Optional, Union
import torch
import torch.nn as nn
import numpy as np
import structlog
from collections import defaultdict

logger = structlog.get_logger()


class ModelEnsemble:
    """
    Ensemble of models for improved predictions
    """
    
    def __init__(
        self,
        models: List[nn.Module],
        weights: Optional[List[float]] = None,
        voting_strategy: str = "average"  # "average", "weighted", "majority"
    ):
        """
        Initialize ensemble
        
        Args:
            models: List of models
            weights: Weights for each model (None = equal weights)
            voting_strategy: Voting strategy
        """
        self.models = models
        self.voting_strategy = voting_strategy
        
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            # Normalize weights
            total = sum(weights)
            self.weights = [w / total for w in weights]
        
        # Set all models to eval mode
        for model in self.models:
            model.eval()
        
        logger.info(
            "ModelEnsemble initialized",
            num_models=len(models),
            strategy=voting_strategy
        )
    
    def predict(
        self,
        inputs: Dict[str, torch.Tensor],
        return_individual: bool = False
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Predict using ensemble
        
        Args:
            inputs: Model inputs
            return_individual: Return individual predictions
            
        Returns:
            Ensemble predictions
        """
        individual_predictions = []
        
        with torch.no_grad():
            for model in self.models:
                try:
                    outputs = model(**inputs)
                    individual_predictions.append(outputs)
                except Exception as e:
                    logger.error("Error in ensemble model prediction", error=str(e))
                    continue
        
        if not individual_predictions:
            raise ValueError("No valid predictions from ensemble")
        
        # Combine predictions
        if self.voting_strategy == "average":
            ensemble_pred = self._average_predictions(individual_predictions)
        elif self.voting_strategy == "weighted":
            ensemble_pred = self._weighted_predictions(individual_predictions)
        elif self.voting_strategy == "majority":
            ensemble_pred = self._majority_vote(individual_predictions)
        else:
            ensemble_pred = self._average_predictions(individual_predictions)
        
        if return_individual:
            return {
                "ensemble": ensemble_pred,
                "individual": individual_predictions
            }
        
        return ensemble_pred
    
    def _average_predictions(
        self,
        predictions: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """Average predictions"""
        combined = defaultdict(list)
        
        for pred in predictions:
            if isinstance(pred, dict):
                for key, value in pred.items():
                    combined[key].append(value)
            else:
                combined["prediction"].append(pred)
        
        result = {}
        for key, values in combined.items():
            stacked = torch.stack(values)
            result[key] = stacked.mean(dim=0)
        
        return result
    
    def _weighted_predictions(
        self,
        predictions: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """Weighted average predictions"""
        combined = defaultdict(list)
        
        for pred in predictions:
            if isinstance(pred, dict):
                for key, value in pred.items():
                    combined[key].append(value)
            else:
                combined["prediction"].append(pred)
        
        result = {}
        for key, values in combined.items():
            stacked = torch.stack(values)
            # Apply weights
            weights_tensor = torch.tensor(
                self.weights[:len(values)],
                device=stacked.device
            ).view(-1, *([1] * (stacked.dim() - 1)))
            
            result[key] = (stacked * weights_tensor).sum(dim=0)
        
        return result
    
    def _majority_vote(
        self,
        predictions: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """Majority vote for classification"""
        # For classification tasks
        combined = defaultdict(list)
        
        for pred in predictions:
            if isinstance(pred, dict):
                for key, value in pred.items():
                    if "logits" in key or "prediction" in key:
                        # Get class predictions
                        class_preds = torch.argmax(value, dim=-1)
                        combined[key].append(class_preds)
            else:
                class_preds = torch.argmax(pred, dim=-1)
                combined["prediction"].append(class_preds)
        
        result = {}
        for key, values in combined.items():
            stacked = torch.stack(values)
            # Majority vote
            result[key] = torch.mode(stacked, dim=0)[0]
        
        return result


class StackingEnsemble:
    """Stacking ensemble with meta-learner"""
    
    def __init__(
        self,
        base_models: List[nn.Module],
        meta_model: Optional[nn.Module] = None
    ):
        """
        Initialize stacking ensemble
        
        Args:
            base_models: Base models
            meta_model: Meta-learner model (optional)
        """
        self.base_models = base_models
        self.meta_model = meta_model
        
        for model in self.base_models:
            model.eval()
        
        if self.meta_model:
            self.meta_model.eval()
        
        logger.info("StackingEnsemble initialized", num_base_models=len(base_models))
    
    def predict(
        self,
        inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Predict using stacking
        
        Args:
            inputs: Model inputs
            
        Returns:
            Stacking predictions
        """
        # Get base predictions
        base_predictions = []
        
        with torch.no_grad():
            for model in self.base_models:
                outputs = model(**inputs)
                if isinstance(outputs, dict):
                    base_predictions.append(outputs.get("logits", outputs.get("prediction")))
                else:
                    base_predictions.append(outputs)
        
        # Stack base predictions
        stacked = torch.stack(base_predictions, dim=1)  # [batch, num_models, num_classes]
        
        # Meta-learner prediction
        if self.meta_model:
            meta_input = stacked.flatten(start_dim=1)  # Flatten for meta-model
            meta_output = self.meta_model(meta_input)
            return {"prediction": meta_output}
        else:
            # Simple average if no meta-model
            return {"prediction": stacked.mean(dim=1)}


# Global ensemble instances
model_ensemble = None  # Initialize when needed




