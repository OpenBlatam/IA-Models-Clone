"""
Evaluation Module
================
Comprehensive evaluation metrics and validation
"""

from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
import structlog
from tqdm import tqdm

logger = structlog.get_logger()


class ModelEvaluator:
    """
    Comprehensive model evaluator with multiple metrics
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None
    ):
        """
        Initialize evaluator
        
        Args:
            model: Model to evaluate
            device: Device (optional, auto-detect)
        """
        self.model = model
        from .deep_learning_models import get_device
        self.device = device or get_device()
        self.model = self.model.to(self.device)
        self.model.eval()
        
        logger.info("ModelEvaluator initialized", device=str(self.device))
    
    def evaluate_classification(
        self,
        data_loader: DataLoader,
        num_classes: int = 3
    ) -> Dict[str, float]:
        """
        Evaluate classification model
        
        Args:
            data_loader: Data loader
            num_classes: Number of classes
            
        Returns:
            Classification metrics
        """
        all_predictions = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                try:
                    # Move to device
                    batch = self._move_batch_to_device(batch)
                    
                    # Forward pass
                    outputs = self.model(batch["input_ids"], attention_mask=batch["attention_mask"])
                    
                    # Get predictions
                    if isinstance(outputs, dict):
                        logits = outputs.get("logits", outputs.get("prediction_logits"))
                    else:
                        logits = outputs
                    
                    probs = torch.softmax(logits, dim=-1)
                    predictions = torch.argmax(probs, dim=-1)
                    
                    # Store results
                    all_predictions.extend(predictions.cpu().numpy())
                    all_labels.extend(batch["labels"].cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())
                    
                except Exception as e:
                    logger.error("Error in evaluation step", error=str(e))
                    continue
        
        # Calculate metrics
        metrics = self._calculate_classification_metrics(
            all_labels,
            all_predictions,
            all_probs,
            num_classes
        )
        
        return metrics
    
    def evaluate_regression(
        self,
        data_loader: DataLoader
    ) -> Dict[str, float]:
        """
        Evaluate regression model
        
        Args:
            data_loader: Data loader
            
        Returns:
            Regression metrics
        """
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                try:
                    batch = self._move_batch_to_device(batch)
                    
                    outputs = self.model(batch["input_ids"], attention_mask=batch["attention_mask"])
                    
                    if isinstance(outputs, dict):
                        predictions = outputs.get("predictions", outputs.get("logits"))
                    else:
                        predictions = outputs
                    
                    all_predictions.extend(predictions.cpu().numpy())
                    all_labels.extend(batch["labels"].cpu().numpy())
                    
                except Exception as e:
                    logger.error("Error in evaluation step", error=str(e))
                    continue
        
        # Calculate metrics
        metrics = self._calculate_regression_metrics(all_labels, all_predictions)
        
        return metrics
    
    def evaluate_personality(
        self,
        data_loader: DataLoader
    ) -> Dict[str, Any]:
        """
        Evaluate personality prediction model
        
        Args:
            data_loader: Data loader
            
        Returns:
            Personality-specific metrics
        """
        traits = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
        trait_metrics = {}
        
        all_predictions = {trait: [] for trait in traits}
        all_labels = {trait: [] for trait in traits}
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating personality"):
                try:
                    batch = self._move_batch_to_device(batch)
                    
                    outputs = self.model(batch["input_ids"], attention_mask=batch["attention_mask"])
                    
                    labels = batch.get("personality_labels")
                    if labels is None:
                        continue
                    
                    for i, trait in enumerate(traits):
                        if trait in outputs:
                            pred = outputs[trait].squeeze().cpu().numpy()
                            label = labels[:, i].cpu().numpy()
                            
                            all_predictions[trait].extend(pred)
                            all_labels[trait].extend(label)
                    
                except Exception as e:
                    logger.error("Error in personality evaluation", error=str(e))
                    continue
        
        # Calculate metrics per trait
        for trait in traits:
            if len(all_predictions[trait]) > 0:
                trait_metrics[trait] = self._calculate_regression_metrics(
                    all_labels[trait],
                    all_predictions[trait]
                )
        
        # Overall metrics
        overall_metrics = {
            "mean_mse": np.mean([m.get("mse", 0) for m in trait_metrics.values()]),
            "mean_mae": np.mean([m.get("mae", 0) for m in trait_metrics.values()]),
            "mean_r2": np.mean([m.get("r2", 0) for m in trait_metrics.values()])
        }
        
        return {
            "traits": trait_metrics,
            "overall": overall_metrics
        }
    
    def _calculate_classification_metrics(
        self,
        labels: List[int],
        predictions: List[int],
        probs: List[List[float]],
        num_classes: int
    ) -> Dict[str, float]:
        """Calculate classification metrics"""
        labels = np.array(labels)
        predictions = np.array(predictions)
        probs = np.array(probs)
        
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels,
            predictions,
            average="weighted",
            zero_division=0
        )
        
        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1)
        }
        
        # ROC AUC for binary or multi-class
        try:
            if num_classes == 2:
                auc = roc_auc_score(labels, probs[:, 1])
                metrics["roc_auc"] = float(auc)
            elif num_classes > 2:
                auc = roc_auc_score(labels, probs, multi_class="ovr", average="weighted")
                metrics["roc_auc"] = float(auc)
        except Exception as e:
            logger.warning("Could not calculate ROC AUC", error=str(e))
        
        return metrics
    
    def _calculate_regression_metrics(
        self,
        labels: List[float],
        predictions: List[float]
    ) -> Dict[str, float]:
        """Calculate regression metrics"""
        labels = np.array(labels)
        predictions = np.array(predictions)
        
        mse = mean_squared_error(labels, predictions)
        mae = mean_absolute_error(labels, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(labels, predictions)
        
        # Additional metrics
        mape = np.mean(np.abs((labels - predictions) / (labels + 1e-8))) * 100
        
        return {
            "mse": float(mse),
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
            "mape": float(mape)
        }
    
    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch to device"""
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }


class CrossValidator:
    """Cross-validation for model evaluation"""
    
    @staticmethod
    def k_fold_cross_validate(
        model_class: type,
        dataset: torch.utils.data.Dataset,
        k: int = 5,
        **model_kwargs
    ) -> Dict[str, List[float]]:
        """
        Perform k-fold cross-validation
        
        Args:
            model_class: Model class
            dataset: Full dataset
            k: Number of folds
            **model_kwargs: Model initialization arguments
            
        Returns:
            Cross-validation results
        """
        from sklearn.model_selection import KFold
        
        kfold = KFold(n_splits=k, shuffle=True, random_state=42)
        results = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": []
        }
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
            logger.info(f"Fold {fold+1}/{k}")
            
            # Create splits
            train_subset = torch.utils.data.Subset(dataset, train_idx)
            val_subset = torch.utils.data.Subset(dataset, val_idx)
            
            # Train model
            model = model_class(**model_kwargs)
            # ... training code ...
            
            # Evaluate
            # ... evaluation code ...
        
        return results




