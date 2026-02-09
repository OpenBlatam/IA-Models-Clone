"""
Evaluation Module
Implements evaluation metrics and validation procedures.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, List
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import logging

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Evaluator class for model evaluation and metrics computation.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: Optional[str] = None,
        use_amp: bool = True,
    ):
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = use_amp and self.device == "cuda"
        self.model.to(self.device)
        self.model.eval()
    
    def evaluate(
        self,
        data_loader: DataLoader,
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Evaluate model on dataset.
        
        Args:
            data_loader: DataLoader for evaluation
            metrics: List of metrics to compute
            
        Returns:
            Dictionary with evaluation metrics
        """
        if metrics is None:
            metrics = ["loss", "accuracy"]
        
        all_predictions = []
        all_labels = []
        total_loss = 0.0
        num_batches = 0
        
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
        with torch.no_grad():
            for batch in data_loader:
                # Move to device
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                
                # Forward pass
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(**batch)
                else:
                    outputs = self.model(**batch)
                
                # Compute loss
                if "loss" in outputs:
                    loss = outputs["loss"]
                elif "logits" in outputs and "labels" in batch:
                    logits = outputs["logits"]
                    labels = batch["labels"]
                    if len(logits.shape) == 3:
                        logits = logits.view(-1, logits.size(-1))
                        labels = labels.view(-1)
                    loss = criterion(logits, labels)
                else:
                    loss = torch.tensor(0.0)
                
                total_loss += loss.item()
                num_batches += 1
                
                # Get predictions
                if "logits" in outputs and "labels" in batch:
                    logits = outputs["logits"]
                    labels = batch["labels"]
                    
                    if len(logits.shape) == 3:
                        # Sequence model
                        predictions = torch.argmax(logits, dim=-1)
                        # Flatten for metrics
                        predictions = predictions.view(-1).cpu().numpy()
                        labels_flat = labels.view(-1).cpu().numpy()
                        # Filter out padding tokens
                        mask = labels_flat != -100
                        all_predictions.extend(predictions[mask])
                        all_labels.extend(labels_flat[mask])
                    else:
                        # Classification model
                        predictions = torch.argmax(logits, dim=-1)
                        all_predictions.extend(predictions.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())
        
        # Compute metrics
        results = {}
        
        if "loss" in metrics:
            results["loss"] = total_loss / num_batches if num_batches > 0 else 0.0
        
        if all_predictions and all_labels:
            if "accuracy" in metrics:
                results["accuracy"] = accuracy_score(all_labels, all_predictions)
            
            if "precision" in metrics or "recall" in metrics or "f1" in metrics:
                precision, recall, f1, _ = precision_recall_fscore_support(
                    all_labels,
                    all_predictions,
                    average="weighted",
                    zero_division=0,
                )
                if "precision" in metrics:
                    results["precision"] = precision
                if "recall" in metrics:
                    results["recall"] = recall
                if "f1" in metrics:
                    results["f1"] = f1
            
            if "confusion_matrix" in metrics:
                results["confusion_matrix"] = confusion_matrix(
                    all_labels,
                    all_predictions,
                ).tolist()
        
        return results
    
    def compute_perplexity(
        self,
        data_loader: DataLoader,
    ) -> float:
        """
        Compute perplexity for language models.
        
        Args:
            data_loader: DataLoader for evaluation
            
        Returns:
            Perplexity score
        """
        total_loss = 0.0
        total_tokens = 0
        
        criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")
        
        with torch.no_grad():
            for batch in data_loader:
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(**batch)
                else:
                    outputs = self.model(**batch)
                
                if "logits" in outputs and "labels" in batch:
                    logits = outputs["logits"]
                    labels = batch["labels"]
                    
                    if len(logits.shape) == 3:
                        logits = logits.view(-1, logits.size(-1))
                        labels = labels.view(-1)
                        loss = criterion(logits, labels)
                        # Count non-ignored tokens
                        num_tokens = (labels != -100).sum().item()
                        total_loss += loss.item()
                        total_tokens += num_tokens
        
        if total_tokens == 0:
            return float("inf")
        
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return perplexity



