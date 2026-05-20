"""
Model Evaluation System - Sistema de evaluación de modelos
===========================================================
Métricas y evaluación completa de modelos
"""

import logging
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from collections import defaultdict

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Sistema de evaluación de modelos"""
    
    def __init__(self):
        self.metrics_history: Dict[str, List[float]] = defaultdict(list)
    
    def evaluate_classification(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        criterion: Optional[nn.Module] = None
    ) -> Dict[str, float]:
        """Evalúa modelo de clasificación"""
        model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, dict):
                    inputs = batch["input"].to(device)
                    labels = batch["label"].to(device)
                else:
                    inputs, labels = batch
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                
                outputs = model(inputs)
                
                if criterion:
                    loss = criterion(outputs, labels)
                    total_loss += loss.item()
                    num_batches += 1
                
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        metrics = {
            "accuracy": accuracy_score(all_labels, all_preds),
            "precision": precision_score(all_labels, all_preds, average="weighted", zero_division=0),
            "recall": recall_score(all_labels, all_preds, average="weighted", zero_division=0),
            "f1_score": f1_score(all_labels, all_preds, average="weighted", zero_division=0)
        }
        
        if criterion:
            metrics["loss"] = total_loss / num_batches if num_batches > 0 else 0.0
        
        return metrics
    
    def evaluate_regression(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        criterion: Optional[nn.Module] = None
    ) -> Dict[str, float]:
        """Evalúa modelo de regresión"""
        model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, dict):
                    inputs = batch["input"].to(device)
                    labels = batch["label"].to(device)
                else:
                    inputs, labels = batch
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                
                outputs = model(inputs)
                
                if criterion:
                    loss = criterion(outputs, labels)
                    total_loss += loss.item()
                    num_batches += 1
                
                all_preds.extend(outputs.cpu().numpy().flatten())
                all_labels.extend(labels.cpu().numpy().flatten())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        metrics = {
            "mse": mean_squared_error(all_labels, all_preds),
            "mae": mean_absolute_error(all_labels, all_preds),
            "rmse": np.sqrt(mean_squared_error(all_labels, all_preds)),
            "r2_score": r2_score(all_labels, all_preds)
        }
        
        if criterion:
            metrics["loss"] = total_loss / num_batches if num_batches > 0 else 0.0
        
        return metrics
    
    def evaluate_generation(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        tokenizer: Any,
        max_length: int = 512
    ) -> Dict[str, float]:
        """Evalúa modelo de generación"""
        model.eval()
        all_generated = []
        all_references = []
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, dict):
                    inputs = batch["input"].to(device)
                    references = batch.get("reference", [])
                else:
                    inputs = batch[0].to(device)
                    references = []
                
                # Generar
                generated = model.generate(
                    inputs,
                    max_length=max_length,
                    num_beams=4,
                    early_stopping=True
                )
                
                # Decodificar
                for gen, ref in zip(generated, references):
                    gen_text = tokenizer.decode(gen, skip_special_tokens=True)
                    all_generated.append(gen_text)
                    if ref:
                        all_references.append(ref)
        
        # Métricas básicas (en producción usar BLEU, ROUGE, etc.)
        avg_length = np.mean([len(text.split()) for text in all_generated])
        
        metrics = {
            "num_samples": len(all_generated),
            "avg_length": avg_length
        }
        
        return metrics
    
    def log_metrics(self, metrics: Dict[str, float], prefix: str = ""):
        """Registra métricas en historial"""
        for key, value in metrics.items():
            full_key = f"{prefix}_{key}" if prefix else key
            self.metrics_history[full_key].append(value)
    
    def get_metrics_history(self) -> Dict[str, List[float]]:
        """Obtiene historial de métricas"""
        return dict(self.metrics_history)




