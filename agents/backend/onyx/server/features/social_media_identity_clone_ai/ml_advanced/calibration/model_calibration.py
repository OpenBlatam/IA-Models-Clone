"""
Calibración de modelos para probabilidades confiables
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
import numpy as np
from sklearn.calibration import calibration_curve
import logging

logger = logging.getLogger(__name__)


class TemperatureScaling:
    """Temperature Scaling para calibración"""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1))
    
    def calibrate(
        self,
        val_logits: torch.Tensor,
        val_labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 50
    ):
        """
        Calibra temperatura en validation set
        
        Args:
            val_logits: Logits de validación
            val_labels: Labels de validación
            lr: Learning rate
            max_iter: Máximo de iteraciones
        """
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        
        def eval():
            optimizer.zero_grad()
            scaled_logits = val_logits / self.temperature
            loss = nn.CrossEntropyLoss()(scaled_logits, val_labels)
            loss.backward()
            return loss
        
        optimizer.step(eval)
        logger.info(f"Temperatura calibrada: {self.temperature.item():.4f}")
    
    def predict_proba(self, logits: torch.Tensor) -> torch.Tensor:
        """Predice probabilidades calibradas"""
        scaled_logits = logits / self.temperature
        return F.softmax(scaled_logits, dim=-1)


class PlattScaling:
    """Platt Scaling (Logistic Regression)"""
    
    def __init__(self):
        from sklearn.linear_model import LogisticRegression
        self.calibrator = LogisticRegression()
        self.is_fitted = False
    
    def fit(self, logits: torch.Tensor, labels: torch.Tensor):
        """Ajusta calibrador"""
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        labels_np = labels.cpu().numpy()
        
        # Usar probabilidad de clase positiva
        if probs.shape[1] == 2:
            self.calibrator.fit(probs[:, 1:2], labels_np)
        else:
            # Multi-class: usar max probability
            max_probs = probs.max(axis=1, keepdims=True)
            self.calibrator.fit(max_probs, labels_np)
        
        self.is_fitted = True
    
    def predict_proba(self, logits: torch.Tensor) -> torch.Tensor:
        """Predice probabilidades calibradas"""
        if not self.is_fitted:
            return F.softmax(logits, dim=-1)
        
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        
        if probs.shape[1] == 2:
            calibrated = self.calibrator.predict_proba(probs[:, 1:2])
            return torch.tensor(calibrated)
        else:
            max_probs = probs.max(axis=1, keepdims=True)
            calibrated = self.calibrator.predict_proba(max_probs)
            return torch.tensor(calibrated)


class CalibrationEvaluator:
    """Evaluador de calibración"""
    
    def __init__(self):
        pass
    
    def evaluate_calibration(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        n_bins: int = 10
    ) -> Dict[str, float]:
        """
        Evalúa calibración usando ECE (Expected Calibration Error)
        
        Args:
            probs: Probabilidades predichas
            labels: Labels verdaderos
            n_bins: Número de bins
            
        Returns:
            Métricas de calibración
        """
        if probs.ndim > 1:
            probs = probs.max(axis=1)  # Usar max probability
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        accuracies = []
        confidences = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (probs > bin_lower) & (probs <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = labels[in_bin].mean()
                avg_confidence_in_bin = probs[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                
                accuracies.append(accuracy_in_bin)
                confidences.append(avg_confidence_in_bin)
        
        return {
            "ece": float(ece),
            "mean_accuracy": float(np.mean(accuracies)) if accuracies else 0.0,
            "mean_confidence": float(np.mean(confidences)) if confidences else 0.0
        }




