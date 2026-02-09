"""
Model Calibration System - Sistema de calibración de modelos
=============================================================
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from sklearn.calibration import calibration_curve

logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    """Resultado de calibración"""
    ece: float  # Expected Calibration Error
    mce: float  # Maximum Calibration Error
    brier_score: float
    reliability_diagram: Optional[Dict[str, Any]] = None


class ModelCalibrator:
    """Calibrador de modelos"""
    
    def __init__(self):
        self.calibration_results: List[CalibrationResult] = []
    
    def calibrate_temperature_scaling(
        self,
        model: nn.Module,
        val_loader: Any,
        device: str = "cuda"
    ) -> nn.Module:
        """Calibración por temperature scaling"""
        device = torch.device(device)
        model = model.to(device)
        model.eval()
        
        logits_list = []
        labels_list = []
        
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, dict):
                    inputs = batch.get("input_ids") or batch.get("inputs")
                    labels = batch.get("labels") or batch.get("targets")
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = model(**batch)
                else:
                    inputs, labels = batch
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = model(inputs)
                
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs
                
                logits_list.append(logits.cpu())
                labels_list.append(labels.cpu())
        
        logits = torch.cat(logits_list, dim=0)
        labels = torch.cat(labels_list, dim=0)
        
        # Optimizar temperatura
        temperature = nn.Parameter(torch.ones(1))
        optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)
        
        def eval():
            optimizer.zero_grad()
            scaled_logits = logits / temperature
            loss = F.cross_entropy(scaled_logits, labels)
            loss.backward()
            return loss
        
        optimizer.step(eval)
        
        # Crear modelo calibrado
        calibrated_model = TemperatureScaledModel(model, temperature.item())
        logger.info(f"Modelo calibrado con temperatura: {temperature.item():.4f}")
        
        return calibrated_model
    
    def evaluate_calibration(
        self,
        model: nn.Module,
        test_loader: Any,
        device: str = "cuda",
        num_bins: int = 10
    ) -> CalibrationResult:
        """Evalúa calibración del modelo"""
        device = torch.device(device)
        model = model.to(device)
        model.eval()
        
        predictions = []
        confidences = []
        labels_list = []
        
        with torch.no_grad():
            for batch in test_loader:
                if isinstance(batch, dict):
                    inputs = batch.get("input_ids") or batch.get("inputs")
                    labels = batch.get("labels") or batch.get("targets")
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = model(**batch)
                else:
                    inputs, labels = batch
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = model(inputs)
                
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs
                
                probs = F.softmax(logits, dim=-1)
                conf, preds = torch.max(probs, dim=-1)
                
                predictions.extend(preds.cpu().numpy())
                confidences.extend(conf.cpu().numpy())
                labels_list.extend(labels.cpu().numpy())
        
        predictions = np.array(predictions)
        confidences = np.array(confidences)
        labels = np.array(labels_list)
        
        # Calcular ECE y MCE
        ece, mce = self._calculate_ece_mce(confidences, predictions == labels, num_bins)
        
        # Calcular Brier Score
        brier_score = self._calculate_brier_score(confidences, predictions == labels)
        
        # Reliability diagram
        try:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                predictions == labels, confidences, n_bins=num_bins
            )
            reliability_diagram = {
                "fraction_of_positives": fraction_of_positives.tolist(),
                "mean_predicted_value": mean_predicted_value.tolist()
            }
        except:
            reliability_diagram = None
        
        result = CalibrationResult(
            ece=ece,
            mce=mce,
            brier_score=brier_score,
            reliability_diagram=reliability_diagram
        )
        
        self.calibration_results.append(result)
        return result
    
    def _calculate_ece_mce(
        self,
        confidences: np.ndarray,
        correct: np.ndarray,
        num_bins: int = 10
    ) -> Tuple[float, float]:
        """Calcula ECE y MCE"""
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        mce = 0.0
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = correct[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
        
        return float(ece), float(mce)
    
    def _calculate_brier_score(
        self,
        confidences: np.ndarray,
        correct: np.ndarray
    ) -> float:
        """Calcula Brier Score"""
        return float(np.mean((confidences - correct) ** 2))


class TemperatureScaledModel(nn.Module):
    """Modelo con temperature scaling"""
    
    def __init__(self, base_model: nn.Module, temperature: float):
        super().__init__()
        self.base_model = base_model
        self.temperature = temperature
    
    def forward(self, *args, **kwargs):
        logits = self.base_model(*args, **kwargs)
        if hasattr(logits, 'logits'):
            logits.logits = logits.logits / self.temperature
        else:
            logits = logits / self.temperature
        return logits




