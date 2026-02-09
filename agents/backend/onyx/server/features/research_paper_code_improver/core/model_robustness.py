"""
Model Robustness Tester - Probador de robustez de modelos
==========================================================
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from .base_classes import BaseManager, BaseConfig
from .common_utils import (
    get_device, move_to_device, get_model_output,
    extract_predictions, calculate_accuracy
)
from .constants import ACCURACY_THRESHOLD

logger = logging.getLogger(__name__)


@dataclass
class RobustnessTestResult:
    """Resultado de test de robustez"""
    test_type: str
    accuracy: float
    robustness_score: float
    passed: bool
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class ModelRobustnessTester(BaseManager):
    """Probador de robustez de modelos"""
    
    def __init__(self, config: Optional[BaseConfig] = None):
        super().__init__(config or BaseConfig())
        self.test_results: List[RobustnessTestResult] = []
        self.thresholds: Dict[str, float] = {
            "noise": ACCURACY_THRESHOLD,
            "adversarial": 0.6,
            "corruption": 0.7,
            "distribution_shift": 0.75
        }
    
    def test_noise_robustness(
        self,
        model: nn.Module,
        test_loader: Any,
        noise_level: float = 0.1,
        device: Optional[str] = None
    ) -> RobustnessTestResult:
        """Prueba robustez a ruido usando utilidades compartidas"""
        device_obj = get_device(device)
        model = model.to(device_obj)
        model.eval()
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                # Mover batch a device usando utilidades compartidas
                batch = move_to_device(batch, device_obj)
                
                # Obtener inputs y labels
                if isinstance(batch, dict):
                    inputs = batch.get("input_ids") or batch.get("inputs")
                    labels = batch.get("labels") or batch.get("targets")
                else:
                    inputs, labels = batch[0], batch[1]
                
                # Agregar ruido
                noise = torch.randn_like(inputs) * noise_level
                noisy_inputs = inputs + noise
                noisy_inputs = torch.clamp(noisy_inputs, 0, 1)
                
                # Predicción usando utilidades compartidas
                outputs = get_model_output(model, noisy_inputs, str(device_obj))
                predictions = extract_predictions(outputs)
                
                all_predictions.append(predictions)
                all_labels.append(labels)
        
        # Calcular accuracy usando utilidades compartidas
        if all_predictions and all_labels:
            predictions_tensor = torch.cat(all_predictions)
            labels_tensor = torch.cat(all_labels)
            accuracy = calculate_accuracy(predictions_tensor, labels_tensor)
        else:
            accuracy = 0.0
        threshold = self.thresholds.get("noise", 0.8)
        passed = accuracy >= threshold
        
        result = RobustnessTestResult(
            test_type="noise",
            accuracy=accuracy,
            robustness_score=accuracy,
            passed=passed,
            details={"noise_level": noise_level}
        )
        
        self.test_results.append(result)
        self.log_event("robustness_test", {"test_type": "noise", "accuracy": accuracy})
        return result
    
    def test_corruption_robustness(
        self,
        model: nn.Module,
        test_loader: Any,
        corruption_type: str = "gaussian",
        severity: int = 1,
        device: str = "cuda"
    ) -> RobustnessTestResult:
        """Prueba robustez a corrupción"""
        device = torch.device(device)
        model = model.to(device)
        model.eval()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in test_loader:
                if isinstance(batch, dict):
                    inputs = batch.get("input_ids") or batch.get("inputs")
                    labels = batch.get("labels") or batch.get("targets")
                else:
                    inputs, labels = batch
                
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Aplicar corrupción
                corrupted_inputs = self._apply_corruption(inputs, corruption_type, severity)
                
                # Predicción
                outputs = model(corrupted_inputs)
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs
                
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        accuracy = correct / total if total > 0 else 0
        threshold = self.thresholds.get("corruption", 0.7)
        passed = accuracy >= threshold
        
        result = RobustnessTestResult(
            test_type="corruption",
            accuracy=accuracy,
            robustness_score=accuracy,
            passed=passed,
            details={"corruption_type": corruption_type, "severity": severity}
        )
        
        self.test_results.append(result)
        return result
    
    def _apply_corruption(
        self,
        inputs: torch.Tensor,
        corruption_type: str,
        severity: int
    ) -> torch.Tensor:
        """Aplica corrupción a inputs"""
        if corruption_type == "gaussian":
            noise = torch.randn_like(inputs) * (severity * 0.1)
            return torch.clamp(inputs + noise, 0, 1)
        elif corruption_type == "salt_pepper":
            mask = torch.rand_like(inputs) < (severity * 0.1)
            corrupted = inputs.clone()
            corrupted[mask] = torch.randint(0, 2, corrupted[mask].shape, device=inputs.device).float()
            return corrupted
        else:
            return inputs
    
    def get_robustness_summary(self) -> Dict[str, Any]:
        """Obtiene resumen de robustez"""
        if not self.test_results:
            return {}
        
        passed_tests = sum(1 for r in self.test_results if r.passed)
        avg_robustness = sum(r.robustness_score for r in self.test_results) / len(self.test_results)
        
        return {
            "total_tests": len(self.test_results),
            "passed_tests": passed_tests,
            "pass_rate": passed_tests / len(self.test_results) if self.test_results else 0,
            "avg_robustness_score": avg_robustness,
            "test_results": [
                {
                    "test_type": r.test_type,
                    "accuracy": r.accuracy,
                    "robustness_score": r.robustness_score,
                    "passed": r.passed
                }
                for r in self.test_results
            ]
        }

