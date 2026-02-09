"""
Model Validation System - Sistema de validación de modelos
============================================================
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime

from .base_classes import BaseEvaluator, BaseConfig
from .common_utils import (
    get_device, move_to_device, get_model_output,
    extract_predictions, calculate_accuracy
)
from .constants import ACCURACY_THRESHOLD

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Resultado de validación"""
    metric_name: str
    value: float
    threshold: float
    passed: bool
    message: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario"""
        return {
            "metric_name": self.metric_name,
            "value": self.value,
            "threshold": self.threshold,
            "passed": self.passed,
            "message": self.message,
            "timestamp": self.timestamp.isoformat()
        }


class ModelValidator(BaseEvaluator):
    """Validador de modelos"""
    
    def __init__(self, config: Optional[BaseConfig] = None):
        super().__init__(config or BaseConfig())
        self.validation_results: List[ValidationResult] = []
        self.thresholds: Dict[str, float] = {
            "accuracy": ACCURACY_THRESHOLD,
            "f1_score": 0.75,
            "precision": 0.7,
            "recall": 0.7,
            "loss": 0.5
        }
    
    def validate_model(
        self,
        model: nn.Module,
        test_loader: Any,
        device: Optional[str] = None,
        loss_fn: Optional[Callable] = None
    ) -> List[ValidationResult]:
        """Valida un modelo usando utilidades compartidas"""
        results = []
        device_obj = get_device(device)
        model = model.to(device_obj)
        model.eval()
        
        # Calcular métricas usando utilidades compartidas
        metrics = self._calculate_metrics(model, test_loader, device_obj, loss_fn)
        
        # Validar cada métrica
        for metric_name, value in metrics.items():
            threshold = self.thresholds.get(metric_name, 0.0)
            passed = value >= threshold if metric_name != "loss" else value <= threshold
            
            result = ValidationResult(
                metric_name=metric_name,
                value=value,
                threshold=threshold,
                passed=passed,
                message=f"{metric_name}: {value:.4f} {'>=' if metric_name != 'loss' else '<='} {threshold:.4f}"
            )
            
            results.append(result)
            self.validation_results.append(result)
            self.evaluation_results.append({"metric": metric_name, "value": value, "passed": passed})
            
            if not passed:
                logger.warning(f"Validación falló: {result.message}")
            else:
                logger.info(f"Validación pasó: {result.message}")
        
        self.log_event("model_validation", {"total_metrics": len(results), "passed": sum(1 for r in results if r.passed)})
        return results
    
    def _calculate_metrics(
        self,
        model: nn.Module,
        test_loader: Any,
        device: torch.device,
        loss_fn: Optional[Callable]
    ) -> Dict[str, float]:
        """Calcula métricas del modelo"""
        correct = 0
        total = 0
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                # Mover batch a device usando utilidades compartidas
                batch = move_to_device(batch, device)
                
                # Obtener labels
                if isinstance(batch, dict):
                    labels = batch.get("labels") or batch.get("targets")
                else:
                    labels = batch[1] if isinstance(batch, tuple) else None
                
                # Obtener outputs usando utilidades compartidas
                outputs = get_model_output(model, batch, str(device))
                
                if loss_fn and labels is not None:
                    loss = loss_fn(outputs, labels)
                    total_loss += loss.item()
                
                # Extraer predicciones usando utilidades compartidas
                predictions = extract_predictions(outputs)
                
                if labels is not None:
                    total += labels.size(0)
                    correct += (predictions == labels).sum().item()
                    all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
        
        accuracy = correct / total if total > 0 else 0
        
        # Calcular métricas adicionales
        try:
            from sklearn.metrics import precision_score, recall_score, f1_score
            
            precision = precision_score(all_targets, all_predictions, average='weighted', zero_division=0)
            recall = recall_score(all_targets, all_predictions, average='weighted', zero_division=0)
            f1 = f1_score(all_targets, all_predictions, average='weighted', zero_division=0)
        except ImportError:
            precision = recall = f1 = 0.0
        
        avg_loss = total_loss / len(test_loader) if test_loader else 0
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "loss": avg_loss
        }
    
    def set_threshold(self, metric_name: str, threshold: float):
        """Establece umbral para una métrica"""
        self.thresholds[metric_name] = threshold
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Obtiene resumen de validación"""
        if not self.validation_results:
            return {}
        
        passed_count = sum(1 for r in self.validation_results if r.passed)
        total_count = len(self.validation_results)
        
        return {
            "total_validations": total_count,
            "passed": passed_count,
            "failed": total_count - passed_count,
            "pass_rate": passed_count / total_count if total_count > 0 else 0,
            "results": [r.to_dict() for r in self.validation_results]
        }
    
    def clear_results(self):
        """Limpia resultados"""
        self.validation_results.clear()

