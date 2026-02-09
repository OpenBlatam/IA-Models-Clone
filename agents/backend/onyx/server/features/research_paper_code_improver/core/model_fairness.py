"""
Model Fairness/Bias Detection - Detección de sesgo y fairness
===============================================================
"""

import logging
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class FairnessMetric:
    """Métrica de fairness"""
    metric_name: str
    value: float
    threshold: float
    passed: bool
    groups: Dict[str, float] = field(default_factory=dict)


@dataclass
class BiasReport:
    """Reporte de sesgo"""
    protected_attribute: str
    metrics: List[FairnessMetric]
    bias_detected: bool
    recommendations: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


class ModelFairnessChecker:
    """Verificador de fairness y sesgo"""
    
    def __init__(self):
        self.fairness_thresholds: Dict[str, float] = {
            "demographic_parity": 0.1,  # 10% difference
            "equalized_odds": 0.1,
            "equal_opportunity": 0.1,
            "calibration": 0.05
        }
        self.bias_reports: List[BiasReport] = []
    
    def check_demographic_parity(
        self,
        predictions: np.ndarray,
        protected_attributes: np.ndarray,
        positive_class: int = 1
    ) -> FairnessMetric:
        """Verifica paridad demográfica"""
        groups = {}
        for group in np.unique(protected_attributes):
            group_mask = protected_attributes == group
            group_positive_rate = (predictions[group_mask] == positive_class).mean()
            groups[str(group)] = float(group_positive_rate)
        
        # Calcular diferencia máxima
        if len(groups) > 1:
            values = list(groups.values())
            max_diff = max(values) - min(values)
        else:
            max_diff = 0.0
        
        threshold = self.fairness_thresholds.get("demographic_parity", 0.1)
        passed = max_diff <= threshold
        
        return FairnessMetric(
            metric_name="demographic_parity",
            value=max_diff,
            threshold=threshold,
            passed=passed,
            groups=groups
        )
    
    def check_equalized_odds(
        self,
        predictions: np.ndarray,
        true_labels: np.ndarray,
        protected_attributes: np.ndarray,
        positive_class: int = 1
    ) -> FairnessMetric:
        """Verifica equalized odds"""
        groups = {}
        for group in np.unique(protected_attributes):
            group_mask = protected_attributes == group
            
            # TPR y FPR por grupo
            group_true = true_labels[group_mask]
            group_pred = predictions[group_mask]
            
            tpr = ((group_pred == positive_class) & (group_true == positive_class)).sum() / \
                  (group_true == positive_class).sum() if (group_true == positive_class).sum() > 0 else 0
            
            fpr = ((group_pred == positive_class) & (group_true != positive_class)).sum() / \
                  (group_true != positive_class).sum() if (group_true != positive_class).sum() > 0 else 0
            
            groups[str(group)] = {"tpr": float(tpr), "fpr": float(fpr)}
        
        # Calcular diferencias
        if len(groups) > 1:
            tprs = [g["tpr"] for g in groups.values()]
            fprs = [g["fpr"] for g in groups.values()]
            max_diff = max(max(tprs) - min(tprs), max(fprs) - min(fprs))
        else:
            max_diff = 0.0
        
        threshold = self.fairness_thresholds.get("equalized_odds", 0.1)
        passed = max_diff <= threshold
        
        return FairnessMetric(
            metric_name="equalized_odds",
            value=max_diff,
            threshold=threshold,
            passed=passed,
            groups=groups
        )
    
    def check_equal_opportunity(
        self,
        predictions: np.ndarray,
        true_labels: np.ndarray,
        protected_attributes: np.ndarray,
        positive_class: int = 1
    ) -> FairnessMetric:
        """Verifica equal opportunity (TPR equality)"""
        groups = {}
        for group in np.unique(protected_attributes):
            group_mask = protected_attributes == group
            
            group_true = true_labels[group_mask]
            group_pred = predictions[group_mask]
            
            tpr = ((group_pred == positive_class) & (group_true == positive_class)).sum() / \
                  (group_true == positive_class).sum() if (group_true == positive_class).sum() > 0 else 0
            
            groups[str(group)] = float(tpr)
        
        if len(groups) > 1:
            values = list(groups.values())
            max_diff = max(values) - min(values)
        else:
            max_diff = 0.0
        
        threshold = self.fairness_thresholds.get("equal_opportunity", 0.1)
        passed = max_diff <= threshold
        
        return FairnessMetric(
            metric_name="equal_opportunity",
            value=max_diff,
            threshold=threshold,
            passed=passed,
            groups=groups
        )
    
    def analyze_bias(
        self,
        model: nn.Module,
        test_data: Any,
        protected_attribute: str,
        protected_values: np.ndarray,
        device: str = "cuda"
    ) -> BiasReport:
        """Analiza sesgo en el modelo"""
        device = torch.device(device)
        model = model.to(device)
        model.eval()
        
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in test_data:
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
                
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                predictions.extend(preds)
                true_labels.extend(labels.cpu().numpy())
        
        predictions = np.array(predictions)
        true_labels = np.array(true_labels)
        
        # Calcular métricas
        metrics = [
            self.check_demographic_parity(predictions, protected_values),
            self.check_equalized_odds(predictions, true_labels, protected_values),
            self.check_equal_opportunity(predictions, true_labels, protected_values)
        ]
        
        bias_detected = any(not m.passed for m in metrics)
        
        # Recomendaciones
        recommendations = []
        if bias_detected:
            recommendations.append("Considerar técnicas de debiasing")
            recommendations.append("Revisar datos de entrenamiento para sesgos")
            recommendations.append("Aplicar post-processing para fairness")
        
        report = BiasReport(
            protected_attribute=protected_attribute,
            metrics=metrics,
            bias_detected=bias_detected,
            recommendations=recommendations
        )
        
        self.bias_reports.append(report)
        return report
    
    def set_fairness_threshold(self, metric_name: str, threshold: float):
        """Establece umbral de fairness"""
        self.fairness_thresholds[metric_name] = threshold




