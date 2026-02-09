"""
Model Quality Assurance - Aseguramiento de calidad de modelos
================================================================
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .base_classes import BaseManager, BaseConfig
from .common_utils import (
    get_device, move_to_device, measure_inference_time,
    calculate_model_size
)
from .constants import (
    LATENCY_THRESHOLD_MS, MEMORY_THRESHOLD_MB, ACCURACY_THRESHOLD
)

logger = logging.getLogger(__name__)


class QualityLevel(Enum):
    """Niveles de calidad"""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    FAILED = "failed"


@dataclass
class QualityCheck:
    """Check de calidad"""
    check_name: str
    passed: bool
    score: float
    threshold: float
    message: str
    level: QualityLevel = QualityLevel.ACCEPTABLE


@dataclass
class QualityReport:
    """Reporte de calidad"""
    model_name: str
    overall_quality: QualityLevel
    checks: List[QualityCheck] = field(default_factory=list)
    overall_score: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    recommendations: List[str] = field(default_factory=list)


class ModelQualityAssurance(BaseManager):
    """Aseguramiento de calidad de modelos"""
    
    def __init__(self, config: Optional[BaseConfig] = None):
        super().__init__(config or BaseConfig())
        self.quality_reports: List[QualityReport] = []
        self.thresholds: Dict[str, float] = {
            "accuracy": ACCURACY_THRESHOLD,
            "latency": LATENCY_THRESHOLD_MS,
            "memory": MEMORY_THRESHOLD_MB,
            "robustness": 0.7,
            "fairness": 0.9
        }
    
    def assess_quality(
        self,
        model: nn.Module,
        model_name: str,
        test_loader: Any,
        eval_fn: Optional[Callable] = None,
        device: Optional[str] = None
    ) -> QualityReport:
        """Evalúa calidad del modelo"""
        device_obj = get_device(device)
        checks = []
        
        # Check de accuracy
        if eval_fn:
            accuracy = eval_fn(model)
            accuracy_check = QualityCheck(
                check_name="accuracy",
                passed=accuracy >= self.thresholds["accuracy"],
                score=accuracy,
                threshold=self.thresholds["accuracy"],
                message=f"Accuracy: {accuracy:.4f}",
                level=self._get_quality_level(accuracy, self.thresholds["accuracy"])
            )
            checks.append(accuracy_check)
        
        # Check de latencia usando utilidades compartidas
        latency = self._measure_latency(model, test_loader, str(device_obj))
        latency_check = QualityCheck(
            check_name="latency",
            passed=latency <= self.thresholds["latency"],
            score=latency,
            threshold=self.thresholds["latency"],
            message=f"Latency: {latency:.2f}ms",
            level=self._get_quality_level(self.thresholds["latency"] - latency, 0)
        )
        checks.append(latency_check)
        
        # Check de memoria usando utilidades compartidas
        memory = calculate_model_size(model)
        memory_check = QualityCheck(
            check_name="memory",
            passed=memory <= self.thresholds["memory"],
            score=memory,
            threshold=self.thresholds["memory"],
            message=f"Memory: {memory:.2f}MB",
            level=self._get_quality_level(self.thresholds["memory"] - memory, 0)
        )
        checks.append(memory_check)
        
        # Calcular score general
        passed_checks = sum(1 for c in checks if c.passed)
        overall_score = passed_checks / len(checks) if checks else 0.0
        
        # Determinar nivel de calidad
        if overall_score >= 0.9:
            overall_quality = QualityLevel.EXCELLENT
        elif overall_score >= 0.7:
            overall_quality = QualityLevel.GOOD
        elif overall_score >= 0.5:
            overall_quality = QualityLevel.ACCEPTABLE
        elif overall_score >= 0.3:
            overall_quality = QualityLevel.POOR
        else:
            overall_quality = QualityLevel.FAILED
        
        # Generar recomendaciones
        recommendations = self._generate_recommendations(checks)
        
        report = QualityReport(
            model_name=model_name,
            overall_quality=overall_quality,
            checks=checks,
            overall_score=overall_score,
            recommendations=recommendations
        )
        
        self.quality_reports.append(report)
        self.log_event("quality_assessment", {"model_name": model_name, "quality": overall_quality.value})
        return report
    
    def _measure_latency(
        self,
        model: nn.Module,
        test_loader: Any,
        device: str,
        num_runs: int = 10
    ) -> float:
        """Mide latencia usando utilidades compartidas"""
        try:
            # Obtener ejemplo de input del loader
            batch = next(iter(test_loader))
            if isinstance(batch, dict):
                example_input = batch.get("input_ids") or batch.get("inputs")
            else:
                example_input = batch[0] if isinstance(batch, tuple) else batch
            
            # Usar utilidades compartidas para medir latencia
            return measure_inference_time(
                model, example_input, num_runs=num_runs, warmup=3, device=device
            )
        except Exception as e:
            logger.warning(f"Error midiendo latencia: {e}")
            return 0.0
    
    # _measure_memory removido - ahora usa calculate_model_size de common_utils
    
    def _get_quality_level(self, score: float, threshold: float) -> QualityLevel:
        """Obtiene nivel de calidad"""
        ratio = score / threshold if threshold > 0 else 0
        
        if ratio >= 1.2:
            return QualityLevel.EXCELLENT
        elif ratio >= 1.0:
            return QualityLevel.GOOD
        elif ratio >= 0.8:
            return QualityLevel.ACCEPTABLE
        elif ratio >= 0.5:
            return QualityLevel.POOR
        else:
            return QualityLevel.FAILED
    
    def _generate_recommendations(self, checks: List[QualityCheck]) -> List[str]:
        """Genera recomendaciones"""
        recommendations = []
        
        for check in checks:
            if not check.passed:
                if check.check_name == "accuracy":
                    recommendations.append("Mejorar accuracy: considerar más datos o mejor arquitectura")
                elif check.check_name == "latency":
                    recommendations.append("Optimizar latencia: considerar cuantización o pruning")
                elif check.check_name == "memory":
                    recommendations.append("Reducir memoria: considerar compresión o cuantización")
        
        return recommendations
    
    def get_quality_summary(self) -> Dict[str, Any]:
        """Obtiene resumen de calidad"""
        if not self.quality_reports:
            return {}
        
        latest = self.quality_reports[-1]
        
        return {
            "model_name": latest.model_name,
            "overall_quality": latest.overall_quality.value,
            "overall_score": latest.overall_score,
            "checks_passed": sum(1 for c in latest.checks if c.passed),
            "total_checks": len(latest.checks),
            "recommendations": latest.recommendations
        }

