"""
Sistema de Cost Analysis
==========================

Sistema para análisis de costos de modelos.
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class CostType(Enum):
    """Tipo de costo"""
    TRAINING = "training"
    INFERENCE = "inference"
    STORAGE = "storage"
    COMPUTE = "compute"
    DATA = "data"


@dataclass
class CostBreakdown:
    """Desglose de costos"""
    breakdown_id: str
    model_id: str
    cost_type: CostType
    cost_amount: float
    currency: str
    period: str  # daily, weekly, monthly
    timestamp: str


@dataclass
class CostReport:
    """Reporte de costos"""
    report_id: str
    model_id: str
    total_cost: float
    breakdown: List[CostBreakdown]
    recommendations: List[str]
    timestamp: str


class CostAnalysis:
    """
    Sistema de Cost Analysis
    
    Proporciona:
    - Análisis de costos de modelos
    - Desglose de costos por tipo
    - Estimación de costos
    - Optimización de costos
    - Reportes de costos
    """
    
    def __init__(self):
        """Inicializar sistema"""
        self.reports: Dict[str, CostReport] = {}
        self.cost_history: List[Dict[str, Any]] = []
        logger.info("CostAnalysis inicializado")
    
    def analyze_costs(
        self,
        model_id: str,
        period: str = "monthly"
    ) -> CostReport:
        """
        Analizar costos de modelo
        
        Args:
            model_id: ID del modelo
            period: Período (daily, weekly, monthly)
        
        Returns:
            Reporte de costos
        """
        report_id = f"cost_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Simulación de análisis de costos
        breakdown = [
            CostBreakdown(
                breakdown_id=f"cost_{i}",
                model_id=model_id,
                cost_type=CostType.TRAINING if i == 0 else CostType.INFERENCE,
                cost_amount=100.0 + i * 50.0,
                currency="USD",
                period=period,
                timestamp=datetime.now().isoformat()
            )
            for i in range(4)
        ]
        
        total_cost = sum(cb.cost_amount for cb in breakdown)
        
        recommendations = [
            "Usar mixed precision para reducir costos de entrenamiento",
            "Implementar caching para reducir inferencias redundantes",
            "Considerar cuantización para reducir costos de almacenamiento"
        ]
        
        report = CostReport(
            report_id=report_id,
            model_id=model_id,
            total_cost=total_cost,
            breakdown=breakdown,
            recommendations=recommendations,
            timestamp=datetime.now().isoformat()
        )
        
        self.reports[report_id] = report
        
        logger.info(f"Análisis de costos completado: {model_id} - Total: ${total_cost:.2f}")
        
        return report
    
    def estimate_training_cost(
        self,
        model_size_mb: float,
        training_hours: float,
        compute_cost_per_hour: float = 2.0
    ) -> float:
        """
        Estimar costo de entrenamiento
        
        Args:
            model_size_mb: Tamaño del modelo en MB
            training_hours: Horas de entrenamiento
            compute_cost_per_hour: Costo por hora de cómputo
        
        Returns:
            Costo estimado
        """
        base_cost = training_hours * compute_cost_per_hour
        storage_cost = model_size_mb * 0.01  # $0.01 por MB
        
        total_cost = base_cost + storage_cost
        
        logger.info(f"Costo estimado de entrenamiento: ${total_cost:.2f}")
        
        return total_cost
    
    def estimate_inference_cost(
        self,
        num_requests: int,
        avg_latency_ms: float,
        cost_per_1k_requests: float = 0.10
    ) -> float:
        """
        Estimar costo de inferencia
        
        Args:
            num_requests: Número de requests
            avg_latency_ms: Latencia promedio en ms
            cost_per_1k_requests: Costo por 1000 requests
        
        Returns:
            Costo estimado
        """
        cost = (num_requests / 1000) * cost_per_1k_requests
        
        logger.info(f"Costo estimado de inferencia: ${cost:.2f}")
        
        return cost


# Instancia global
_cost_analysis: Optional[CostAnalysis] = None


def get_cost_analysis() -> CostAnalysis:
    """Obtener instancia global del sistema"""
    global _cost_analysis
    if _cost_analysis is None:
        _cost_analysis = CostAnalysis()
    return _cost_analysis


