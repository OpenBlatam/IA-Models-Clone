"""
Cost Optimizer - Optimizador de Costos
======================================

Sistema para optimizar costos basado en uso y predicciones.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class CostRecord:
    """Registro de costo."""
    cost_id: str
    resource_type: str
    cost: float
    usage: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CostOptimizationSuggestion:
    """Sugerencia de optimización de costos."""
    suggestion_id: str
    resource_type: str
    current_cost: float
    estimated_savings: float
    recommendation: str
    confidence: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


class CostOptimizer:
    """Optimizador de costos."""
    
    def __init__(self):
        self.cost_records: List[CostRecord] = []
        self.cost_per_unit: Dict[str, float] = {}
        self.suggestions: List[CostOptimizationSuggestion] = []
        self._lock = asyncio.Lock()
    
    def set_cost_per_unit(
        self,
        resource_type: str,
        cost_per_unit: float,
    ):
        """Establecer costo por unidad de recurso."""
        self.cost_per_unit[resource_type] = cost_per_unit
        logger.info(f"Set cost per unit for {resource_type}: ${cost_per_unit:.4f}")
    
    async def record_cost(
        self,
        resource_type: str,
        usage: float,
        cost: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Registrar costo de uso.
        
        Args:
            resource_type: Tipo de recurso
            usage: Cantidad de uso
            cost: Costo (opcional, se calcula si no se proporciona)
            metadata: Metadatos adicionales
        """
        if cost is None:
            cost_per_unit = self.cost_per_unit.get(resource_type, 0.0)
            cost = usage * cost_per_unit
        
        record = CostRecord(
            cost_id=f"cost_{resource_type}_{datetime.now().timestamp()}",
            resource_type=resource_type,
            cost=cost,
            usage=usage,
            timestamp=datetime.now(),
            metadata=metadata or {},
        )
        
        async with self._lock:
            self.cost_records.append(record)
            if len(self.cost_records) > 10000:
                self.cost_records = self.cost_records[-10000:]
    
    async def analyze_and_suggest(
        self,
        resource_type: Optional[str] = None,
    ) -> List[CostOptimizationSuggestion]:
        """
        Analizar costos y generar sugerencias.
        
        Args:
            resource_type: Tipo de recurso (opcional, todos si None)
        
        Returns:
            Lista de sugerencias
        """
        records = self.cost_records
        
        if resource_type:
            records = [r for r in records if r.resource_type == resource_type]
        
        if not records:
            return []
        
        # Agrupar por tipo de recurso
        by_resource: Dict[str, List[CostRecord]] = defaultdict(list)
        for record in records:
            by_resource[record.resource_type].append(record)
        
        suggestions = []
        
        for res_type, res_records in by_resource.items():
            # Calcular costos promedio
            total_cost = sum(r.cost for r in res_records)
            total_usage = sum(r.usage for r in res_records)
            avg_cost_per_unit = total_cost / total_usage if total_usage > 0 else 0.0
            
            # Buscar oportunidades de optimización
            recent_records = res_records[-100:] if len(res_records) > 100 else res_records
            recent_avg = sum(r.cost for r in recent_records) / len(recent_records)
            
            # Si el costo reciente es significativamente mayor que el promedio
            if recent_avg > total_cost / len(res_records) * 1.2:
                estimated_savings = recent_avg - (total_cost / len(res_records))
                suggestion = CostOptimizationSuggestion(
                    suggestion_id=f"suggestion_{res_type}_{datetime.now().timestamp()}",
                    resource_type=res_type,
                    current_cost=recent_avg,
                    estimated_savings=estimated_savings,
                    recommendation=f"Recent costs are {((recent_avg / (total_cost / len(res_records))) - 1) * 100:.1f}% higher than average. Consider optimizing usage patterns.",
                    confidence=0.7,
                    timestamp=datetime.now(),
                    metadata={
                        "avg_cost_per_unit": avg_cost_per_unit,
                        "total_usage": total_usage,
                        "sample_size": len(res_records),
                    },
                )
                suggestions.append(suggestion)
        
        async with self._lock:
            self.suggestions.extend(suggestions)
            if len(self.suggestions) > 1000:
                self.suggestions = self.suggestions[-1000:]
        
        return suggestions
    
    def get_cost_summary(
        self,
        resource_type: Optional[str] = None,
        days: int = 30,
    ) -> Dict[str, Any]:
        """Obtener resumen de costos."""
        cutoff = datetime.now() - timedelta(days=days)
        records = [
            r for r in self.cost_records
            if r.timestamp >= cutoff
        ]
        
        if resource_type:
            records = [r for r in records if r.resource_type == resource_type]
        
        if not records:
            return {
                "total_cost": 0.0,
                "total_usage": 0.0,
                "avg_cost_per_unit": 0.0,
                "record_count": 0,
            }
        
        total_cost = sum(r.cost for r in records)
        total_usage = sum(r.usage for r in records)
        avg_cost_per_unit = total_cost / total_usage if total_usage > 0 else 0.0
        
        # Costos por tipo
        by_resource: Dict[str, float] = defaultdict(float)
        for record in records:
            by_resource[record.resource_type] += record.cost
        
        return {
            "total_cost": total_cost,
            "total_usage": total_usage,
            "avg_cost_per_unit": avg_cost_per_unit,
            "record_count": len(records),
            "cost_by_resource": dict(by_resource),
            "period_days": days,
        }
    
    def get_recent_costs(
        self,
        resource_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Obtener costos recientes."""
        records = self.cost_records
        
        if resource_type:
            records = [r for r in records if r.resource_type == resource_type]
        
        return [
            {
                "cost_id": r.cost_id,
                "resource_type": r.resource_type,
                "cost": r.cost,
                "usage": r.usage,
                "timestamp": r.timestamp.isoformat(),
            }
            for r in records[-limit:]
        ]
















