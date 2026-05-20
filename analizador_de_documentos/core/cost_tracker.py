"""
Sistema de Seguimiento de Costos
==================================

Sistema para tracking de uso y costos de API.
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class CostRecord:
    """Registro de costo"""
    timestamp: str
    operation: str
    cost: float
    tokens_used: int
    model: str
    user_id: Optional[str] = None


class CostTracker:
    """
    Tracker de costos
    
    Proporciona:
    - Tracking de costos por operación
    - Costos por modelo
    - Costos por usuario/tenant
    - Estimación de costos
    - Reportes de costos
    """
    
    def __init__(self):
        """Inicializar tracker"""
        self.records: List[CostRecord] = []
        self.cost_per_token: Dict[str, float] = {
            "default": 0.0001,  # $0.0001 por token
            "gpt-4": 0.03,
            "gpt-3.5-turbo": 0.002,
            "bert-base": 0.0001
        }
        self.cost_per_operation: Dict[str, float] = {
            "classification": 0.001,
            "summarization": 0.005,
            "embedding": 0.0005,
            "ocr": 0.01
        }
        logger.info("CostTracker inicializado")
    
    def record_cost(
        self,
        operation: str,
        tokens_used: int,
        model: str = "default",
        user_id: Optional[str] = None
    ):
        """Registrar costo de operación"""
        cost_per_token = self.cost_per_token.get(model, self.cost_per_token["default"])
        cost = tokens_used * cost_per_token
        
        # Agregar costo base de operación si existe
        if operation in self.cost_per_operation:
            cost += self.cost_per_operation[operation]
        
        record = CostRecord(
            timestamp=datetime.now().isoformat(),
            operation=operation,
            cost=cost,
            tokens_used=tokens_used,
            model=model,
            user_id=user_id
        )
        
        self.records.append(record)
        
        # Mantener solo últimos 10000 registros
        if len(self.records) > 10000:
            self.records = self.records[-10000:]
    
    def get_total_cost(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> float:
        """Obtener costo total"""
        filtered = self._filter_records(start_date, end_date, user_id)
        return sum(r.cost for r in filtered)
    
    def get_cost_by_operation(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, float]:
        """Obtener costos por operación"""
        filtered = self._filter_records(start_date, end_date)
        
        costs = defaultdict(float)
        for record in filtered:
            costs[record.operation] += record.cost
        
        return dict(costs)
    
    def get_cost_by_model(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, float]:
        """Obtener costos por modelo"""
        filtered = self._filter_records(start_date, end_date)
        
        costs = defaultdict(float)
        for record in filtered:
            costs[record.model] += record.cost
        
        return dict(costs)
    
    def get_cost_by_user(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, float]:
        """Obtener costos por usuario"""
        filtered = self._filter_records(start_date, end_date)
        
        costs = defaultdict(float)
        for record in filtered:
            if record.user_id:
                costs[record.user_id] += record.cost
        
        return dict(costs)
    
    def get_daily_cost(
        self,
        days: int = 7
    ) -> List[Dict[str, Any]]:
        """Obtener costos diarios"""
        cutoff = datetime.now() - timedelta(days=days)
        
        daily_costs = defaultdict(float)
        daily_tokens = defaultdict(int)
        
        for record in self.records:
            record_date = datetime.fromisoformat(record.timestamp)
            if record_date >= cutoff:
                date_key = record_date.date().isoformat()
                daily_costs[date_key] += record.cost
                daily_tokens[date_key] += record.tokens_used
        
        return [
            {
                "date": date,
                "cost": cost,
                "tokens": daily_tokens[date]
            }
            for date, cost in sorted(daily_costs.items())
        ]
    
    def estimate_cost(
        self,
        operation: str,
        estimated_tokens: int,
        model: str = "default"
    ) -> float:
        """Estimar costo de operación"""
        cost_per_token = self.cost_per_token.get(model, self.cost_per_token["default"])
        cost = estimated_tokens * cost_per_token
        
        if operation in self.cost_per_operation:
            cost += self.cost_per_operation[operation]
        
        return cost
    
    def _filter_records(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> List[CostRecord]:
        """Filtrar registros"""
        filtered = self.records
        
        if start_date:
            start = datetime.fromisoformat(start_date)
            filtered = [r for r in filtered if datetime.fromisoformat(r.timestamp) >= start]
        
        if end_date:
            end = datetime.fromisoformat(end_date)
            filtered = [r for r in filtered if datetime.fromisoformat(r.timestamp) <= end]
        
        if user_id:
            filtered = [r for r in filtered if r.user_id == user_id]
        
        return filtered


# Instancia global
_cost_tracker: Optional[CostTracker] = None


def get_cost_tracker() -> CostTracker:
    """Obtener instancia global del tracker"""
    global _cost_tracker
    if _cost_tracker is None:
        _cost_tracker = CostTracker()
    return _cost_tracker
















