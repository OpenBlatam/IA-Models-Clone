"""
Predictive Scaling - Auto-Scaling Predictivo
============================================

Sistema de auto-scaling basado en predicciones de demanda para escalar recursos proactivamente.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class ScalingAction(Enum):
    """Acción de scaling."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    NO_ACTION = "no_action"


@dataclass
class ScalingDecision:
    """Decisión de scaling."""
    decision_id: str
    action: ScalingAction
    resource_type: str
    current_capacity: int
    recommended_capacity: int
    reason: str
    confidence: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


class PredictiveScaler:
    """Auto-scaler predictivo."""
    
    def __init__(
        self,
        min_capacity: int = 1,
        max_capacity: int = 100,
        scale_up_threshold: float = 0.8,
        scale_down_threshold: float = 0.3,
        prediction_horizon_minutes: int = 5,
    ):
        self.min_capacity = min_capacity
        self.max_capacity = max_capacity
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.prediction_horizon_minutes = prediction_horizon_minutes
        self.current_capacity: Dict[str, int] = {}
        self.scaling_history: List[ScalingDecision] = []
        self._lock = asyncio.Lock()
    
    async def evaluate_scaling(
        self,
        resource_type: str,
        current_utilization: float,
        predicted_demand: Optional[float] = None,
    ) -> ScalingDecision:
        """
        Evaluar si es necesario hacer scaling.
        
        Args:
            resource_type: Tipo de recurso
            current_utilization: Utilización actual (0-1)
            predicted_demand: Demanda predicha (opcional)
        
        Returns:
            Decisión de scaling
        """
        current_cap = self.current_capacity.get(resource_type, self.min_capacity)
        
        # Usar demanda predicha si está disponible
        if predicted_demand is not None:
            utilization = predicted_demand / current_cap if current_cap > 0 else 1.0
        else:
            utilization = current_utilization
        
        action = ScalingAction.NO_ACTION
        recommended_capacity = current_cap
        reason = ""
        confidence = 0.5
        
        # Evaluar necesidad de scaling
        if utilization >= self.scale_up_threshold:
            # Necesitamos más capacidad
            if current_cap < self.max_capacity:
                # Calcular capacidad recomendada basada en demanda
                if predicted_demand is not None:
                    recommended_capacity = max(
                        self.min_capacity,
                        min(
                            self.max_capacity,
                            int(predicted_demand / 0.7) + 1  # Mantener 70% utilización
                        )
                    )
                else:
                    recommended_capacity = min(
                        self.max_capacity,
                        int(current_cap * 1.5) + 1
                    )
                
                action = ScalingAction.SCALE_UP
                reason = f"High utilization ({utilization:.1%}) - scaling up to {recommended_capacity}"
                confidence = 0.9 if predicted_demand is not None else 0.7
            else:
                reason = "At max capacity - cannot scale up"
        
        elif utilization <= self.scale_down_threshold:
            # Podemos reducir capacidad
            if current_cap > self.min_capacity:
                if predicted_demand is not None:
                    recommended_capacity = max(
                        self.min_capacity,
                        int(predicted_demand / 0.5)  # Mantener 50% utilización
                    )
                else:
                    recommended_capacity = max(
                        self.min_capacity,
                        int(current_cap * 0.7)
                    )
                
                action = ScalingAction.SCALE_DOWN
                reason = f"Low utilization ({utilization:.1%}) - scaling down to {recommended_capacity}"
                confidence = 0.8 if predicted_demand is not None else 0.6
            else:
                reason = "At min capacity - cannot scale down"
        else:
            reason = f"Optimal utilization ({utilization:.1%}) - no action needed"
            confidence = 0.9
        
        decision = ScalingDecision(
            decision_id=f"decision_{resource_type}_{datetime.now().timestamp()}",
            action=action,
            resource_type=resource_type,
            current_capacity=current_cap,
            recommended_capacity=recommended_capacity,
            reason=reason,
            confidence=confidence,
            timestamp=datetime.now(),
            metadata={
                "current_utilization": current_utilization,
                "predicted_demand": predicted_demand,
                "utilization": utilization,
            },
        )
        
        async with self._lock:
            self.scaling_history.append(decision)
            if len(self.scaling_history) > 1000:
                self.scaling_history.pop(0)
        
        return decision
    
    async def apply_scaling(
        self,
        resource_type: str,
        new_capacity: int,
    ) -> bool:
        """
        Aplicar cambio de capacidad.
        
        Args:
            resource_type: Tipo de recurso
            new_capacity: Nueva capacidad
        
        Returns:
            True si se aplicó exitosamente
        """
        if new_capacity < self.min_capacity or new_capacity > self.max_capacity:
            logger.warning(
                f"Capacity {new_capacity} out of bounds "
                f"({self.min_capacity}-{self.max_capacity})"
            )
            return False
        
        async with self._lock:
            old_capacity = self.current_capacity.get(resource_type, self.min_capacity)
            self.current_capacity[resource_type] = new_capacity
        
        logger.info(
            f"Scaled {resource_type} from {old_capacity} to {new_capacity}"
        )
        
        return True
    
    def get_scaling_history(
        self,
        resource_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Obtener historial de scaling."""
        history = self.scaling_history
        
        if resource_type:
            history = [d for d in history if d.resource_type == resource_type]
        
        return [
            {
                "decision_id": d.decision_id,
                "action": d.action.value,
                "resource_type": d.resource_type,
                "current_capacity": d.current_capacity,
                "recommended_capacity": d.recommended_capacity,
                "reason": d.reason,
                "confidence": d.confidence,
                "timestamp": d.timestamp.isoformat(),
            }
            for d in history[-limit:]
        ]
    
    def get_current_capacity(self, resource_type: str) -> int:
        """Obtener capacidad actual."""
        return self.current_capacity.get(resource_type, self.min_capacity)
















