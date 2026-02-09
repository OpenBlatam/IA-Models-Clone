"""
Sistema de Auto-Scaling

Proporciona:
- Escalado basado en métricas
- Escalado predictivo
- Escalado reactivo
- Políticas de escalado configurables
"""

import logging
import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
from enum import Enum

logger = logging.getLogger(__name__)


class ScalingAction(Enum):
    """Acciones de escalado"""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    NO_ACTION = "no_action"


@dataclass
class ScalingPolicy:
    """Política de escalado"""
    name: str
    metric: str  # cpu, memory, requests, queue_size
    threshold_up: float
    threshold_down: float
    min_replicas: int = 1
    max_replicas: int = 10
    scale_up_cooldown: int = 60  # segundos
    scale_down_cooldown: int = 300  # segundos
    enabled: bool = True


@dataclass
class ScalingDecision:
    """Decisión de escalado"""
    action: ScalingAction
    current_replicas: int
    target_replicas: int
    reason: str
    timestamp: datetime = field(default_factory=datetime.now)


class AutoScaler:
    """Auto-scaler avanzado"""
    
    def __init__(self):
        self.policies: Dict[str, ScalingPolicy] = {}
        self.current_replicas: int = 1
        self.metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.last_scale_up: Optional[datetime] = None
        self.last_scale_down: Optional[datetime] = None
        self.scaling_history: List[ScalingDecision] = []
        
        # Políticas por defecto
        self._setup_default_policies()
        
        logger.info("AutoScaler initialized")
    
    def _setup_default_policies(self):
        """Configura políticas por defecto"""
        self.add_policy(ScalingPolicy(
            name="cpu_based",
            metric="cpu",
            threshold_up=80.0,
            threshold_down=30.0,
            min_replicas=1,
            max_replicas=10
        ))
        
        self.add_policy(ScalingPolicy(
            name="memory_based",
            metric="memory",
            threshold_up=85.0,
            threshold_down=40.0,
            min_replicas=1,
            max_replicas=10
        ))
        
        self.add_policy(ScalingPolicy(
            name="queue_based",
            metric="queue_size",
            threshold_up=100,
            threshold_down=20,
            min_replicas=1,
            max_replicas=10
        ))
    
    def add_policy(self, policy: ScalingPolicy):
        """Agrega una política de escalado"""
        self.policies[policy.name] = policy
        logger.info(f"Scaling policy added: {policy.name}")
    
    def record_metric(self, metric_name: str, value: float):
        """Registra una métrica"""
        self.metric_history[metric_name].append({
            "value": value,
            "timestamp": datetime.now()
        })
    
    def evaluate_scaling(self) -> Optional[ScalingDecision]:
        """
        Evalúa si se necesita escalar
        
        Returns:
            Decisión de escalado o None
        """
        decisions = []
        
        for policy_name, policy in self.policies.items():
            if not policy.enabled:
                continue
            
            # Obtener valor actual de la métrica
            current_value = self._get_current_metric_value(policy.metric)
            
            if current_value is None:
                continue
            
            # Evaluar umbrales
            if current_value >= policy.threshold_up:
                # Necesita scale up
                target_replicas = min(
                    self.current_replicas + 1,
                    policy.max_replicas
                )
                
                if target_replicas > self.current_replicas:
                    # Verificar cooldown
                    if self._can_scale_up(policy):
                        decisions.append(ScalingDecision(
                            action=ScalingAction.SCALE_UP,
                            current_replicas=self.current_replicas,
                            target_replicas=target_replicas,
                            reason=f"{policy.metric} = {current_value:.2f} >= {policy.threshold_up}"
                        ))
            
            elif current_value <= policy.threshold_down:
                # Necesita scale down
                target_replicas = max(
                    self.current_replicas - 1,
                    policy.min_replicas
                )
                
                if target_replicas < self.current_replicas:
                    # Verificar cooldown
                    if self._can_scale_down(policy):
                        decisions.append(ScalingDecision(
                            action=ScalingAction.SCALE_DOWN,
                            current_replicas=self.current_replicas,
                            target_replicas=target_replicas,
                            reason=f"{policy.metric} = {current_value:.2f} <= {policy.threshold_down}"
                        ))
        
        # Si hay múltiples decisiones, priorizar scale up
        if decisions:
            scale_up_decisions = [d for d in decisions if d.action == ScalingAction.SCALE_UP]
            if scale_up_decisions:
                return scale_up_decisions[0]
            return decisions[0]
        
        return None
    
    def _get_current_metric_value(self, metric_name: str) -> Optional[float]:
        """Obtiene el valor actual de una métrica"""
        history = self.metric_history.get(metric_name)
        if not history:
            return None
        
        # Obtener promedio de últimos valores
        recent_values = [
            entry["value"]
            for entry in list(history)[-10:]
        ]
        
        if not recent_values:
            return None
        
        return sum(recent_values) / len(recent_values)
    
    def _can_scale_up(self, policy: ScalingPolicy) -> bool:
        """Verifica si se puede hacer scale up"""
        if self.last_scale_up is None:
            return True
        
        elapsed = (datetime.now() - self.last_scale_up).total_seconds()
        return elapsed >= policy.scale_up_cooldown
    
    def _can_scale_down(self, policy: ScalingPolicy) -> bool:
        """Verifica si se puede hacer scale down"""
        if self.last_scale_down is None:
            return True
        
        elapsed = (datetime.now() - self.last_scale_down).total_seconds()
        return elapsed >= policy.scale_down_cooldown
    
    def apply_scaling(self, decision: ScalingDecision) -> bool:
        """
        Aplica una decisión de escalado
        
        Args:
            decision: Decisión de escalado
        
        Returns:
            True si se aplicó correctamente
        """
        try:
            # En producción, esto llamaría a la API de Kubernetes/Docker
            # Por ahora solo actualizamos el estado
            self.current_replicas = decision.target_replicas
            
            if decision.action == ScalingAction.SCALE_UP:
                self.last_scale_up = datetime.now()
            elif decision.action == ScalingAction.SCALE_DOWN:
                self.last_scale_down = datetime.now()
            
            self.scaling_history.append(decision)
            
            logger.info(
                f"Scaling {decision.action.value}: "
                f"{decision.current_replicas} -> {decision.target_replicas} "
                f"({decision.reason})"
            )
            
            return True
        
        except Exception as e:
            logger.error(f"Error applying scaling: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del auto-scaler"""
        return {
            "current_replicas": self.current_replicas,
            "policies": len(self.policies),
            "enabled_policies": sum(1 for p in self.policies.values() if p.enabled),
            "scaling_history_count": len(self.scaling_history),
            "last_scale_up": self.last_scale_up.isoformat() if self.last_scale_up else None,
            "last_scale_down": self.last_scale_down.isoformat() if self.last_scale_down else None,
            "recent_decisions": [
                {
                    "action": d.action.value,
                    "replicas": f"{d.current_replicas} -> {d.target_replicas}",
                    "reason": d.reason,
                    "timestamp": d.timestamp.isoformat()
                }
                for d in self.scaling_history[-10:]
            ]
        }


# Instancia global
_auto_scaler: Optional[AutoScaler] = None


def get_auto_scaler() -> AutoScaler:
    """Obtiene la instancia global del auto-scaler"""
    global _auto_scaler
    if _auto_scaler is None:
        _auto_scaler = AutoScaler()
    return _auto_scaler

