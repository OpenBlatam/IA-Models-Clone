"""
Auto Scaler - Escalador Automático
===================================

Sistema de auto-escalado basado en métricas de carga y predicción de demanda.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
import statistics

logger = logging.getLogger(__name__)


class ScalingAction(Enum):
    """Acción de escalado."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    NO_ACTION = "no_action"


class ScalingStrategy(Enum):
    """Estrategia de escalado."""
    REACTIVE = "reactive"
    PREDICTIVE = "predictive"
    HYBRID = "hybrid"


@dataclass
class ScalingRule:
    """Regla de escalado."""
    rule_id: str
    metric_name: str
    threshold_up: float
    threshold_down: float
    min_instances: int = 1
    max_instances: int = 10
    scale_up_step: int = 1
    scale_down_step: int = 1
    cooldown_seconds: float = 300.0
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScalingEvent:
    """Evento de escalado."""
    event_id: str
    timestamp: datetime
    action: ScalingAction
    rule_id: str
    current_instances: int
    target_instances: int
    reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class AutoScaler:
    """Escalador automático."""
    
    def __init__(self):
        self.rules: Dict[str, ScalingRule] = {}
        self.current_instances: int = 1
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.scaling_history: deque = deque(maxlen=10000)
        self.last_scaling_time: Optional[datetime] = None
        self._lock = asyncio.Lock()
        self._scaling_active = False
    
    def add_rule(
        self,
        rule_id: str,
        metric_name: str,
        threshold_up: float,
        threshold_down: float,
        min_instances: int = 1,
        max_instances: int = 10,
        scale_up_step: int = 1,
        scale_down_step: int = 1,
        cooldown_seconds: float = 300.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Agregar regla de escalado."""
        rule = ScalingRule(
            rule_id=rule_id,
            metric_name=metric_name,
            threshold_up=threshold_up,
            threshold_down=threshold_down,
            min_instances=min_instances,
            max_instances=max_instances,
            scale_up_step=scale_up_step,
            scale_down_step=scale_down_step,
            cooldown_seconds=cooldown_seconds,
            metadata=metadata or {},
        )
        
        async def save_rule():
            async with self._lock:
                self.rules[rule_id] = rule
        
        asyncio.create_task(save_rule())
        
        logger.info(f"Added scaling rule: {rule_id}")
        return rule_id
    
    def record_metric(
        self,
        metric_name: str,
        value: float,
        timestamp: Optional[datetime] = None,
    ):
        """Registrar métrica."""
        timestamp = timestamp or datetime.now()
        
        async def save_metric():
            async with self._lock:
                self.metrics[metric_name].append({
                    "value": value,
                    "timestamp": timestamp,
                })
        
        asyncio.create_task(save_metric())
        
        # Evaluar escalado si está activo
        if self._scaling_active:
            asyncio.create_task(self._evaluate_scaling())
    
    async def _evaluate_scaling(self):
        """Evaluar necesidad de escalado."""
        # Verificar cooldown
        if self.last_scaling_time:
            time_since_last = (datetime.now() - self.last_scaling_time).total_seconds()
            min_cooldown = min(
                (r.cooldown_seconds for r in self.rules.values() if r.enabled),
                default=300.0
            )
            if time_since_last < min_cooldown:
                return
        
        # Evaluar cada regla
        for rule in self.rules.values():
            if not rule.enabled:
                continue
            
            metric_history = self.metrics.get(rule.metric_name, deque())
            if not metric_history:
                continue
            
            # Calcular métrica actual (promedio de últimos valores)
            recent_values = [m["value"] for m in list(metric_history)[-10:]]
            if not recent_values:
                continue
            
            current_metric = statistics.mean(recent_values)
            
            # Determinar acción
            action = ScalingAction.NO_ACTION
            target_instances = self.current_instances
            reason = ""
            
            if current_metric >= rule.threshold_up:
                # Escalar arriba
                target_instances = min(
                    rule.max_instances,
                    self.current_instances + rule.scale_up_step
                )
                if target_instances > self.current_instances:
                    action = ScalingAction.SCALE_UP
                    reason = f"Metric {rule.metric_name} ({current_metric:.2f}) >= threshold_up ({rule.threshold_up})"
            
            elif current_metric <= rule.threshold_down:
                # Escalar abajo
                target_instances = max(
                    rule.min_instances,
                    self.current_instances - rule.scale_down_step
                )
                if target_instances < self.current_instances:
                    action = ScalingAction.SCALE_DOWN
                    reason = f"Metric {rule.metric_name} ({current_metric:.2f}) <= threshold_down ({rule.threshold_down})"
            
            # Ejecutar escalado si es necesario
            if action != ScalingAction.NO_ACTION:
                await self._execute_scaling(action, rule, target_instances, reason)
                break  # Solo una acción por ciclo
    
    async def _execute_scaling(
        self,
        action: ScalingAction,
        rule: ScalingRule,
        target_instances: int,
        reason: str,
    ):
        """Ejecutar escalado."""
        old_instances = self.current_instances
        
        async with self._lock:
            self.current_instances = target_instances
            self.last_scaling_time = datetime.now()
        
        # Registrar evento
        event = ScalingEvent(
            event_id=f"scale_{datetime.now().timestamp()}",
            timestamp=datetime.now(),
            action=action,
            rule_id=rule.rule_id,
            current_instances=old_instances,
            target_instances=target_instances,
            reason=reason,
        )
        
        async with self._lock:
            self.scaling_history.append(event)
        
        logger.info(f"Scaling {action.value}: {old_instances} -> {target_instances}. Reason: {reason}")
        
        # En producción, aquí se ejecutaría la acción real de escalado
        # await self._perform_scaling_action(action, target_instances)
    
    def set_instances(self, instances: int):
        """Establecer número de instancias manualmente."""
        async def update():
            async with self._lock:
                self.current_instances = instances
        
        asyncio.create_task(update())
        logger.info(f"Instances set to {instances}")
    
    def start_scaling(self):
        """Iniciar auto-escalado."""
        self._scaling_active = True
        logger.info("Auto-scaling started")
    
    def stop_scaling(self):
        """Detener auto-escalado."""
        self._scaling_active = False
        logger.info("Auto-scaling stopped")
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Obtener estado de escalado."""
        return {
            "scaling_active": self._scaling_active,
            "current_instances": self.current_instances,
            "total_rules": len(self.rules),
            "active_rules": len([r for r in self.rules.values() if r.enabled]),
            "last_scaling_time": self.last_scaling_time.isoformat() if self.last_scaling_time else None,
        }
    
    def get_scaling_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Obtener historial de escalado."""
        history = list(self.scaling_history)[-limit:]
        
        return [
            {
                "event_id": e.event_id,
                "timestamp": e.timestamp.isoformat(),
                "action": e.action.value,
                "rule_id": e.rule_id,
                "current_instances": e.current_instances,
                "target_instances": e.target_instances,
                "reason": e.reason,
            }
            for e in history
        ]
    
    def get_auto_scaler_summary(self) -> Dict[str, Any]:
        """Obtener resumen del escalador."""
        by_action: Dict[str, int] = defaultdict(int)
        
        for event in self.scaling_history:
            by_action[event.action.value] += 1
        
        return {
            "scaling_active": self._scaling_active,
            "current_instances": self.current_instances,
            "total_rules": len(self.rules),
            "scaling_events_by_action": dict(by_action),
            "total_history": len(self.scaling_history),
        }


