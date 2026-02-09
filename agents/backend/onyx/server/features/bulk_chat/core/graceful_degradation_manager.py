"""
Graceful Degradation Manager - Gestor de Degradación Gradual
==============================================================

Sistema de gestión de degradación gradual con fallbacks, circuitos y modos de operación degradados.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class DegradationLevel(Enum):
    """Nivel de degradación."""
    NORMAL = "normal"
    DEGRADED = "degraded"
    MINIMAL = "minimal"
    EMERGENCY = "emergency"
    OFFLINE = "offline"


class ServiceState(Enum):
    """Estado de servicio."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILING = "failing"
    DOWN = "down"


@dataclass
class ServiceHealth:
    """Salud de servicio."""
    service_id: str
    state: ServiceState
    error_rate: float = 0.0
    response_time: float = 0.0
    last_check: datetime = field(default_factory=datetime.now)
    consecutive_failures: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FallbackStrategy:
    """Estrategia de fallback."""
    strategy_id: str
    service_id: str
    fallback_handler: Callable
    conditions: Dict[str, Any]
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DegradationRule:
    """Regla de degradación."""
    rule_id: str
    service_id: str
    metric_name: str
    threshold: float
    degradation_level: DegradationLevel
    action: Optional[Callable] = None
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class GracefulDegradationManager:
    """Gestor de degradación gradual."""
    
    def __init__(self):
        self.services: Dict[str, ServiceHealth] = {}
        self.fallbacks: Dict[str, List[FallbackStrategy]] = defaultdict(list)
        self.degradation_rules: List[DegradationRule] = []
        self.current_level: DegradationLevel = DegradationLevel.NORMAL
        self.degradation_history: deque = deque(maxlen=10000)
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._lock = asyncio.Lock()
    
    def register_service(
        self,
        service_id: str,
        initial_state: ServiceState = ServiceState.HEALTHY,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Registrar servicio."""
        health = ServiceHealth(
            service_id=service_id,
            state=initial_state,
            metadata=metadata or {},
        )
        
        async def save_service():
            async with self._lock:
                self.services[service_id] = health
        
        asyncio.create_task(save_service())
        
        logger.info(f"Registered service: {service_id}")
        return service_id
    
    def register_fallback(
        self,
        strategy_id: str,
        service_id: str,
        fallback_handler: Callable,
        conditions: Optional[Dict[str, Any]] = None,
        priority: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Registrar estrategia de fallback."""
        strategy = FallbackStrategy(
            strategy_id=strategy_id,
            service_id=service_id,
            fallback_handler=fallback_handler,
            conditions=conditions or {},
            priority=priority,
            metadata=metadata or {},
        )
        
        async def save_fallback():
            async with self._lock:
                self.fallbacks[service_id].append(strategy)
                self.fallbacks[service_id].sort(key=lambda s: s.priority, reverse=True)
        
        asyncio.create_task(save_fallback())
        
        logger.info(f"Registered fallback strategy: {strategy_id} for service {service_id}")
        return strategy_id
    
    def add_degradation_rule(
        self,
        rule_id: str,
        service_id: str,
        metric_name: str,
        threshold: float,
        degradation_level: DegradationLevel,
        action: Optional[Callable] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Agregar regla de degradación."""
        rule = DegradationRule(
            rule_id=rule_id,
            service_id=service_id,
            metric_name=metric_name,
            threshold=threshold,
            degradation_level=degradation_level,
            action=action,
            metadata=metadata or {},
        )
        
        async def save_rule():
            async with self._lock:
                self.degradation_rules.append(rule)
        
        asyncio.create_task(save_rule())
        
        logger.info(f"Added degradation rule: {rule_id}")
        return rule_id
    
    async def record_metric(
        self,
        service_id: str,
        metric_name: str,
        value: float,
        timestamp: Optional[datetime] = None,
    ):
        """Registrar métrica."""
        timestamp = timestamp or datetime.now()
        
        async with self._lock:
            key = f"{service_id}_{metric_name}"
            self.metrics[key].append({
                "value": value,
                "timestamp": timestamp,
            })
        
        # Evaluar reglas de degradación
        asyncio.create_task(self._evaluate_degradation_rules(service_id, metric_name, value))
    
    async def record_service_call(
        self,
        service_id: str,
        success: bool,
        response_time: float,
    ):
        """Registrar llamada a servicio."""
        service = self.services.get(service_id)
        if not service:
            return
        
        async with self._lock:
            # Actualizar métricas
            if not success:
                service.consecutive_failures += 1
                service.error_rate = min(1.0, service.error_rate + 0.1)
            else:
                service.consecutive_failures = 0
                service.error_rate = max(0.0, service.error_rate - 0.05)
            
            service.response_time = (service.response_time * 0.9) + (response_time * 0.1)
            service.last_check = datetime.now()
            
            # Actualizar estado
            if service.error_rate > 0.5 or service.consecutive_failures > 5:
                service.state = ServiceState.DOWN
            elif service.error_rate > 0.2 or service.consecutive_failures > 2:
                service.state = ServiceState.FAILING
            elif service.error_rate > 0.05:
                service.state = ServiceState.DEGRADED
            else:
                service.state = ServiceState.HEALTHY
        
        # Evaluar degradación global
        asyncio.create_task(self._evaluate_global_degradation())
    
    async def _evaluate_degradation_rules(self, service_id: str, metric_name: str, value: float):
        """Evaluar reglas de degradación."""
        rules = [
            r for r in self.degradation_rules
            if r.service_id == service_id
            and r.metric_name == metric_name
            and r.enabled
        ]
        
        for rule in rules:
            if value >= rule.threshold:
                # Trigger degradación
                await self._apply_degradation(rule.degradation_level, rule.service_id, rule.action)
    
    async def _evaluate_global_degradation(self):
        """Evaluar degradación global."""
        # Contar servicios por estado
        healthy_count = sum(1 for s in self.services.values() if s.state == ServiceState.HEALTHY)
        degraded_count = sum(1 for s in self.services.values() if s.state == ServiceState.DEGRADED)
        failing_count = sum(1 for s in self.services.values() if s.state == ServiceState.FAILING)
        down_count = sum(1 for s in self.services.values() if s.state == ServiceState.DOWN)
        
        total = len(self.services)
        if total == 0:
            return
        
        # Determinar nivel de degradación
        healthy_ratio = healthy_count / total
        
        if healthy_ratio < 0.2 or down_count > total * 0.5:
            new_level = DegradationLevel.EMERGENCY
        elif healthy_ratio < 0.5 or failing_count > total * 0.3:
            new_level = DegradationLevel.MINIMAL
        elif healthy_ratio < 0.8 or degraded_count > total * 0.4:
            new_level = DegradationLevel.DEGRADED
        else:
            new_level = DegradationLevel.NORMAL
        
        if new_level != self.current_level:
            await self._apply_degradation(new_level)
    
    async def _apply_degradation(
        self,
        level: DegradationLevel,
        service_id: Optional[str] = None,
        action: Optional[Callable] = None,
    ):
        """Aplicar degradación."""
        old_level = self.current_level
        self.current_level = level
        
        async with self._lock:
            self.degradation_history.append({
                "timestamp": datetime.now().isoformat(),
                "old_level": old_level.value,
                "new_level": level.value,
                "service_id": service_id,
            })
        
        # Ejecutar acción si existe
        if action:
            try:
                if asyncio.iscoroutinefunction(action):
                    await action(level, service_id)
                else:
                    action(level, service_id)
            except Exception as e:
                logger.error(f"Error executing degradation action: {e}")
        
        logger.info(f"Degradation level changed: {old_level.value} -> {level.value} (service: {service_id})")
    
    async def execute_with_fallback(
        self,
        service_id: str,
        primary_handler: Callable,
        *args,
        **kwargs,
    ) -> Any:
        """Ejecutar con fallback."""
        service = self.services.get(service_id)
        
        # Intentar ejecución primaria
        try:
            if asyncio.iscoroutinefunction(primary_handler):
                result = await primary_handler(*args, **kwargs)
            else:
                result = primary_handler(*args, **kwargs)
            
            # Registrar éxito
            await self.record_service_call(service_id, True, 0.0)
            return result
        
        except Exception as e:
            # Registrar fallo
            await self.record_service_call(service_id, False, 0.0)
            
            # Intentar fallback
            fallbacks = self.fallbacks.get(service_id, [])
            
            for fallback in fallbacks:
                try:
                    if asyncio.iscoroutinefunction(fallback.fallback_handler):
                        result = await fallback.fallback_handler(*args, **kwargs)
                    else:
                        result = fallback.fallback_handler(*args, **kwargs)
                    
                    logger.info(f"Fallback executed for {service_id}: {fallback.strategy_id}")
                    return result
                
                except Exception as fallback_error:
                    logger.warning(f"Fallback {fallback.strategy_id} failed: {fallback_error}")
                    continue
            
            # Todos los fallbacks fallaron
            raise Exception(f"All fallbacks failed for {service_id}. Last error: {str(e)}")
    
    def get_service_health(self, service_id: str) -> Optional[Dict[str, Any]]:
        """Obtener salud de servicio."""
        service = self.services.get(service_id)
        if not service:
            return None
        
        return {
            "service_id": service.service_id,
            "state": service.state.value,
            "error_rate": service.error_rate,
            "response_time": service.response_time,
            "consecutive_failures": service.consecutive_failures,
            "last_check": service.last_check.isoformat(),
        }
    
    def get_degradation_status(self) -> Dict[str, Any]:
        """Obtener estado de degradación."""
        by_state: Dict[str, int] = defaultdict(int)
        
        for service in self.services.values():
            by_state[service.state.value] += 1
        
        return {
            "current_level": self.current_level.value,
            "total_services": len(self.services),
            "services_by_state": dict(by_state),
            "total_fallbacks": sum(len(fallbacks) for fallbacks in self.fallbacks.values()),
            "degradation_rules": len(self.degradation_rules),
        }
    
    def get_degradation_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Obtener historial de degradación."""
        return list(self.degradation_history)[-limit:]
    
    def get_graceful_degradation_manager_summary(self) -> Dict[str, Any]:
        """Obtener resumen del gestor."""
        return {
            "current_level": self.current_level.value,
            "total_services": len(self.services),
            "total_fallbacks": sum(len(fallbacks) for fallbacks in self.fallbacks.values()),
            "degradation_rules": len(self.degradation_rules),
            "total_history": len(self.degradation_history),
        }



