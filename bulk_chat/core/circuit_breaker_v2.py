"""
Circuit Breaker V2 - Circuit Breaker Avanzado
==============================================

Sistema avanzado de circuit breaker con múltiples estrategias y auto-recovery.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Estado del circuito."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class FailureStrategy(Enum):
    """Estrategia de fallo."""
    COUNT_BASED = "count_based"
    PERCENTAGE_BASED = "percentage_based"
    TIME_BASED = "time_based"
    HYBRID = "hybrid"


@dataclass
class CircuitBreaker:
    """Circuit breaker."""
    circuit_id: str
    name: str
    failure_threshold: int = 5
    failure_percentage: float = 0.5
    timeout_seconds: float = 60.0
    half_open_max_calls: int = 3
    failure_strategy: FailureStrategy = FailureStrategy.COUNT_BASED
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    total_calls: int = 0
    last_failure_time: Optional[datetime] = None
    opened_at: Optional[datetime] = None
    call_history: deque = field(default_factory=lambda: deque(maxlen=100))
    metadata: Dict[str, Any] = field(default_factory=dict)


class CircuitBreakerV2:
    """Gestor de circuit breakers avanzado."""
    
    def __init__(self):
        self.circuits: Dict[str, CircuitBreaker] = {}
        self._lock = asyncio.Lock()
    
    def create_circuit(
        self,
        circuit_id: str,
        name: str,
        failure_threshold: int = 5,
        failure_percentage: float = 0.5,
        timeout_seconds: float = 60.0,
        half_open_max_calls: int = 3,
        failure_strategy: FailureStrategy = FailureStrategy.COUNT_BASED,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Crear circuit breaker."""
        circuit = CircuitBreaker(
            circuit_id=circuit_id,
            name=name,
            failure_threshold=failure_threshold,
            failure_percentage=failure_percentage,
            timeout_seconds=timeout_seconds,
            half_open_max_calls=half_open_max_calls,
            failure_strategy=failure_strategy,
            metadata=metadata or {},
        )
        
        async def save_circuit():
            async with self._lock:
                self.circuits[circuit_id] = circuit
        
        asyncio.create_task(save_circuit())
        
        logger.info(f"Created circuit breaker: {circuit_id} - {name}")
        return circuit_id
    
    async def call(
        self,
        circuit_id: str,
        func: Callable,
        *args,
        **kwargs,
    ) -> Any:
        """Ejecutar función con circuit breaker."""
        circuit = self.circuits.get(circuit_id)
        if not circuit:
            # Si no hay circuit breaker, ejecutar directamente
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        
        # Verificar estado del circuito
        await self._check_circuit_state(circuit)
        
        if circuit.state == CircuitState.OPEN:
            raise Exception(f"Circuit breaker {circuit_id} is OPEN")
        
        # Ejecutar función
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Registrar éxito
            await self._record_success(circuit)
            return result
        
        except Exception as e:
            # Registrar fallo
            await self._record_failure(circuit)
            raise
    
    async def _check_circuit_state(self, circuit: CircuitBreaker):
        """Verificar y actualizar estado del circuito."""
        now = datetime.now()
        
        if circuit.state == CircuitState.OPEN:
            # Verificar si es tiempo de intentar half-open
            if circuit.opened_at:
                time_since_open = (now - circuit.opened_at).total_seconds()
                if time_since_open >= circuit.timeout_seconds:
                    circuit.state = CircuitState.HALF_OPEN
                    circuit.success_count = 0
                    circuit.failure_count = 0
                    logger.info(f"Circuit {circuit.circuit_id} moved to HALF_OPEN")
        
        elif circuit.state == CircuitState.HALF_OPEN:
            # Si se alcanzó el máximo de llamadas en half-open, evaluar
            if circuit.total_calls >= circuit.half_open_max_calls:
                if circuit.failure_count == 0:
                    # Todo funcionó, cerrar circuito
                    circuit.state = CircuitState.CLOSED
                    circuit.failure_count = 0
                    circuit.success_count = 0
                    circuit.total_calls = 0
                    logger.info(f"Circuit {circuit.circuit_id} moved to CLOSED")
                else:
                    # Sigue fallando, abrir circuito
                    circuit.state = CircuitState.OPEN
                    circuit.opened_at = now
                    logger.warning(f"Circuit {circuit.circuit_id} moved to OPEN")
    
    async def _record_success(self, circuit: CircuitBreaker):
        """Registrar éxito."""
        now = datetime.now()
        
        async with self._lock:
            circuit.success_count += 1
            circuit.total_calls += 1
            circuit.call_history.append({
                "timestamp": now,
                "success": True,
            })
        
        # Si está en half-open y todo va bien, podría cerrarse
        if circuit.state == CircuitState.HALF_OPEN:
            await self._check_circuit_state(circuit)
    
    async def _record_failure(self, circuit: CircuitBreaker):
        """Registrar fallo."""
        now = datetime.now()
        
        async with self._lock:
            circuit.failure_count += 1
            circuit.total_calls += 1
            circuit.last_failure_time = now
            circuit.call_history.append({
                "timestamp": now,
                "success": False,
            })
        
        # Evaluar si se debe abrir el circuito
        should_open = False
        
        if circuit.failure_strategy == FailureStrategy.COUNT_BASED:
            should_open = circuit.failure_count >= circuit.failure_threshold
        
        elif circuit.failure_strategy == FailureStrategy.PERCENTAGE_BASED:
            if circuit.total_calls > 0:
                failure_rate = circuit.failure_count / circuit.total_calls
                should_open = failure_rate >= circuit.failure_percentage
        
        elif circuit.failure_strategy == FailureStrategy.TIME_BASED:
            # Abrir si hay fallos recientes
            recent_failures = sum(
                1 for call in circuit.call_history
                if not call["success"]
                and (now - call["timestamp"]).total_seconds() < circuit.timeout_seconds
            )
            should_open = recent_failures >= circuit.failure_threshold
        
        elif circuit.failure_strategy == FailureStrategy.HYBRID:
            # Combinar count y percentage
            count_check = circuit.failure_count >= circuit.failure_threshold
            percentage_check = False
            if circuit.total_calls > 0:
                failure_rate = circuit.failure_count / circuit.total_calls
                percentage_check = failure_rate >= circuit.failure_percentage
            should_open = count_check or percentage_check
        
        if should_open and circuit.state == CircuitState.CLOSED:
            circuit.state = CircuitState.OPEN
            circuit.opened_at = now
            logger.warning(f"Circuit {circuit.circuit_id} opened due to failures")
    
    def get_circuit(self, circuit_id: str) -> Optional[Dict[str, Any]]:
        """Obtener información de circuit breaker."""
        circuit = self.circuits.get(circuit_id)
        if not circuit:
            return None
        
        failure_rate = 0.0
        if circuit.total_calls > 0:
            failure_rate = circuit.failure_count / circuit.total_calls
        
        return {
            "circuit_id": circuit.circuit_id,
            "name": circuit.name,
            "state": circuit.state.value,
            "failure_count": circuit.failure_count,
            "success_count": circuit.success_count,
            "total_calls": circuit.total_calls,
            "failure_rate": failure_rate,
            "last_failure_time": circuit.last_failure_time.isoformat() if circuit.last_failure_time else None,
            "opened_at": circuit.opened_at.isoformat() if circuit.opened_at else None,
        }
    
    async def reset_circuit(self, circuit_id: str) -> bool:
        """Resetear circuit breaker."""
        circuit = self.circuits.get(circuit_id)
        if not circuit:
            return False
        
        async with self._lock:
            circuit.state = CircuitState.CLOSED
            circuit.failure_count = 0
            circuit.success_count = 0
            circuit.total_calls = 0
            circuit.opened_at = None
            circuit.last_failure_time = None
        
        logger.info(f"Reset circuit breaker: {circuit_id}")
        return True
    
    def get_circuit_breaker_v2_summary(self) -> Dict[str, Any]:
        """Obtener resumen del gestor."""
        by_state: Dict[str, int] = {}
        for state in CircuitState:
            by_state[state.value] = 0
        
        for circuit in self.circuits.values():
            by_state[circuit.state.value] += 1
        
        return {
            "total_circuits": len(self.circuits),
            "circuits_by_state": by_state,
            "open_circuits": len([c for c in self.circuits.values() if c.state == CircuitState.OPEN]),
            "half_open_circuits": len([c for c in self.circuits.values() if c.state == CircuitState.HALF_OPEN]),
        }


