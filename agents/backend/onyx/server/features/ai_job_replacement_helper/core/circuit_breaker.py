"""
Circuit Breaker Service - Circuit breaker pattern
=================================================

Sistema de circuit breaker para manejar fallos en servicios externos.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    """Estados del circuit breaker"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreaker:
    """Circuit breaker"""
    name: str
    failure_threshold: int = 5
    success_threshold: int = 2
    timeout_seconds: int = 60
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[datetime] = None
    opened_at: Optional[datetime] = None


class CircuitBreakerService:
    """Servicio de circuit breaker"""
    
    def __init__(self):
        """Inicializar servicio"""
        self.circuits: Dict[str, CircuitBreaker] = {}
        logger.info("CircuitBreakerService initialized")
    
    def create_circuit(
        self,
        name: str,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        timeout_seconds: int = 60
    ) -> CircuitBreaker:
        """Crear circuit breaker"""
        circuit = CircuitBreaker(
            name=name,
            failure_threshold=failure_threshold,
            success_threshold=success_threshold,
            timeout_seconds=timeout_seconds,
        )
        
        self.circuits[name] = circuit
        
        logger.info(f"Circuit breaker created: {name}")
        return circuit
    
    async def call(
        self,
        circuit_name: str,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Ejecutar función a través del circuit breaker"""
        circuit = self.circuits.get(circuit_name)
        if not circuit:
            circuit = self.create_circuit(circuit_name)
        
        # Verificar estado
        if circuit.state == CircuitState.OPEN:
            # Verificar si debe intentar half-open
            if circuit.opened_at:
                time_since_open = (datetime.now() - circuit.opened_at).total_seconds()
                if time_since_open >= circuit.timeout_seconds:
                    circuit.state = CircuitState.HALF_OPEN
                    circuit.success_count = 0
                    logger.info(f"Circuit {circuit_name} entering half-open state")
                else:
                    raise Exception(f"Circuit breaker {circuit_name} is OPEN")
        
        # Ejecutar función
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Éxito
            self._record_success(circuit)
            return result
        
        except Exception as e:
            # Falla
            self._record_failure(circuit)
            raise e
    
    def _record_success(self, circuit: CircuitBreaker):
        """Registrar éxito"""
        if circuit.state == CircuitState.HALF_OPEN:
            circuit.success_count += 1
            if circuit.success_count >= circuit.success_threshold:
                circuit.state = CircuitState.CLOSED
                circuit.failure_count = 0
                circuit.opened_at = None
                logger.info(f"Circuit {circuit.name} closed after recovery")
        else:
            circuit.failure_count = 0
    
    def _record_failure(self, circuit: CircuitBreaker):
        """Registrar falla"""
        circuit.failure_count += 1
        circuit.last_failure_time = datetime.now()
        
        if circuit.state == CircuitState.HALF_OPEN:
            circuit.state = CircuitState.OPEN
            circuit.opened_at = datetime.now()
            logger.warning(f"Circuit {circuit.name} opened again after half-open")
        elif circuit.failure_count >= circuit.failure_threshold:
            circuit.state = CircuitState.OPEN
            circuit.opened_at = datetime.now()
            logger.warning(f"Circuit {circuit.name} opened due to failures")
    
    def get_circuit_status(self, circuit_name: str) -> Dict[str, Any]:
        """Obtener estado del circuit breaker"""
        circuit = self.circuits.get(circuit_name)
        if not circuit:
            return {"exists": False}
        
        return {
            "name": circuit.name,
            "state": circuit.state.value,
            "failure_count": circuit.failure_count,
            "success_count": circuit.success_count,
            "last_failure_time": circuit.last_failure_time.isoformat() if circuit.last_failure_time else None,
            "opened_at": circuit.opened_at.isoformat() if circuit.opened_at else None,
        }

