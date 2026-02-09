"""
Service Mesh System
===================
Sistema de service mesh para comunicación entre servicios
"""

import time
import json
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum


class TrafficPolicy(Enum):
    """Políticas de tráfico"""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    RANDOM = "random"
    WEIGHTED = "weighted"


class CircuitBreakerState(Enum):
    """Estados de circuit breaker"""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class ServiceMeshConfig:
    """Configuración de service mesh"""
    service_name: str
    retry_count: int = 3
    timeout: float = 30.0
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 60.0
    traffic_policy: TrafficPolicy = TrafficPolicy.ROUND_ROBIN


@dataclass
class ServiceCall:
    """Llamada a servicio"""
    id: str
    service_name: str
    method: str
    endpoint: str
    started_at: float
    completed_at: Optional[float] = None
    status_code: Optional[int] = None
    error: Optional[str] = None
    retry_count: int = 0


class ServiceMesh:
    """
    Sistema de service mesh
    """
    
    def __init__(self):
        self.services: Dict[str, ServiceMeshConfig] = {}
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}  # service_name -> circuit_breaker_state
        self.service_calls: List[ServiceCall] = []
        self.load_balancers: Dict[str, List[str]] = {}  # service_name -> [endpoint_ids]
    
    def register_service(self, config: ServiceMeshConfig):
        """Registrar servicio en mesh"""
        self.services[config.service_name] = config
        self.circuit_breakers[config.service_name] = {
            'state': CircuitBreakerState.CLOSED.value,
            'failure_count': 0,
            'last_failure_time': None,
            'success_count': 0
        }
    
    def call_service(
        self,
        service_name: str,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Llamar a servicio a través del mesh
        
        Args:
            service_name: Nombre del servicio
            method: Método HTTP
            endpoint: Endpoint
            data: Datos a enviar
            headers: Headers HTTP
        """
        if service_name not in self.services:
            raise ValueError(f"Service {service_name} not registered")
        
        config = self.services[service_name]
        
        # Verificar circuit breaker
        if not self._check_circuit_breaker(service_name, config):
            raise Exception(f"Circuit breaker is OPEN for service {service_name}")
        
        call_id = f"call_{int(time.time() * 1000)}"
        service_call = ServiceCall(
            id=call_id,
            service_name=service_name,
            method=method,
            endpoint=endpoint,
            started_at=time.time()
        )
        
        self.service_calls.append(service_call)
        
        # Retry logic
        last_error = None
        for attempt in range(config.retry_count):
            service_call.retry_count = attempt
            
            try:
                # En implementación real, hacer llamada HTTP real
                result = self._make_http_call(
                    service_name,
                    method,
                    endpoint,
                    data,
                    headers,
                    config.timeout
                )
                
                service_call.completed_at = time.time()
                service_call.status_code = 200
                
                # Registrar éxito en circuit breaker
                self._record_success(service_name, config)
                
                return result
                
            except Exception as e:
                last_error = e
                service_call.error = str(e)
                
                if attempt < config.retry_count - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        # Registrar fallo
        service_call.completed_at = time.time()
        service_call.status_code = 500
        self._record_failure(service_name, config)
        
        raise Exception(f"Service call failed after {config.retry_count} attempts: {last_error}")
    
    def _check_circuit_breaker(
        self,
        service_name: str,
        config: ServiceMeshConfig
    ) -> bool:
        """Verificar circuit breaker"""
        cb = self.circuit_breakers[service_name]
        
        if cb['state'] == CircuitBreakerState.OPEN.value:
            # Verificar si debe pasar a HALF_OPEN
            if cb['last_failure_time']:
                if time.time() - cb['last_failure_time'] > config.circuit_breaker_timeout:
                    cb['state'] = CircuitBreakerState.HALF_OPEN.value
                    cb['success_count'] = 0
                    return True
            return False
        
        return True
    
    def _record_success(self, service_name: str, config: ServiceMeshConfig):
        """Registrar éxito en circuit breaker"""
        cb = self.circuit_breakers[service_name]
        
        if cb['state'] == CircuitBreakerState.HALF_OPEN.value:
            cb['success_count'] += 1
            if cb['success_count'] >= 3:  # 3 éxitos consecutivos
                cb['state'] = CircuitBreakerState.CLOSED.value
                cb['failure_count'] = 0
        else:
            cb['failure_count'] = 0
    
    def _record_failure(self, service_name: str, config: ServiceMeshConfig):
        """Registrar fallo en circuit breaker"""
        cb = self.circuit_breakers[service_name]
        cb['failure_count'] += 1
        cb['last_failure_time'] = time.time()
        
        if cb['failure_count'] >= config.circuit_breaker_threshold:
            cb['state'] = CircuitBreakerState.OPEN.value
    
    def _make_http_call(
        self,
        service_name: str,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]],
        headers: Optional[Dict[str, str]],
        timeout: float
    ) -> Dict[str, Any]:
        """Hacer llamada HTTP (placeholder)"""
        # En implementación real, usar requests o httpx
        # Por ahora, simular llamada
        time.sleep(0.1)  # Simular latencia
        
        return {
            'status': 'success',
            'data': {'result': 'mock_response'}
        }
    
    def get_service_health(self, service_name: str) -> Dict[str, Any]:
        """Obtener salud de servicio"""
        if service_name not in self.services:
            return {}
        
        cb = self.circuit_breakers[service_name]
        recent_calls = [
            call for call in self.service_calls[-100:]
            if call.service_name == service_name
        ]
        
        successful = len([c for c in recent_calls if c.status_code == 200])
        failed = len([c for c in recent_calls if c.status_code != 200 and c.status_code])
        
        return {
            'service_name': service_name,
            'circuit_breaker_state': cb['state'],
            'failure_count': cb['failure_count'],
            'recent_calls': len(recent_calls),
            'success_rate': successful / len(recent_calls) if recent_calls else 0,
            'failed_calls': failed
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas del mesh"""
        return {
            'total_services': len(self.services),
            'total_calls': len(self.service_calls),
            'circuit_breakers': {
                name: cb['state']
                for name, cb in self.circuit_breakers.items()
            },
            'services': [
                self.get_service_health(name)
                for name in self.services.keys()
            ]
        }


# Instancia global
service_mesh = ServiceMesh()

