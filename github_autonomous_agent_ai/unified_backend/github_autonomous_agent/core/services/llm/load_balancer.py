"""
Load Balancer - Sistema de balanceo de carga para múltiples modelos LLM.

Características:
- Distribución inteligente de requests
- Health checking de modelos
- Failover automático
- Estadísticas de rendimiento
- Estrategias de balanceo (round-robin, least-connections, weighted)
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
import asyncio

from config.logging_config import get_logger

logger = get_logger(__name__)


class LoadBalanceStrategy(str, Enum):
    """Estrategias de balanceo de carga."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED = "weighted"
    RANDOM = "random"
    LEAST_LATENCY = "least_latency"
    LEAST_ERRORS = "least_errors"


@dataclass
class ModelHealth:
    """Estado de salud de un modelo."""
    model: str
    is_healthy: bool
    last_check: datetime
    consecutive_failures: int = 0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_latency_ms: float = 0.0
    current_connections: int = 0
    max_connections: int = 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            "model": self.model,
            "is_healthy": self.is_healthy,
            "last_check": self.last_check.isoformat(),
            "consecutive_failures": self.consecutive_failures,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": (
                self.successful_requests / self.total_requests
                if self.total_requests > 0 else 0.0
            ),
            "avg_latency_ms": self.avg_latency_ms,
            "current_connections": self.current_connections,
            "max_connections": self.max_connections
        }


@dataclass
class LoadBalanceConfig:
    """Configuración de balanceo de carga."""
    strategy: LoadBalanceStrategy = LoadBalanceStrategy.ROUND_ROBIN
    health_check_interval: float = 30.0  # segundos
    max_failures_before_unhealthy: int = 3
    recovery_threshold: int = 2  # requests exitosos para recuperar
    weights: Dict[str, float] = field(default_factory=dict)
    enable_health_checks: bool = True


class LoadBalancer:
    """
    Sistema de balanceo de carga para modelos LLM.
    
    Características:
    - Múltiples estrategias de balanceo
    - Health checking automático
    - Failover automático
    - Estadísticas de rendimiento
    """
    
    def __init__(self, config: Optional[LoadBalanceConfig] = None):
        """
        Inicializar load balancer.
        
        Args:
            config: Configuración de balanceo
        """
        self.config = config or LoadBalanceConfig()
        
        # Estado de modelos
        self.model_health: Dict[str, ModelHealth] = {}
        
        # Índices para round-robin
        self.round_robin_index: Dict[str, int] = defaultdict(int)
        
        # Estadísticas
        self.stats = {
            "total_requests": 0,
            "total_successful": 0,
            "total_failed": 0,
            "total_failovers": 0
        }
        
        # Lock para thread safety
        self._lock = asyncio.Lock()
        
        # Health check task
        self._health_check_task: Optional[asyncio.Task] = None
        self._running = False
    
    def register_model(
        self,
        model: str,
        max_connections: int = 100,
        initial_weight: float = 1.0
    ):
        """
        Registrar un modelo para balanceo.
        
        Args:
            model: Nombre del modelo
            max_connections: Máximo de conexiones concurrentes
            initial_weight: Peso inicial (para estrategia weighted)
        """
        if model not in self.model_health:
            self.model_health[model] = ModelHealth(
                model=model,
                is_healthy=True,
                last_check=datetime.now(),
                max_connections=max_connections
            )
            
            if model not in self.config.weights:
                self.config.weights[model] = initial_weight
            
            logger.info(f"Modelo {model} registrado en load balancer")
    
    async def select_model(
        self,
        available_models: Optional[List[str]] = None
    ) -> Optional[str]:
        """
        Seleccionar modelo según estrategia de balanceo.
        
        Args:
            available_models: Lista de modelos disponibles (opcional)
            
        Returns:
            Modelo seleccionado o None si no hay modelos disponibles
        """
        async with self._lock:
            # Filtrar modelos disponibles
            if available_models:
                models = [
                    m for m in available_models
                    if m in self.model_health
                ]
            else:
                models = list(self.model_health.keys())
            
            # Filtrar solo modelos saludables
            healthy_models = [
                m for m in models
                if self.model_health[m].is_healthy
                and self.model_health[m].current_connections < self.model_health[m].max_connections
            ]
            
            if not healthy_models:
                # Si no hay modelos saludables, usar todos
                healthy_models = models
            
            if not healthy_models:
                return None
            
            # Seleccionar según estrategia
            if self.config.strategy == LoadBalanceStrategy.ROUND_ROBIN:
                return self._select_round_robin(healthy_models)
            elif self.config.strategy == LoadBalanceStrategy.LEAST_CONNECTIONS:
                return self._select_least_connections(healthy_models)
            elif self.config.strategy == LoadBalanceStrategy.WEIGHTED:
                return self._select_weighted(healthy_models)
            elif self.config.strategy == LoadBalanceStrategy.LEAST_LATENCY:
                return self._select_least_latency(healthy_models)
            elif self.config.strategy == LoadBalanceStrategy.LEAST_ERRORS:
                return self._select_least_errors(healthy_models)
            else:  # RANDOM
                import random
                return random.choice(healthy_models)
    
    def _select_round_robin(self, models: List[str]) -> str:
        """Seleccionar modelo usando round-robin."""
        if not models:
            return None
        
        key = ",".join(sorted(models))
        index = self.round_robin_index[key] % len(models)
        self.round_robin_index[key] += 1
        return models[index]
    
    def _select_least_connections(self, models: List[str]) -> str:
        """Seleccionar modelo con menos conexiones."""
        if not models:
            return None
        
        return min(
            models,
            key=lambda m: self.model_health[m].current_connections
        )
    
    def _select_weighted(self, models: List[str]) -> str:
        """Seleccionar modelo usando pesos."""
        if not models:
            return None
        
        import random
        
        # Calcular pesos normalizados
        weights = [
            self.config.weights.get(m, 1.0) * (
                1.0 if self.model_health[m].is_healthy else 0.1
            )
            for m in models
        ]
        
        total_weight = sum(weights)
        if total_weight == 0:
            return random.choice(models)
        
        # Selección aleatoria ponderada
        r = random.uniform(0, total_weight)
        cumulative = 0
        
        for i, weight in enumerate(weights):
            cumulative += weight
            if r <= cumulative:
                return models[i]
        
        return models[-1]
    
    def _select_least_latency(self, models: List[str]) -> str:
        """Seleccionar modelo con menor latencia promedio."""
        if not models:
            return None
        
        return min(
            models,
            key=lambda m: self.model_health[m].avg_latency_ms
        )
    
    def _select_least_errors(self, models: List[str]) -> str:
        """Seleccionar modelo con menor tasa de errores."""
        if not models:
            return None
        
        return min(
            models,
            key=lambda m: (
                self.model_health[m].failed_requests / max(
                    self.model_health[m].total_requests, 1
                )
            )
        )
    
    async def record_request(
        self,
        model: str,
        success: bool,
        latency_ms: float
    ):
        """
        Registrar resultado de un request.
        
        Args:
            model: Modelo usado
            success: Si fue exitoso
            latency_ms: Latencia en milisegundos
        """
        async with self._lock:
            if model not in self.model_health:
                self.register_model(model)
            
            health = self.model_health[model]
            health.total_requests += 1
            health.last_check = datetime.now()
            
            if success:
                health.successful_requests += 1
                health.consecutive_failures = 0
                
                # Actualizar latencia promedio
                if health.total_requests == 1:
                    health.avg_latency_ms = latency_ms
                else:
                    health.avg_latency_ms = (
                        (health.avg_latency_ms * (health.total_requests - 1) + latency_ms)
                        / health.total_requests
                    )
                
                # Recuperar si estaba no saludable
                if not health.is_healthy:
                    if health.successful_requests >= self.config.recovery_threshold:
                        health.is_healthy = True
                        logger.info(f"Modelo {model} recuperado")
            else:
                health.failed_requests += 1
                health.consecutive_failures += 1
                
                # Marcar como no saludable si excede umbral
                if (health.consecutive_failures >= self.config.max_failures_before_unhealthy
                    and health.is_healthy):
                    health.is_healthy = False
                    logger.warning(
                        f"Modelo {model} marcado como no saludable "
                        f"({health.consecutive_failures} fallos consecutivos)"
                    )
            
            self.stats["total_requests"] += 1
            if success:
                self.stats["total_successful"] += 1
            else:
                self.stats["total_failed"] += 1
    
    async def increment_connection(self, model: str):
        """Incrementar contador de conexiones."""
        async with self._lock:
            if model in self.model_health:
                self.model_health[model].current_connections += 1
    
    async def decrement_connection(self, model: str):
        """Decrementar contador de conexiones."""
        async with self._lock:
            if model in self.model_health:
                self.model_health[model].current_connections = max(
                    0,
                    self.model_health[model].current_connections - 1
                )
    
    async def start_health_checks(self):
        """Iniciar health checks automáticos."""
        if self._running or not self.config.enable_health_checks:
            return
        
        self._running = True
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        logger.info("Health checks iniciados")
    
    async def stop_health_checks(self):
        """Detener health checks."""
        self._running = False
        if self._health_check_task:
            await self._health_check_task
        logger.info("Health checks detenidos")
    
    async def _health_check_loop(self):
        """Loop de health checks."""
        while self._running:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                
                # Health check básico: verificar si modelos tienen muchos fallos
                async with self._lock:
                    for model, health in self.model_health.items():
                        # Si no ha tenido requests recientes, considerar saludable
                        time_since_last = (datetime.now() - health.last_check).total_seconds()
                        if time_since_last > self.config.health_check_interval * 2:
                            # Resetear fallos consecutivos si ha pasado mucho tiempo
                            if health.consecutive_failures > 0:
                                health.consecutive_failures = 0
                                if not health.is_healthy:
                                    health.is_healthy = True
                                    logger.info(f"Modelo {model} recuperado por timeout")
                
            except Exception as e:
                logger.error(f"Error en health check: {e}")
    
    def get_health_status(self, model: Optional[str] = None) -> Dict[str, Any]:
        """
        Obtener estado de salud.
        
        Args:
            model: Modelo específico (opcional, todos si no se proporciona)
            
        Returns:
            Estado de salud
        """
        if model:
            if model in self.model_health:
                return self.model_health[model].to_dict()
            return {}
        
        return {
            m: h.to_dict()
            for m, h in self.model_health.items()
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del load balancer."""
        return {
            **self.stats,
            "success_rate": (
                self.stats["total_successful"] / self.stats["total_requests"]
                if self.stats["total_requests"] > 0 else 0.0
            ),
            "models_count": len(self.model_health),
            "healthy_models": sum(
                1 for h in self.model_health.values() if h.is_healthy
            )
        }


def get_load_balancer() -> LoadBalancer:
    """Factory function para obtener instancia singleton del load balancer."""
    if not hasattr(get_load_balancer, "_instance"):
        get_load_balancer._instance = LoadBalancer()
    return get_load_balancer._instance



