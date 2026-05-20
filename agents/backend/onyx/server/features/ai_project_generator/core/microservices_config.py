"""
Microservices Configuration - Configuración avanzada para arquitectura de microservicios
=========================================================================================

Configuración para soportar arquitectura de microservicios, serverless,
y patrones cloud-native siguiendo mejores prácticas de FastAPI.
"""

import os
from typing import Optional, Dict, Any, List
from enum import Enum
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class DeploymentType(str, Enum):
    """Tipos de despliegue soportados"""
    STANDARD = "standard"
    SERVERLESS = "serverless"
    CONTAINER = "container"
    KUBERNETES = "kubernetes"


class MessageBrokerType(str, Enum):
    """Tipos de message brokers soportados"""
    RABBITMQ = "rabbitmq"
    KAFKA = "kafka"
    REDIS = "redis"
    SQS = "sqs"
    NONE = "none"


class CacheBackend(str, Enum):
    """Backends de cache soportados"""
    REDIS = "redis"
    MEMCACHED = "memcached"
    IN_MEMORY = "in_memory"
    NONE = "none"


class WorkerBackend(str, Enum):
    """Backends de workers asíncronos"""
    CELERY = "celery"
    RQ = "rq"
    ARQ = "arq"
    NONE = "none"


class APIGatewayType(str, Enum):
    """Tipos de API Gateway soportados"""
    KONG = "kong"
    AWS_API_GATEWAY = "aws_api_gateway"
    AZURE_API_MANAGEMENT = "azure_api_management"
    TRAEFIK = "traefik"
    NGINX = "nginx"
    NONE = "none"


class MicroservicesConfig(BaseSettings):
    """Configuración para arquitectura de microservicios"""
    
    # Deployment
    deployment_type: DeploymentType = Field(
        default=DeploymentType.STANDARD,
        description="Tipo de despliegue"
    )
    
    # Serverless
    serverless_provider: Optional[str] = Field(
        default=None,
        description="Proveedor serverless (aws_lambda, azure_functions, gcp_cloud_functions)"
    )
    minimize_cold_start: bool = Field(
        default=True,
        description="Minimizar cold start times"
    )
    serverless_timeout: int = Field(
        default=30,
        description="Timeout para funciones serverless (segundos)"
    )
    
    # API Gateway
    api_gateway_type: APIGatewayType = Field(
        default=APIGatewayType.NONE,
        description="Tipo de API Gateway"
    )
    api_gateway_url: Optional[str] = Field(
        default=None,
        description="URL del API Gateway"
    )
    api_gateway_key: Optional[str] = Field(
        default=None,
        description="API Key para el API Gateway"
    )
    
    # Message Broker
    message_broker_type: MessageBrokerType = Field(
        default=MessageBrokerType.NONE,
        description="Tipo de message broker"
    )
    message_broker_url: Optional[str] = Field(
        default=None,
        description="URL del message broker"
    )
    message_broker_username: Optional[str] = Field(
        default=None,
        description="Usuario del message broker"
    )
    message_broker_password: Optional[str] = Field(
        default=None,
        description="Contraseña del message broker"
    )
    
    # Cache
    cache_backend: CacheBackend = Field(
        default=CacheBackend.REDIS,
        description="Backend de cache"
    )
    cache_url: Optional[str] = Field(
        default=None,
        description="URL del cache (ej: redis://localhost:6379)"
    )
    cache_ttl: int = Field(
        default=3600,
        description="TTL por defecto para cache (segundos)"
    )
    
    # Async Workers
    worker_backend: WorkerBackend = Field(
        default=WorkerBackend.CELERY,
        description="Backend de workers asíncronos"
    )
    worker_broker_url: Optional[str] = Field(
        default=None,
        description="URL del broker para workers"
    )
    worker_result_backend: Optional[str] = Field(
        default=None,
        description="Backend de resultados para workers"
    )
    
    # Circuit Breaker
    circuit_breaker_enabled: bool = Field(
        default=True,
        description="Habilitar circuit breakers"
    )
    circuit_breaker_failure_threshold: int = Field(
        default=5,
        description="Umbral de fallos para abrir circuit breaker"
    )
    circuit_breaker_timeout: int = Field(
        default=60,
        description="Timeout para circuit breaker (segundos)"
    )
    
    # Retry
    retry_enabled: bool = Field(
        default=True,
        description="Habilitar retry logic"
    )
    retry_max_attempts: int = Field(
        default=3,
        description="Máximo número de intentos"
    )
    retry_backoff_factor: float = Field(
        default=2.0,
        description="Factor de backoff exponencial"
    )
    
    # Monitoring
    prometheus_enabled: bool = Field(
        default=True,
        description="Habilitar métricas Prometheus"
    )
    prometheus_port: int = Field(
        default=9090,
        description="Puerto para métricas Prometheus"
    )
    
    # Tracing
    opentelemetry_enabled: bool = Field(
        default=True,
        description="Habilitar OpenTelemetry tracing"
    )
    opentelemetry_endpoint: Optional[str] = Field(
        default=None,
        description="Endpoint para OpenTelemetry"
    )
    
    # Logging
    structured_logging: bool = Field(
        default=True,
        description="Usar logging estructurado (JSON)"
    )
    log_level: str = Field(
        default="INFO",
        description="Nivel de logging"
    )
    
    # Security
    oauth2_enabled: bool = Field(
        default=False,
        description="Habilitar OAuth2"
    )
    oauth2_provider: Optional[str] = Field(
        default=None,
        description="Proveedor OAuth2 (google, github, auth0, etc.)"
    )
    rate_limiting_enabled: bool = Field(
        default=True,
        description="Habilitar rate limiting"
    )
    ddos_protection_enabled: bool = Field(
        default=True,
        description="Habilitar protección DDoS"
    )
    
    # Load Balancing
    load_balancer_enabled: bool = Field(
        default=False,
        description="Habilitar load balancing"
    )
    service_mesh_enabled: bool = Field(
        default=False,
        description="Habilitar service mesh (Istio, Linkerd)"
    )
    
    class Config:
        env_prefix = "MICROSERVICES_"
        case_sensitive = False


def get_microservices_config() -> MicroservicesConfig:
    """Obtiene la configuración de microservicios"""
    return MicroservicesConfig()


def get_cache_config() -> Dict[str, Any]:
    """Obtiene configuración de cache"""
    config = get_microservices_config()
    return {
        "backend": config.cache_backend.value,
        "url": config.cache_url or os.getenv("REDIS_URL", "redis://localhost:6379"),
        "ttl": config.cache_ttl,
    }


def get_worker_config() -> Dict[str, Any]:
    """Obtiene configuración de workers"""
    config = get_microservices_config()
    return {
        "backend": config.worker_backend.value,
        "broker_url": config.worker_broker_url or config.message_broker_url,
        "result_backend": config.worker_result_backend or config.cache_url,
    }


def get_circuit_breaker_config() -> Dict[str, Any]:
    """Obtiene configuración de circuit breaker"""
    config = get_microservices_config()
    return {
        "enabled": config.circuit_breaker_enabled,
        "failure_threshold": config.circuit_breaker_failure_threshold,
        "timeout": config.circuit_breaker_timeout,
    }


def get_retry_config() -> Dict[str, Any]:
    """Obtiene configuración de retry"""
    config = get_microservices_config()
    return {
        "enabled": config.retry_enabled,
        "max_attempts": config.retry_max_attempts,
        "backoff_factor": config.retry_backoff_factor,
    }















