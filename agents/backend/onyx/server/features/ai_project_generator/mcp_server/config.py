"""
MCP Configuration - Configuración desde archivos
================================================
"""

import os
import json
import yaml
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class MCPConfig(BaseModel):
    """Configuración del servidor MCP"""
    
    # Servidor
    host: str = Field(default="0.0.0.0", description="Host del servidor")
    port: int = Field(default=8020, description="Puerto del servidor")
    debug: bool = Field(default=False, description="Modo debug")
    
    # Seguridad
    secret_key: str = Field(..., description="Clave secreta para JWT")
    token_expire_minutes: int = Field(default=30, description="Minutos de expiración del token")
    
    # Rate Limiting
    rate_limiting_enabled: bool = Field(default=True, description="Habilitar rate limiting")
    default_rate_limit_per_minute: int = Field(default=60, description="Límite por defecto por minuto")
    default_rate_limit_per_hour: int = Field(default=1000, description="Límite por defecto por hora")
    
    # Cache
    cache_enabled: bool = Field(default=True, description="Habilitar cache")
    cache_default_ttl: int = Field(default=300, description="TTL por defecto del cache en segundos")
    
    # CORS
    cors_enabled: bool = Field(default=True, description="Habilitar CORS")
    cors_origins: list[str] = Field(default_factory=lambda: ["*"], description="Orígenes permitidos")
    
    # Observabilidad
    observability_enabled: bool = Field(default=True, description="Habilitar observabilidad")
    tracing_enabled: bool = Field(default=True, description="Habilitar tracing")
    metrics_enabled: bool = Field(default=True, description="Habilitar métricas")
    otlp_endpoint: Optional[str] = Field(None, description="Endpoint OTLP para tracing")
    
    # Manifests
    manifests_path: Optional[str] = Field(None, description="Ruta a directorio de manifests")
    
    # Retry
    retry_enabled: bool = Field(default=True, description="Habilitar retry")
    retry_max_attempts: int = Field(default=3, description="Máximo de intentos de retry")
    retry_initial_delay: float = Field(default=1.0, description="Delay inicial de retry")
    
    # Circuit Breaker
    circuit_breaker_enabled: bool = Field(default=True, description="Habilitar circuit breaker")
    circuit_breaker_failure_threshold: int = Field(default=5, description="Umbral de fallos")
    circuit_breaker_recovery_timeout: float = Field(default=60.0, description="Timeout de recuperación")
    
    # Batch
    batch_max_concurrent: int = Field(default=10, description="Máximo de operaciones concurrentes en batch")
    
    # Webhooks
    webhooks_enabled: bool = Field(default=False, description="Habilitar webhooks")
    
    # Logging
    log_level: str = Field(default="INFO", description="Nivel de logging")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Formato de logging"
    )
    
    @classmethod
    def from_file(cls, file_path: str) -> "MCPConfig":
        """
        Carga configuración desde archivo
        
        Args:
            file_path: Ruta al archivo de configuración (JSON o YAML)
            
        Returns:
            MCPConfig cargado
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {file_path}")
        
        with open(path, "r", encoding="utf-8") as f:
            if path.suffix.lower() in [".yaml", ".yml"]:
                data = yaml.safe_load(f)
            elif path.suffix.lower() == ".json":
                data = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {path.suffix}")
        
        return cls(**data)
    
    @classmethod
    def from_env(cls) -> "MCPConfig":
        """
        Carga configuración desde variables de entorno
        
        Returns:
            MCPConfig cargado desde env
        """
        return cls(
            host=os.getenv("MCP_HOST", "0.0.0.0"),
            port=int(os.getenv("MCP_PORT", "8020")),
            debug=os.getenv("MCP_DEBUG", "false").lower() == "true",
            secret_key=os.getenv("MCP_SECRET_KEY", "change-me-in-production"),
            token_expire_minutes=int(os.getenv("MCP_TOKEN_EXPIRE_MINUTES", "30")),
            rate_limiting_enabled=os.getenv("MCP_RATE_LIMITING_ENABLED", "true").lower() == "true",
            cache_enabled=os.getenv("MCP_CACHE_ENABLED", "true").lower() == "true",
            cors_enabled=os.getenv("MCP_CORS_ENABLED", "true").lower() == "true",
            observability_enabled=os.getenv("MCP_OBSERVABILITY_ENABLED", "true").lower() == "true",
            manifests_path=os.getenv("MCP_MANIFESTS_PATH"),
            otlp_endpoint=os.getenv("OTLP_ENDPOINT"),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte configuración a diccionario"""
        return self.dict(exclude_none=True)
    
    def save(self, file_path: str, format: str = "yaml"):
        """
        Guarda configuración a archivo
        
        Args:
            file_path: Ruta donde guardar
            format: Formato (yaml o json)
        """
        path = Path(file_path)
        data = self.to_dict()
        
        with open(path, "w", encoding="utf-8") as f:
            if format.lower() == "yaml":
                yaml.dump(data, f, default_flow_style=False)
            elif format.lower() == "json":
                json.dump(data, f, indent=2)
            else:
                raise ValueError(f"Unsupported format: {format}")

