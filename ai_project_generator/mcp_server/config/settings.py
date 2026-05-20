"""
MCP Settings - Configuración mejorada con validaciones
======================================================

Configuración del servidor MCP con validaciones robustas,
soporte para múltiples fuentes y valores por defecto sensatos.
"""

import os
import logging
from typing import Optional, List, Dict, Any
from pathlib import Path
from pydantic import BaseModel, Field, validator, root_validator

logger = logging.getLogger(__name__)


class ServerSettings(BaseModel):
    """Configuración del servidor"""
    
    host: str = Field(default="0.0.0.0", description="Host del servidor")
    port: int = Field(default=8020, description="Puerto del servidor", ge=1, le=65535)
    debug: bool = Field(default=False, description="Modo debug")
    reload: bool = Field(default=False, description="Auto-reload en desarrollo")
    workers: int = Field(default=1, description="Número de workers", ge=1)
    
    @validator("host")
    def validate_host(cls, v: str) -> str:
        """Validar host"""
        if not v or not v.strip():
            raise ValueError("host cannot be empty")
        return v.strip()


class SecuritySettings(BaseModel):
    """Configuración de seguridad"""
    
    secret_key: str = Field(..., description="Clave secreta para JWT")
    token_expire_minutes: int = Field(default=30, description="Minutos de expiración", ge=1, le=1440)
    algorithm: str = Field(default="HS256", description="Algoritmo JWT")
    require_https: bool = Field(default=False, description="Requerir HTTPS en producción")
    
    @validator("secret_key")
    def validate_secret_key(cls, v: str) -> str:
        """Validar que secret_key tenga longitud mínima"""
        if not v or len(v) < 32:
            raise ValueError("secret_key must be at least 32 characters long")
        if v == "change-me-in-production":
            logger.warning("Using default secret_key - change in production!")
        return v


class RateLimitSettings(BaseModel):
    """Configuración de rate limiting"""
    
    enabled: bool = Field(default=True, description="Habilitar rate limiting")
    requests_per_minute: int = Field(default=60, description="Límite por minuto", ge=1)
    requests_per_hour: int = Field(default=1000, description="Límite por hora", ge=1)
    per_user: bool = Field(default=True, description="Rate limit por usuario")
    per_ip: bool = Field(default=True, description="Rate limit por IP")
    
    @root_validator
    def validate_limits(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validar que límites sean consistentes"""
        per_minute = values.get("requests_per_minute", 60)
        per_hour = values.get("requests_per_hour", 1000)
        
        if per_minute > per_hour:
            raise ValueError("requests_per_minute cannot be greater than requests_per_hour")
        
        return values


class CacheSettings(BaseModel):
    """Configuración de cache"""
    
    enabled: bool = Field(default=True, description="Habilitar cache")
    default_ttl: int = Field(default=300, description="TTL por defecto en segundos", ge=1)
    max_size: int = Field(default=1000, description="Tamaño máximo del cache", ge=1)
    strategy: str = Field(default="lru", description="Estrategia de cache (lru, fifo, lfu)")


class CORSSettings(BaseModel):
    """Configuración de CORS"""
    
    enabled: bool = Field(default=True, description="Habilitar CORS")
    origins: List[str] = Field(default_factory=lambda: ["*"], description="Orígenes permitidos")
    allow_credentials: bool = Field(default=True, description="Permitir credenciales")
    allow_methods: List[str] = Field(
        default_factory=lambda: ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        description="Métodos permitidos"
    )
    allow_headers: List[str] = Field(
        default_factory=lambda: ["*"],
        description="Headers permitidos"
    )
    max_age: int = Field(default=3600, description="Max age en segundos", ge=0)


class ObservabilitySettings(BaseModel):
    """Configuración de observabilidad"""
    
    enabled: bool = Field(default=True, description="Habilitar observabilidad")
    tracing_enabled: bool = Field(default=True, description="Habilitar tracing")
    metrics_enabled: bool = Field(default=True, description="Habilitar métricas")
    otlp_endpoint: Optional[str] = Field(None, description="Endpoint OTLP para tracing")
    log_level: str = Field(default="INFO", description="Nivel de logging")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Formato de logging"
    )
    
    @validator("log_level")
    def validate_log_level(cls, v: str) -> str:
        """Validar nivel de logging"""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return v.upper()


class MCPSettings(BaseModel):
    """
    Configuración completa del servidor MCP.
    
    Combina todas las configuraciones en una sola clase
    con validaciones y valores por defecto.
    """
    
    server: ServerSettings = Field(default_factory=ServerSettings)
    security: SecuritySettings
    rate_limiting: RateLimitSettings = Field(default_factory=RateLimitSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    cors: CORSSettings = Field(default_factory=CORSSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)
    
    # Configuración adicional
    manifests_path: Optional[str] = Field(None, description="Ruta a directorio de manifests")
    
    class Config:
        """Configuración de Pydantic"""
        env_prefix = "MCP_"
        case_sensitive = False
        extra = "ignore"
    
    @validator("manifests_path")
    def validate_manifests_path(cls, v: Optional[str]) -> Optional[str]:
        """Validar que manifests_path exista si se proporciona"""
        if v is None:
            return None
        
        path = Path(v)
        if not path.exists():
            logger.warning(f"Manifests path does not exist: {v}")
        elif not path.is_dir():
            raise ValueError(f"manifests_path must be a directory: {v}")
        
        return str(path.absolute())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir configuración a diccionario"""
        return self.dict(exclude_none=True)
    
    def validate(self) -> List[str]:
        """
        Validar configuración completa.
        
        Returns:
            Lista de advertencias (vacía si todo está bien)
        """
        warnings: List[str] = []
        
        # Validar secret_key en producción
        if not self.server.debug and self.security.secret_key == "change-me-in-production":
            warnings.append("Using default secret_key in production is insecure")
        
        # Validar HTTPS en producción
        if not self.server.debug and not self.security.require_https:
            warnings.append("HTTPS not required in production configuration")
        
        # Validar manifests_path
        if self.manifests_path and not Path(self.manifests_path).exists():
            warnings.append(f"Manifests path does not exist: {self.manifests_path}")
        
        return warnings


def get_settings() -> MCPSettings:
    """
    Obtener configuración desde variables de entorno.
    
    Returns:
        MCPSettings cargado desde env
    """
    return MCPSettings(
        server=ServerSettings(
            host=os.getenv("MCP_HOST", "0.0.0.0"),
            port=int(os.getenv("MCP_PORT", "8020")),
            debug=os.getenv("MCP_DEBUG", "false").lower() == "true",
            reload=os.getenv("MCP_RELOAD", "false").lower() == "true",
            workers=int(os.getenv("MCP_WORKERS", "1")),
        ),
        security=SecuritySettings(
            secret_key=os.getenv("MCP_SECRET_KEY", "change-me-in-production"),
            token_expire_minutes=int(os.getenv("MCP_TOKEN_EXPIRE_MINUTES", "30")),
            algorithm=os.getenv("MCP_JWT_ALGORITHM", "HS256"),
            require_https=os.getenv("MCP_REQUIRE_HTTPS", "false").lower() == "true",
        ),
        rate_limiting=RateLimitSettings(
            enabled=os.getenv("MCP_RATE_LIMITING_ENABLED", "true").lower() == "true",
            requests_per_minute=int(os.getenv("MCP_RATE_LIMIT_PER_MINUTE", "60")),
            requests_per_hour=int(os.getenv("MCP_RATE_LIMIT_PER_HOUR", "1000")),
            per_user=os.getenv("MCP_RATE_LIMIT_PER_USER", "true").lower() == "true",
            per_ip=os.getenv("MCP_RATE_LIMIT_PER_IP", "true").lower() == "true",
        ),
        cache=CacheSettings(
            enabled=os.getenv("MCP_CACHE_ENABLED", "true").lower() == "true",
            default_ttl=int(os.getenv("MCP_CACHE_TTL", "300")),
            max_size=int(os.getenv("MCP_CACHE_MAX_SIZE", "1000")),
            strategy=os.getenv("MCP_CACHE_STRATEGY", "lru"),
        ),
        cors=CORSSettings(
            enabled=os.getenv("MCP_CORS_ENABLED", "true").lower() == "true",
            origins=os.getenv("MCP_CORS_ORIGINS", "*").split(","),
            allow_credentials=os.getenv("MCP_CORS_ALLOW_CREDENTIALS", "true").lower() == "true",
            max_age=int(os.getenv("MCP_CORS_MAX_AGE", "3600")),
        ),
        observability=ObservabilitySettings(
            enabled=os.getenv("MCP_OBSERVABILITY_ENABLED", "true").lower() == "true",
            tracing_enabled=os.getenv("MCP_TRACING_ENABLED", "true").lower() == "true",
            metrics_enabled=os.getenv("MCP_METRICS_ENABLED", "true").lower() == "true",
            otlp_endpoint=os.getenv("OTLP_ENDPOINT"),
            log_level=os.getenv("MCP_LOG_LEVEL", "INFO"),
        ),
        manifests_path=os.getenv("MCP_MANIFESTS_PATH"),
    )


def load_settings(
    file_path: Optional[str] = None,
    use_env: bool = True,
    validate: bool = True
) -> MCPSettings:
    """
    Cargar configuración desde archivo y/o variables de entorno.
    
    Args:
        file_path: Ruta al archivo de configuración (opcional)
        use_env: Si usar variables de entorno (default: True)
        validate: Si validar configuración (default: True)
        
    Returns:
        MCPSettings cargado
        
    Raises:
        FileNotFoundError: Si el archivo no existe
        ValueError: Si la configuración es inválida
    """
    settings: Optional[MCPSettings] = None
    
    # Cargar desde archivo si se proporciona
    if file_path:
        from .loader import load_config_from_file
        settings = load_config_from_file(file_path)
    
    # Cargar desde env si se solicita
    if use_env:
        env_settings = get_settings()
        
        # Combinar: archivo primero, luego env (env tiene prioridad)
        if settings:
            # Actualizar con valores de env (env tiene prioridad)
            settings_dict = settings.dict()
            env_dict = env_settings.dict()
            
            # Merge profundo
            def deep_update(base: dict, update: dict) -> dict:
                for key, value in update.items():
                    if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                        base[key] = deep_update(base[key], value)
                    else:
                        base[key] = value
                return base
            
            settings_dict = deep_update(settings_dict, env_dict)
            settings = MCPSettings(**settings_dict)
        else:
            settings = env_settings
    
    if settings is None:
        raise ValueError("No configuration source provided")
    
    # Validar si se solicita
    if validate:
        warnings = settings.validate()
        if warnings:
            for warning in warnings:
                logger.warning(f"Configuration warning: {warning}")
    
    return settings

