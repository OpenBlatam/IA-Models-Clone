"""
Configuration Manager - Refactored
===================================

Gestión centralizada y type-safe de configuración del sistema.
Soporta múltiples fuentes: variables de entorno, archivos, y valores por defecto.
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class AgentSettings:
    """Configuración del agente."""
    check_interval: float = 1.0
    max_concurrent_tasks: int = 5
    task_timeout: float = 300.0
    auto_restart: bool = True
    persistent_storage: bool = True
    storage_path: str = "./data/agent_state.json"
    command_file: Optional[str] = None
    watch_directory: Optional[str] = None
    enable_devin_persona: bool = True
    devin_mode: str = "standard"
    devin_language: str = "es"


@dataclass
class APISettings:
    """Configuración de la API."""
    host: str = "0.0.0.0"
    port: int = 8024
    cors_origins: list[str] = field(default_factory=lambda: ["*"])
    debug: bool = False
    reload: bool = False


@dataclass
class CursorAPISettings:
    """Configuración de Cursor API."""
    api_key: Optional[str] = None
    api_url: str = "https://api.cursor.sh"
    webhook_url: Optional[str] = None


@dataclass
class AWSSettings:
    """Configuración de AWS."""
    region: str = "us-east-1"
    lambda_function_name: Optional[str] = None
    dynamodb_table: str = "cursor-agent-state"
    redis_endpoint: Optional[str] = None
    cache_type: str = "elasticache"
    cloudwatch_log_group: str = "/aws/cursor-agent-24-7"


@dataclass
class RedisSettings:
    """Configuración de Redis."""
    url: Optional[str] = None
    endpoint: Optional[str] = None
    port: int = 6379
    db: int = 0


@dataclass
class SystemConfig:
    """
    Configuración completa del sistema.
    
    Agrupa todas las configuraciones en un solo objeto type-safe.
    """
    agent: AgentSettings = field(default_factory=AgentSettings)
    api: APISettings = field(default_factory=APISettings)
    cursor_api: CursorAPISettings = field(default_factory=CursorAPISettings)
    aws: AWSSettings = field(default_factory=AWSSettings)
    redis: RedisSettings = field(default_factory=RedisSettings)
    
    @classmethod
    def from_env(cls) -> "SystemConfig":
        """
        Crear configuración desde variables de entorno.
        
        Returns:
            Instancia de SystemConfig cargada desde variables de entorno.
        """
        base_dir = Path(__file__).parent.parent.parent
        
        # Agent settings
        agent = AgentSettings(
            check_interval=float(os.getenv("AGENT_CHECK_INTERVAL", "1.0")),
            max_concurrent_tasks=int(os.getenv("AGENT_MAX_CONCURRENT_TASKS", "5")),
            task_timeout=float(os.getenv("AGENT_TASK_TIMEOUT", "300.0")),
            auto_restart=os.getenv("AGENT_AUTO_RESTART", "true").lower() == "true",
            persistent_storage=os.getenv("AGENT_PERSISTENT_STORAGE", "true").lower() == "true",
            storage_path=os.getenv(
                "AGENT_STORAGE_PATH",
                str(base_dir / "data" / "agent_state.json")
            ),
            command_file=os.getenv("AGENT_COMMAND_FILE"),
            watch_directory=os.getenv("AGENT_WATCH_DIRECTORY"),
            enable_devin_persona=os.getenv("AGENT_ENABLE_DEVIN", "true").lower() == "true",
            devin_mode=os.getenv("AGENT_DEVIN_MODE", "standard"),
            devin_language=os.getenv("AGENT_DEVIN_LANGUAGE", "es"),
        )
        
        # API settings
        api = APISettings(
            host=os.getenv("API_HOST", "0.0.0.0"),
            port=int(os.getenv("API_PORT", "8024")),
            cors_origins=os.getenv("API_CORS_ORIGINS", "*").split(","),
            debug=os.getenv("API_DEBUG", "false").lower() == "true",
            reload=os.getenv("API_RELOAD", "false").lower() == "true",
        )
        
        # Cursor API settings
        cursor_api = CursorAPISettings(
            api_key=os.getenv("CURSOR_API_KEY"),
            api_url=os.getenv("CURSOR_API_URL", "https://api.cursor.sh"),
            webhook_url=os.getenv("CURSOR_WEBHOOK_URL"),
        )
        
        # AWS settings
        aws = AWSSettings(
            region=os.getenv("AWS_REGION", "us-east-1"),
            lambda_function_name=os.getenv("AWS_LAMBDA_FUNCTION_NAME"),
            dynamodb_table=os.getenv("DYNAMODB_TABLE_NAME", "cursor-agent-state"),
            redis_endpoint=os.getenv("REDIS_ENDPOINT"),
            cache_type=os.getenv("CACHE_TYPE", "elasticache"),
            cloudwatch_log_group=os.getenv(
                "CLOUDWATCH_LOG_GROUP",
                "/aws/cursor-agent-24-7"
            ),
        )
        
        # Redis settings
        redis = RedisSettings(
            url=os.getenv("REDIS_URL"),
            endpoint=os.getenv("REDIS_ENDPOINT"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            db=int(os.getenv("REDIS_DB", "0")),
        )
        
        return cls(
            agent=agent,
            api=api,
            cursor_api=cursor_api,
            aws=aws,
            redis=redis
        )
    
    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> "SystemConfig":
        """
        Cargar configuración desde archivo JSON.
        
        Args:
            config_path: Ruta al archivo de configuración.
        
        Returns:
            Instancia de SystemConfig cargada desde archivo.
        
        Raises:
            FileNotFoundError: Si el archivo no existe.
            json.JSONDecodeError: Si el archivo no es JSON válido.
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SystemConfig":
        """
        Crear configuración desde diccionario.
        
        Args:
            data: Diccionario con valores de configuración.
        
        Returns:
            Instancia de SystemConfig.
        """
        agent_data = data.get("agent", {})
        api_data = data.get("api", {})
        cursor_api_data = data.get("cursor_api", {})
        aws_data = data.get("aws", {})
        redis_data = data.get("redis", {})
        
        return cls(
            agent=AgentSettings(**agent_data),
            api=APISettings(**api_data),
            cursor_api=CursorAPISettings(**cursor_api_data),
            aws=AWSSettings(**aws_data),
            redis=RedisSettings(**redis_data),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convertir configuración a diccionario.
        
        Returns:
            Diccionario con toda la configuración.
        """
        return {
            "agent": {
                "check_interval": self.agent.check_interval,
                "max_concurrent_tasks": self.agent.max_concurrent_tasks,
                "task_timeout": self.agent.task_timeout,
                "auto_restart": self.agent.auto_restart,
                "persistent_storage": self.agent.persistent_storage,
                "storage_path": self.agent.storage_path,
                "command_file": self.agent.command_file,
                "watch_directory": self.agent.watch_directory,
                "enable_devin_persona": self.agent.enable_devin_persona,
                "devin_mode": self.agent.devin_mode,
                "devin_language": self.agent.devin_language,
            },
            "api": {
                "host": self.api.host,
                "port": self.api.port,
                "cors_origins": self.api.cors_origins,
                "debug": self.api.debug,
                "reload": self.api.reload,
            },
            "cursor_api": {
                "api_key": self.cursor_api.api_key,
                "api_url": self.cursor_api.api_url,
                "webhook_url": self.cursor_api.webhook_url,
            },
            "aws": {
                "region": self.aws.region,
                "lambda_function_name": self.aws.lambda_function_name,
                "dynamodb_table": self.aws.dynamodb_table,
                "redis_endpoint": self.aws.redis_endpoint,
                "cache_type": self.aws.cache_type,
                "cloudwatch_log_group": self.aws.cloudwatch_log_group,
            },
            "redis": {
                "url": self.redis.url,
                "endpoint": self.redis.endpoint,
                "port": self.redis.port,
                "db": self.redis.db,
            },
        }
    
    def save_to_file(self, config_path: Union[str, Path]) -> None:
        """
        Guardar configuración en archivo JSON.
        
        Args:
            config_path: Ruta donde guardar la configuración.
        """
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


# Instancia global de configuración
_config: Optional[SystemConfig] = None


def get_config() -> SystemConfig:
    """
    Obtener configuración del sistema (singleton).
    
    La primera llamada carga la configuración desde variables de entorno.
    Llamadas subsecuentes retornan la misma instancia.
    
    Returns:
        Instancia de SystemConfig.
    """
    global _config
    if _config is None:
        _config = SystemConfig.from_env()
        logger.debug("Configuration loaded from environment variables")
    return _config


def set_config(config: SystemConfig) -> None:
    """
    Establecer configuración del sistema.
    
    Args:
        config: Nueva configuración.
    """
    global _config
    _config = config
    logger.debug("Configuration updated")


def get_config_manager() -> SystemConfig:
    """
    Obtener gestor de configuración (alias de get_config para compatibilidad).
    
    Returns:
        Instancia de SystemConfig.
    """
    return get_config()


def reload_config() -> SystemConfig:
    """
    Recargar configuración desde variables de entorno.
    
    Returns:
        Nueva instancia de SystemConfig.
    """
    global _config
    _config = SystemConfig.from_env()
    logger.info("Configuration reloaded from environment variables")
    return _config




