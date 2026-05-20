"""
MCP Config - Configuración del servidor MCP
===========================================

Configuración flexible para el servidor MCP.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path


@dataclass
class MCPServerConfig:
    """Configuración del servidor MCP"""
    
    host: str = "localhost"
    port: int = 8025
    enable_cors: bool = True
    enable_auth: bool = False
    enable_cache: bool = True
    enable_metrics: bool = True
    enable_circuit_breaker: bool = True
    rate_limit_max_requests: int = 100
    rate_limit_window_seconds: int = 60
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_recovery_timeout: float = 60.0
    cache_max_size: int = 1000
    cache_default_ttl: Optional[float] = 300.0
    websocket_timeout: float = 300.0
    max_command_length: int = 10000
    allowed_origins: List[str] = field(default_factory=lambda: ["*"])
    api_version: str = "v1"
    enable_batch_operations: bool = True
    enable_webhooks: bool = False
    webhook_urls: List[str] = field(default_factory=list)
    user_rate_limit_max_requests: int = 50
    user_rate_limit_window_seconds: int = 60
    websocket_heartbeat_interval: Optional[float] = 30.0
    use_token_bucket_rate_limiting: bool = False
    enable_request_queue: bool = False
    request_queue_max_size: int = 1000
    request_queue_max_workers: int = 10
    connection_pool_max_connections: int = 10
    enable_request_deduplication: bool = False
    deduplication_window_seconds: float = 60.0
    deduplication_max_cache_size: int = 10000
    max_websocket_connections: Optional[int] = None
    enable_adaptive_rate_limiting: bool = False
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "MCPServerConfig":
        """Crear configuración desde diccionario"""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
    
    @classmethod
    def from_file(cls, config_path: str) -> "MCPServerConfig":
        """Cargar configuración desde archivo"""
        import json
        path = Path(config_path)
        if path.exists():
            with open(path, 'r') as f:
                return cls.from_dict(json.load(f))
        return cls()
    
    def save_to_file(self, config_path: str) -> None:
        """Guardar configuración a archivo"""
        import json
        path = Path(config_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

