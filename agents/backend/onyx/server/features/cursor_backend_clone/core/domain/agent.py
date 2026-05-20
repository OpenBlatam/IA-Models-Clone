"""
Domain Models - Agent Status, Config, and Task
==============================================

Domain models for the Cursor Agent system.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from datetime import datetime


class AgentStatus(Enum):
    """Estados del agente"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class AgentConfig:
    """
    Configuración del agente.
    
    Soporta carga desde variables de entorno con prefijo AGENT_.
    """
    check_interval: float = 1.0
    max_concurrent_tasks: int = 5
    task_timeout: float = 300.0
    auto_restart: bool = True
    persistent_storage: bool = True
    storage_path: str = "./data/agent_state.json"
    command_file: Optional[str] = None
    watch_directory: Optional[str] = None
    
    @classmethod
    def from_constants(cls) -> "AgentConfig":
        """Crear configuración desde constantes del sistema"""
        from ..constants import (
            DEFAULT_CHECK_INTERVAL,
            MAX_CONCURRENT_TASKS,
            DEFAULT_TASK_TIMEOUT,
            DEFAULT_STORAGE_PATH
        )
        return cls(
            check_interval=DEFAULT_CHECK_INTERVAL,
            max_concurrent_tasks=MAX_CONCURRENT_TASKS,
            task_timeout=DEFAULT_TASK_TIMEOUT,
            storage_path=DEFAULT_STORAGE_PATH
        )
    
    @classmethod
    def from_env(cls) -> "AgentConfig":
        """
        Crear configuración desde variables de entorno.
        
        Variables de entorno soportadas:
        - AGENT_CHECK_INTERVAL: Intervalo de verificación en segundos
        - AGENT_MAX_CONCURRENT_TASKS: Máximo de tareas concurrentes
        - AGENT_TASK_TIMEOUT: Timeout de tareas en segundos
        - AGENT_AUTO_RESTART: Auto reinicio (true/false)
        - AGENT_PERSISTENT_STORAGE: Almacenamiento persistente (true/false)
        - AGENT_STORAGE_PATH: Ruta de almacenamiento
        - AGENT_COMMAND_FILE: Archivo de comandos a monitorear
        - AGENT_WATCH_DIRECTORY: Directorio a monitorear
        
        Returns:
            AgentConfig configurado desde variables de entorno
        """
        import os
        from ..constants import (
            DEFAULT_CHECK_INTERVAL,
            MAX_CONCURRENT_TASKS,
            DEFAULT_TASK_TIMEOUT,
            DEFAULT_STORAGE_PATH
        )
        
        def get_bool_env(key: str, default: bool) -> bool:
            """Obtener valor booleano de variable de entorno"""
            value = os.getenv(key)
            if value is None:
                return default
            return value.lower() in ('true', '1', 'yes', 'on')
        
        def get_float_env(key: str, default: float) -> float:
            """Obtener valor float de variable de entorno"""
            value = os.getenv(key)
            if value is None:
                return default
            try:
                return float(value)
            except ValueError:
                return default
        
        def get_int_env(key: str, default: int) -> int:
            """Obtener valor int de variable de entorno"""
            value = os.getenv(key)
            if value is None:
                return default
            try:
                return int(value)
            except ValueError:
                return default
        
        return cls(
            check_interval=get_float_env("AGENT_CHECK_INTERVAL", DEFAULT_CHECK_INTERVAL),
            max_concurrent_tasks=get_int_env("AGENT_MAX_CONCURRENT_TASKS", MAX_CONCURRENT_TASKS),
            task_timeout=get_float_env("AGENT_TASK_TIMEOUT", DEFAULT_TASK_TIMEOUT),
            auto_restart=get_bool_env("AGENT_AUTO_RESTART", True),
            persistent_storage=get_bool_env("AGENT_PERSISTENT_STORAGE", True),
            storage_path=os.getenv("AGENT_STORAGE_PATH", DEFAULT_STORAGE_PATH),
            command_file=os.getenv("AGENT_COMMAND_FILE"),
            watch_directory=os.getenv("AGENT_WATCH_DIRECTORY")
        )
    
    @classmethod
    def from_env_or_constants(cls) -> "AgentConfig":
        """
        Crear configuración desde variables de entorno o constantes.
        
        Prioriza variables de entorno, usa constantes como fallback.
        
        Returns:
            AgentConfig configurado
        """
        try:
            return cls.from_env()
        except Exception:
            return cls.from_constants()


@dataclass
class Task:
    """Tarea a ejecutar"""
    id: str
    command: str
    timestamp: datetime = field(default_factory=datetime.now)
    status: str = "pending"
    priority: int = 0
    result: Optional[str] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}






