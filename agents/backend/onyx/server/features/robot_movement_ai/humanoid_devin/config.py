"""
Configuración para robot humanoide (optimizado)

Incluye settings, constantes, y configuración de la aplicación.
"""

import os
from typing import Optional, List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Configuración de la aplicación para robot humanoide"""
    
    # Aplicación
    app_name: str = "Humanoid Devin Robot API"
    app_version: str = "1.0.0"
    debug: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # Robot
    robot_ip: str = os.getenv("ROBOT_IP", "192.168.1.100")
    robot_port: int = int(os.getenv("ROBOT_PORT", "30001"))
    dof: int = int(os.getenv("DOF", "32"))  # Grados de libertad
    
    # Servidor
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8020"))
    
    # LLM
    llm_provider: Optional[str] = os.getenv("LLM_PROVIDER", None)
    llm_api_key: Optional[str] = os.getenv("LLM_API_KEY", None)
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4")
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    llm_max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "1000"))
    
    # CORS
    cors_origins: List[str] = os.getenv(
        "CORS_ORIGINS",
        "*"
    ).split(",") if os.getenv("CORS_ORIGINS") != "*" else ["*"]
    cors_allow_credentials: bool = True
    cors_allow_methods: List[str] = ["*"]
    cors_allow_headers: List[str] = ["*"]
    
    # Movimiento
    default_speed: float = float(os.getenv("DEFAULT_SPEED", "0.5"))
    max_speed: float = float(os.getenv("MAX_SPEED", "1.0"))
    default_distance: float = float(os.getenv("DEFAULT_DISTANCE", "1.0"))
    max_distance: float = float(os.getenv("MAX_DISTANCE", "10.0"))
    
    # Modelos de IA
    tensorflow_enabled: bool = os.getenv("TENSORFLOW_ENABLED", "True").lower() == "true"
    pytorch_enabled: bool = os.getenv("PYTORCH_ENABLED", "True").lower() == "true"
    model_cache_size: int = int(os.getenv("MODEL_CACHE_SIZE", "10"))
    
    # Visión
    vision_enabled: bool = os.getenv("VISION_ENABLED", "True").lower() == "true"
    camera_resolution: tuple = tuple(
        map(int, os.getenv("CAMERA_RESOLUTION", "640,480").split(","))
    )
    vision_fps: int = int(os.getenv("VISION_FPS", "30"))
    
    # ROS 2
    ros2_enabled: bool = os.getenv("ROS2_ENABLED", "False").lower() == "true"
    ros2_namespace: str = os.getenv("ROS2_NAMESPACE", "/humanoid")
    
    # Timeouts
    connection_timeout: float = float(os.getenv("CONNECTION_TIMEOUT", "10.0"))
    command_timeout: float = float(os.getenv("COMMAND_TIMEOUT", "30.0"))
    movement_timeout: float = float(os.getenv("MOVEMENT_TIMEOUT", "60.0"))
    
    # Seguridad
    max_concurrent_commands: int = int(os.getenv("MAX_CONCURRENT_COMMANDS", "5"))
    rate_limit_enabled: bool = os.getenv("RATE_LIMIT_ENABLED", "False").lower() == "true"
    rate_limit_requests: int = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
    rate_limit_window: int = int(os.getenv("RATE_LIMIT_WINDOW", "60"))
    
    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_file: Optional[str] = os.getenv("LOG_FILE", None)
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        validate_assignment = True
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._validate_settings()
    
    def _validate_settings(self) -> None:
        """
        Valida la configuración después de la inicialización (optimizado).
        
        Raises:
            ValueError: Si la configuración es inválida
        """
        # Validar robot
        if not self.robot_ip or not self.robot_ip.strip():
            raise ValueError("robot_ip cannot be empty")
        if not 1 <= self.robot_port <= 65535:
            raise ValueError("robot_port must be between 1 and 65535")
        if self.dof <= 0:
            raise ValueError("dof must be positive")
        
        # Validar servidor
        if not 1 <= self.port <= 65535:
            raise ValueError("port must be between 1 and 65535")
        
        # Validar movimiento
        if self.default_speed <= 0 or self.default_speed > self.max_speed:
            raise ValueError("default_speed must be positive and <= max_speed")
        if self.max_speed <= 0:
            raise ValueError("max_speed must be positive")
        if self.default_distance <= 0 or self.default_distance > self.max_distance:
            raise ValueError("default_distance must be positive and <= max_distance")
        if self.max_distance <= 0:
            raise ValueError("max_distance must be positive")
        
        # Validar timeouts
        if self.connection_timeout <= 0:
            raise ValueError("connection_timeout must be positive")
        if self.command_timeout <= 0:
            raise ValueError("command_timeout must be positive")
        if self.movement_timeout <= 0:
            raise ValueError("movement_timeout must be positive")
        
        # Validar límites
        if self.max_concurrent_commands <= 0:
            raise ValueError("max_concurrent_commands must be positive")


# Instancia global de settings
settings = Settings()

