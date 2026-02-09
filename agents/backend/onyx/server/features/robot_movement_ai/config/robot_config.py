"""
Robot Configuration
===================

Configuración centralizada para el sistema de movimiento robótico.
"""

import os
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from ..core.exceptions import ConfigurationError
from ..core.config_utils import ensure_dir, get_env_int, get_env_bool, get_env


class RobotBrand(Enum):
    """Marcas de robots soportadas."""
    KUKA = "kuka"
    ABB = "abb"
    FANUC = "fanuc"
    UNIVERSAL_ROBOTS = "universal_robots"
    GENERIC = "generic"


@dataclass
class RobotConfig:
    """Configuración del sistema de movimiento robótico."""
    
    # General settings
    robot_brand: RobotBrand = RobotBrand.GENERIC
    ros_enabled: bool = True
    log_level: str = "INFO"
    
    # Performance settings
    feedback_frequency: int = 1000  # Hz
    trajectory_update_rate: int = 100  # Hz
    visual_processing_rate: int = 30  # FPS
    
    # Paths
    storage_path: str = field(default_factory=lambda: str(Path(__file__).parent.parent / "storage"))
    logs_directory: str = field(default_factory=lambda: str(Path(__file__).parent.parent / "logs"))
    models_directory: str = field(default_factory=lambda: str(Path(__file__).parent.parent / "models"))
    
    # ROS settings
    ros_master_uri: str = os.getenv("ROS_MASTER_URI", "http://localhost:11311")
    ros_node_name: str = "robot_movement_ai"
    
    # AI Model settings
    rl_model_path: Optional[str] = None
    cnn_model_path: Optional[str] = None
    ik_model_path: Optional[str] = None
    
    # Robot connection settings
    robot_ip: Optional[str] = get_env("ROBOT_IP", default="192.168.1.100")
    robot_port: int = get_env_int("ROBOT_PORT", default=30001)
    connection_timeout: float = get_env("ROBOT_CONNECTION_TIMEOUT", default=10.0, env_type=float)
    
    # Safety settings
    max_velocity: float = 1.0  # m/s
    max_acceleration: float = 2.0  # m/s²
    max_joint_velocity: float = 1.57  # rad/s
    collision_detection_enabled: bool = True
    emergency_stop_enabled: bool = True
    
    # Trajectory optimization
    trajectory_optimization_enabled: bool = True
    energy_optimization_enabled: bool = True
    vibration_compensation_enabled: bool = True
    
    # Visual processing
    camera_enabled: bool = True
    camera_resolution: tuple = (1920, 1080)
    object_detection_enabled: bool = True
    
    # Chat settings
    chat_enabled: bool = get_env_bool("CHAT_ENABLED", default=True)
    llm_provider: str = get_env("LLM_PROVIDER", default="openai")
    llm_model: str = get_env("LLM_MODEL", default="gpt-4")
    llm_api_key: Optional[str] = get_env("OPENAI_API_KEY") or get_env("ANTHROPIC_API_KEY")
    
    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8010
    api_cors_enabled: bool = True
    api_cors_origins: list = field(default_factory=lambda: ["*"])
    
    def __post_init__(self):
        """Validar y crear directorios necesarios."""
        self._validate()
        self._create_directories()
    
    def _validate(self):
        """Validar configuración."""
        # Validar frecuencia de feedback
        if self.feedback_frequency <= 0 or self.feedback_frequency > 2000:
            raise ConfigurationError(
                f"Feedback frequency must be between 1 and 2000 Hz, got {self.feedback_frequency}"
            )
        
        # Validar límites de seguridad
        if self.max_velocity <= 0:
            raise ConfigurationError("Max velocity must be positive")
        
        if self.max_acceleration <= 0:
            raise ConfigurationError("Max acceleration must be positive")
        
        if self.max_joint_velocity <= 0:
            raise ConfigurationError("Max joint velocity must be positive")
        
        # Validar resolución de cámara
        if len(self.camera_resolution) != 2:
            raise ConfigurationError("Camera resolution must be a tuple of 2 elements")
        
        if any(r <= 0 for r in self.camera_resolution):
            raise ConfigurationError("Camera resolution values must be positive")
        
        # Validar puerto
        if not (1 <= self.api_port <= 65535):
            raise ConfigurationError(f"API port must be between 1 and 65535, got {self.api_port}")
        
        # Validar log level
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level.upper() not in valid_log_levels:
            raise ConfigurationError(
                f"Log level must be one of {valid_log_levels}, got {self.log_level}"
            )
    
    def _create_directories(self):
        """Crear directorios necesarios."""
        try:
            ensure_dir(self.storage_path)
            ensure_dir(self.logs_directory)
            ensure_dir(self.models_directory)
        except ConfigurationError as e:
            # Re-lanzar con contexto adicional
            raise ConfigurationError(
                f"Failed to create required directories: {e.message}",
                error_code=e.error_code,
                details={**e.details, "storage_path": self.storage_path}
            ) from e
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir configuración a diccionario."""
        return {
            "robot_brand": self.robot_brand.value,
            "ros_enabled": self.ros_enabled,
            "log_level": self.log_level,
            "feedback_frequency": self.feedback_frequency,
            "trajectory_update_rate": self.trajectory_update_rate,
            "visual_processing_rate": self.visual_processing_rate,
            "max_velocity": self.max_velocity,
            "max_acceleration": self.max_acceleration,
            "max_joint_velocity": self.max_joint_velocity,
            "collision_detection_enabled": self.collision_detection_enabled,
            "emergency_stop_enabled": self.emergency_stop_enabled,
            "trajectory_optimization_enabled": self.trajectory_optimization_enabled,
            "energy_optimization_enabled": self.energy_optimization_enabled,
            "vibration_compensation_enabled": self.vibration_compensation_enabled,
            "camera_enabled": self.camera_enabled,
            "camera_resolution": self.camera_resolution,
            "object_detection_enabled": self.object_detection_enabled,
            "chat_enabled": self.chat_enabled,
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "api_host": self.api_host,
            "api_port": self.api_port,
            "api_cors_enabled": self.api_cors_enabled,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RobotConfig":
        """
        Crear configuración desde diccionario.
        
        Args:
            data: Diccionario con configuración
        
        Returns:
            Instancia de RobotConfig
        """
        if "robot_brand" in data:
            data["robot_brand"] = RobotBrand(data["robot_brand"])
        return cls(**data)
    
    def update(self, **kwargs) -> "RobotConfig":
        """
        Actualizar configuración con nuevos valores.
        
        Args:
            **kwargs: Valores a actualizar
        
        Returns:
            Nueva instancia con valores actualizados
        """
        current_dict = self.to_dict()
        current_dict.update(kwargs)
        return self.from_dict(current_dict)

