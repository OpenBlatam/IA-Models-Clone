"""
Config Loader for Humanoid Devin Robot (Optimizado)
====================================================

Cargador de configuración con validación y valores por defecto.
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)


def ErrorCode(description: str):
    """Decorador para anotar excepciones con códigos de error y descripciones."""
    def decorator(cls):
        cls._error_description = description
        return cls
    return decorator


@ErrorCode(description="Error in configuration loading")
class ConfigError(Exception):
    """Excepción para errores de configuración."""
    
    def __init__(self):
        """Initialize exception with description from @ErrorCode decorator."""
        message = getattr(self.__class__, '_error_description', "Error in configuration loading")
        super().__init__(message)
        self.message = message


class ConfigLoader:
    """
    Cargador de configuración para el robot humanoide.
    
    Soporta archivos YAML y JSON con validación y valores por defecto.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Inicializar cargador de configuración.
        
        Args:
            config_path: Ruta al archivo de configuración (opcional)
        """
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        
        if config_path:
            self.load_config(config_path)
        else:
            self.load_defaults()
    
    def load_config(self, config_path: str) -> None:
        """
        Cargar configuración desde archivo.
        
        Args:
            config_path: Ruta al archivo de configuración
            
        Raises:
            ConfigError: Si hay error al cargar la configuración
        """
        config_file = Path(config_path)
        
        if not config_file.exists():
            raise ConfigError(f"Config file not found: {config_path}")
        
        if not config_file.is_file():
            raise ConfigError(f"Config path is not a file: {config_path}")
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                if config_file.suffix in ['.yaml', '.yml']:
                    self.config = yaml.safe_load(f) or {}
                elif config_file.suffix == '.json':
                    import json
                    self.config = json.load(f)
                else:
                    raise ConfigError(f"Unsupported config file format: {config_file.suffix}")
            
            logger.info(f"Config loaded from {config_path}")
            self._validate_config()
            
        except yaml.YAMLError as e:
            raise ConfigError(f"Error parsing YAML config: {e}") from e
        except Exception as e:
            raise ConfigError(f"Error loading config: {e}") from e
    
    def load_defaults(self) -> None:
        """Cargar configuración por defecto."""
        # Configuración por defecto básica
        self.config = {
            "robot": {
                "ip": "192.168.1.100",
                "port": 30001,
                "dof": 32,
                "robot_type": "generic"
            },
            "integrations": {
                "use_ml": True,
                "use_diffusion": True,
                "use_ros2": True,
                "use_moveit2": True,
                "use_opencv": True,
                "use_ai": True,
                "use_pcl": True,
                "use_nav2": True
            },
            "control": {
                "default_speed": 0.5,
                "max_speed": 1.0,
                "min_speed": 0.1
            },
            "logging": {
                "level": "INFO"
            }
        }
        
        logger.info("Default config loaded")
    
    def _validate_config(self) -> None:
        """
        Validar configuración cargada.
        
        Raises:
            ConfigError: Si la configuración es inválida
        """
        # Validar sección robot
        if "robot" not in self.config:
            raise ConfigError("Config must contain 'robot' section")
        
        robot_config = self.config["robot"]
        
        if "ip" not in robot_config:
            raise ConfigError("Robot config must contain 'ip'")
        
        if "port" not in robot_config:
            raise ConfigError("Robot config must contain 'port'")
        
        if not isinstance(robot_config["port"], int) or not (1 <= robot_config["port"] <= 65535):
            raise ConfigError(f"Robot port must be between 1 and 65535, got {robot_config['port']}")
        
        if "dof" not in robot_config:
            raise ConfigError("Robot config must contain 'dof'")
        
        if not isinstance(robot_config["dof"], int) or not (1 <= robot_config["dof"] <= 100):
            raise ConfigError(f"Robot dof must be between 1 and 100, got {robot_config['dof']}")
        
        logger.debug("Config validation passed")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Obtener valor de configuración usando notación de puntos.
        
        Args:
            key: Clave en notación de puntos (ej: "robot.ip")
            default: Valor por defecto si no se encuentra
            
        Returns:
            Valor de configuración o default
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_robot_config(self) -> Dict[str, Any]:
        """
        Obtener configuración del robot.
        
        Returns:
            Dict con configuración del robot
        """
        return self.config.get("robot", {})
    
    def get_integrations_config(self) -> Dict[str, Any]:
        """
        Obtener configuración de integraciones.
        
        Returns:
            Dict con configuración de integraciones
        """
        return self.config.get("integrations", {})
    
    def get_control_config(self) -> Dict[str, Any]:
        """
        Obtener configuración de control.
        
        Returns:
            Dict con configuración de control
        """
        return self.config.get("control", {})
    
    def get_deep_learning_config(self) -> Dict[str, Any]:
        """
        Obtener configuración de deep learning.
        
        Returns:
            Dict con configuración de deep learning
        """
        return self.config.get("deep_learning", {})
    
    def create_driver_config(self) -> Dict[str, Any]:
        """
        Crear diccionario de configuración para el driver.
        
        Returns:
            Dict con parámetros para HumanoidDevinDriver
        """
        robot_config = self.get_robot_config()
        integrations = self.get_integrations_config()
        
        return {
            "robot_ip": robot_config.get("ip", "192.168.1.100"),
            "robot_port": robot_config.get("port", 30001),
            "dof": robot_config.get("dof", 32),
            "robot_type": robot_config.get("robot_type", "generic"),
            "use_ml": integrations.get("use_ml", True),
            "use_diffusion": integrations.get("use_diffusion", True),
            "use_ros2": integrations.get("use_ros2", True),
            "use_moveit2": integrations.get("use_moveit2", True),
            "use_opencv": integrations.get("use_opencv", True),
            "use_ai": integrations.get("use_ai", True),
            "use_pcl": integrations.get("use_pcl", True),
            "use_nav2": integrations.get("use_nav2", True)
        }


def load_config(config_path: Optional[str] = None) -> ConfigLoader:
    """
    Función helper para cargar configuración.
    
    Args:
        config_path: Ruta al archivo de configuración (opcional)
        
    Returns:
        ConfigLoader con configuración cargada
    """
    return ConfigLoader(config_path)

