"""
Config Manager - Gestión de configuraciones YAML/JSON.

Sigue principios de configuración centralizada.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import yaml
import json

from config.logging_config import get_logger

logger = get_logger(__name__)


class ConfigManager:
    """
    Gestor de configuraciones para LLM Service.
    
    Soporta YAML y JSON.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Inicializar ConfigManager.
        
        Args:
            config_path: Ruta al directorio de configuraciones
        """
        self.config_path = Path(config_path) if config_path else Path("./config/llm")
        self.config_path.mkdir(parents=True, exist_ok=True)
        
        self.configs: Dict[str, Dict[str, Any]] = {}
    
    def load_config(
        self,
        name: str,
        filepath: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Cargar configuración desde archivo.
        
        Args:
            name: Nombre de la configuración
            filepath: Ruta al archivo (opcional)
            
        Returns:
            Diccionario con configuración
        """
        if filepath is None:
            # Buscar en config_path
            yaml_path = self.config_path / f"{name}.yaml"
            json_path = self.config_path / f"{name}.json"
            
            if yaml_path.exists():
                filepath = str(yaml_path)
            elif json_path.exists():
                filepath = str(json_path)
            else:
                raise FileNotFoundError(f"Configuración '{name}' no encontrada")
        
        path = Path(filepath)
        
        if path.suffix == ".yaml" or path.suffix == ".yml":
            with open(path, 'r') as f:
                config = yaml.safe_load(f)
        elif path.suffix == ".json":
            with open(path, 'r') as f:
                config = json.load(f)
        else:
            raise ValueError(f"Formato no soportado: {path.suffix}")
        
        self.configs[name] = config
        logger.info(f"Configuración '{name}' cargada desde {filepath}")
        
        return config
    
    def save_config(
        self,
        name: str,
        config: Dict[str, Any],
        format: str = "yaml"
    ) -> None:
        """
        Guardar configuración a archivo.
        
        Args:
            name: Nombre de la configuración
            config: Diccionario con configuración
            format: Formato (yaml o json)
        """
        if format == "yaml":
            filepath = self.config_path / f"{name}.yaml"
            with open(filepath, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        elif format == "json":
            filepath = self.config_path / f"{name}.json"
            with open(filepath, 'w') as f:
                json.dump(config, f, indent=2)
        else:
            raise ValueError(f"Formato no soportado: {format}")
        
        self.configs[name] = config
        logger.info(f"Configuración '{name}' guardada en {filepath}")
    
    def get_config(self, name: str) -> Optional[Dict[str, Any]]:
        """Obtener configuración cargada."""
        return self.configs.get(name)
    
    def create_default_configs(self) -> None:
        """Crear configuraciones por defecto."""
        
        # Configuración de modelos
        models_config = {
            "default_models": [
                "openai/gpt-4o-mini",
                "anthropic/claude-3.5-sonnet",
                "google/gemini-pro-1.5"
            ],
            "model_settings": {
                "openai/gpt-4o-mini": {
                    "temperature": 0.7,
                    "max_tokens": 2000,
                    "top_p": 1.0
                },
                "anthropic/claude-3.5-sonnet": {
                    "temperature": 0.7,
                    "max_tokens": 4000,
                    "top_p": 0.9
                }
            }
        }
        
        # Configuración de prompts
        prompts_config = {
            "system_prompts": {
                "code_analysis": "Eres un experto analista de código...",
                "documentation": "Eres un experto en documentación...",
                "refactoring": "Eres un experto en refactorización..."
            },
            "default_temperature": 0.7,
            "default_max_tokens": 2000
        }
        
        # Configuración de optimización
        optimization_config = {
            "cache": {
                "enabled": True,
                "ttl": 3600,
                "max_size": 1000
            },
            "retry": {
                "max_attempts": 3,
                "min_wait": 1.0,
                "max_wait": 10.0
            },
            "rate_limiting": {
                "enabled": True,
                "requests_per_minute": 50
            }
        }
        
        self.save_config("models", models_config)
        self.save_config("prompts", prompts_config)
        self.save_config("optimization", optimization_config)
        
        logger.info("Configuraciones por defecto creadas")


# Instancia global
_config_manager = ConfigManager()


def get_config_manager() -> ConfigManager:
    """Obtener instancia global del config manager."""
    return _config_manager



