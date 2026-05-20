"""
Config Manager - Gestor de configuración
"""

import logging
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class ConfigManager:
    """Gestor de configuración del sistema"""

    def __init__(self, config_path: Optional[str] = None):
        """
        Inicializar el gestor de configuración.

        Args:
            config_path: Ruta al archivo de configuración
        """
        self.config_path = config_path or self._get_default_config_path()
        self.config: Dict[str, Any] = {}

    def _get_default_config(self) -> Dict[str, Any]:
        """Obtener configuración por defecto"""
        return {
            "app": {
                "name": "Addition Removal AI",
                "version": "1.0.0"
            },
            "editor": {
                "max_content_length": 1000000,
                "enable_validation": True,
                "enable_history": True
            },
            "ai": {
                "model": "gpt-3.5-turbo",
                "temperature": 0.7,
                "max_tokens": 2000
            },
            "api": {
                "host": "0.0.0.0",
                "port": 8010,
                "cors_origins": ["*"]
            }
        }

    def _get_default_config_path(self) -> Path:
        """Obtener ruta por defecto del archivo de configuración"""
        base_dir = Path(__file__).parent.parent
        return base_dir / "config" / "config.yaml"

    def get_config(self) -> Dict[str, Any]:
        """
        Obtener la configuración.

        Returns:
            Diccionario con la configuración
        """
        if not self.config:
            self.load_config()
        return self.config

    def load_config(self):
        """Cargar configuración desde archivo"""
        default_config = self._get_default_config()
        
        if self.config_path and Path(self.config_path).exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    file_config = yaml.safe_load(f) or {}
                # Combinar con configuración por defecto
                self.config = self._merge_config(default_config, file_config)
                logger.info(f"Configuración cargada desde {self.config_path}")
            except Exception as e:
                logger.warning(f"Error al cargar configuración: {e}. Usando valores por defecto.")
                self.config = default_config
        else:
            logger.info("No se encontró archivo de configuración. Usando valores por defecto.")
            self.config = default_config

    def _merge_config(self, default: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Combinar configuraciones recursivamente"""
        result = default.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_config(result[key], value)
            else:
                result[key] = value
        return result

    def save_config(self, config: Optional[Dict[str, Any]] = None):
        """
        Guardar configuración en archivo.

        Args:
            config: Configuración a guardar (opcional, usa la actual si no se especifica)
        """
        if config:
            self.config = config
        
        if self.config_path:
            try:
                Path(self.config_path).parent.mkdir(parents=True, exist_ok=True)
                with open(self.config_path, 'w', encoding='utf-8') as f:
                    yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
                logger.info(f"Configuración guardada en {self.config_path}")
            except Exception as e:
                logger.error(f"Error al guardar configuración: {e}")






