"""
Config Manager
==============
Gestor centralizado de configuración
"""

import json
import os
from typing import Dict, Any, Optional
from pathlib import Path
from ..base.base_manager import BaseManager


class ConfigManager(BaseManager):
    """
    Gestor centralizado de configuración
    """
    
    def __init__(self, name: str = "ConfigManager"):
        super().__init__(name)
        self.config: Dict[str, Any] = {}
        self.config_file: Optional[Path] = None
        self.defaults: Dict[str, Any] = {}
    
    def _initialize(self):
        """Inicializar config manager"""
        self._load_defaults()
        self._load_from_file()
        self._load_from_environment()
    
    def _shutdown(self):
        """Cerrar config manager"""
        self._save_to_file()
    
    def _load_defaults(self):
        """Cargar valores por defecto"""
        self.defaults = {
            'model': {
                'device': 'cuda',
                'dtype': 'float16',
                'max_batch_size': 4
            },
            'processing': {
                'max_resolution': 1024,
                'quality_threshold': 0.7
            },
            'api': {
                'timeout': 30,
                'retry_count': 3
            },
            'cache': {
                'enabled': True,
                'ttl': 3600
            }
        }
        self.config = self.defaults.copy()
    
    def _load_from_file(self, config_file: Optional[str] = None):
        """Cargar configuración desde archivo"""
        if config_file:
            self.config_file = Path(config_file)
        elif not self.config_file:
            # Buscar archivo de configuración
            possible_paths = [
                Path('config.json'),
                Path('config/config.json'),
                Path.home() / '.character_clothing_changer' / 'config.json'
            ]
            
            for path in possible_paths:
                if path.exists():
                    self.config_file = path
                    break
        
        if self.config_file and self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    file_config = json.load(f)
                    self._merge_config(file_config)
            except Exception as e:
                self._log_error(f"Error loading config file: {e}")
    
    def _load_from_environment(self):
        """Cargar configuración desde variables de entorno"""
        env_mappings = {
            'MODEL_DEVICE': ('model', 'device'),
            'MODEL_DTYPE': ('model', 'dtype'),
            'API_TIMEOUT': ('api', 'timeout'),
            'CACHE_ENABLED': ('cache', 'enabled'),
        }
        
        for env_var, (section, key) in env_mappings.items():
            value = os.getenv(env_var)
            if value:
                if section not in self.config:
                    self.config[section] = {}
                
                # Convertir tipos
                if value.lower() in ('true', 'false'):
                    value = value.lower() == 'true'
                elif value.isdigit():
                    value = int(value)
                elif '.' in value and value.replace('.', '').isdigit():
                    value = float(value)
                
                self.config[section][key] = value
    
    def _merge_config(self, new_config: Dict[str, Any]):
        """Fusionar configuración nueva con existente"""
        def merge_dict(base: Dict, new: Dict):
            for key, value in new.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    merge_dict(base[key], value)
                else:
                    base[key] = value
        
        merge_dict(self.config, new_config)
    
    def _save_to_file(self):
        """Guardar configuración en archivo"""
        if self.config_file:
            try:
                self.config_file.parent.mkdir(parents=True, exist_ok=True)
                with open(self.config_file, 'w') as f:
                    json.dump(self.config, f, indent=2)
            except Exception as e:
                self._log_error(f"Error saving config file: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Obtener valor de configuración
        
        Args:
            key: Clave en formato 'section.key' o 'section.subsection.key'
            default: Valor por defecto si no existe
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """
        Establecer valor de configuración
        
        Args:
            key: Clave en formato 'section.key' o 'section.subsection.key'
            value: Valor a establecer
        """
        keys = key.split('.')
        config = self.config
        
        # Crear estructura anidada si no existe
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Obtener sección completa de configuración"""
        return self.config.get(section, {}).copy()
    
    def set_section(self, section: str, values: Dict[str, Any]):
        """Establecer sección completa de configuración"""
        self.config[section] = values.copy()
    
    def reset_to_defaults(self):
        """Resetear a valores por defecto"""
        self.config = self.defaults.copy()
    
    def reload(self):
        """Recargar configuración"""
        self._load_from_file()
        self._load_from_environment()


# Instancia global
config_manager = ConfigManager()

