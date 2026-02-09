"""YAML Loader - Carga de archivos YAML"""
from typing import Dict, Any
from pathlib import Path


class YAMLLoader:
    """Cargador de archivos YAML"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
    
    def load(self) -> Dict[str, Any]:
        """Carga configuración desde YAML"""
        if not self.config_path.exists():
            return {}
        
        try:
            import yaml
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f) or {}
        except Exception:
            return {}

