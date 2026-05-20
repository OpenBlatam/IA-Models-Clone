"""
Config Serializer
Serialize and deserialize configurations
"""

from typing import Dict, Any, Optional
import logging
import json
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)


class ConfigSerializer:
    """Serialize and deserialize configurations"""
    
    @staticmethod
    def save_json(config: Dict[str, Any], path: str):
        """Save config as JSON"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Config saved to {path}")
    
    @staticmethod
    def load_json(path: str) -> Dict[str, Any]:
        """Load config from JSON"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(path, 'r') as f:
            config = json.load(f)
        
        logger.info(f"Config loaded from {path}")
        return config
    
    @staticmethod
    def save_yaml(config: Dict[str, Any], path: str):
        """Save config as YAML"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        logger.info(f"Config saved to {path}")
    
    @staticmethod
    def load_yaml(path: str) -> Dict[str, Any]:
        """Load config from YAML"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Config loaded from {path}")
        return config



