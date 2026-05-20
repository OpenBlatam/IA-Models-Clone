"""
Configuration Loader
Load configurations from various sources
"""

from pathlib import Path
from typing import Dict, Any, Optional
import json
import yaml
import logging

logger = logging.getLogger(__name__)


class ConfigLoader:
    """
    Load configurations from files
    """
    
    @staticmethod
    def load_yaml(file_path: Path) -> Dict[str, Any]:
        """
        Load configuration from YAML file
        
        Args:
            file_path: Path to YAML file
            
        Returns:
            Configuration dictionary
        """
        file_path = Path(file_path)
        with open(file_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded config from {file_path}")
        return config
    
    @staticmethod
    def load_json(file_path: Path) -> Dict[str, Any]:
        """
        Load configuration from JSON file
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Configuration dictionary
        """
        file_path = Path(file_path)
        with open(file_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded config from {file_path}")
        return config
    
    @staticmethod
    def save_yaml(config: Dict[str, Any], file_path: Path) -> None:
        """
        Save configuration to YAML file
        
        Args:
            config: Configuration dictionary
            file_path: Path to save file
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        logger.info(f"Saved config to {file_path}")
    
    @staticmethod
    def save_json(config: Dict[str, Any], file_path: Path) -> None:
        """
        Save configuration to JSON file
        
        Args:
            config: Configuration dictionary
            file_path: Path to save file
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Saved config to {file_path}")



