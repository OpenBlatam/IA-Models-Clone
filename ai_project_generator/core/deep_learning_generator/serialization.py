"""
Serialization Module

Import/export configurations to/from various formats.
"""

from typing import Dict, Any, Optional, Union
from pathlib import Path
import json
import yaml
import logging

logger = logging.getLogger(__name__)


class ConfigSerializer:
    """
    Serialize and deserialize generator configurations.
    """
    
    @staticmethod
    def to_dict(config: Dict[str, Any]) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return config.copy()
    
    @staticmethod
    def to_json(config: Dict[str, Any], indent: int = 2) -> str:
        """Serialize configuration to JSON string."""
        return json.dumps(config, indent=indent, default=str)
    
    @staticmethod
    def to_yaml(config: Dict[str, Any]) -> str:
        """Serialize configuration to YAML string."""
        try:
            return yaml.dump(config, default_flow_style=False, sort_keys=False)
        except ImportError:
            logger.warning("PyYAML not available, falling back to JSON")
            return ConfigSerializer.to_json(config)
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> Dict[str, Any]:
        """Deserialize from dictionary."""
        return data.copy()
    
    @staticmethod
    def from_json(json_str: str) -> Dict[str, Any]:
        """Deserialize from JSON string."""
        return json.loads(json_str)
    
    @staticmethod
    def from_yaml(yaml_str: str) -> Dict[str, Any]:
        """Deserialize from YAML string."""
        try:
            return yaml.safe_load(yaml_str)
        except ImportError:
            logger.warning("PyYAML not available, trying JSON")
            return ConfigSerializer.from_json(yaml_str)
    
    @staticmethod
    def save_to_file(
        config: Dict[str, Any],
        filepath: Union[str, Path],
        format: str = "json"
    ) -> None:
        """
        Save configuration to file.
        
        Args:
            config: Configuration to save
            filepath: Path to save file
            format: Format (json, yaml)
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == "json":
            content = ConfigSerializer.to_json(config)
        elif format.lower() == "yaml":
            content = ConfigSerializer.to_yaml(config)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        filepath.write_text(content, encoding="utf-8")
        logger.info(f"Configuration saved to {filepath}")
    
    @staticmethod
    def load_from_file(
        filepath: Union[str, Path]
    ) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Args:
            filepath: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        content = filepath.read_text(encoding="utf-8")
        
        if filepath.suffix.lower() in [".yaml", ".yml"]:
            return ConfigSerializer.from_yaml(content)
        elif filepath.suffix.lower() == ".json":
            return ConfigSerializer.from_json(content)
        else:
            # Try to detect format from content
            try:
                return ConfigSerializer.from_json(content)
            except json.JSONDecodeError:
                return ConfigSerializer.from_yaml(content)


def save_config(
    config: Dict[str, Any],
    filepath: Union[str, Path],
    format: str = "json"
) -> None:
    """Save configuration to file."""
    ConfigSerializer.save_to_file(config, filepath, format)


def load_config(filepath: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from file."""
    return ConfigSerializer.load_from_file(filepath)















