"""
Advanced Serialization
======================

Advanced serialization utilities with support for multiple formats.
"""

import json
import pickle
import yaml
import logging
from typing import Any, Dict, Optional, Union
from pathlib import Path
from datetime import datetime
import base64

logger = logging.getLogger(__name__)


class Serializer:
    """
    Advanced serializer with support for:
    - JSON
    - YAML
    - Pickle
    - Custom formats
    """
    
    @staticmethod
    def serialize_json(
        data: Any,
        indent: int = 2,
        ensure_ascii: bool = False
    ) -> str:
        """
        Serialize to JSON.
        
        Args:
            data: Data to serialize
            indent: Indentation level
            ensure_ascii: Ensure ASCII encoding
        
        Returns:
            JSON string
        
        Raises:
            ValueError: If serialization fails
        """
        try:
            return json.dumps(
                data,
                indent=indent,
                ensure_ascii=ensure_ascii,
                default=Serializer._json_default
            )
        except (TypeError, ValueError) as e:
            raise ValueError(f"JSON serialization failed: {str(e)}")
    
    @staticmethod
    def deserialize_json(json_str: str) -> Any:
        """
        Deserialize from JSON.
        
        Args:
            json_str: JSON string
        
        Returns:
            Deserialized data
        
        Raises:
            ValueError: If deserialization fails
        """
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON deserialization failed: {str(e)}")
    
    @staticmethod
    def serialize_yaml(data: Any, default_flow_style: bool = False) -> str:
        """
        Serialize to YAML.
        
        Args:
            data: Data to serialize
            default_flow_style: Use flow style
        
        Returns:
            YAML string
        
        Raises:
            ValueError: If serialization fails
        """
        try:
            return yaml.dump(
                data,
                default_flow_style=default_flow_style,
                allow_unicode=True
            )
        except Exception as e:
            raise ValueError(f"YAML serialization failed: {str(e)}")
    
    @staticmethod
    def deserialize_yaml(yaml_str: str) -> Any:
        """
        Deserialize from YAML.
        
        Args:
            yaml_str: YAML string
        
        Returns:
            Deserialized data
        
        Raises:
            ValueError: If deserialization fails
        """
        try:
            return yaml.safe_load(yaml_str)
        except Exception as e:
            raise ValueError(f"YAML deserialization failed: {str(e)}")
    
    @staticmethod
    def serialize_pickle(data: Any) -> bytes:
        """
        Serialize to pickle.
        
        Args:
            data: Data to serialize
        
        Returns:
            Pickle bytes
        
        Raises:
            ValueError: If serialization fails
        """
        try:
            return pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            raise ValueError(f"Pickle serialization failed: {str(e)}")
    
    @staticmethod
    def deserialize_pickle(pickle_bytes: bytes) -> Any:
        """
        Deserialize from pickle.
        
        Args:
            pickle_bytes: Pickle bytes
        
        Returns:
            Deserialized data
        
        Raises:
            ValueError: If deserialization fails
        """
        try:
            return pickle.loads(pickle_bytes)
        except Exception as e:
            raise ValueError(f"Pickle deserialization failed: {str(e)}")
    
    @staticmethod
    def save_to_file(
        data: Any,
        filepath: Union[str, Path],
        format: str = "json"
    ) -> None:
        """
        Save data to file.
        
        Args:
            data: Data to save
            filepath: Path to file
            format: Format ("json", "yaml", "pickle")
        
        Raises:
            ValueError: If format is not supported
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "json":
            content = Serializer.serialize_json(data)
            filepath.write_text(content, encoding="utf-8")
        elif format == "yaml":
            content = Serializer.serialize_yaml(data)
            filepath.write_text(content, encoding="utf-8")
        elif format == "pickle":
            content = Serializer.serialize_pickle(data)
            filepath.write_bytes(content)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Data saved to {filepath} ({format})")
    
    @staticmethod
    def load_from_file(
        filepath: Union[str, Path],
        format: Optional[str] = None
    ) -> Any:
        """
        Load data from file.
        
        Args:
            filepath: Path to file
            format: Format (None = auto-detect from extension)
        
        Returns:
            Loaded data
        
        Raises:
            ValueError: If format is not supported
            FileNotFoundError: If file doesn't exist
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # Auto-detect format from extension
        if format is None:
            ext = filepath.suffix.lower()
            format_map = {
                ".json": "json",
                ".yaml": "yaml",
                ".yml": "yaml",
                ".pkl": "pickle",
                ".pickle": "pickle"
            }
            format = format_map.get(ext, "json")
        
        if format == "json":
            content = filepath.read_text(encoding="utf-8")
            return Serializer.deserialize_json(content)
        elif format == "yaml":
            content = filepath.read_text(encoding="utf-8")
            return Serializer.deserialize_yaml(content)
        elif format == "pickle":
            content = filepath.read_bytes()
            return Serializer.deserialize_pickle(content)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    @staticmethod
    def _json_default(obj: Any) -> Any:
        """Default JSON encoder for special types."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, bytes):
            return base64.b64encode(obj).decode("utf-8")
        elif hasattr(obj, "__dict__"):
            return obj.__dict__
        else:
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
