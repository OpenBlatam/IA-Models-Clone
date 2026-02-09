"""
Optimizer Serializer
===================

Handles serialization and deserialization of optimizer configurations.
"""

import json
import pickle
import logging
from pathlib import Path
from typing import Dict, Any, Union

logger = logging.getLogger(__name__)


class OptimizerSerializer:
    """
    Handles serialization and deserialization of optimizer configurations.
    
    Responsibilities:
    - Serialize optimizer configs to dictionaries
    - Deserialize optimizer configs from dictionaries
    - Save/load optimizer configs to/from files
    - Validate serializable data
    """
    
    @staticmethod
    def is_serializable(obj: Any) -> bool:
        """
        Check if an object is JSON serializable.
        
        Args:
            obj: Object to check
        
        Returns:
            True if serializable, False otherwise
        """
        try:
            json.dumps(obj)
            return True
        except (TypeError, ValueError):
            return False
    
    @staticmethod
    def serialize(
        optimizer_type: str,
        learning_rate: float,
        kwargs: Dict[str, Any],
        use_core: bool,
        version: str = '1.0'
    ) -> Dict[str, Any]:
        """
        Serialize optimizer configuration to dictionary.
        
        Args:
            optimizer_type: Type of optimizer
            learning_rate: Learning rate
            kwargs: Optimizer parameters
            use_core: Whether to use optimization_core
            version: Version string
        
        Returns:
            Dictionary with serialized configuration
        """
        return {
            'optimizer_type': optimizer_type,
            'learning_rate': learning_rate,
            'kwargs': {
                k: v for k, v in kwargs.items()
                if OptimizerSerializer.is_serializable(v)
            },
            'use_core': use_core,
            'version': version
        }
    
    @staticmethod
    def save(config: Dict[str, Any], filepath: Union[str, Path]) -> None:
        """
        Save optimizer configuration to file.
        
        Args:
            config: Configuration dictionary
            filepath: Path to save file
        
        Raises:
            IOError: If file cannot be written
        """
        path = Path(filepath)
        
        if path.suffix == '.json':
            with open(path, 'w') as f:
                json.dump(config, f, indent=2)
        else:
            with open(path, 'wb') as f:
                pickle.dump(config, f)
        
        logger.info(f"✅ Saved optimizer config to {path}")
    
    @staticmethod
    def load(filepath: Union[str, Path]) -> Dict[str, Any]:
        """
        Load optimizer configuration from file.
        
        Args:
            filepath: Path to load file from
        
        Returns:
            Configuration dictionary
        
        Raises:
            IOError: If file cannot be read
            ValueError: If file format is invalid
        """
        path = Path(filepath)
        
        if path.suffix == '.json':
            with open(path, 'r') as f:
                config = json.load(f)
        else:
            with open(path, 'rb') as f:
                config = pickle.load(f)
        
        logger.info(f"✅ Loaded optimizer config from {path}")
        return config

