"""
Serialization Utilities
Advanced serialization and deserialization
"""

import torch
import pickle
import json
from pathlib import Path
from typing import Any, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class Serializer:
    """
    Advanced serialization utilities
    """
    
    @staticmethod
    def save_pickle(obj: Any, filepath: Path) -> None:
        """
        Save object as pickle
        
        Args:
            obj: Object to save
            filepath: Path to save file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f)
        logger.info(f"Saved pickle: {filepath}")
    
    @staticmethod
    def load_pickle(filepath: Path) -> Any:
        """
        Load object from pickle
        
        Args:
            filepath: Path to pickle file
            
        Returns:
            Loaded object
        """
        filepath = Path(filepath)
        with open(filepath, 'rb') as f:
            obj = pickle.load(f)
        logger.info(f"Loaded pickle: {filepath}")
        return obj
    
    @staticmethod
    def save_json(obj: Any, filepath: Path, indent: int = 2) -> None:
        """
        Save object as JSON
        
        Args:
            obj: Object to save (must be JSON serializable)
            filepath: Path to save file
            indent: JSON indentation
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(obj, f, indent=indent)
        logger.info(f"Saved JSON: {filepath}")
    
    @staticmethod
    def load_json(filepath: Path) -> Any:
        """
        Load object from JSON
        
        Args:
            filepath: Path to JSON file
            
        Returns:
            Loaded object
        """
        filepath = Path(filepath)
        with open(filepath, 'r') as f:
            obj = json.load(f)
        logger.info(f"Loaded JSON: {filepath}")
        return obj
    
    @staticmethod
    def save_state_dict(
        model: torch.nn.Module,
        filepath: Path,
        metadata: Optional[Dict] = None,
    ) -> None:
        """
        Save model state dict with metadata
        
        Args:
            model: Model to save
            filepath: Path to save file
            metadata: Additional metadata (optional)
        """
        state = {
            'state_dict': model.state_dict(),
            'metadata': metadata or {},
        }
        torch.save(state, filepath)
        logger.info(f"Saved state dict: {filepath}")



