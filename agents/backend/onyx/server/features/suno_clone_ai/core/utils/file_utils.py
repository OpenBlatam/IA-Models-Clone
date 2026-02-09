"""
File Utilities

Utilities for file operations and path management.
"""

import logging
import json
import pickle
from pathlib import Path
from typing import Any, Optional, Dict
import yaml

logger = logging.getLogger(__name__)


class FileManager:
    """Manage file operations."""
    
    @staticmethod
    def ensure_dir(path: str) -> Path:
        """
        Ensure directory exists.
        
        Args:
            path: Directory path
            
        Returns:
            Path object
        """
        path_obj = Path(path)
        path_obj.mkdir(parents=True, exist_ok=True)
        return path_obj
    
    @staticmethod
    def save_json(
        data: Any,
        file_path: str,
        indent: int = 2
    ) -> None:
        """
        Save data to JSON file.
        
        Args:
            data: Data to save
            file_path: File path
            indent: JSON indent
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=indent)
        
        logger.debug(f"Saved JSON to: {file_path}")
    
    @staticmethod
    def load_json(file_path: str) -> Any:
        """
        Load data from JSON file.
        
        Args:
            file_path: File path
            
        Returns:
            Loaded data
        """
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        logger.debug(f"Loaded JSON from: {file_path}")
        return data
    
    @staticmethod
    def save_yaml(
        data: Any,
        file_path: str
    ) -> None:
        """
        Save data to YAML file.
        
        Args:
            data: Data to save
            file_path: File path
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
        
        logger.debug(f"Saved YAML to: {file_path}")
    
    @staticmethod
    def load_yaml(file_path: str) -> Any:
        """
        Load data from YAML file.
        
        Args:
            file_path: File path
            
        Returns:
            Loaded data
        """
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        
        logger.debug(f"Loaded YAML from: {file_path}")
        return data
    
    @staticmethod
    def save_pickle(
        data: Any,
        file_path: str
    ) -> None:
        """
        Save data to pickle file.
        
        Args:
            data: Data to save
            file_path: File path
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.debug(f"Saved pickle to: {file_path}")
    
    @staticmethod
    def load_pickle(file_path: str) -> Any:
        """
        Load data from pickle file.
        
        Args:
            file_path: File path
            
        Returns:
            Loaded data
        """
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        logger.debug(f"Loaded pickle from: {file_path}")
        return data


def ensure_dir(path: str) -> Path:
    """Ensure directory exists."""
    return FileManager.ensure_dir(path)


def save_json(data: Any, file_path: str, **kwargs) -> None:
    """Save data to JSON."""
    FileManager.save_json(data, file_path, **kwargs)


def load_json(file_path: str) -> Any:
    """Load data from JSON."""
    return FileManager.load_json(file_path)



