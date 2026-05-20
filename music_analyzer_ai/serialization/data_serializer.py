"""
Data Serializer
Serialize and deserialize data
"""

from typing import Any, Optional
import logging
import pickle
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")


class DataSerializer:
    """Serialize and deserialize data"""
    
    @staticmethod
    def save_pickle(data: Any, path: str):
        """Save data as pickle"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Data saved to {path}")
    
    @staticmethod
    def load_pickle(path: str) -> Any:
        """Load data from pickle"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        logger.info(f"Data loaded from {path}")
        return data
    
    @staticmethod
    def save_numpy(data: np.ndarray, path: str):
        """Save numpy array"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        np.save(path, data)
        logger.info(f"NumPy array saved to {path}")
    
    @staticmethod
    def load_numpy(path: str) -> np.ndarray:
        """Load numpy array"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"NumPy file not found: {path}")
        
        data = np.load(path)
        logger.info(f"NumPy array loaded from {path}")
        return data
    
    @staticmethod
    def save_tensor(data: torch.Tensor, path: str):
        """Save PyTorch tensor"""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save(data, path)
        logger.info(f"Tensor saved to {path}")
    
    @staticmethod
    def load_tensor(path: str, map_location: Optional[str] = None) -> torch.Tensor:
        """Load PyTorch tensor"""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Tensor file not found: {path}")
        
        data = torch.load(path, map_location=map_location)
        logger.info(f"Tensor loaded from {path}")
        return data



