"""
Model Exporter
Export models to different formats
"""

from typing import Dict, Any, Optional, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")


class ModelExporter:
    """Export models"""
    
    @staticmethod
    def export_pytorch(
        model: nn.Module,
        path: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Export PyTorch model"""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        export_dict = {
            "model_state_dict": model.state_dict(),
            "metadata": metadata or {}
        }
        
        torch.save(export_dict, path)
        logger.info(f"Model exported to {path}")
    
    @staticmethod
    def export_state_dict(
        model: nn.Module,
        path: str
    ):
        """Export only state dict"""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save(model.state_dict(), path)
        logger.info(f"State dict exported to {path}")



