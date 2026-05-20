"""
NaN/Inf Detector
Specialized detection and handling of NaN/Inf values
"""

from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    import numpy as np
    TORCH_AVAILABLE = True
    NUMPY_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    NUMPY_AVAILABLE = False
    logger.warning("PyTorch or NumPy not available")


class NaNDetector:
    """Detect and handle NaN/Inf values"""
    
    def __init__(self, auto_fix: bool = True):
        self.auto_fix = auto_fix
        self.detections: List[Dict[str, Any]] = []
    
    def detect(
        self,
        tensor: Any,
        name: str = "tensor",
        fix: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Detect NaN/Inf values
        
        Args:
            tensor: Tensor to check
            name: Tensor name
            fix: Whether to fix (overrides auto_fix)
        
        Returns:
            Detection results
        """
        fix = fix if fix is not None else self.auto_fix
        
        detection = {
            "name": name,
            "has_nan": False,
            "has_inf": False,
            "nan_count": 0,
            "inf_count": 0,
            "fixed": False
        }
        
        if isinstance(tensor, torch.Tensor) and TORCH_AVAILABLE:
            detection["has_nan"] = torch.isnan(tensor).any().item()
            detection["has_inf"] = torch.isinf(tensor).any().item()
            
            if detection["has_nan"]:
                detection["nan_count"] = torch.isnan(tensor).sum().item()
                logger.warning(f"NaN detected in {name}: {detection['nan_count']} values")
            
            if detection["has_inf"]:
                detection["inf_count"] = torch.isinf(tensor).sum().item()
                logger.warning(f"Inf detected in {name}: {detection['inf_count']} values")
            
            if fix and (detection["has_nan"] or detection["has_inf"]):
                tensor = torch.nan_to_num(
                    tensor,
                    nan=0.0,
                    posinf=1.0,
                    neginf=-1.0
                )
                detection["fixed"] = True
        
        elif isinstance(tensor, np.ndarray) and NUMPY_AVAILABLE:
            detection["has_nan"] = np.isnan(tensor).any()
            detection["has_inf"] = np.isinf(tensor).any()
            
            if detection["has_nan"]:
                detection["nan_count"] = np.isnan(tensor).sum()
                logger.warning(f"NaN detected in {name}: {detection['nan_count']} values")
            
            if detection["has_inf"]:
                detection["inf_count"] = np.isinf(tensor).sum()
                logger.warning(f"Inf detected in {name}: {detection['inf_count']} values")
            
            if fix and (detection["has_nan"] or detection["has_inf"]):
                tensor = np.nan_to_num(
                    tensor,
                    nan=0.0,
                    posinf=1.0,
                    neginf=-1.0
                )
                detection["fixed"] = True
        
        self.detections.append(detection)
        return detection
    
    def detect_in_model(
        self,
        model: torch.nn.Module,
        fix: Optional[bool] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Detect NaN/Inf in all model parameters
        
        Args:
            model: Model to check
            fix: Whether to fix
        
        Returns:
            Dictionary of detections per parameter
        """
        fix = fix if fix is not None else self.auto_fix
        detections = {}
        
        for name, param in model.named_parameters():
            detection = self.detect(param, name=f"param_{name}", fix=fix)
            detections[name] = detection
        
        return detections
    
    def get_detections(self) -> List[Dict[str, Any]]:
        """Get all detections"""
        return self.detections.copy()
    
    def clear_detections(self):
        """Clear detection history"""
        self.detections.clear()



