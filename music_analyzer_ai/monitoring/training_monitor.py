"""
Training Monitor
Monitor training progress and metrics
"""

from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")


class TrainingMonitor:
    """Monitor training progress"""
    
    def __init__(self):
        self.epoch_metrics: List[Dict[str, Any]] = []
        self.batch_metrics: List[Dict[str, Any]] = []
        self.current_epoch = 0
        self.current_batch = 0
    
    def record_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]] = None
    ):
        """Record epoch metrics"""
        self.current_epoch = epoch
        epoch_data = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics or {}
        }
        self.epoch_metrics.append(epoch_data)
    
    def record_batch(
        self,
        batch_idx: int,
        metrics: Dict[str, float]
    ):
        """Record batch metrics"""
        self.current_batch = batch_idx
        batch_data = {
            "batch": batch_idx,
            "epoch": self.current_epoch,
            **metrics
        }
        self.batch_metrics.append(batch_data)
    
    def get_epoch_history(self) -> List[Dict[str, Any]]:
        """Get epoch history"""
        return self.epoch_metrics.copy()
    
    def get_batch_history(self) -> List[Dict[str, Any]]:
        """Get batch history"""
        return self.batch_metrics.copy()
    
    def get_latest_metrics(self) -> Dict[str, Any]:
        """Get latest metrics"""
        if not self.epoch_metrics:
            return {}
        
        latest = self.epoch_metrics[-1]
        return {
            "epoch": latest["epoch"],
            "train": latest.get("train", {}),
            "val": latest.get("val", {})
        }
    
    def get_best_epoch(self, metric: str = "val_loss", mode: str = "min") -> Optional[Dict[str, Any]]:
        """Get best epoch based on metric"""
        if not self.epoch_metrics:
            return None
        
        best_epoch = None
        best_value = float('inf') if mode == "min" else float('-inf')
        
        for epoch_data in self.epoch_metrics:
            val_metrics = epoch_data.get("val", {})
            if metric in val_metrics:
                value = val_metrics[metric]
                is_better = (
                    value < best_value if mode == "min"
                    else value > best_value
                )
                if is_better:
                    best_value = value
                    best_epoch = epoch_data
        
        return best_epoch
    
    def clear(self):
        """Clear all metrics"""
        self.epoch_metrics.clear()
        self.batch_metrics.clear()



