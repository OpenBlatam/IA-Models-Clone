"""
Checkpoint Manager Module
=========================

Advanced checkpoint management and utilities.

Author: BUL System
Date: 2024
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
import json

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages training checkpoints with metadata and utilities.
    
    Provides:
    - Checkpoint listing and management
    - Metadata storage and retrieval
    - Checkpoint comparison
    - Automatic cleanup
    
    Example:
        >>> manager = CheckpointManager("./checkpoints")
        >>> checkpoints = manager.list_checkpoints()
        >>> best = manager.get_best_checkpoint()
    """
    
    def __init__(self, checkpoint_dir: Path):
        """
        Initialize CheckpointManager.
        
        Args:
            checkpoint_dir: Directory containing checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        List all available checkpoints.
        
        Returns:
            List of checkpoint information dictionaries
        """
        checkpoints = []
        
        if not self.checkpoint_dir.exists():
            return checkpoints
        
        for item in self.checkpoint_dir.iterdir():
            if item.is_dir() and (item / "pytorch_model.bin").exists() or (item / "model.safetensors").exists():
                metadata = self.get_checkpoint_metadata(item)
                checkpoints.append({
                    "path": str(item),
                    "name": item.name,
                    **metadata
                })
        
        # Sort by step number if available
        checkpoints.sort(key=lambda x: x.get("step", 0), reverse=True)
        return checkpoints
    
    def get_checkpoint_metadata(self, checkpoint_path: Path) -> Dict[str, Any]:
        """
        Get metadata for a checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint directory
            
        Returns:
            Dictionary with checkpoint metadata
        """
        metadata = {
            "path": str(checkpoint_path),
            "exists": checkpoint_path.exists(),
        }
        
        # Try to read training state or metadata file
        training_state_file = checkpoint_path / "training_state.json"
        if training_state_file.exists():
            try:
                with open(training_state_file, 'r') as f:
                    state = json.load(f)
                    metadata.update(state)
            except Exception as e:
                logger.debug(f"Could not read training state: {e}")
        
        # Extract step number from directory name
        if "checkpoint-" in checkpoint_path.name:
            try:
                step = int(checkpoint_path.name.split("checkpoint-")[-1])
                metadata["step"] = step
            except ValueError:
                pass
        
        return metadata
    
    def get_best_checkpoint(self, metric: str = "eval_loss", mode: str = "min") -> Optional[Path]:
        """
        Get the best checkpoint based on a metric.
        
        Args:
            metric: Metric name to use for comparison
            mode: "min" or "max" - whether to minimize or maximize the metric
            
        Returns:
            Path to best checkpoint or None
        """
        checkpoints = self.list_checkpoints()
        
        if not checkpoints:
            return None
        
        # Filter checkpoints with the metric
        checkpoints_with_metric = [
            c for c in checkpoints
            if metric in c.get("metrics", {}) or metric in c
        ]
        
        if not checkpoints_with_metric:
            # Return latest checkpoint if no metric available
            return Path(checkpoints[0]["path"])
        
        # Find best based on metric
        if mode == "min":
            best = min(checkpoints_with_metric, key=lambda x: x.get(metric, x.get("metrics", {}).get(metric, float('inf'))))
        else:
            best = max(checkpoints_with_metric, key=lambda x: x.get(metric, x.get("metrics", {}).get(metric, float('-inf'))))
        
        return Path(best["path"])
    
    def get_latest_checkpoint(self) -> Optional[Path]:
        """
        Get the latest checkpoint.
        
        Returns:
            Path to latest checkpoint or None
        """
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            return None
        
        # Return checkpoint with highest step number
        latest = max(checkpoints, key=lambda x: x.get("step", 0))
        return Path(latest["path"])
    
    def cleanup_old_checkpoints(self, keep: int = 3) -> List[Path]:
        """
        Clean up old checkpoints, keeping only the most recent ones.
        
        Args:
            keep: Number of checkpoints to keep
            
        Returns:
            List of deleted checkpoint paths
        """
        checkpoints = self.list_checkpoints()
        
        if len(checkpoints) <= keep:
            return []
        
        # Sort by step (newest first)
        checkpoints.sort(key=lambda x: x.get("step", 0), reverse=True)
        
        # Get checkpoints to delete
        to_delete = checkpoints[keep:]
        
        deleted = []
        for checkpoint_info in to_delete:
            checkpoint_path = Path(checkpoint_info["path"])
            try:
                # Delete checkpoint directory
                import shutil
                shutil.rmtree(checkpoint_path)
                deleted.append(checkpoint_path)
                logger.info(f"Deleted old checkpoint: {checkpoint_path}")
            except Exception as e:
                logger.warning(f"Could not delete checkpoint {checkpoint_path}: {e}")
        
        return deleted

