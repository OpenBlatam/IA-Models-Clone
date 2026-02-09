"""
Resume Training Manager Module
===============================

Utilities for resuming training from checkpoints.

Author: BUL System
Date: 2024
"""

import logging
from pathlib import Path
from typing import Optional, Union, Dict, Any

logger = logging.getLogger(__name__)


class ResumeManager:
    """
    Manages resuming training from checkpoints.
    
    Provides utilities to:
    - Find latest checkpoint
    - Validate checkpoint integrity
    - Prepare resume configuration
    
    Example:
        >>> manager = ResumeManager("./checkpoints")
        >>> checkpoint = manager.find_latest_checkpoint()
        >>> if checkpoint:
        ...     trainer.train(resume_from_checkpoint=checkpoint)
    """
    
    def __init__(self, checkpoint_dir: Path):
        """
        Initialize ResumeManager.
        
        Args:
            checkpoint_dir: Directory containing checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
    
    def find_latest_checkpoint(self) -> Optional[Union[str, bool]]:
        """
        Find the latest checkpoint for resuming.
        
        Returns:
            Path to latest checkpoint, True if should auto-resume, or None
        """
        if not self.checkpoint_dir.exists():
            return None
        
        # Look for checkpoint directories
        checkpoints = []
        for item in self.checkpoint_dir.iterdir():
            if item.is_dir() and "checkpoint" in item.name.lower():
                # Check if it's a valid checkpoint
                if self._is_valid_checkpoint(item):
                    # Extract step number
                    step = self._extract_step_number(item.name)
                    checkpoints.append((step, item))
        
        if not checkpoints:
            return None
        
        # Sort by step number (highest first)
        checkpoints.sort(key=lambda x: x[0], reverse=True)
        latest_checkpoint = checkpoints[0][1]
        
        logger.info(f"Found latest checkpoint: {latest_checkpoint}")
        return str(latest_checkpoint)
    
    def _is_valid_checkpoint(self, checkpoint_path: Path) -> bool:
        """
        Check if a checkpoint directory is valid.
        
        Args:
            checkpoint_path: Path to checkpoint directory
            
        Returns:
            True if checkpoint is valid
        """
        # Check for required files
        required_files = ["config.json"]
        model_files = ["pytorch_model.bin", "model.safetensors"]
        
        has_config = any((checkpoint_path / f).exists() for f in required_files)
        has_model = any((checkpoint_path / f).exists() for f in model_files)
        
        return has_config and has_model
    
    def _extract_step_number(self, checkpoint_name: str) -> int:
        """
        Extract step number from checkpoint name.
        
        Args:
            checkpoint_name: Name of checkpoint directory
            
        Returns:
            Step number (0 if not found)
        """
        try:
            # Format: checkpoint-500, checkpoint-1000, etc.
            if "checkpoint-" in checkpoint_name:
                step_str = checkpoint_name.split("checkpoint-")[-1]
                return int(step_str)
            # Format: step_500, step_1000, etc.
            elif "step_" in checkpoint_name:
                step_str = checkpoint_name.split("step_")[-1]
                return int(step_str)
        except (ValueError, IndexError):
            pass
        
        return 0
    
    def get_resume_info(self, checkpoint_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        Get information about resuming from a checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint (None to find latest)
            
        Returns:
            Dictionary with resume information
        """
        if checkpoint_path is None:
            checkpoint_path = self.find_latest_checkpoint()
        
        if checkpoint_path is None:
            return {
                "can_resume": False,
                "checkpoint_path": None,
                "message": "No checkpoint found"
            }
        
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            return {
                "can_resume": False,
                "checkpoint_path": str(checkpoint_path),
                "message": "Checkpoint path does not exist"
            }
        
        if not self._is_valid_checkpoint(checkpoint_path):
            return {
                "can_resume": False,
                "checkpoint_path": str(checkpoint_path),
                "message": "Checkpoint is invalid or incomplete"
            }
        
        step = self._extract_step_number(checkpoint_path.name)
        
        return {
            "can_resume": True,
            "checkpoint_path": str(checkpoint_path),
            "step": step,
            "message": f"Can resume from step {step}"
        }

