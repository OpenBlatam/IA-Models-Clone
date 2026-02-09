from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import os
import json
import logging
import time
import shutil
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
import pickle
import zipfile
import tempfile
    import torch
    import numpy as np
from typing import Any, List, Dict, Optional
import asyncio
"""
Checkpoint Management System
===========================

This module provides comprehensive checkpoint management for AI video generation models.

Features:
- Model checkpointing with metadata
- Checkpoint versioning and tagging
- Automatic checkpoint cleanup
- Checkpoint validation and recovery
- Distributed checkpointing support
- Checkpoint compression and optimization
"""


# Optional imports
try:
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class CheckpointMetadata:
    """Metadata for a checkpoint."""
    
    # Basic information
    checkpoint_id: str
    checkpoint_path: str
    model_name: str
    model_version: str
    
    # Training information
    epoch: int
    step: int
    total_steps: int
    
    # Performance metrics
    metrics: Dict[str, float] = field(default_factory=dict)
    best_metrics: Dict[str, float] = field(default_factory=dict)
    
    # System information
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    duration: float = 0.0  # Training duration in seconds
    gpu_memory_used: float = 0.0
    cpu_memory_used: float = 0.0
    
    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    
    # Tags and notes
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    
    # File information
    file_size: int = 0
    checksum: str = ""
    compression_ratio: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CheckpointMetadata':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class CheckpointConfig:
    """Configuration for checkpoint management."""
    
    # Directory settings
    checkpoint_dir: str = "checkpoints"
    backup_dir: str = "checkpoint_backups"
    
    # Checkpoint settings
    save_frequency: int = 1000  # Save every N steps
    save_best_only: bool = True
    max_checkpoints: int = 5
    max_backups: int = 10
    
    # Metrics for best checkpoint selection
    primary_metric: str = "val_loss"
    secondary_metrics: List[str] = field(default_factory=lambda: ["loss", "accuracy"])
    
    # Compression and optimization
    use_compression: bool = True
    compression_level: int = 6
    remove_optimizer_state: bool = False  # Save space by removing optimizer state
    
    # Validation
    validate_checkpoints: bool = True
    verify_checksum: bool = True
    
    # Distributed settings
    is_distributed: bool = False
    local_rank: int = 0
    world_size: int = 1


class CheckpointManager:
    """Main checkpoint management class."""
    
    def __init__(self, config: CheckpointConfig):
        
    """__init__ function."""
self.config = config
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.backup_dir = Path(config.backup_dir)
        
        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing checkpoints
        self.checkpoints: List[CheckpointMetadata] = []
        self.best_checkpoint: Optional[CheckpointMetadata] = None
        
        # Load existing checkpoints
        self._load_existing_checkpoints()
        
        logger.info(f"Checkpoint manager initialized: {self.checkpoint_dir}")
    
    def _load_existing_checkpoints(self) -> Any:
        """Load existing checkpoints from directory."""
        metadata_files = list(self.checkpoint_dir.glob("*.metadata.json"))
        
        for metadata_file in metadata_files:
            try:
                with open(metadata_file, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    metadata_dict = json.load(f)
                
                metadata = CheckpointMetadata.from_dict(metadata_dict)
                
                # Verify checkpoint file exists
                checkpoint_path = Path(metadata.checkpoint_path)
                if checkpoint_path.exists():
                    self.checkpoints.append(metadata)
                    
                    # Update best checkpoint
                    if self._is_better_checkpoint(metadata):
                        self.best_checkpoint = metadata
                else:
                    logger.warning(f"Checkpoint file not found: {checkpoint_path}")
                    
            except Exception as e:
                logger.warning(f"Failed to load checkpoint metadata {metadata_file}: {e}")
        
        # Sort checkpoints by step
        self.checkpoints.sort(key=lambda x: x.step)
        logger.info(f"Loaded {len(self.checkpoints)} existing checkpoints")
    
    def _is_better_checkpoint(self, checkpoint: CheckpointMetadata) -> bool:
        """Check if checkpoint is better than current best."""
        if self.best_checkpoint is None:
            return True
        
        # Compare primary metric
        if self.config.primary_metric in checkpoint.metrics:
            current_best = self.best_checkpoint.metrics.get(self.config.primary_metric, float('inf'))
            new_value = checkpoint.metrics[self.config.primary_metric]
            
            # Lower is better for loss metrics
            if "loss" in self.config.primary_metric.lower():
                return new_value < current_best
            else:
                return new_value > current_best
        
        return False
    
    def _generate_checkpoint_id(self, model_name: str, epoch: int, step: int) -> str:
        """Generate unique checkpoint ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{model_name}_epoch_{epoch}_step_{step}_{timestamp}"
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of file."""
        sha256_hash = hashlib.sha256()
        
        with open(file_path, "rb") as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            for chunk in iter(lambda: f.read(4096), b""):
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                sha256_hash.update(chunk)
        
        return sha256_hash.hexdigest()
    
    def _compress_checkpoint(self, checkpoint_path: Path) -> Tuple[Path, float]:
        """Compress checkpoint file."""
        if not self.config.use_compression:
            return checkpoint_path, 1.0
        
        compressed_path = checkpoint_path.with_suffix('.zip')
        
        with zipfile.ZipFile(compressed_path, 'w', zipfile.ZIP_DEFLATED, 
                           compresslevel=self.config.compression_level) as zipf:
            zipf.write(checkpoint_path, checkpoint_path.name)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        
        # Calculate compression ratio
        original_size = checkpoint_path.stat().st_size
        compressed_size = compressed_path.stat().st_size
        compression_ratio = compressed_size / original_size
        
        # Remove original file
        checkpoint_path.unlink()
        
        return compressed_path, compression_ratio
    
    def _decompress_checkpoint(self, compressed_path: Path) -> Path:
        """Decompress checkpoint file."""
        if not compressed_path.suffix == '.zip':
            return compressed_path
        
        checkpoint_path = compressed_path.with_suffix('')
        
        with zipfile.ZipFile(compressed_path, 'r') as zipf:
            zipf.extractall(compressed_path.parent)
        
        return checkpoint_path
    
    def save_checkpoint(
        self,
        model,
        optimizer=None,
        scheduler=None,
        metrics: Optional[Dict[str, float]] = None,
        epoch: int = 0,
        step: int = 0,
        total_steps: int = 0,
        config: Optional[Dict[str, Any]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        notes: str = "",
        duration: float = 0.0,
        force_save: bool = False
    ) -> Optional[str]:
        """Save model checkpoint."""
        if metrics is None:
            metrics = {}
        if config is None:
            config = {}
        if hyperparameters is None:
            hyperparameters = {}
        if tags is None:
            tags = []
        
        # Check if we should save
        if not force_save and step % self.config.save_frequency != 0:
            return None
        
        # Generate checkpoint ID and path
        model_name = config.get("model_name", "model")
        checkpoint_id = self._generate_checkpoint_id(model_name, epoch, step)
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.pth"
        
        # Prepare checkpoint data
        checkpoint_data = {
            "model_state": model.state_dict() if TORCH_AVAILABLE else {},
            "epoch": epoch,
            "step": step,
            "total_steps": total_steps,
            "metrics": metrics,
            "config": config,
            "hyperparameters": hyperparameters
        }
        
        # Add optimizer and scheduler states
        if optimizer and not self.config.remove_optimizer_state:
            checkpoint_data["optimizer_state"] = optimizer.state_dict()
        if scheduler and not self.config.remove_optimizer_state:
            checkpoint_data["scheduler_state"] = scheduler.state_dict()
        
        try:
            # Save checkpoint
            if TORCH_AVAILABLE:
                torch.save(checkpoint_data, checkpoint_path)
            else:
                with open(checkpoint_path, 'wb') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    pickle.dump(checkpoint_data, f)
            
            # Calculate file size and checksum
            file_size = checkpoint_path.stat().st_size
            checksum = self._calculate_checksum(checkpoint_path)
            
            # Compress if enabled
            compression_ratio = 1.0
            if self.config.use_compression:
                checkpoint_path, compression_ratio = self._compress_checkpoint(checkpoint_path)
            
            # Create metadata
            metadata = CheckpointMetadata(
                checkpoint_id=checkpoint_id,
                checkpoint_path=str(checkpoint_path),
                model_name=model_name,
                model_version=config.get("model_version", "1.0.0"),
                epoch=epoch,
                step=step,
                total_steps=total_steps,
                metrics=metrics,
                best_metrics=self.best_checkpoint.metrics if self.best_checkpoint else {},
                timestamp=datetime.now().isoformat(),
                duration=duration,
                config=config,
                hyperparameters=hyperparameters,
                tags=tags,
                notes=notes,
                file_size=file_size,
                checksum=checksum,
                compression_ratio=compression_ratio
            )
            
            # Save metadata
            metadata_path = checkpoint_path.with_suffix('.metadata.json')
            with open(metadata_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                json.dump(metadata.to_dict(), f, indent=2)
            
            # Add to checkpoints list
            self.checkpoints.append(metadata)
            
            # Update best checkpoint
            if self._is_better_checkpoint(metadata):
                self.best_checkpoint = metadata
                logger.info(f"New best checkpoint: {checkpoint_id}")
            
            # Cleanup old checkpoints
            if self.config.save_best_only:
                self._cleanup_old_checkpoints()
            
            logger.info(f"Checkpoint saved: {checkpoint_path}")
            return str(checkpoint_path)
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return None
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        model,
        optimizer=None,
        scheduler=None,
        device: str = "cpu"
    ) -> Optional[Dict[str, Any]]:
        """Load checkpoint from file."""
        try:
            checkpoint_path = Path(checkpoint_path)
            
            # Decompress if needed
            if checkpoint_path.suffix == '.zip':
                checkpoint_path = self._decompress_checkpoint(checkpoint_path)
            
            # Load checkpoint data
            if TORCH_AVAILABLE:
                checkpoint_data = torch.load(checkpoint_path, map_location=device)
            else:
                with open(checkpoint_path, 'rb') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    checkpoint_data = pickle.load(f)
            
            # Load model state
            if "model_state" in checkpoint_data:
                model.load_state_dict(checkpoint_data["model_state"])
            
            # Load optimizer state
            if optimizer and "optimizer_state" in checkpoint_data:
                optimizer.load_state_dict(checkpoint_data["optimizer_state"])
            
            # Load scheduler state
            if scheduler and "scheduler_state" in checkpoint_data:
                scheduler.load_state_dict(checkpoint_data["scheduler_state"])
            
            logger.info(f"Checkpoint loaded: {checkpoint_path}")
            return checkpoint_data
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
            return None
    
    def load_latest_checkpoint(self, model, optimizer=None, scheduler=None, device: str = "cpu"):
        """Load the latest checkpoint."""
        if not self.checkpoints:
            logger.warning("No checkpoints available")
            return None
        
        latest_checkpoint = max(self.checkpoints, key=lambda x: x.step)
        return self.load_checkpoint(latest_checkpoint.checkpoint_path, model, optimizer, scheduler, device)
    
    def load_best_checkpoint(self, model, optimizer=None, scheduler=None, device: str = "cpu"):
        """Load the best checkpoint."""
        if not self.best_checkpoint:
            logger.warning("No best checkpoint available")
            return None
        
        return self.load_checkpoint(self.best_checkpoint.checkpoint_path, model, optimizer, scheduler, device)
    
    def _cleanup_old_checkpoints(self) -> Any:
        """Remove old checkpoints to save space."""
        if len(self.checkpoints) <= self.config.max_checkpoints:
            return
        
        # Sort by step and keep most recent
        self.checkpoints.sort(key=lambda x: x.step, reverse=True)
        
        checkpoints_to_remove = self.checkpoints[self.config.max_checkpoints:]
        for checkpoint in checkpoints_to_remove:
            try:
                # Remove checkpoint file
                checkpoint_path = Path(checkpoint.checkpoint_path)
                if checkpoint_path.exists():
                    checkpoint_path.unlink()
                
                # Remove metadata file
                metadata_path = checkpoint_path.with_suffix('.metadata.json')
                if metadata_path.exists():
                    metadata_path.unlink()
                
                logger.info(f"Removed old checkpoint: {checkpoint.checkpoint_id}")
                
            except Exception as e:
                logger.warning(f"Failed to remove checkpoint {checkpoint.checkpoint_id}: {e}")
        
        # Update checkpoints list
        self.checkpoints = self.checkpoints[:self.config.max_checkpoints]
    
    def create_backup(self, checkpoint_id: str) -> bool:
        """Create backup of a checkpoint."""
        try:
            # Find checkpoint
            checkpoint = next((c for c in self.checkpoints if c.checkpoint_id == checkpoint_id), None)
            if not checkpoint:
                logger.warning(f"Checkpoint not found: {checkpoint_id}")
                return False
            
            checkpoint_path = Path(checkpoint.checkpoint_path)
            if not checkpoint_path.exists():
                logger.warning(f"Checkpoint file not found: {checkpoint_path}")
                return False
            
            # Create backup
            backup_path = self.backup_dir / f"{checkpoint_id}_backup.pth"
            shutil.copy2(checkpoint_path, backup_path)
            
            # Backup metadata
            metadata_path = checkpoint_path.with_suffix('.metadata.json')
            if metadata_path.exists():
                backup_metadata_path = self.backup_dir / f"{checkpoint_id}_backup.metadata.json"
                shutil.copy2(metadata_path, backup_metadata_path)
            
            logger.info(f"Backup created: {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            return False
    
    def restore_backup(self, checkpoint_id: str) -> bool:
        """Restore checkpoint from backup."""
        try:
            backup_path = self.backup_dir / f"{checkpoint_id}_backup.pth"
            if not backup_path.exists():
                logger.warning(f"Backup not found: {backup_path}")
                return False
            
            # Restore checkpoint
            checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.pth"
            shutil.copy2(backup_path, checkpoint_path)
            
            # Restore metadata
            backup_metadata_path = self.backup_dir / f"{checkpoint_id}_backup.metadata.json"
            if backup_metadata_path.exists():
                metadata_path = checkpoint_path.with_suffix('.metadata.json')
                shutil.copy2(backup_metadata_path, metadata_path)
            
            logger.info(f"Backup restored: {checkpoint_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore backup: {e}")
            return False
    
    def list_checkpoints(self) -> List[CheckpointMetadata]:
        """List all checkpoints."""
        return sorted(self.checkpoints, key=lambda x: x.step)
    
    def get_checkpoint_info(self, checkpoint_id: str) -> Optional[CheckpointMetadata]:
        """Get information about a specific checkpoint."""
        return next((c for c in self.checkpoints if c.checkpoint_id == checkpoint_id), None)
    
    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete a checkpoint."""
        try:
            checkpoint = self.get_checkpoint_info(checkpoint_id)
            if not checkpoint:
                logger.warning(f"Checkpoint not found: {checkpoint_id}")
                return False
            
            # Remove checkpoint file
            checkpoint_path = Path(checkpoint.checkpoint_path)
            if checkpoint_path.exists():
                checkpoint_path.unlink()
            
            # Remove metadata file
            metadata_path = checkpoint_path.with_suffix('.metadata.json')
            if metadata_path.exists():
                metadata_path.unlink()
            
            # Remove from list
            self.checkpoints = [c for c in self.checkpoints if c.checkpoint_id != checkpoint_id]
            
            # Update best checkpoint if needed
            if self.best_checkpoint and self.best_checkpoint.checkpoint_id == checkpoint_id:
                self.best_checkpoint = max(self.checkpoints, key=lambda x: x.step) if self.checkpoints else None
            
            logger.info(f"Checkpoint deleted: {checkpoint_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete checkpoint: {e}")
            return False
    
    def get_checkpoint_summary(self) -> Dict[str, Any]:
        """Get summary of all checkpoints."""
        if not self.checkpoints:
            return {"total_checkpoints": 0}
        
        total_size = sum(c.file_size for c in self.checkpoints)
        avg_compression = np.mean([c.compression_ratio for c in self.checkpoints]) if NUMPY_AVAILABLE else 1.0
        
        return {
            "total_checkpoints": len(self.checkpoints),
            "total_size_mb": total_size / (1024 * 1024),
            "average_compression_ratio": avg_compression,
            "best_checkpoint": self.best_checkpoint.checkpoint_id if self.best_checkpoint else None,
            "latest_checkpoint": max(self.checkpoints, key=lambda x: x.step).checkpoint_id,
            "checkpoint_range": {
                "min_step": min(c.step for c in self.checkpoints),
                "max_step": max(c.step for c in self.checkpoints)
            }
        }


# Convenience functions
def create_checkpoint_manager(
    checkpoint_dir: str = "checkpoints",
    save_frequency: int = 1000,
    max_checkpoints: int = 5,
    use_compression: bool = True
) -> CheckpointManager:
    """Create checkpoint manager with default settings."""
    config = CheckpointConfig(
        checkpoint_dir=checkpoint_dir,
        save_frequency=save_frequency,
        max_checkpoints=max_checkpoints,
        use_compression=use_compression
    )
    return CheckpointManager(config)


if __name__ == "__main__":
    # Example usage
    print("💾 Checkpoint Management System")
    print("=" * 40)
    
    # Create checkpoint manager
    manager = create_checkpoint_manager(
        checkpoint_dir="test_checkpoints",
        save_frequency=100,
        max_checkpoints=3
    )
    
    # Mock model and optimizer
    class MockModel:
        def state_dict(self) -> Any:
            return {"weights": [1, 2, 3]}
        
        def load_state_dict(self, state_dict) -> Any:
            pass
    
    class MockOptimizer:
        def state_dict(self) -> Any:
            return {"lr": 1e-4}
        
        def load_state_dict(self, state_dict) -> Any:
            pass
    
    model = MockModel()
    optimizer = MockOptimizer()
    
    # Simulate training
    for step in range(500):
        if step % 100 == 0:
            metrics = {
                "loss": 1.0 / (step + 1),
                "val_loss": 1.2 / (step + 1),
                "accuracy": 0.5 + 0.4 * (1 - np.exp(-step / 100))
            }
            
            config = {
                "model_name": "test_model",
                "model_version": "1.0.0",
                "learning_rate": 1e-4
            }
            
            manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                metrics=metrics,
                epoch=step // 100,
                step=step,
                total_steps=500,
                config=config,
                tags=["test", "example"]
            )
    
    # List checkpoints
    checkpoints = manager.list_checkpoints()
    print(f"Total checkpoints: {len(checkpoints)}")
    
    # Get summary
    summary = manager.get_checkpoint_summary()
    print(f"Checkpoint summary: {summary}")
    
    # Load best checkpoint
    best_data = manager.load_best_checkpoint(model, optimizer)
    if best_data:
        print(f"Loaded best checkpoint at step {best_data['step']}")
    
    print("✅ Checkpoint management example completed!") 