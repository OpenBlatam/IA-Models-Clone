"""
🔒 Advanced Checkpoint Management System
Provides comprehensive checkpointing capabilities for ML models and experiments.
"""

import os
import json
import shutil
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
import torch
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class CheckpointMetadata:
    """Metadata for a checkpoint"""
    checkpoint_id: str
    timestamp: str
    epoch: int
    step: int
    model_name: str
    experiment_name: str
    file_size: int
    checksum: str
    metrics: Dict[str, float]
    config: Dict[str, Any]
    tags: List[str]
    description: str = ""
    is_best: bool = False
    is_latest: bool = False

@dataclass
class CheckpointConfig:
    """Configuration for checkpoint management"""
    checkpoint_dir: str = "checkpoints"
    save_interval: int = 1  # Save every N epochs
    max_checkpoints: int = 10  # Keep only N most recent checkpoints
    save_best_only: bool = False  # Save only best checkpoints
    monitor_metric: str = "val_loss"
    monitor_mode: str = "min"  # min or max
    save_optimizer: bool = True
    save_scheduler: bool = True
    save_metadata: bool = True
    compression: bool = False
    backup_checkpoints: bool = True
    validate_checkpoints: bool = True

class CheckpointManager:
    """
    Advanced checkpoint management system with automatic checkpointing,
    validation, comparison, and recovery capabilities.
    """
    
    def __init__(self, config: CheckpointConfig, experiment_name: str = "experiment"):
        self.config = config
        self.experiment_name = experiment_name
        self.checkpoint_dir = Path(config.checkpoint_dir) / experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Metadata storage
        self.metadata_file = self.checkpoint_dir / "checkpoint_metadata.json"
        self.checkpoints: Dict[str, CheckpointMetadata] = {}
        self.best_checkpoint: Optional[str] = None
        self.latest_checkpoint: Optional[str] = None
        
        # Load existing metadata
        self._load_metadata()
        
        logger.info(f"CheckpointManager initialized for experiment: {experiment_name}")
        logger.info(f"Checkpoint directory: {self.checkpoint_dir}")
    
    def _load_metadata(self) -> None:
        """Load existing checkpoint metadata"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)
                    self.checkpoints = {
                        k: CheckpointMetadata(**v) for k, v in data.get('checkpoints', {}).items()
                    }
                    self.best_checkpoint = data.get('best_checkpoint')
                    self.latest_checkpoint = data.get('latest_checkpoint')
                logger.info(f"Loaded metadata for {len(self.checkpoints)} checkpoints")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint metadata: {e}")
    
    def _save_metadata(self) -> None:
        """Save checkpoint metadata"""
        try:
            data = {
                'checkpoints': {k: asdict(v) for k, v in self.checkpoints.items()},
                'best_checkpoint': self.best_checkpoint,
                'latest_checkpoint': self.latest_checkpoint,
                'last_updated': datetime.now().isoformat()
            }
            with open(self.metadata_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save checkpoint metadata: {e}")
    
    def _generate_checkpoint_id(self, epoch: int, step: int) -> str:
        """Generate unique checkpoint ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"checkpoint_epoch_{epoch:04d}_step_{step:06d}_{timestamp}"
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of checkpoint file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def _validate_checkpoint(self, checkpoint_path: Path, expected_checksum: str) -> bool:
        """Validate checkpoint file integrity"""
        if not checkpoint_path.exists():
            logger.error(f"Checkpoint file not found: {checkpoint_path}")
            return False
        
        try:
            actual_checksum = self._calculate_checksum(checkpoint_path)
            if actual_checksum != expected_checksum:
                logger.error(f"Checksum mismatch for {checkpoint_path}")
                return False
            
            # Try loading the checkpoint to ensure it's valid
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            required_keys = ['model_state_dict', 'epoch', 'step']
            if not all(key in checkpoint for key in required_keys):
                logger.error(f"Invalid checkpoint format: {checkpoint_path}")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Checkpoint validation failed for {checkpoint_path}: {e}")
            return False
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        epoch: int = 0,
        step: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        description: str = "",
        force_save: bool = False
    ) -> Optional[str]:
        """
        Save a comprehensive checkpoint with metadata.
        
        Args:
            model: PyTorch model to save
            optimizer: Optimizer state (optional)
            scheduler: Scheduler state (optional)
            epoch: Current epoch
            step: Current step
            metrics: Current metrics
            config: Model/training configuration
            tags: Tags for the checkpoint
            description: Description of the checkpoint
            force_save: Force save even if conditions not met
        
        Returns:
            Checkpoint ID if saved, None otherwise
        """
        try:
            # Check if we should save based on interval
            if not force_save and epoch % self.config.save_interval != 0:
                return None
            
            # Check if we should save only best checkpoints
            if self.config.save_best_only and metrics:
                if not self._should_save_best(metrics):
                    return None
            
            # Generate checkpoint ID and path
            checkpoint_id = self._generate_checkpoint_id(epoch, step)
            checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.pt"
            
            # Prepare checkpoint data
            checkpoint_data = {
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'step': step,
                'timestamp': datetime.now().isoformat(),
                'checkpoint_id': checkpoint_id,
                'experiment_name': self.experiment_name
            }
            
            if self.config.save_optimizer and optimizer:
                checkpoint_data['optimizer_state_dict'] = optimizer.state_dict()
            
            if self.config.save_scheduler and scheduler:
                checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
            
            if metrics:
                checkpoint_data['metrics'] = metrics
            
            if config:
                checkpoint_data['config'] = config
            
            # Save checkpoint
            torch.save(checkpoint_data, checkpoint_path)
            
            # Calculate file size and checksum
            file_size = checkpoint_path.stat().st_size
            checksum = self._calculate_checksum(checkpoint_path)
            
            # Create metadata
            metadata = CheckpointMetadata(
                checkpoint_id=checkpoint_id,
                timestamp=datetime.now().isoformat(),
                epoch=epoch,
                step=step,
                model_name=model.__class__.__name__,
                experiment_name=self.experiment_name,
                file_size=file_size,
                checksum=checksum,
                metrics=metrics or {},
                config=config or {},
                tags=tags or [],
                description=description,
                is_best=self._is_best_checkpoint(metrics),
                is_latest=True
            )
            
            # Update metadata
            self.checkpoints[checkpoint_id] = metadata
            self.latest_checkpoint = checkpoint_id
            
            if metadata.is_best:
                self.best_checkpoint = checkpoint_id
            
            # Update is_latest flag for all checkpoints
            for cp in self.checkpoints.values():
                cp.is_latest = (cp.checkpoint_id == checkpoint_id)
            
            # Save metadata
            self._save_metadata()
            
            # Cleanup old checkpoints
            self._cleanup_old_checkpoints()
            
            # Create backup if enabled
            if self.config.backup_checkpoints:
                self._create_backup(checkpoint_path, checkpoint_id)
            
            logger.info(f"Checkpoint saved: {checkpoint_id} (epoch {epoch}, step {step})")
            return checkpoint_id
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return None
    
    def _should_save_best(self, metrics: Dict[str, float]) -> bool:
        """Check if current metrics indicate best performance"""
        if not metrics or self.config.monitor_metric not in metrics:
            return False
        
        current_metric = metrics[self.config.monitor_metric]
        
        if self.best_checkpoint is None:
            return True
        
        best_metadata = self.checkpoints.get(self.best_checkpoint)
        if not best_metadata or self.config.monitor_metric not in best_metadata.metrics:
            return True
        
        best_metric = best_metadata.metrics[self.config.monitor_metric]
        
        if self.config.monitor_mode == "min":
            return current_metric < best_metric
        else:  # max
            return current_metric > best_metric
    
    def _is_best_checkpoint(self, metrics: Optional[Dict[str, float]]) -> bool:
        """Check if this checkpoint is the best so far"""
        if not metrics or self.config.monitor_metric not in metrics:
            return False
        
        current_metric = metrics[self.config.monitor_metric]
        
        for metadata in self.checkpoints.values():
            if (metadata.metrics and 
                self.config.monitor_metric in metadata.metrics):
                best_metric = metadata.metrics[self.config.monitor_metric]
                
                if self.config.monitor_mode == "min":
                    if current_metric >= best_metric:
                        return False
                else:  # max
                    if current_metric <= best_metric:
                        return False
        
        return True
    
    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints to stay within max_checkpoints limit"""
        if len(self.checkpoints) <= self.config.max_checkpoints:
            return
        
        # Sort checkpoints by timestamp (oldest first)
        sorted_checkpoints = sorted(
            self.checkpoints.items(),
            key=lambda x: x[1].timestamp
        )
        
        # Keep best checkpoint and latest checkpoint
        checkpoints_to_keep = set()
        if self.best_checkpoint:
            checkpoints_to_keep.add(self.best_checkpoint)
        if self.latest_checkpoint:
            checkpoints_to_keep.add(self.latest_checkpoint)
        
        # Add most recent checkpoints up to max_checkpoints
        for checkpoint_id, _ in sorted_checkpoints[-self.config.max_checkpoints:]:
            checkpoints_to_keep.add(checkpoint_id)
        
        # Remove old checkpoints
        for checkpoint_id, metadata in sorted_checkpoints:
            if checkpoint_id not in checkpoints_to_keep:
                checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.pt"
                if checkpoint_path.exists():
                    checkpoint_path.unlink()
                del self.checkpoints[checkpoint_id]
                logger.info(f"Removed old checkpoint: {checkpoint_id}")
    
    def _create_backup(self, checkpoint_path: Path, checkpoint_id: str) -> None:
        """Create a backup of the checkpoint"""
        backup_dir = self.checkpoint_dir / "backups"
        backup_dir.mkdir(exist_ok=True)
        
        backup_path = backup_dir / f"{checkpoint_id}.pt"
        shutil.copy2(checkpoint_path, backup_path)
        logger.debug(f"Backup created: {backup_path}")
    
    def load_checkpoint(
        self,
        checkpoint_id: Optional[str] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
        model: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Load a checkpoint.
        
        Args:
            checkpoint_id: ID of checkpoint to load (from metadata)
            checkpoint_path: Direct path to checkpoint file
            model: Model to load state into
            optimizer: Optimizer to load state into
            scheduler: Scheduler to load state into
            device: Device to load checkpoint on
        
        Returns:
            Checkpoint data dictionary
        """
        try:
            # Determine checkpoint path
            if checkpoint_path:
                checkpoint_path = Path(checkpoint_path)
            elif checkpoint_id:
                if checkpoint_id not in self.checkpoints:
                    logger.error(f"Checkpoint ID not found: {checkpoint_id}")
                    return None
                checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.pt"
            else:
                # Load latest checkpoint
                if not self.latest_checkpoint:
                    logger.error("No latest checkpoint available")
                    return None
                checkpoint_path = self.checkpoint_dir / f"{self.latest_checkpoint}.pt"
            
            if not checkpoint_path.exists():
                logger.error(f"Checkpoint file not found: {checkpoint_path}")
                return None
            
            # Validate checkpoint if enabled
            if self.config.validate_checkpoints and checkpoint_id:
                metadata = self.checkpoints[checkpoint_id]
                if not self._validate_checkpoint(checkpoint_path, metadata.checksum):
                    logger.error(f"Checkpoint validation failed: {checkpoint_id}")
                    return None
            
            # Load checkpoint
            checkpoint_data = torch.load(checkpoint_path, map_location=device or 'cpu')
            
            # Load into model if provided
            if model and 'model_state_dict' in checkpoint_data:
                model.load_state_dict(checkpoint_data['model_state_dict'])
                logger.info(f"Model state loaded from checkpoint")
            
            # Load into optimizer if provided
            if optimizer and 'optimizer_state_dict' in checkpoint_data:
                optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
                logger.info(f"Optimizer state loaded from checkpoint")
            
            # Load into scheduler if provided
            if scheduler and 'scheduler_state_dict' in checkpoint_data:
                scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
                logger.info(f"Scheduler state loaded from checkpoint")
            
            logger.info(f"Checkpoint loaded: {checkpoint_path}")
            return checkpoint_data
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None
    
    def get_checkpoint_info(self, checkpoint_id: str) -> Optional[CheckpointMetadata]:
        """Get metadata for a specific checkpoint"""
        return self.checkpoints.get(checkpoint_id)
    
    def list_checkpoints(self, sort_by: str = "timestamp", reverse: bool = True) -> List[CheckpointMetadata]:
        """List all available checkpoints"""
        checkpoints = list(self.checkpoints.values())
        
        if sort_by == "timestamp":
            checkpoints.sort(key=lambda x: x.timestamp, reverse=reverse)
        elif sort_by == "epoch":
            checkpoints.sort(key=lambda x: x.epoch, reverse=reverse)
        elif sort_by == "step":
            checkpoints.sort(key=lambda x: x.step, reverse=reverse)
        elif sort_by == "file_size":
            checkpoints.sort(key=lambda x: x.file_size, reverse=reverse)
        
        return checkpoints
    
    def get_best_checkpoint(self) -> Optional[CheckpointMetadata]:
        """Get the best checkpoint metadata"""
        if self.best_checkpoint:
            return self.checkpoints.get(self.best_checkpoint)
        return None
    
    def get_latest_checkpoint(self) -> Optional[CheckpointMetadata]:
        """Get the latest checkpoint metadata"""
        if self.latest_checkpoint:
            return self.checkpoints.get(self.latest_checkpoint)
        return None
    
    def compare_checkpoints(self, checkpoint_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple checkpoints"""
        if len(checkpoint_ids) < 2:
            logger.error("Need at least 2 checkpoints to compare")
            return {}
        
        comparison = {
            'checkpoints': {},
            'metrics_comparison': {},
            'file_sizes': {},
            'timestamps': {}
        }
        
        for checkpoint_id in checkpoint_ids:
            if checkpoint_id not in self.checkpoints:
                logger.warning(f"Checkpoint not found: {checkpoint_id}")
                continue
            
            metadata = self.checkpoints[checkpoint_id]
            comparison['checkpoints'][checkpoint_id] = asdict(metadata)
            comparison['file_sizes'][checkpoint_id] = metadata.file_size
            comparison['timestamps'][checkpoint_id] = metadata.timestamp
            
            # Compare metrics
            for metric_name, metric_value in metadata.metrics.items():
                if metric_name not in comparison['metrics_comparison']:
                    comparison['metrics_comparison'][metric_name] = {}
                comparison['metrics_comparison'][metric_name][checkpoint_id] = metric_value
        
        return comparison
    
    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete a specific checkpoint"""
        if checkpoint_id not in self.checkpoints:
            logger.error(f"Checkpoint not found: {checkpoint_id}")
            return False
        
        try:
            # Remove file
            checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.pt"
            if checkpoint_path.exists():
                checkpoint_path.unlink()
            
            # Remove from metadata
            del self.checkpoints[checkpoint_id]
            
            # Update best/latest references
            if self.best_checkpoint == checkpoint_id:
                self.best_checkpoint = None
            if self.latest_checkpoint == checkpoint_id:
                self.latest_checkpoint = None
            
            # Save updated metadata
            self._save_metadata()
            
            logger.info(f"Checkpoint deleted: {checkpoint_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete checkpoint {checkpoint_id}: {e}")
            return False
    
    def export_checkpoint(self, checkpoint_id: str, export_path: Union[str, Path]) -> bool:
        """Export a checkpoint to a different location"""
        if checkpoint_id not in self.checkpoints:
            logger.error(f"Checkpoint not found: {checkpoint_id}")
            return False
        
        try:
            source_path = self.checkpoint_dir / f"{checkpoint_id}.pt"
            export_path = Path(export_path)
            
            if not source_path.exists():
                logger.error(f"Checkpoint file not found: {source_path}")
                return False
            
            # Copy checkpoint file
            shutil.copy2(source_path, export_path)
            
            # Export metadata
            metadata = self.checkpoints[checkpoint_id]
            metadata_path = export_path.parent / f"{checkpoint_id}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(asdict(metadata), f, indent=2)
            
            logger.info(f"Checkpoint exported to: {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export checkpoint {checkpoint_id}: {e}")
            return False
    
    def get_checkpoint_summary(self) -> Dict[str, Any]:
        """Get a summary of all checkpoints"""
        if not self.checkpoints:
            return {"message": "No checkpoints available"}
        
        total_size = sum(cp.file_size for cp in self.checkpoints.values())
        metrics_summary = defaultdict(list)
        
        for metadata in self.checkpoints.values():
            for metric_name, metric_value in metadata.metrics.items():
                metrics_summary[metric_name].append(metric_value)
        
        return {
            "total_checkpoints": len(self.checkpoints),
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "best_checkpoint": self.best_checkpoint,
            "latest_checkpoint": self.latest_checkpoint,
            "metrics_summary": {
                metric: {
                    "min": min(values),
                    "max": max(values),
                    "mean": np.mean(values),
                    "std": np.std(values)
                } for metric, values in metrics_summary.items()
            },
            "checkpoint_ids": list(self.checkpoints.keys())
        }






