from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import torch
import torch.nn as nn
import json
import yaml
import hashlib
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
import pickle
import zipfile
import tempfile
import sqlite3
from git_manager import GitManager, GitConfig
        import argparse
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
Model Checkpointer
Comprehensive model checkpointing system with Git version control integration.
"""


logger = logging.getLogger(__name__)


@dataclass
class CheckpointConfig:
    """Configuration for model checkpointing."""
    # Checkpoint settings
    checkpoint_dir: str = "./checkpoints"
    save_best_only: bool = True
    save_last: bool = True
    save_top_k: int = 3
    save_interval: int = 10  # Save every N epochs
    save_interval_steps: int = 1000  # Save every N steps
    
    # Model information
    model_name: str = "model"
    model_version: str = "1.0.0"
    model_type: str = "pytorch"
    
    # Checkpoint content
    save_model_state: bool = True
    save_optimizer_state: bool = True
    save_scheduler_state: bool = True
    save_scaler_state: bool = True
    save_metadata: bool = True
    save_config: bool = True
    save_metrics: bool = True
    
    # Compression and optimization
    compress_checkpoints: bool = False
    use_torch_script: bool = False
    use_onnx: bool = False
    quantization: bool = False
    quantization_type: str = "int8"  # int8, fp16
    
    # Git integration
    git_enabled: bool = True
    git_auto_commit: bool = True
    git_tag_checkpoints: bool = True
    git_tag_format: str = "checkpoint-{version}-{metric}"
    
    # Backup settings
    create_backups: bool = True
    backup_interval: int = 86400  # 24 hours
    max_backups: int = 10
    
    # Metadata
    created_by: str = "Deep Learning Team"
    description: str = ""
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    def save(self, filepath: str):
        """Save config to file."""
        with open(filepath, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'CheckpointConfig':
        """Load config from file."""
        with open(filepath, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            config_dict = json.load(f)
        return cls(**config_dict)


@dataclass
class CheckpointInfo:
    """Information about a model checkpoint."""
    checkpoint_id: str = field(default_factory=lambda: str(hashlib.md5(datetime.now().isoformat().encode()).hexdigest()[:8]))
    checkpoint_path: str = ""
    model_name: str = ""
    model_version: str = ""
    model_type: str = "pytorch"
    
    # Training information
    epoch: int = 0
    step: int = 0
    global_step: int = 0
    
    # Performance metrics
    metrics: Dict[str, float] = field(default_factory=dict)
    best_metric: Optional[str] = None
    best_value: Optional[float] = None
    
    # Model information
    model_size: int = 0
    num_parameters: int = 0
    trainable_parameters: int = 0
    
    # File information
    file_size: int = 0
    file_hash: str = ""
    compression_ratio: float = 1.0
    
    # System information
    created_date: str = field(default_factory=lambda: datetime.now().isoformat())
    created_by: str = ""
    system_info: Dict[str, Any] = field(default_factory=dict)
    
    # Git information
    git_commit: str = ""
    git_branch: str = ""
    git_tag: str = ""
    
    # Metadata
    description: str = ""
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert checkpoint info to dictionary."""
        return asdict(self)
    
    def save(self, filepath: str):
        """Save checkpoint info to file."""
        with open(filepath, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'CheckpointInfo':
        """Load checkpoint info from file."""
        with open(filepath, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            info_dict = json.load(f)
        return cls(**info_dict)


class ModelCheckpointer:
    """Comprehensive model checkpointing system."""
    
    def __init__(self, config: CheckpointConfig):
        
    """__init__ function."""
self.config = config
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self.db_path = self.checkpoint_dir / "checkpoints.db"
        self._init_database()
        
        # Initialize Git manager if enabled
        self.git_manager = None
        if config.git_enabled:
            git_config = GitConfig(
                repo_path=str(self.checkpoint_dir),
                author_name=config.created_by,
                author_email="checkpoint@example.com",
                track_models=True,
                auto_commit=config.git_auto_commit
            )
            self.git_manager = GitManager(git_config)
        
        # Track best checkpoints
        self.best_checkpoints = {}
        self.checkpoint_history = []
        
        logger.info(f"Initialized model checkpointer: {config.model_name}")
    
    def _init_database(self) -> Any:
        """Initialize SQLite database for checkpoint tracking."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create checkpoints table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS checkpoints (
                checkpoint_id TEXT PRIMARY KEY,
                checkpoint_path TEXT NOT NULL,
                model_name TEXT,
                model_version TEXT,
                model_type TEXT,
                epoch INTEGER,
                step INTEGER,
                global_step INTEGER,
                metrics TEXT,
                best_metric TEXT,
                best_value REAL,
                model_size INTEGER,
                num_parameters INTEGER,
                trainable_parameters INTEGER,
                file_size INTEGER,
                file_hash TEXT,
                compression_ratio REAL,
                created_date TEXT,
                created_by TEXT,
                system_info TEXT,
                git_commit TEXT,
                git_branch TEXT,
                git_tag TEXT,
                description TEXT,
                tags TEXT,
                notes TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_checkpoint(self, model: nn.Module, optimizer=None, scheduler=None, scaler=None,
                       epoch: int = 0, step: int = 0, global_step: int = 0,
                       metrics: Dict[str, float] = None, config: Dict[str, Any] = None,
                       description: str = "", tags: List[str] = None) -> str:
        """Save a model checkpoint."""
        # Generate checkpoint ID
        checkpoint_id = self._generate_checkpoint_id(epoch, step, metrics)
        
        # Create checkpoint directory
        checkpoint_dir = self.checkpoint_dir / checkpoint_id
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Create checkpoint info
        checkpoint_info = CheckpointInfo(
            checkpoint_id=checkpoint_id,
            checkpoint_path=str(checkpoint_dir),
            model_name=self.config.model_name,
            model_version=self.config.model_version,
            model_type=self.config.model_type,
            epoch=epoch,
            step=step,
            global_step=global_step,
            metrics=metrics or {},
            created_by=self.config.created_by,
            description=description,
            tags=tags or []
        )
        
        # Save model state
        if self.config.save_model_state:
            model_path = checkpoint_dir / "model.pth"
            torch.save(model.state_dict(), model_path)
            checkpoint_info.checkpoint_path = str(model_path)
        
        # Save optimizer state
        if self.config.save_optimizer_state and optimizer:
            optimizer_path = checkpoint_dir / "optimizer.pth"
            torch.save(optimizer.state_dict(), optimizer_path)
        
        # Save scheduler state
        if self.config.save_scheduler_state and scheduler:
            scheduler_path = checkpoint_dir / "scheduler.pth"
            torch.save(scheduler.state_dict(), scheduler_path)
        
        # Save scaler state
        if self.config.save_scaler_state and scaler:
            scaler_path = checkpoint_dir / "scaler.pth"
            torch.save(scaler.state_dict(), scaler_path)
        
        # Save configuration
        if self.config.save_config and config:
            config_path = checkpoint_dir / "config.yaml"
            with open(config_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                yaml.dump(config, f, default_flow_style=False, indent=2)
        
        # Save metrics
        if self.config.save_metrics and metrics:
            metrics_path = checkpoint_dir / "metrics.json"
            with open(metrics_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                json.dump(metrics, f, indent=2)
        
        # Calculate model information
        checkpoint_info.num_parameters = sum(p.numel() for p in model.parameters())
        checkpoint_info.trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Calculate file information
        if checkpoint_info.checkpoint_path:
            file_path = Path(checkpoint_info.checkpoint_path)
            checkpoint_info.file_size = file_path.stat().st_size
            checkpoint_info.file_hash = self._calculate_file_hash(file_path)
        
        # Save checkpoint info
        info_path = checkpoint_dir / "checkpoint_info.json"
        checkpoint_info.save(str(info_path))
        
        # Update best checkpoints
        self._update_best_checkpoints(checkpoint_info)
        
        # Save to database
        self._insert_checkpoint_to_db(checkpoint_info)
        
        # Git operations
        if self.git_manager:
            self._git_commit_checkpoint(checkpoint_info)
        
        # Add to history
        self.checkpoint_history.append(checkpoint_info)
        
        logger.info(f"Saved checkpoint: {checkpoint_id} (epoch {epoch}, step {step})")
        return checkpoint_id
    
    def load_checkpoint(self, checkpoint_id: str, model: nn.Module = None,
                       optimizer=None, scheduler=None, scaler=None,
                       map_location: str = "cpu") -> Dict[str, Any]:
        """Load a model checkpoint."""
        checkpoint_dir = self.checkpoint_dir / checkpoint_id
        info_path = checkpoint_dir / "checkpoint_info.json"
        
        if not info_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_id}")
        
        # Load checkpoint info
        checkpoint_info = CheckpointInfo.load(str(info_path))
        
        # Load model state
        model_state = None
        if self.config.save_model_state:
            model_path = checkpoint_dir / "model.pth"
            if model_path.exists():
                model_state = torch.load(model_path, map_location=map_location)
                if model:
                    model.load_state_dict(model_state)
        
        # Load optimizer state
        optimizer_state = None
        if self.config.save_optimizer_state:
            optimizer_path = checkpoint_dir / "optimizer.pth"
            if optimizer_path.exists() and optimizer:
                optimizer_state = torch.load(optimizer_path, map_location=map_location)
                optimizer.load_state_dict(optimizer_state)
        
        # Load scheduler state
        scheduler_state = None
        if self.config.save_scheduler_state:
            scheduler_path = checkpoint_dir / "scheduler.pth"
            if scheduler_path.exists() and scheduler:
                scheduler_state = torch.load(scheduler_path, map_location=map_location)
                scheduler.load_state_dict(scheduler_state)
        
        # Load scaler state
        scaler_state = None
        if self.config.save_scaler_state:
            scaler_path = checkpoint_dir / "scaler.pth"
            if scaler_path.exists() and scaler:
                scaler_state = torch.load(scaler_path, map_location=map_location)
                scaler.load_state_dict(scaler_state)
        
        # Load configuration
        config = None
        if self.config.save_config:
            config_path = checkpoint_dir / "config.yaml"
            if config_path.exists():
                with open(config_path, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    config = yaml.safe_load(f)
        
        # Load metrics
        metrics = None
        if self.config.save_metrics:
            metrics_path = checkpoint_dir / "metrics.json"
            if metrics_path.exists():
                with open(metrics_path, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    metrics = json.load(f)
        
        logger.info(f"Loaded checkpoint: {checkpoint_id}")
        
        return {
            'checkpoint_info': checkpoint_info,
            'model_state': model_state,
            'optimizer_state': optimizer_state,
            'scheduler_state': scheduler_state,
            'scaler_state': scaler_state,
            'config': config,
            'metrics': metrics
        }
    
    def get_best_checkpoint(self, metric: str, maximize: bool = True) -> Optional[CheckpointInfo]:
        """Get the best checkpoint based on a metric."""
        if metric not in self.best_checkpoints:
            return None
        
        return self.best_checkpoints[metric]
    
    def get_latest_checkpoint(self) -> Optional[CheckpointInfo]:
        """Get the latest checkpoint."""
        if not self.checkpoint_history:
            return None
        
        return self.checkpoint_history[-1]
    
    def list_checkpoints(self, limit: int = None) -> List[CheckpointInfo]:
        """List all checkpoints."""
        if limit:
            return self.checkpoint_history[-limit:]
        return self.checkpoint_history.copy()
    
    def delete_checkpoint(self, checkpoint_id: str):
        """Delete a checkpoint."""
        checkpoint_dir = self.checkpoint_dir / checkpoint_id
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
            
            # Remove from history
            self.checkpoint_history = [c for c in self.checkpoint_history if c.checkpoint_id != checkpoint_id]
            
            # Remove from database
            self._delete_checkpoint_from_db(checkpoint_id)
            
            logger.info(f"Deleted checkpoint: {checkpoint_id}")
        else:
            logger.warning(f"Checkpoint not found: {checkpoint_id}")
    
    def export_checkpoint(self, checkpoint_id: str, export_path: str = None) -> str:
        """Export a checkpoint to a zip file."""
        checkpoint_dir = self.checkpoint_dir / checkpoint_id
        
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_id}")
        
        if export_path is None:
            export_path = f"checkpoint_{checkpoint_id}.zip"
        
        with zipfile.ZipFile(export_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in checkpoint_dir.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(checkpoint_dir)
                    zipf.write(file_path, arcname)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        
        logger.info(f"Exported checkpoint to: {export_path}")
        return export_path
    
    def import_checkpoint(self, import_path: str) -> str:
        """Import a checkpoint from a zip file."""
        checkpoint_id = None
        
        with zipfile.ZipFile(import_path, 'r') as zipf:
            # Find checkpoint info file
            info_files = [f for f in zipf.namelist() if f.endswith('checkpoint_info.json')]
            if not info_files:
                raise ValueError("No checkpoint info file found in archive")
            
            # Read checkpoint info
            with zipf.open(info_files[0]) as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                info_data = json.load(f)
                checkpoint_id = info_data.get('checkpoint_id')
            
            if not checkpoint_id:
                raise ValueError("Invalid checkpoint info file")
            
            # Extract to checkpoint directory
            checkpoint_dir = self.checkpoint_dir / checkpoint_id
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            zipf.extractall(checkpoint_dir)
        
        # Load checkpoint info
        info_path = checkpoint_dir / "checkpoint_info.json"
        checkpoint_info = CheckpointInfo.load(str(info_path))
        
        # Add to history and database
        self.checkpoint_history.append(checkpoint_info)
        self._insert_checkpoint_to_db(checkpoint_info)
        
        logger.info(f"Imported checkpoint: {checkpoint_id}")
        return checkpoint_id
    
    def compare_checkpoints(self, checkpoint_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple checkpoints."""
        checkpoints = []
        for checkpoint_id in checkpoint_ids:
            checkpoint_dir = self.checkpoint_dir / checkpoint_id
            info_path = checkpoint_dir / "checkpoint_info.json"
            if info_path.exists():
                checkpoint_info = CheckpointInfo.load(str(info_path))
                checkpoints.append(checkpoint_info)
        
        if len(checkpoints) < 2:
            return {}
        
        comparison = {
            'checkpoints': [c.to_dict() for c in checkpoints],
            'metrics_comparison': {},
            'size_comparison': {
                'file_sizes': [c.file_size for c in checkpoints],
                'model_sizes': [c.num_parameters for c in checkpoints],
                'compression_ratios': [c.compression_ratio for c in checkpoints]
            },
            'performance_comparison': {
                'epochs': [c.epoch for c in checkpoints],
                'steps': [c.step for c in checkpoints],
                'global_steps': [c.global_step for c in checkpoints]
            }
        }
        
        # Compare metrics
        all_metrics = set()
        for checkpoint in checkpoints:
            all_metrics.update(checkpoint.metrics.keys())
        
        for metric in all_metrics:
            values = []
            for checkpoint in checkpoints:
                if metric in checkpoint.metrics:
                    values.append(checkpoint.metrics[metric])
                else:
                    values.append(None)
            
            comparison['metrics_comparison'][metric] = {
                'values': values,
                'min': min(v for v in values if v is not None) if any(v is not None for v in values) else None,
                'max': max(v for v in values if v is not None) if any(v is not None for v in values) else None,
                'avg': sum(v for v in values if v is not None) / len([v for v in values if v is not None]) if any(v is not None for v in values) else None
            }
        
        return comparison
    
    def _generate_checkpoint_id(self, epoch: int, step: int, metrics: Dict[str, float] = None) -> str:
        """Generate a unique checkpoint ID."""
        timestamp = datetime.now().isoformat()
        metric_str = ""
        if metrics:
            metric_str = "_".join([f"{k}_{v:.4f}" for k, v in sorted(metrics.items())])
        
        content = f"{self.config.model_name}_{epoch}_{step}_{timestamp}_{metric_str}"
        return hashlib.md5(content.encode()).hexdigest()[:8]
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of a file."""
        hash_sha256 = hashlib.sha256()
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
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def _update_best_checkpoints(self, checkpoint_info: CheckpointInfo):
        """Update best checkpoints tracking."""
        for metric_name, metric_value in checkpoint_info.metrics.items():
            if metric_name not in self.best_checkpoints:
                self.best_checkpoints[metric_name] = checkpoint_info
                checkpoint_info.best_metric = metric_name
                checkpoint_info.best_value = metric_value
            else:
                current_best = self.best_checkpoints[metric_name]
                current_value = current_best.metrics.get(metric_name, float('-inf'))
                
                # Update if better (assuming higher is better for most metrics)
                if metric_value > current_value:
                    self.best_checkpoints[metric_name] = checkpoint_info
                    checkpoint_info.best_metric = metric_name
                    checkpoint_info.best_value = metric_value
    
    def _git_commit_checkpoint(self, checkpoint_info: CheckpointInfo):
        """Commit checkpoint to Git."""
        if not self.git_manager:
            return
        
        checkpoint_dir = Path(checkpoint_info.checkpoint_path).parent
        
        # Add all files in checkpoint directory
        for file_path in checkpoint_dir.rglob('*'):
            if file_path.is_file():
                self.git_manager.add_file(str(file_path), f"Save checkpoint: {checkpoint_info.checkpoint_id}"f")
        
        # Create tag if enabled
        if self.config.git_tag_checkpoints:
            tag_name = self.config.git_tag_format"
            self.git_manager.create_tag(tag_name, f"Checkpoint {checkpoint_info.checkpoint_id}")
    
    def _insert_checkpoint_to_db(self, checkpoint_info: CheckpointInfo):
        """Insert checkpoint into database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO checkpoints (
                checkpoint_id, checkpoint_path, model_name, model_version, model_type,
                epoch, step, global_step, metrics, best_metric, best_value,
                model_size, num_parameters, trainable_parameters, file_size,
                file_hash, compression_ratio, created_date, created_by,
                system_info, git_commit, git_branch, git_tag, description,
                tags, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            checkpoint_info.checkpoint_id, checkpoint_info.checkpoint_path,
            checkpoint_info.model_name, checkpoint_info.model_version,
            checkpoint_info.model_type, checkpoint_info.epoch, checkpoint_info.step,
            checkpoint_info.global_step, json.dumps(checkpoint_info.metrics),
            checkpoint_info.best_metric, checkpoint_info.best_value,
            checkpoint_info.model_size, checkpoint_info.num_parameters,
            checkpoint_info.trainable_parameters, checkpoint_info.file_size,
            checkpoint_info.file_hash, checkpoint_info.compression_ratio,
            checkpoint_info.created_date, checkpoint_info.created_by,
            json.dumps(checkpoint_info.system_info), checkpoint_info.git_commit,
            checkpoint_info.git_branch, checkpoint_info.git_tag,
            checkpoint_info.description, json.dumps(checkpoint_info.tags),
            checkpoint_info.notes
        ))
        
        conn.commit()
        conn.close()
    
    def _delete_checkpoint_from_db(self, checkpoint_id: str):
        """Delete checkpoint from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM checkpoints WHERE checkpoint_id = ?', (checkpoint_id,))
        
        conn.commit()
        conn.close()


class ModelCheckpointerCLI:
    """Command-line interface for model checkpointing."""
    
    def __init__(self, checkpointer: ModelCheckpointer):
        
    """__init__ function."""
self.checkpointer = checkpointer
    
    def create_parser(self) -> Any:
        """Create command-line argument parser."""
        
        parser = argparse.ArgumentParser(description="Model Checkpointing CLI")
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # List checkpoints
        list_parser = subparsers.add_parser('list', help='List checkpoints')
        list_parser.add_argument('--limit', type=int, help='Number of checkpoints to show')
        
        # Get checkpoint
        get_parser = subparsers.add_parser('get', help='Get checkpoint information')
        get_parser.add_argument('checkpoint_id', help='Checkpoint ID')
        
        # Load checkpoint
        load_parser = subparsers.add_parser('load', help='Load checkpoint')
        load_parser.add_argument('checkpoint_id', help='Checkpoint ID')
        load_parser.add_argument('--output', help='Output file for model state')
        
        # Delete checkpoint
        delete_parser = subparsers.add_parser('delete', help='Delete checkpoint')
        delete_parser.add_argument('checkpoint_id', help='Checkpoint ID')
        
        # Export checkpoint
        export_parser = subparsers.add_parser('export', help='Export checkpoint')
        export_parser.add_argument('checkpoint_id', help='Checkpoint ID')
        export_parser.add_argument('--output', help='Output file path')
        
        # Import checkpoint
        import_parser = subparsers.add_parser('import', help='Import checkpoint')
        import_parser.add_argument('import_path', help='Import file path')
        
        # Compare checkpoints
        compare_parser = subparsers.add_parser('compare', help='Compare checkpoints')
        compare_parser.add_argument('checkpoint_ids', nargs='+', help='Checkpoint IDs to compare')
        
        return parser
    
    def run(self, args=None) -> Any:
        """Run the CLI."""
        parser = self.create_parser()
        parsed_args = parser.parse_args(args)
        
        if parsed_args.command == 'list':
            checkpoints = self.checkpointer.list_checkpoints(parsed_args.limit)
            for checkpoint in checkpoints:
                print(f"{checkpoint.checkpoint_id}: {checkpoint.model_name} (epoch {checkpoint.epoch}) - {checkpoint.created_date}")
        elif parsed_args.command == 'get':
            checkpoint_dir = self.checkpointer.checkpoint_dir / parsed_args.checkpoint_id
            info_path = checkpoint_dir / "checkpoint_info.json"
            if info_path.exists():
                checkpoint_info = CheckpointInfo.load(str(info_path))
                print(json.dumps(checkpoint_info.to_dict(), indent=2))
            else:
                print("Checkpoint not found")
        elif parsed_args.command == 'load':
            try:
                result = self.checkpointer.load_checkpoint(parsed_args.checkpoint_id)
                if parsed_args.output and result['model_state']:
                    torch.save(result['model_state'], parsed_args.output)
                    print(f"Model state saved to: {parsed_args.output}")
                else:
                    print("Checkpoint loaded successfully")
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
        elif parsed_args.command == 'delete':
            self.checkpointer.delete_checkpoint(parsed_args.checkpoint_id)
            print(f"Deleted checkpoint: {parsed_args.checkpoint_id}")
        elif parsed_args.command == 'export':
            try:
                export_path = self.checkpointer.export_checkpoint(parsed_args.checkpoint_id, parsed_args.output)
                print(f"Exported to: {export_path}")
            except Exception as e:
                print(f"Error exporting checkpoint: {e}")
        elif parsed_args.command == 'import':
            try:
                checkpoint_id = self.checkpointer.import_checkpoint(parsed_args.import_path)
                print(f"Imported checkpoint: {checkpoint_id}")
            except Exception as e:
                print(f"Error importing checkpoint: {e}")
        elif parsed_args.command == 'compare':
            comparison = self.checkpointer.compare_checkpoints(parsed_args.checkpoint_ids)
            print(json.dumps(comparison, indent=2))
        else:
            parser.print_help()


# Example usage
if __name__ == "__main__":
    # Create checkpoint configuration
    config = CheckpointConfig(
        model_name="transformer_classifier",
        model_version="1.0.0",
        checkpoint_dir="./checkpoints",
        save_best_only=True,
        save_last=True,
        git_enabled=True,
        git_auto_commit=True
    )
    
    # Create model checkpointer
    checkpointer = ModelCheckpointer(config)
    
    # Create CLI
    cli = ModelCheckpointerCLI(checkpointer)
    
    # Run CLI
    cli.run() 