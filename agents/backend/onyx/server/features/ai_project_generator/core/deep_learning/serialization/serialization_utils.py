"""
Serialization Utilities
========================

Advanced serialization and versioning utilities.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import json
import pickle
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)


def save_model_complete(
    model: nn.Module,
    output_path: Path,
    config: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    include_optimizer: bool = False,
    optimizer: Optional[torch.optim.Optimizer] = None
) -> Path:
    """
    Save model with complete information.
    
    Args:
        model: PyTorch model
        output_path: Output file path
        config: Model configuration
        metadata: Additional metadata
        include_optimizer: Include optimizer state
        optimizer: Optimizer to save
        
    Returns:
        Path to saved model
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_type': type(model).__name__,
        'timestamp': datetime.now().isoformat(),
    }
    
    if config:
        checkpoint['config'] = config
    
    if metadata:
        checkpoint['metadata'] = metadata
    
    if include_optimizer and optimizer:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    # Calculate hash
    model_bytes = pickle.dumps(model.state_dict())
    model_hash = hashlib.sha256(model_bytes).hexdigest()
    checkpoint['model_hash'] = model_hash
    
    torch.save(checkpoint, output_path)
    
    logger.info(f"Model saved completely: {output_path}")
    return output_path


def load_model_complete(
    checkpoint_path: Path,
    model_class: type,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Load model with complete information.
    
    Args:
        checkpoint_path: Path to checkpoint
        model_class: Model class
        device: Target device
        
    Returns:
        Dictionary with model and metadata
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Create model
    config = checkpoint.get('config', {})
    model = model_class(**config)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if device:
        model = model.to(device)
    
    result = {
        'model': model,
        'config': checkpoint.get('config', {}),
        'metadata': checkpoint.get('metadata', {}),
        'timestamp': checkpoint.get('timestamp', ''),
        'model_hash': checkpoint.get('model_hash', ''),
    }
    
    if 'optimizer_state_dict' in checkpoint:
        result['optimizer_state_dict'] = checkpoint['optimizer_state_dict']
    
    logger.info(f"Model loaded completely: {checkpoint_path}")
    return result


def serialize_config(config: Dict[str, Any], output_path: Path) -> Path:
    """
    Serialize configuration to JSON.
    
    Args:
        config: Configuration dictionary
        output_path: Output file path
        
    Returns:
        Path to saved config
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert non-serializable types
    serializable_config = {}
    for key, value in config.items():
        if isinstance(value, (int, float, str, bool, list, dict, type(None))):
            serializable_config[key] = value
        else:
            serializable_config[key] = str(value)
    
    with open(output_path, 'w') as f:
        json.dump(serializable_config, f, indent=2)
    
    logger.info(f"Config serialized: {output_path}")
    return output_path


def deserialize_config(config_path: Path) -> Dict[str, Any]:
    """
    Deserialize configuration from JSON.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    logger.info(f"Config deserialized: {config_path}")
    return config


class ModelVersionManager:
    """
    Manage model versions.
    """
    
    def __init__(self, models_dir: Path):
        """
        Initialize version manager.
        
        Args:
            models_dir: Directory for models
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.version_file = self.models_dir / "versions.json"
        self.versions = self._load_versions()
    
    def _load_versions(self) -> Dict[str, Any]:
        """Load version information."""
        if self.version_file.exists():
            with open(self.version_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_versions(self) -> None:
        """Save version information."""
        with open(self.version_file, 'w') as f:
            json.dump(self.versions, f, indent=2)
    
    def save_version(
        self,
        model: nn.Module,
        version: str,
        config: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Save model version.
        
        Args:
            model: PyTorch model
            version: Version string
            config: Model configuration
            metadata: Additional metadata
            
        Returns:
            Path to saved model
        """
        model_path = self.models_dir / f"model_v{version}.pt"
        
        save_model_complete(model, model_path, config, metadata)
        
        # Update versions
        self.versions[version] = {
            'path': str(model_path),
            'timestamp': datetime.now().isoformat(),
            'config': config or {},
            'metadata': metadata or {}
        }
        self._save_versions()
        
        logger.info(f"Model version {version} saved")
        return model_path
    
    def load_version(
        self,
        version: str,
        model_class: type,
        device: Optional[torch.device] = None
    ) -> Dict[str, Any]:
        """
        Load model version.
        
        Args:
            version: Version string
            model_class: Model class
            device: Target device
            
        Returns:
            Dictionary with model and metadata
        """
        if version not in self.versions:
            raise ValueError(f"Version {version} not found")
        
        model_path = Path(self.versions[version]['path'])
        return load_model_complete(model_path, model_class, device)
    
    def list_versions(self) -> List[str]:
        """List all versions."""
        return sorted(self.versions.keys(), reverse=True)
    
    def get_latest_version(self) -> Optional[str]:
        """Get latest version."""
        versions = self.list_versions()
        return versions[0] if versions else None



