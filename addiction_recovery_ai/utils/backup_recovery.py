"""
Backup and Recovery for Models and Data
"""

import torch
import shutil
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
import hashlib

logger = logging.getLogger(__name__)


class ModelBackup:
    """Backup and recovery for models"""
    
    def __init__(self, backup_dir: str = "backups"):
        """
        Initialize model backup
        
        Args:
            backup_dir: Backup directory
        """
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)
        
        self.manifest_file = self.backup_dir / "manifest.json"
        self.manifest = self._load_manifest()
        
        logger.info(f"ModelBackup initialized: {backup_dir}")
    
    def _load_manifest(self) -> Dict[str, Any]:
        """Load backup manifest"""
        if self.manifest_file.exists():
            try:
                with open(self.manifest_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load manifest: {e}")
        
        return {"backups": []}
    
    def _save_manifest(self):
        """Save backup manifest"""
        with open(self.manifest_file, 'w') as f:
            json.dump(self.manifest, f, indent=2)
    
    def backup_model(
        self,
        model: torch.nn.Module,
        model_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Backup model
        
        Args:
            model: Model to backup
            model_name: Model name
            metadata: Optional metadata
        
        Returns:
            Backup path
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{model_name}_{timestamp}"
        backup_path = self.backup_dir / backup_name
        backup_path.mkdir(exist_ok=True)
        
        # Save model
        model_path = backup_path / "model.pth"
        torch.save(model.state_dict(), model_path)
        
        # Save metadata
        metadata = metadata or {}
        metadata.update({
            "model_name": model_name,
            "backup_time": timestamp,
            "model_path": str(model_path)
        })
        
        metadata_path = backup_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Calculate hash
        model_hash = self._calculate_hash(model_path)
        
        # Update manifest
        backup_entry = {
            "backup_name": backup_name,
            "model_name": model_name,
            "timestamp": timestamp,
            "path": str(backup_path),
            "hash": model_hash,
            "metadata": metadata
        }
        
        self.manifest["backups"].append(backup_entry)
        self._save_manifest()
        
        logger.info(f"Model backed up: {backup_name}")
        return str(backup_path)
    
    def _calculate_hash(self, filepath: Path) -> str:
        """Calculate file hash"""
        hasher = hashlib.sha256()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def list_backups(self, model_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List backups
        
        Args:
            model_name: Optional model name filter
        
        Returns:
            List of backups
        """
        backups = self.manifest.get("backups", [])
        
        if model_name:
            backups = [b for b in backups if b["model_name"] == model_name]
        
        return sorted(backups, key=lambda x: x["timestamp"], reverse=True)
    
    def restore_model(
        self,
        backup_name: str,
        model_class: torch.nn.Module
    ) -> torch.nn.Module:
        """
        Restore model from backup
        
        Args:
            backup_name: Backup name
            model_class: Model class to instantiate
        
        Returns:
            Restored model
        """
        backups = self.list_backups()
        backup = next((b for b in backups if b["backup_name"] == backup_name), None)
        
        if backup is None:
            raise ValueError(f"Backup not found: {backup_name}")
        
        model_path = Path(backup["path"]) / "model.pth"
        
        model = model_class()
        model.load_state_dict(torch.load(model_path))
        
        logger.info(f"Model restored from backup: {backup_name}")
        return model
    
    def delete_backup(self, backup_name: str):
        """Delete backup"""
        backups = self.list_backups()
        backup = next((b for b in backups if b["backup_name"] == backup_name), None)
        
        if backup is None:
            raise ValueError(f"Backup not found: {backup_name}")
        
        # Delete directory
        backup_path = Path(backup["path"])
        if backup_path.exists():
            shutil.rmtree(backup_path)
        
        # Remove from manifest
        self.manifest["backups"] = [
            b for b in self.manifest["backups"]
            if b["backup_name"] != backup_name
        ]
        self._save_manifest()
        
        logger.info(f"Backup deleted: {backup_name}")


class DataBackup:
    """Backup and recovery for data"""
    
    def __init__(self, backup_dir: str = "data_backups"):
        """
        Initialize data backup
        
        Args:
            backup_dir: Backup directory
        """
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)
        
        logger.info(f"DataBackup initialized: {backup_dir}")
    
    def backup_data(
        self,
        data: Any,
        data_name: str,
        format: str = "json"
    ) -> str:
        """
        Backup data
        
        Args:
            data: Data to backup
            data_name: Data name
            format: Backup format (json, pickle)
        
        Returns:
            Backup path
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{data_name}_{timestamp}"
        backup_path = self.backup_dir / backup_name
        backup_path.mkdir(exist_ok=True)
        
        if format == "json":
            import json
            filepath = backup_path / f"{data_name}.json"
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        else:
            import pickle
            filepath = backup_path / f"{data_name}.pkl"
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
        
        logger.info(f"Data backed up: {backup_name}")
        return str(backup_path)
    
    def restore_data(self, backup_name: str, format: str = "json") -> Any:
        """
        Restore data from backup
        
        Args:
            backup_name: Backup name
            format: Backup format
        
        Returns:
            Restored data
        """
        backup_path = self.backup_dir / backup_name
        
        if format == "json":
            import json
            files = list(backup_path.glob("*.json"))
        else:
            import pickle
            files = list(backup_path.glob("*.pkl"))
        
        if not files:
            raise ValueError(f"No data files found in backup: {backup_name}")
        
        filepath = files[0]
        
        if format == "json":
            import json
            with open(filepath, 'r') as f:
                data = json.load(f)
        else:
            import pickle
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
        
        logger.info(f"Data restored from backup: {backup_name}")
        return data

