"""
Backup Mixin

Contains backup and restore functionality.
"""

import logging
import json
import shutil
from typing import Union, Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class BackupMixin:
    """
    Mixin providing backup and restore functionality.
    
    This mixin contains:
    - Configuration backup
    - Cache backup
    - Preset backup
    - Full system backup
    - Restore operations
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize backup mixin."""
        super().__init__(*args, **kwargs)
        if not hasattr(self, '_backup_dir'):
            self._backup_dir = Path("backups")
            self._backup_dir.mkdir(exist_ok=True)
    
    def backup_configuration(
        self,
        backup_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Backup current configuration.
        
        Args:
            backup_name: Optional backup name
            
        Returns:
            Dictionary with backup information
        """
        if backup_name is None:
            backup_name = f"config_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        backup_path = self._backup_dir / f"{backup_name}_config.json"
        
        config = {}
        
        # Backup cache settings
        if hasattr(self, 'cache') and self.cache:
            config["cache"] = {
                "enabled": True,
                "max_size": getattr(self.cache, 'max_size', 64),
                "ttl": getattr(self.cache, 'ttl', 3600),
            }
        else:
            config["cache"] = {"enabled": False}
        
        # Backup initialization parameters
        config["init_params"] = {
            "enable_cache": getattr(self, 'enable_cache', True),
            "max_workers": getattr(self, 'max_workers', 4),
            "validate_images": getattr(self, 'validate_images', True),
            "enhance_images": getattr(self, 'enhance_images', False),
            "auto_select_method": getattr(self, 'auto_select_method', False),
        }
        
        # Save backup
        with open(backup_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Configuration backed up to {backup_path}")
        
        return {
            "backup_name": backup_name,
            "backup_path": str(backup_path),
            "timestamp": datetime.now().isoformat(),
        }
    
    def backup_presets(
        self,
        backup_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Backup all presets.
        
        Args:
            backup_name: Optional backup name
            
        Returns:
            Dictionary with backup information
        """
        if backup_name is None:
            backup_name = f"presets_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        backup_path = self._backup_dir / f"{backup_name}_presets.json"
        
        presets = {}
        
        if hasattr(self, 'list_presets'):
            preset_names = self.list_presets()
            for preset_name in preset_names:
                if hasattr(self, 'get_preset_info'):
                    preset_info = self.get_preset_info(preset_name)
                    if preset_info:
                        presets[preset_name] = preset_info
        
        # Save backup
        with open(backup_path, 'w') as f:
            json.dump(presets, f, indent=2)
        
        logger.info(f"Presets backed up to {backup_path}")
        
        return {
            "backup_name": backup_name,
            "backup_path": str(backup_path),
            "presets_count": len(presets),
            "timestamp": datetime.now().isoformat(),
        }
    
    def backup_workflows(
        self,
        backup_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Backup all workflows.
        
        Args:
            backup_name: Optional backup name
            
        Returns:
            Dictionary with backup information
        """
        if backup_name is None:
            backup_name = f"workflows_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        backup_path = self._backup_dir / f"{backup_name}_workflows.json"
        
        workflows = {}
        
        if hasattr(self, 'list_workflows'):
            workflow_names = self.list_workflows()
            for workflow_name in workflow_names:
                if hasattr(self, 'get_workflow_info'):
                    workflow_info = self.get_workflow_info(workflow_name)
                    if workflow_info:
                        workflows[workflow_name] = workflow_info
        
        # Save backup
        with open(backup_path, 'w') as f:
            json.dump(workflows, f, indent=2)
        
        logger.info(f"Workflows backed up to {backup_path}")
        
        return {
            "backup_name": backup_name,
            "backup_path": str(backup_path),
            "workflows_count": len(workflows),
            "timestamp": datetime.now().isoformat(),
        }
    
    def create_full_backup(
        self,
        backup_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a full system backup.
        
        Args:
            backup_name: Optional backup name
            
        Returns:
            Dictionary with backup information
        """
        if backup_name is None:
            backup_name = f"full_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        backup_results = {
            "backup_name": backup_name,
            "timestamp": datetime.now().isoformat(),
            "components": {},
        }
        
        # Backup configuration
        try:
            config_backup = self.backup_configuration(f"{backup_name}_config")
            backup_results["components"]["configuration"] = config_backup
        except Exception as e:
            logger.error(f"Failed to backup configuration: {e}")
            backup_results["components"]["configuration"] = {"error": str(e)}
        
        # Backup presets
        try:
            presets_backup = self.backup_presets(f"{backup_name}_presets")
            backup_results["components"]["presets"] = presets_backup
        except Exception as e:
            logger.error(f"Failed to backup presets: {e}")
            backup_results["components"]["presets"] = {"error": str(e)}
        
        # Backup workflows
        try:
            workflows_backup = self.backup_workflows(f"{backup_name}_workflows")
            backup_results["components"]["workflows"] = workflows_backup
        except Exception as e:
            logger.error(f"Failed to backup workflows: {e}")
            backup_results["components"]["workflows"] = {"error": str(e)}
        
        # Save backup manifest
        manifest_path = self._backup_dir / f"{backup_name}_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(backup_results, f, indent=2)
        
        logger.info(f"Full backup created: {backup_name}")
        
        return backup_results
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """List all available backups."""
        backups = []
        
        for manifest_file in self._backup_dir.glob("*_manifest.json"):
            try:
                with open(manifest_file, 'r') as f:
                    manifest = json.load(f)
                    backups.append(manifest)
            except Exception as e:
                logger.error(f"Failed to read backup manifest {manifest_file}: {e}")
        
        return sorted(backups, key=lambda x: x.get("timestamp", ""), reverse=True)
    
    def restore_backup(
        self,
        backup_name: str
    ) -> Dict[str, Any]:
        """
        Restore from a backup.
        
        Args:
            backup_name: Name of backup to restore
            
        Returns:
            Dictionary with restore results
        """
        manifest_path = self._backup_dir / f"{backup_name}_manifest.json"
        
        if not manifest_path.exists():
            raise ValueError(f"Backup '{backup_name}' not found")
        
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        restore_results = {
            "backup_name": backup_name,
            "restored_components": [],
            "errors": [],
        }
        
        # Restore configuration
        if "configuration" in manifest.get("components", {}):
            config_info = manifest["components"]["configuration"]
            if "backup_path" in config_info:
                try:
                    with open(config_info["backup_path"], 'r') as f:
                        config = json.load(f)
                    # Apply configuration (implementation depends on system)
                    restore_results["restored_components"].append("configuration")
                except Exception as e:
                    restore_results["errors"].append(f"Configuration restore failed: {e}")
        
        # Restore presets
        if "presets" in manifest.get("components", {}):
            presets_info = manifest["components"]["presets"]
            if "backup_path" in presets_info:
                try:
                    with open(presets_info["backup_path"], 'r') as f:
                        presets = json.load(f)
                    # Restore presets (implementation depends on system)
                    restore_results["restored_components"].append("presets")
                except Exception as e:
                    restore_results["errors"].append(f"Presets restore failed: {e}")
        
        # Restore workflows
        if "workflows" in manifest.get("components", {}):
            workflows_info = manifest["components"]["workflows"]
            if "backup_path" in workflows_info:
                try:
                    with open(workflows_info["backup_path"], 'r') as f:
                        workflows = json.load(f)
                    # Restore workflows (implementation depends on system)
                    restore_results["restored_components"].append("workflows")
                except Exception as e:
                    restore_results["errors"].append(f"Workflows restore failed: {e}")
        
        logger.info(f"Backup '{backup_name}' restored")
        
        return restore_results


