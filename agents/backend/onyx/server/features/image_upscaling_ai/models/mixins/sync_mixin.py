"""
Sync Mixin

Contains synchronization functionality.
"""

import logging
import json
from typing import Union, Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class SyncMixin:
    """
    Mixin providing synchronization functionality.
    
    This mixin contains:
    - Sync configurations
    - Sync presets
    - Sync workflows
    - Sync state
    - Conflict resolution
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize sync mixin."""
        super().__init__(*args, **kwargs)
        if not hasattr(self, '_sync_state'):
            self._sync_state = {
                "last_sync": None,
                "sync_metadata": {},
            }
    
    def export_sync_data(
        self,
        include_presets: bool = True,
        include_workflows: bool = True,
        include_config: bool = True
    ) -> Dict[str, Any]:
        """
        Export data for synchronization.
        
        Args:
            include_presets: Include presets
            include_workflows: Include workflows
            include_config: Include configuration
            
        Returns:
            Dictionary with sync data
        """
        sync_data = {
            "exported_at": datetime.now().isoformat(),
            "version": "1.0",
            "data": {},
        }
        
        # Export presets
        if include_presets and hasattr(self, 'list_presets'):
            presets = {}
            for preset_name in self.list_presets():
                if hasattr(self, 'get_preset_info'):
                    preset_info = self.get_preset_info(preset_name)
                    if preset_info:
                        presets[preset_name] = preset_info
            sync_data["data"]["presets"] = presets
        
        # Export workflows
        if include_workflows and hasattr(self, 'list_workflows'):
            workflows = {}
            for workflow_name in self.list_workflows():
                if hasattr(self, 'get_workflow_info'):
                    workflow_info = self.get_workflow_info(workflow_name)
                    if workflow_info:
                        workflows[workflow_name] = workflow_info
            sync_data["data"]["workflows"] = workflows
        
        # Export configuration
        if include_config and hasattr(self, 'get_configuration'):
            sync_data["data"]["configuration"] = self.get_configuration()
        
        return sync_data
    
    def import_sync_data(
        self,
        sync_data: Dict[str, Any],
        conflict_resolution: str = "skip"
    ) -> Dict[str, Any]:
        """
        Import synchronized data.
        
        Args:
            sync_data: Sync data dictionary
            conflict_resolution: How to handle conflicts ('skip', 'overwrite', 'rename')
            
        Returns:
            Dictionary with import results
        """
        results = {
            "imported": 0,
            "skipped": 0,
            "errors": [],
        }
        
        data = sync_data.get("data", {})
        
        # Import presets
        if "presets" in data:
            for preset_name, preset_info in data["presets"].items():
                try:
                    if hasattr(self, 'get_preset_info'):
                        existing = self.get_preset_info(preset_name)
                        if existing:
                            if conflict_resolution == "skip":
                                results["skipped"] += 1
                                continue
                            elif conflict_resolution == "rename":
                                preset_name = f"{preset_name}_imported"
                    
                    if hasattr(self, 'create_preset'):
                        self.create_preset(preset_name, preset_info)
                        results["imported"] += 1
                except Exception as e:
                    results["errors"].append(f"Preset {preset_name}: {str(e)}")
        
        # Import workflows
        if "workflows" in data:
            for workflow_name, workflow_info in data["workflows"].items():
                try:
                    if hasattr(self, 'get_workflow_info'):
                        existing = self.get_workflow_info(workflow_name)
                        if existing:
                            if conflict_resolution == "skip":
                                results["skipped"] += 1
                                continue
                            elif conflict_resolution == "rename":
                                workflow_name = f"{workflow_name}_imported"
                    
                    if hasattr(self, 'create_workflow'):
                        steps = workflow_info.get("steps", [])
                        description = workflow_info.get("description", "")
                        self.create_workflow(workflow_name, steps, description)
                        results["imported"] += 1
                except Exception as e:
                    results["errors"].append(f"Workflow {workflow_name}: {str(e)}")
        
        # Import configuration
        if "configuration" in data and hasattr(self, 'update_configuration'):
            try:
                self.update_configuration(data["configuration"])
                results["imported"] += 1
            except Exception as e:
                results["errors"].append(f"Configuration: {str(e)}")
        
        # Update sync state
        self._sync_state["last_sync"] = datetime.now().isoformat()
        
        logger.info(f"Sync data imported: {results['imported']} items, {results['skipped']} skipped")
        
        return results
    
    def sync_with_remote(
        self,
        remote_data: Dict[str, Any],
        conflict_resolution: str = "skip"
    ) -> Dict[str, Any]:
        """
        Synchronize with remote data.
        
        Args:
            remote_data: Remote sync data
            conflict_resolution: How to handle conflicts
            
        Returns:
            Dictionary with sync results
        """
        # Export local data
        local_data = self.export_sync_data()
        
        # Compare timestamps
        local_time = datetime.fromisoformat(local_data["exported_at"])
        remote_time = datetime.fromisoformat(remote_data.get("exported_at", "1970-01-01T00:00:00"))
        
        # Import remote data
        import_results = self.import_sync_data(remote_data, conflict_resolution)
        
        return {
            "local_timestamp": local_data["exported_at"],
            "remote_timestamp": remote_data.get("exported_at"),
            "sync_direction": "import",
            "import_results": import_results,
        }
    
    def get_sync_state(self) -> Dict[str, Any]:
        """Get current sync state."""
        return self._sync_state.copy()
    
    def reset_sync_state(self) -> None:
        """Reset sync state."""
        self._sync_state = {
            "last_sync": None,
            "sync_metadata": {},
        }
        logger.info("Sync state reset")


