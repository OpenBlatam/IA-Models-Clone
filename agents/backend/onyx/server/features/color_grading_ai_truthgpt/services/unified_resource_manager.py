"""
Unified Resource Manager for Color Grading AI
=============================================

Consolidates resource management services:
- TemplateManager (templates)
- PresetManager (presets)
- LUTManager (LUTs)
- VersionManager (versions)
- HistoryManager (history)
- BackupManager (backups)

Features:
- Unified interface for all resource types
- Common CRUD operations
- Search and filtering
- Version control
- Backup/restore
"""

import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Type, TypeVar
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

from .template_manager import TemplateManager, ColorGradingTemplate
from .preset_manager import PresetManager, ColorPreset
from .lut_manager import LUTManager, LUTInfo
from .version_manager import VersionManager, Version
from .history_manager import HistoryManager, ProcessingHistory
from .backup_manager import BackupManager

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ResourceType(Enum):
    """Resource types."""
    TEMPLATE = "template"
    PRESET = "preset"
    LUT = "lut"
    VERSION = "version"
    HISTORY = "history"
    BACKUP = "backup"


@dataclass
class ResourceMetadata:
    """Resource metadata."""
    resource_type: ResourceType
    name: str
    path: str
    created_at: datetime
    updated_at: datetime
    size: int = 0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class UnifiedResourceManager:
    """
    Unified resource manager.
    
    Consolidates:
    - TemplateManager: Templates
    - PresetManager: Presets
    - LUTManager: LUTs
    - VersionManager: Versions
    - HistoryManager: History
    - BackupManager: Backups
    
    Features:
    - Unified interface for all resources
    - Common CRUD operations
    - Search and filtering
    - Version control
    - Backup/restore
    """
    
    def __init__(
        self,
        base_dir: str = "resources",
        templates_dir: Optional[str] = None,
        presets_dir: Optional[str] = None,
        luts_dir: Optional[str] = None,
        versions_dir: Optional[str] = None,
        history_dir: Optional[str] = None,
        backups_dir: Optional[str] = None
    ):
        """
        Initialize unified resource manager.
        
        Args:
            base_dir: Base directory for resources
            templates_dir: Optional templates directory
            presets_dir: Optional presets directory
            luts_dir: Optional LUTs directory
            versions_dir: Optional versions directory
            history_dir: Optional history directory
            backups_dir: Optional backups directory
        """
        base_path = Path(base_dir)
        
        # Initialize individual managers
        self.template_manager = TemplateManager(
            templates_dir=templates_dir or str(base_path / "templates")
        )
        self.preset_manager = PresetManager(
            presets_dir=presets_dir or str(base_path / "presets")
        )
        self.lut_manager = LUTManager(
            luts_dir=luts_dir or str(base_path / "luts")
        )
        self.version_manager = VersionManager(
            versions_dir=versions_dir or str(base_path / "versions")
        )
        self.history_manager = HistoryManager(
            history_dir=history_dir or str(base_path / "history")
        )
        self.backup_manager = BackupManager(
            backup_dir=backups_dir or str(base_path / "backups")
        )
        
        # Map resource types to managers
        self._managers = {
            ResourceType.TEMPLATE: self.template_manager,
            ResourceType.PRESET: self.preset_manager,
            ResourceType.LUT: self.lut_manager,
            ResourceType.VERSION: self.version_manager,
            ResourceType.HISTORY: self.history_manager,
            ResourceType.BACKUP: self.backup_manager,
        }
        
        logger.info("Initialized UnifiedResourceManager")
    
    def get_resource(
        self,
        resource_type: ResourceType,
        name: str
    ) -> Optional[Any]:
        """
        Get resource by type and name.
        
        Args:
            resource_type: Resource type
            name: Resource name
            
        Returns:
            Resource object or None
        """
        manager = self._managers.get(resource_type)
        if not manager:
            return None
        
        if resource_type == ResourceType.TEMPLATE:
            return manager.get_template(name)
        elif resource_type == ResourceType.PRESET:
            return manager.get_preset(name)
        elif resource_type == ResourceType.LUT:
            return manager.get_lut(name)
        elif resource_type == ResourceType.VERSION:
            return manager.get_version(name)
        elif resource_type == ResourceType.HISTORY:
            return manager.get_history(name)
        
        return None
    
    def list_resources(
        self,
        resource_type: ResourceType,
        filter_func: Optional[callable] = None
    ) -> List[Any]:
        """
        List resources by type.
        
        Args:
            resource_type: Resource type
            filter_func: Optional filter function
            
        Returns:
            List of resources
        """
        manager = self._managers.get(resource_type)
        if not manager:
            return []
        
        if resource_type == ResourceType.TEMPLATE:
            resources = manager.list_templates()
        elif resource_type == ResourceType.PRESET:
            resources = manager.list_presets()
        elif resource_type == ResourceType.LUT:
            resources = manager.list_luts()
        elif resource_type == ResourceType.VERSION:
            resources = manager.list_versions()
        elif resource_type == ResourceType.HISTORY:
            resources = manager.search(limit=1000)
        else:
            resources = []
        
        if filter_func:
            resources = [r for r in resources if filter_func(r)]
        
        return resources
    
    def save_resource(
        self,
        resource_type: ResourceType,
        resource: Any,
        name: Optional[str] = None
    ) -> bool:
        """
        Save resource.
        
        Args:
            resource_type: Resource type
            resource: Resource object
            name: Optional resource name
            
        Returns:
            True if saved
        """
        manager = self._managers.get(resource_type)
        if not manager:
            return False
        
        try:
            if resource_type == ResourceType.TEMPLATE:
                manager.save_template(resource)
            elif resource_type == ResourceType.PRESET:
                manager.save_preset(resource)
            elif resource_type == ResourceType.LUT:
                manager.add_lut(resource)
            elif resource_type == ResourceType.VERSION:
                manager.create_version(resource)
            elif resource_type == ResourceType.HISTORY:
                manager.add_history(resource)
            
            return True
        except Exception as e:
            logger.error(f"Error saving resource: {e}")
            return False
    
    def delete_resource(
        self,
        resource_type: ResourceType,
        name: str
    ) -> bool:
        """
        Delete resource.
        
        Args:
            resource_type: Resource type
            name: Resource name
            
        Returns:
            True if deleted
        """
        manager = self._managers.get(resource_type)
        if not manager:
            return False
        
        try:
            if resource_type == ResourceType.TEMPLATE:
                manager.delete_template(name)
            elif resource_type == ResourceType.PRESET:
                manager.delete_preset(name)
            elif resource_type == ResourceType.LUT:
                manager.remove_lut(name)
            elif resource_type == ResourceType.VERSION:
                manager.delete_version(name)
            elif resource_type == ResourceType.HISTORY:
                manager.delete_history(name)
            
            return True
        except Exception as e:
            logger.error(f"Error deleting resource: {e}")
            return False
    
    def search_resources(
        self,
        resource_type: ResourceType,
        query: str,
        limit: int = 100
    ) -> List[Any]:
        """
        Search resources.
        
        Args:
            resource_type: Resource type
            query: Search query
            limit: Result limit
            
        Returns:
            List of matching resources
        """
        manager = self._managers.get(resource_type)
        if not manager:
            return []
        
        if resource_type == ResourceType.TEMPLATE:
            return manager.search_templates(query, limit=limit)
        elif resource_type == ResourceType.PRESET:
            return manager.search_presets(query, limit=limit)
        elif resource_type == ResourceType.LUT:
            return manager.search_luts(query, limit=limit)
        elif resource_type == ResourceType.HISTORY:
            return manager.search(query=query, limit=limit)
        
        return []
    
    def create_backup(
        self,
        resource_type: ResourceType,
        backup_name: Optional[str] = None
    ) -> Optional[str]:
        """
        Create backup of resource type.
        
        Args:
            resource_type: Resource type
            backup_name: Optional backup name
            
        Returns:
            Backup path or None
        """
        manager = self._managers.get(resource_type)
        if not manager:
            return None
        
        try:
            if resource_type == ResourceType.TEMPLATE:
                resources_dir = manager.templates_dir
            elif resource_type == ResourceType.PRESET:
                resources_dir = manager.presets_dir
            elif resource_type == ResourceType.LUT:
                resources_dir = manager.luts_dir
            else:
                return None
            
            return self.backup_manager.create_backup(
                str(resources_dir),
                backup_name=backup_name
            )
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get resource statistics."""
        return {
            "templates": len(self.template_manager.list_templates()),
            "presets": len(self.preset_manager.list_presets()),
            "luts": len(self.lut_manager.list_luts()),
            "versions": len(self.version_manager.list_versions()),
            "history_entries": len(self.history_manager.search(limit=10000)),
        }


