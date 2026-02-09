"""
Collaboration Mixin

Contains collaboration and sharing functionality.
"""

import logging
import json
from typing import Union, Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
from PIL import Image

logger = logging.getLogger(__name__)


class CollaborationMixin:
    """
    Mixin providing collaboration and sharing functionality.
    
    This mixin contains:
    - Share configurations
    - Share presets
    - Share workflows
    - Collaboration sessions
    - Shared resources
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize collaboration mixin."""
        super().__init__(*args, **kwargs)
        if not hasattr(self, '_shared_resources'):
            self._shared_resources = {}
        if not hasattr(self, '_collaboration_sessions'):
            self._collaboration_sessions = {}
    
    def share_preset(
        self,
        preset_name: str,
        share_id: Optional[str] = None,
        description: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Share a preset with others.
        
        Args:
            preset_name: Name of preset to share
            share_id: Optional share ID
            description: Optional description
            
        Returns:
            Dictionary with share information
        """
        if not hasattr(self, 'get_preset_info'):
            raise AttributeError("ConfigurationMixin required for sharing presets")
        
        preset_info = self.get_preset_info(preset_name)
        if not preset_info:
            raise ValueError(f"Preset '{preset_name}' not found")
        
        if share_id is None:
            share_id = f"preset_{preset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        share_info = {
            "share_id": share_id,
            "type": "preset",
            "resource_name": preset_name,
            "resource_data": preset_info,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "created_by": "system",  # Could be user ID in real implementation
        }
        
        if not hasattr(self, '_shared_resources'):
            self._shared_resources = {}
        self._shared_resources[share_id] = share_info
        
        logger.info(f"Preset '{preset_name}' shared with ID: {share_id}")
        
        return share_info
    
    def share_workflow(
        self,
        workflow_name: str,
        share_id: Optional[str] = None,
        description: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Share a workflow with others.
        
        Args:
            workflow_name: Name of workflow to share
            share_id: Optional share ID
            description: Optional description
            
        Returns:
            Dictionary with share information
        """
        if not hasattr(self, 'get_workflow_info'):
            raise AttributeError("WorkflowMixin required for sharing workflows")
        
        workflow_info = self.get_workflow_info(workflow_name)
        if not workflow_info:
            raise ValueError(f"Workflow '{workflow_name}' not found")
        
        if share_id is None:
            share_id = f"workflow_{workflow_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        share_info = {
            "share_id": share_id,
            "type": "workflow",
            "resource_name": workflow_name,
            "resource_data": workflow_info,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "created_by": "system",
        }
        
        if not hasattr(self, '_shared_resources'):
            self._shared_resources = {}
        self._shared_resources[share_id] = share_info
        
        logger.info(f"Workflow '{workflow_name}' shared with ID: {share_id}")
        
        return share_info
    
    def get_shared_resource(self, share_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a shared resource.
        
        Args:
            share_id: Share ID
            
        Returns:
            Dictionary with resource information or None
        """
        if not hasattr(self, '_shared_resources'):
            return None
        return self._shared_resources.get(share_id)
    
    def import_shared_resource(
        self,
        share_id: str,
        new_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Import a shared resource.
        
        Args:
            share_id: Share ID
            new_name: Optional new name for imported resource
            
        Returns:
            Dictionary with import result
        """
        share_info = self.get_shared_resource(share_id)
        if not share_info:
            return {"success": False, "error": "Share ID not found"}
        
        resource_type = share_info["type"]
        resource_data = share_info["resource_data"]
        original_name = share_info["resource_name"]
        new_name = new_name or f"{original_name}_imported"
        
        try:
            if resource_type == "preset":
                if hasattr(self, 'create_preset'):
                    self.create_preset(new_name, resource_data)
                    return {"success": True, "type": "preset", "name": new_name}
            elif resource_type == "workflow":
                if hasattr(self, 'create_workflow'):
                    steps = resource_data.get("steps", [])
                    self.create_workflow(new_name, steps, resource_data.get("description", ""))
                    return {"success": True, "type": "workflow", "name": new_name}
        except Exception as e:
            return {"success": False, "error": str(e)}
        
        return {"success": False, "error": f"Unknown resource type: {resource_type}"}
    
    def list_shared_resources(self) -> List[Dict[str, Any]]:
        """List all shared resources."""
        if not hasattr(self, '_shared_resources'):
            return []
        return list(self._shared_resources.values())
    
    def create_collaboration_session(
        self,
        session_name: str,
        participants: List[str],
        description: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a collaboration session.
        
        Args:
            session_name: Name of session
            participants: List of participant IDs
            description: Optional description
            
        Returns:
            Dictionary with session information
        """
        session_id = f"session_{session_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        session_info = {
            "session_id": session_id,
            "session_name": session_name,
            "participants": participants,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "shared_resources": [],
        }
        
        if not hasattr(self, '_collaboration_sessions'):
            self._collaboration_sessions = {}
        self._collaboration_sessions[session_id] = session_info
        
        logger.info(f"Collaboration session '{session_name}' created: {session_id}")
        
        return session_info
    
    def add_resource_to_session(
        self,
        session_id: str,
        share_id: str
    ) -> bool:
        """
        Add a shared resource to a collaboration session.
        
        Args:
            session_id: Session ID
            share_id: Share ID
            
        Returns:
            True if successful
        """
        if not hasattr(self, '_collaboration_sessions'):
            return False
        
        if session_id not in self._collaboration_sessions:
            return False
        
        session = self._collaboration_sessions[session_id]
        if share_id not in session["shared_resources"]:
            session["shared_resources"].append(share_id)
        
        return True
    
    def get_collaboration_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get collaboration session information."""
        if not hasattr(self, '_collaboration_sessions'):
            return None
        return self._collaboration_sessions.get(session_id)


