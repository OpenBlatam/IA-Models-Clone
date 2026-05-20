"""
Workflow Manager
===============
Manages workflow template loading and caching
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class WorkflowManager:
    """
    Manages workflow template loading and caching.
    """
    
    def __init__(self, workflow_path: str):
        """
        Initialize workflow manager.
        
        Args:
            workflow_path: Path to workflow template file
        """
        self.workflow_path = workflow_path
        self._workflow_template: Optional[Dict[str, Any]] = None
    
    def load_template(self) -> Dict[str, Any]:
        """
        Load the ComfyUI workflow template.
        
        Returns:
            Workflow dictionary with nodes, links, and configuration
        """
        if self._workflow_template is not None:
            return self._workflow_template
        
        workflow_file = Path(self.workflow_path)
        if not workflow_file.is_absolute():
            # Try relative to services directory
            workflow_file = Path(__file__).parent.parent.parent / workflow_file
        
        if not workflow_file.exists():
            logger.warning(f"Workflow file not found at {workflow_file}, using default structure")
            self._workflow_template = self._get_default_workflow()
        else:
            try:
                with open(workflow_file, 'r', encoding='utf-8') as f:
                    self._workflow_template = json.load(f)
                logger.info(f"Loaded workflow template from {workflow_file}")
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in workflow file: {e}")
                self._workflow_template = self._get_default_workflow()
            except Exception as e:
                logger.error(f"Error loading workflow file: {e}")
                self._workflow_template = self._get_default_workflow()
        
        return self._workflow_template
    
    def _get_default_workflow(self) -> Dict[str, Any]:
        """
        Get default workflow structure.
        
        Returns:
            Default workflow dictionary
        """
        return {
            "last_node_id": 419,
            "last_link_id": 665,
            "nodes": [],
            "links": [],
            "groups": [],
            "config": {},
            "extra": {},
            "version": 0.4
        }
    
    def get_template(self) -> Dict[str, Any]:
        """
        Get cached workflow template.
        
        Returns:
            Workflow template dictionary
        """
        return self.load_template()
    
    def clear_cache(self) -> None:
        """Clear cached workflow template"""
        self._workflow_template = None

