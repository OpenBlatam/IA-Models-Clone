"""
Workflow Template Manager
==========================

Manages workflow template loading, caching, and validation.
"""

import json
import logging
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class WorkflowTemplateManager:
    """Manages workflow template operations."""
    
    def __init__(self, workflow_path: str):
        """
        Initialize Workflow Template Manager.
        
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
        
        workflow_file = Path(__file__).parent.parent / self.workflow_path
        if not workflow_file.is_absolute():
            workflow_file = Path(__file__).parent.parent / workflow_file
        
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
        """Get default workflow structure"""
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
    
    def validate_structure(self, workflow: Dict[str, Any], required_node_ids: list) -> Tuple[bool, Optional[str]]:
        """
        Validate workflow structure.
        
        Args:
            workflow: Workflow dictionary to validate
            required_node_ids: List of required node IDs
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(workflow, dict):
            return False, "Workflow must be a dictionary"
        
        if "nodes" not in workflow:
            return False, "Workflow missing 'nodes' key"
        
        if not isinstance(workflow["nodes"], list):
            return False, "Workflow 'nodes' must be a list"
        
        if len(workflow["nodes"]) == 0:
            return False, "Workflow has no nodes"
        
        # Check for required nodes
        node_ids = [node.get("id") for node in workflow["nodes"]]
        missing_nodes = [nid for nid in required_node_ids if nid not in node_ids]
        if missing_nodes:
            return False, f"Workflow missing required nodes: {missing_nodes}"
        
        return True, None
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded workflow template.
        
        Returns:
            Dictionary with workflow information
        """
        workflow = self.load_template()
        
        nodes = workflow.get("nodes", [])
        links = workflow.get("links", [])
        node_types = list(set(node.get("type", "unknown") for node in nodes))
        
        return {
            "node_count": len(nodes),
            "link_count": len(links),
            "node_types": node_types,
            "version": workflow.get("version", "unknown")
        }

