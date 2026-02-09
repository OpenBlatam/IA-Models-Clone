"""
Workflow Node Manager
====================
Manages workflow node operations and updates
"""

import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

# Node IDs from the workflow JSON
NODE_IDS = {
    "LOAD_IMAGE": 239,
    "LOAD_NEW_FACE": 240,
    "CLIP_TEXT_ENCODE": 343,
    "FLUX_GUIDANCE": 345,
    "KSAMPLER": 346,
    "VAE_DECODE": 214,
    "IMAGE_CROP": 228,
    "INPAINT_CROP": 411,
    "INPAINT_STITCH": 412,
    "CONDITIONING_ZERO_OUT": 404,  # For negative prompt
}


class WorkflowNodeManager:
    """
    Manages workflow node operations and updates.
    """
    
    def __init__(self, node_ids: Optional[Dict[str, int]] = None):
        """
        Initialize workflow node manager.
        
        Args:
            node_ids: Optional custom node ID mapping
        """
        self.node_ids = node_ids or NODE_IDS
    
    def find_node(
        self,
        workflow: Dict[str, Any],
        node_id: int
    ) -> Optional[Dict[str, Any]]:
        """
        Find a node in workflow by ID.
        
        Args:
            workflow: Workflow dictionary
            node_id: Node ID to find
            
        Returns:
            Node dictionary or None if not found
        """
        nodes = workflow.get("nodes", [])
        for node in nodes:
            if node.get("id") == node_id:
                return node
        return None
    
    def update_node_widget(
        self,
        workflow: Dict[str, Any],
        node_id: int,
        widget_index: int,
        value: Any
    ) -> bool:
        """
        Update a widget value in a node.
        
        Args:
            workflow: Workflow dictionary
            node_id: Node ID
            widget_index: Widget index
            value: New value
            
        Returns:
            True if update successful, False otherwise
        """
        node = self.find_node(workflow, node_id)
        if not node:
            logger.warning(f"Node {node_id} not found in workflow")
            return False
        
        widgets_values = node.get("widgets_values", [])
        if 0 <= widget_index < len(widgets_values):
            widgets_values[widget_index] = value
            logger.debug(f"Updated node {node_id} widget {widget_index} to {value}")
            return True
        
        logger.warning(
            f"Widget index {widget_index} out of range for node {node_id} "
            f"(has {len(widgets_values)} widgets)"
        )
        return False
    
    def get_node(self, workflow: Dict[str, Any], node_name: str) -> Optional[Dict[str, Any]]:
        """
        Get node by name using node_ids mapping.
        
        Args:
            workflow: Workflow dictionary
            node_name: Node name (key in NODE_IDS)
            
        Returns:
            Node dictionary or None if not found
        """
        node_id = self.node_ids.get(node_name)
        if node_id is None:
            logger.warning(f"Node name '{node_name}' not found in node_ids mapping")
            return None
        
        return self.find_node(workflow, node_id)
    
    def update_node_by_name(
        self,
        workflow: Dict[str, Any],
        node_name: str,
        widget_index: int,
        value: Any
    ) -> bool:
        """
        Update node widget by node name.
        
        Args:
            workflow: Workflow dictionary
            node_name: Node name (key in NODE_IDS)
            widget_index: Widget index
            value: New value
            
        Returns:
            True if update successful, False otherwise
        """
        node_id = self.node_ids.get(node_name)
        if node_id is None:
            logger.warning(f"Node name '{node_name}' not found in node_ids mapping")
            return False
        
        return self.update_node_widget(workflow, node_id, widget_index, value)

